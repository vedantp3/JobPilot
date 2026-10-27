"""Small provider-agnostic LLM interface.

Callers use `complete_json` (schema-constrained) and `complete_text`. Pass a stable
`key` (e.g. "extract/<sha1 of post>") so responses can be recorded and replayed in
tests and --dry-run without spending API calls, and an `effort` ("low" | "medium" |
"high") for how hard reasoning models should think; providers without such a
setting ignore it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code  # HTTP status from the provider, when there is one


class LLM(Protocol):
    name: str

    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any],
        key: str | None = None, max_tokens: int = 8000, effort: str | None = None,
    ) -> Any: ...

    async def complete_text(
        self, *, system: str, user: str, key: str | None = None, max_tokens: int = 2000,
        effort: str | None = None,
    ) -> str: ...


def content_key(task: str, *parts: str) -> str:
    """Stable replay key: task name + sha1 of the identifying inputs."""
    h = hashlib.sha1("\x1e".join(p.strip() for p in parts).encode("utf-8")).hexdigest()[:16]
    return f"{task}/{h}"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    return t


def _parse_json(text: str) -> Any:
    try:
        return json.loads(_strip_fences(text))
    except json.JSONDecodeError as e:
        raise LLMError(f"invalid JSON from model: {e}") from e


def _schema_prompt(system: str, schema: dict) -> str:
    return f"{system}\n\nReturn only a JSON object matching this JSON Schema:\n{json.dumps(schema)}"


class _Throttled:
    """Spaces calls at least `min_interval` seconds apart."""

    min_interval: float = 0.0

    def _init_throttle(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


def _retry_after_header(e) -> float | None:
    resp = getattr(e, "response", None)
    header = resp.headers.get("retry-after") if resp is not None else None
    try:
        return float(header) if header else None
    except ValueError:
        return None


# --- OpenAI ----------------------------------------------------------------------------
_OPENAI_BILLING = re.compile(r"insufficient_quota|exceeded your current quota|billing|hard limit", re.I)


class OpenAILLM(_Throttled):
    """OpenAI Responses API with strict JSON-schema output and reasoning effort."""

    RETRYABLE = (429, 500, 502, 503, 504)

    def __init__(self, api_key: str, model: str, min_interval: float = 0.0, max_retries: int = 5,
                 max_wait: float = 600.0):
        import openai

        self._openai = openai
        self.client = openai.AsyncOpenAI(api_key=api_key or None, max_retries=0)  # we retry ourselves
        self.model = model
        self.name = f"openai:{model}"
        self.max_retries = max_retries
        self.max_wait = max_wait
        self._init_throttle(min_interval)
        self._use_schema = True

    async def _respond(self, system: str, user: str, max_tokens: int, effort: str | None,
                       text_format: dict[str, Any] | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model, "instructions": system, "input": user, "store": False,
            # reasoning tokens count against max_output_tokens: leave plenty of room
            "max_output_tokens": max(max_tokens * 2, 16000),
            "reasoning": {"effort": effort or "low"},
        }
        if text_format:
            kwargs["text"] = {"format": text_format}
        openai = self._openai
        for attempt in range(self.max_retries + 1):
            await self._throttle()
            try:
                resp = await self.client.responses.create(**kwargs)
            except openai.APIStatusError as e:
                if e.status_code == 429 and _OPENAI_BILLING.search(str(e)):
                    raise LLMError(f"OpenAI quota/billing limit reached: {e.message}", code=429) from e
                if e.status_code in self.RETRYABLE and attempt < self.max_retries:
                    delay = _retry_after_header(e)
                    delay = delay + 0.5 if delay is not None else min(60.0, 2.0 * 2 ** attempt)
                    if delay > self.max_wait:
                        raise LLMError(f"OpenAI {e.status_code}: would need to wait {delay:.0f}s",
                                       code=e.status_code) from e
                    log.warning("OpenAI %s (attempt %d); retrying in %.1fs", e.status_code, attempt + 1, delay)
                    await asyncio.sleep(delay)
                    continue
                raise LLMError(f"OpenAI API error {e.status_code}: {e.message}", code=e.status_code) from e
            except openai.APIConnectionError as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(min(30.0, 2.0 * 2 ** attempt))
                    continue
                raise LLMError(f"OpenAI connection error: {e}") from e
            if getattr(resp, "status", "completed") == "incomplete":
                reason = getattr(getattr(resp, "incomplete_details", None), "reason", "unknown")
                raise LLMError(f"OpenAI response incomplete ({reason})")
            text = resp.output_text
            if not text:
                raise LLMError("empty response from OpenAI")
            return text
        raise LLMError("unreachable")

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        if self._use_schema:
            try:
                return _parse_json(await self._respond(system, user, max_tokens, effort, {
                    "type": "json_schema", "name": "response", "schema": schema, "strict": True,
                }))
            except LLMError as e:
                if e.code != 400:
                    raise
                log.warning("OpenAI rejected the strict schema (%s); using JSON mode", e)
                self._use_schema = False
        return _parse_json(await self._respond(_schema_prompt(system, schema), user, max_tokens, effort,
                                               {"type": "json_object"}))

    async def complete_text(self, *, system, user, key=None, max_tokens=2000, effort=None):
        return await self._respond(system, user, max_tokens, effort)


# --- Anthropic ---------------------------------------------------------------------------
class AnthropicLLM:
    def __init__(self, api_key: str, model: str, effort: str = "low"):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key or None, max_retries=3)
        self.model = model
        self.effort = effort
        self.name = f"anthropic:{model}"

    async def _create(self, effort: str | None, **kwargs: Any) -> str:
        output_config = {"effort": effort or self.effort, **kwargs.pop("output_config", {})}
        try:
            resp = await self.client.messages.create(model=self.model, output_config=output_config, **kwargs)
        except self._anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}", code=e.status_code) from e
        except self._anthropic.APIConnectionError as e:
            raise LLMError(f"Anthropic connection error: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("model refused the request")
        if resp.stop_reason == "max_tokens":
            raise LLMError("response truncated (max_tokens)")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if not text:
            raise LLMError("empty response")
        return text

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        return _parse_json(await self._create(
            effort, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        ))

    async def complete_text(self, *, system, user, key=None, max_tokens=2000, effort=None):
        return await self._create(effort, max_tokens=max_tokens, system=system,
                                  messages=[{"role": "user", "content": user}])


# --- Gemini ----------------------------------------------------------------------------------
_RETRY_DELAY = re.compile(r"retry(?:Delay| in)['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.I)
_DAILY_QUOTA = re.compile(r"per.?day|PerDay|daily", re.I)


class GeminiLLM(_Throttled):
    """Google Gemini (free tier by default). Spaces calls `min_interval` seconds apart and
    retries rate-limit / server errors with backoff; a daily-quota 429 fails fast."""

    RETRYABLE = (429, 500, 502, 503, 504)

    def __init__(self, api_key: str, model: str, min_interval: float = 4.0, max_retries: int = 5,
                 max_wait: float = 600.0):
        from google import genai
        from google.genai import errors

        self._errors = errors
        self.client = genai.Client(api_key=api_key or None)  # None -> GEMINI_API_KEY / GOOGLE_API_KEY env
        self.model = model
        self.name = f"gemini:{model}"
        self.max_retries = max_retries
        self.max_wait = max_wait  # a longer suggested wait fails fast (lets a fallback provider take over)
        self._init_throttle(min_interval)
        self._use_schema = True

    async def _generate(self, system: str, user: str, config: dict[str, Any]) -> str:
        for attempt in range(self.max_retries + 1):
            await self._throttle()
            try:
                resp = await self.client.aio.models.generate_content(
                    model=self.model, contents=user, config={"system_instruction": system, **config}
                )
            except self._errors.APIError as e:
                detail = f"{e.message or ''} {e.details or ''}"
                if e.code == 429 and _DAILY_QUOTA.search(detail):
                    raise LLMError(f"Gemini free daily quota exhausted for {self.model}; it resets at "
                                   "midnight Pacific. Try another LLM_MODEL meanwhile.", code=429) from e
                if e.code in self.RETRYABLE and attempt < self.max_retries:
                    m = _RETRY_DELAY.search(detail)
                    delay = float(m.group(1)) + 1 if m else min(60.0, 5.0 * 2 ** attempt)
                    if delay > self.max_wait:
                        raise LLMError(f"Gemini {e.code}: would need to wait {delay:.0f}s", code=e.code) from e
                    log.warning("Gemini %s (attempt %d); retrying in %.0fs", e.code, attempt + 1, delay)
                    await asyncio.sleep(delay)
                    continue
                raise LLMError(f"Gemini API error {e.code}: {e.message}", code=e.code) from e
            text = resp.text
            if not text:
                reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
                raise LLMError(f"empty response from Gemini ({reason})")
            return text
        raise LLMError("unreachable")

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        # thinking tokens count against max_output_tokens, so leave headroom
        config: dict[str, Any] = {"response_mime_type": "application/json",
                                  "max_output_tokens": max(max_tokens, 8192)}
        if self._use_schema:
            try:
                return _parse_json(await self._generate(system, user, {**config, "response_json_schema": schema}))
            except LLMError as e:
                if e.code != 400:
                    raise
                # Gemini supports a subset of JSON Schema; fall back to schema-in-prompt
                log.warning("Gemini rejected the response schema (%s); using prompt-only JSON", e)
                self._use_schema = False
        return _parse_json(await self._generate(_schema_prompt(system, schema), user, config))

    async def complete_text(self, *, system, user, key=None, max_tokens=2000, effort=None):
        return await self._generate(system, user, {"max_output_tokens": max(max_tokens, 8192)})


# --- Groq ---------------------------------------------------------------------------------------
_GROQ_DAILY = re.compile(r"per day|\(TPD\)|\(RPD\)", re.I)
_GROQ_WAIT = re.compile(r"try again in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?", re.I)
# models that accept response_format json_schema (strict) on Groq
_GROQ_SCHEMA_MODELS = re.compile(r"^(?:openai/gpt-oss|qwen/|meta-models/muse)", re.I)


def _groq_retry_after(e) -> float | None:
    """Seconds to wait from the retry-after header, else from 'Please try again in 7m12.5s'."""
    if (header := _retry_after_header(e)) is not None:
        return header
    m = _GROQ_WAIT.search(str(e))
    if not m or not any(m.groups()):
        return None
    h, mnt, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mnt * 60 + s + ms / 1000


class GroqLLM(_Throttled):
    """Groq (free tier). Fast; tight per-minute token limits are handled by waiting the
    server's retry-after, or failing fast past `max_wait` so a fallback provider takes over."""

    RETRYABLE = (429, 500, 502, 503, 504)

    def __init__(self, api_key: str, model: str, min_interval: float = 2.0, max_retries: int = 5,
                 max_wait: float = 600.0):
        import groq

        self._groq = groq
        self.client = groq.AsyncGroq(api_key=api_key or None, max_retries=0)  # we retry ourselves
        self.model = model
        self.name = f"groq:{model}"
        self.max_retries = max_retries
        self.max_wait = max_wait
        self._init_throttle(min_interval)
        self._use_schema = bool(_GROQ_SCHEMA_MODELS.match(model))

    async def _chat(self, system: str, user: str, max_tokens: int, effort: str | None,
                    response_format: dict[str, Any] | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if self.model.startswith("openai/gpt-oss"):
            # reasoning tokens count against the per-minute token limit; default low
            kwargs["reasoning_effort"] = effort if effort in ("low", "medium", "high") else "low"
        groq = self._groq
        for attempt in range(self.max_retries + 1):
            await self._throttle()
            try:
                resp = await self.client.chat.completions.create(**kwargs)
            except groq.APIStatusError as e:
                if e.status_code == 429 and _GROQ_DAILY.search(str(e)):
                    raise LLMError(f"Groq daily limit reached for {self.model}: {e}", code=429) from e
                if e.status_code in self.RETRYABLE and attempt < self.max_retries:
                    delay = _groq_retry_after(e)
                    delay = delay + 0.5 if delay is not None else min(60.0, 3.0 * 2 ** attempt)
                    if delay > self.max_wait:
                        raise LLMError(f"Groq {e.status_code}: would need to wait {delay:.0f}s",
                                       code=e.status_code) from e
                    log.warning("Groq %s (attempt %d); retrying in %.1fs", e.status_code, attempt + 1, delay)
                    await asyncio.sleep(delay)
                    continue
                raise LLMError(f"Groq API error {e.status_code}: {e.message}", code=e.status_code) from e
            except groq.APIConnectionError as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(min(30.0, 2.0 * 2 ** attempt))
                    continue
                raise LLMError(f"Groq connection error: {e}") from e
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                raise LLMError("Groq response truncated (max tokens)")
            if not choice.message.content:
                raise LLMError(f"empty response from Groq ({choice.finish_reason})")
            return choice.message.content
        raise LLMError("unreachable")

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        if self._use_schema:
            try:
                return _parse_json(await self._chat(system, user, max_tokens, effort, {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True, "schema": schema},
                }))
            except LLMError as e:
                if e.code == 400:
                    log.warning("Groq rejected strict schema output (%s); using JSON mode", e)
                    self._use_schema = False
                elif not str(e).startswith("invalid JSON"):
                    raise
                # invalid JSON: fall through to JSON mode once
        return _parse_json(await self._chat(_schema_prompt(system, schema), user, max_tokens, effort,
                                            {"type": "json_object"}))

    async def complete_text(self, *, system, user, key=None, max_tokens=2000, effort=None):
        return await self._chat(system, user, max_tokens, effort)


# --- composition ------------------------------------------------------------------------------
class FallbackLLM:
    """Tries each LLM in order; any LLMError (rate limit, quota, outage, bad key) moves to the next."""

    def __init__(self, llms: list[LLM]):
        if not llms:
            raise ValueError("FallbackLLM needs at least one LLM")
        self.llms = llms
        self.name = " > ".join(l.name for l in llms)

    async def _call(self, method: str, **kwargs: Any) -> Any:
        errors: list[str] = []
        for i, llm in enumerate(self.llms):
            try:
                return await getattr(llm, method)(**kwargs)
            except LLMError as e:
                errors.append(f"{llm.name}: {e}")
                if i + 1 < len(self.llms):
                    log.warning("%s failed (%s); falling back to %s", llm.name, e, self.llms[i + 1].name)
        raise LLMError("all LLM providers failed: " + " | ".join(errors))

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        return await self._call("complete_json", system=system, user=user, schema=schema, key=key,
                                max_tokens=max_tokens, effort=effort)

    async def complete_text(self, *, system, user, key=None, max_tokens=2000, effort=None):
        return await self._call("complete_text", system=system, user=user, key=key, max_tokens=max_tokens,
                                effort=effort)


class ReplayLLM:
    """Returns recorded responses from `<dir>/<key>.json`. Raises if a key is missing."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.name = f"replay:{self.dir}"

    def _load(self, key: str | None) -> Any:
        if key is None:
            raise LLMError("ReplayLLM needs a key")
        path = self.dir / f"{key}.json"
        if not path.exists():
            raise LLMError(f"no recorded response for {key} ({path}); record with --fixtures --live-llm")
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        return self._load(key)

    async def complete_text(self, *, system, user, key=None, max_tokens=2000, effort=None):
        return self._load(key)


class RecordingLLM:
    """Wraps a live LLM and saves every keyed response for later replay."""

    def __init__(self, inner: LLM, directory: Path):
        self.inner, self.dir = inner, Path(directory)
        self.name = f"record:{inner.name}"

    def _save(self, key: str | None, response: Any) -> None:
        if key is None:
            return
        path = self.dir / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"response": response}, indent=2, ensure_ascii=False), encoding="utf-8")

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        r = await self.inner.complete_json(system=system, user=user, schema=schema, key=key,
                                           max_tokens=max_tokens, effort=effort)
        self._save(key, r)
        return r

    async def complete_text(self, *, system, user, key=None, max_tokens=2000, effort=None):
        r = await self.inner.complete_text(system=system, user=user, key=key, max_tokens=max_tokens, effort=effort)
        self._save(key, r)
        return r


def get_llm(*, replay_dir: Path | None = None, record_dir: Path | None = None) -> LLM:
    """Build the configured LLM. `replay_dir` forces replay; `record_dir` records live calls."""
    if replay_dir is not None:
        return ReplayLLM(replay_dir)
    from jobpilot.config import get_config, get_secrets

    sec, cfg = get_secrets(), get_config()
    keyed = [p for p in sec.providers if getattr(sec, f"{p}_api_key")]
    for p in sec.providers:
        if p not in keyed:
            log.info("%s_API_KEY not set in .env; %s", p.upper(),
                     "skipping it in the provider chain" if keyed else "calls will fail")
    providers = keyed or sec.providers
    # in a chain, don't sit out long rate-limit waits: hand the call to the next provider
    chained_wait = 30.0
    llms: list[LLM] = []
    for i, provider in enumerate(providers):
        model = sec.model_for(provider)
        key = getattr(sec, f"{provider}_api_key")
        max_wait = chained_wait if i + 1 < len(providers) else 600.0
        if provider == "openai":
            llms.append(OpenAILLM(key, model, max_wait=max_wait))
        elif provider == "groq":
            llms.append(GroqLLM(key, model, min_interval=cfg.llm_min_interval_seconds / 2, max_wait=max_wait))
        elif provider == "gemini":
            llms.append(GeminiLLM(key, model, min_interval=cfg.llm_min_interval_seconds, max_wait=max_wait))
        else:
            llms.append(AnthropicLLM(key, model))
    llm: LLM = llms[0] if len(llms) == 1 else FallbackLLM(llms)
    return RecordingLLM(llm, record_dir) if record_dir is not None else llm
