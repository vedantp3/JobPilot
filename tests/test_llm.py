"""GeminiLLM behaviour against a fake google-genai client (no network)."""
from types import SimpleNamespace

import pytest
from google.genai import errors

from jobpilot import llm as llm_mod
from jobpilot.llm import GeminiLLM, LLMError


def api_error(code, message, details=()):
    return errors.APIError(code, {"error": {"code": code, "message": message, "details": list(details)}})


def ok(text):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason="STOP")])


class FakeModels:
    def __init__(self, *results):
        self.results, self.calls = list(results), []

    async def generate_content(self, *, model, contents, config):
        self.calls.append(config)
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def sleeps(monkeypatch):
    waited = []

    async def fake_sleep(s):
        waited.append(s)

    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    return waited


def gemini(*results):
    g = GeminiLLM("test-key", "gemini-3.5-flash-lite", min_interval=0)
    g.client = SimpleNamespace(aio=SimpleNamespace(models=FakeModels(*results)))
    return g, g.client.aio.models


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


async def test_json_with_schema(sleeps):
    g, models = gemini(ok('{"a": 1}'))
    assert await g.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 1}
    cfg = models.calls[0]
    assert cfg["response_json_schema"] == SCHEMA and cfg["response_mime_type"] == "application/json"
    assert cfg["system_instruction"] == "s"


async def test_strips_code_fences(sleeps):
    g, _ = gemini(ok('```json\n{"a": 2}\n```'))
    assert await g.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 2}


async def test_rate_limit_retries_with_server_delay(sleeps):
    g, models = gemini(
        api_error(429, "limit: 10 per minute. Please retry in 7.5s.",
                  [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]),
        api_error(503, "overloaded"),
        ok('{"a": 3}'),
    )
    assert await g.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 3}
    assert len(models.calls) == 3
    assert sleeps[0] == pytest.approx(8.5)   # server-suggested delay + 1s
    assert sleeps[1] == 10.0                 # exponential backoff (5 * 2**1)


async def test_daily_quota_fails_fast(sleeps):
    g, models = gemini(api_error(429, "Quota exceeded for metric: generate_content_free_tier_requests, "
                                      "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    with pytest.raises(LLMError, match="daily quota") as ei:
        await g.complete_json(system="s", user="u", schema=SCHEMA)
    assert ei.value.code == 429 and len(models.calls) == 1 and sleeps == []


async def test_gives_up_after_max_retries(sleeps):
    g, models = gemini(*[api_error(500, "boom")] * 6)
    with pytest.raises(LLMError, match="500"):
        await g.complete_json(system="s", user="u", schema=SCHEMA)
    assert len(models.calls) == 6


async def test_schema_rejected_falls_back_to_prompt(sleeps):
    g, models = gemini(api_error(400, "Invalid JSON payload: unsupported schema keyword"), ok('{"a": 4}'),
                       ok('{"a": 5}'))
    assert await g.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 4}
    assert "response_json_schema" not in models.calls[1]
    assert '"required": ["a"]' in models.calls[1]["system_instruction"]
    # remembered for later calls
    assert await g.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 5}
    assert "response_json_schema" not in models.calls[2]


async def test_other_client_errors_not_retried(sleeps):
    g, models = gemini(api_error(403, "API key not valid"))
    with pytest.raises(LLMError, match="403"):
        await g.complete_text(system="s", user="u")
    assert len(models.calls) == 1


async def test_empty_response(sleeps):
    g, _ = gemini(SimpleNamespace(text=None, candidates=[SimpleNamespace(finish_reason="SAFETY")]))
    with pytest.raises(LLMError, match="SAFETY"):
        await g.complete_text(system="s", user="u")


async def test_spacing_between_calls(monkeypatch):
    waited = []

    async def fake_sleep(s):
        waited.append(s)

    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    g, _ = gemini(ok("x"), ok("y"))
    g.min_interval = 4.0
    await g.complete_text(system="s", user="u")
    await g.complete_text(system="s", user="u")
    assert len(waited) == 1 and 3.5 < waited[0] <= 4.0


def test_provider_defaults(monkeypatch):
    from jobpilot.config import Secrets

    for k in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_MODEL", "GEMINI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    s = Secrets(_env_file=None)
    assert s.providers == ["openai", "groq", "gemini"] and s.model == "gpt-6-luna"
    assert s.model_for("groq") == "openai/gpt-oss-120b"
    assert s.model_for("gemini") == "gemini-3.5-flash-lite"
    assert s.llm_key_names == ["openai_api_key", "groq_api_key", "gemini_api_key"]
    s = Secrets(_env_file=None, llm_provider="", llm_model="gpt-6.1-sol")    # blank -> default chain
    assert s.providers[0] == "openai" and s.model_for("openai") == "gpt-6.1-sol"
    s = Secrets(_env_file=None, llm_provider=" Gemini , groq,gemini", llm_model="gemini-3.8-flash")
    assert s.providers == ["gemini", "groq"] and s.model == "gemini-3.8-flash"
    assert s.model_for("groq") == "openai/gpt-oss-120b"
    # LLM_MODEL goes to the provider it belongs to, not blindly to the first one
    s = Secrets(_env_file=None, llm_provider="groq,gemini", llm_model="gemini-3.8-flash")
    assert (s.model_for("groq"), s.model_for("gemini")) == ("openai/gpt-oss-120b", "gemini-3.8-flash")
    s = Secrets(_env_file=None, llm_provider="groq,gemini", llm_model="openai/gpt-oss-20b")
    assert (s.model_for("groq"), s.model_for("gemini")) == ("openai/gpt-oss-20b", "gemini-3.5-flash-lite")
    s = Secrets(_env_file=None, llm_provider="groq,gemini", llm_model="x", groq_model="llama-3.3-70b-versatile")
    assert s.model_for("groq") == "llama-3.3-70b-versatile"
    with pytest.raises(ValueError):
        Secrets(_env_file=None, llm_provider="groq,mistral")


# --- Groq ---------------------------------------------------------------------------
import httpx  # noqa: E402
import groq  # noqa: E402

from jobpilot.llm import FallbackLLM, GroqLLM  # noqa: E402


def groq_error(code, message, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(code, headers=headers, request=httpx.Request("POST", "https://api.groq.com"))
    cls = {429: groq.RateLimitError, 400: groq.BadRequestError, 401: groq.AuthenticationError}.get(
        code, groq.InternalServerError)
    return cls(message, response=resp, body={"error": {"message": message}})


def completion(text, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                                                    message=SimpleNamespace(content=text))])


class FakeCompletions:
    def __init__(self, *results):
        self.results, self.calls = list(results), []

    async def create(self, **kw):
        self.calls.append(kw)
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def groq_llm(*results, model="openai/gpt-oss-120b", max_wait=600.0):
    g = GroqLLM("test-key", model, min_interval=0, max_wait=max_wait)
    g.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(*results)))
    return g, g.client.chat.completions


async def test_groq_strict_schema(sleeps):
    g, comp = groq_llm(completion('{"a": 1}'))
    assert await g.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 1}
    kw = comp.calls[0]
    assert kw["response_format"]["type"] == "json_schema"
    assert kw["response_format"]["json_schema"]["strict"] is True
    assert kw["reasoning_effort"] == "low"
    assert kw["messages"][0] == {"role": "system", "content": "s"}


async def test_groq_llama_uses_json_mode(sleeps):
    g, comp = groq_llm(completion('{"a": 2}'), model="llama-3.3-70b-versatile")
    assert await g.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 2}
    kw = comp.calls[0]
    assert kw["response_format"] == {"type": "json_object"} and "reasoning_effort" not in kw
    assert "JSON Schema" in kw["messages"][0]["content"]


async def test_groq_retry_after_header(sleeps):
    g, comp = groq_llm(groq_error(429, "Rate limit reached on tokens per minute (TPM)", retry_after=2),
                       completion('{"a": 3}'))
    assert await g.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 3}
    assert sleeps == [2.5]


async def test_groq_retry_delay_from_message(sleeps):
    g, _ = groq_llm(groq_error(429, "Rate limit reached on TPM. Please try again in 1m2.5s."),
                    completion("ok"))
    assert await g.complete_text(system="s", user="u") == "ok"
    assert sleeps == [pytest.approx(63.0)]


async def test_groq_daily_limit_fails_fast(sleeps):
    g, comp = groq_llm(groq_error(429, "Rate limit reached on tokens per day (TPD): Limit 200000. "
                                       "Please try again in 7m12s."))
    with pytest.raises(LLMError, match="daily limit"):
        await g.complete_text(system="s", user="u")
    assert len(comp.calls) == 1 and sleeps == []


async def test_groq_long_wait_fails_fast_when_chained(sleeps):
    g, _ = groq_llm(groq_error(429, "TPM limit", retry_after=90), max_wait=30)
    with pytest.raises(LLMError, match="wait 90s"):
        await g.complete_text(system="s", user="u")
    assert sleeps == []


async def test_groq_schema_400_falls_back_to_json_mode(sleeps):
    g, comp = groq_llm(groq_error(400, "json_schema not supported"), completion('{"a": 4}'))
    assert await g.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 4}
    assert comp.calls[1]["response_format"] == {"type": "json_object"}


async def test_groq_truncated(sleeps):
    g, _ = groq_llm(completion('{"a"', finish="length"))
    with pytest.raises(LLMError, match="truncated"):
        await g.complete_text(system="s", user="u")


async def test_fallback_chain(sleeps):
    first, c1 = groq_llm(groq_error(429, "tokens per day (TPD) exhausted"))
    second, c2 = gemini(ok('{"a": 5}'))
    chain = FallbackLLM([first, second])
    assert await chain.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 5}
    assert len(c1.calls) == 1 and len(c2.calls) == 1


async def test_fallback_all_fail(sleeps):
    first, _ = groq_llm(groq_error(401, "Invalid API Key"))
    second, _ = gemini(api_error(403, "API key not valid"))
    with pytest.raises(LLMError, match="all LLM providers failed.*401.*403"):
        await FallbackLLM([first, second]).complete_text(system="s", user="u")


# --- OpenAI -----------------------------------------------------------------------------
import openai  # noqa: E402

from jobpilot.llm import OpenAILLM  # noqa: E402


def openai_error(code, message, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(code, headers=headers, request=httpx.Request("POST", "https://api.openai.com"))
    cls = {429: openai.RateLimitError, 400: openai.BadRequestError, 401: openai.AuthenticationError}.get(
        code, openai.InternalServerError)
    return cls(message, response=resp, body={"error": {"message": message}})


def oa_response(text, status="completed", reason=None):
    return SimpleNamespace(output_text=text, status=status,
                           incomplete_details=SimpleNamespace(reason=reason) if reason else None)


class FakeResponses:
    def __init__(self, *results):
        self.results, self.calls = list(results), []

    async def create(self, **kw):
        self.calls.append(kw)
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def openai_llm(*results, max_wait=600.0):
    o = OpenAILLM("sk-test", "gpt-6-luna", max_wait=max_wait)
    o.client = SimpleNamespace(responses=FakeResponses(*results))
    return o, o.client.responses


async def test_openai_strict_schema_and_effort(sleeps):
    o, r = openai_llm(oa_response('{"a": 1}'))
    assert await o.complete_json(system="s", user="u", schema=SCHEMA, effort="medium") == {"a": 1}
    kw = r.calls[0]
    assert kw["model"] == "gpt-6-luna" and kw["instructions"] == "s" and kw["input"] == "u"
    assert kw["text"]["format"] == {"type": "json_schema", "name": "response", "schema": SCHEMA, "strict": True}
    assert kw["reasoning"] == {"effort": "medium"} and kw["store"] is False
    assert kw["max_output_tokens"] >= 16000


async def test_openai_default_effort_low_for_text(sleeps):
    o, r = openai_llm(oa_response("hello"))
    assert await o.complete_text(system="s", user="u") == "hello"
    assert r.calls[0]["reasoning"] == {"effort": "low"} and "text" not in r.calls[0]


async def test_openai_retries_with_retry_after(sleeps):
    o, r = openai_llm(openai_error(429, "Rate limit reached for requests", retry_after=3),
                      openai_error(503, "server_is_overloaded"), oa_response('{"a": 2}'))
    assert await o.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 2}
    assert sleeps[0] == 3.5 and len(r.calls) == 3


async def test_openai_billing_quota_fails_fast(sleeps):
    o, r = openai_llm(openai_error(429, "You exceeded your current quota, please check your plan and billing"))
    with pytest.raises(LLMError, match="quota/billing") as ei:
        await o.complete_text(system="s", user="u")
    assert ei.value.code == 429 and len(r.calls) == 1 and sleeps == []


async def test_openai_incomplete_response(sleeps):
    o, _ = openai_llm(oa_response("", status="incomplete", reason="max_output_tokens"))
    with pytest.raises(LLMError, match="max_output_tokens"):
        await o.complete_text(system="s", user="u")


async def test_openai_schema_rejected_uses_json_mode(sleeps):
    o, r = openai_llm(openai_error(400, "Invalid schema for response_format"), oa_response('{"a": 3}'))
    assert await o.complete_json(system="s", user="u", schema=SCHEMA) == {"a": 3}
    assert r.calls[1]["text"]["format"] == {"type": "json_object"}
    assert "JSON Schema" in r.calls[1]["instructions"]


async def test_effort_passes_through_fallback(sleeps):
    o, r = openai_llm(openai_error(401, "bad key"))
    g, comp = groq_llm(completion('{"a": 4}'))
    assert await FallbackLLM([o, g]).complete_json(system="s", user="u", schema=SCHEMA, effort="medium") == {"a": 4}
    assert comp.calls[0]["reasoning_effort"] == "medium"


def test_get_llm_skips_keyless_providers(monkeypatch):
    from jobpilot import config
    from jobpilot.llm import get_llm

    monkeypatch.setattr(config, "get_secrets", lambda: config.Secrets(
        _env_file=None, llm_provider="groq,gemini", groq_api_key="", gemini_api_key="g-key"))
    llm = get_llm()
    assert llm.name.startswith("gemini:")
    monkeypatch.setattr(config, "get_secrets", lambda: config.Secrets(
        _env_file=None, llm_provider="", openai_api_key="sk", groq_api_key="", gemini_api_key="g"))
    chain = get_llm()
    assert [l.name.split(":")[0] for l in chain.llms] == ["openai", "gemini"]
    monkeypatch.setattr(config, "get_secrets", lambda: config.Secrets(
        _env_file=None, llm_provider="groq,gemini", groq_api_key="k", gemini_api_key="g"))
    chain = get_llm()
    assert isinstance(chain, FallbackLLM) and [l.name.split(":")[0] for l in chain.llms] == ["groq", "gemini"]
    assert chain.llms[0].max_wait == 30.0 and chain.llms[1].max_wait == 600.0
