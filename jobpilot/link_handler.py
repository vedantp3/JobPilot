"""Link jobs: classification, answer sheets, and Playwright prefill (never submits).

- `classify_link` decides what kind of application link a URL is.
- `draft_answers` writes short, validated answers to free-text form questions.
- `build_answer_sheet` makes a copy-ready Telegram message (standard fields + answers).
- `Prefiller` opens the form in a headed, persistent Chromium window and fills what it can.
  It never clicks anything (so it can't submit); the user reviews and submits.
"""
from __future__ import annotations

import html
import json
import logging
import re
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ValidationError

from jobpilot.config import AppConfig, Profile
from jobpilot.email_drafter import (
    _EMAIL_RE, _EMOJI_RE, _NAMED_THING, _norm_text, _numbers, email_facts, never_claim_patterns,
)
from jobpilot.llm import LLM, LLMError, content_key
from jobpilot.normalize import extract_urls, url_key

log = logging.getLogger(__name__)

# --- classification ---------------------------------------------------------------------
# (link_type, host suffixes, path prefix or None)
_RULES: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    ("google_form", ("docs.google.com",), "/forms"),
    ("google_form", ("forms.gle",), None),
    ("lever", ("jobs.lever.co",), None),
    ("greenhouse", ("boards.greenhouse.io", "job-boards.greenhouse.io"), None),
    ("ashby", ("jobs.ashbyhq.com",), None),
    ("linkedin", ("linkedin.com", "lnkd.in"), None),
    ("naukri", ("naukri.com",), None),
    ("workday", ("myworkdayjobs.com",), None),
    ("whatsapp", ("wa.me", "whatsapp.com"), None),
    ("telegram", ("t.me", "telegram.me"), None),
)
PREFILLABLE = {"google_form", "lever", "greenhouse", "ashby"}
CHAT_BASED = {"whatsapp", "telegram"}

LABELS = {
    "google_form": "Google Form", "lever": "Lever", "greenhouse": "Greenhouse", "ashby": "Ashby",
    "linkedin": "LinkedIn", "naukri": "Naukri", "workday": "Workday", "whatsapp": "WhatsApp (chat-based apply)",
    "telegram": "Telegram (chat-based apply)", "other_portal": "careers portal",
}


def classify_link(url: str) -> str:
    parts = urlsplit(url if "://" in url else "https://" + url)
    host = parts.netloc.lower().split(":")[0]
    for link_type, hosts, path in _RULES:
        if any(host == h or host.endswith("." + h) for h in hosts) and (path is None or parts.path.startswith(path)):
            return link_type
    return "other_portal"


def application_url(url: str, link_type: str) -> str:
    """The page that holds the application form (no clicking needed to reach it)."""
    p = urlsplit(url)
    path = p.path.rstrip("/")
    if link_type == "lever" and not path.endswith("/apply"):
        path += "/apply"
    elif link_type == "ashby" and not path.endswith("/application"):
        path += "/application"
    elif link_type == "google_form" and path.endswith("/edit"):
        path = path[: -len("/edit")] + "/viewform"
    return urlunsplit((p.scheme, p.netloc, path, p.query, p.fragment))


# --- profile field map ------------------------------------------------------------------------
def _first_last(name: str) -> tuple[str, str]:
    parts = name.split()
    return (parts[0], " ".join(parts[1:])) if len(parts) > 1 else (name, "")


# (pattern on "label + name attribute", profile value getter). Order matters: specific first.
FIELD_RULES: tuple[tuple[str, callable], ...] = (
    # school-level marks aren't in profile.json: leave for the user
    (r"\b(?:10th|12th|ssc|hsc|tenth|twelfth|class\s*(?:x|xii|10|12))\b", None),
    (r"\bfirst\s*name\b|\bgiven\s*name\b", lambda p: _first_last(p.name)[0]),
    (r"\blast\s*name\b|\bsurname\b|\bfamily\s*name\b", lambda p: _first_last(p.name)[1]),
    (r"\b(?:company|organi[sz]ation|employer|school|father'?s?|mother'?s?|guardian'?s?|reference|referr\w*|"
     r"manager'?s?)\s*name\b", None),
    (r"\b(?:college|university|institut(?:e|ion))\s*name\b", lambda p: p.college),
    (r"\bfull\s*name\b|\byour\s*name\b|\bcandidate'?s?\s*name\b|\bapplicant'?s?\s*name\b|^\s*name\s*\*?\s*$|^name\b",
     lambda p: p.name),
    (r"\be-?mail\b|\bmail\s*id\b", lambda p: p.email),
    (r"\bphone\b|\bmobile\b|\bcontact\s*(?:no|number)\b|\bwhats\s*app\b|\bcell\b", lambda p: p.phone),
    (r"\blinked\s*in\b", lambda p: p.linkedin),
    (r"\bgit\s*hub\b", lambda p: p.github),
    (r"\bportfolio\b|\bpersonal\s*(?:website|site)\b|\bwebsite\b", lambda p: p.portfolio),
    (r"\bcgpa\b|\bgpa\b|\bcpi\b|\bpercentage\b|\baggregate\b|\bacademic\s*score\b", lambda p: p.cgpa),
    (r"\b(?:graduation|passing|passout|pass-out|grad)\s*year\b|\byear\s*of\s*(?:graduation|passing)\b|\bbatch\b",
     lambda p: str(p.grad_year)),
    (r"\bcollege\b|\buniversity\b|\binstitut(?:e|ion)\b", lambda p: p.college),
    (r"\bdegree\b|\bqualification\b|\bcourse\b|\bbranch\b|\bstream\b|\bspeciali[sz]ation\b", lambda p: p.degree),
    (r"\bnotice\s*period\b", lambda p: p.notice_period),
    (r"\bexpected\s*(?:ctc|salary|stipend|compensation)\b|\bsalary\s*expectation", lambda p: p.expected_ctc),
    (r"\bcurrent\s*(?:ctc|salary)\b", None),
    (r"\bwhen\s*can\s*you\s*(?:join|start)\b|\bjoining\s*date\b|\bavailability\b|\bstart\s*date\b",
     lambda p: p.availability),
    (r"\bcurrent\s*(?:location|city)\b|\blocation\b|\bcity\b|\bwhere\s*are\s*you\s*based\b", lambda p: p.location),
)
_COMPILED_RULES = [(re.compile(pat, re.I), get) for pat, get in FIELD_RULES]
_QUESTION = re.compile(
    r"\?\s*\*?\s*$"
    r"|^\s*(?:why|what|how|which|when|where|who|describe|tell|explain|share|briefly|write|mention|provide|"
    r"give|list|elaborate|highlight|walk\s+us|talk\s+about|summari[sz]e|outline|detail|please|in\s+\d+\s+words|"
    r"any(?:thing)?\s+else|are\s+you|do\s+you|have\s+you|can\s+you|would\s+you|is\s+there)\b",
    re.I,
)
QUESTION_MIN_WORDS = 5   # an unmatched field whose label is a sentence this long is treated as a question
_RESUME_FIELD = re.compile(r"\bresume\b|\bcv\b|\bcurriculum\b", re.I)


def match_profile_field(text: str, profile: Profile) -> tuple[bool, str | None]:
    """(matched, value). matched=True with value None means 'known field we must leave blank'."""
    for pat, get in _COMPILED_RULES:
        if pat.search(text):
            value = get(profile) if get else None
            return True, (value or None)
    return False, None


def looks_like_question(label: str, tag: str) -> bool:
    """Free-text question the LLM should answer (vs. a short factual field we can't fill)."""
    label = label.strip().rstrip("*").strip()
    if not label:
        return False
    return (bool(_QUESTION.search(label)) or (tag == "textarea" and len(label) >= 12)
            or len(re.findall(r"[A-Za-z]+", label)) >= QUESTION_MIN_WORDS)


# --- answers ----------------------------------------------------------------------------------
MAX_ANSWER_WORDS = 80
MAX_QUESTIONS = 8


class _Answer(BaseModel):
    question: str
    answer: str


class _Answers(BaseModel):
    answers: list[_Answer]


ANSWERS_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answers"],
    "properties": {"answers": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["question", "answer"],
        "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    }}},
}

JOB_FIELDS = ("company", "role", "role_category", "location", "work_mode", "stipend_or_ctc", "required_skills",
              "jd_summary")


def validate_answer(answer: str, profile: Profile, job: dict) -> list[str]:
    """Same honesty rules as emails, sized for form answers. Empty = valid (empty answers are allowed)."""
    if not answer.strip():
        return []
    errors = []
    words = len(answer.split())
    if words > MAX_ANSWER_WORDS:
        errors.append(f"{words} words; max {MAX_ANSWER_WORDS}")
    if _EMOJI_RE.search(answer):
        errors.append("contains an emoji")
    if re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", answer):
        errors.append("uses a list; write prose")
    profile_text = profile.model_dump_json()
    job_text = json.dumps(job, default=str, ensure_ascii=False)
    prose = _EMAIL_RE.sub(" ", answer)
    for u in extract_urls(prose):
        prose = prose.replace(u.raw_value, " ")
    unknown = sorted(n for n in _numbers(prose) if n not in _numbers(profile_text) | _numbers(job_text))
    if unknown:
        errors.append(f"uses numbers not in profile.json or the job post: {', '.join(unknown)}")
    allowed = {url_key(u) for u in profile.allowed_links}
    for u in extract_urls(answer):
        if url_key(u.value) not in allowed:
            errors.append(f"contains a link that isn't yours: {u.value}")
    for m in _NAMED_THING.finditer(answer):
        phrase = m.group(1).rstrip(".+-&")
        if _norm_text(phrase) not in _norm_text(profile_text) and _norm_text(phrase) not in _norm_text(job_text):
            errors.append(f"names {phrase!r}, which isn't in profile.json")
    for pat in never_claim_patterns(profile.never_claim):
        if m := pat.search(answer):
            errors.append(f"mentions {m.group(0)!r} (never_claim)")
    return errors


def standard_answers_for(profile: Profile, track: str | None) -> dict[str, str]:
    """Base answers for this job: 'about_me:ai_ml' style keys override plain 'about_me' for that track."""
    general = {k: v for k, v in profile.standard_answers.items() if v and ":" not in k}
    specific = {k.split(":", 1)[0]: v for k, v in profile.standard_answers.items()
                if v and ":" in k and k.split(":", 1)[1] == track}
    return general | specific


async def draft_answers(questions: list[str], job: dict, profile: Profile, cfg: AppConfig,
                        llm: LLM) -> dict[str, str]:
    """Draft answers to free-text questions. Answers that still break the rules after one
    retry come back empty (the user fills them in)."""
    questions = list(dict.fromkeys(q.strip() for q in questions if q.strip()))[:MAX_QUESTIONS]
    if not questions:
        return {}
    system = (cfg.path("prompts_dir") / "answers.md").read_text(encoding="utf-8")
    request = json.dumps({
        "JOB": {k: job.get(k) for k in JOB_FIELDS},
        "QUESTIONS": questions,
        "STANDARD_ANSWERS": standard_answers_for(profile, job.get("resume_track")),
        "FEATURED_PROJECTS": profile.featured_projects,
        "PROFILE": email_facts(profile) | {"location": profile.location, "notice_period": profile.notice_period,
                                           "expected_ctc": profile.expected_ctc},
    }, indent=1, ensure_ascii=False, sort_keys=True)
    key = content_key("answers", request)
    answers: dict[str, str] = {q: "" for q in questions}
    feedback = ""
    for attempt in range(2):
        try:
            raw = await llm.complete_json(system=system, user=request + feedback, schema=ANSWERS_SCHEMA,
                                          key=key if attempt == 0 else f"{key}.retry", max_tokens=4000,
                                          effort="medium")
            got = _Answers.model_validate(raw).answers
        except (ValidationError, LLMError) as e:
            log.warning("answer drafting failed: %s", e)
            continue
        by_q = {a.question.strip(): a.answer.strip() for a in got}
        bad: dict[str, list[str]] = {}
        for i, q in enumerate(questions):
            ans = by_q.get(q) or (got[i].answer.strip() if i < len(got) else "")
            errs = validate_answer(ans, profile, job)
            if errs:
                bad[q] = errs
            elif ans:
                answers[q] = ans
        if not bad:
            break
        feedback = "\n\nThese answers broke the rules; rewrite them:\n" + "\n".join(
            f"- {q}: {'; '.join(e)}" for q, e in bad.items())
    return answers


# --- answer sheet -------------------------------------------------------------------------------
SHEET_QUESTIONS = (
    "Tell us about yourself.",
    "Why do you want to join {company}?",
    "Why are you a good fit for the {role} role?",
    "What are your key strengths?",
)


def standard_fields(profile: Profile) -> list[tuple[str, str]]:
    first, last = _first_last(profile.name)
    rows = [
        ("Full name", profile.name), ("First name", first), ("Last name", last), ("Email", profile.email),
        ("Phone", profile.phone), ("College", profile.college), ("Degree", profile.degree),
        ("Graduation year", str(profile.grad_year)), ("CGPA", profile.cgpa), ("Location", profile.location),
        ("LinkedIn", profile.linkedin), ("GitHub", profile.github), ("Portfolio", profile.portfolio),
        ("Notice period", profile.notice_period), ("Availability", profile.availability),
        ("Expected CTC", profile.expected_ctc),
    ]
    return [(k, v) for k, v in rows if v]


def sheet_questions(job: dict) -> list[str]:
    company = job.get("company") or "the company"
    return [q.format(company=company, role=job.get("role") or "this") for q in SHEET_QUESTIONS]


def build_answer_sheet(job: dict, links: list[str], answers: dict[str, str], profile: Profile,
                       resume_name: str | None) -> str:
    """Telegram HTML: every value in its own <code> block so a tap copies it."""
    e = lambda s: html.escape(str(s), quote=False)  # noqa: E731
    lines = [f"📋 <b>Answer sheet: {e(job['role'])} — {e(job.get('company') or '?')}</b>  #{job['id']}"]
    for u in links:
        lines.append(f'<a href="{html.escape(u, quote=True)}">{e(u)}</a>')
    if resume_name:
        lines.append(f"Resume to upload: <b>{e(resume_name)}</b> (in your resumes folder)")
    lines.append("")
    for label, value in standard_fields(profile):
        lines.append(f"{e(label)}: <code>{e(value)}</code>")
    for q, a in answers.items():
        lines += ["", f"<b>{e(q)}</b>", f"<code>{e(a)}</code>" if a else "<i>(fill this in yourself)</i>"]
    lines += ["", "Tap any grey block to copy it. Submit on the site yourself, then tap ✔️ Mark submitted."]
    return "\n".join(lines)


# --- prefill -------------------------------------------------------------------------------------
def prefill_supported() -> bool:
    """A visible browser needs a desktop: always on Windows/macOS, on Linux only with a display."""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@dataclass
class PrefillReport:
    url: str
    filled: list[str] = field(default_factory=list)          # labels filled from profile.json
    drafted: dict[str, str] = field(default_factory=dict)    # question -> LLM answer (highlighted)
    uploaded: list[str] = field(default_factory=list)        # file fields that got the resume
    left_blank: list[str] = field(default_factory=list)      # fields the user must handle
    error: str | None = None
    needs_login: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


_COLLECT_FIELDS_JS = r"""
() => {
  const out = [];
  const textOf = (n) => (n && (n.innerText || n.textContent) || '').replace(/\s+/g, ' ').trim();
  document.querySelectorAll('input, textarea, select').forEach((el, i) => {
    const tag = el.tagName.toLowerCase();
    const type = (tag === 'input' ? (el.getAttribute('type') || 'text') : tag).toLowerCase();
    if (['hidden', 'submit', 'button', 'image', 'reset', 'password', 'search'].includes(type)) return;
    const r = el.getBoundingClientRect();
    const visible = (r.width > 0 && r.height > 0) || type === 'file';
    if (!visible || el.disabled || el.readOnly) return;
    let label = el.getAttribute('aria-label') || '';
    if (!label && el.id) {
      const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (l) label = textOf(l);
    }
    if (!label && el.getAttribute('aria-labelledby')) {
      label = el.getAttribute('aria-labelledby').split(/\s+/)
        .map((id) => textOf(document.getElementById(id))).join(' ');
    }
    if (!label) { const l = el.closest('label'); if (l) label = textOf(l); }
    if (!label) {
      const item = el.closest('[role=listitem], .application-question, .field, fieldset, li');
      if (item) label = textOf(item.querySelector('[role=heading], legend, .application-label, label'));
    }
    if (!label) label = el.getAttribute('placeholder') || '';
    el.setAttribute('data-jobpilot-idx', String(i));
    out.push({
      idx: i, tag, type, label: label.slice(0, 300), name: el.getAttribute('name') || '',
      value: (type === 'checkbox' || type === 'radio') ? '' : (el.value || ''),
    });
  });
  return out;
}
"""

_BANNER_JS = r"""
(msg) => {
  if (document.getElementById('jobpilot-banner')) return;
  const b = document.createElement('div');
  b.id = 'jobpilot-banner';
  b.textContent = msg;
  b.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:2147483647;padding:10px 16px;' +
    'background:#f5b700;color:#111;font:600 14px/1.4 system-ui,sans-serif;text-align:center;' +
    'box-shadow:0 2px 6px rgba(0,0,0,.2)';
  document.body.appendChild(b);
}
"""
BANNER = ("JobPilot prefilled this form. Yellow boxes are drafted answers: review everything, "
          "then submit it yourself.")


@dataclass
class FormField:
    idx: int
    tag: str
    type: str
    label: str
    name: str
    value: str = ""

    @property
    def text(self) -> str:
        return f"{self.label} {self.name.replace('_', ' ').replace('[', ' ').replace(']', ' ')}"


def plan_fill(fields: list[FormField], profile: Profile) -> tuple[dict[int, str], list[FormField], list[int], list[str]]:
    """Decide what to do with each field.

    Returns (profile_values by idx, question fields for the LLM, resume upload idxs, labels left blank).
    Fields that already have a value are left alone.
    """
    values: dict[int, str] = {}
    questions: list[FormField] = []
    uploads: list[int] = []
    blank: list[str] = []
    seen_choice_groups: set[str] = set()
    for f in fields:
        label = f.label or f.name or f"field {f.idx}"
        if f.type == "file":
            if _RESUME_FIELD.search(f.text) or not f.label:
                uploads.append(f.idx)
            else:
                blank.append(f"{label} (file)")
            continue
        if f.type in ("checkbox", "radio") or f.tag == "select":
            key = f.name or label
            if key not in seen_choice_groups:
                seen_choice_groups.add(key)
                blank.append(f"{label} (choice)")
            continue
        if f.value.strip():
            continue
        matched, value = match_profile_field(f.text, profile)
        if matched and value:
            values[f.idx] = value
        elif matched:
            blank.append(label)
        elif looks_like_question(f.label, f.tag):
            questions.append(f)
        else:
            blank.append(label)
    return values, questions, uploads, blank


BROWSER_CHOICES = ("auto", "chrome", "msedge", "chromium")
GOOGLE_LOGIN_URL = "https://accounts.google.com/"


def _browser_paths(channel: str) -> list[Path]:
    env = os.environ.get
    if sys.platform == "win32":
        roots = [env("ProgramFiles", r"C:\Program Files"), env("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                 env("LOCALAPPDATA", "")]
        rel = {"chrome": r"Google\Chrome\Application\chrome.exe",
               "msedge": r"Microsoft\Edge\Application\msedge.exe"}.get(channel)
        return [Path(r) / rel for r in roots if r and rel]
    if sys.platform == "darwin":
        app = {"chrome": "Google Chrome", "msedge": "Microsoft Edge"}.get(channel)
        return [Path(f"/Applications/{app}.app/Contents/MacOS/{app}")] if app else []
    import shutil
    names = {"chrome": ("google-chrome", "google-chrome-stable"), "msedge": ("microsoft-edge",)}.get(channel, ())
    return [Path(p) for n in names if (p := shutil.which(n))]


def find_browser(channel: str) -> Path | None:
    """Installed Chrome/Edge executable, for opening a normal (non-automated) window."""
    return next((p for p in _browser_paths(channel) if p.exists()), None)


class Prefiller:
    """Owns one persistent browser context; forms open as tabs and stay open for review.

    browser="auto" tries the installed Google Chrome, then Microsoft Edge, then Playwright's
    bundled Chromium, and sticks with the first that starts. Each gets its own profile folder
    under `user_data_dir`, separate from your everyday browser profile.
    """

    def __init__(self, user_data_dir: Path, *, headless: bool = False, browser: str = "auto"):
        if browser not in BROWSER_CHOICES:
            raise ValueError(f"browser must be one of {BROWSER_CHOICES}")
        self.user_data_dir = Path(user_data_dir)
        self.headless = headless
        self.browser = browser
        self.channel: str | None = None      # what actually launched
        self._pw = None
        self._ctx = None

    def _candidates(self) -> list[str]:
        return ["chrome", "msedge", "chromium"] if self.browser == "auto" else [self.browser]

    async def _context(self):
        """The open browser window, relaunched if the user closed it."""
        if self._ctx is not None:
            return self._ctx
        from playwright.async_api import async_playwright

        if self._pw is None:
            self._pw = await async_playwright().start()
        candidates = [self.channel] if self.channel else self._candidates()
        errors = []
        for channel in candidates:
            profile_dir = self.user_data_dir / channel
            profile_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._ctx = await self._pw.chromium.launch_persistent_context(
                    str(profile_dir), headless=self.headless, viewport=None,
                    channel=None if channel == "chromium" else channel,
                    # look less like an automated browser (fewer "browser may not be secure" blocks)
                    ignore_default_args=["--enable-automation"],
                    args=["--disable-blink-features=AutomationControlled"]
                         + ([] if self.headless else ["--start-maximized"]),
                )
            except Exception as e:  # noqa: BLE001 - try the next browser
                msg = str(e)
                if re.search(r"ProcessSingleton|already in use|profile.*in use|existing browser session|lock",
                             msg, re.I):
                    raise RuntimeError("JobPilot's browser profile is already open in another window (the "
                                       "/login window, or another JobPilot run): close that window completely, "
                                       "then try again") from e
                errors.append(f"{channel}: {msg.splitlines()[0][:150]}")
                continue
            self.channel = channel
            self._ctx.on("close", lambda *_: setattr(self, "_ctx", None))
            log.info("prefill browser: %s", channel)
            return self._ctx
        raise RuntimeError("no browser could be started (" + "; ".join(errors) + "). Install Google Chrome, "
                           "or run: python -m playwright install --force chromium")

    async def close(self) -> None:
        try:
            if self._ctx is not None:
                await self._ctx.close()
        finally:
            self._ctx = None
            if self._pw is not None:
                await self._pw.stop()
                self._pw = None

    async def open_for_login(self, url: str = GOOGLE_LOGIN_URL, popen=None) -> str:
        """Open a NORMAL (not automated) browser window on JobPilot's profile so the user can sign in
        to Google; sites refuse sign-in inside automated windows. Cookies land in the same profile,
        so later prefills are signed in. Returns the browser used."""
        import subprocess

        await self.close()   # the profile can only be open in one browser at a time
        for channel in ([self.channel] if self.channel in ("chrome", "msedge") else ["chrome", "msedge"]):
            exe = find_browser(channel)
            if exe is None:
                continue
            profile_dir = self.user_data_dir / channel
            profile_dir.mkdir(parents=True, exist_ok=True)
            (popen or subprocess.Popen)([str(exe), f"--user-data-dir={profile_dir}", "--no-first-run",
                                         "--no-default-browser-check", url])
            self.channel = channel
            return channel
        raise RuntimeError("Google Chrome or Microsoft Edge isn't installed, so a normal sign-in window "
                           "can't be opened")

    async def prefill(self, url: str, link_type: str, job: dict, profile: Profile, cfg: AppConfig, llm: LLM,
                      resume: Path | None) -> PrefillReport:
        """Open the form, fill profile fields, upload the resume, draft free-text answers.
        Never clicks anything; the tab stays open for the user to review and submit."""
        target = application_url(url, link_type)
        report = PrefillReport(target)
        try:
            ctx = await self._context()
            page = await ctx.new_page()
            await page.goto(target, wait_until="domcontentloaded", timeout=45_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:  # noqa: BLE001 - busy pages never go idle; fields are usually there
                pass
            if "accounts.google.com" in page.url:
                report.needs_login = True
                report.error = ("this form needs a Google sign-in. Google blocks sign-in inside the "
                                "automated window, so send /login: it opens a normal window where you sign "
                                "in once (it's remembered). Close that window, then tap Prefill again")
                await page.close()
                return report
            raw = await page.evaluate(_COLLECT_FIELDS_JS)
            fields = [FormField(**{k: r[k] for k in ("idx", "tag", "type", "label", "name", "value")}) for r in raw]
            if not fields:
                report.error = "no form fields found on the page (it may need a click to open the form)"
                return report
            values, questions, uploads, blank = plan_fill(fields, profile)
            labels = {f.idx: f.label or f.name for f in fields}

            for idx, value in values.items():
                try:
                    await page.locator(f'[data-jobpilot-idx="{idx}"]').fill(value, timeout=5_000)
                    report.filled.append(labels[idx])
                except Exception as e:  # noqa: BLE001
                    log.debug("fill %s failed: %s", labels[idx], e)
                    report.left_blank.append(labels[idx])
            for idx in uploads:
                if resume is None or not resume.exists():
                    report.left_blank.append(f"{labels[idx] or 'resume'} (file: resume missing)")
                    continue
                try:
                    await page.locator(f'[data-jobpilot-idx="{idx}"]').set_input_files(str(resume), timeout=5_000)
                    report.uploaded.append(labels[idx] or "resume")
                except Exception as e:  # noqa: BLE001
                    log.debug("upload failed: %s", e)
                    report.left_blank.append(f"{labels[idx] or 'resume'} (file)")

            if questions:
                answers = await draft_answers([q.label for q in questions], job, profile, cfg, llm)
                for q in questions:
                    ans = answers.get(q.label.strip(), "")
                    if not ans:
                        report.left_blank.append(q.label)
                        continue
                    loc = page.locator(f'[data-jobpilot-idx="{q.idx}"]')
                    try:
                        await loc.fill(ans, timeout=5_000)
                        await loc.evaluate("e => { e.style.outline = '3px solid #f5b700';"
                                           " e.style.background = '#fff8d6'; }")
                        report.drafted[q.label] = ans
                    except Exception as e:  # noqa: BLE001
                        log.debug("fill answer failed: %s", e)
                        report.left_blank.append(q.label)
            report.left_blank += blank
            await page.evaluate(_BANNER_JS, BANNER)
            await page.bring_to_front()
        except Exception as e:  # noqa: BLE001 - caller falls back to the answer sheet
            log.exception("prefill failed for %s", target)
            report.error = f"{type(e).__name__}: {e}"
        return report
