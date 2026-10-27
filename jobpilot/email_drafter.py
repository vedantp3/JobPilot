"""Draft + validate tailored application emails.

The LLM writes only the body paragraphs. Code builds the subject, greeting and
signature, and `validate_email` enforces the spec's hard constraints (plus: no emojis).
A failing draft is regenerated up to MAX_REGENERATIONS times with the errors fed back.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from jobpilot.config import AppConfig, Profile
from jobpilot.llm import LLM, LLMError, content_key
from jobpilot.normalize import _EMAIL_RE, _EMOJI_RE, extract_urls, url_key

log = logging.getLogger(__name__)

MAX_WORDS = 150
MAX_REGENERATIONS = 2


# --- fixed parts --------------------------------------------------------------
def _strip_parens(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", s).strip()


def build_subject(role: str, profile: Profile) -> str:
    return f"{role.strip()} — {profile.name} ({profile.college}, {profile.grad_year})"


_REQUESTED_SUBJECT = re.compile(
    r"(?im)\bsubject(?:\s*line)?\s*"
    r"(?:[:\-–]\s*[\"'“‘]?(?P<s1>[^\n\"'”’]{3,120})"                     # Subject: X  /  Subject - "X"
    r"|(?:should\s+be|must\s+be|as)\s*[:\-–]?\s*[\"'“‘](?P<s2>[^\n\"'”’]{3,120})[\"'”’])"  # subject as 'X'
)
_NAME_PLACEHOLDER = re.compile(r"[<\[{(]\s*(?:your\s*)?(?:full\s*)?name\s*[>\]})]", re.I)
_ROLE_PLACEHOLDER = re.compile(r"[<\[{(]\s*(?:job\s*|role\s*)?(?:role|position|title)\s*[>\]})]", re.I)
_ANY_PLACEHOLDER = re.compile(r"[<\[{]\s*[^>\]}]{1,30}[>\]}]")


def requested_subject(post_text: str, role: str, profile: Profile) -> str | None:
    """The subject line a post explicitly asks for ("Subject: ML Intern - <Your Name>"), filled in.

    Returns None if there's none, or if it has placeholders we can't fill.
    """
    m = _REQUESTED_SUBJECT.search(post_text or "")
    if not m:
        return None
    s = _EMOJI_RE.sub("", m.group("s1") or m.group("s2")).strip(" .,:;")
    s = _NAME_PLACEHOLDER.sub(profile.name, s)
    s = _ROLE_PLACEHOLDER.sub(role, s)
    if _ANY_PLACEHOLDER.search(s) or len(s) < 3:
        return None
    return s


def build_greeting(person_name: str | None, company: str | None) -> str:
    if person_name and person_name.strip():
        return f"Hi {person_name.strip().split()[0].title()},"
    return f"Hi {_strip_parens(company)} Team," if company and _strip_parens(company) else "Hi Hiring Team,"


def build_signature(profile: Profile) -> str:
    return f"Best,\n{profile.name}\n{profile.phone}"


def assemble(greeting: str, body: str, profile: Profile) -> str:
    return f"{greeting}\n\n{body.strip()}\n\n{build_signature(profile)}"


def split_signature(full: str, profile: Profile) -> str:
    """Email text without the signature block."""
    sig = build_signature(profile)
    return full[: -len(sig)].rstrip() if full.endswith(sig) else full


def attachment_name(profile: Profile) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", profile.name).strip("_") + "_Resume.pdf"


# --- validator -----------------------------------------------------------------
_BANNED_PHRASES = ("writing to apply", "interested in applying")
_LIST_LINE = re.compile(r"(?m)^\s*(?:[-*•·▪◦‣–]|\d+[.)])\s+")
_SIGNOFF_LINE = re.compile(
    r"(?im)^\s*(?:best(?: regards| wishes)?|regards|warm regards|kind regards|sincerely|cheers|thanks|"
    r"thank you|yours (?:truly|sincerely))[,!.]?\s*$"
)
# standalone numbers only: digits inside words (n8n, S/4HANA, EfficientNetV2B0) aren't metrics
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?![A-Za-z])")
_NAMED_THING = re.compile(
    r"\b(?:built|developed|created|shipped|designed|led|architected|my)\s+(?:an?\s+|the\s+)?"
    r"((?:[A-Z][\w&.+-]*)(?:\s+[A-Z][\w&.+-]*)*)"
)
_GENERIC_FIRST_WORDS = {"global", "the", "indian", "national", "data", "smart", "tech", "digital"}
_NEVER_CLAIM_FILLER = {"experience", "project", "projects", "contributions", "contribution", "work",
                       "any", "skills", "skill"}


def _norm_text(s: str) -> str:
    return re.sub(r"[\s\-_]+", " ", s.lower())


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def org_aliases(org: str) -> list[str]:
    full = org.strip()
    base = _strip_parens(full)
    aliases = [full, base]
    first = base.split()[0] if base else ""
    if len(first) >= 5 and first.lower() not in _GENERIC_FIRST_WORDS:
        aliases.append(first)
    return list(dict.fromkeys(a for a in aliases if a))


def never_claim_patterns(items: list[str]) -> list[re.Pattern]:
    pats = []
    for item in items:
        if item.lower().startswith("any "):          # meta rule; enforced by the other checks
            continue
        words = [w for w in re.split(r"[\s\-]+", item) if w.lower() not in _NEVER_CLAIM_FILLER]
        if words:
            pats.append(re.compile(r"\b" + r"[\s\-]+".join(re.escape(w) for w in words) + r"\b", re.I))
        pats += [re.compile(rf"\b{re.escape(w)}\b") for w in words if w.isupper() and len(w) >= 2]
    return pats


def validate_email(subject: str, full_text: str, profile: Profile, job: dict,
                   projects_used: list[str] = (), employers_used: list[str] = (),
                   max_words: int = MAX_WORDS) -> list[str]:
    """Return a list of human-readable violations (empty = valid)."""
    errors: list[str] = []
    text = split_signature(full_text, profile)
    low = text.lower()

    words = len(re.findall(r"\S+", text))
    if words >= max_words:
        errors.append(f"{words} words; must be under {max_words} excluding the signature")
    if _LIST_LINE.search(text):
        errors.append("contains a bullet point or numbered list; use plain paragraphs")
    for phrase in _BANNED_PHRASES:
        if phrase in low:
            errors.append(f"uses the banned phrase {phrase!r}")
    if _EMOJI_RE.search(text) or _EMOJI_RE.search(subject):
        errors.append("contains an emoji or decorative symbol")
    body_lines = text.split("\n")[1:]  # skip the greeting
    if any(_SIGNOFF_LINE.match(l) for l in body_lines):
        errors.append("contains its own sign-off; the signature is added automatically")

    # numbers: at least one non-year metric, and every number must come from profile or job.
    # Digits inside links/addresses (e.g. 'vedantp3') don't count.
    profile_text = profile.model_dump_json()
    job_text = json.dumps(job, default=str, ensure_ascii=False)
    allowed_numbers = _numbers(profile_text) | _numbers(job_text)
    prose = _EMAIL_RE.sub(" ", text)
    for u in extract_urls(prose):
        prose = prose.replace(u.raw_value, " ")
    nums = _numbers(prose)
    metrics = {n for n in nums if not re.fullmatch(r"(?:19|20)\d\d", n)}
    if not metrics:
        errors.append("has no digit-bearing metric (e.g. '30% better relevance')")
    unknown = sorted(n for n in nums if n not in allowed_numbers)
    if unknown:
        errors.append(f"uses numbers not found in profile.json or the job post: {', '.join(unknown)}")

    # projects / employers
    project_names = {p.name.lower(): p.name for p in profile.projects}
    aliases = {a.lower() for e in profile.experience for a in org_aliases(e.org)}
    mentioned = [p for p in project_names if p in low] + [a for a in aliases if a in low]
    if not mentioned:
        errors.append("doesn't name any project or experience entry from profile.json")
    for name in projects_used:
        if name.lower() not in project_names:
            errors.append(f"names project {name!r}, which isn't in profile.json")
    for name in employers_used:
        if name.lower() not in aliases and _strip_parens(name).lower() not in aliases:
            errors.append(f"names employer {name!r}, which isn't in profile.json")
    profile_low = _norm_text(profile_text)
    for m in _NAMED_THING.finditer(text):
        phrase = m.group(1).rstrip(".+-&")
        if _norm_text(phrase) not in profile_low and _norm_text(phrase) not in _norm_text(job_text):
            errors.append(f"names {phrase!r}, which isn't in profile.json")

    # links and addresses
    allowed_links = {url_key(u) for u in profile.allowed_links}
    for u in extract_urls(text):
        if url_key(u.value) not in allowed_links:
            errors.append(f"contains a link that isn't your GitHub/portfolio/LinkedIn: {u.value}")
    for m in _EMAIL_RE.finditer(text):
        if m.group(0).lower() != profile.email.lower():
            errors.append(f"contains an email address: {m.group(0)}")
    present = {url_key(u.value) for u in extract_urls(text)}
    for label, url in (("GitHub", profile.github), ("portfolio", profile.portfolio)):
        # accept "https://x.app", "x.app" or "www.x.app" (bare domains aren't extracted as links)
        if url and url_key(url) not in present and url_key(url) not in low:
            errors.append(f"missing your {label} URL ({url})")
    if "resume attached" not in low:
        errors.append("missing the sentence 'Resume attached.'")

    for pat in never_claim_patterns(profile.never_claim):
        if m := pat.search(text):
            errors.append(f"mentions {m.group(0)!r}, which is on your never_claim list")
    return list(dict.fromkeys(errors))


# --- LLM drafting -----------------------------------------------------------------
class EmailOut(BaseModel):
    body: str
    projects_used: list[str] = Field(default_factory=list)
    employers_used: list[str] = Field(default_factory=list)


EMAIL_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["body", "projects_used", "employers_used"],
    "properties": {
        "body": {"type": "string"},
        "projects_used": {"type": "array", "items": {"type": "string"}},
        "employers_used": {"type": "array", "items": {"type": "string"}},
    },
}

JOB_EMAIL_FIELDS = ("company", "role", "role_category", "location", "work_mode", "stipend_or_ctc",
                    "required_skills", "jd_summary")


def skill_gaps(job: dict) -> list[str]:
    return [f.split(":", 1)[1].strip() for f in job.get("flags") or [] if f.startswith("skill gap:")]


def email_facts(profile: Profile) -> dict:
    return profile.model_dump(include={
        "name", "college", "degree", "grad_year", "cgpa", "availability", "github", "portfolio",
        "linkedin", "certifications", "skills", "projects", "experience", "achievements", "never_claim",
    })


def build_request(job: dict, also_roles: list[str], recipient: str, track: str,
                  profile: Profile) -> str:
    request = {
        "JOB": {k: job.get(k) for k in JOB_EMAIL_FIELDS},
        "ALSO_ROLES": also_roles,
        "GAPS": skill_gaps(job),
        "RECIPIENT": recipient,
        "RESUME_TRACK": track,
        "PROFILE": email_facts(profile),
    }
    if profile.featured_projects:
        request["FEATURED_PROJECTS"] = profile.featured_projects
    return json.dumps(request, indent=1, ensure_ascii=False, sort_keys=True)


@dataclass
class Draft:
    subject: str
    text: str                      # full email incl. greeting and signature
    errors: list[str] = field(default_factory=list)
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


async def draft_email(job: dict, contact: dict, also_jobs: list[dict], profile: Profile,
                      cfg: AppConfig, llm: LLM, post_text: str = "") -> Draft:
    """Generate + validate; regenerate up to MAX_REGENERATIONS times. Returns the best attempt.

    Uses the subject line the post asks for, if any; otherwise the standard format.
    """
    system = load_email_prompt(cfg)
    greeting = build_greeting(contact.get("person_name"), job.get("company"))
    subject = requested_subject(post_text, job["role"], profile) or build_subject(job["role"], profile)
    request = build_request(job, [j["role"] for j in also_jobs], greeting, job.get("resume_track") or "ai_ml",
                            profile)
    return await _generate(llm, system, request, content_key("email", request), greeting, subject, job, profile,
                           cfg.email_max_words)


LOCAL_EXAMPLES = "email_examples.local.md"   # your own example emails; private (gitignored)


def load_email_prompt(cfg: AppConfig) -> str:
    prompts = cfg.path("prompts_dir")
    text = (prompts / "email.md").read_text(encoding="utf-8")
    if (local := prompts / LOCAL_EXAMPLES).exists():
        text = text.rstrip() + "\n\n" + local.read_text(encoding="utf-8")
    return text.replace("{max_words}", str(cfg.email_max_words))


REVISE_INSTRUCTIONS = """

## Revising an existing email
You are revising CURRENT_BODY according to the candidate's INSTRUCTION. Change only what the instruction asks for and keep everything else. If INSTRUCTION is itself a complete email body, use its wording as far as the rules above allow. All rules above still apply; if the instruction conflicts with a hard constraint, follow the constraint."""


def body_of(full_text: str, profile: Profile) -> tuple[str, str]:
    """Split a stored email into (greeting, body), dropping the signature."""
    text = split_signature(full_text, profile)
    greeting, _, body = text.partition("\n")
    return greeting.strip(), body.strip()


async def revise_email(job: dict, contact: dict, also_jobs: list[dict], profile: Profile, cfg: AppConfig,
                       llm: LLM, current_text: str, subject: str, instruction: str) -> Draft:
    """Apply the user's edit instruction to an existing email; same validation and retries."""
    system = load_email_prompt(cfg) + REVISE_INSTRUCTIONS
    greeting, current_body = body_of(current_text, profile)
    greeting = greeting or build_greeting(contact.get("person_name"), job.get("company"))
    base = json.loads(build_request(job, [j["role"] for j in also_jobs], greeting,
                                    job.get("resume_track") or "ai_ml", profile))
    request = json.dumps({**base, "CURRENT_BODY": current_body, "INSTRUCTION": instruction},
                         indent=1, ensure_ascii=False, sort_keys=True)
    return await _generate(llm, system, request, content_key("revise", request), greeting, subject, job, profile,
                           cfg.email_max_words)


async def _generate(llm: LLM, system: str, request: str, key: str, greeting: str, subject: str,
                    job: dict, profile: Profile, max_words: int = MAX_WORDS) -> Draft:
    best: Draft | None = None
    feedback = ""
    for attempt in range(MAX_REGENERATIONS + 1):
        user = request + feedback
        try:
            raw = await llm.complete_json(system=system, user=user, schema=EMAIL_SCHEMA,
                                          key=key if attempt == 0 else f"{key}.retry{attempt}",
                                          max_tokens=4000, effort="medium")
            out = EmailOut.model_validate(raw)
        except (ValidationError, LLMError) as e:
            errors = [f"generation failed: {str(e).splitlines()[0]}"]
            draft = Draft(subject, "", errors, attempt + 1)
        else:
            text = assemble(greeting, out.body, profile)
            errors = validate_email(subject, text, profile, job, out.projects_used, out.employers_used,
                                    max_words=max_words)
            draft = Draft(subject, text, errors, attempt + 1)
        if draft.ok:
            return draft
        # keep the attempt with text and the fewest violations
        if best is None or (draft.text and (not best.text or len(draft.errors) < len(best.errors))):
            best = draft
        log.info("email draft attempt %d for %r failed: %s", attempt + 1, job["role"], "; ".join(draft.errors))
        feedback = ("\n\nYour previous body broke these rules; rewrite it fixing every one:\n- "
                    + "\n- ".join(draft.errors))
    return best
