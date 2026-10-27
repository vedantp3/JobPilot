"""Hard rules + LLM fit check -> proceed / flag / skip.

Flags are always shown to the user, never hidden. Hard-rule skips don't call the LLM.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from jobpilot import db
from jobpilot.config import AppConfig, Profile
from jobpilot.llm import LLM, LLMError, content_key

log = logging.getLogger(__name__)

PROCEED, FLAG, SKIP = "proceed", "flag", "skip"
FLAG_BAND = 10  # scores in [threshold, threshold + FLAG_BAND) are flagged


# --- batch --------------------------------------------------------------------
def check_batch(batch: str | None, target: int) -> tuple[str, str | None]:
    """Return ("ok" | "excluded" | "unknown", note).

    Understands "2025/2026", "2024-2026", "2025 onwards", "2024 or earlier", "2025 & above".
    """
    if not batch or not batch.strip():
        return "unknown", "batch not stated"
    s = batch.lower()
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", s)]
    if not years:
        return "unknown", f"batch unclear ({batch})"
    if re.search(r"(20\d\d)\s*(?:-|–|to)\s*(20\d\d)", s):
        lo, hi = min(years), max(years)
        ok = lo <= target <= hi
    elif re.search(r"onwards|or later|and later|& later|and above|& above|\+|or after|and after", s):
        ok = target >= min(years)
    elif re.search(r"or earlier|and earlier|or before|and before|& below|and below|or older", s):
        ok = target <= max(years)
    else:
        ok = target in years
    return ("ok", None) if ok else ("excluded", f"batch {batch} excludes {target}")


# --- keywords / scams -----------------------------------------------------------
SCAM_PATTERNS = (
    r"\b(?:registration|processing|training|joining|enrol?l?ment|application|onboarding|"
    r"verification|kit|interview)\s+(?:fees?|charges?|amount)\b",
    r"\bsecurity\s+(?:deposit|fees?|amount)\b",
    r"\bpay\s+to\s+apply\b",
    r"\brefundable\s+(?:amount|fees?|deposit)\b",
    r"\bfees?\s+of\s+(?:rs\.?|inr|₹)\s*\d",
)
_NEGATION = re.compile(r"\b(?:no|zero|without|not|never|free of|don'?t|doesn'?t)\b[\w\s]{0,15}$", re.I)


def _find(pattern: str, text: str) -> str | None:
    """First non-negated match of `pattern` in text ("no registration fee" doesn't count)."""
    for m in re.finditer(pattern, text, re.I):
        if not _NEGATION.search(text[max(0, m.start() - 30):m.start()]):
            return m.group(0)
    return None


def keyword_pattern(keyword: str) -> str:
    return r"\b" + r"\s+".join(re.escape(w) for w in keyword.split()) + r"\b"


def find_reject_keywords(text: str, keywords: list[str]) -> list[str]:
    return [k for k in keywords if _find(keyword_pattern(k), text)]


def find_scam(text: str) -> str | None:
    for p in SCAM_PATTERNS:
        if hit := _find(p, text):
            return hit
    return None


# --- stipend --------------------------------------------------------------------
def parse_monthly_inr(s: str | None) -> int | None:
    """Best-effort monthly INR from strings like '₹30,000/month', '20k pm', '4.5 LPA', '6-8 LPA'.

    Returns the lower bound of a range, 0 for unpaid, None if not parseable.
    """
    if not s:
        return None
    t = s.lower().replace(",", "")
    if re.search(r"\bunpaid\b|\bno stipend\b", t):
        return 0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(k|l\b|lpa|lakhs?|lacs?|lac|cr)?", t)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2) or ""
    # a unit may only appear after the second number of a range ("6-8 LPA")
    if not unit:
        rng = re.search(r"\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*(k|l\b|lpa|lakhs?|lacs?|cr)", t)
        if rng:
            unit = rng.group(1)
    if unit == "k":
        value *= 1_000
    elif unit in ("l", "lpa", "lakh", "lakhs", "lac", "lacs"):
        value *= 100_000
    elif unit == "cr":
        value *= 10_000_000
    annual = bool(re.search(r"lpa|per annum|p\.?a\.?\b|/\s*(?:year|yr|annum)|per year|annual|ctc", t))
    monthly = bool(re.search(r"/\s*(?:month|mo)\b|per month|\bpm\b|p\.m\.|monthly|/m\b", t))
    if unit in ("l", "lakh", "lakhs", "lac", "lacs", "lpa", "cr") and not monthly:
        annual = True
    if not annual and not monthly:
        annual = value >= 100_000  # bare big numbers are yearly CTC
    return int(value / 12 if annual else value)


_RANGE_SEP = re.compile(r"\d+(?:\.\d+)?\s*(?:k|l\b|lpa|lakhs?|lacs?)?\s*(?:-|–|to)\s*(?=\d)", re.I)


def parse_monthly_range(s: str | None) -> tuple[int, int] | None:
    """(low, high) monthly INR. A single figure gives (x, x); '15-25k/month' gives (15000, 25000)."""
    lo = parse_monthly_inr(s)
    if lo is None:
        return None
    t = s.lower().replace(",", "")
    m = _RANGE_SEP.search(t)
    hi = parse_monthly_inr(t[m.end():]) if m else None
    return lo, max(lo, hi if hi is not None else lo)


def stipend_flags(stipend: str | None, min_monthly: int) -> tuple[str | None, list[str]]:
    """Return (skip_reason, flags). With a minimum set, a job is skipped only when even the top of
    its pay range is below it; a range that merely starts below it is flagged."""
    if not stipend:
        return None, ["stipend/CTC not stated"]
    flags = []
    if re.search(r"performance|incentive|commission", stipend, re.I):
        flags.append(f"pay is performance/incentive based ({stipend})")
    rng = parse_monthly_range(stipend)
    if rng is None:
        return None, flags
    lo, hi = rng
    if hi == 0:
        flags.append("unpaid")
    elif hi < 5_000:
        flags.append(f"very low pay (~₹{hi:,}/month)")
    elif lo > 250_000:
        flags.append(f"unusually high pay (~₹{lo:,}/month), double-check the post")
    if min_monthly and hi < min_monthly:
        return f"pay ~₹{hi:,}/month is below your minimum of ₹{min_monthly:,}/month", flags
    if min_monthly and lo < min_monthly:
        flags.append(f"pay range starts below your ₹{min_monthly:,}/month minimum (₹{lo:,}-₹{hi:,})")
    return None, flags


# --- location / contacts -----------------------------------------------------------
def location_flags(location: str | None, work_mode: str | None, ok: list[str]) -> list[str]:
    if work_mode == "remote" or not location:
        return []
    loc = location.lower()
    if any(o.lower() in loc for o in ok if o.lower() != "any"):
        return []
    if "any" in [o.lower() for o in ok] and re.search(r"\bany\b|anywhere|pan india|multiple", loc):
        return []
    return [f"location {location!r} isn't in your locations_ok"]


PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.in", "yahoo.co.in", "outlook.com",
    "hotmail.com", "live.com", "rediffmail.com", "icloud.com", "proton.me", "protonmail.com",
    "aol.com", "zoho.com", "ymail.com",
}


def contact_flags(contacts: list[dict]) -> list[str]:
    flags = []
    for c in contacts:
        if c["kind"] == "email" and c["value"].split("@")[-1] in PERSONAL_EMAIL_DOMAINS:
            flags.append(f"{c['value']} is a personal email address, not a company domain (heads-up)")
    return flags


# --- LLM fit check -------------------------------------------------------------------
class FitResult(BaseModel):
    fit_score: int
    fit_reason: str
    gaps: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)

    @field_validator("fit_score")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(0, min(100, v))


FIT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fit_score", "fit_reason", "gaps", "flags"],
    "properties": {
        "fit_score": {"type": "integer"},
        "fit_reason": {"type": "string"},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "flags": {"type": "array", "items": {"type": "string"}},
    },
}

JOB_FIT_FIELDS = (
    "company", "role", "role_category", "batch_eligibility", "location", "work_mode",
    "stipend_or_ctc", "required_skills", "jd_summary", "deadline",
)


def profile_facts(profile: Profile) -> dict[str, Any]:
    """Profile fields relevant to fit (no phone/email)."""
    return profile.model_dump(include={
        "college", "degree", "grad_year", "cgpa", "location", "availability", "certifications",
        "skills", "projects", "experience", "achievements", "never_claim",
    })


def fit_request(job: dict, profile: Profile, cfg: AppConfig) -> tuple[str, str]:
    """Return (user_message, replay_key) for the fit check."""
    payload = {
        "JOB": {k: job.get(k) for k in JOB_FIT_FIELDS},
        "PROFILE": profile_facts(profile),
        "PREFERENCES": {"target_batch": cfg.target_batch, "locations_ok": cfg.locations_ok},
    }
    user = json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True)
    return user, content_key("fit", user)


async def fit_check(job: dict, profile: Profile, cfg: AppConfig, llm: LLM) -> FitResult:
    system = (cfg.path("prompts_dir") / "fit.md").read_text(encoding="utf-8")
    user, key = fit_request(job, profile, cfg)
    error = None
    for attempt in range(2):
        msg = user if error is None else f"{user}\n\nYour previous answer was invalid ({error}). Return corrected JSON."
        try:
            raw = await llm.complete_json(system=system, user=msg, schema=FIT_SCHEMA,
                                          key=key if attempt == 0 else f"{key}.retry", max_tokens=2000,
                                          effort="medium")
            return FitResult.model_validate(raw)
        except (ValidationError, LLMError) as e:
            error = str(e)[:500]
    raise LLMError(f"fit check failed twice: {error}")


# --- decision ----------------------------------------------------------------------
@dataclass
class FilterResult:
    decision: str                 # proceed | flag | skip
    fit_score: int | None = None
    fit_reason: str | None = None
    gaps: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    scam: bool = False

    @property
    def status(self) -> str:
        return {PROCEED: db.EXTRACTED, FLAG: db.FLAGGED, SKIP: db.SKIPPED_FILTER}[self.decision]

    def all_flags(self) -> list[str]:
        out = list(self.flags) + [f"skill gap: {g}" for g in self.gaps]
        if self.skip_reason:
            out.insert(0, f"skipped: {self.skip_reason}")
        return out


def score_decision(score: int, threshold: int) -> str:
    if score < threshold:
        return SKIP
    return FLAG if score < threshold + FLAG_BAND else PROCEED


def hard_rules(job: dict, post_text: str, cfg: AppConfig) -> FilterResult | None:
    """Return a SKIP result if a hard rule fires, else None."""
    batch_state, batch_note = check_batch(job.get("batch_eligibility"), cfg.target_batch)
    if batch_state == "excluded":
        return FilterResult(SKIP, skip_reason=batch_note)

    haystack = "\n".join(filter(None, [post_text, job.get("role"), job.get("jd_summary"),
                                       job.get("stipend_or_ctc")]))
    scam = find_scam(haystack)
    rejects = find_reject_keywords(haystack, cfg.reject_keywords)
    fee_keywords = [k for k in rejects if re.search(r"fee|deposit|pay to apply", k, re.I)]
    if scam or fee_keywords:
        evidence = scam or fee_keywords[0]
        return FilterResult(SKIP, skip_reason=f"asks candidates to pay ({evidence!r})",
                            flags=[f"likely scam: asks for money ({evidence!r})"], scam=True)
    if rejects:
        return FilterResult(SKIP, skip_reason=f"reject keyword(s): {', '.join(rejects)}")

    if job.get("role_category") not in cfg.accept_roles:
        return FilterResult(SKIP, skip_reason=f"role category {job.get('role_category')!r} not in accept_roles")
    return None


async def evaluate(job: dict, contacts: list[dict], post_text: str, profile: Profile,
                   cfg: AppConfig, llm: LLM) -> FilterResult:
    if (hard := hard_rules(job, post_text, cfg)) is not None:
        return hard

    flags: list[str] = []
    _, batch_note = check_batch(job.get("batch_eligibility"), cfg.target_batch)
    if batch_note:
        flags.append(batch_note)
    pay_skip, pay_flags = stipend_flags(job.get("stipend_or_ctc"), cfg.min_stipend_monthly_inr)
    flags += pay_flags
    if pay_skip:
        return FilterResult(SKIP, skip_reason=pay_skip, flags=flags)
    flags += location_flags(job.get("location"), job.get("work_mode"), cfg.locations_ok)
    flags += contact_flags(contacts)

    try:
        fit = await fit_check(job, profile, cfg, llm)
    except LLMError as e:
        log.warning("fit check failed for job %s: %s", job.get("id"), e)
        return FilterResult(FLAG, flags=flags + [f"fit check failed, review manually ({e})"])
    decision = score_decision(fit.fit_score, cfg.fit_threshold)
    return FilterResult(
        decision, fit_score=fit.fit_score, fit_reason=fit.fit_reason, gaps=fit.gaps,
        flags=flags + fit.flags,
        skip_reason=f"fit {fit.fit_score} below threshold {cfg.fit_threshold}" if decision == SKIP else None,
    )
