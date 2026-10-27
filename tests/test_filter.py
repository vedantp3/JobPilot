import pytest

from jobpilot import db
from jobpilot.config import PROJECT_ROOT, AppConfig, load_profile
from jobpilot.filter import (
    FLAG, PROCEED, SKIP, check_batch, contact_flags, evaluate, find_reject_keywords, find_scam,
    hard_rules, location_flags, parse_monthly_inr, score_decision, stipend_flags,
)
from jobpilot.llm import LLMError

CFG = AppConfig(
    reject_keywords=["unpaid", "pay to apply", "registration fee", "security deposit"],
    locations_ok=["remote", "pune", "mumbai", "navi mumbai", "bangalore", "hyderabad", "any"],
)
PROFILE = load_profile(PROJECT_ROOT / "profile.example.json")


@pytest.mark.parametrize("batch, state", [
    ("2025/2026", "ok"),
    ("2026", "ok"),
    ("2024 and 2025 ONLY", "excluded"),
    ("2024/2025 only", "excluded"),
    ("2024-2026", "ok"),
    ("2022 to 2025", "excluded"),
    ("2025 onwards", "ok"),
    ("2027 and above", "excluded"),
    ("2026 or earlier", "ok"),
    ("2024 or earlier", "excluded"),
    ("2025/2026, 0-1 yrs", "ok"),
    ("0-1 years experience", "unknown"),
    (None, "unknown"),
])
def test_check_batch(batch, state):
    assert check_batch(batch, 2026)[0] == state


def test_reject_keywords_word_boundaries_and_negation():
    assert find_reject_keywords("This is an UNPAID internship", ["unpaid"]) == ["unpaid"]
    assert find_reject_keywords("prepaid cards team", ["unpaid"]) == []
    assert find_reject_keywords("There is no registration fee.", ["registration fee"]) == []
    assert find_reject_keywords("registration   fee of 500", ["registration fee"]) == ["registration fee"]


@pytest.mark.parametrize("text, hit", [
    ("A refundable registration fee of Rs. 999 is required", True),
    ("Pay a security deposit of 2000", True),
    ("training fees apply", True),
    ("fee of ₹1500 for the kit", True),
    ("No registration fee, no charges at all", False),
    ("We never charge any registration fee", False),
    ("Stipend: 20k, free of training fees", False),
])
def test_scam_patterns(text, hit):
    assert bool(find_scam(text)) is hit


@pytest.mark.parametrize("s, monthly", [
    ("₹30,000/month", 30000),
    ("20k/month", 20000),
    ("25,000 INR per month", 25000),
    ("15k pm", 15000),
    ("4.5 LPA", 37500),
    ("6-8 LPA", 50000),
    ("Rs 600000 per annum", 50000),
    ("CTC 12 L", 100000),
    ("unpaid", 0),
    ("performance based", None),
    (None, None),
])
def test_parse_monthly_inr(s, monthly):
    assert parse_monthly_inr(s) == monthly


def test_stipend_flags():
    assert stipend_flags(None, 0) == (None, ["stipend/CTC not stated"])
    assert stipend_flags("₹2,000/month", 0)[1] == ["very low pay (~₹2,000/month)"]
    assert "performance" in stipend_flags("Performance-based incentives only", 0)[1][0]
    skip, _ = stipend_flags("10k/month", 15000)
    assert skip and "below your minimum" in skip
    assert stipend_flags("30k/month", 15000) == (None, [])


@pytest.mark.parametrize("pay, outcome", [
    ("₹15,000/month", "skip"),
    ("19999 per month", "skip"),
    ("20k/month", "ok"),
    ("₹30,000/month", "ok"),
    ("2.4 LPA", "ok"),                 # 2.4 lakh/year = 20,000/month
    ("2 LPA", "skip"),
    ("15-25k/month", "flag"),          # top of the range reaches 20k
    ("10k - 15k per month", "skip"),
    ("3-5 LPA", "ok"),
    ("Performance based", "ok"),       # can't parse: flagged, never skipped
    (None, "ok"),                      # not stated: flagged, not skipped
])
def test_minimum_stipend_20k(pay, outcome):
    skip, flags = stipend_flags(pay, 20000)
    got = "skip" if skip else ("flag" if any("starts below" in f for f in flags) else "ok")
    assert got == outcome, (pay, skip, flags)


def test_parse_monthly_range():
    from jobpilot.filter import parse_monthly_range
    assert parse_monthly_range("15-25k/month") == (15000, 25000)
    assert parse_monthly_range("6-8 LPA") == (50000, 66666)
    assert parse_monthly_range("₹30,000/month") == (30000, 30000)
    assert parse_monthly_range("unpaid") == (0, 0)


def test_location_flags():
    assert location_flags("Navi Mumbai", "onsite", CFG.locations_ok) == []
    assert location_flags("Delhi", "remote", CFG.locations_ok) == []
    assert location_flags("Delhi NCR", "onsite", CFG.locations_ok)
    assert location_flags("Anywhere in India", "hybrid", CFG.locations_ok) == []


def test_personal_email_flag():
    flags = contact_flags([{"kind": "email", "value": "hr.x@gmail.com"},
                           {"kind": "email", "value": "hr@x.com"}])
    assert len(flags) == 1 and "hr.x@gmail.com" in flags[0]


def job(**kw):
    return {"id": 1, "company": "Acme", "role": "ML Intern", "role_category": "ai_ml",
            "batch_eligibility": "2026", "location": "Remote", "work_mode": "remote",
            "stipend_or_ctc": "30k/month", "required_skills": ["Python"], "jd_summary": "x", **kw}


def test_hard_rules():
    assert hard_rules(job(), "", CFG) is None
    assert hard_rules(job(batch_eligibility="2024/2025"), "", CFG).decision == SKIP
    r = hard_rules(job(), "Apply now! Registration fee Rs 499", CFG)
    assert r.decision == SKIP and r.scam and "likely scam" in r.flags[0]
    r = hard_rules(job(), "This is an unpaid role", CFG)
    assert r.decision == SKIP and not r.scam
    assert hard_rules(job(role_category="other"), "", CFG).decision == SKIP


@pytest.mark.parametrize("score, decision", [(59, SKIP), (60, FLAG), (69, FLAG), (70, PROCEED), (95, PROCEED)])
def test_score_decision(score, decision):
    assert score_decision(score, 60) == decision


class FitLLM:
    name = "fit"

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    async def complete_json(self, **kw):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def fit(score, gaps=(), flags=()):
    return {"fit_score": score, "fit_reason": "reason", "gaps": list(gaps), "flags": list(flags)}


async def test_evaluate_combines_rules_and_fit():
    r = await evaluate(job(stipend_or_ctc=None), [{"kind": "email", "value": "a@gmail.com"}], "",
                       PROFILE, CFG, FitLLM(fit(65, ["PyTorch"], ["JD vague"])))
    assert r.decision == FLAG and r.status == db.FLAGGED and r.fit_score == 65
    flags = r.all_flags()
    assert "stipend/CTC not stated" in flags and "JD vague" in flags and "skill gap: PyTorch" in flags
    assert any("personal email" in f for f in flags)


async def test_evaluate_hard_skip_does_not_call_llm():
    llm = FitLLM()
    r = await evaluate(job(batch_eligibility="2025"), [], "", PROFILE, CFG, llm)
    assert r.decision == SKIP and llm.calls == 0


async def test_evaluate_low_fit_skips_with_reason():
    r = await evaluate(job(), [], "", PROFILE, CFG, FitLLM(fit(40)))
    assert r.decision == SKIP and r.status == db.SKIPPED_FILTER and "below threshold" in r.skip_reason


async def test_evaluate_llm_failure_flags_for_manual_review():
    r = await evaluate(job(), [], "", PROFILE, CFG, FitLLM(LLMError("down"), {"fit_score": "x"}))
    assert r.decision == FLAG and any("fit check failed" in f for f in r.flags)


async def test_fit_score_clamped():
    r = await evaluate(job(), [], "", PROFILE, CFG, FitLLM(fit(140)))
    assert r.fit_score == 100
