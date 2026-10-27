import pytest

from jobpilot.config import get_config, load_profile
from jobpilot.email_drafter import (
    MAX_REGENERATIONS, assemble, attachment_name, build_greeting, build_subject, draft_email,
    requested_subject, validate_email,
)
from jobpilot.llm import LLMError
from jobpilot.main import FIXTURE_PROFILE

PROFILE = load_profile(FIXTURE_PROFILE)
JOB = {"id": 1, "company": "Acme AI", "role": "ML Intern", "role_category": "ai_ml",
       "required_skills": ["Python", "PyTorch"], "jd_summary": "Build RAG pipelines for a 6-month internship.",
       "flags": ["skill gap: Kubernetes"], "resume_track": "ai_ml"}
GH, PF = PROFILE.github, PROFILE.portfolio
TAIL = f"I graduate from VIT Vellore in 2026 and can join immediately. GitHub: {GH} Portfolio: {PF} Resume attached."
GOOD = ("At Walstar Technologies I redesigned a RAG chatbot (Mistral, LangChain), improving answer quality "
        "by 30% across 100+ test cases. Your stack of Python and PyTorch is what I use daily.\n\n" + TAIL)


def check(body, **kw):
    subject = build_subject(JOB["role"], PROFILE)
    return validate_email(subject, assemble("Hi Priya,", body, PROFILE), PROFILE, JOB, **kw)


def test_fixed_parts():
    assert build_subject("ML Intern", PROFILE) == "ML Intern — Vedant Patil (VIT Vellore, 2026)"
    assert build_greeting("priya sharma", "Acme") == "Hi Priya,"
    assert build_greeting(None, "Custom Logic Innovation (CLI)") == "Hi Custom Logic Innovation Team,"
    assert build_greeting(None, None) == "Hi Hiring Team,"
    assert assemble("Hi Priya,", "Body.", PROFILE).endswith(f"Best,\nVedant Patil\n{PROFILE.phone}")
    assert attachment_name(PROFILE) == "Vedant_Patil_Resume.pdf"


def test_good_email_passes():
    assert check(GOOD, projects_used=[], employers_used=["Walstar Technologies"]) == []


@pytest.mark.parametrize("body, needle", [
    (GOOD + " " + "word " * 120, "words"),
    ("- built things at Walstar Technologies with 30% gains\n" + TAIL, "bullet"),
    ("1. Walstar Technologies, 30% better\n" + TAIL, "bullet"),
    ("I am writing to apply. Walstar Technologies, 30% better. " + TAIL, "writing to apply"),
    ("I am interested in applying. Walstar Technologies, 30% better. " + TAIL, "interested in applying"),
    ("At Walstar Technologies I did great work. " + TAIL, "no digit-bearing metric"),
    ("At Walstar Technologies I improved accuracy by 47%. " + TAIL, "not found in profile.json"),
    ("Big fan 🚀 of Walstar Technologies, 30% better. " + TAIL, "emoji"),
    ("I improved answer quality by 30% in a chatbot. " + TAIL, "doesn't name any project"),
    ("At Walstar Technologies, 30% better. I built QuantumTrader for options. " + TAIL, "QuantumTrader"),
    ("At Walstar Technologies, 30% better. See https://myblog.dev/post. " + TAIL, "isn't your GitHub"),
    ("At Walstar Technologies, 30% better. Mail me at me@other.com. " + TAIL, "email address"),
    ("At Walstar Technologies, 30% better. I graduate in 2026. Resume attached.", "missing your GitHub"),
    (f"At Walstar Technologies, 30% better. GitHub: {GH} Portfolio: {PF}", "Resume attached"),
    ("At Walstar Technologies, 30% better, and I built an NSE stock analyzer. " + TAIL, "never_claim"),
    ("At Walstar Technologies, 30% better; I love open source work. " + TAIL, "never_claim"),
    (GOOD + "\n\nBest regards,", "sign-off"),
])
def test_violations(body, needle):
    errors = check(body)
    assert any(needle.lower() in e.lower() for e in errors), errors


def test_digits_inside_words_and_bare_links():
    body = ("At Walstar Technologies I improved answer quality by 30%, using n8n and S/4HANA style flows.\n\n"
            "I graduate from VIT Vellore in 2026.\nGitHub: github.com/vedantp3\n"
            "Portfolio: vedantpatil.vercel.app\nResume attached.")
    assert check(body) == []


def test_word_limit_is_configurable():
    long_body = GOOD + " " + "word " * 120
    assert any("words" in e for e in check(long_body))
    subject = build_subject(JOB["role"], PROFILE)
    text = assemble("Hi Priya,", long_body, PROFILE)
    assert validate_email(subject, text, PROFILE, JOB, max_words=250) == []


def test_prompt_word_limit_and_private_examples(tmp_path):
    import shutil
    from jobpilot.email_drafter import LOCAL_EXAMPLES, load_email_prompt
    shutil.copy(get_config().path("prompts_dir") / "email.md", tmp_path / "email.md")
    cfg = get_config().model_copy(update={"email_max_words": 180, "prompts_dir": str(tmp_path)})
    prompt = load_email_prompt(cfg)
    assert "Under 180 words" in prompt and "{max_words}" not in prompt
    assert "<example>" not in prompt                       # the public prompt carries no personal email
    (tmp_path / LOCAL_EXAMPLES).write_text("<example>\nMy shortlisted email.\n</example>", encoding="utf-8")
    assert load_email_prompt(cfg).rstrip().endswith("<example>\nMy shortlisted email.\n</example>")


def test_declared_names_must_exist():
    errors = check(GOOD, projects_used=["Crypto Bot"], employers_used=["Google"])
    assert any("Crypto Bot" in e for e in errors) and any("Google" in e for e in errors)


def test_job_numbers_allowed_and_short_org_alias():
    body = ("At Walstar I improved answer quality by 30%, and I can commit to your 6-month internship. " + TAIL)
    assert check(body) == []


@pytest.mark.parametrize("post, expected", [
    ("Subject line: ML Intern - <Your Name>", "ML Intern - Vedant Patil"),
    ("Mention subject as 'Application for [Role] - [Name]'", "Application for ML Intern - Vedant Patil"),
    ("Subject: Backend Intern Application", "Backend Intern Application"),
    ("Subject: <Your Name> | <College> | <CGPA>", None),     # placeholders we can't fill
    ("Please send a clear subject line and your CV", None),  # prose, not an instruction
    ("No special instructions", None),
])
def test_requested_subject(post, expected):
    assert requested_subject(post, "ML Intern", PROFILE) == expected


class EmailLLM:
    name = "email"

    def __init__(self, *responses):
        self.responses, self.calls, self.keys = list(responses), [], []

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        self.calls.append(user)
        self.keys.append(key)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def out(body, **kw):
    return {"body": body, "projects_used": kw.get("p", []), "employers_used": kw.get("e", [])}


async def test_draft_first_try():
    llm = EmailLLM(out(GOOD))
    d = await draft_email(JOB, {"person_name": "Priya"}, [], PROFILE, get_config(), llm)
    assert d.ok and d.attempts == 1 and d.text.startswith("Hi Priya,")
    assert '"GAPS": [\n  "Kubernetes"' in llm.calls[0] and '"RESUME_TRACK": "ai_ml"' in llm.calls[0]


async def test_draft_regenerates_with_errors_then_passes():
    llm = EmailLLM(out("I am writing to apply. " + GOOD), out(GOOD))
    d = await draft_email(JOB, {}, [], PROFILE, get_config(), llm)
    assert d.ok and d.attempts == 2 and d.text.startswith("Hi Acme AI Team,")
    assert "writing to apply" in llm.calls[1] and llm.keys[1].endswith(".retry1")


async def test_draft_gives_up_and_returns_best_attempt():
    bad = [out("I am writing to apply. " + GOOD), LLMError("rate limited"), out("🚀 " + GOOD + " 47%")]
    assert len(bad) == MAX_REGENERATIONS + 1
    d = await draft_email(JOB, {}, [], PROFILE, get_config(), EmailLLM(*bad))
    assert not d.ok and d.attempts == 1 and "writing to apply" in d.errors[0]


async def test_featured_projects_in_email_request():
    llm = EmailLLM(out(GOOD))
    p = PROFILE.model_copy(update={"featured_projects": ["StockPilot"]})
    await draft_email(JOB, {}, [], p, get_config(), llm)
    assert '"FEATURED_PROJECTS": [\n  "StockPilot"' in llm.calls[0]
    llm2 = EmailLLM(out(GOOD))
    await draft_email(JOB, {}, [], PROFILE, get_config(), llm2)       # none featured: key absent
    assert "FEATURED_PROJECTS" not in llm2.calls[0]


async def test_draft_uses_requested_subject_and_also_roles():
    llm = EmailLLM(out(GOOD))
    d = await draft_email(JOB, {}, [{"role": "DevOps Intern"}], PROFILE, get_config(), llm,
                          post_text="Subject line: ML Intern - <Your Name>")
    assert d.subject == "ML Intern - Vedant Patil"
    assert '"ALSO_ROLES": [\n  "DevOps Intern"' in llm.calls[0]
