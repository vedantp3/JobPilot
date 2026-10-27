import asyncio
from pathlib import Path

import pytest

from jobpilot.config import get_config, load_profile
from jobpilot.link_handler import (
    FormField, Prefiller, application_url, build_answer_sheet, classify_link, draft_answers, looks_like_question,
    match_profile_field, plan_fill, sheet_questions, validate_answer,
)
from jobpilot.llm import LLMError
from jobpilot.main import FIXTURE_PROFILE

PROFILE = load_profile(FIXTURE_PROFILE)
JOB = {"id": 7, "company": "Kestrel Systems", "role": "Software Engineer Intern", "role_category": "software_engineering",
       "required_skills": ["Java", "Spring Boot"], "jd_summary": "SWE internship for 6 months", "resume_track": "sde"}


def test_application_url():
    assert application_url("https://jobs.lever.co/kestrel/abc", "lever") == "https://jobs.lever.co/kestrel/abc/apply"
    assert application_url("https://jobs.lever.co/kestrel/abc/apply", "lever").endswith("/abc/apply")
    assert application_url("https://jobs.ashbyhq.com/acme/123", "ashby").endswith("/123/application")
    assert application_url("https://docs.google.com/forms/d/e/x/viewform", "google_form").endswith("/viewform")
    assert classify_link("https://boards.greenhouse.io/acme/jobs/1") == "greenhouse"


@pytest.mark.parametrize("label, expected", [
    ("Full Name *", PROFILE.name),
    ("Name", PROFILE.name),
    ("First Name", "Vedant"),
    ("Last name", "Patil"),
    ("Email Address", PROFILE.email),
    ("Mobile Number", PROFILE.phone),
    ("WhatsApp number", PROFILE.phone),
    ("LinkedIn Profile URL", PROFILE.linkedin),
    ("GitHub", PROFILE.github),
    ("Portfolio / Website", PROFILE.portfolio),
    ("College Name", PROFILE.college),
    ("University", PROFILE.college),
    ("Degree / Branch", PROFILE.degree),
    ("Year of Graduation", "2026"),
    ("Batch", "2026"),
    ("CGPA (out of 10)", PROFILE.cgpa),
    ("Notice Period", PROFILE.notice_period),
    ("Expected CTC", PROFILE.expected_ctc),
    ("Current Location", PROFILE.location),
])
def test_profile_fields(label, expected):
    assert match_profile_field(label, PROFILE) == (True, expected)


@pytest.mark.parametrize("label", ["Current Company Name", "Father's Name", "10th Percentage", "12th marks %",
                                   "Current CTC", "Reference name"])
def test_known_fields_left_blank(label):
    assert match_profile_field(label, PROFILE) == (True, None)


def test_unknown_field():
    assert match_profile_field("Favourite colour", PROFILE) == (False, None)


@pytest.mark.parametrize("label, tag, q", [
    ("Why do you want to join us?", "input", True),
    ("Describe a project you're proud of", "input", True),
    ("Cover letter / anything else", "textarea", True),
    ("Provide details of one AI project you have worked on *", "input", True),
    ("Give an example of a time you shipped fast", "input", True),
    ("List the tools you use daily", "input", True),
    ("Briefly", "input", True),
    ("Your experience with LLM agents in production", "input", True),   # long sentence, no keyword
    ("Pronouns", "input", False),
    ("Hometown", "input", False),
    ("", "input", False),
])
def test_question_detection(label, tag, q):
    assert looks_like_question(label, tag) is q


def test_plan_fill():
    fields = [
        FormField(0, "input", "text", "Full Name", "name"),
        FormField(1, "input", "email", "Email", "email"),
        FormField(2, "input", "text", "Phone", "phone", value="+91 already typed"),     # keep user's value
        FormField(3, "input", "file", "Resume/CV", "resume"),
        FormField(4, "textarea", "textarea", "Why do you want to work at Kestrel?", "q1"),
        FormField(5, "input", "radio", "Are you authorised to work in India?", "auth"),
        FormField(6, "input", "radio", "Are you authorised to work in India?", "auth"),
        FormField(7, "select", "select", "Gender", "gender"),
        FormField(8, "input", "text", "Current CTC", "ctc"),
        FormField(9, "input", "text", "Pronouns", "pronouns"),
        FormField(10, "input", "file", "Cover photo", "photo"),
    ]
    values, questions, uploads, blank = plan_fill(fields, PROFILE)
    assert values == {0: PROFILE.name, 1: PROFILE.email}
    assert [q.idx for q in questions] == [4] and uploads == [3]
    assert blank == ["Are you authorised to work in India? (choice)", "Gender (choice)", "Current CTC", "Pronouns",
                     "Cover photo (file)"]


@pytest.mark.parametrize("answer, needle", [
    ("word " * 90, "words"),
    ("I love this 🚀 and Walstar Technologies.", "emoji"),
    ("- point one\n- point two", "list"),
    ("At Walstar Technologies I improved accuracy by 47%.", "numbers"),
    ("See https://myblog.dev/post for more.", "link"),
    ("I built QuantumTrader for options trading.", "QuantumTrader"),
    ("I also built an NSE stock analyzer.", "never_claim"),
])
def test_validate_answer(answer, needle):
    assert any(needle in e for e in validate_answer(answer, PROFILE, JOB))


def test_validate_answer_ok_and_empty():
    ok = ("At Walstar Technologies I built Flask REST APIs and an OCR pipeline with 95% accuracy, so a 6 month "
          "SWE internship at Kestrel Systems fits well.")
    assert validate_answer(ok, PROFILE, JOB) == []
    assert validate_answer("", PROFILE, JOB) == []


class AnswersLLM:
    name = "answers"

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        self.calls.append(user)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def answers(*pairs):
    return {"answers": [{"question": q, "answer": a} for q, a in pairs]}


GOOD = "At Walstar Technologies I built Flask REST APIs with 95% accuracy on the OCR pipeline."


async def test_draft_answers_retry_then_blank():
    qs = ["Why Kestrel?", "Current CTC?"]
    llm = AnswersLLM(answers(("Why Kestrel?", "🚀 " + GOOD), ("Current CTC?", "")),
                     answers(("Why Kestrel?", GOOD), ("Current CTC?", "")))
    got = await draft_answers(qs, JOB, PROFILE, get_config(), llm)
    assert got == {"Why Kestrel?": GOOD, "Current CTC?": ""}
    assert "emoji" in llm.calls[1] and '"QUESTIONS"' in llm.calls[0]


async def test_draft_answers_give_up_leaves_blank():
    llm = AnswersLLM(answers(("Why?", "47% better")), LLMError("down"))
    assert await draft_answers(["Why?"], JOB, PROFILE, get_config(), llm) == {"Why?": ""}


def test_standard_answers_per_track():
    from jobpilot.link_handler import standard_answers_for
    p = PROFILE.model_copy(update={"standard_answers": {
        "about_me": "general", "about_me:sde": "sde version", "strengths": "s", "why_this_company": "",
        "about_me:product": "product version"}})
    assert standard_answers_for(p, "sde") == {"about_me": "sde version", "strengths": "s"}
    assert standard_answers_for(p, "ai_ml") == {"about_me": "general", "strengths": "s"}
    assert standard_answers_for(p, None) == {"about_me": "general", "strengths": "s"}


async def test_featured_projects_reach_the_prompt():
    p = PROFILE.model_copy(update={"featured_projects": ["StockPilot"]})
    llm = AnswersLLM(answers(("Tell us about a project", GOOD)))
    await draft_answers(["Tell us about a project"], JOB, p, get_config(), llm)
    assert '"FEATURED_PROJECTS": [\n  "StockPilot"' in llm.calls[0]


def test_featured_projects_must_exist():
    import json
    from pydantic import ValidationError
    from jobpilot.config import Profile
    data = json.loads(FIXTURE_PROFILE.read_text(encoding="utf-8"))
    assert Profile.model_validate({**data, "featured_projects": ["StockPilot"]}).featured_projects == ["StockPilot"]
    with pytest.raises(ValidationError, match="not found"):
        Profile.model_validate({**data, "featured_projects": ["Imaginary App"]})


def test_answer_sheet_html():
    qs = sheet_questions({**JOB, "company": "R&D <Labs>"})
    assert qs[1] == "Why do you want to join R&D <Labs>?"
    sheet = build_answer_sheet({**JOB, "company": "R&D <Labs>"}, ["https://jobs.lever.co/k/1"],
                               {qs[0]: GOOD, qs[1]: ""}, PROFILE, "sde.pdf")
    assert f"Email: <code>{PROFILE.email}</code>" in sheet
    assert "R&amp;D &lt;Labs&gt;" in sheet and f"<code>{GOOD}</code>" in sheet
    assert "(fill this in yourself)" in sheet and "sde.pdf" in sheet


async def test_prefiller_falls_back_through_browsers(tmp_path):
    tried = []

    class Ctx:
        def on(self, *a):
            pass

    class Chromium:
        async def launch_persistent_context(self, user_data_dir, channel=None, **kw):
            tried.append((channel, Path(user_data_dir).name))
            if channel == "chrome":
                raise RuntimeError("chrome not installed")
            return Ctx()

    pf = Prefiller(tmp_path, headless=True)
    pf._pw = type("PW", (), {"chromium": Chromium()})()
    await pf._context()
    assert tried == [("chrome", "chrome"), ("msedge", "msedge")] and pf.channel == "msedge"
    pf._ctx = None                                   # user closed the window: relaunch the same browser
    await pf._context()
    assert tried[-1] == ("msedge", "msedge")

    class Broken:
        async def launch_persistent_context(self, *a, **kw):
            raise RuntimeError("spawn UNKNOWN")

    # automation markers are removed from the launch
    seen = {}

    class Recorder:
        async def launch_persistent_context(self, user_data_dir, **kw):
            seen.update(kw)
            return Ctx()

    pf3 = Prefiller(tmp_path, headless=False)
    pf3._pw = type("PW", (), {"chromium": Recorder()})()
    await pf3._context()
    assert seen["ignore_default_args"] == ["--enable-automation"]
    assert "--disable-blink-features=AutomationControlled" in seen["args"]

    class Locked:
        async def launch_persistent_context(self, *a, **kw):
            raise RuntimeError("Failed to create a ProcessSingleton for your profile directory")

    pf4 = Prefiller(tmp_path, headless=True)
    pf4._pw = type("PW", (), {"chromium": Locked()})()
    with pytest.raises(RuntimeError, match="already open in another window"):
        await pf4._context()

    pf2 = Prefiller(tmp_path, headless=True)
    pf2._pw = type("PW", (), {"chromium": Broken()})()
    with pytest.raises(RuntimeError, match="no browser could be started.*chrome.*msedge.*chromium"):
        await pf2._context()
    with pytest.raises(ValueError):
        Prefiller(tmp_path, browser="firefox")


async def test_open_for_login_uses_normal_browser_on_same_profile(tmp_path, monkeypatch):
    import jobpilot.link_handler as lh
    exe = tmp_path / "chrome.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(lh, "find_browser", lambda ch: exe if ch == "chrome" else None)
    launched = []
    pf = Prefiller(tmp_path / "bp")
    assert await pf.open_for_login(popen=launched.append) == "chrome"
    cmd = launched[0]
    assert cmd[0] == str(exe) and f"--user-data-dir={tmp_path / 'bp' / 'chrome'}" in cmd
    assert cmd[-1] == "https://accounts.google.com/" and not any("automation" in a for a in cmd)
    monkeypatch.setattr(lh, "find_browser", lambda ch: None)
    with pytest.raises(RuntimeError, match="isn't installed"):
        await Prefiller(tmp_path / "bp2").open_for_login(popen=launched.append)


def test_find_browser_on_this_machine():
    from jobpilot.link_handler import _browser_paths
    import sys
    paths = _browser_paths("chrome")
    if sys.platform == "win32":
        assert any(str(p).endswith(r"Google\Chrome\Application\chrome.exe") for p in paths)


# --- real browser against local pages (no network) ------------------------------------------------
FORM_HTML = """<!doctype html><html><body>
<form id="f" action="https://example.invalid/submit" onsubmit="window.__submitted = true; return false;">
  <div role="listitem"><div role="heading" id="h1">Full Name</div>
    <input type="text" aria-labelledby="h1"></div>
  <label for="em">Email address</label><input id="em" type="email" name="email">
  <label>Mobile number <input type="tel" name="phone"></label>
  <label for="li">LinkedIn URL</label><input id="li" type="url">
  <label for="cv">Resume</label><input id="cv" type="file" name="resume">
  <label for="why">Why do you want to join Kestrel Systems?</label><textarea id="why"></textarea>
  <label for="ctc">Current CTC</label><input id="ctc" type="text">
  <fieldset><legend>Work mode</legend><input type="radio" name="mode" value="r"> Remote</fieldset>
  <button type="submit" id="submit">Submit</button>
</form></body></html>"""


def _chromium_available() -> bool:
    async def probe():
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            b = await p.chromium.launch(headless=True)
            await b.close()
    try:
        asyncio.run(probe())
        return True
    except Exception:  # noqa: BLE001
        return False


browser = pytest.mark.skipif(not _chromium_available(), reason="Playwright Chromium not installed")


@browser
async def test_prefill_local_form(tmp_path):
    page_file = tmp_path / "form.html"
    page_file.write_text(FORM_HTML, encoding="utf-8")
    resume = tmp_path / "sde.pdf"
    resume.write_bytes(b"%PDF-1.4 test")
    why = "Why do you want to join Kestrel Systems?"
    llm = AnswersLLM(answers((why, GOOD)))
    pf = Prefiller(tmp_path / "profile", headless=True)
    try:
        report = await pf.prefill(page_file.as_uri(), "other_portal", JOB, PROFILE, get_config(), llm, resume)
        assert report.ok, report.error
        assert set(report.filled) >= {"Full Name", "Email address", "Mobile number", "LinkedIn URL"}
        assert report.uploaded == ["Resume"] and report.drafted == {why: GOOD}
        assert "Current CTC" in report.left_blank and "Work mode (choice)" in report.left_blank
        page = (await pf._context()).pages[-1]
        vals = await page.evaluate("""() => ({
            name: document.querySelector('[aria-labelledby=h1]').value,
            email: document.getElementById('em').value,
            why: document.getElementById('why').value,
            whyOutline: document.getElementById('why').style.outline,
            file: document.getElementById('cv').files[0].name,
            submitted: !!window.__submitted,
            banner: !!document.getElementById('jobpilot-banner'),
        })""")
        assert vals["name"] == PROFILE.name and vals["email"] == PROFILE.email and vals["why"] == GOOD
        assert "solid" in vals["whyOutline"] and vals["file"] == "sde.pdf" and vals["banner"]
        assert vals["submitted"] is False                     # never submits
        assert page.url == page_file.as_uri()
    finally:
        await pf.close()


@browser
async def test_prefill_no_fields_reports_error(tmp_path):
    page_file = tmp_path / "empty.html"
    page_file.write_text("<html><body><p>Click Apply to start</p></body></html>", encoding="utf-8")
    pf = Prefiller(tmp_path / "profile", headless=True)
    try:
        report = await pf.prefill(page_file.as_uri(), "other_portal", JOB, PROFILE, get_config(), AnswersLLM(), None)
        assert not report.ok and "no form fields" in report.error
    finally:
        await pf.close()
