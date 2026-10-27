"""End-to-end ingest + decide over tests/fixtures/posts with recorded LLM responses."""
import httpx
import pytest

from jobpilot import db
from jobpilot.config import RESUME_TRACKS, get_config, load_profile
from jobpilot.dedupe import job_hash
from jobpilot.llm import ReplayLLM
from jobpilot.main import FIXTURE_LLM, FIXTURE_POSTS, FIXTURE_PROFILE
from jobpilot.pipeline import Pipeline, run_fixtures

pytestmark = pytest.mark.llm

PROFILE = load_profile(FIXTURE_PROFILE)


@pytest.fixture
def cfg(tmp_path):
    """Real config, but resumes come from a temp folder of fake PDFs."""
    resumes = tmp_path / "resumes"
    resumes.mkdir()
    for t in RESUME_TRACKS:
        (resumes / f"{t}.pdf").write_bytes(b"%PDF fake " + t.encode())
    return get_config().model_copy(update={"resumes_dir": str(resumes), "fit_threshold": 60})


class FakeGmail:
    def __init__(self, fail=False):
        self.drafts, self.fail = [], fail

    async def create_draft(self, to, subject, body, attachment=None, attachment_filename=None):
        if self.fail:
            raise RuntimeError("quota exceeded")
        self.drafts.append({"to": to, "subject": subject, "body": body,
                            "attachment": attachment, "filename": attachment_filename})
        return f"draft-{len(self.drafts)}"


def offline_http():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "bit.ly":
            return httpx.Response(301, headers={"Location": "https://forms.gle/veltrixBTP"})
        if req.url.host == "forms.gle":
            return httpx.Response(302, headers={"Location": "https://docs.google.com/forms/d/e/veltrix/viewform"})
        return httpx.Response(200)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def make_pipe(conn, out, notes=None, cfg=None, dry_run=True, **kw):
    async def notify(t):
        (notes if notes is not None else out).append(t)
    return Pipeline(conn, ReplayLLM(FIXTURE_LLM), PROFILE, cfg=cfg or get_config(), dry_run=dry_run,
                    echo=out.append, notify=notify, **kw)


@pytest.fixture
async def ran(conn, cfg):
    out: list[str] = []
    notes: list[str] = []
    async with offline_http() as http:
        ids = await run_fixtures(make_pipe(conn, out, notes, cfg=cfg, http=http), FIXTURE_POSTS)
    return conn, ids, "\n".join(out), notes


def by_role(conn, role):
    return next(j for j in db.jobs_by_status(conn, *db.ALL_STATUSES) if j["role"] == role)


def planned(conn, role):
    job = by_role(conn, role)
    return [conn.execute("SELECT value FROM contacts WHERE id=?", (a["contact_id"],)).fetchone()[0]
            for a in db.applications_for_job(conn, job["id"])]


async def test_all_fixtures_processed(ran):
    conn, ids, out, _ = ran
    assert len(ids) == 11
    assert conn.execute("SELECT COUNT(*) FROM raw_posts WHERE processed != 1").fetchone()[0] == 0
    assert "Error" not in out
    assert db.status_counts(conn) == {
        db.DRAFTED: 4, db.EXTRACTED: 2, db.FLAGGED: 1, db.SKIPPED_FILTER: 3, db.DUPLICATE: 2,
    }


@pytest.mark.parametrize("role, status, track", [
    ("Machine Learning Intern", db.DRAFTED, "ai_ml"),
    ("Data Analyst Intern", db.DRAFTED, "ai_ml"),
    ("Backend Developer Intern", db.EXTRACTED, "sde"),          # link job: no email
    ("Full Stack Developer", db.DRAFTED, "sde"),
    ("AI Engineer Intern", db.DRAFTED, "ai_ml"),
    ("SAP BTP Developer (Fresher)", db.EXTRACTED, "sap"),       # link job: no email
    ("Software Engineer Intern", db.SKIPPED_FILTER, None),
    ("Software Engineer Trainee", db.SKIPPED_FILTER, None),
    ("Data Science Intern", db.SKIPPED_FILTER, None),
])
async def test_decisions(ran, role, status, track):
    conn, *_ = ran
    j = by_role(conn, role)
    assert (j["status"], j["resume_track"]) == (status, track)


async def test_batch_and_scam_skips_never_reach_llm(ran):
    conn, *_ = ran
    assert by_role(conn, "Software Engineer Trainee")["fit_score"] is None
    scam = by_role(conn, "Data Science Intern")
    assert scam["fit_score"] is None and any(f.startswith("likely scam") for f in scam["flags"])


async def test_planned_emails(ran):
    conn, *_ = ran
    assert planned(conn, "Machine Learning Intern") == ["priya.sharma@acme.ai"]
    assert planned(conn, "Full Stack Developer") == ["rahul@orbitly.io"]
    assert planned(conn, "AI Engineer Intern") == ["neha@zentrix.ai"]          # named founder over careers@
    assert planned(conn, "Software Engineer Trainee") == []                     # skipped
    total = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    assert total == 4


async def test_same_recipient_multi_role_single_email(ran):
    conn, *_ = ran
    assert planned(conn, "Data Analyst Intern") == ["data.hiring@orbitly.io"]
    (app,) = db.applications_for_job(conn, by_role(conn, "Data Analyst Intern")["id"])
    devops = by_role(conn, "DevOps Intern")
    assert app["also_job_ids"] == [devops["id"]]
    assert planned(conn, "DevOps Intern") == [] and devops["status"] == db.DUPLICATE


async def test_same_recipient_across_posts_is_held(ran):
    conn, _, _, notes = ran
    assert planned(conn, "Data Engineer Intern") == []
    assert by_role(conn, "Data Engineer Intern")["status"] == db.FLAGGED
    assert len(notes) == 1 and "priya.sharma@acme.ai" in notes[0] and "Combine / Send anyway / Skip?" in notes[0]


async def test_short_link_resolved_and_raw_kept(ran):
    conn, *_ = ran
    sap = by_role(conn, "SAP BTP Developer (Fresher)")
    (c,) = db.contacts_for_job(conn, sap["id"])
    assert c["value"] == "https://docs.google.com/forms/d/e/veltrix/viewform"
    assert c["raw_value"] == "https://bit.ly/veltrix-btp26"


async def test_hidden_hyperlink(ran):
    conn, *_ = ran
    j = by_role(conn, "Software Engineer Intern")
    assert db.contacts_for_job(conn, j["id"])[0]["value"] == "https://jobs.lever.co/kestrel/7f3c2a91-intern-swe"


async def test_repost_is_duplicate_and_merges_contacts(ran):
    conn, _, out, _ = ran
    ml = by_role(conn, "Machine Learning Intern")
    assert {c["value"] for c in db.contacts_for_job(conn, ml["id"])} == {
        "priya.sharma@acme.ai", "careers@acme.ai"
    }
    dup = by_role(conn, "ML Intern")
    assert dup["status"] == db.DUPLICATE and dup["fit_score"] is None
    assert db.applications_for_job(conn, dup["id"]) == []
    assert "[duplicate]" in out and "careers@acme.ai" in out
    assert job_hash("Acme AI", "ML Intern") == job_hash("Acme AI Pvt. Ltd.", "Machine Learning Internship")


async def test_rerun_is_idempotent(ran, cfg):
    conn, *_ = ran
    pipe = make_pipe(conn, [], cfg=cfg, resolve_links=False)
    assert await run_fixtures(pipe, FIXTURE_POSTS) == []
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 12
    assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 4


async def test_emails_drafted_and_valid(ran):
    conn, _, out, _ = ran
    rows = conn.execute("SELECT a.subject, a.body, a.resume_file, a.gmail_draft_id, j.role FROM applications a "
                        "JOIN jobs j ON j.id = a.job_id ORDER BY a.id").fetchall()
    assert len(rows) == 4
    for r in rows:
        assert r["body"].startswith("Hi ") and r["body"].endswith(f"Best,\nVedant Patil\n{PROFILE.phone}")
        assert "Resume attached." in r["body"] and r["gmail_draft_id"] is None   # dry run: no Gmail
    by = {r["role"]: r for r in rows}
    assert by["Machine Learning Intern"]["subject"] == "ML Intern - Vedant Patil"      # the post asked for it
    assert by["Full Stack Developer"]["subject"] == "Full Stack Developer — Vedant Patil (VIT Vellore, 2026)"
    assert by["Full Stack Developer"]["body"].startswith("Hi Rahul,")
    assert by["Data Analyst Intern"]["body"].startswith("Hi Orbitly Team,")
    assert "DevOps Intern" in by["Data Analyst Intern"]["body"]                         # also-role mention
    assert by["Full Stack Developer"]["resume_file"] == "sde.pdf"
    ml = by_role(conn, "Machine Learning Intern")
    assert any(f.startswith("subject line taken from the post") for f in ml["flags"])
    assert "Subject: ML Intern - Vedant Patil" in out


async def test_live_mode_creates_gmail_drafts(conn, cfg):
    gmail = FakeGmail()
    pipe = make_pipe(conn, [], cfg=cfg, dry_run=False, gmail=gmail, resolve_links=False)
    await run_fixtures(pipe, FIXTURE_POSTS)
    assert len(gmail.drafts) == 4
    d = next(d for d in gmail.drafts if d["to"] == "rahul@orbitly.io")
    assert d["attachment"].name == "sde.pdf" and d["filename"] == "Vedant_Patil_Resume.pdf"
    ids = {r[0] for r in conn.execute("SELECT gmail_draft_id FROM applications")}
    assert ids == {"draft-1", "draft-2", "draft-3", "draft-4"}


async def test_gmail_failure_marks_job_failed(conn, cfg):
    notes: list[str] = []
    pipe = make_pipe(conn, [], notes, cfg=cfg, dry_run=False, gmail=FakeGmail(fail=True), resolve_links=False)
    await run_fixtures(pipe, FIXTURE_POSTS)
    assert db.status_counts(conn)[db.FAILED] == 4
    assert any("Gmail error" in n and "quota exceeded" in n for n in notes)


async def test_missing_resume_fails_job(conn, cfg):
    (cfg.path("resumes_dir") / "sde.pdf").unlink()
    notes: list[str] = []
    await run_fixtures(make_pipe(conn, [], notes, cfg=cfg, resolve_links=False), FIXTURE_POSTS)
    fs = by_role(conn, "Full Stack Developer")
    assert fs["status"] == db.FAILED and any("sde.pdf is missing" in f for f in fs["flags"])
    assert any("sde.pdf is missing" in n for n in notes)


async def test_edited_post_redecides_without_duplicating(ran, cfg):
    conn, *_ = ran
    ml = by_role(conn, "Machine Learning Intern")
    db.update_job(conn, ml["id"], status=db.EXTRACTED)        # undo the draft (dry run: no Gmail draft)
    db.mark_post_processed(conn, ml["raw_post_id"], 0)        # simulate an edit
    pipe = make_pipe(conn, [], cfg=cfg, resolve_links=False)
    assert await pipe.process_post(ml["raw_post_id"]) == [ml["id"]]
    assert planned(conn, "Machine Learning Intern") == ["priya.sharma@acme.ai"]
    assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 4


async def test_failed_extraction_marks_post_and_notifies(conn, cfg):
    notes: list[str] = []
    pid, _ = db.upsert_raw_post(conn, tg_message_id=99, channel="c", text="unrecorded post")
    pipe = make_pipe(conn, [], notes, cfg=cfg, resolve_links=False)
    assert await pipe.process_post(pid) == []
    assert db.get_raw_post(conn, pid)["processed"] == -1
    assert notes and "#99" in notes[0]
