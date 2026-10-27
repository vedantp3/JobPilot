"""Approval bot logic end to end over the fixture posts, with a fake Telegram and fake Gmail."""
import asyncio
import csv
import io
from types import SimpleNamespace

import pytest

from jobpilot import db
from jobpilot.approval_bot import (
    CARD_KV, EDIT_KV, OVERRIDE_KV, ApprovalService, Button, TelegramMessenger, build_application, export_csv,
)
from jobpilot.config import RESUME_TRACKS, get_config, load_profile
from jobpilot.gmail_client import SendQueue
from jobpilot.link_handler import classify_link
from jobpilot.llm import ReplayLLM
from jobpilot.main import FIXTURE_LLM, FIXTURE_POSTS, FIXTURE_PROFILE
from jobpilot.pipeline import Pipeline, run_fixtures

pytestmark = pytest.mark.llm
PROFILE = load_profile(FIXTURE_PROFILE)
GOOD_BODY = ("At Walstar Technologies I redesigned a RAG chatbot (Mistral, LangChain), improving answer quality "
             "by 30% across 100+ test cases.\n\nI graduate from VIT Vellore in 2026 and can join immediately. "
             f"GitHub: {PROFILE.github} Portfolio: {PROFILE.portfolio} Resume attached.")


class FakeMessenger:
    def __init__(self):
        self.messages: dict[int, dict] = {}
        self.files: list[tuple[bytes, str]] = []
        self._next = 100

    async def send(self, text, buttons=None):
        self._next += 1
        self.messages[self._next] = {"text": text, "buttons": buttons or []}
        return self._next

    async def edit(self, message_id, text, buttons=None):
        self.messages[message_id] = {"text": text, "buttons": buttons or []}

    async def send_file(self, data, filename, caption=""):
        self.files.append((data, filename))

    def last(self):
        return self.messages[max(self.messages)]

    def datas(self, message_id):
        return [b.data for row in self.messages[message_id]["buttons"] for b in row]


class FakeGmail:
    def __init__(self):
        self.drafts, self.updates, self.sent = {}, [], []

    async def create_draft(self, to, subject, body, attachment=None, attachment_filename=None):
        draft_id = f"d{len(self.drafts) + 1}"
        self.drafts[draft_id] = {"to": to, "subject": subject, "body": body, "attachment": attachment}
        return draft_id

    async def update_draft(self, draft_id, to, subject, body, attachment=None, attachment_filename=None):
        self.updates.append(draft_id)
        self.drafts[draft_id] = {"to": to, "subject": subject, "body": body, "attachment": attachment}
        return draft_id

    async def send_draft(self, draft_id):
        self.sent.append(draft_id)
        return f"m-{draft_id}"


class StubLLM(ReplayLLM):
    """Replays fixtures; unrecorded email/revise calls return `email_out` (a body or a list of bodies)."""

    def __init__(self):
        super().__init__(FIXTURE_LLM)
        self.bodies: list[str] = []
        self.fit_overrides: dict[str, int] = {}     # role -> score, for /rescore tests

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        if key.startswith("fit/"):
            import json as _json
            role = _json.loads(user)["JOB"]["role"]
            if role in self.fit_overrides:
                return {"fit_score": self.fit_overrides[role], "fit_reason": "re-scored", "gaps": [], "flags": []}
        if key.startswith("answers/"):
            import json as _json
            qs = _json.loads(user.split("\n\nThese answers", 1)[0])["QUESTIONS"]
            return {"answers": [{"question": q, "answer": "" if "strengths" in q else
                                 "At Walstar Technologies I built Flask REST APIs."} for q in qs]}
        if key.startswith(("email/", "revise/")) and not (self.dir / f"{key}.json").exists():
            body = self.bodies.pop(0) if self.bodies else GOOD_BODY
            return {"body": body, "projects_used": [], "employers_used": ["Walstar Technologies"]}
        return await super().complete_json(system=system, user=user, schema=schema, key=key)


@pytest.fixture
def cfg(tmp_path):
    resumes = tmp_path / "resumes"
    resumes.mkdir()
    for t in RESUME_TRACKS:
        (resumes / f"{t}.pdf").write_bytes(b"%PDF " + t.encode())
    return get_config().model_copy(update={"resumes_dir": str(resumes), "fit_threshold": 60})


@pytest.fixture
async def bot(conn, cfg):
    msgr, gmail, llm = FakeMessenger(), FakeGmail(), StubLLM()

    async def no_sleep(s):
        pass

    pipe = Pipeline(conn, llm, PROFILE, cfg=cfg, gmail=gmail, resolve_links=False, echo=lambda s: None)
    queue = SendQueue(conn, gmail, spacing=(90, 240), sleep=no_sleep)
    svc = ApprovalService(conn, pipe, queue, msgr)
    queue.notify, queue.on_sent, queue.on_refused = svc.notify, svc.on_sent, svc.on_refused
    pipe.notify, pipe.on_ready = svc.notify, svc.post_cards
    await run_fixtures(pipe, FIXTURE_POSTS)
    return SimpleNamespace(svc=svc, msgr=msgr, gmail=gmail, conn=conn, queue=queue, llm=llm)


def job(conn, role):
    return next(j for j in db.jobs_by_status(conn, *db.ALL_STATUSES) if j["role"] == role)


def app_of(conn, role):
    return [a for a in db.applications_for_job(conn, job(conn, role)["id"]) if a["body"]][-1]


def card(b, role):
    return int(db.kv_get(b.conn, CARD_KV.format(job(b.conn, role)["id"])))


async def drain(queue):
    task = asyncio.create_task(queue.run())
    await queue._queue.join()
    task.cancel()


# --- cards -------------------------------------------------------------------------------
async def test_cards_posted_for_every_actionable_job(bot):
    c = bot.conn
    for role in ("Machine Learning Intern", "Data Analyst Intern", "Full Stack Developer", "AI Engineer Intern",
                 "Backend Developer Intern", "SAP BTP Developer (Fresher)"):
        assert job(c, role)["status"] == db.AWAITING_APPROVAL, role
        assert card(bot, role)
    assert job(c, "Data Engineer Intern")["status"] == db.FLAGGED and card(bot, "Data Engineer Intern")
    for role in ("Software Engineer Trainee", "Data Science Intern", "ML Intern", "DevOps Intern"):
        assert db.kv_get(c, CARD_KV.format(job(c, role)["id"])) is None, role
    assert len(bot.gmail.drafts) == 4
    # no plain "already waiting" notice: the conflict card carries it
    assert not any(m["text"].startswith("⚠️ An email") for m in bot.msgr.messages.values())


async def test_email_card_contents(bot):
    m = bot.msgr.messages[card(bot, "Full Stack Developer")]
    t = m["text"]
    assert t.startswith("📧 <b>Full Stack Developer — Orbitly</b>")
    assert "Fit 72 · Resume: sde (clearly full-stack scoped)" in t
    assert "To: rahul@orbitly.io (Rahul)" in t and "<b>Subject:</b>" in t and "Hi Rahul," in t
    assert "⚠️ Flags:" in t and "gaps: MongoDB" in t
    app = app_of(bot.conn, "Full Stack Developer")
    assert bot.msgr.datas(card(bot, "Full Stack Developer")) == [
        f"send:{app['id']}", f"edit:{app['id']}", f"resume:{app['id']}", f"skip:{job(bot.conn, 'Full Stack Developer')['id']}"]
    assert app["approval_msg_id"] == card(bot, "Full Stack Developer")
    da = bot.msgr.messages[card(bot, "Data Analyst Intern")]["text"]
    assert "Also covers: DevOps Intern" in da


async def test_link_and_conflict_cards(bot):
    link = bot.msgr.messages[card(bot, "Backend Developer Intern")]
    assert link["text"].startswith("🔗") and "Apply via: Google Form" in link["text"]
    assert link["buttons"][1][0].url.startswith("https://docs.google.com/forms/")
    conflict = bot.msgr.messages[card(bot, "Data Engineer Intern")]
    assert conflict["text"].startswith("⏸") and "already waiting for your approval" in conflict["text"]
    jid = job(bot.conn, "Data Engineer Intern")["id"]
    assert bot.msgr.datas(card(bot, "Data Engineer Intern")) == [f"combine:{jid}", f"anyway:{jid}", f"skip:{jid}"]


async def test_html_is_escaped(bot):
    c = bot.conn
    jid = job(c, "Full Stack Developer")["id"]
    db.update_job(c, jid, company="R&D <Labs>")
    await bot.svc.refresh_card(jid)
    assert "R&amp;D &lt;Labs&gt;" in bot.msgr.messages[card(bot, "Full Stack Developer")]["text"]


async def test_cards_not_reposted(bot):
    before = len(bot.msgr.messages)
    assert await bot.svc.post_pending_cards() == 0
    assert len(bot.msgr.messages) == before


# --- send ------------------------------------------------------------------------------------
async def test_send_flow(bot):
    app = app_of(bot.conn, "Full Stack Developer")
    assert await bot.svc.handle_callback(f"send:{app['id']}") == "Queued (1 in line)"
    m = bot.msgr.messages[card(bot, "Full Stack Developer")]
    assert "⏳ <b>Queued</b>" in m["text"] and m["buttons"] == []
    assert await bot.svc.handle_callback(f"send:{app['id']}") == "Already queued"
    await drain(bot.queue)
    assert bot.gmail.sent == [app["gmail_draft_id"]]
    assert job(bot.conn, "Full Stack Developer")["status"] == db.SENT
    assert "✅ <b>Sent</b>" in bot.msgr.messages[card(bot, "Full Stack Developer")]["text"]
    assert await bot.svc.handle_callback(f"send:{app['id']}") == "Already sent"


async def test_send_refused_for_cooldown_offers_send_anyway(bot):
    app = app_of(bot.conn, "Full Stack Developer")
    db.record_send(bot.conn, "rahul@orbitly.io")          # emailed elsewhere meanwhile
    await bot.svc.handle_callback(f"send:{app['id']}")
    await drain(bot.queue)
    assert bot.gmail.sent == []
    last = bot.msgr.last()
    assert "Not sent" in last["text"] and [b.data for b in last["buttons"][0]][0] == f"force:{app['id']}"
    assert await bot.svc.handle_callback(f"force:{app['id']}") == "Queued (1 in line)"
    await drain(bot.queue)
    assert bot.gmail.sent == [app["gmail_draft_id"]]


async def test_skip_and_mark_submitted(bot):
    c = bot.conn
    fs = job(c, "Full Stack Developer")
    assert await bot.svc.handle_callback(f"skip:{fs['id']}") == "Skipped"
    assert job(c, "Full Stack Developer")["status"] == db.SKIPPED_BY_USER
    assert "❌ <b>Skipped</b>" in bot.msgr.messages[card(bot, "Full Stack Developer")]["text"]
    assert (await bot.svc.handle_callback(f"send:{app_of(c, 'Full Stack Developer')['id']}")).startswith("Can't send")
    be = job(c, "Backend Developer Intern")
    assert await bot.svc.handle_callback(f"sub:{be['id']}") == "Marked submitted"
    assert job(c, "Backend Developer Intern")["status"] == db.SUBMITTED


async def test_change_resume_updates_gmail_draft(bot):
    app = app_of(bot.conn, "Full Stack Developer")
    assert await bot.svc.handle_callback(f"resume:{app['id']}") == "Pick a resume"
    picker = bot.msgr.datas(card(bot, "Full Stack Developer"))
    assert f"rset:{app['id']}:hybrid" in picker
    assert await bot.svc.handle_callback(f"rset:{app['id']}:hybrid") == "Resume: hybrid"
    assert job(bot.conn, "Full Stack Developer")["resume_track"] == "hybrid"
    assert app_of(bot.conn, "Full Stack Developer")["resume_file"] == "hybrid.pdf"
    assert bot.gmail.updates == [app["gmail_draft_id"]]
    assert bot.gmail.drafts[app["gmail_draft_id"]]["attachment"].name == "hybrid.pdf"


# --- edit flow ---------------------------------------------------------------------------------
async def test_edit_flow(bot):
    app = app_of(bot.conn, "Full Stack Developer")
    old_card = card(bot, "Full Stack Developer")
    assert await bot.svc.handle_callback(f"edit:{app['id']}") == "Send your changes"
    assert db.kv_get(bot.conn, EDIT_KV) == str(app["id"])
    await bot.svc.handle_text("shorter please")
    new = app_of(bot.conn, "Full Stack Developer")
    assert new["body"].startswith("Hi Rahul,\n\nAt Walstar Technologies") and new["id"] == app["id"]
    assert bot.gmail.drafts[app["gmail_draft_id"]]["body"] == new["body"]
    assert "Replaced by the edited version" in bot.msgr.messages[old_card]["text"]
    assert bot.msgr.messages[old_card]["buttons"] == []
    assert card(bot, "Full Stack Developer") != old_card
    assert db.kv_get(bot.conn, EDIT_KV) == ""


async def test_edit_that_breaks_rules_keeps_old_draft(bot):
    app = app_of(bot.conn, "Full Stack Developer")
    bot.llm.bodies = ["🚀 " + GOOD_BODY] * 3
    await bot.svc.handle_callback(f"edit:{app['id']}")
    await bot.svc.handle_text("add rocket emoji")
    assert app_of(bot.conn, "Full Stack Developer")["body"] == app["body"]
    last = bot.msgr.last()
    assert "Couldn't apply that edit" in last["text"] and "emoji" in last["text"]
    assert last["buttons"][0][0].data == f"edit:{app['id']}"


async def test_text_without_edit_gets_help(bot):
    await bot.svc.handle_text("hello")
    assert "Use the buttons" in bot.msgr.last()["text"]


# --- cooldown choices ------------------------------------------------------------------------------
async def test_combine(bot):
    c = bot.conn
    de, ml = job(c, "Data Engineer Intern"), job(c, "Machine Learning Intern")
    old_ml_card = card(bot, "Machine Learning Intern")
    assert await bot.svc.handle_callback(f"combine:{de['id']}") == "Combined"
    assert job(c, "Data Engineer Intern")["status"] == db.DUPLICATE
    ml_app = app_of(c, "Machine Learning Intern")
    assert ml_app["also_job_ids"] == [de["id"]]
    assert job(c, "Machine Learning Intern")["status"] == db.AWAITING_APPROVAL
    assert "Also covers: Data Engineer Intern" in bot.msgr.messages[card(bot, "Machine Learning Intern")]["text"]
    assert "Replaced" in bot.msgr.messages[old_ml_card]["text"]
    assert ml_app["gmail_draft_id"] in bot.gmail.updates           # same Gmail draft, updated
    assert ml["id"] != de["id"]


async def test_send_anyway(bot):
    c = bot.conn
    de = job(c, "Data Engineer Intern")
    assert await bot.svc.handle_callback(f"anyway:{de['id']}") == "Drafted"
    app = app_of(c, "Data Engineer Intern")
    assert db.kv_get(c, OVERRIDE_KV.format(app["id"])) == "1"
    assert job(c, "Data Engineer Intern")["status"] == db.AWAITING_APPROVAL
    assert card(bot, "Data Engineer Intern") and app["gmail_draft_id"]


async def test_conflict_buttons_need_flagged_status(bot):
    fs = job(bot.conn, "Full Stack Developer")
    assert (await bot.svc.handle_callback(f"combine:{fs['id']}")).startswith("Can't")
    assert await bot.svc.handle_callback("nonsense:1") == "Unknown button"


# --- commands ------------------------------------------------------------------------------------------
async def test_commands(bot):
    s, c = bot.svc, bot.conn
    await s.cmd_pending([])
    assert "7 pending" in bot.msgr.last()["text"]
    await s.cmd_stats([])
    t = bot.msgr.last()["text"]
    assert t.startswith("<pre>") and "awaiting_approval" in t and "Emails sent today: 0/25" in t
    await s.cmd_status(["5", "interview"])
    assert "Can't" in bot.msgr.last()["text"]                     # DevOps is a duplicate, not sent
    sub = job(c, "Backend Developer Intern")["id"]
    await s.handle_callback(f"sub:{sub}")
    await s.cmd_status([str(sub), "interview"])
    assert job(c, "Backend Developer Intern")["status"] == db.INTERVIEW
    await s.cmd_status(["x"])
    assert "Usage" in bot.msgr.last()["text"]
    await s.cmd_pause([])
    assert db.kv_get(c, "paused") == "1"
    await s.cmd_resume([])
    assert db.kv_get(c, "paused") == "0" and "Resumed" in bot.msgr.last()["text"]
    await s.cmd_backfill(["5"])
    assert "only works while the listener" in bot.msgr.last()["text"]
    n = len(bot.msgr.messages)
    await s.cmd_card([str(sub)])
    assert len(bot.msgr.messages) == n + 1
    await s.cmd_help([])
    assert "/pending" in bot.msgr.last()["text"]


async def test_skipped_lists_with_draft_anyway_buttons(bot):
    await bot.svc.cmd_skipped([])
    m = bot.msgr.last()
    assert "Last 3 skipped" in m["text"]
    assert "batch 2024/2025 only excludes 2026" in m["text"] and "fit 58" in m["text"]
    datas = [b.data for row in m["buttons"] for b in row]
    assert datas == [f"rescue:{job(bot.conn, r)['id']}" for r in
                     ("Data Science Intern", "Software Engineer Trainee", "Software Engineer Intern")]


async def test_draft_anyway_rescues_batch_skip(bot):
    c = bot.conn
    gt = job(c, "Software Engineer Trainee")
    assert await bot.svc.handle_callback(f"rescue:{gt['id']}") == "Card posted below"
    gt = job(c, "Software Engineer Trainee")
    assert gt["status"] == db.AWAITING_APPROVAL and gt["resume_track"] == "sde"
    assert any(f.startswith("you chose Draft anyway (it was skipped: batch") for f in gt["flags"])
    assert not any(f.startswith("skipped:") for f in gt["flags"])
    app = app_of(c, "Software Engineer Trainee")
    assert app["gmail_draft_id"] and bot.gmail.drafts[app["gmail_draft_id"]]["to"] == "talent@greyfin.com"
    assert bot.msgr.messages[card(bot, "Software Engineer Trainee")]["text"].startswith("📧")
    assert await bot.svc.handle_callback(f"rescue:{gt['id']}") == "Already awaiting_approval"


async def test_draft_anyway_link_job(bot):
    k = job(bot.conn, "Software Engineer Intern")       # Kestrel, link-only, fit 58
    assert await bot.svc.handle_callback(f"rescue:{k['id']}") == "Card posted below"
    assert job(bot.conn, "Software Engineer Intern")["status"] == db.AWAITING_APPROVAL
    assert bot.msgr.messages[card(bot, "Software Engineer Intern")]["text"].startswith("🔗")


async def test_rescore(bot):
    c = bot.conn
    bot.llm.fit_overrides = {"Software Engineer Intern": 66}
    await bot.svc.cmd_rescore([])
    summary = bot.msgr.last()["text"]
    assert "Re-scored 1; 1 now pass" in summary and "58 → 66" in summary
    k = job(c, "Software Engineer Intern")
    assert k["status"] == db.AWAITING_APPROVAL and k["fit_score"] == 66
    assert not any(f.startswith("skipped:") for f in k["flags"])
    # batch/scam skips aren't re-scored
    assert job(c, "Software Engineer Trainee")["status"] == db.SKIPPED_FILTER


async def test_rescore_can_still_skip(bot):
    bot.llm.fit_overrides = {"Software Engineer Intern": 30}
    await bot.svc.cmd_rescore(["5"])
    assert "Re-scored 1; 0 now pass" in bot.msgr.last()["text"]
    k = job(bot.conn, "Software Engineer Intern")
    assert k["status"] == db.SKIPPED_FILTER and k["fit_score"] == 30
    assert sum(f.startswith("skipped:") for f in k["flags"]) == 1


class FakePrefiller:
    def __init__(self, report):
        self.report, self.calls = report, []

    async def prefill(self, url, link_type, job, profile, cfg, llm, resume):
        self.calls.append((url, link_type, resume.name))
        return self.report


async def test_link_card_buttons_with_and_without_prefill(bot):
    be = job(bot.conn, "Backend Developer Intern")
    datas = bot.msgr.datas(card(bot, "Backend Developer Intern"))
    assert datas[:1] == [f"sheet:{be['id']}"] and f"prefill:{be['id']}" not in datas   # no prefiller in tests
    from jobpilot.link_handler import PrefillReport
    bot.svc.prefiller = FakePrefiller(PrefillReport("u"))
    await bot.svc.refresh_card(be["id"])
    assert bot.msgr.datas(card(bot, "Backend Developer Intern"))[:2] == [f"prefill:{be['id']}", f"sheet:{be['id']}"]
    # SAP's link is an unresolved bit.ly (other portal): answer sheet only
    sap = job(bot.conn, "SAP BTP Developer (Fresher)")
    await bot.svc.refresh_card(sap["id"])
    assert f"prefill:{sap['id']}" not in bot.msgr.datas(card(bot, "SAP BTP Developer (Fresher)"))


async def test_answer_sheet(bot):
    be = job(bot.conn, "Backend Developer Intern")
    assert await bot.svc.handle_callback(f"sheet:{be['id']}") == "Answer sheet below"
    sheet = bot.msgr.last()
    t = sheet["text"]
    assert t.startswith("📋 <b>Answer sheet: Backend Developer Intern — Nimbus Labs</b>")
    assert f"Email: <code>{PROFILE.email}</code>" in t and "Resume to upload: <b>sde.pdf</b>" in t
    assert "<b>Why do you want to join Nimbus Labs?</b>" in t and "(fill this in yourself)" in t
    assert sheet["buttons"][0][0].data == f"sub:{be['id']}"
    assert db.kv_get(bot.conn, f"answers:{be['id']}")                         # cached
    await bot.svc.handle_callback(f"sheet:{be['id']}")
    assert bot.msgr.last()["text"] == t


async def test_prefill_success_report(bot):
    from jobpilot.link_handler import PrefillReport
    be = job(bot.conn, "Backend Developer Intern")
    rep = PrefillReport("https://docs.google.com/forms/x", filled=["Full Name", "Email"], uploaded=["Resume"],
                        drafted={"Why us?": "Because <reasons>."}, left_blank=["Gender (choice)"])
    bot.svc.prefiller = FakePrefiller(rep)
    assert await bot.svc.handle_callback(f"prefill:{be['id']}") == "Prefilled: check the browser"
    url, link_type, resume = bot.svc.prefiller.calls[0]
    assert link_type == "google_form" and resume == "sde.pdf"
    t = bot.msgr.last()["text"]
    assert "<b>Nothing was submitted.</b>" in t and "Filled from your profile: Full Name, Email" in t
    assert "Because &lt;reasons&gt;." in t and "Left for you: Gender (choice)" in t


async def test_prefill_failure_falls_back_to_sheet(bot):
    from jobpilot.link_handler import PrefillReport
    be = job(bot.conn, "Backend Developer Intern")
    bot.svc.prefiller = FakePrefiller(PrefillReport("u", error="TimeoutError: page didn't load"))
    assert await bot.svc.handle_callback(f"prefill:{be['id']}") == "Prefill didn't work (see message)"
    msgs = [m["text"] for m in list(bot.msgr.messages.values())[-2:]]
    assert "Couldn't prefill: TimeoutError" in msgs[0] and msgs[1].startswith("📋")


async def test_prefill_needs_google_login_no_sheet(bot):
    from jobpilot.link_handler import PrefillReport
    be = job(bot.conn, "Backend Developer Intern")
    bot.svc.prefiller = FakePrefiller(PrefillReport("u", error="needs a Google sign-in", needs_login=True))
    n = len(bot.msgr.messages)
    await bot.svc.handle_callback(f"prefill:{be['id']}")
    assert len(bot.msgr.messages) == n + 1 and "Google sign-in" in bot.msgr.last()["text"]


async def test_backlog_command_and_drop(bot):
    c = bot.conn
    await bot.svc.cmd_backlog([])
    assert "No backlog" in bot.msgr.last()["text"]
    from datetime import datetime, timedelta, timezone
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(timespec="seconds")
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    db.upsert_raw_post(c, tg_message_id=900, channel="ch", text="old post", posted_at=old)
    db.upsert_raw_post(c, tg_message_id=901, channel="ch", text="recent post", posted_at=recent)
    await bot.svc.cmd_backlog([])
    m = bot.msgr.last()
    assert "2 saved post(s) waiting" in m["text"]
    assert [b.data for row in m["buttons"] for b in row] == ["drop:24", "drop:all", "drop:keep"]
    assert await bot.svc.handle_callback("drop:24") == "Dropped 1"
    assert "1 still waiting" in bot.msgr.last()["text"]
    assert await bot.svc.handle_callback("drop:keep") == "Keeping the backlog"
    assert await bot.svc.handle_callback("drop:all") == "Dropped 1"
    assert db.backlog_stats(c) == (0, None)
    await bot.svc.cmd_stats([])
    assert "posts waiting: 0" in bot.msgr.last()["text"]


async def test_date_command(bot):
    await bot.svc.cmd_date(["23-10"])
    assert "only works while the listener" in bot.msgr.last()["text"]
    got = []

    async def by_date(args):
        from jobpilot.listener import parse_date_range
        parse_date_range(args)          # raises ValueError on bad input, like the real one
        got.append(args)
        return "3 post(s) from 23 Oct 2026: 2 queued, 1 already processed."

    bot.svc.by_date = by_date
    await bot.svc.cmd_date([])
    assert "Usage: /date" in bot.msgr.last()["text"]
    await bot.svc.cmd_date(["23-10"])
    assert bot.msgr.last()["text"].startswith("📅 3 post(s) from 23 Oct 2026: 2 queued")
    await bot.svc.cmd_date(["someday"])
    assert "can't read the date" in bot.msgr.last()["text"] and got == [["23-10"]]


async def test_source_channel_and_channels_command(bot):
    c = bot.conn
    await bot.svc.cmd_channels([])
    assert "No channels" in bot.msgr.last()["text"]
    ref = c.execute("SELECT channel FROM raw_posts LIMIT 1").fetchone()[0]
    db.kv_set(c, f"channel_title:{ref}", "Fixture Jobs")
    # one channel: no source line on cards
    bot.svc.channels = [(ref, "Fixture Jobs")]
    fs = job(c, "Full Stack Developer")
    await bot.svc.refresh_card(fs["id"])
    assert "📢 From:" not in bot.msgr.messages[card(bot, "Full Stack Developer")]["text"]
    # several channels: cards say where the job came from
    bot.svc.channels = [(ref, "Fixture Jobs"), ("@other", "Other Jobs")]
    await bot.svc.refresh_card(fs["id"])
    assert "📢 From: Fixture Jobs" in bot.msgr.messages[card(bot, "Full Stack Developer")]["text"]
    await bot.svc.cmd_channels([])
    t = bot.msgr.last()["text"]
    assert "Watching 2 channel(s)" in t and "Fixture Jobs" in t and "10 posts · 12 jobs · 0 applied" in t
    assert "Other Jobs" in t and "0 posts · 0 jobs" in t


async def test_login_command(bot):
    await bot.svc.cmd_login([])
    assert "isn't enabled" in bot.msgr.last()["text"]

    class P:
        async def open_for_login(self):
            return "chrome"

    bot.svc.prefiller = P()
    await bot.svc.cmd_login([])
    t = bot.msgr.last()["text"]
    assert "normal Chrome window" in t and "Close that window completely" in t


async def test_export(bot):
    await bot.svc.cmd_export([])
    data, name = bot.msgr.files[0]
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    assert name == "jobpilot.csv" and len(rows) == 12
    fs = next(r for r in rows if r["role"] == "Full Stack Developer")
    assert fs["emailed_to"] == "rahul@orbitly.io" and fs["status"] == "awaiting_approval"


# --- telegram wiring ---------------------------------------------------------------------------------
def test_classify_link():
    assert classify_link("https://docs.google.com/forms/d/e/x/viewform") == "google_form"
    assert classify_link("forms.gle/abc") == "google_form"
    assert classify_link("https://docs.google.com/document/d/x") == "other_portal"
    assert classify_link("https://jobs.lever.co/acme/1") == "lever"
    assert classify_link("https://job-boards.greenhouse.io/acme/jobs/1") == "greenhouse"
    assert classify_link("https://www.linkedin.com/jobs/view/1") == "linkedin"
    assert classify_link("https://acme.wd5.myworkdayjobs.com/x") == "workday"
    assert classify_link("https://wa.me/919999999999") == "whatsapp"


def test_markup():
    m = TelegramMessenger._markup([[Button("Send", "send:1"), Button("Open", url="https://x.io")]])
    b1, b2 = m.inline_keyboard[0]
    assert (b1.text, b1.callback_data, b2.url) == ("Send", "send:1", "https://x.io")
    assert TelegramMessenger._markup([]) is None


async def test_non_owner_button_presses_are_ignored(conn):
    calls = []

    class Svc:
        async def handle_callback(self, data):
            calls.append(data)
            return "ok"

        def __getattr__(self, name):          # cmd_* handlers: not exercised here
            async def noop(*a):
                pass
            return noop

    app = build_application("123456:TEST-TOKEN", 42, Svc())
    from telegram.ext import CallbackQueryHandler, CommandHandler

    handlers = app.handlers[0]
    cb = next(h for h in handlers if isinstance(h, CallbackQueryHandler))
    answered = []

    async def answer(text=None):
        answered.append(text)

    stranger = SimpleNamespace(effective_chat=SimpleNamespace(id=999),
                               callback_query=SimpleNamespace(data="send:1", answer=answer))
    await cb.callback(stranger, None)
    assert calls == [] and answered == [None]
    owner = SimpleNamespace(effective_chat=SimpleNamespace(id=42),
                            callback_query=SimpleNamespace(data="send:1", answer=answer))
    await cb.callback(owner, None)
    assert calls == ["send:1"] and answered[-1] == "ok"
    # every command is restricted to the owner chat
    cmds = [h for h in handlers if isinstance(h, CommandHandler)]
    assert {next(iter(h.commands)) for h in cmds} >= {"pending", "stats", "status", "export", "pause", "resume",
                                                      "backfill"}
    assert all("42" in repr(h.filters) for h in cmds)
