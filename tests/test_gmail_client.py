import base64
import email
from datetime import datetime, timedelta, timezone

import pytest

from jobpilot import db
from jobpilot.gmail_client import SendQueue, SendRefused, _raw, build_message, local_midnight_utc


def test_build_message_with_attachment(tmp_path):
    pdf = tmp_path / "sde.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    msg = build_message("hr@acme.ai", "ML Intern — Vedant Patil (VIT Vellore, 2026)", "Hi,\n\nBody.",
                        pdf, "Vedant_Patil_Resume.pdf")
    parsed = email.message_from_bytes(base64.urlsafe_b64decode(_raw(msg)), policy=email.policy.default)
    assert parsed["To"] == "hr@acme.ai"
    assert parsed["Subject"] == "ML Intern — Vedant Patil (VIT Vellore, 2026)"
    parts = list(parsed.iter_attachments())
    assert len(parts) == 1 and parts[0].get_filename() == "Vedant_Patil_Resume.pdf"
    assert parts[0].get_content() == b"%PDF-1.4 fake"
    assert parsed.get_body(("plain",)).get_content().strip() == "Hi,\n\nBody."


class FakeGmail:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    async def send_draft(self, draft_id):
        if self.fail:
            raise RuntimeError("gmail down")
        self.sent.append(draft_id)
        return f"msg-{draft_id}"


def approved_app(conn, n, email_addr="hr@acme.ai", status=db.AWAITING_APPROVAL, draft=True):
    pid = db.upsert_raw_post(conn, tg_message_id=n, channel="c", text=f"p{n}")[0]
    jid = db.insert_job(conn, raw_post_id=pid, job_hash=f"h{n}", role=f"Role {n}", status=status)
    cid = db.add_contact(conn, jid, kind="email", value=email_addr)
    return db.insert_application(conn, job_id=jid, contact_id=cid,
                                 gmail_draft_id=f"d{n}" if draft else None), jid


def queue(conn, gmail=None, **kw):
    notes, sleeps = [], []

    async def notify(t):
        notes.append(t)

    async def sleep(s):
        sleeps.append(s)

    q = SendQueue(conn, gmail or FakeGmail(), notify=notify, sleep=sleep, **kw)
    return q, notes, sleeps


async def test_send_one_happy_path(conn):
    app_id, jid = approved_app(conn, 1)
    q, _, _ = queue(conn)
    assert await q.send_one(app_id) == "msg-d1"
    app = db.get_application(conn, app_id)
    assert app["gmail_message_id"] == "msg-d1" and app["sent_at"]
    assert db.get_job(conn, jid)["status"] == db.SENT
    assert db.get_recipient(conn, "hr@acme.ai")["total_emails"] == 1


@pytest.mark.parametrize("status", [db.EXTRACTED, db.FLAGGED, db.DRAFTED, db.SKIPPED_BY_USER, db.SENT])
async def test_refuses_unapproved_jobs(conn, status):
    app_id, _ = approved_app(conn, 1, status=status)
    q, _, _ = queue(conn)
    with pytest.raises(SendRefused, match="not awaiting_approval"):
        await q.send_one(app_id)
    assert q.gmail.sent == []


async def test_refuses_without_draft_or_twice(conn):
    app_id, _ = approved_app(conn, 1, draft=False)
    q, _, _ = queue(conn)
    with pytest.raises(SendRefused, match="no Gmail draft"):
        await q.send_one(app_id)
    app2, jid2 = approved_app(conn, 2, email_addr="b@x.io")
    await q.send_one(app2)
    db.set_status(conn, jid2, db.AWAITING_APPROVAL, force=True)
    with pytest.raises(SendRefused, match="already sent"):
        await q.send_one(app2)


async def test_cooldown_blocks_unless_overridden(conn):
    db.record_send(conn, "hr@acme.ai", at=(datetime.now(timezone.utc) - timedelta(days=2)).isoformat())
    app_id, _ = approved_app(conn, 1)
    q, _, _ = queue(conn)
    with pytest.raises(SendRefused, match="already emailed"):
        await q.send_one(app_id)
    assert await q.send_one(app_id, override_cooldown=True) == "msg-d1"


async def test_daily_cap(conn):
    q, _, _ = queue(conn, max_per_day=2)
    a1, _ = approved_app(conn, 1, "a@x.io")
    a2, _ = approved_app(conn, 2, "b@x.io")
    a3, _ = approved_app(conn, 3, "c@x.io")
    await q.send_one(a1)
    await q.send_one(a2)
    with pytest.raises(SendRefused, match="daily cap"):
        await q.send_one(a3)


async def test_gmail_failure_marks_failed(conn):
    app_id, jid = approved_app(conn, 1)
    q, _, _ = queue(conn, FakeGmail(fail=True))
    with pytest.raises(RuntimeError, match="gmail down"):
        await q.send_one(app_id)
    assert db.get_job(conn, jid)["status"] == db.FAILED
    assert db.get_application(conn, app_id)["sent_at"] is None


async def test_worker_spaces_sends_and_reports_refusals(conn):
    q, notes, sleeps = queue(conn, spacing=(90, 240))
    a1, _ = approved_app(conn, 1, "a@x.io")
    a2, _ = approved_app(conn, 2, "b@x.io")
    bad, _ = approved_app(conn, 3, "c@x.io", status=db.DRAFTED)
    for a in (a1, bad, a2):
        await q.enqueue(a)
    import asyncio
    task = asyncio.create_task(q.run())
    await q._queue.join()
    task.cancel()
    assert q.gmail.sent == ["d1", "d2"]
    assert len(sleeps) == 2 and all(90 <= s <= 240 for s in sleeps)   # no delay before the first send
    assert len(notes) == 1 and "Not sent" in notes[0]


async def test_worker_holds_at_cap_and_notifies_once(conn):
    now = [datetime(2026, 10, 23, 10, 0, tzinfo=timezone.utc)]
    q, notes, sleeps = queue(conn, max_per_day=1, clock=lambda: now[0])
    a1, _ = approved_app(conn, 1, "a@x.io")
    a2, _ = approved_app(conn, 2, "b@x.io")
    await q.send_one(a1)

    async def sleep(s):
        sleeps.append(s)
        now[0] += timedelta(seconds=s)   # time passes until tomorrow

    q.sleep = sleep
    await q.enqueue(a2)
    import asyncio
    task = asyncio.create_task(q.run())
    await q._queue.join()
    task.cancel()
    assert len([n for n in notes if "Daily cap" in n]) == 1
    assert q.gmail.sent == ["d1", "d2"]


def test_local_midnight_is_utc_aware():
    m = local_midnight_utc(datetime(2026, 10, 23, 12, tzinfo=timezone.utc))
    assert m.tzinfo == timezone.utc and m <= datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
