from datetime import datetime, timedelta, timezone

import pytest

from jobpilot import db


def test_raw_post_upsert_and_edit(conn):
    pid, changed = db.upsert_raw_post(conn, tg_message_id=10, channel="@jobs", text="hi",
                                      entities=["https://a.com"])
    assert changed
    assert db.upsert_raw_post(conn, tg_message_id=10, channel="@jobs", text="hi",
                              entities=["https://a.com"]) == (pid, False)

    db.mark_post_processed(conn, pid)
    _, changed = db.upsert_raw_post(conn, tg_message_id=10, channel="@jobs", text="hi (fixed)",
                                    edited=True)
    assert changed
    post = db.get_raw_post(conn, pid)
    assert post["processed"] == 0 and post["edited_at"] and post["entities"] == []
    assert db.last_message_id(conn, "@jobs") == 10
    assert [p["id"] for p in db.unprocessed_posts(conn)] == [pid]


def test_job_roundtrip_json_columns(conn):
    jid = db.insert_job(conn, job_hash="h1", company="Acme", role="ML Intern",
                        required_skills=["python", "pytorch"], flags=[])
    job = db.get_job(conn, jid)
    assert job["status"] == db.EXTRACTED
    assert job["required_skills"] == ["python", "pytorch"]
    assert db.get_job_by_hash(conn, "h1")["id"] == jid
    with pytest.raises(ValueError):
        db.insert_job(conn, job_hash="h2", bogus=1)


def test_status_transitions(conn):
    jid = db.insert_job(conn, job_hash="h")
    db.set_status(conn, jid, db.DRAFTED)
    db.set_status(conn, jid, db.AWAITING_APPROVAL)
    with pytest.raises(db.InvalidTransition):
        db.set_status(conn, jid, db.OFFER)
    db.set_status(conn, jid, db.SENT)
    db.set_status(conn, jid, db.INTERVIEW)
    assert db.status_counts(conn) == {db.INTERVIEW: 1}


def test_contacts_dedupe_per_job(conn):
    jid = db.insert_job(conn, job_hash="h")
    a = db.add_contact(conn, jid, kind="email", value="hr@acme.ai", person_name="Priya")
    b = db.add_contact(conn, jid, kind="email", value="hr@acme.ai")
    assert a == b and len(db.contacts_for_job(conn, jid)) == 1


def test_recipient_cooldown(conn):
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    db.record_send(conn, "Old@X.com", at=old)
    db.record_send(conn, "new@x.com")
    assert db.recently_emailed(conn, "old@x.com", 7) is None
    assert db.recently_emailed(conn, "new@x.com", 7)["total_emails"] == 1
    db.record_send(conn, "new@x.com")
    assert db.get_recipient(conn, "NEW@x.com")["total_emails"] == 2


def test_daily_send_count(conn):
    jid = db.insert_job(conn, job_hash="h")
    db.insert_application(conn, job_id=jid, sent_at=db.now())
    db.insert_application(conn, job_id=jid)  # draft only
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).isoformat()
    assert db.emails_sent_since(conn, start) == 1


def test_kv(conn):
    assert db.kv_get(conn, "paused", "0") == "0"
    db.kv_set(conn, "paused", "1")
    db.kv_set(conn, "paused", "0")
    assert db.kv_get(conn, "paused") == "0"
