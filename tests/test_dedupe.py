from datetime import datetime, timedelta, timezone

import pytest

from jobpilot import db
from jobpilot.dedupe import (
    DUPLICATE, NEW, UNCHANGED, UPDATED, contact_preference, job_hash, plan_emails,
    recipient_conflict, register_job,
)
from jobpilot.extractor import ExtractedContact, ExtractedJob


@pytest.mark.parametrize("a, b", [
    (("Acme AI", "ML Intern"), ("Acme AI Pvt. Ltd.", "Machine Learning Internship")),
    (("Orbitly", "SDE Intern"), ("orbitly", "Software Engineer - Intern")),
    (("Kestrel", "Full-Stack Developer"), ("Kestrel Inc", "Fullstack Engineer")),
    (("Acme", "We are hiring: Backend Intern!"), ("ACME", "Backend")),
])
def test_hash_equivalent(a, b):
    assert job_hash(*a) == job_hash(*b)


@pytest.mark.parametrize("a, b", [
    (("Acme", "ML Intern"), ("Acme", "Data Engineer Intern")),
    (("Acme", "ML Intern"), ("Zentrix", "ML Intern")),
])
def test_hash_distinct(a, b):
    assert job_hash(*a) != job_hash(*b)


def ej(role="ML Intern", company="Acme", emails=(), links=()):
    contacts = [ExtractedContact(kind="email", value=e) for e in emails]
    contacts += [ExtractedContact(kind="link", value=u) for u in links]
    return ExtractedJob(company=company, role=role, role_category="ai_ml", contacts=contacts)


def post(conn, n):
    return db.upsert_raw_post(conn, tg_message_id=n, channel="c", text=f"post {n}")[0]


def test_register_new_duplicate_and_edit(conn):
    p1, p2 = post(conn, 1), post(conn, 2)
    r1 = register_job(conn, p1, ej(emails=["a@acme.ai"]), {}, ["note"])
    assert r1.kind == NEW and r1.new_contacts == ["a@acme.ai"]

    r2 = register_job(conn, p2, ej(role="Machine Learning Internship", emails=["a@acme.ai", "b@acme.ai"]), {}, [])
    assert r2.kind == DUPLICATE and r2.existing_id == r1.job_id and r2.new_contacts == ["b@acme.ai"]
    dup = db.get_job(conn, r2.job_id)
    assert dup["status"] == db.DUPLICATE and dup["job_hash"] is None
    assert f"duplicate of job #{r1.job_id}" in dup["flags"][0]
    assert {c["value"] for c in db.contacts_for_job(conn, r1.job_id)} == {"a@acme.ai", "b@acme.ai"}

    # edit of the original post while still undecided -> re-decide
    assert register_job(conn, p1, ej(emails=["a@acme.ai"]), {}, []).kind == UPDATED
    # once drafted, an edit only attaches contacts
    db.insert_application(conn, job_id=r1.job_id, gmail_draft_id="d1")
    assert register_job(conn, p1, ej(emails=["c@acme.ai"]), {}, []).kind == UNCHANGED


def test_register_stores_resolved_link(conn):
    r = register_job(conn, post(conn, 1), ej(links=["https://bit.ly/x"]), {"https://bit.ly/x": "https://forms.gle/y"}, [])
    (c,) = db.contacts_for_job(conn, r.job_id)
    assert (c["value"], c["raw_value"]) == ("https://forms.gle/y", "https://bit.ly/x")


def decided(conn, raw_post_id, role, emails, score=80, status=db.EXTRACTED, names=None, links=()):
    r = register_job(conn, raw_post_id, ej(role=role, emails=emails, links=links), {}, [])
    for c in db.contacts_for_job(conn, r.job_id):
        if names and c["value"] in names:
            conn.execute("UPDATE contacts SET person_name=? WHERE id=?", (names[c["value"]], c["id"]))
    conn.commit()
    db.update_job(conn, r.job_id, fit_score=score, status=status)
    return r.job_id


def planned_to(conn, job_id):
    return [conn.execute("SELECT value FROM contacts WHERE id=?", (a["contact_id"],)).fetchone()[0]
            for a in db.applications_for_job(conn, job_id)]


def test_same_recipient_one_email_best_fit(conn):
    p = post(conn, 1)
    a = decided(conn, p, "Data Analyst", ["jobs@orbitly.io"], score=66, status=db.FLAGGED)
    b = decided(conn, p, "DevOps Intern", ["jobs@orbitly.io"], score=75)
    c = decided(conn, p, "Backend Intern", ["jobs@orbitly.io"], score=50, status=db.SKIPPED_FILTER)
    plan = plan_emails(conn, [a, b, c], p, 7, {a: "ai_ml.pdf", b: "sde.pdf"})
    assert len(plan.applications) == 1
    (app,) = db.applications_for_job(conn, b)
    assert app["also_job_ids"] == [a] and app["resume_file"] == "sde.pdf"
    assert db.applications_for_job(conn, a) == [] and db.get_job(conn, a)["status"] == db.DUPLICATE
    assert db.applications_for_job(conn, c) == []


def test_covered_job_with_link_keeps_status(conn):
    p = post(conn, 1)
    a = decided(conn, p, "Data Analyst", ["jobs@o.io"], score=90)
    b = decided(conn, p, "DevOps", ["jobs@o.io"], score=70, links=["https://forms.gle/z"])
    plan_emails(conn, [a, b], p, 7, {})
    assert db.get_job(conn, b)["status"] == db.EXTRACTED
    assert any("covered by the one email" in f for f in db.get_job(conn, b)["flags"])


def test_prefers_named_person_then_personal_over_generic():
    assert sorted(
        [{"value": "careers@z.ai", "person_name": None}, {"value": "neha@z.ai", "person_name": None},
         {"value": "hr@z.ai", "person_name": "Priya"}],
        key=contact_preference,
    )[0]["value"] == "hr@z.ai"
    assert contact_preference({"value": "neha@z.ai"}) < contact_preference({"value": "careers@z.ai"})


def test_cooldown_after_send(conn):
    p1, p2 = post(conn, 1), post(conn, 2)
    j1 = decided(conn, p1, "SDE Intern", ["priya@x.com"])
    plan_emails(conn, [j1], p1, 7, {})
    (app,) = db.applications_for_job(conn, j1)
    sent = datetime.now(timezone.utc) - timedelta(days=2)
    db.update_application(conn, app["id"], sent_at=sent.isoformat())
    db.set_status(conn, j1, db.DRAFTED)
    db.set_status(conn, j1, db.AWAITING_APPROVAL)
    db.set_status(conn, j1, db.SENT)
    db.record_send(conn, "priya@x.com", at=sent.isoformat())

    j2 = decided(conn, p2, "ML Intern", ["priya@x.com"])
    plan = plan_emails(conn, [j2], p2, 7, {})
    assert plan.applications == []
    assert plan.notices[0].startswith(f"Already emailed priya@x.com on {sent.strftime('%d %b')} for SDE Intern")
    assert "this post is for ML Intern" in plan.notices[0] and plan.notices[0].endswith("Combine / Send anyway / Skip?")
    assert db.get_job(conn, j2)["status"] == db.FLAGGED


def test_cooldown_expired(conn):
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    db.record_send(conn, "priya@x.com", at=old)
    assert recipient_conflict(conn, "priya@x.com", 7) is None


def test_pending_email_blocks_second_post_but_falls_back(conn):
    p1, p2, p3 = post(conn, 1), post(conn, 2), post(conn, 3)
    j1 = decided(conn, p1, "ML Intern", ["priya@acme.ai"])
    plan_emails(conn, [j1], p1, 7, {})

    j2 = decided(conn, p2, "Data Engineer", ["priya@acme.ai"])
    plan = plan_emails(conn, [j2], p2, 7, {})
    assert plan.applications == [] and "already waiting for your approval" in plan.notices[0]

    # a post with another usable address uses that one instead
    j3 = decided(conn, p3, "Platform Intern", ["priya@acme.ai", "careers@acme.ai"])
    plan = plan_emails(conn, [j3], p3, 7, {})
    assert plan.notices == [] and planned_to(conn, j3) == ["careers@acme.ai"]


def test_skipped_job_frees_recipient(conn):
    p1, p2 = post(conn, 1), post(conn, 2)
    j1 = decided(conn, p1, "ML Intern", ["priya@acme.ai"])
    plan_emails(conn, [j1], p1, 7, {})
    db.update_job(conn, j1, status=db.SKIPPED_BY_USER)
    j2 = decided(conn, p2, "Data Engineer", ["priya@acme.ai"])
    assert plan_emails(conn, [j2], p2, 7, {}).applications
