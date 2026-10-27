"""Job hashing, duplicate detection, recipient cooldown and per-post email planning."""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from jobpilot import db
from jobpilot.extractor import ExtractedJob

log = logging.getLogger(__name__)

# --- hashing ------------------------------------------------------------------
_ROLE_NOISE = {
    "intern", "interns", "internship", "internships", "hiring", "we", "are", "is", "for", "the",
    "a", "an", "urgent", "urgently", "opening", "openings", "position", "role", "job", "jobs",
    "vacancy", "required", "wanted", "fresher", "freshers",
}
_COMPANY_NOISE = {
    "pvt", "private", "ltd", "limited", "inc", "llp", "llc", "corp", "corporation", "co",
    "the", "hiring", "is", "are", "we",
}
_ROLE_SYNONYMS = {
    "ml": "machine learning", "ai/ml": "ai machine learning", "sde": "software engineer",
    "swe": "software engineer", "developer": "engineer", "dev": "engineer", "fullstack": "full stack",
    "full-stack": "full stack", "front-end": "frontend", "back-end": "backend", "sr": "senior",
    "jr": "junior",
}


def _norm(s: str | None, noise: set[str], synonyms: dict[str, str] | None = None) -> str:
    s = (s or "").lower().replace("&", " and ")
    words = re.sub(r"[^a-z0-9+#/-]+", " ", s).split()
    if synonyms:
        words = " ".join(synonyms.get(w, w) for w in words).split()
    words = re.sub(r"[^a-z0-9+# ]+", " ", " ".join(words)).split()
    return " ".join(w for w in words if w not in noise)


def job_hash(company: str | None, role: str | None) -> str:
    key = f"{_norm(company, _COMPANY_NOISE)}|{_norm(role, _ROLE_NOISE, _ROLE_SYNONYMS)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# --- registration (new / edited / duplicate) ------------------------------------
NEW, UPDATED, DUPLICATE, UNCHANGED = "new", "updated", "duplicate", "unchanged"
# a job from an edited post is re-decided only while nothing has been drafted or sent
REDECIDABLE = (db.EXTRACTED, db.FLAGGED, db.SKIPPED_FILTER)


@dataclass
class Registered:
    job_id: int                    # the job to decide on (or the duplicate row)
    kind: str                      # new | updated | duplicate | unchanged
    existing_id: int | None = None
    new_contacts: list[str] = field(default_factory=list)


def _job_fields(job: ExtractedJob, flags: list[str]) -> dict:
    return dict(
        company=job.company, role=job.role, role_category=job.role_category,
        batch_eligibility=job.batch_eligibility, location=job.location, work_mode=job.work_mode,
        stipend_or_ctc=job.stipend_or_ctc, required_skills=job.required_skills,
        jd_summary=job.jd_summary, deadline=job.deadline, apply_method=job.apply_method, flags=flags,
    )


def _attach_contacts(conn: sqlite3.Connection, job_id: int, job: ExtractedJob,
                     resolved: dict[str, str]) -> list[str]:
    before = {c["value"] for c in db.contacts_for_job(conn, job_id)}
    added = []
    for c in job.contacts:
        value = resolved.get(c.value, c.value) if c.kind == "link" else c.value
        db.add_contact(conn, job_id, kind=c.kind, value=value, raw_value=c.value,
                       person_name=c.person_name)
        if value not in before:
            added.append(value)
            before.add(value)
    return added


def register_job(conn: sqlite3.Connection, raw_post_id: int, job: ExtractedJob,
                 resolved: dict[str, str], flags: list[str]) -> Registered:
    """Store an extracted job, or recognise it as an edit or a duplicate."""
    h = job_hash(job.company, job.role)
    existing = db.get_job_by_hash(conn, h)

    if existing is None:
        job_id = db.insert_job(conn, raw_post_id=raw_post_id, job_hash=h, **_job_fields(job, flags))
        return Registered(job_id, NEW, new_contacts=_attach_contacts(conn, job_id, job, resolved))

    if existing["raw_post_id"] == raw_post_id:
        # same post seen again (edited): refresh it if it hasn't moved past the filter
        added = _attach_contacts(conn, existing["id"], job, resolved)
        if existing["status"] in REDECIDABLE and not any(
            a["gmail_draft_id"] or a["sent_at"] for a in db.applications_for_job(conn, existing["id"])
        ):
            db.update_job(conn, existing["id"], **_job_fields(job, flags))
            return Registered(existing["id"], UPDATED, existing["id"], added)
        return Registered(existing["id"], UNCHANGED, existing["id"], added)

    # a different post for a job we already have: record the duplicate, merge contacts
    added = _attach_contacts(conn, existing["id"], job, resolved)
    dup_flags = [f"duplicate of job #{existing['id']} ({existing['status']})"]
    if added:
        dup_flags.append(f"new contacts added to job #{existing['id']}: {', '.join(added)}")
    dup_id = db.insert_job(conn, raw_post_id=raw_post_id, job_hash=None, status=db.DUPLICATE,
                           **_job_fields(job, dup_flags))
    return Registered(dup_id, DUPLICATE, existing["id"], added)


# --- recipient cooldown -----------------------------------------------------------
@dataclass
class RecipientConflict:
    email: str
    kind: str                 # "sent" (emailed within cooldown) | "pending" (unsent email planned)
    other_role: str
    other_job_id: int
    when: str | None = None   # ISO timestamp of the last send

    def message(self, this_role: str, this_job_id: int) -> str:
        if self.kind == "sent":
            day = datetime.fromisoformat(self.when).strftime("%d %b") if self.when else "recently"
            head = f"Already emailed {self.email} on {day} for {self.other_role} (job #{self.other_job_id})"
        else:
            head = (f"An email to {self.email} for {self.other_role} (job #{self.other_job_id}) "
                    f"is already waiting for your approval")
        return f"{head}; this post is for {this_role} (job #{this_job_id}). Combine / Send anyway / Skip?"


def recipient_conflict(conn: sqlite3.Connection, email: str, cooldown_days: int,
                       exclude_raw_post_id: int | None = None) -> RecipientConflict | None:
    if rec := db.recently_emailed(conn, email, cooldown_days):
        last = db.last_sent_application_to(conn, email)
        return RecipientConflict(
            email, "sent", last["job_role"] if last else "an earlier role",
            last["job_id"] if last else 0, rec["last_emailed_at"],
        )
    if pending := db.open_applications_to(conn, email, exclude_raw_post_id):
        p = pending[0]
        return RecipientConflict(email, "pending", p["job_role"], p["job_id"])
    return None


# --- per-post email planning ---------------------------------------------------------
_GENERIC_LOCAL = re.compile(r"^(?:hr|careers?|jobs?|hiring|recruit\w*|talent\w*|info|contact|apply|"
                            r"people|team|admin|hello|support|internships?)\b", re.I)


def contact_preference(contact: dict) -> tuple[int, int]:
    """Lower sorts first: a named person, then a personal-looking address, then generic inboxes."""
    named = 0 if contact.get("person_name") else 1
    generic = 1 if _GENERIC_LOCAL.match(contact["value"].split("@")[0]) else 0
    return named, generic


@dataclass
class EmailPlan:
    applications: list[int] = field(default_factory=list)   # application ids created
    notices: list[str] = field(default_factory=list)        # cooldown messages for the user


def plan_emails(conn: sqlite3.Connection, job_ids: list[int], raw_post_id: int,
                cooldown_days: int, resume_tracks: dict[int, str]) -> EmailPlan:
    """Create one planned application per recipient for the decided jobs of one post.

    - each job emails its preferred contact (named person > personal > generic inbox),
      skipping contacts in cooldown or with another unsent email pending;
    - a recipient picked by several jobs gets ONE email, for the best-fit job; other roles
      that also passed the filter are listed in `also_job_ids`;
    - jobs whose every email contact is blocked are flagged and the user is notified.
    """
    plan = EmailPlan()
    jobs = {jid: db.get_job(conn, jid) for jid in job_ids}
    eligible = {jid: j for jid, j in jobs.items() if j["status"] in (db.EXTRACTED, db.FLAGGED)}

    chosen: dict[str, list[tuple[int, dict]]] = {}     # email -> [(job_id, contact)]
    for jid, job in eligible.items():
        emails = sorted((c for c in db.contacts_for_job(conn, jid) if c["kind"] == "email"),
                        key=contact_preference)
        if not emails:
            continue
        conflicts = []
        pick = None
        for c in emails:
            conflict = recipient_conflict(conn, c["value"], cooldown_days, exclude_raw_post_id=raw_post_id)
            if conflict is None:
                pick = c
                break
            conflicts.append(conflict)
        if pick is None:
            msgs = [c.message(job["role"], jid) for c in conflicts]
            db.update_job(conn, jid, status=db.FLAGGED,
                          flags=_merge(job["flags"], [f"cooldown: {m}" for m in msgs]))
            plan.notices += msgs
            continue
        others = [c["value"] for c in emails if c["value"] != pick["value"]]
        if others:
            db.update_job(conn, jid, flags=_merge(job["flags"], [f"other contacts: {', '.join(others)}"]))
        chosen.setdefault(pick["value"], []).append((jid, pick))

    for email, picks in chosen.items():
        picks.sort(key=lambda p: (-(eligible[p[0]]["fit_score"] or 0), p[0]))
        (primary_id, contact), rest = picks[0], picks[1:]
        also = [jid for jid, _ in rest]
        plan.applications.append(db.insert_application(
            conn, job_id=primary_id, contact_id=contact["id"],
            resume_file=resume_tracks.get(primary_id), also_job_ids=also or None,
        ))
        for jid in also:
            job = db.get_job(conn, jid)
            note = f"covered by the one email to {email} for job #{primary_id} (same recipient)"
            has_link = any(c["kind"] == "link" for c in db.contacts_for_job(conn, jid))
            db.update_job(conn, jid, flags=_merge(job["flags"], [note]),
                          **({} if has_link else {"status": db.DUPLICATE}))
    return plan


def _merge(existing: list[str] | None, new: list[str]) -> list[str]:
    out = list(existing or [])
    seen = {f.lower() for f in out}
    for f in new:
        if f.lower() not in seen:
            out.append(f)
            seen.add(f.lower())
    return out
