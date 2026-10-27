"""SQLite schema and queries. Timestamps are stored as ISO-8601 UTC strings."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

# --- job status state machine -------------------------------------------------
EXTRACTED = "extracted"
SKIPPED_FILTER = "skipped_filter"
DUPLICATE = "duplicate"
FLAGGED = "flagged"
DRAFTED = "drafted"
AWAITING_APPROVAL = "awaiting_approval"
SENT = "sent"
SUBMITTED = "submitted"
SKIPPED_BY_USER = "skipped_by_user"
FAILED = "failed"
INTERVIEW = "interview"
REJECTED = "rejected"
OFFER = "offer"

TRANSITIONS: dict[str, set[str]] = {
    EXTRACTED: {SKIPPED_FILTER, DUPLICATE, FLAGGED, DRAFTED, AWAITING_APPROVAL, FAILED},
    FLAGGED: {DRAFTED, AWAITING_APPROVAL, SKIPPED_BY_USER, FAILED},
    DRAFTED: {AWAITING_APPROVAL, FAILED},
    AWAITING_APPROVAL: {SENT, SUBMITTED, SKIPPED_BY_USER, FAILED, DRAFTED},
    FAILED: {DRAFTED, AWAITING_APPROVAL, SKIPPED_BY_USER},
    SENT: {INTERVIEW, REJECTED, OFFER},
    SUBMITTED: {INTERVIEW, REJECTED, OFFER},
    INTERVIEW: {REJECTED, OFFER},
    SKIPPED_FILTER: {AWAITING_APPROVAL},  # manual rescue
    DUPLICATE: set(),
    SKIPPED_BY_USER: set(),
    REJECTED: set(),
    OFFER: set(),
}
ALL_STATUSES = tuple(TRANSITIONS)
MANUAL_STATUSES = (INTERVIEW, REJECTED, OFFER)

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_posts (
    id              INTEGER PRIMARY KEY,
    tg_message_id   INTEGER NOT NULL,
    channel         TEXT NOT NULL,
    text            TEXT,
    entities_json   TEXT,                 -- hidden hyperlink URLs from message.entities
    posted_at       TIMESTAMP,
    edited_at       TIMESTAMP,
    processed       INTEGER DEFAULT 0,    -- 0 pending, 1 done, -1 extraction failed, 2 dropped by user
    UNIQUE (channel, tg_message_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id                INTEGER PRIMARY KEY,
    raw_post_id       INTEGER REFERENCES raw_posts(id),
    job_hash          TEXT UNIQUE,
    company           TEXT,
    role              TEXT,
    role_category     TEXT,
    batch_eligibility TEXT,
    location          TEXT,
    work_mode         TEXT,
    stipend_or_ctc    TEXT,
    required_skills   TEXT,               -- JSON array
    jd_summary        TEXT,
    deadline          TEXT,
    apply_method      TEXT,               -- email | link | both | unknown
    fit_score         INTEGER,
    fit_reason        TEXT,
    flags             TEXT,               -- JSON array
    resume_track      TEXT,
    status            TEXT NOT NULL DEFAULT 'extracted',
    created_at        TIMESTAMP,
    updated_at        TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS contacts (
    id          INTEGER PRIMARY KEY,
    job_id      INTEGER REFERENCES jobs(id),
    kind        TEXT NOT NULL,            -- email | link
    value       TEXT NOT NULL,            -- normalized email or resolved URL
    raw_value   TEXT,
    person_name TEXT,
    link_type   TEXT,
    UNIQUE (job_id, kind, value)
);

CREATE TABLE IF NOT EXISTS applications (
    id               INTEGER PRIMARY KEY,
    job_id           INTEGER REFERENCES jobs(id),
    contact_id       INTEGER REFERENCES contacts(id),
    gmail_draft_id   TEXT,
    gmail_message_id TEXT,
    subject          TEXT,
    body             TEXT,
    resume_file      TEXT,
    sent_at          TIMESTAMP,
    approval_msg_id  INTEGER,
    also_job_ids     TEXT                 -- JSON array: other roles from the same post this email covers
);
CREATE INDEX IF NOT EXISTS idx_applications_sent ON applications(sent_at);

CREATE TABLE IF NOT EXISTS recipients (
    email           TEXT PRIMARY KEY,
    last_emailed_at TIMESTAMP,
    total_emails    INTEGER DEFAULT 0
);

-- small key/value store for runtime flags (paused, last polled message id, ...)
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

JSON_COLUMNS = ("required_skills", "flags", "also_job_ids")

# columns added after the first release: (table, column, type)
MIGRATIONS = (("applications", "also_job_ids", "TEXT"),)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    in_memory = str(path) == ":memory:"
    if not in_memory:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not in_memory:
        conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, type_ in MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            with conn:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_}")


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for col in JSON_COLUMNS:
        if col in d and d[col] is not None:
            d[col] = json.loads(d[col])
    return d


# --- raw_posts ----------------------------------------------------------------
def upsert_raw_post(
    conn: sqlite3.Connection, *, tg_message_id: int, channel: str, text: str,
    entities: Iterable[str] = (), posted_at: str | None = None, edited: bool = False,
) -> tuple[int, bool]:
    """Insert or update a post. Returns (raw_post_id, changed).

    `changed` is False when an identical post already exists, so the caller can
    skip re-processing. An edit with new text resets `processed` to 0.
    """
    entities_json = json.dumps(sorted(set(entities)))
    existing = conn.execute(
        "SELECT id, text, entities_json FROM raw_posts WHERE channel=? AND tg_message_id=?",
        (channel, tg_message_id),
    ).fetchone()
    with conn:
        if existing is None:
            cur = conn.execute(
                "INSERT INTO raw_posts (tg_message_id, channel, text, entities_json, posted_at)"
                " VALUES (?,?,?,?,?)",
                (tg_message_id, channel, text, entities_json, posted_at or now()),
            )
            return cur.lastrowid, True
        if existing["text"] == text and existing["entities_json"] == entities_json:
            return existing["id"], False
        conn.execute(
            "UPDATE raw_posts SET text=?, entities_json=?, processed=0, edited_at=? WHERE id=?",
            (text, entities_json, now() if edited else None, existing["id"]),
        )
        return existing["id"], True


def get_raw_post(conn: sqlite3.Connection, raw_post_id: int) -> dict[str, Any] | None:
    row = _row(conn.execute("SELECT * FROM raw_posts WHERE id=?", (raw_post_id,)).fetchone())
    if row is not None:
        row["entities"] = json.loads(row.pop("entities_json") or "[]")
    return row


def unprocessed_posts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    ids = conn.execute("SELECT id FROM raw_posts WHERE processed=0 ORDER BY id").fetchall()
    return [get_raw_post(conn, r["id"]) for r in ids]


def mark_post_processed(conn: sqlite3.Connection, raw_post_id: int, value: int = 1) -> None:
    with conn:
        conn.execute("UPDATE raw_posts SET processed=? WHERE id=?", (value, raw_post_id))


POST_DROPPED = 2


def backlog_stats(conn: sqlite3.Connection) -> tuple[int, str | None]:
    """(number of posts waiting to be processed, posted_at of the oldest)."""
    row = conn.execute("SELECT COUNT(*) AS n, MIN(posted_at) AS oldest FROM raw_posts WHERE processed=0").fetchone()
    return row["n"], row["oldest"]


def drop_backlog(conn: sqlite3.Connection, older_than: str | None = None) -> int:
    """Mark waiting posts as dropped (never processed). `older_than`: ISO time; None = all waiting."""
    q, args = "UPDATE raw_posts SET processed=? WHERE processed=0", [POST_DROPPED]
    if older_than:
        q += " AND posted_at < ?"
        args.append(older_than)
    with conn:
        return conn.execute(q, args).rowcount


def last_message_id(conn: sqlite3.Connection, channel: str) -> int:
    row = conn.execute(
        "SELECT MAX(tg_message_id) AS m FROM raw_posts WHERE channel=?", (channel,)
    ).fetchone()
    return row["m"] or 0


# --- jobs ---------------------------------------------------------------------
JOB_FIELDS = (
    "raw_post_id", "job_hash", "company", "role", "role_category", "batch_eligibility",
    "location", "work_mode", "stipend_or_ctc", "required_skills", "jd_summary", "deadline",
    "apply_method", "fit_score", "fit_reason", "flags", "resume_track", "status",
)


def _encode(fields: dict[str, Any]) -> dict[str, Any]:
    out = dict(fields)
    for col in JSON_COLUMNS:
        if col in out and not isinstance(out[col], (str, type(None))):
            out[col] = json.dumps(out[col])
    return out


def insert_job(conn: sqlite3.Connection, **fields: Any) -> int:
    unknown = set(fields) - set(JOB_FIELDS)
    if unknown:
        raise ValueError(f"unknown job fields: {sorted(unknown)}")
    fields = _encode({"status": EXTRACTED, **fields})
    ts = now()
    cols = list(fields) + ["created_at", "updated_at"]
    with conn:
        cur = conn.execute(
            f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [*fields.values(), ts, ts],
        )
    return cur.lastrowid


def get_job(conn: sqlite3.Connection, job_id: int) -> dict[str, Any] | None:
    return _row(conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())


def get_job_by_hash(conn: sqlite3.Connection, job_hash: str) -> dict[str, Any] | None:
    return _row(conn.execute("SELECT * FROM jobs WHERE job_hash=?", (job_hash,)).fetchone())


def update_job(conn: sqlite3.Connection, job_id: int, **fields: Any) -> None:
    unknown = set(fields) - set(JOB_FIELDS)
    if unknown:
        raise ValueError(f"unknown job fields: {sorted(unknown)}")
    if not fields:
        return
    fields = _encode(fields)
    sets = ", ".join(f"{k}=?" for k in fields)
    with conn:
        conn.execute(
            f"UPDATE jobs SET {sets}, updated_at=? WHERE id=?",
            [*fields.values(), now(), job_id],
        )


class InvalidTransition(ValueError):
    pass


def set_status(conn: sqlite3.Connection, job_id: int, status: str, *, force: bool = False) -> None:
    job = get_job(conn, job_id)
    if job is None:
        raise KeyError(f"job {job_id} not found")
    if status not in TRANSITIONS:
        raise InvalidTransition(f"unknown status {status!r}")
    if not force and status != job["status"] and status not in TRANSITIONS[job["status"]]:
        raise InvalidTransition(f"job {job_id}: {job['status']} -> {status} not allowed")
    update_job(conn, job_id, status=status)


def jobs_by_status(conn: sqlite3.Connection, *statuses: str) -> list[dict[str, Any]]:
    q = f"SELECT * FROM jobs WHERE status IN ({', '.join('?' * len(statuses))}) ORDER BY id"
    return [_row(r) for r in conn.execute(q, statuses).fetchall()]


def status_counts(conn: sqlite3.Connection, since: str | None = None) -> dict[str, int]:
    q, args = "SELECT status, COUNT(*) AS n FROM jobs", []
    if since:
        q += " WHERE created_at >= ?"
        args.append(since)
    q += " GROUP BY status"
    return {r["status"]: r["n"] for r in conn.execute(q, args).fetchall()}


# --- contacts -----------------------------------------------------------------
def add_contact(
    conn: sqlite3.Connection, job_id: int, *, kind: str, value: str,
    raw_value: str | None = None, person_name: str | None = None, link_type: str | None = None,
) -> int:
    """Insert a contact; returns the existing id if this job already has it."""
    if kind not in ("email", "link"):
        raise ValueError(f"bad contact kind {kind!r}")
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO contacts (job_id, kind, value, raw_value, person_name, link_type)"
            " VALUES (?,?,?,?,?,?)",
            (job_id, kind, value, raw_value, person_name, link_type),
        )
    return conn.execute(
        "SELECT id FROM contacts WHERE job_id=? AND kind=? AND value=?", (job_id, kind, value)
    ).fetchone()["id"]


def contacts_for_job(conn: sqlite3.Connection, job_id: int) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM contacts WHERE job_id=? ORDER BY id", (job_id,))
    return [dict(r) for r in rows.fetchall()]


# --- applications -------------------------------------------------------------
APPLICATION_FIELDS = (
    "job_id", "contact_id", "gmail_draft_id", "gmail_message_id", "subject", "body",
    "resume_file", "sent_at", "approval_msg_id", "also_job_ids",
)


def insert_application(conn: sqlite3.Connection, **fields: Any) -> int:
    unknown = set(fields) - set(APPLICATION_FIELDS)
    if unknown:
        raise ValueError(f"unknown application fields: {sorted(unknown)}")
    fields = _encode(fields)
    cols = list(fields)
    with conn:
        cur = conn.execute(
            f"INSERT INTO applications ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            list(fields.values()),
        )
    return cur.lastrowid


def update_application(conn: sqlite3.Connection, app_id: int, **fields: Any) -> None:
    unknown = set(fields) - set(APPLICATION_FIELDS)
    if unknown:
        raise ValueError(f"unknown application fields: {sorted(unknown)}")
    if not fields:
        return
    fields = _encode(fields)
    sets = ", ".join(f"{k}=?" for k in fields)
    with conn:
        conn.execute(f"UPDATE applications SET {sets} WHERE id=?", [*fields.values(), app_id])


def get_application(conn: sqlite3.Connection, app_id: int) -> dict[str, Any] | None:
    return _row(conn.execute("SELECT * FROM applications WHERE id=?", (app_id,)).fetchone())


def applications_for_job(conn: sqlite3.Connection, job_id: int) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM applications WHERE job_id=? ORDER BY id", (job_id,))
    return [_row(r) for r in rows.fetchall()]


def delete_planned_applications(conn: sqlite3.Connection, job_id: int) -> int:
    """Remove applications that were planned but never drafted or sent (used on re-decide)."""
    with conn:
        cur = conn.execute(
            "DELETE FROM applications WHERE job_id=? AND gmail_draft_id IS NULL AND sent_at IS NULL",
            (job_id,),
        )
    return cur.rowcount


# statuses where a planned (unsent) email still counts as "on its way"; FAILED is
# retryable, so it keeps holding its recipient until the user skips it
OPEN_STATUSES = (EXTRACTED, FLAGGED, DRAFTED, AWAITING_APPROVAL, FAILED)


def open_applications_to(
    conn: sqlite3.Connection, email: str, exclude_raw_post_id: int | None = None,
) -> list[dict[str, Any]]:
    """Unsent applications to `email` whose job is still open, newest first."""
    q = (
        "SELECT a.*, j.role AS job_role, j.company AS job_company, j.raw_post_id"
        " FROM applications a JOIN contacts c ON c.id = a.contact_id JOIN jobs j ON j.id = a.job_id"
        f" WHERE c.value = ? AND a.sent_at IS NULL AND j.status IN ({', '.join('?' * len(OPEN_STATUSES))})"
    )
    args: list[Any] = [email.lower(), *OPEN_STATUSES]
    if exclude_raw_post_id is not None:
        q += " AND j.raw_post_id != ?"
        args.append(exclude_raw_post_id)
    return [_row(r) for r in conn.execute(q + " ORDER BY a.id DESC", args).fetchall()]


def last_sent_application_to(conn: sqlite3.Connection, email: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT a.*, j.role AS job_role, j.company AS job_company"
        " FROM applications a JOIN contacts c ON c.id = a.contact_id JOIN jobs j ON j.id = a.job_id"
        " WHERE c.value = ? AND a.sent_at IS NOT NULL ORDER BY a.sent_at DESC LIMIT 1",
        (email.lower(),),
    ).fetchone()
    return _row(row)


def emails_sent_since(conn: sqlite3.Connection, since: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM applications WHERE sent_at IS NOT NULL AND sent_at >= ?",
        (since,),
    ).fetchone()["n"]


# --- recipients ---------------------------------------------------------------
def record_send(conn: sqlite3.Connection, email: str, at: str | None = None) -> None:
    at = at or now()
    with conn:
        conn.execute(
            "INSERT INTO recipients (email, last_emailed_at, total_emails) VALUES (?,?,1)"
            " ON CONFLICT(email) DO UPDATE SET last_emailed_at=excluded.last_emailed_at,"
            " total_emails=total_emails+1",
            (email.lower(), at),
        )


def get_recipient(conn: sqlite3.Connection, email: str) -> dict[str, Any] | None:
    return _row(conn.execute("SELECT * FROM recipients WHERE email=?", (email.lower(),)).fetchone())


def recently_emailed(conn: sqlite3.Connection, email: str, days: int) -> dict[str, Any] | None:
    """Return the recipient row if they were emailed within `days`, else None."""
    rec = get_recipient(conn, email)
    if rec is None or rec["last_emailed_at"] is None:
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return rec if datetime.fromisoformat(rec["last_emailed_at"]) >= cutoff else None


# --- kv -----------------------------------------------------------------------
def kv_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def kv_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
