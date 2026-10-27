"""Per-post processing: extract -> resolve links -> register (dedupe) -> filter -> resume -> plan emails.

Phase 4 drafts the planned applications; Phase 5 turns cards into Telegram approvals.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from jobpilot import db
from jobpilot.config import AppConfig, Profile, get_config
from jobpilot.dedupe import DUPLICATE, NEW, UPDATED, Registered, plan_emails, register_job
from jobpilot.email_drafter import attachment_name, build_subject, draft_email
from jobpilot.extractor import ExtractedJob, ExtractionError, extract
from jobpilot.filter import SKIP, evaluate
from jobpilot.gmail_client import GmailClient
from jobpilot.llm import LLM
from jobpilot.normalize import resolve_all
from jobpilot.resume_picker import pick_resume

log = logging.getLogger(__name__)

Notify = Callable[[str], Awaitable[None]]


async def log_notify(text: str) -> None:
    log.warning("NOTIFY: %s", text)


@dataclass
class Pipeline:
    conn: sqlite3.Connection
    llm: LLM
    profile: Profile
    cfg: AppConfig = field(default_factory=get_config)
    dry_run: bool = False
    resolve_links: bool = True
    notify: Notify = log_notify
    http: httpx.AsyncClient | None = None
    gmail: GmailClient | None = None          # None (or dry_run) -> emails are written but no Gmail draft
    echo: Callable[[str], None] = print       # where dry-run cards go
    on_ready: Callable[[list[int]], Awaitable[None]] | None = None   # approval bot: post cards for these jobs
    _own_http: bool = field(default=False, init=False)

    async def __aenter__(self) -> "Pipeline":
        if self.http is None and self.resolve_links:
            self.http = httpx.AsyncClient(
                follow_redirects=True, timeout=10.0,
                headers={"User-Agent": "Mozilla/5.0 (JobPilot link resolver)"},
            )
            self._own_http = True
        return self

    async def __aexit__(self, *exc) -> None:
        if self._own_http and self.http is not None:
            await self.http.aclose()

    async def process_post(self, raw_post_id: int) -> list[int]:
        """Process one stored post. Returns ids of jobs decided on. Never raises;
        failures are reported via notify."""
        post = db.get_raw_post(self.conn, raw_post_id)
        if post is None:
            return []
        try:
            return await self._process(post)
        except ExtractionError as e:
            db.mark_post_processed(self.conn, raw_post_id, -1)
            await self.notify(f"Could not extract post #{post['tg_message_id']}: {e}")
        except Exception as e:  # noqa: BLE001 - one bad post must not stop the listener
            log.exception("processing post %s failed", raw_post_id)
            db.mark_post_processed(self.conn, raw_post_id, -1)
            await self.notify(f"Error processing post #{post['tg_message_id']}: {type(e).__name__}: {e}")
        return []

    async def _process(self, post: dict) -> list[int]:
        result = await extract(post["text"] or "", post["entities"], self.llm)
        resolved = await self._resolve(result.jobs)

        regs: list[Registered] = []
        for job in result.jobs:
            flags = list(job.post_quality_notes)
            if result.dropped:
                flags.append(f"removed contact(s) not present in the post: {', '.join(result.dropped)}")
            regs.append(register_job(self.conn, post["id"], job, resolved, flags))

        decide = [r.job_id for r in regs if r.kind in (NEW, UPDATED)]
        tracks: dict[int, str] = {}
        for r in regs:
            if r.kind == UPDATED:
                db.delete_planned_applications(self.conn, r.job_id)
        for jid in decide:
            tracks[jid] = await self._decide(jid, post["text"] or "")
        plan = plan_emails(self.conn, decide, post["id"], self.cfg.recipient_cooldown_days,
                           {jid: self.cfg.resume_file(t).name for jid, t in tracks.items() if t})
        for app_id in plan.applications:
            await self.draft_application(app_id)
        db.mark_post_processed(self.conn, post["id"])

        if self.on_ready is not None:
            await self.on_ready(decide)     # the approval bot posts cards (incl. cooldown choices)
        else:
            for msg in plan.notices:
                await self.notify(msg)
        if self.dry_run:
            if not result.jobs:
                self.echo(f"[post #{post['tg_message_id']}] no jobs found")
            for r in regs:
                if r.kind == DUPLICATE:
                    self.echo(f"[duplicate] job #{r.job_id} is a repost of job #{r.existing_id}; "
                              f"new contacts attached: {', '.join(r.new_contacts) or 'none'}")
                elif r.kind in (NEW, UPDATED):
                    self.echo(render_card(self.conn, r.job_id))
        log.info("post #%s -> %d job(s), %d planned email(s)", post["tg_message_id"],
                 len(decide), len(plan.applications))
        return decide

    async def _resolve(self, jobs: list[ExtractedJob]) -> dict[str, str]:
        urls = sorted({c.value for j in jobs for c in j.contacts if c.kind == "link"})
        if not urls or not self.resolve_links:
            return {u: u for u in urls}
        return await resolve_all(urls, self.http)

    def resume_path(self, app: dict) -> Path:
        return self.cfg.path("resumes_dir") / app["resume_file"]

    async def sync_gmail_draft(self, app_id: int) -> None:
        """Create or update the Gmail draft from the application's stored subject/body/resume.
        No-op in dry-run or without Gmail. Raises on Gmail errors."""
        if self.gmail is None or self.dry_run:
            return
        app = db.get_application(self.conn, app_id)
        to = self.conn.execute("SELECT value FROM contacts WHERE id=?", (app["contact_id"],)).fetchone()[0]
        args = (to, app["subject"], app["body"], self.resume_path(app), attachment_name(self.profile))
        if app["gmail_draft_id"]:
            await self.gmail.update_draft(app["gmail_draft_id"], *args)
        else:
            db.update_application(self.conn, app_id, gmail_draft_id=await self.gmail.create_draft(*args))

    async def draft_application(self, app_id: int) -> bool:
        """Write + validate the email for a planned application, then create the Gmail draft
        (skipped in dry-run). Failures mark the job `failed` and notify; nothing is sent here.
        Returns True if the job ends up `drafted`."""
        app = db.get_application(self.conn, app_id)
        job = db.get_job(self.conn, app["job_id"])
        contact = next(c for c in db.contacts_for_job(self.conn, job["id"]) if c["id"] == app["contact_id"])
        also = [db.get_job(self.conn, j) for j in app["also_job_ids"] or []]
        resume = self.cfg.resume_file(job["resume_track"] or "ai_ml")

        async def fail(reason: str) -> None:
            db.update_job(self.conn, job["id"], status=db.FAILED,
                          flags=list(job["flags"] or []) + [f"email draft failed: {reason}"])
            await self.notify(f"Job #{job['id']} ({job['role']} - {job['company']}): email not drafted. {reason}")

        if not resume.exists():
            await fail(f"resume {resume.name} is missing from {resume.parent}")
            return False
        post = db.get_raw_post(self.conn, job["raw_post_id"]) or {}
        draft = await draft_email(job, contact, also, self.profile, self.cfg, self.llm,
                                  post_text=post.get("text") or "")
        db.update_application(self.conn, app_id, subject=draft.subject, body=draft.text,
                              resume_file=resume.name)
        if not draft.ok:
            await fail(f"it broke the email rules after {draft.attempts} attempts: {'; '.join(draft.errors)}")
            return False
        if draft.subject != build_subject(job["role"], self.profile):
            note = f"subject line taken from the post's instructions: {draft.subject!r}"
            if note not in (job["flags"] or []):
                db.update_job(self.conn, job["id"], flags=list(job["flags"] or []) + [note])
        try:
            await self.sync_gmail_draft(app_id)
        except Exception as e:  # noqa: BLE001 - report and keep going
            log.exception("Gmail draft failed for job %s", job["id"])
            await fail(f"Gmail error: {type(e).__name__}: {e}")
            return False
        db.update_job(self.conn, job["id"], status=db.DRAFTED)
        return True

    async def rescue_job(self, job_id: int, *, rescore: bool) -> str:
        """Bring back a job the filter skipped. Returns its new status.

        rescore=True re-runs the filter (it may skip again); rescore=False is the user's
        "Draft anyway": the job goes through as flagged. Either way, a passing job gets
        its email planned and drafted like a fresh one (cooldown rules still apply).
        """
        job = db.get_job(self.conn, job_id)
        post = db.get_raw_post(self.conn, job["raw_post_id"]) or {}
        flags = list(job["flags"] or [])
        skipped = [f for f in flags if f.startswith("skipped:")]
        if rescore:
            db.update_job(self.conn, job_id, flags=[f for f in flags if not f.startswith(("skipped:", "skill gap:"))])
            await self._decide(job_id, post.get("text") or "")
        else:
            track = job["resume_track"] or pick_resume(job)[0]
            note = "you chose Draft anyway" + (f" (it was {skipped[0]})" if skipped else "")
            db.update_job(self.conn, job_id, status=db.FLAGGED, resume_track=track,
                          flags=[f for f in flags if not f.startswith("skipped:")] + [note])
        job = db.get_job(self.conn, job_id)
        if job["status"] in (db.EXTRACTED, db.FLAGGED) and job["resume_track"]:
            db.delete_planned_applications(self.conn, job_id)
            plan = plan_emails(self.conn, [job_id], job["raw_post_id"], self.cfg.recipient_cooldown_days,
                               {job_id: self.cfg.resume_file(job["resume_track"]).name})
            for app_id in plan.applications:
                await self.draft_application(app_id)
        return db.get_job(self.conn, job_id)["status"]

    async def _decide(self, job_id: int, post_text: str) -> str | None:
        """Filter one job and pick its resume. Returns the resume track (None if skipped)."""
        job = db.get_job(self.conn, job_id)
        contacts = db.contacts_for_job(self.conn, job_id)
        fr = await evaluate(job, contacts, post_text, self.profile, self.cfg, self.llm)
        track = None if fr.decision == SKIP else pick_resume(job)[0]
        flags = list(job["flags"] or [])
        flags += [f for f in fr.all_flags() if f.lower() not in {x.lower() for x in flags}]
        db.update_job(self.conn, job_id, status=fr.status, fit_score=fr.fit_score,
                      fit_reason=fr.fit_reason or fr.skip_reason, flags=flags, resume_track=track)
        return track


def render_card(conn: sqlite3.Connection, job_id: int) -> str:
    """Plain-text job card for the console (the Telegram card comes in Phase 5)."""
    j = db.get_job(conn, job_id)
    decision = {db.EXTRACTED: "MATCH", db.FLAGGED: "FLAGGED", db.SKIPPED_FILTER: "SKIP",
                db.DUPLICATE: "COVERED"}.get(j["status"], j["status"].upper())
    fit = f"Fit {j['fit_score']}" if j["fit_score"] is not None else "Fit -"
    lines = [
        "=" * 64,
        f"#{j['id']}  {j['role']} - {j['company'] or '?'}   [{j['role_category']}]",
        f"  {decision} · {fit} · {j['fit_reason'] or ''}".rstrip(" ·"),
    ]
    if j["resume_track"]:
        lines.append(f"  Resume: {j['resume_track']} - {pick_resume(j)[1]}")
    meta = [
        ("Batch", j["batch_eligibility"]), ("Location", j["location"]), ("Mode", j["work_mode"]),
        ("Pay", j["stipend_or_ctc"]), ("Deadline", j["deadline"]),
    ]
    lines.append("  " + (" | ".join(f"{k}: {v}" for k, v in meta if v) or "(no details)"))
    if j["required_skills"]:
        lines.append(f"  Skills: {', '.join(j['required_skills'])}")
    if j["jd_summary"]:
        lines.append(f"  JD: {j['jd_summary']}")
    contacts = {c["id"]: c for c in db.contacts_for_job(conn, job_id)}
    for c in contacts.values():
        who = f" ({c['person_name']})" if c["person_name"] else ""
        via = f"  <- {c['raw_value']}" if c["raw_value"] and c["raw_value"] != c["value"] else ""
        lines.append(f"  {c['kind']:5}: {c['value']}{who}{via}")
    flags = [f for f in j["flags"] or [] if not f.startswith("skill gap:")]
    gaps = [f.split(":", 1)[1].strip() for f in j["flags"] or [] if f.startswith("skill gap:")]
    if flags or gaps:
        lines.append("  Flags:")
        lines += [f"    - {f}" for f in flags]
        if gaps:
            more = f" (+{len(gaps) - MAX_GAPS_SHOWN} more)" if len(gaps) > MAX_GAPS_SHOWN else ""
            lines.append(f"    - skill gaps: {', '.join(gaps[:MAX_GAPS_SHOWN])}{more}")
    for a in db.applications_for_job(conn, job_id):
        also = f", also covering job(s) {', '.join(f'#{x}' for x in a['also_job_ids'])}" if a["also_job_ids"] else ""
        state = "Gmail draft ready" if a["gmail_draft_id"] else ("drafted" if a["body"] else "planned")
        lines.append(f"  -> email to {contacts[a['contact_id']]['value']} with {a['resume_file']}{also} [{state}]")
        if a["body"]:
            lines.append("  " + "-" * 62)
            lines.append(f"  Subject: {a['subject']}")
            lines += [f"  | {l}" for l in a["body"].splitlines()]
    return "\n".join(lines)


MAX_GAPS_SHOWN = 5


# --- fixtures -----------------------------------------------------------------
ENTITY_MARKER = "---entities---"


def load_fixture(path: Path) -> tuple[str, list[str]]:
    """A fixture is post text, optionally followed by a line '---entities---' and
    one hidden hyperlink URL per line (simulating MessageEntityTextUrl)."""
    raw = path.read_text(encoding="utf-8")
    text, _, ents = raw.partition(f"\n{ENTITY_MARKER}\n")
    return text.strip("\n"), [u.strip() for u in ents.splitlines() if u.strip()]


async def run_fixtures(pipeline: Pipeline, directory: Path) -> list[int]:
    job_ids: list[int] = []
    for i, path in enumerate(sorted(directory.glob("*.txt")), start=1):
        text, entities = load_fixture(path)
        pid, changed = db.upsert_raw_post(
            pipeline.conn, tg_message_id=i, channel=f"fixtures:{directory.name}",
            text=text, entities=entities,
        )
        if pipeline.dry_run:
            pipeline.echo(f"\n>>> {path.name}")
        if changed or db.get_raw_post(pipeline.conn, pid)["processed"] == 0:
            job_ids += await pipeline.process_post(pid)
    return job_ids
