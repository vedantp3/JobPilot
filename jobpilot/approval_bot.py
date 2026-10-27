"""Owner-only Telegram approval bot: cards, buttons, edit flow, commands.

`ApprovalService` holds all the logic and talks to Telegram only through a small
`Messenger` interface, so it can be tested without a real bot. `build_application`
wires it into python-telegram-bot and drops every update that isn't from the owner.

Nothing is sent without a button tap here: ✅ Send enqueues the application in the
`SendQueue`, which itself refuses anything not `awaiting_approval`.
"""
from __future__ import annotations

import asyncio
import csv
import html
import io
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Protocol

from jobpilot import db
from jobpilot.config import RESUME_TRACKS
from jobpilot.dedupe import contact_preference, recipient_conflict
from jobpilot.email_drafter import revise_email
from jobpilot.gmail_client import SendQueue, local_midnight_utc
from jobpilot.link_handler import (
    CHAT_BASED, LABELS, PREFILLABLE, Prefiller, PrefillReport, build_answer_sheet, classify_link, draft_answers,
    sheet_questions,
)
from jobpilot.pipeline import MAX_GAPS_SHOWN, Pipeline
from jobpilot.resume_picker import pick_resume

log = logging.getLogger(__name__)

CARD_KV = "card:{}"          # job_id -> Telegram message id of its card
ANSWERS_KV = "answers:{}"    # job_id -> JSON {question: answer} for the answer sheet
EDIT_KV = "edit_app"         # application id waiting for an edit instruction
OVERRIDE_KV = "override:{}"  # app_id -> "1" when the user chose "Send anyway"
PAUSED_KV = "paused"
MAX_BODY_CHARS = 3000        # keeps cards under Telegram's 4096-char limit


@dataclass(frozen=True)
class Button:
    text: str
    data: str | None = None   # callback data (<= 64 bytes)
    url: str | None = None


Keyboard = list[list[Button]]


class Messenger(Protocol):
    async def send(self, text: str, buttons: Keyboard | None = None) -> int: ...
    async def edit(self, message_id: int, text: str, buttons: Keyboard | None = None) -> None: ...
    async def send_file(self, data: bytes, filename: str, caption: str = "") -> None: ...


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""), quote=False)


def _local_time(iso: str | None) -> str:
    if not iso:
        return ""
    return datetime.fromisoformat(iso).astimezone().strftime("%d %b %H:%M")


@dataclass
class ApprovalService:
    conn: sqlite3.Connection
    pipeline: Pipeline
    queue: SendQueue | None
    messenger: Messenger | None = None
    backfill: Callable[[int], Awaitable[int]] | None = None   # set by main when the listener runs
    by_date: Callable[[list[str]], Awaitable[str]] | None = None   # /date: fetch + queue posts from dates
    prefiller: Prefiller | None = None                         # None when there's no desktop (server)
    channels: list = field(default_factory=list)               # [(channel ref, title)] being watched

    @property
    def cfg(self):
        return self.pipeline.cfg

    @property
    def profile(self):
        return self.pipeline.profile

    # --- notifications -------------------------------------------------------------
    async def notify(self, text: str) -> None:
        await self.messenger.send("⚠️ " + esc(text))

    # --- data helpers ----------------------------------------------------------------
    def _app_for_job(self, job_id: int) -> dict | None:
        apps = [a for a in db.applications_for_job(self.conn, job_id) if a["body"]]
        return apps[-1] if apps else None

    def _contact(self, contact_id: int) -> dict:
        return dict(self.conn.execute("SELECT * FROM contacts WHERE id=?", (contact_id,)).fetchone())

    def _card_id(self, job_id: int) -> int | None:
        v = db.kv_get(self.conn, CARD_KV.format(job_id))
        return int(v) if v else None

    def _kind(self, job: dict) -> str:
        """What sort of card this job gets, from what it has (not its status)."""
        if self._app_for_job(job["id"]):
            return "email"
        flags = job["flags"] or []
        if any(f.startswith("cooldown:") for f in flags):
            return "conflict"
        contacts = db.contacts_for_job(self.conn, job["id"])
        if any(c["kind"] == "link" for c in contacts):
            return "link"
        return "nocontact" if not contacts else "info"

    def needs_card(self, job: dict) -> bool:
        if self._card_id(job["id"]) is not None:
            return False
        st = job["status"]
        kind = self._kind(job)
        if kind == "email":
            return st == db.DRAFTED
        if kind in ("link", "nocontact", "conflict"):
            return st in (db.EXTRACTED, db.FLAGGED)
        return False

    # --- rendering ---------------------------------------------------------------------
    def _header(self, job: dict, icon: str) -> list[str]:
        lines = [f"{icon} <b>{esc(job['role'])} — {esc(job['company'] or '?')}</b>  #{job['id']}"]
        meta = [f"Fit {job['fit_score']}" if job["fit_score"] is not None else "Fit -"]
        if job["resume_track"]:
            meta.append(f"Resume: {job['resume_track']} ({esc(pick_resume(job)[1])})")
        lines.append(" · ".join(meta))
        details = [job.get("location"), job.get("work_mode"), job.get("stipend_or_ctc"),
                   f"deadline {job['deadline']}" if job.get("deadline") else None]
        if any(details):
            lines.append(esc(" · ".join(d for d in details if d)))
        if len(self.channels) > 1 and (source := self._source_channel(job)):
            lines.append(f"📢 From: {esc(source)}")
        return lines

    def _source_channel(self, job: dict) -> str | None:
        row = self.conn.execute("SELECT channel FROM raw_posts WHERE id=?", (job["raw_post_id"],)).fetchone()
        if row is None:
            return None
        return db.kv_get(self.conn, f"channel_title:{row['channel']}") or row["channel"]

    def _flags_line(self, job: dict) -> list[str]:
        flags = [f for f in job["flags"] or [] if not f.startswith(("skill gap:", "cooldown:"))]
        gaps = [f.split(":", 1)[1].strip() for f in job["flags"] or [] if f.startswith("skill gap:")]
        if gaps:
            more = f" +{len(gaps) - MAX_GAPS_SHOWN}" if len(gaps) > MAX_GAPS_SHOWN else ""
            flags.append(f"gaps: {', '.join(gaps[:MAX_GAPS_SHOWN])}{more}")
        return [f"⚠️ Flags: {esc('; '.join(flags))}"] if flags else []

    def _footer(self, job: dict, app: dict | None) -> list[str]:
        st = job["status"]
        if app and app["sent_at"]:
            return [f"✅ <b>Sent</b> {_local_time(app['sent_at'])}"]
        if app and self.queue and self.queue.is_queued(app["id"]):
            lo, hi = self.cfg.send_spacing_seconds
            return [f"⏳ <b>Queued</b>, sends are spaced {lo}-{hi}s apart"]
        return {
            db.SKIPPED_BY_USER: ["❌ <b>Skipped</b>"],
            db.SUBMITTED: ["✔️ <b>Submitted</b>"],
            db.DUPLICATE: ["↪️ <b>Combined into another email</b>"],
            db.FAILED: ["⚠️ <b>Failed</b>, see the flags; use /card to retry later"],
            db.INTERVIEW: ["🎯 <b>Interview</b>"], db.REJECTED: ["<b>Rejected</b>"], db.OFFER: ["🎉 <b>Offer</b>"],
        }.get(st, [])

    def render(self, job_id: int) -> tuple[str, Keyboard]:
        job = db.get_job(self.conn, job_id)
        kind = self._kind(job)
        renderer = {"email": self._render_email, "link": self._render_link, "conflict": self._render_conflict}
        return renderer.get(kind, self._render_plain)(job)

    def _render_email(self, job: dict) -> tuple[str, Keyboard]:
        app = self._app_for_job(job["id"])
        contact = self._contact(app["contact_id"])
        lines = self._header(job, "📧")
        who = f" ({esc(contact['person_name'])})" if contact["person_name"] else ""
        lines.append(f"To: {esc(contact['value'])}{who}")
        if app["also_job_ids"]:
            roles = [db.get_job(self.conn, j)["role"] for j in app["also_job_ids"]]
            lines.append(f"Also covers: {esc(', '.join(roles))}")
        lines += self._flags_line(job)
        body = app["body"] if len(app["body"]) <= MAX_BODY_CHARS else app["body"][:MAX_BODY_CHARS] + "…"
        lines += ["───", f"<b>Subject:</b> {esc(app['subject'])}", "", esc(body)]
        footer = self._footer(job, app)
        if footer:
            lines += [""] + footer
        buttons: Keyboard = []
        queued = self.queue is not None and self.queue.is_queued(app["id"])
        if job["status"] == db.AWAITING_APPROVAL and not app["sent_at"] and not queued:
            buttons = [
                [Button("✅ Send", f"send:{app['id']}"), Button("✏️ Edit", f"edit:{app['id']}")],
                [Button("📄 Resume ▾", f"resume:{app['id']}"), Button("❌ Skip", f"skip:{job['id']}")],
            ]
        return "\n".join(lines), buttons

    def _render_link(self, job: dict) -> tuple[str, Keyboard]:
        links = [c for c in db.contacts_for_job(self.conn, job["id"]) if c["kind"] == "link"]
        types = [classify_link(c["value"]) for c in links]
        lines = self._header(job, "🔗")
        lines.append("Apply via: " + esc(", ".join(dict.fromkeys(LABELS.get(t, t) for t in types))))
        lines += self._flags_line(job)
        if any(t in CHAT_BASED for t in types):
            lines.append("💬 Chat-based apply: message them yourself.")
        for c in links:
            lines.append(f'<a href="{html.escape(c["value"], quote=True)}">{esc(c["value"])}</a>')
        footer = self._footer(job, None)
        if footer:
            lines += [""] + footer
        buttons: Keyboard = []
        if job["status"] == db.AWAITING_APPROVAL:
            first = []
            if self.prefiller is not None and any(t in PREFILLABLE for t in types):
                first.append(Button("🖥 Prefill", f"prefill:{job['id']}"))
            first.append(Button("📋 Answer sheet", f"sheet:{job['id']}"))
            buttons = [first, [Button("🌐 Open link", url=links[0]["value"])],
                       [Button("✔️ Mark submitted", f"sub:{job['id']}"), Button("❌ Skip", f"skip:{job['id']}")]]
        return "\n".join(lines), buttons

    def _render_conflict(self, job: dict) -> tuple[str, Keyboard]:
        lines = self._header(job, "⏸")
        lines += [esc(f.split(":", 1)[1].strip()) for f in job["flags"] or [] if f.startswith("cooldown:")]
        lines += self._flags_line(job)
        footer = self._footer(job, None)
        if footer:
            lines += [""] + footer
        buttons: Keyboard = []
        if job["status"] == db.FLAGGED:
            pending = self._pending_conflict(job) is not None
            row = [Button("🔗 Combine", f"combine:{job['id']}")] if pending else []
            row += [Button("📨 Send anyway", f"anyway:{job['id']}"), Button("❌ Skip", f"skip:{job['id']}")]
            buttons = [row]
        return "\n".join(lines), buttons

    def _render_plain(self, job: dict) -> tuple[str, Keyboard]:
        lines = self._header(job, "❓" if self._kind(job) == "nocontact" else "ℹ️")
        if self._kind(job) == "nocontact":
            lines.append("No email or application link was found in the post.")
        lines += self._flags_line(job)
        lines.append(f"Status: {job['status']}")
        buttons = [[Button("❌ Skip", f"skip:{job['id']}")]] if job["status"] in (
            db.AWAITING_APPROVAL, db.EXTRACTED, db.FLAGGED) else []
        return "\n".join(lines), buttons

    # --- posting cards ----------------------------------------------------------------
    async def post_cards(self, job_ids: list[int]) -> None:
        """Pipeline hook: post a card for every decided job that needs one."""
        for jid in job_ids:
            try:
                job = db.get_job(self.conn, jid)
                if job and self.needs_card(job):
                    await self.post_card(jid)
            except Exception as e:  # noqa: BLE001 - one bad card must not stop the rest
                log.exception("posting card for job %s failed", jid)
                await self.notify(f"Couldn't post the card for job #{jid}: {e}")

    async def post_pending_cards(self) -> int:
        """Post cards for jobs that were decided while the bot wasn't running."""
        rows = self.conn.execute(
            "SELECT id FROM jobs WHERE status IN (?, ?, ?) ORDER BY id",
            (db.DRAFTED, db.EXTRACTED, db.FLAGGED),
        ).fetchall()
        todo = [r["id"] for r in rows if self.needs_card(db.get_job(self.conn, r["id"]))]
        await self.post_cards(todo)
        return len(todo)

    async def post_card(self, job_id: int) -> int:
        job = db.get_job(self.conn, job_id)
        kind = self._kind(job)
        if kind in ("email", "link", "nocontact") and job["status"] in (db.DRAFTED, db.EXTRACTED, db.FLAGGED):
            db.update_job(self.conn, job_id, status=db.AWAITING_APPROVAL)
        text, buttons = self.render(job_id)
        message_id = await self.messenger.send(text, buttons)
        db.kv_set(self.conn, CARD_KV.format(job_id), str(message_id))
        if kind == "email":
            db.update_application(self.conn, self._app_for_job(job_id)["id"], approval_msg_id=message_id)
        return message_id

    async def refresh_card(self, job_id: int, *, buttons: Keyboard | None = None) -> None:
        text, default_buttons = self.render(job_id)
        mid = self._card_id(job_id)
        if mid is None:
            await self.post_card(job_id)
            return
        await self.messenger.edit(mid, text, default_buttons if buttons is None else buttons)

    async def retire_card(self, job_id: int, note: str) -> None:
        if (mid := self._card_id(job_id)) is not None:
            text, _ = self.render(job_id)
            await self.messenger.edit(mid, f"{text}\n\n{note}", [])

    # --- send queue hooks ----------------------------------------------------------------
    async def on_sent(self, app_id: int, message_id: str) -> None:
        await self.refresh_card(db.get_application(self.conn, app_id)["job_id"])

    async def on_refused(self, app_id: int, reason: str) -> None:
        app = db.get_application(self.conn, app_id)
        await self.refresh_card(app["job_id"])
        if "already emailed" in reason:
            await self.messenger.send(
                f"⏸ Not sent: {esc(reason)}",
                [[Button("📨 Send anyway", f"force:{app_id}"), Button("❌ Skip", f"skip:{app['job_id']}")]],
            )
        else:
            await self.notify(f"Not sent (job #{app['job_id']}): {reason}")

    # --- callbacks ------------------------------------------------------------------------
    async def handle_callback(self, data: str) -> str:
        """Dispatch a button press. Returns a short toast for the user."""
        action, _, rest = data.partition(":")
        handler = getattr(self, f"_cb_{action}", None)
        if handler is None:
            return "Unknown button"
        try:
            return await handler(*rest.split(":")) or ""
        except Exception as e:  # noqa: BLE001 - report instead of crashing the bot
            log.exception("button %s failed", data)
            await self.notify(f"That button failed: {type(e).__name__}: {e}")
            return "Error"

    async def _enqueue(self, app_id: int, override: bool) -> str:
        app = db.get_application(self.conn, app_id)
        job = db.get_job(self.conn, app["job_id"])
        if self.queue is None:
            return "Sending isn't available in this mode"
        if app["sent_at"]:
            return "Already sent"
        if job["status"] != db.AWAITING_APPROVAL:
            return f"Can't send: job is {job['status']}"
        override = override or db.kv_get(self.conn, OVERRIDE_KV.format(app_id)) == "1"
        position = await self.queue.enqueue(app_id, override_cooldown=override)
        if position is None:
            return "Already queued"
        await self.refresh_card(job["id"])
        return f"Queued ({position} in line)"

    async def _cb_send(self, app_id: str) -> str:
        return await self._enqueue(int(app_id), override=False)

    async def _cb_force(self, app_id: str) -> str:
        db.kv_set(self.conn, OVERRIDE_KV.format(app_id), "1")
        return await self._enqueue(int(app_id), override=True)

    async def _cb_edit(self, app_id: str) -> str:
        app = db.get_application(self.conn, int(app_id))
        job = db.get_job(self.conn, app["job_id"])
        if job["status"] != db.AWAITING_APPROVAL or app["sent_at"]:
            return f"Can't edit: job is {job['status']}"
        db.kv_set(self.conn, EDIT_KV, app_id)
        await self.messenger.send(
            f"✏️ Editing the email for <b>{esc(job['role'])}</b> (#{job['id']}).\n"
            "Reply with what to change, e.g. <i>shorter</i> or <i>mention StockPilot instead</i>, "
            "or paste a full new body. /cancel to stop."
        )
        return "Send your changes"

    async def _cb_resume(self, app_id: str) -> str:
        app = db.get_application(self.conn, int(app_id))
        job = db.get_job(self.conn, app["job_id"])
        current = job["resume_track"]
        rows = [[Button(("✓ " if t == current else "") + t, f"rset:{app_id}:{t}")] for t in RESUME_TRACKS]
        rows.append([Button("← Back", f"back:{job['id']}")])
        await self.refresh_card(job["id"], buttons=rows)
        return "Pick a resume"

    async def _cb_rset(self, app_id: str, track: str) -> str:
        if track not in RESUME_TRACKS:
            return "Unknown resume"
        app = db.get_application(self.conn, int(app_id))
        resume = self.cfg.resume_file(track)
        if not resume.exists():
            return f"{resume.name} is missing"
        db.update_job(self.conn, app["job_id"], resume_track=track)
        db.update_application(self.conn, app["id"], resume_file=resume.name)
        try:
            await self.pipeline.sync_gmail_draft(app["id"])
        except Exception as e:  # noqa: BLE001
            await self.notify(f"Resume changed, but updating the Gmail draft failed: {e}")
        await self.refresh_card(app["job_id"])
        return f"Resume: {track}"

    async def _cb_back(self, job_id: str) -> str:
        await self.refresh_card(int(job_id))
        return ""

    async def _cb_skip(self, job_id: str) -> str:
        job = db.get_job(self.conn, int(job_id))
        if job["status"] in (db.SENT, db.SUBMITTED, db.INTERVIEW, db.REJECTED, db.OFFER):
            return f"Already {job['status']}"
        db.set_status(self.conn, job["id"], db.SKIPPED_BY_USER, force=True)
        await self.refresh_card(job["id"])
        return "Skipped"

    async def _cb_sub(self, job_id: str) -> str:
        job = db.get_job(self.conn, int(job_id))
        if job["status"] != db.AWAITING_APPROVAL:
            return f"Can't: job is {job['status']}"
        db.set_status(self.conn, job["id"], db.SUBMITTED)
        await self.refresh_card(job["id"])
        return "Marked submitted"

    # --- link jobs: answer sheet + prefill ------------------------------------------------------
    def _links(self, job_id: int) -> list[str]:
        return [c["value"] for c in db.contacts_for_job(self.conn, job_id) if c["kind"] == "link"]

    def _resume_for(self, job: dict):
        track = job["resume_track"] or pick_resume(job)[0]
        return self.cfg.resume_file(track)

    async def _sheet_answers(self, job: dict) -> dict[str, str]:
        cached = db.kv_get(self.conn, ANSWERS_KV.format(job["id"]))
        if cached:
            return json.loads(cached)
        answers = await draft_answers(sheet_questions(job), job, self.profile, self.cfg, self.pipeline.llm)
        db.kv_set(self.conn, ANSWERS_KV.format(job["id"]), json.dumps(answers, ensure_ascii=False))
        return answers

    async def send_answer_sheet(self, job_id: int) -> None:
        job = db.get_job(self.conn, job_id)
        answers = await self._sheet_answers(job)
        resume = self._resume_for(job)
        await self.messenger.send(
            build_answer_sheet(job, self._links(job_id), answers, self.profile, resume.name),
            [[Button("✔️ Mark submitted", f"sub:{job_id}")]],
        )

    async def _cb_sheet(self, job_id: str) -> str:
        if db.get_job(self.conn, int(job_id)) is None:
            return "No such job"
        await self.send_answer_sheet(int(job_id))
        return "Answer sheet below"

    async def _cb_prefill(self, job_id: str) -> str:
        job = db.get_job(self.conn, int(job_id))
        if job is None:
            return "No such job"
        if job["status"] != db.AWAITING_APPROVAL:
            return f"Can't: job is {job['status']}"
        links = [u for u in self._links(job["id"]) if classify_link(u) in PREFILLABLE]
        if self.prefiller is None or not links:
            await self.send_answer_sheet(job["id"])
            return "Prefill isn't available here; answer sheet below"
        report = await self.prefiller.prefill(links[0], classify_link(links[0]), job, self.profile, self.cfg,
                                              self.pipeline.llm, self._resume_for(job))
        await self.messenger.send(self.render_prefill_report(job, report),
                                  [[Button("✔️ Mark submitted", f"sub:{job['id']}")]])
        if not report.ok and not report.needs_login:
            await self.send_answer_sheet(job["id"])
        return "Prefilled: check the browser" if report.ok else "Prefill didn't work (see message)"

    @staticmethod
    def render_prefill_report(job: dict, r: PrefillReport) -> str:
        head = f"🖥 <b>{esc(job['role'])} — {esc(job['company'] or '?')}</b>  #{job['id']}"
        if not r.ok:
            tail = "" if r.needs_login else "\nHere's the answer sheet instead."
            return f"{head}\nCouldn't prefill: {esc(r.error)}{tail}"
        lines = [head, "Opened and prefilled in the browser window. <b>Nothing was submitted.</b>"]
        if r.filled:
            lines.append(f"Filled from your profile: {esc(', '.join(r.filled))}")
        if r.uploaded:
            lines.append(f"Resume uploaded: {esc(', '.join(r.uploaded))}")
        if r.drafted:
            lines.append("Drafted answers (highlighted yellow, review them):")
            lines += [f"• <b>{esc(q)}</b>\n  {esc(a)}" for q, a in r.drafted.items()]
        if r.left_blank:
            lines.append(f"Left for you: {esc(', '.join(r.left_blank))}")
        lines.append("Review it, submit it on the site yourself, then tap ✔️ Mark submitted.")
        return "\n".join(lines)

    def _pending_conflict(self, job: dict) -> tuple[str, dict] | None:
        """(email, open application) this job's recipient is already waiting on, if any."""
        emails = sorted((c for c in db.contacts_for_job(self.conn, job["id"]) if c["kind"] == "email"),
                        key=contact_preference)
        for c in emails:
            conflict = recipient_conflict(self.conn, c["value"], self.cfg.recipient_cooldown_days,
                                          exclude_raw_post_id=job["raw_post_id"])
            if conflict and conflict.kind == "pending":
                apps = db.open_applications_to(self.conn, c["value"], job["raw_post_id"])
                if apps:
                    return c["value"], apps[0]
        return None

    async def _cb_combine(self, job_id: str) -> str:
        job = db.get_job(self.conn, int(job_id))
        if job["status"] != db.FLAGGED:
            return f"Can't: job is {job['status']}"
        found = self._pending_conflict(job)
        if found is None:
            return "Nothing pending to combine with; use Send anyway or Skip"
        email, other_app = found
        if self.queue is not None and self.queue.is_queued(other_app["id"]):
            return "That email is already queued to send; use Send anyway or Skip"
        other_job_id = other_app["job_id"]
        await self.retire_card(other_job_id, "↪️ Replaced: this email now also covers "
                                             f"<b>{esc(job['role'])}</b>. See the updated card below.")
        db.update_application(self.conn, other_app["id"],
                              also_job_ids=list(other_app["also_job_ids"] or []) + [job["id"]])
        if not await self.pipeline.draft_application(other_app["id"]):
            return "Couldn't redraft the combined email (see message)"
        db.update_job(self.conn, job["id"], status=db.DUPLICATE, flags=list(job["flags"] or []) + [
            f"combined into the email to {email} for job #{other_job_id}"])
        await self.post_card(other_job_id)
        await self.refresh_card(job["id"])
        return "Combined"

    async def _cb_anyway(self, job_id: str) -> str:
        job = db.get_job(self.conn, int(job_id))
        if job["status"] != db.FLAGGED:
            return f"Can't: job is {job['status']}"
        emails = sorted((c for c in db.contacts_for_job(self.conn, job["id"]) if c["kind"] == "email"),
                        key=contact_preference)
        if not emails:
            return "No email address for this job"
        track = job["resume_track"] or pick_resume(job)[0]
        db.update_job(self.conn, job["id"], resume_track=track)
        app_id = db.insert_application(self.conn, job_id=job["id"], contact_id=emails[0]["id"],
                                       resume_file=self.cfg.resume_file(track).name)
        db.kv_set(self.conn, OVERRIDE_KV.format(app_id), "1")
        await self.retire_card(job["id"], "↪️ Drafting the email below…")
        if not await self.pipeline.draft_application(app_id):
            return "Draft failed (see message)"
        await self.post_card(job["id"])
        return "Drafted"

    # --- edit flow ------------------------------------------------------------------------
    async def handle_text(self, text: str) -> None:
        app_id = db.kv_get(self.conn, EDIT_KV)
        if not app_id:
            await self.messenger.send("Use the buttons on a card, or /help for commands.")
            return
        db.kv_set(self.conn, EDIT_KV, "")
        app = db.get_application(self.conn, int(app_id))
        job = db.get_job(self.conn, app["job_id"])
        if job["status"] != db.AWAITING_APPROVAL or app["sent_at"]:
            await self.messenger.send(f"That email can't be edited any more (job is {job['status']}).")
            return
        contact = self._contact(app["contact_id"])
        also = [db.get_job(self.conn, j) for j in app["also_job_ids"] or []]
        await self.messenger.send("Revising…")
        draft = await revise_email(job, contact, also, self.profile, self.cfg, self.pipeline.llm,
                                   app["body"], app["subject"], text)
        if not draft.ok:
            await self.messenger.send(
                "Couldn't apply that edit without breaking the email rules:\n• "
                + "\n• ".join(esc(e) for e in draft.errors) + "\nThe current draft is unchanged.",
                [[Button("✏️ Try again", f"edit:{app_id}")]],
            )
            return
        await self.retire_card(job["id"], "✏️ Replaced by the edited version below.")
        db.update_application(self.conn, app["id"], body=draft.text)
        try:
            await self.pipeline.sync_gmail_draft(app["id"])
        except Exception as e:  # noqa: BLE001
            await self.notify(f"Edited, but updating the Gmail draft failed: {e}")
        await self.post_card(job["id"])

    async def cmd_cancel(self, args: list[str] | None = None) -> None:
        db.kv_set(self.conn, EDIT_KV, "")
        await self.messenger.send("Cancelled.")

    # --- commands ---------------------------------------------------------------------------
    async def cmd_help(self, args: list[str] | None = None) -> None:
        await self.messenger.send(
            "<b>JobPilot</b>: matched jobs arrive here as cards. Nothing is sent until you tap ✅ Send.\n\n"
            "/pending: jobs waiting for you\n/card N: show job N's card again\n"
            "/skipped [N]: recent skipped jobs, with a Draft-anyway button\n"
            "/rescore [N]: re-run the fit check on the last N low-fit skips\n"
            "/stats: counts today, last 7 days, all time\n/status N interview|rejected|offer\n"
            "/export: tracker as CSV\n/pause, /resume: stop/restart processing new posts\n"
            "/channels: the channels being watched, with counts\n"
            "/backfill N: process the last N posts of each channel\n"
            "/date DAY [TO]: process posts from a day or range, e.g. /date 20-10 23-10\n"
            "/backlog: how many old posts are waiting; drop them\n"
            "/login: sign in to Google for Prefill (forms that need an account)\n/cancel: cancel an edit"
        )

    async def cmd_pending(self, args: list[str] | None = None) -> None:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE status IN (?, ?, ?) ORDER BY id",
            (db.AWAITING_APPROVAL, db.FLAGGED, db.FAILED),
        ).fetchall()
        if not rows:
            await self.messenger.send("Nothing pending. 🎉")
            return
        lines = [f"<b>{len(rows)} pending</b>"]
        for r in rows:
            job = db.get_job(self.conn, r["id"])
            kind = self._kind(job)
            icon = {"email": "📧", "link": "🔗", "conflict": "⏸"}.get(kind, "❓")
            lines.append(f"{icon} #{job['id']} {esc(job['role'])} — {esc(job['company'] or '?')} "
                         f"<i>({job['status']})</i>")
        lines.append("\nUse /card N to see one.")
        await self.messenger.send("\n".join(lines))

    @staticmethod
    def _skip_reason(job: dict) -> str:
        if job["fit_score"] is None:
            return job["fit_reason"] or "skipped"
        reason = (job["fit_reason"] or "").strip()
        return f"fit {job['fit_score']}" + (f": {reason[:90]}" if reason else "")

    async def cmd_skipped(self, args: list[str]) -> None:
        n = max(1, min(int(args[0]) if args and args[0].isdigit() else 15, 40))
        rows = self.conn.execute("SELECT id FROM jobs WHERE status=? ORDER BY id DESC LIMIT ?",
                                 (db.SKIPPED_FILTER, n)).fetchall()
        if not rows:
            await self.messenger.send("No skipped jobs.")
            return
        lines = [f"<b>Last {len(rows)} skipped</b> (tap a button to draft it anyway)"]
        buttons: Keyboard = []
        for r in rows:
            job = db.get_job(self.conn, r["id"])
            lines.append(f"#{job['id']} {esc(job['role'])} — {esc(job['company'] or '?')}\n"
                         f"    <i>{esc(self._skip_reason(job))}</i>")
            buttons.append([Button(f"✏️ Draft #{job['id']} anyway", f"rescue:{job['id']}")])
        lines.append("\n/rescore N re-runs the fit check on the last N low-fit skips.")
        await self.messenger.send("\n".join(lines), buttons)

    async def _cb_rescue(self, job_id: str) -> str:
        job = db.get_job(self.conn, int(job_id))
        if job is None:
            return "No such job"
        if job["status"] != db.SKIPPED_FILTER:
            return f"Already {job['status']}"
        status = await self.pipeline.rescue_job(job["id"], rescore=False)
        job = db.get_job(self.conn, job["id"])
        if self.needs_card(job):
            await self.post_card(job["id"])
            return "Card posted below"
        return "Draft failed (see message)" if status == db.FAILED else f"Now {status}"

    async def cmd_rescore(self, args: list[str]) -> None:
        n = max(1, min(int(args[0]) if args and args[0].isdigit() else 10, 50))
        rows = self.conn.execute(
            "SELECT id FROM jobs WHERE status=? AND fit_score IS NOT NULL ORDER BY id DESC LIMIT ?",
            (db.SKIPPED_FILTER, n),
        ).fetchall()
        if not rows:
            await self.messenger.send("No low-fit skips to re-score. (Batch, scam and keyword skips aren't "
                                      "re-scored; use /skipped to draft one anyway.)")
            return
        await self.messenger.send(f"Re-scoring {len(rows)} job(s)…")
        lines, passed = [], 0
        for r in rows:
            before = db.get_job(self.conn, r["id"])
            status = await self.pipeline.rescue_job(r["id"], rescore=True)
            job = db.get_job(self.conn, r["id"])
            note = ""
            if status != db.SKIPPED_FILTER:
                passed += 1
                if self.needs_card(job):
                    await self.post_card(job["id"])
                    note = ", card posted"
            lines.append(f"#{job['id']} {esc(job['role'])}: {before['fit_score']} → {job['fit_score']} "
                         f"<i>({status}{note})</i>")
        await self.messenger.send(f"<b>Re-scored {len(rows)}; {passed} now pass</b>\n" + "\n".join(lines))

    async def cmd_card(self, args: list[str]) -> None:
        if not args or not args[0].lstrip("#").isdigit():
            await self.messenger.send("Usage: /card N")
            return
        job_id = int(args[0].lstrip("#"))
        if db.get_job(self.conn, job_id) is None:
            await self.messenger.send(f"No job #{job_id}.")
            return
        await self.post_card(job_id)

    async def cmd_stats(self, args: list[str] | None = None) -> None:
        today = local_midnight_utc()
        periods = {"today": today, "7d": today - timedelta(days=6), "all": None}
        counts = {p: db.status_counts(self.conn, since.isoformat(timespec="seconds") if since else None)
                  for p, since in periods.items()}
        statuses = [s for s in db.ALL_STATUSES if any(c.get(s) for c in counts.values())]
        rows = [f"{'status':<18}{'today':>6}{'7d':>5}{'all':>6}"]
        rows += [f"{s:<18}{counts['today'].get(s, 0):>6}{counts['7d'].get(s, 0):>5}{counts['all'].get(s, 0):>6}"
                 for s in statuses]
        sent_today = self.queue.sent_today() if self.queue else db.emails_sent_since(
            self.conn, today.isoformat(timespec="seconds"))
        waiting, _ = db.backlog_stats(self.conn)
        await self.messenger.send(
            "<pre>" + esc("\n".join(rows)) + "</pre>\n"
            f"Emails sent today: {sent_today}/{self.cfg.max_emails_per_day} · posts waiting: {waiting}"
            + (" · ⏸ processing paused" if db.kv_get(self.conn, PAUSED_KV) == "1" else "")
        )

    async def cmd_status(self, args: list[str]) -> None:
        if len(args) != 2 or not args[0].lstrip("#").isdigit() or args[1].lower() not in db.MANUAL_STATUSES:
            await self.messenger.send("Usage: /status N interview|rejected|offer")
            return
        job_id, status = int(args[0].lstrip("#")), args[1].lower()
        job = db.get_job(self.conn, job_id)
        if job is None:
            await self.messenger.send(f"No job #{job_id}.")
            return
        try:
            db.set_status(self.conn, job_id, status)
        except db.InvalidTransition as e:
            await self.messenger.send(f"Can't: {esc(e)}")
            return
        await self.messenger.send(f"#{job_id} {esc(job['role'])} → <b>{status}</b>")
        if self._card_id(job_id) is not None:
            await self.refresh_card(job_id)

    async def cmd_export(self, args: list[str] | None = None) -> None:
        await self.messenger.send_file(export_csv(self.conn), "jobpilot.csv", "JobPilot tracker")

    async def cmd_channels(self, args: list[str] | None = None) -> None:
        if not self.channels:
            await self.messenger.send("No channels are being watched right now (the listener isn't running).")
            return
        lines = [f"<b>Watching {len(self.channels)} channel(s)</b>"]
        for ref, title in self.channels:
            posts = self.conn.execute("SELECT COUNT(*) FROM raw_posts WHERE channel=?", (ref,)).fetchone()[0]
            jobs = self.conn.execute(
                "SELECT COUNT(*) AS n, SUM(j.status IN (?, ?, ?)) AS applied FROM jobs j "
                "JOIN raw_posts r ON r.id = j.raw_post_id WHERE r.channel=?",
                (db.SENT, db.SUBMITTED, db.INTERVIEW, ref),
            ).fetchone()
            lines.append(f"📢 <b>{esc(title)}</b> <code>{esc(ref)}</code>\n"
                         f"    {posts} posts · {jobs['n']} jobs · {jobs['applied'] or 0} applied")
        lines.append("\nAdd or remove channels in TG_CHANNEL in .env (comma-separated), then restart.")
        await self.messenger.send("\n".join(lines))

    async def cmd_date(self, args: list[str]) -> None:
        if self.by_date is None:
            await self.messenger.send("/date only works while the listener is running.")
            return
        if not args:
            await self.messenger.send(
                "Usage: /date DAY or /date FROM TO\n"
                "e.g. <code>/date today</code>, <code>/date 23-10</code>, <code>/date 20-10 23-10</code>, "
                "<code>/date 21 Oct</code>")
            return
        try:
            summary = await self.by_date(args)
        except ValueError as e:
            await self.messenger.send(f"⚠️ {esc(e)}")
            return
        await self.messenger.send(f"📅 {esc(summary)} Cards arrive as each one is processed.")

    async def cmd_login(self, args: list[str] | None = None) -> None:
        if self.prefiller is None:
            await self.messenger.send("Prefill isn't enabled on this machine, so there's nothing to sign in to.")
            return
        try:
            browser = await self.prefiller.open_for_login()
        except Exception as e:  # noqa: BLE001
            await self.notify(f"Couldn't open the sign-in window: {e}")
            return
        name = {"chrome": "Chrome", "msedge": "Edge"}.get(browser, browser)
        await self.messenger.send(
            f"🔑 Opened a normal {name} window with JobPilot's browser profile.\n"
            "1. Sign in to your Google account there (and any job sites you use).\n"
            "2. <b>Close that window completely.</b>\n"
            "3. Tap 🖥 Prefill again: the form window will now be signed in.\n"
            "Your everyday browser profile isn't touched."
        )

    async def cmd_backlog(self, args: list[str] | None = None) -> None:
        n, oldest = db.backlog_stats(self.conn)
        if n == 0:
            await self.messenger.send("No backlog: every saved post has been processed.")
            return
        await self.messenger.send(
            f"<b>{n} saved post(s) waiting</b> (oldest from {_local_time(oldest)}).\n"
            "New channel posts are always processed before these. Drop the old ones?",
            [[Button("🗑 Drop posts older than 24h", "drop:24")],
             [Button(f"🗑 Drop all {n} waiting", "drop:all")],
             [Button("Keep processing them", "drop:keep")]],
        )

    async def _cb_drop(self, which: str) -> str:
        if which == "keep":
            return "Keeping the backlog"
        cutoff = None
        if which != "all":
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=int(which))).isoformat(timespec="seconds")
        dropped = db.drop_backlog(self.conn, cutoff)
        left, _ = db.backlog_stats(self.conn)
        await self.messenger.send(f"🗑 Dropped {dropped} old post(s); {left} still waiting. "
                                  "Use /backfill N if you want some of them back.")
        return f"Dropped {dropped}"

    async def cmd_pause(self, args: list[str] | None = None) -> None:
        db.kv_set(self.conn, PAUSED_KV, "1")
        await self.messenger.send("⏸ Paused. New posts are saved but not processed until /resume.")

    async def cmd_resume(self, args: list[str] | None = None) -> None:
        db.kv_set(self.conn, PAUSED_KV, "0")
        waiting = len(db.unprocessed_posts(self.conn))
        await self.messenger.send(f"▶️ Resumed. {waiting} saved post(s) will be processed.")

    async def cmd_backfill(self, args: list[str]) -> None:
        if self.backfill is None:
            await self.messenger.send("Backfill only works while the listener is running.")
            return
        n = int(args[0]) if args and args[0].isdigit() else 10
        n = max(1, min(n, 100))
        queued = await self.backfill(n)
        await self.messenger.send(f"Queued {queued} post(s) from the last {n}.")


def export_csv(conn: sqlite3.Connection) -> bytes:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["job_id", "created_at", "company", "role", "category", "status", "fit_score", "resume_track",
                "apply_method", "emails", "links", "emailed_to", "subject", "sent_at", "flags"])
    for job in db.jobs_by_status(conn, *db.ALL_STATUSES):
        contacts = db.contacts_for_job(conn, job["id"])
        apps = [a for a in db.applications_for_job(conn, job["id"]) if a["body"]]
        app = apps[-1] if apps else None
        to = conn.execute("SELECT value FROM contacts WHERE id=?", (app["contact_id"],)).fetchone()[0] if app else ""
        w.writerow([
            job["id"], job["created_at"], job["company"], job["role"], job["role_category"], job["status"],
            job["fit_score"], job["resume_track"], job["apply_method"],
            " ".join(c["value"] for c in contacts if c["kind"] == "email"),
            " ".join(c["value"] for c in contacts if c["kind"] == "link"),
            to, app["subject"] if app else "", app["sent_at"] if app else "", "; ".join(job["flags"] or []),
        ])
    return out.getvalue().encode("utf-8-sig")   # BOM so Excel opens UTF-8 correctly


# --- python-telegram-bot wiring ----------------------------------------------------------------
class TelegramMessenger:
    """Messenger over a python-telegram-bot `Bot`, always to the owner's chat, HTML formatted."""

    def __init__(self, bot, chat_id: int):
        self.bot, self.chat_id = bot, chat_id

    @staticmethod
    def _markup(buttons: Keyboard | None):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        if not buttons:
            return None
        return InlineKeyboardMarkup([
            [InlineKeyboardButton(b.text, callback_data=b.data) if b.data else InlineKeyboardButton(b.text, url=b.url)
             for b in row] for row in buttons
        ])

    async def send(self, text: str, buttons: Keyboard | None = None) -> int:
        from telegram import LinkPreviewOptions
        from telegram.constants import ParseMode

        msg = await self.bot.send_message(
            self.chat_id, text[:4096], parse_mode=ParseMode.HTML, reply_markup=self._markup(buttons),
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
        return msg.message_id

    async def edit(self, message_id: int, text: str, buttons: Keyboard | None = None) -> None:
        from telegram import LinkPreviewOptions
        from telegram.constants import ParseMode
        from telegram.error import BadRequest

        try:
            await self.bot.edit_message_text(
                text[:4096], chat_id=self.chat_id, message_id=message_id, parse_mode=ParseMode.HTML,
                reply_markup=self._markup(buttons), link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                raise

    async def send_file(self, data: bytes, filename: str, caption: str = "") -> None:
        from telegram import InputFile

        await self.bot.send_document(self.chat_id, document=InputFile(io.BytesIO(data), filename=filename),
                                     caption=caption)


SLOW_CALLBACK_SECONDS = 8.0   # answer the button press early if the action takes longer (LLM redraft)


def build_application(token: str, owner_chat_id: int, service: ApprovalService):
    """python-telegram-bot Application that only ever responds to the owner's chat."""
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

    app = Application.builder().token(token).build()
    owner = filters.Chat(chat_id=owner_chat_id)

    def command(fn):
        async def handler(update, context):
            await fn(list(context.args or []))
        return handler

    for name, fn in {
        "start": service.cmd_help, "help": service.cmd_help, "pending": service.cmd_pending,
        "card": service.cmd_card, "stats": service.cmd_stats, "status": service.cmd_status,
        "export": service.cmd_export, "pause": service.cmd_pause, "resume": service.cmd_resume,
        "backfill": service.cmd_backfill, "cancel": service.cmd_cancel,
        "skipped": service.cmd_skipped, "rescore": service.cmd_rescore, "backlog": service.cmd_backlog,
        "login": service.cmd_login, "date": service.cmd_date, "channels": service.cmd_channels,
    }.items():
        app.add_handler(CommandHandler(name, command(fn), filters=owner))

    async def on_callback(update, context):
        query = update.callback_query
        if update.effective_chat is None or update.effective_chat.id != owner_chat_id:
            await query.answer()
            return
        task = asyncio.create_task(service.handle_callback(query.data or ""))
        try:
            toast = await asyncio.wait_for(asyncio.shield(task), SLOW_CALLBACK_SECONDS)
        except asyncio.TimeoutError:
            await query.answer("Working on it…")
            toast = await task
            if toast:
                await service.messenger.send(esc(toast))
            return
        await query.answer(toast[:190] if toast else None)

    async def on_text(update, context):
        await service.handle_text(update.effective_message.text or "")

    async def on_error(update, context):
        log.error("bot error", exc_info=context.error)
        try:
            await service.notify(f"Bot error: {type(context.error).__name__}: {context.error}")
        except Exception:  # noqa: BLE001 - never loop on errors
            pass

    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(owner & filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)
    return app
