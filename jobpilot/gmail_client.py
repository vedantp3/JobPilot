"""Gmail OAuth, drafts, and the rate-limited send queue.

Sending only happens through `SendQueue.send_one`, which refuses any job that isn't
`awaiting_approval` (i.e. hasn't been approved from the owner's Telegram card), any
recipient inside the cooldown window, and anything past the daily cap.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Awaitable, Callable

from jobpilot import db

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]


class GmailAuthError(RuntimeError):
    pass


def build_message(to: str, subject: str, body: str, attachment: Path | None = None,
                  attachment_filename: str | None = None) -> EmailMessage:
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    if attachment is not None:
        msg.add_attachment(Path(attachment).read_bytes(), maintype="application", subtype="pdf",
                           filename=attachment_filename or Path(attachment).name)
    return msg


def _raw(msg: EmailMessage) -> str:
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")


class GmailClient:
    """Thin async wrapper over the Gmail API (the google client is sync; calls run in a thread)."""

    def __init__(self, credentials_path: Path, token_path: Path, interactive: bool = True):
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.interactive = interactive
        self._service = None

    # -- auth --
    def _credentials(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self.token_path.write_text(creds.to_json(), encoding="utf-8")
                return creds
            except Exception as e:  # noqa: BLE001 - e.g. revoked / 7-day testing-mode expiry
                log.warning("Gmail token refresh failed (%s); re-authorisation needed", e)
        if not self.interactive:
            raise GmailAuthError("Gmail token missing or expired; run: python -m jobpilot.main --gmail-auth")
        if not self.credentials_path.exists():
            raise GmailAuthError(f"{self.credentials_path} not found; see README: Gmail OAuth")
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
        creds = flow.run_local_server(port=0, prompt="consent")
        self.token_path.write_text(creds.to_json(), encoding="utf-8")
        return creds

    @property
    def service(self):
        if self._service is None:
            from googleapiclient.discovery import build

            self._service = build("gmail", "v1", credentials=self._credentials(), cache_discovery=False)
        return self._service

    async def _run(self, fn):
        return await asyncio.to_thread(lambda: fn().execute())

    async def account_email(self) -> str:
        prof = await self._run(lambda: self.service.users().getProfile(userId="me"))
        return prof["emailAddress"]

    # -- drafts --
    async def create_draft(self, to: str, subject: str, body: str, attachment: Path | None = None,
                           attachment_filename: str | None = None) -> str:
        raw = _raw(build_message(to, subject, body, attachment, attachment_filename))
        draft = await self._run(lambda: self.service.users().drafts().create(
            userId="me", body={"message": {"raw": raw}}))
        return draft["id"]

    async def update_draft(self, draft_id: str, to: str, subject: str, body: str,
                           attachment: Path | None = None, attachment_filename: str | None = None) -> str:
        raw = _raw(build_message(to, subject, body, attachment, attachment_filename))
        draft = await self._run(lambda: self.service.users().drafts().update(
            userId="me", id=draft_id, body={"id": draft_id, "message": {"raw": raw}}))
        return draft["id"]

    async def send_draft(self, draft_id: str) -> str:
        sent = await self._run(lambda: self.service.users().drafts().send(userId="me", body={"id": draft_id}))
        return sent["id"]


# --- send queue -------------------------------------------------------------------
Notify = Callable[[str], Awaitable[None]]


async def _log_notify(text: str) -> None:
    log.warning("NOTIFY: %s", text)


def local_midnight_utc(now: datetime | None = None) -> datetime:
    """Start of the current local day, as an aware UTC datetime."""
    local = (now or datetime.now(timezone.utc)).astimezone()
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


class SendRefused(RuntimeError):
    pass


QUEUE_KV = "send_queue"   # persisted queue: JSON [[app_id, override], ...]


@dataclass
class SendQueue:
    conn: sqlite3.Connection
    gmail: GmailClient
    max_per_day: int = 25
    spacing: tuple[int, int] = (90, 240)
    cooldown_days: int = 7
    notify: Notify = _log_notify
    on_sent: Callable[[int, str], Awaitable[None]] | None = None        # (app_id, message_id)
    on_refused: Callable[[int, str], Awaitable[None]] | None = None     # (app_id, reason)
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue, init=False)
    _queued: dict = field(default_factory=dict, init=False)     # app_id -> override
    _sent_once: bool = field(default=False, init=False)
    _cap_notified: bool = field(default=False, init=False)

    def sent_today(self) -> int:
        return db.emails_sent_since(self.conn, local_midnight_utc(self.clock()).isoformat(timespec="seconds"))

    def _persist(self) -> None:
        db.kv_set(self.conn, QUEUE_KV, json.dumps([[a, o] for a, o in self._queued.items()]))

    def is_queued(self, app_id: int) -> bool:
        return app_id in self._queued

    async def enqueue(self, app_id: int, *, override_cooldown: bool = False) -> int | None:
        """Queue an approved application. Returns its position, or None if it's already queued."""
        if app_id in self._queued:
            return None
        self._queued[app_id] = override_cooldown
        self._persist()
        await self._queue.put((app_id, override_cooldown))
        return len(self._queued)

    async def restore(self) -> int:
        """Re-queue sends that were approved before a restart and haven't gone out yet."""
        restored = 0
        for app_id, override in json.loads(db.kv_get(self.conn, QUEUE_KV) or "[]"):
            app = db.get_application(self.conn, app_id)
            if app and not app["sent_at"] and await self.enqueue(app_id, override_cooldown=override):
                restored += 1
        return restored

    def check(self, app_id: int, *, override_cooldown: bool = False) -> tuple[dict, dict, dict]:
        """Raise SendRefused unless this application may be sent now."""
        app = db.get_application(self.conn, app_id)
        if app is None:
            raise SendRefused(f"application {app_id} not found")
        if app["sent_at"]:
            raise SendRefused(f"application {app_id} was already sent")
        if not app["gmail_draft_id"]:
            raise SendRefused(f"application {app_id} has no Gmail draft")
        job = db.get_job(self.conn, app["job_id"])
        if job["status"] != db.AWAITING_APPROVAL:
            raise SendRefused(f"job #{job['id']} is {job['status']}, not awaiting_approval")
        contact = self.conn.execute("SELECT * FROM contacts WHERE id=?", (app["contact_id"],)).fetchone()
        if contact is None:
            raise SendRefused(f"application {app_id} has no recipient")
        if not override_cooldown and (rec := db.recently_emailed(self.conn, contact["value"], self.cooldown_days)):
            raise SendRefused(f"{contact['value']} was already emailed at {rec['last_emailed_at']} "
                              f"(within {self.cooldown_days} days)")
        return app, job, dict(contact)

    async def send_one(self, app_id: int, *, override_cooldown: bool = False) -> str:
        """Send one approved application now (no spacing). Returns the Gmail message id."""
        if self.sent_today() >= self.max_per_day:
            raise SendRefused(f"daily cap of {self.max_per_day} emails reached")
        app, job, contact = self.check(app_id, override_cooldown=override_cooldown)
        try:
            message_id = await self.gmail.send_draft(app["gmail_draft_id"])
        except Exception as e:
            db.set_status(self.conn, job["id"], db.FAILED)
            raise RuntimeError(f"Gmail send failed for job #{job['id']}: {e}") from e
        now = self.clock().isoformat(timespec="seconds")
        db.update_application(self.conn, app_id, gmail_message_id=message_id, sent_at=now)
        db.set_status(self.conn, job["id"], db.SENT)
        db.record_send(self.conn, contact["value"], at=now)
        log.info("sent job #%s to %s", job["id"], contact["value"])
        return message_id

    async def _wait_for_capacity(self) -> None:
        while self.sent_today() >= self.max_per_day:
            if not self._cap_notified:
                await self.notify(f"Daily cap of {self.max_per_day} emails reached; "
                                  f"{self._queue.qsize() + 1} approved email(s) held until tomorrow.")
                self._cap_notified = True
            tomorrow = local_midnight_utc(self.clock()) + timedelta(days=1, minutes=1)
            await self.sleep(max(60.0, (tomorrow - self.clock()).total_seconds()))
        self._cap_notified = False

    async def run(self) -> None:
        """Worker: send queued applications one by one, spaced by a random delay."""
        while True:
            app_id, override = await self._queue.get()
            try:
                await self._wait_for_capacity()
                if self._sent_once:
                    await self.sleep(random.uniform(*self.spacing))
                message_id = await self.send_one(app_id, override_cooldown=override)
                self._sent_once = True
                if self.on_sent:
                    await self.on_sent(app_id, message_id)
            except SendRefused as e:
                if self.on_refused:
                    await self.on_refused(app_id, str(e))
                else:
                    await self.notify(f"Not sent: {e}")
            except Exception as e:  # noqa: BLE001 - keep the worker alive
                log.exception("send failed")
                await self.notify(str(e))
            finally:
                self._queued.pop(app_id, None)
                self._persist()
                self._queue.task_done()
