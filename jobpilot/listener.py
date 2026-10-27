"""Telethon listener: read-only channel watcher, --backfill, --poll.

The user account is strictly read-only: `ReadOnlyTelegramClient` refuses every
send/forward/edit/delete/react method and raw request, so a bug elsewhere can't
make the account post anything.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession
from telethon.tl.types import MessageEntityTextUrl, MessageEntityUrl

from jobpilot import db
from jobpilot.config import PROJECT_ROOT, Secrets

log = logging.getLogger(__name__)

POLL_FIRST_RUN_LIMIT = 20
CHANNEL_TITLE_KV = "channel_title:{}"   # configured channel ref -> its display title


MAX_DATE_RANGE_DAYS = 31
MAX_DATE_POSTS = 500
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTH_RE = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
             r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")


def parse_day(text: str, today: date | None = None) -> date:
    """'today', 'yesterday', '23-10', '23/10/2026', '2026-10-23', '23 Oct', 'Oct 23' -> date.
    Day-month order (Indian style); a missing year means the most recent such date."""
    today = today or datetime.now().astimezone().date()
    s = text.strip().lower().replace(",", " ")
    if s == "today":
        return today
    if s == "yesterday":
        return today - timedelta(days=1)
    if m := re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s):
        return date(int(m[1]), int(m[2]), int(m[3]))
    day = month = year = None
    if m := re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})(?:[-/.](\d{2,4}))?", s):
        day, month, year = int(m[1]), int(m[2]), m[3]
    elif m := re.fullmatch(rf"(\d{{1,2}})\s*{_MONTH_RE}\.?(?:\s+(\d{{2,4}}))?", s):
        day, month, year = int(m[1]), _MONTHS[m[2][:3]], m[3]
    elif m := re.fullmatch(rf"{_MONTH_RE}\.?\s*(\d{{1,2}})(?:\s+(\d{{2,4}}))?", s):
        day, month, year = int(m[2]), _MONTHS[m[1][:3]], m[3]
    if not day or not month:
        raise ValueError(f"can't read the date {text!r}; try 23-10, 23/10/2026, 2026-10-23 or 23 Oct")
    if year:
        y = int(year)
        d = date(y + 2000 if y < 100 else y, month, day)
    else:
        d = date(today.year, month, day)
        if d > today:            # "23-12" in January means last December
            d = date(today.year - 1, month, day)
    return d


def parse_date_range(args: list[str], today: date | None = None) -> tuple[date, date]:
    """One date -> that day; two dates -> inclusive range (order doesn't matter)."""
    if not args or len(args) > 2:
        raise ValueError("give one date or two: e.g. 23-10, or 20-10 23-10")
    start = parse_day(args[0], today)
    end = parse_day(args[1], today) if len(args) == 2 else start
    if start > end:
        start, end = end, start
    if (end - start).days + 1 > MAX_DATE_RANGE_DAYS:
        raise ValueError(f"that's more than {MAX_DATE_RANGE_DAYS} days; pick a shorter range")
    return start, end


def local_day_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    """[start 00:00, end+1 00:00) in local time, as aware datetimes."""
    tz = datetime.now().astimezone().tzinfo
    lo = datetime.combine(start, time.min).replace(tzinfo=tz)
    hi = datetime.combine(end + timedelta(days=1), time.min).replace(tzinfo=tz)
    return lo, hi


def describe_range(start: date, end: date) -> str:
    fmt = lambda d: d.strftime("%d %b %Y").lstrip("0")  # noqa: E731
    return fmt(start) if start == end else f"{fmt(start)} – {fmt(end)}"


@dataclass
class DateFetch:
    start: date
    end: date
    found: int                 # posts in the channel in that range
    queued: list[int]          # raw_post_ids that still need processing
    truncated: bool = False
    channel: str = ""

    def summary(self) -> str:
        done = self.found - len(self.queued)
        s = (f"{self.found} post(s) from {describe_range(self.start, self.end)}: "
             f"{len(self.queued)} queued, {done} already processed.")
        if self.truncated:
            s += f" (Stopped at {MAX_DATE_POSTS}; pick a shorter range for the rest.)"
        return s


def combined_summary(fetches: list[DateFetch]) -> str:
    """One line per channel when there are several."""
    if len(fetches) == 1:
        return fetches[0].summary()
    head = (f"{sum(f.found for f in fetches)} post(s) from {describe_range(fetches[0].start, fetches[0].end)} "
            f"across {len(fetches)} channels: {sum(len(f.queued) for f in fetches)} queued.")
    lines = [f"• {f.channel}: {f.found} found, {len(f.queued)} queued"
             + (" (stopped at the limit)" if f.truncated else "") for f in fetches]
    return "\n".join([head, *lines])


class PostQueue:
    """Posts to process: live channel posts (LIVE) always go before backlog/backfill (BACKLOG);
    first-in-first-out within each priority. `put(pid)` defaults to LIVE, which is what the
    listener's event handlers use."""

    LIVE, BACKLOG = 0, 1

    def __init__(self):
        self._q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = itertools.count()

    async def put(self, raw_post_id: int, priority: int = LIVE) -> None:
        await self._q.put((priority, next(self._seq), raw_post_id))

    async def get(self) -> int:
        return (await self._q.get())[2]

    def task_done(self) -> None:
        self._q.task_done()

    def qsize(self) -> int:
        return self._q.qsize()


class ReadOnlyViolation(PermissionError):
    pass


_BLOCKED_METHODS = (
    "send_message", "send_file", "forward_messages", "edit_message", "delete_messages",
    "pin_message", "unpin_message", "send_read_acknowledge", "delete_dialog", "edit_admin",
    "edit_permissions", "kick_participant", "edit_folder", "send_reaction",
)
# Raw TL requests that would write anything visible to others.
_BLOCKED_REQUEST = re.compile(
    r"^(Send|Forward|Edit|Delete|Join|Leave|Report|Block|Save|Set|Update|Toggle|Invite|"
    r"Import|Create|Upload|Read|Vote|Pin|Unpin|Start|Accept|Hide|Install|Migrate)\w*Request$"
)
# Requests matching the pattern above that login/Telethon internals need; none is visible to others.
_ALLOWED_REQUEST = {
    "SendCodeRequest", "ImportAuthorizationRequest", "UpdateStatusRequest", "SaveAppLogRequest",
}


class ReadOnlyTelegramClient(TelegramClient):
    def __getattribute__(self, name: str):
        if name in _BLOCKED_METHODS:
            raise ReadOnlyViolation(f"JobPilot listener is read-only: {name}() is disabled")
        return super().__getattribute__(name)

    async def __call__(self, request, *args, **kwargs):
        reqs = request if isinstance(request, (list, tuple)) else [request]
        for r in reqs:
            n = type(r).__name__
            if _BLOCKED_REQUEST.match(n) and n not in _ALLOWED_REQUEST:
                raise ReadOnlyViolation(f"JobPilot listener is read-only: {n} is disabled")
        return await super().__call__(request, *args, **kwargs)


def build_client(secrets: Secrets) -> ReadOnlyTelegramClient:
    if not secrets.tg_api_id or not secrets.tg_api_hash:
        raise RuntimeError("TG_API_ID and TG_API_HASH must be set in .env")
    session = (StringSession(secrets.tg_session_string) if secrets.tg_session_string
               else str(PROJECT_ROOT / "jobpilot"))
    return ReadOnlyTelegramClient(
        session, secrets.tg_api_id, secrets.tg_api_hash,
        flood_sleep_threshold=24 * 3600,  # always sleep through FloodWait instead of raising
        receive_updates=True,
    )


async def make_session_string(secrets: Secrets) -> str:
    """Interactive login that prints a StringSession for headless hosting."""
    client = ReadOnlyTelegramClient(StringSession(), secrets.tg_api_id, secrets.tg_api_hash)
    await client.start(phone=secrets.tg_phone or None)
    s = client.session.save()
    await client.disconnect()
    return s


def entity_urls(message) -> list[str]:
    """Hidden hyperlink targets (MessageEntityTextUrl). Plain URLs are already in the text,
    but MessageEntityUrl is included too in case the text was truncated oddly."""
    urls: list[str] = []
    text = message.message or ""
    for ent in message.entities or []:
        if isinstance(ent, MessageEntityTextUrl):
            urls.append(ent.url)
        elif isinstance(ent, MessageEntityUrl):
            # offsets are UTF-16 code units
            u16 = text.encode("utf-16-le")
            urls.append(u16[ent.offset * 2:(ent.offset + ent.length) * 2].decode("utf-16-le"))
    return urls


def message_text(message) -> str:
    """Raw message text (no markdown), with any media caption included."""
    return message.message or ""


class Listener:
    def __init__(self, client: ReadOnlyTelegramClient, channel: str, conn: sqlite3.Connection,
                 queue: "PostQueue | asyncio.Queue[int]"):
        self.client, self.channel, self.conn, self.queue = client, channel, conn, queue
        self._entity = None

    @staticmethod
    def _channel_ref(channel: str):
        return int(channel) if re.fullmatch(r"-?\d+", channel) else channel

    async def entity(self):
        if self._entity is None:
            self._entity = await self.client.get_entity(self._channel_ref(self.channel))
        return self._entity

    def store(self, message, *, edited: bool = False) -> int | None:
        """Store a message; returns raw_post_id if it needs (re)processing."""
        text = message_text(message)
        if not text.strip():
            return None
        pid, changed = db.upsert_raw_post(
            self.conn, tg_message_id=message.id, channel=self.channel, text=text,
            entities=entity_urls(message),
            posted_at=message.date.isoformat() if message.date else None, edited=edited,
        )
        return pid if changed else None

    async def _enqueue(self, message, edited: bool) -> None:
        try:
            pid = self.store(message, edited=edited)
            if pid is not None:
                await self.queue.put(pid)
        except Exception:  # noqa: BLE001 - never let one message kill the handler
            log.exception("failed to store message %s", getattr(message, "id", "?"))

    @property
    def title(self) -> str:
        return getattr(self._entity, "title", None) or self.channel

    async def attach(self) -> str | None:
        """Start receiving live posts. Returns a warning if live posts won't arrive (not a member)."""
        chat = await self.entity()
        db.kv_set(self.conn, CHANNEL_TITLE_KV.format(self.channel), self.title)

        @self.client.on(events.NewMessage(chats=chat))
        async def _new(event):
            await self._enqueue(event.message, edited=False)

        @self.client.on(events.MessageEdited(chats=chat))
        async def _edited(event):
            await self._enqueue(event.message, edited=True)

        log.info("listening to %s (%s)", self.title, self.channel)
        if getattr(chat, "left", False):
            return (f"Your Telegram account hasn't joined {self.title} ({self.channel}), so its new posts won't "
                    "arrive live. Join it in the Telegram app; /backfill and /date still work meanwhile.")
        return None

    async def _iter(self, **kwargs) -> list:
        while True:
            try:
                return [m async for m in self.client.iter_messages(await self.entity(), **kwargs)]
            except FloodWaitError as e:
                log.warning("FloodWait: sleeping %ss", e.seconds)
                await asyncio.sleep(e.seconds + 1)

    async def backfill(self, n: int) -> list[int]:
        """Store the last n posts (oldest first). Returns raw_post_ids to process."""
        msgs = await self._iter(limit=n)
        return self._store_all(reversed(msgs))

    async def poll(self) -> list[int]:
        """Store posts newer than the last stored message id."""
        last = db.last_message_id(self.conn, self.channel)
        if not last:  # first run: start from the most recent posts, not the whole history
            return await self.backfill(POLL_FIRST_RUN_LIMIT)
        return self._store_all(await self._iter(min_id=last, reverse=True))

    async def by_date(self, start: date, end: date) -> "DateFetch":
        """Store the channel's posts from local days start..end (inclusive), oldest first.
        Posts already processed aren't re-queued."""
        lo, hi = local_day_bounds(start, end)
        msgs = []
        while True:
            try:
                msgs = []
                # newest first, starting just before the end of the range
                async for m in self.client.iter_messages(await self.entity(), offset_date=hi):
                    if m.date is None:
                        continue
                    if m.date < lo:
                        break
                    if m.date < hi and (m.message or "").strip():
                        msgs.append(m)
                    if len(msgs) >= MAX_DATE_POSTS:
                        break
                break
            except FloodWaitError as e:
                log.warning("FloodWait: sleeping %ss", e.seconds)
                await asyncio.sleep(e.seconds + 1)
        with self.conn:   # picking a date explicitly brings back posts dropped via /backlog
            for m in msgs:
                self.conn.execute(
                    "UPDATE raw_posts SET processed=0 WHERE channel=? AND tg_message_id=? AND processed=?",
                    (self.channel, m.id, db.POST_DROPPED),
                )
        to_process = self._store_all(reversed(msgs))
        return DateFetch(start, end, found=len(msgs), queued=to_process,
                         truncated=len(msgs) >= MAX_DATE_POSTS, channel=self.title)

    def _store_all(self, msgs) -> list[int]:
        ids = []
        for m in msgs:
            pid = self.store(m)
            if pid is None:  # unchanged; still process if never finished
                existing = self.conn.execute(
                    "SELECT id FROM raw_posts WHERE channel=? AND tg_message_id=? AND processed=0",
                    (self.channel, m.id),
                ).fetchone()
                pid = existing["id"] if existing else None
            if pid is not None:
                ids.append(pid)
        return ids


def session_file_exists() -> bool:
    return Path(PROJECT_ROOT / "jobpilot.session").exists()
