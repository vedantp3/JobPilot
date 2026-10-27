from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from telethon.sessions import StringSession
from telethon.tl.functions.messages import GetHistoryRequest, SendMessageRequest, SendReactionRequest
from telethon.tl.types import MessageEntityBold, MessageEntityTextUrl, MessageEntityUrl

from jobpilot import db
from jobpilot.listener import Listener, ReadOnlyTelegramClient, ReadOnlyViolation, entity_urls


def msg(id, text, entities=(), date=None):
    return SimpleNamespace(id=id, message=text, entities=list(entities),
                           date=date or datetime(2026, 10, 23, tzinfo=timezone.utc))


def test_entity_urls_hidden_and_utf16_offsets():
    text = "🚀 Apply here: https://a.io/x now"
    # '🚀' is 2 UTF-16 units, so the URL starts at offset 15
    m = msg(1, text, [
        MessageEntityTextUrl(offset=3, length=10, url="https://jobs.lever.co/k/1"),
        MessageEntityUrl(offset=15, length=14),
        MessageEntityBold(offset=0, length=2),
    ])
    assert entity_urls(m) == ["https://jobs.lever.co/k/1", "https://a.io/x"]


@pytest.fixture
def client():
    return ReadOnlyTelegramClient(StringSession(), 1, "0" * 32)


def test_send_methods_blocked(client):
    for name in ("send_message", "forward_messages", "send_file", "edit_message", "delete_messages"):
        with pytest.raises(ReadOnlyViolation):
            getattr(client, name)


@pytest.mark.asyncio
async def test_raw_write_requests_blocked(client):
    with pytest.raises(ReadOnlyViolation):
        await client(SendMessageRequest(peer="me", message="hi"))
    with pytest.raises(ReadOnlyViolation):
        await client(SendReactionRequest(peer="me", msg_id=1))


@pytest.mark.asyncio
async def test_read_requests_pass_guard(client, monkeypatch):
    seen = []

    async def fake_call(self, request, *a, **k):
        seen.append(type(request).__name__)

    monkeypatch.setattr("telethon.TelegramClient.__call__", fake_call)
    await client(GetHistoryRequest(peer="me", offset_id=0, offset_date=None, add_offset=0,
                                   limit=1, max_id=0, min_id=0, hash=0))
    assert seen == ["GetHistoryRequest"]


async def test_post_queue_live_before_backlog():
    from jobpilot.listener import PostQueue
    q = PostQueue()
    for pid in (1, 2, 3):
        await q.put(pid, PostQueue.BACKLOG)
    await q.put(10)          # live post arrives while backlog is waiting
    await q.put(11)
    await q.put(4, PostQueue.BACKLOG)
    assert [await q.get() for _ in range(6)] == [10, 11, 1, 2, 3, 4]


def test_drop_backlog(conn):
    old = db.upsert_raw_post(conn, tg_message_id=1, channel="c", text="a", posted_at="2026-10-20T08:00:00+00:00")[0]
    new = db.upsert_raw_post(conn, tg_message_id=2, channel="c", text="b", posted_at="2026-10-23T08:00:00+00:00")[0]
    done = db.upsert_raw_post(conn, tg_message_id=3, channel="c", text="c", posted_at="2026-10-19T08:00:00+00:00")[0]
    db.mark_post_processed(conn, done)
    assert db.backlog_stats(conn) == (2, "2026-10-20T08:00:00+00:00")
    assert db.drop_backlog(conn, "2026-10-22T00:00:00+00:00") == 1
    assert db.get_raw_post(conn, old)["processed"] == db.POST_DROPPED
    assert [p["id"] for p in db.unprocessed_posts(conn)] == [new]       # dropped posts don't come back
    assert db.drop_backlog(conn) == 1 and db.backlog_stats(conn) == (0, None)
    assert db.get_raw_post(conn, done)["processed"] == 1


from datetime import date as _date, timedelta as _td  # noqa: E402

from jobpilot.listener import describe_range, local_day_bounds, parse_date_range, parse_day  # noqa: E402

TODAY = _date(2026, 10, 24)


@pytest.mark.parametrize("text, expected", [
    ("today", TODAY), ("yesterday", _date(2026, 10, 23)),
    ("23-10", _date(2026, 10, 23)), ("23/10", _date(2026, 10, 23)), ("23.10.2026", _date(2026, 10, 23)),
    ("23-10-26", _date(2026, 10, 23)), ("2026-10-23", _date(2026, 10, 23)),
    ("23 Oct", _date(2026, 10, 23)), ("23 october 2026", _date(2026, 10, 23)), ("Oct 23", _date(2026, 10, 23)),
    ("3 Sept", _date(2026, 9, 3)), ("1 June", _date(2026, 6, 1)),
    ("5-11", _date(2025, 11, 5)),          # no year + in the future -> last year
])
def test_parse_day(text, expected):
    assert parse_day(text, TODAY) == expected


@pytest.mark.parametrize("bad", ["soon", "32-10", "23-13", "Octember 3"])
def test_parse_day_rejects(bad):
    with pytest.raises(ValueError):
        parse_day(bad, TODAY)


def test_parse_date_range():
    assert parse_date_range(["20-10", "23-10"], TODAY) == (_date(2026, 10, 20), _date(2026, 10, 23))
    assert parse_date_range(["23-10", "20-10"], TODAY) == (_date(2026, 10, 20), _date(2026, 10, 23))
    assert parse_date_range(["today"], TODAY) == (TODAY, TODAY)
    with pytest.raises(ValueError, match="more than 31 days"):
        parse_date_range(["1-8", "24-10"], TODAY)
    with pytest.raises(ValueError):
        parse_date_range([], TODAY)
    assert describe_range(_date(2026, 10, 20), _date(2026, 10, 23)) == "20 Oct 2026 – 23 Oct 2026"


class FakeChannelClient:
    """iter_messages(offset_date=X) yields messages older than X, newest first, like Telethon."""

    def __init__(self, msgs):
        self.msgs = sorted(msgs, key=lambda m: m.date, reverse=True)

    async def get_entity(self, ref):
        return "chan"

    async def iter_messages(self, entity, offset_date=None, **kw):
        for m in self.msgs:
            if offset_date is None or m.date < offset_date:
                yield m


async def test_by_date_fetches_only_that_range(conn):
    lo, hi = local_day_bounds(_date(2026, 10, 21), _date(2026, 10, 22))
    msgs = [
        msg(1, "too old", date=lo - _td(minutes=1)),
        msg(2, "first in range", date=lo + _td(hours=1)),
        msg(3, "   ", date=lo + _td(hours=2)),                # no text: ignored
        msg(4, "last in range", date=hi - _td(minutes=1)),
        msg(5, "too new", date=hi + _td(minutes=1)),
    ]
    lis = Listener(FakeChannelClient(msgs), "@jobs", conn, queue=None)
    fetch = await lis.by_date(_date(2026, 10, 21), _date(2026, 10, 22))
    assert fetch.found == 2 and len(fetch.queued) == 2
    stored = [db.get_raw_post(conn, pid)["tg_message_id"] for pid in fetch.queued]
    assert stored == [2, 4]                                   # oldest first
    assert "2 post(s) from 21 Oct 2026 – 22 Oct 2026: 2 queued, 0 already processed." == fetch.summary()

    # processed posts aren't re-queued; dropped ones come back when you pick their date
    db.mark_post_processed(conn, fetch.queued[0])
    db.mark_post_processed(conn, fetch.queued[1], db.POST_DROPPED)
    again = await lis.by_date(_date(2026, 10, 21), _date(2026, 10, 22))
    assert [db.get_raw_post(conn, p)["tg_message_id"] for p in again.queued] == [4]
    assert "1 queued, 1 already processed" in again.summary()


def test_channels_from_env():
    from jobpilot.config import Secrets
    s = Secrets(_env_file=None, tg_channel="@jobs_a, @jobs_b,-1001234567890  @jobs_a\n@jobs_c")
    assert s.channels == ["@jobs_a", "@jobs_b", "-1001234567890", "@jobs_c"]
    assert Secrets(_env_file=None, tg_channel="@solo").channels == ["@solo"]
    assert Secrets(_env_file=None, tg_channel="").channels == []


class AttachClient(FakeChannelClient):
    def __init__(self, entity):
        super().__init__([])
        self._e, self.handlers = entity, []

    async def get_entity(self, ref):
        return self._e

    def on(self, event):
        def deco(fn):
            self.handlers.append(fn)
            return fn
        return deco


async def test_attach_saves_title_and_warns_when_not_joined(conn):
    from types import SimpleNamespace as NS
    from jobpilot.listener import CHANNEL_TITLE_KV
    joined = Listener(AttachClient(NS(title="Tech Jobs India", left=False)), "@tji", conn, queue=None)
    assert await joined.attach() is None and len(joined.client.handlers) == 2
    assert db.kv_get(conn, CHANNEL_TITLE_KV.format("@tji")) == "Tech Jobs India"
    left = Listener(AttachClient(NS(title="Fresher Jobs", left=True)), "@fj", conn, queue=None)
    warning = await left.attach()
    assert "hasn't joined Fresher Jobs" in warning and "Join it in the Telegram app" in warning


def test_combined_summary():
    from jobpilot.listener import DateFetch, combined_summary
    d = _date(2026, 10, 23)
    one = DateFetch(d, d, found=3, queued=[1, 2], channel="A")
    assert combined_summary([one]) == one.summary()
    two = DateFetch(d, d, found=5, queued=[3], channel="B", truncated=True)
    s = combined_summary([one, two])
    assert s.startswith("8 post(s) from 23 Oct 2026 across 2 channels: 3 queued.")
    assert "• A: 3 found, 2 queued" in s and "• B: 5 found, 1 queued (stopped at the limit)" in s


def test_store_new_edit_and_empty(conn):
    lis = Listener(client=None, channel="@jobs", conn=conn, queue=None)
    pid = lis.store(msg(5, "Hiring!", [MessageEntityTextUrl(offset=0, length=6, url="https://x.io/a")]))
    assert pid and db.get_raw_post(conn, pid)["entities"] == ["https://x.io/a"]
    assert lis.store(msg(5, "Hiring!", [MessageEntityTextUrl(offset=0, length=6, url="https://x.io/a")])) is None
    db.mark_post_processed(conn, pid)
    assert lis.store(msg(5, "Hiring! (fixed link)"), edited=True) == pid
    assert db.get_raw_post(conn, pid)["processed"] == 0
    assert lis.store(msg(6, "   ")) is None  # media-only post without caption
