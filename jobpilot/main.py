"""JobPilot entrypoint.

    python -m jobpilot.main                      # run listener + approval bot + send queue
    python -m jobpilot.main --bot-only           # approval bot + send queue only (with --poll from cron)
    python -m jobpilot.main --poll               # fetch new posts once, process, exit (cron)
    python -m jobpilot.main --backfill 20        # process the last 20 channel posts once
    python -m jobpilot.main --date 20-10 23-10   # process posts from those days once
    python -m jobpilot.main --fixtures           # run tests/fixtures/posts offline
    python -m jobpilot.main --dry-run ...        # no Gmail, no Playwright; print cards to console
    python -m jobpilot.main --check              # validate config/secrets/files
    python -m jobpilot.main --init-db            # create the SQLite schema
    python -m jobpilot.main --make-session-string  # print a TG_SESSION_STRING for hosting
    python -m jobpilot.main --gmail-auth         # one-time Google sign-in for Gmail drafts/sending
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from jobpilot import db
from jobpilot.config import PROJECT_ROOT, RESUME_TRACKS, get_config, get_secrets, load_profile

log = logging.getLogger("jobpilot")

FIXTURE_POSTS = PROJECT_ROOT / "tests" / "fixtures" / "posts"
FIXTURE_LLM = PROJECT_ROOT / "tests" / "fixtures" / "llm"
FIXTURE_PROFILE = PROJECT_ROOT / "tests" / "fixtures" / "profile.json"
DRY_RUN_DB = "data/dryrun.db"


def check_setup() -> bool:
    """Print a setup checklist. Returns True when everything required is present."""
    cfg, sec = get_config(), get_secrets()
    ok = True

    def report(good: bool, label: str, hint: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {label}" + (f"  -> {hint}" if hint and not good else ""))

    print("Secrets (.env):")
    for group, names in {
        "Telegram listener": ("tg_api_id", "tg_api_hash", "tg_channel"),
        "Approval bot": ("bot_token", "bot_owner_chat_id"),
        f"LLM ({' > '.join(sec.providers)})": tuple(sec.llm_key_names),
    }.items():
        missing = sec.missing(*names)
        report(not missing, group, f"set {', '.join(missing)}")
    if not sec.tg_session_string:
        report(bool(sec.tg_phone), "Telegram login", "set TG_PHONE or TG_SESSION_STRING")
    print("  [..] LLM models: " + ", ".join(f"{p}={sec.model_for(p)}" for p in sec.providers))
    print(f"  [..] channels ({len(sec.channels)}): {', '.join(sec.channels) or '-'}")

    print("Files:")
    try:
        profile = load_profile(cfg.path("profile_path"))
        report(True, f"profile.json ({profile.name}, {len(profile.projects)} projects)")
        empty = [p.name for p in profile.projects if not p.proof]
        if empty:
            print(f"  [..] projects without proof/metrics yet: {', '.join(empty)}")
    except Exception as e:  # noqa: BLE001 - surface any profile problem
        report(False, "profile.json", str(e))
    for track in RESUME_TRACKS:
        f = cfg.resume_file(track)
        report(f.exists(), f"resume {f.name}", f"add {f}")
    creds = PROJECT_ROOT / sec.gmail_credentials_path
    report(creds.exists(), "Gmail credentials.json", "see README: Gmail OAuth")
    token = PROJECT_ROOT / sec.gmail_token_path
    report(token.exists(), "Gmail signed in (token.json)", "run: python -m jobpilot.main --gmail-auth")

    print("Database:")
    conn = db.connect(cfg.path("db_path"))
    counts = db.status_counts(conn)
    print(f"  [ok] {cfg.path('db_path')} ({sum(counts.values())} jobs)")
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="jobpilot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--poll", action="store_true", help="fetch posts newer than the last stored one, process, exit")
    mode.add_argument("--backfill", type=int, metavar="N", help="process the last N channel posts once")
    mode.add_argument("--date", nargs="+", metavar="DAY",
                      help="process posts from one day or a range once, e.g. --date 23-10 or --date 20-10 23-10")
    mode.add_argument("--fixtures", nargs="?", const=str(FIXTURE_POSTS), metavar="DIR",
                      help="process fixture posts offline (default tests/fixtures/posts)")
    mode.add_argument("--check", action="store_true", help="validate setup and exit")
    mode.add_argument("--init-db", action="store_true", help="create the database schema and exit")
    mode.add_argument("--make-session-string", action="store_true", help="log in and print TG_SESSION_STRING")
    mode.add_argument("--gmail-auth", action="store_true", help="sign in to Gmail once and save token.json")
    mode.add_argument("--bot-only", action="store_true",
                      help="run only the approval bot + send queue (pair with --poll from cron)")
    p.add_argument("--dry-run", action="store_true", help="no Gmail/Playwright calls; print cards to console")
    p.add_argument("--live-llm", action="store_true",
                   help="with --fixtures: call the real LLM and record responses instead of replaying")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    return p.parse_args(argv)


def _db_path(args: argparse.Namespace) -> Path:
    cfg = get_config()
    return PROJECT_ROOT / DRY_RUN_DB if args.dry_run else cfg.path("db_path")


async def _console_notify(text: str) -> None:
    print(f"[notify] {text}")


async def _run_fixtures(args: argparse.Namespace) -> int:
    from jobpilot.llm import get_llm
    from jobpilot.pipeline import Pipeline, run_fixtures

    # replayed responses were recorded against the fixture profile
    profile_path = get_config().path("profile_path")
    if not args.live_llm or not profile_path.exists():
        profile_path = FIXTURE_PROFILE
    llm = get_llm(record_dir=FIXTURE_LLM) if args.live_llm else get_llm(replay_dir=FIXTURE_LLM)
    conn = db.connect(":memory:")
    async with Pipeline(conn, llm, load_profile(profile_path), dry_run=True,
                        notify=_console_notify) as pipe:
        job_ids = await run_fixtures(pipe, Path(args.fixtures))
    print(f"\n{len(job_ids)} job(s) decided from fixtures; status counts: {db.status_counts(conn)}")
    return 0


async def _run_telegram(args: argparse.Namespace) -> int:
    """Listener (+ approval bot and send queue unless --dry-run).

    Long-running by default; --backfill/--poll process once, post cards and exit;
    --bot-only runs just the approval bot + send queue (no listener)."""
    from jobpilot.llm import get_llm
    from jobpilot.pipeline import Pipeline

    sec, cfg = get_secrets(), get_config()
    needed = [] if args.bot_only else ["tg_api_id", "tg_api_hash", "tg_channel"]
    if not args.dry_run:
        needed += ["bot_token", "bot_owner_chat_id"]
    if missing := sec.missing(*needed):
        print(f"Missing in .env: {', '.join(missing)}", file=sys.stderr)
        return 1
    try:
        profile = load_profile()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    gmail = None
    if not args.dry_run:
        try:
            gmail = await _gmail_client(profile, interactive=False)
        except Exception as e:  # noqa: BLE001 - explain and stop
            print(f"Gmail isn't ready: {e}\nRun --gmail-auth first, or use --dry-run.", file=sys.stderr)
            return 1
    date_range = None
    if args.date:
        from jobpilot.listener import parse_date_range
        try:
            date_range = parse_date_range(args.date)
        except ValueError as e:
            print(f"--date: {e}", file=sys.stderr)
            return 1
    conn = db.connect(_db_path(args))
    long_running = not (args.backfill or args.poll or date_range)

    async with Pipeline(conn, get_llm(), profile, dry_run=args.dry_run, notify=_console_notify,
                        gmail=gmail) as pipe:
        bot = service = send_queue = None
        if not args.dry_run:
            bot, service, send_queue = _build_bot(conn, pipe, gmail, sec, cfg)
            await bot.initialize()
        tasks: list[asyncio.Task] = []
        client = None
        try:
            if service is not None:
                posted = await service.post_pending_cards()   # anything decided while we were offline
                if posted:
                    log.info("posted %d pending card(s)", posted)
            if bot is not None and long_running:
                await bot.start()
                await bot.updater.start_polling(allowed_updates=["message", "callback_query"])
                restored = await send_queue.restore()
                tasks.append(asyncio.create_task(send_queue.run()))
                log.info("approval bot running%s", f"; {restored} queued send(s) restored" if restored else "")
            if args.bot_only:
                await asyncio.Event().wait()   # until Ctrl+C
                return 0

            from jobpilot.listener import Listener, PostQueue, build_client, combined_summary

            posts = PostQueue()   # live channel posts jump ahead of the backlog
            client = build_client(sec)
            await client.start(phone=sec.tg_phone or None)
            notify = service.notify if service is not None else _console_notify
            listeners: list[Listener] = []
            for ch in sec.channels:           # one listener per channel, all feeding the same queue
                lis = Listener(client, ch, conn, posts)
                try:
                    await lis.entity()
                except Exception as e:  # noqa: BLE001 - a bad channel mustn't stop the others
                    await notify(f"Can't open channel {ch}: {type(e).__name__}: {e}. Check TG_CHANNEL in .env.")
                    continue
                listeners.append(lis)
            if not listeners:
                print("None of the channels in TG_CHANNEL could be opened.", file=sys.stderr)
                return 1
            pending = [p["id"] for p in db.unprocessed_posts(conn)]  # left over from a previous run

            async def fetch_dates(start, end) -> tuple[list[int], str]:
                fetches = [await lis.by_date(start, end) for lis in listeners]
                return [pid for f in fetches for pid in f.queued], combined_summary(fetches)

            if not long_running:
                if date_range:
                    ids, summary = await fetch_dates(*date_range)
                    print(summary)
                    pending = []                       # only that range, not older leftovers
                else:
                    ids = []
                    for lis in listeners:
                        ids += await (lis.backfill(args.backfill) if args.backfill else lis.poll())
                for pid in dict.fromkeys(pending + ids):
                    await pipe.process_post(pid)
                return 0

            if service is not None:
                async def backfill(n: int) -> int:
                    queued = 0
                    for lis in listeners:          # the last N posts of EACH channel
                        for pid in await lis.backfill(n):
                            await posts.put(pid, PostQueue.BACKLOG)
                            queued += 1
                    return queued

                async def by_date(day_args: list[str]) -> str:
                    from jobpilot.listener import parse_date_range
                    ids, summary = await fetch_dates(*parse_date_range(day_args))
                    for pid in ids:
                        await posts.put(pid, PostQueue.BACKLOG)
                    return summary

                service.backfill = backfill
                service.by_date = by_date
                service.channels = [(lis.channel, lis.title) for lis in listeners]
            for pid in pending:
                await posts.put(pid, PostQueue.BACKLOG)
            for lis in listeners:
                if warning := await lis.attach():
                    await notify(warning)
            log.info("watching %d channel(s): %s", len(listeners), ", ".join(lis.title for lis in listeners))

            async def worker() -> None:
                while True:
                    pid = await posts.get()
                    try:
                        while db.kv_get(conn, "paused") == "1":   # /pause: keep saving posts, process later
                            await asyncio.sleep(5)
                        post = db.get_raw_post(conn, pid)
                        if post is not None and post["processed"] == 0:   # skip done or dropped (/backlog)
                            await pipe.process_post(pid)  # never raises
                    finally:
                        posts.task_done()

            tasks.append(asyncio.create_task(worker()))
            await client.run_until_disconnected()
            return 0
        finally:
            for t in tasks:
                t.cancel()
            if client is not None:
                await client.disconnect()
            if bot is not None:
                if bot.updater and bot.updater.running:
                    await bot.updater.stop()
                if bot.running:
                    await bot.stop()
                await bot.shutdown()
            if service is not None and service.prefiller is not None:
                await service.prefiller.close()


def _build_bot(conn, pipe, gmail, sec, cfg):
    """Approval bot + send queue, wired into the pipeline's notifications and card hook."""
    from jobpilot.approval_bot import ApprovalService, TelegramMessenger, build_application
    from jobpilot.gmail_client import SendQueue

    send_queue = SendQueue(conn, gmail, max_per_day=cfg.max_emails_per_day,
                           spacing=tuple(cfg.send_spacing_seconds), cooldown_days=cfg.recipient_cooldown_days)
    service = ApprovalService(conn, pipe, send_queue)
    from jobpilot.link_handler import Prefiller, prefill_supported

    if cfg.link_prefill and prefill_supported():
        service.prefiller = Prefiller(cfg.path("browser_profile_dir"),    # launches a browser on first use
                                      browser=cfg.prefill_browser)
    bot = build_application(sec.bot_token, sec.bot_owner_chat_id, service)
    service.messenger = TelegramMessenger(bot.bot, sec.bot_owner_chat_id)
    send_queue.notify = service.notify
    send_queue.on_sent = service.on_sent
    send_queue.on_refused = service.on_refused
    pipe.notify = service.notify
    pipe.on_ready = service.post_cards
    return bot, service, send_queue


async def _gmail_client(profile, *, interactive: bool):
    """Build an authorised Gmail client and check it's the profile's account."""
    from jobpilot.gmail_client import GmailClient

    sec = get_secrets()
    gmail = GmailClient(PROJECT_ROOT / sec.gmail_credentials_path, PROJECT_ROOT / sec.gmail_token_path,
                        interactive=interactive)
    address = await gmail.account_email()
    if address.lower() != profile.email.lower():
        log.warning("Gmail is signed in as %s but profile.json email is %s; emails will come from %s",
                    address, profile.email, address)
    return gmail


async def _gmail_auth() -> int:
    profile = load_profile()
    gmail = await _gmail_client(profile, interactive=True)
    print(f"Gmail ready: drafts will be created in {await gmail.account_email()}. "
          f"Token saved to {gmail.token_path.name}.")
    return 0


async def _make_session_string() -> int:
    from jobpilot.listener import make_session_string

    s = await make_session_string(get_secrets())
    print("\nTG_SESSION_STRING (treat it like a password):\n" + s)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("telethon", "httpx", "httpcore", "anthropic", "google_genai", "googleapiclient", "groq",
                  "telegram", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.init_db:
        path = get_config().path("db_path")
        db.connect(path).close()
        print(f"Database ready at {path}")
        return 0
    if args.check:
        return 0 if check_setup() else 1
    if args.make_session_string:
        return asyncio.run(_make_session_string())
    if args.gmail_auth:
        return asyncio.run(_gmail_auth())
    if args.fixtures:
        return asyncio.run(_run_fixtures(args))
    try:
        return asyncio.run(_run_telegram(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
