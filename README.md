# JobPilot

Watches a Telegram job channel, extracts each job with its contact emails or apply links, filters them against your profile, drafts tailored applications, and sends **you** an approval card on Telegram. Nothing is sent or submitted until you tap a button.

Full design: [JOBPILOT_SPEC.md](JOBPILOT_SPEC.md).

**Build status:** Phases 1–3 done. Each post becomes jobs that are deduplicated, filtered, given a resume track, and (for email jobs) a planned application with one recipient. Drafting, Gmail and the approval bot come next.

| Phase | Scope | Status |
|---|---|---|
| 1 | Skeleton: layout, config, DB schema, setup docs | ✅ |
| 2 | Ingest: listener, `--backfill`, normalize, extractor | ✅ |
| 3 | Decide: dedupe, filter, resume picker | ✅ |
| 4 | Email path: drafter, validator, Gmail, rate limiter | ✅ |
| 5 | Approval bot | ✅ |
| 6 | Link path: classifier, answer sheets, Playwright prefill | ✅ |
| 7 | Hardening, hosting | ⏳ |

---

## Setup

### 1. Python environment
Python 3.11+.
```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```
(On Linux/macOS use `.venv/bin/python`.) For link prefill (Phase 6), also install the browser:
```bash
.venv/Scripts/python -m playwright install chromium
```

### 2. Your files
- Copy `.env.example` to `.env` and fill it in (steps below).
- Copy `profile.example.json` to `profile.json`. Fill in **real** `stack` and `proof` (with metrics) for every project. The LLM may only use facts from this file, so anything missing here can't appear in an email.
- Put your resumes in `resumes/` as `ai_ml.pdf`, `sde.pdf`, `product.pdf`, `sap.pdf`, `hybrid.pdf`.
- `profile.json` also takes a `skills` map (group → list) and `achievements`, copied from your resumes; the fit check counts a skill only if it appears there or in a project/experience entry.
- Tune `config.yaml` (batch, roles, keywords, locations, limits).

`.env`, `profile.json`, `resumes/`, `data/`, `*.session`, `token.json` and `credentials.json` are gitignored.

### 3. Telegram API id/hash (for reading the channel)
The channel is read through **your user account** with Telethon, because a bot can't read a channel it isn't an admin of. JobPilot only reads; it never sends, forwards or reacts from your account.

1. Go to <https://my.telegram.org>, log in with your phone number.
2. Open **API development tools**, create an app (any name / short name; platform "Desktop").
3. Copy `api_id` → `TG_API_ID` and `api_hash` → `TG_API_HASH`. Set `TG_PHONE` (e.g. `+91…`).
4. `TG_CHANNEL`: the channel's `@username`, or its numeric id (`-100…`) if it's private. **Several channels:** separate them with commas, e.g. `TG_CHANNEL=@jobs_a, @jobs_b, -1001234567890`. Your account must have joined each one (JobPilot is read-only and can't join for you; it warns in the bot if you haven't). A job posted in several channels is recognised as a duplicate, so it's only emailed once; with more than one channel, cards show which channel the job came from, and `/channels` lists them with counts. `/backfill N` takes the last N posts of each channel; `/date` covers all of them.

On first run Telethon asks for the login code Telegram sends you and creates `jobpilot.session` locally. **Treat the session file like a password.** For headless hosting, generate a session string once and put it in `TG_SESSION_STRING`:
```bash
.venv/Scripts/python -m jobpilot.main --make-session-string
```

The Telethon client is wrapped in `ReadOnlyTelegramClient`, which refuses every send/forward/edit/delete/react call, so even a bug can't make your account post.

### 4. Approval bot (BotFather)
1. In Telegram, message **@BotFather** → `/newbot` → pick a name and username.
2. Copy the token → `BOT_TOKEN`.
3. Message **@userinfobot** to get your numeric chat id → `BOT_OWNER_CHAT_ID`. The bot ignores everyone else.
4. Send `/start` to your new bot once so it is allowed to message you.

### 5. LLM: OpenAI, then Groq + Gemini as free fallbacks
The default chain is `openai,groq,gemini`: each call goes to the first provider with a key, and moves to the next on a rate limit, quota or outage. A provider without a key is skipped.

- **OpenAI** (primary): <https://platform.openai.com/api-keys> → `OPENAI_API_KEY`. Default model `gpt-6-luna` (OpenAI's high-volume model, $0.10/$0.50 per 1M tokens, ~2M tokens/min), via the Responses API with strict JSON-schema output. Reasoning effort is set per task: low for extraction, medium for fit checks and emails. For higher-quality emails try `OPENAI_MODEL=gpt-6.1-sol` (about 20x the price).

- **Groq** (primary, fast): <https://console.groq.com/keys> → **Create API Key** → `GROQ_API_KEY`. Default model `openai/gpt-oss-120b` (strict JSON output). Free limits are per model, roughly 30 requests/min and 1,000/day with a tight tokens-per-minute cap; see <https://console.groq.com/settings/limits>. Groq says it doesn't retain inference inputs by default.
- **Gemini** (fallback): <https://aistudio.google.com/apikey> → **Create API key** → `GEMINI_API_KEY`. Default model `gemini-3.5-flash-lite`; `gemini-3.8-flash` is smarter with a lower daily quota (`GEMINI_MODEL`). On the free tier Google may use what you send to improve its products.

JobPilot spaces calls (`llm_min_interval_seconds` in config.yaml), waits out short rate limits using the server's retry time, hands longer waits to the next provider, and stops with a clear message when every provider's daily quota is used up. `ANTHROPIC_API_KEY` + `anthropic` in the chain remains available as a paid option.

### 6. Gmail OAuth
1. <https://console.cloud.google.com> → create a project.
2. **APIs & Services → Library** → enable **Gmail API**.
3. **Google Auth Platform → Audience** (older UI: *OAuth consent screen*) → External. Under **Test users** click **+ Add users** and add the Gmail address you'll send from. Skipping this gives `Error 403: access_denied` ("can only be accessed by developer-approved testers"). Scopes: `gmail.compose`, `gmail.send`.
4. **Credentials → Create credentials → OAuth client ID → Desktop app**. Download the JSON as `credentials.json` in the project root.
5. Sign in once (opens a browser; pick the account in `profile.json` → `email`). The refresh token is saved to `token.json`:
   ```bash
   .venv/Scripts/python -m jobpilot.main --gmail-auth
   ```
   Google will warn that the app isn't verified: that's your own OAuth app; choose *Advanced → continue*.

> While the app is in "Testing" mode, Google expires the sign-in after 7 days, so you'd rerun `--gmail-auth` weekly. To avoid that, click **Publish app** on the Audience page: for a personal app no review is needed, you just keep seeing the "unverified app" warning at sign-in.

### 7. Check and initialise
```bash
.venv/Scripts/python -m jobpilot.main --check
```
```bash
.venv/Scripts/python -m jobpilot.main --init-db
```
`--check` lists anything still missing (secrets, profile, resumes, Gmail credentials).

---

## Usage

| Command | What it does |
|---|---|
| `python -m jobpilot.main` | Long-running: listener + approval bot + send queue. **This is the normal way to run it.** |
| `python -m jobpilot.main --bot-only` | Approval bot + send queue only (pair with `--poll` from cron) |
| `python -m jobpilot.main --backfill 20` | Process the last 20 posts once |
| `python -m jobpilot.main --date 20-10 23-10` | Process only posts from those days (or one day: `--date 23-10`, `--date yesterday`) once |
| `python -m jobpilot.main --poll` | Process posts newer than the last stored one, then exit |
| `python -m jobpilot.main --dry-run --backfill 5` | Full pipeline, no Gmail/Playwright; cards printed to console. Uses `data/dryrun.db` so your real DB isn't touched |
| `python -m jobpilot.main --fixtures` | Run `tests/fixtures/posts/*.txt` offline with recorded LLM responses (in-memory DB) |
| `python -m jobpilot.main --fixtures --live-llm` | Same, but calls the real LLM and re-records `tests/fixtures/llm/` |

### Fixtures
Each file in `tests/fixtures/posts/` is one channel post. To simulate a hidden hyperlink ("Apply here" with the URL only in the message entities), add a line `---entities---` followed by one URL per line. Paste real channel posts there too, then run `--fixtures --live-llm` once to record their extractions.

The recordings in `tests/fixtures/llm/` (extraction, fit checks and emails) for the 10 synthetic posts were hand-written to match the expected results, against `tests/fixtures/profile.json` (a copy of your profile with a fake phone number). Replay mode always uses that file; `--live-llm` uses your `profile.json` if it exists. Re-record with `--live-llm` once your API key is set to see what the real model does.

## How an email is written
The LLM writes only the body; code adds the subject (`Role — Vedant Patil (VIT Vellore, 2026)`, or the exact subject a post asks for), the greeting (`Hi Priya,` / `Hi Acme Team,`) and the signature. `email_drafter.validate_email` then checks every draft, and a failing draft is rewritten up to 2 more times with the violations fed back:
under `email_max_words` (config.yaml, 200 by default here) · no bullet or numbered lists · no "writing to apply" / "interested in applying" · at least one metric, and every number must appear in `profile.json` or the job post · only projects/employers from `profile.json` · only your GitHub, portfolio and LinkedIn links · GitHub, portfolio and "Resume attached." present · nothing from `never_claim` · no emojis · no sign-off of its own.

If all 3 attempts fail, the job is marked `failed` and you're notified; nothing invalid is ever drafted in Gmail. Otherwise a **Gmail draft** is created with the chosen resume attached as `Vedant_Patil_Resume.pdf`, and the job becomes `drafted`. In `--dry-run`, emails are written and shown in the console cards, but no Gmail draft is made.

**Sending** is done only by the approval bot (Phase 5): a job must be `awaiting_approval` (i.e. you tapped ✅ Send on its card). Sends are spaced randomly by `send_spacing_seconds`, capped at `max_emails_per_day` (the rest wait for tomorrow and you're told), and a recipient emailed within `recipient_cooldown_days` is refused unless you choose *Send anyway*. There is deliberately no command-line send.

The prompt [jobpilot/prompts/email.md](jobpilot/prompts/email.md) contains one of your emails that got a shortlist as a style example; add more successful emails there in `<example>` tags. The model copies their structure and tone, never their facts.

## How a job is decided
1. **Duplicate?** Same company + role (normalised: "ML Intern" = "Machine Learning Internship") as an earlier post → recorded as `duplicate`, new contacts are added to the original job.
2. **Hard rules** (no LLM call): batch explicitly excludes `target_batch`; any money-for-a-job wording (registration fee, security deposit, ...) → skipped and flagged *likely scam*; other `reject_keywords`; `role_category` not in `accept_roles`.
3. **Flags** (always shown): batch not stated, pay not stated / very low / very high / performance-based, location outside `locations_ok`, personal Gmail used as HR address, plus whatever the LLM fit check flags and every skill gap.
4. **Fit score** from the LLM: below `fit_threshold` → `skipped_filter`; within 10 points above → `flagged`; otherwise a match.
5. **Resume track**: rule-based (`sap` > `product` > `sde` for backend/full-stack/DevOps > `hybrid` for SWE+AI > `ai_ml` default).
6. **Email plan**: one recipient per job (named person > personal address > generic inbox). One email per recipient per post, for the best-fit role, mentioning other roles that also passed. A recipient emailed within `recipient_cooldown_days`, or with another email still awaiting approval, is skipped; if that leaves no address, the job is flagged and you get a *Combine / Send anyway / Skip?* notice.

### The approval bot
Every matched job arrives in your bot chat as a card. Only `BOT_OWNER_CHAT_ID` can use it; everyone else is ignored. Send `/start` to the bot once first, or Telegram won't let it message you.

| Card | Buttons |
|---|---|
| 📧 email job: fit, resume + reason, recipient, flags, full email | ✅ Send · ✏️ Edit · 📄 Resume ▾ · ❌ Skip |
| 🔗 link job: fit, link type, link | 🖥 Prefill (Google Forms, Lever, Greenhouse, Ashby) · 📋 Answer sheet · 🌐 Open link · ✔️ Mark submitted · ❌ Skip |
| ⏸ recipient already emailed / has an email waiting | 🔗 Combine (fold this role into the waiting email) · 📨 Send anyway · ❌ Skip |

- **✅ Send** queues the Gmail draft; the card shows *Queued*, then *Sent* with the time. Sends are spaced and capped (see *How an email is written*). Queued sends survive a restart.
- **✏️ Edit**: your next message is the instruction ("shorter", "mention StockPilot instead", or a full new body). The LLM revises, the rules are re-checked, the Gmail draft is updated and a new card replaces the old one. If the edit would break a rule, the old draft stays and you're told why.
- **📄 Resume ▾** swaps the attached resume (and updates the Gmail draft).

**Link jobs:**
- **🖥 Prefill** opens the form in a Chrome window on this computer and fills what it can: your name, email, phone, college, degree, grad year, CGPA, links, location, notice period and expected CTC from `profile.json`; the resume for that job's track into file fields; and LLM-drafted answers (max 80 words, same honesty checks as emails) to free-text questions, highlighted in yellow. **It never clicks anything, so it can't submit.** The bot then lists what was filled, the drafted answers, and what's left for you (choice questions, 10th/12th marks, current CTC, ...). Review, submit on the site, tap ✔️ Mark submitted. It uses your installed Google Chrome (else Edge, else Playwright's bundled Chromium; `prefill_browser` in config.yaml) with its own profile in `browser_profile/`, separate from your everyday browser, so a Google sign-in for forms that need one is remembered there. Prefill needs a desktop: on a server (or with `link_prefill: false`) the button isn't shown, and a failed prefill sends the answer sheet instead.
- **📋 Answer sheet** (any link, incl. LinkedIn, Naukri, Workday, which JobPilot never logs into) sends your standard fields and drafted answers to the usual questions, each in a tap-to-copy block. Fill `standard_answers` in `profile.json` and the drafts start from your own wording.

**Skipped jobs:** `/skipped [N]` lists recent skips with their reason and a ✏️ *Draft anyway* button each (the job goes through as flagged; cooldown rules still apply). `/rescore [N]` re-runs the fit check on the last N low-fit skips, e.g. after you improve `profile.json` or the prompt; jobs that now pass get cards.

**By date:** `/date 23-10` processes the channel's posts from that day, `/date 20-10 23-10` a range (up to 31 days; local time; formats like `23/10/2026`, `2026-10-23`, `23 Oct`, `today`, `yesterday`). Already-processed posts are skipped; posts you dropped via `/backlog` come back when you pick their date.

Commands: `/pending`, `/card N`, `/date DAY [TO]`, `/skipped [N]`, `/rescore [N]`, `/stats`, `/status N interview|rejected|offer`, `/export` (CSV), `/pause`, `/resume`, `/backfill N`, `/cancel`, `/help`. Errors from any stage are posted to the chat; one bad post never stops the listener.

New channel posts are always processed before older ones still waiting (from `/backfill` or a restart); `/backlog` shows how many are waiting and can drop the old ones.

`--backfill N` and `--poll` also post cards to the bot, but the buttons only work while a long-running `python -m jobpilot.main` (or `--bot-only`) is up; Telegram keeps button presses for 24 hours.

## Tests
```bash
.venv/Scripts/python -m pytest
```
LLM-dependent tests are marked `llm` and replay recorded responses by default.

## Hosting
Documented in Phase 7. Short version: run the long-lived process on your laptop (needed for Playwright prefill) or a free VM via `systemd`/`pm2`; or run `--poll` from cron/GitHub Actions with the bot hosted separately.
