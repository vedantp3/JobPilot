# JobPilot — Telegram Job Channel → Application Pipeline

**Build spec for Claude Code.** Read this whole document before writing code. Build in the phase order given in §12, and stop for my review at the end of each phase.

---

## 1. Goal

A Telegram channel I follow posts job and internship listings. Each post contains either **application links** or **email IDs of company members** (HR, recruiters, founders, engineers). Applying quickly and well has gotten me real interviews.

JobPilot should:

1. Listen to the channel in real time.
2. Extract every job in each post, along with its contact email(s) or application link(s).
3. Filter jobs against my profile and flag problems.
4. For **email jobs**: draft a tailored application email with the right resume attached, as a Gmail draft.
5. For **link jobs**: classify the link, prefill simple forms where possible, and otherwise queue the link with copy-ready answers.
6. Send me a Telegram approval card for every matched job. **Nothing is ever sent or submitted without my tap.**
7. Track everything in a local database.

### Non-goals (do not build)
- Auto-sending emails or auto-submitting forms without approval.
- Sending Telegram messages *from my user account* to anyone. Outbound Telegram messages come only from my own bot, and only to me.
- Scraping job boards outside the channel.
- Logging into LinkedIn, Naukri or Workday, or solving CAPTCHAs.

---

## 2. Architecture

```
Telegram channel
      │  (Telethon user session, read-only)
      ▼
[listener] ──► raw_posts table
      │
      ▼
[extractor]  regex (emails/URLs) + LLM (structure, role↔contact mapping)
      │
      ▼
[dedupe]  job hash + recipient cooldown
      │
      ▼
[filter]  rules + LLM fit check → match / flag / skip
      │
      ├── email job ──► [email_drafter] ──► Gmail draft (resume attached)
      └── link job  ──► [link_handler]  ──► classify → prefill / queue
      │
      ▼
[approval_bot]  python-telegram-bot → card to me with buttons
      │
      ├── ✅ Send     → Gmail send draft (rate-limited)  / open prefilled form
      ├── ✏️ Edit     → I reply with changes → LLM revises → new card
      └── ❌ Skip     → mark skipped
      ▼
[tracker]  SQLite + /stats, /pending, /export commands
```

Run a single long-running process with asyncio, holding both the Telethon client and the bot. Also provide a `--poll` mode for cron or GitHub Actions hosting (§11).

---

## 3. Tech stack

- Python 3.11+
- **Telethon**: reads the channel as my user account. A Bot API bot can't read a channel unless it's an admin there, and I'm not.
- **python-telegram-bot** v21+: the approval bot (a separate bot from @BotFather).
- **Anthropic API** (Claude). Make the model configurable, with a cheap or fast model as the default. Put LLM calls behind a small `llm.py` interface so Gemini can be swapped in.
- **SQLite** via `sqlite3` or SQLModel.
- **Gmail API** (google-api-python-client, OAuth desktop flow, `gmail.compose` + `gmail.send` scopes).
- **Playwright** (Chromium, headed): prefills Google Forms.
- **httpx**: resolves shortened links.
- **pydantic**: config and LLM output validation.
- **pydantic-settings / python-dotenv**: secrets.

---

## 4. Project layout

```
jobpilot/
├── .env.example
├── .gitignore              # .env, *.session, token.json, credentials.json, profile.json, resumes/, data/
├── README.md               # setup steps, incl. getting api_id/hash, BotFather, Gmail OAuth
├── config.yaml             # non-secret settings (channel, filters, limits)
├── profile.json            # my details + standard answers (gitignored)
├── profile.example.json
├── resumes/                # ai_ml.pdf, sde.pdf, product.pdf, sap.pdf, hybrid.pdf (gitignored)
├── data/jobpilot.db
├── jobpilot/
│   ├── main.py             # entrypoint: run listener + bot; --poll; --backfill N
│   ├── config.py
│   ├── db.py               # schema + queries
│   ├── listener.py         # Telethon
│   ├── extractor.py        # regex + LLM extraction
│   ├── normalize.py        # email deobfuscation, URL resolve
│   ├── dedupe.py
│   ├── filter.py
│   ├── resume_picker.py
│   ├── email_drafter.py
│   ├── gmail_client.py
│   ├── link_handler.py     # classify + Playwright prefill
│   ├── approval_bot.py
│   ├── llm.py
│   └── prompts/
│       ├── extract.md
│       ├── fit.md
│       └── email.md
└── tests/
    ├── fixtures/posts/     # sample channel posts (.txt)
    └── test_*.py
```

---

## 5. Configuration

### `.env`
```
TG_API_ID=
TG_API_HASH=
TG_PHONE=
TG_SESSION_STRING=          # optional, for headless/cron hosting
TG_CHANNEL=                 # @username or numeric id
BOT_TOKEN=                  # approval bot
BOT_OWNER_CHAT_ID=          # only this chat id may use the bot
ANTHROPIC_API_KEY=
LLM_MODEL=
GMAIL_CREDENTIALS_PATH=credentials.json
GMAIL_TOKEN_PATH=token.json
```

### `config.yaml`
```yaml
target_batch: 2026
accept_roles: [ai_ml, software_engineering, backend, full_stack, data, product, sap, devops]
reject_keywords: [unpaid, "pay to apply", "registration fee", "security deposit"]
locations_ok: [remote, pune, mumbai, navi mumbai, bangalore, hyderabad, any]
min_stipend_monthly_inr: 0          # 0 = don't filter, only flag
recipient_cooldown_days: 7
send_spacing_seconds: [90, 240]     # random delay between sends
max_emails_per_day: 25
fit_threshold: 60                   # 0-100; below → skip, near → flag
```

### `profile.json` (example values; I'll fill in the real ones)
```json
{
  "name": "Vedant Patil",
  "email": "your.name@gmail.com",
  "phone": "+91XXXXXXXXXX",
  "college": "VIT Vellore",
  "degree": "B.Tech Computer Science",
  "grad_year": 2026,
  "cgpa": "8.49",
  "location": "Maharashtra, India",
  "github": "https://github.com/vedantp3",
  "portfolio": "https://vedantpatil.vercel.app",
  "linkedin": "https://linkedin.com/in/vedant-patil-8739a6240",
  "availability": "Immediate joiner; open to remote, hybrid or onsite",
  "notice_period": "Immediate",
  "expected_ctc": "As per company norms",
  "certifications": ["SAP Certified Generative AI Developer", "Google Cloud Generative AI", "Complete SQL Bootcamp"],
  "projects": [
    {"name": "Multimodal Math Mentor", "tags": ["ai_ml", "agents", "llm"], "stack": "", "proof": ""},
    {"name": "RAG Chatbot", "tags": ["ai_ml", "rag", "backend"], "stack": "", "proof": ""},
    {"name": "NutriChamp", "tags": ["ai_ml", "cv"], "stack": "", "proof": ""},
    {"name": "StockPilot", "tags": ["devops", "backend"], "stack": "", "proof": ""},
    {"name": "Property Rental System", "tags": ["full_stack"], "stack": "", "proof": ""},
    {"name": "Clara Pipeline", "tags": ["ai_ml", "data"], "stack": "", "proof": ""}
  ],
  "experience": [
    {"org": "Walstar Technologies", "role": "AI/ML & Software Engineering Intern", "proof": "RAG chatbot, OMR pipeline, Flask APIs"},
    {"org": "Fyaril AB (Sweden, remote)", "role": "Frontend Developer Intern", "proof": "Help Center portal revamp"},
    {"org": "Global IoT School", "role": "Assistant Manager / Emerging Technology Coordinator", "proof": "AI/IoT/Cybersecurity/SAP training partnerships across 3 districts"}
  ],
  "standard_answers": {
    "about_me": "",
    "why_this_company": "",
    "strengths": ""
  },
  "never_claim": ["reinforcement learning experience", "NSE stock analyzer project", "open-source contributions", "any project, link or metric not listed in this file"]
}
```
I'll fill in `stack` and `proof` (with metrics) per project. **The LLM may only use facts from this file.**

---

## 6. Database schema (SQLite)

```sql
raw_posts(
  id INTEGER PK, tg_message_id INTEGER UNIQUE, channel TEXT,
  text TEXT, posted_at TIMESTAMP, processed INTEGER DEFAULT 0
)

jobs(
  id INTEGER PK, raw_post_id INTEGER FK,
  job_hash TEXT UNIQUE,                -- sha1(normalized company + normalized role)
  company TEXT, role TEXT, role_category TEXT,
  batch_eligibility TEXT, location TEXT, work_mode TEXT,
  stipend_or_ctc TEXT, required_skills TEXT,   -- JSON array
  jd_summary TEXT, deadline TEXT,
  apply_method TEXT,                   -- 'email' | 'link' | 'both' | 'unknown'
  fit_score INTEGER, fit_reason TEXT,
  flags TEXT,                          -- JSON array of strings
  resume_track TEXT,
  status TEXT,                         -- see state machine below
  created_at TIMESTAMP, updated_at TIMESTAMP
)

contacts(
  id INTEGER PK, job_id INTEGER FK,
  kind TEXT,                           -- 'email' | 'link'
  value TEXT,                          -- normalized email or resolved URL
  raw_value TEXT, person_name TEXT, link_type TEXT
)

applications(
  id INTEGER PK, job_id INTEGER FK, contact_id INTEGER FK,
  gmail_draft_id TEXT, gmail_message_id TEXT,
  subject TEXT, body TEXT, resume_file TEXT,
  sent_at TIMESTAMP, approval_msg_id INTEGER
)

recipients(
  email TEXT PK, last_emailed_at TIMESTAMP, total_emails INTEGER
)
```

**Job status state machine:**
`extracted → skipped_filter | duplicate | flagged | drafted → awaiting_approval → sent | submitted | skipped_by_user | failed`
Later manual updates: `interview`, `rejected`, `offer`.

---

## 7. Module specs

### 7.1 listener.py
- Use a Telethon `events.NewMessage(chats=TG_CHANNEL)` handler. Store the post in `raw_posts`, then enqueue it for processing.
- Also handle `MessageEdited`, since channels often fix typos or links. Re-extract the post, but don't create duplicate approvals for a job that's already been sent.
- Also support `--backfill N`, which processes the last N posts once. This is useful for testing.
- Use the session file locally, or `StringSession(TG_SESSION_STRING)` when hosted.
- **Read-only.** The listener must never send, forward or react from the user account.
- On a FloodWait, sleep for the required time and continue.

### 7.2 normalize.py
- **Email deobfuscation** before matching: handle `[at]`, `(at)`, ` at `, `[dot]`, `(dot)`, ` dot `, and stray spaces around `@`/`.`. Lowercase emails, and strip trailing punctuation and emoji.
- **Email regex:** a standard RFC-ish pattern. Reject placeholder addresses like `example.com` and `xyz@`.
- **URL extraction:** catch plain URLs and Telegram entity URLs (`MessageEntityTextUrl`). Hyperlinked text hides its URL in the message entities, so read `message.entities`, not just the text.
- **URL resolution:** follow redirects with httpx (HEAD, falling back to GET, 10s timeout) for bit.ly, lnkd.in, forms.gle, tinyurl, t.co, and the like. Store both the raw and the resolved URL.

### 7.3 extractor.py
- Run regex first to get the candidate emails and URLs. Then call the LLM with the post text plus those candidates, using `prompts/extract.md`, which returns **strict JSON** validated by pydantic:
```json
{
  "jobs": [
    {
      "company": "", "role": "", "role_category": "ai_ml|software_engineering|backend|full_stack|data|product|sap|devops|other",
      "batch_eligibility": "e.g. 2025/2026, or null",
      "location": "", "work_mode": "remote|onsite|hybrid|null",
      "stipend_or_ctc": "", "required_skills": [], "jd_summary": "≤60 words",
      "deadline": null,
      "contacts": [{"kind": "email|link", "value": "", "person_name": null}],
      "post_quality_notes": []
    }
  ]
}
```
- One post can contain several jobs. The LLM must assign each contact to the correct job.
- `post_quality_notes` covers things like vague JD, inconsistent details, looks like two merged postings, or asks for a fee.
- **Guard:** every contact value in the LLM output must exist among the regex candidates. Drop anything the LLM invented.
- If validation fails, retry once with the validation error included in the prompt. If it fails again, mark the post `processed=-1` and notify me through the bot.

### 7.4 dedupe.py
- `job_hash` = sha1 of the lowercased company + role, with punctuation and words like "intern", "internship" and "hiring" removed. If the hash already exists, set status `duplicate` and attach any new contacts to the existing job.
- **Recipient cooldown:** if an email address was emailed within `recipient_cooldown_days`, don't draft a new email. Notify me instead: *"Already emailed priya@x.com on 18 Oct for SDE Intern; this post is for ML Intern. Combine / Send anyway / Skip?"*
- **Same recipient, multiple roles in one post:** draft **one** email that covers the best-fit role, mentioning a second role only if it also fits. Never send several emails to one person from one post.

### 7.5 filter.py
Apply hard rules first:
- Skip if the batch is explicit and excludes 2026.
- Skip if `reject_keywords` match. For fee or deposit requests, also flag the post as a likely scam.
- Skip if `role_category` isn't in `accept_roles`.

Then an LLM fit check (`prompts/fit.md`) takes the job and `profile.json` and returns `{fit_score, fit_reason, gaps: [], flags: []}`.

**Flags are shown to me, never hidden.** They include:
- Skill gaps.
- Batch or eligibility mismatch or ambiguity.
- An unusual stipend or CTC (very low, very high, or "performance-based only").
- A vague or inconsistent JD.
- A post that looks like two merged postings.
- Location conflicts.
- A personal Gmail address used as a "company HR" email. Just a heads-up, not a skip.

Results: below `fit_threshold` → `skipped_filter`; within 10 points of it → `flagged`, still sent to me with the flag shown; otherwise → proceed.

### 7.6 resume_picker.py
Tracks: `ai_ml`, `sde`, `product`, `sap`, `hybrid`.
- **Default to `ai_ml`** when in doubt.
- Use `sde` when the role is clearly backend, full-stack or DevOps scoped.
- Use `product` for PM, APM or Founder's Office roles.
- Use `sap` for SAP, BTP or S/4HANA roles.
- Use `hybrid` for generalist roles that mix AI and SWE.

Return the track plus a one-line reason, which is shown on the approval card. I can override the choice from the card.

### 7.7 email_drafter.py — **follow these rules exactly**
Prompt: `prompts/email.md`. Inputs: job, contact (with `person_name`), the `profile.json` projects and experience, and the chosen resume track.

**Format rules:**
- **Subject:** `[Role] — Vedant Patil (VIT Vellore, 2026)`
- **Greeting:** `Hi <FirstName>,` if the post gave a name, otherwise `Hi <Company> Team,`
- **Body order:**
  1. Open with proof, not intent. Name 1–2 specific projects or experience entries from `profile.json` matched to the JD, with their stack and **at least one metric**.
  2. Name the company's exact stack from the JD.
  3. **Gap line**, only if a required skill is missing: one honest sentence framing it as transferable plus a fast learner. Never pretend to have the skill.
  4. A closing tech-stack coverage line.
  5. Grad line, remote/availability, GitHub, portfolio, "Resume attached."
  6. Sign-off: `Best,\nVedant Patil\n+91…`

**Hard constraints** (enforce these in code with a validator after generation, and regenerate on any failure, up to 2 times):
- Under 150 words, excluding the signature.
- No bullet points or numbered lists.
- Must not start with "I am writing to apply" or any variant (check case-insensitively for "writing to apply" and "I am interested in applying").
- At least one digit-bearing metric present.
- Every project or employer named must exist in `profile.json`. Check this with string matching against the names.
- No links other than GitHub, portfolio and LinkedIn from `profile.json`.
- Nothing from `never_claim`.

Then create a **Gmail draft** with the chosen resume PDF attached and store the `gmail_draft_id`.

### 7.8 gmail_client.py
- OAuth installed-app flow on first run, then a refresh token stored in `token.json`.
- `create_draft(to, subject, body, attachment_path) -> draft_id`
- `send_draft(draft_id) -> message_id`
- `update_draft(draft_id, …)`
- **Rate limiting:** a send queue with a random delay in `send_spacing_seconds` between sends, and a hard cap of `max_emails_per_day`. If the cap is hit, hold the queue and tell me.
- Update `recipients` after every send.

### 7.9 link_handler.py
Classify the resolved URL by domain:

| `link_type` | Domains | Action |
|---|---|---|
| `google_form` | docs.google.com/forms, forms.gle | Playwright prefill (below) |
| `lever` | jobs.lever.co | Prefill attempt (name/email/phone/links/resume upload) |
| `greenhouse` | boards.greenhouse.io, job-boards.greenhouse.io | Prefill attempt |
| `ashby` | jobs.ashbyhq.com | Prefill attempt |
| `linkedin` | linkedin.com | Queue only |
| `naukri` / `workday` / `other_portal` | naukri.com, myworkdayjobs.com, others | Queue only |
| `whatsapp` / `telegram` | wa.me, t.me | Queue only, flagged "chat-based apply" |

**Prefill behaviour:**
- Run Playwright in **headed** mode on my machine with a persistent browser context, so Google sign-in survives between runs.
- Match fields by label text against a field map built from `profile.json`: name, email, phone, college, degree, grad year, CGPA, GitHub, LinkedIn, portfolio, location, notice period, expected CTC. Upload the resume to file inputs.
- For free-text questions ("Why do you want to join?"), generate an answer with the LLM using job plus profile, under 80 words, following the same never-claim rules. Fill it in, but highlight it for my review.
- **Never click Submit.** Leave the tab open for me to review and submit. Afterwards I tap "Mark submitted" on the bot card.
- If prefill fails, fall back to queue-only.

**Queue-only output:** a bot card with the link plus an "answer sheet" message. The sheet contains my standard fields and the LLM-drafted answers to that portal's common questions, formatted for one-tap copy (each value in its own `code` block).

### 7.10 approval_bot.py
- Respond only to `BOT_OWNER_CHAT_ID` and ignore everyone else.
- **Card for an email job:**
  ```
  📧 ML Engineer Intern — Acme AI
  Fit 82 · Resume: ai_ml (RAG + agents role)
  To: priya@acme.ai (Priya)
  ⚠️ Flags: JD asks for PyTorch Lightning (gap); stipend not stated
  ───
  Subject: …
  <full email body>
  ```
  Buttons: `✅ Send` `✏️ Edit` `📄 Resume ▾` `❌ Skip`
- **Card for a link job:** role, company, fit, flags, link type, link. Buttons: `🖥 Prefill` (google_form/lever/greenhouse/ashby) or `📋 Answer sheet`, plus `✔️ Mark submitted` and `❌ Skip`.
- **Edit flow:** after ✏️, my next text message is treated as an instruction ("shorter", "mention StockPilot instead", or a fully rewritten body). The LLM revises the draft, the validator re-runs, the Gmail draft is updated, and a new card is posted.
- **Commands:**
  - `/pending`: list jobs awaiting approval.
  - `/stats`: counts by status for today, this week, and all time.
  - `/status <job_id> interview|rejected|offer`
  - `/export`: send the tracker as CSV.
  - `/pause` and `/resume`: stop or restart processing.
  - `/backfill N`
- Post an error summary to the bot if any stage crashes. One bad post must never stop the listener.

---

## 8. Prompts (put in `prompts/`, keep editable)

- **extract.md:** instructions plus the JSON schema from §7.3. Tell the model: "Only use emails and URLs from the CANDIDATES list. If a field isn't stated, use null; don't guess. Split multi-role posts into separate jobs."
- **fit.md:** takes the job plus profile and returns a score, reason, gaps and flags. Tell the model to be conservative and to flag rather than assume.
- **email.md:** the rules from §7.7 verbatim, plus the full profile as the only allowed source of facts. Include 2 good example emails that I'll write later; leave a placeholder section for them.

---

## 9. Safety and honesty rules (enforce in code, not just prompts)

1. Nothing is sent or submitted without an explicit button tap from the owner chat.
2. No fabricated projects, links, metrics, open-source work or experience. Validate output against `profile.json`.
3. Never mention the NSE stock analyzer project. Never claim RL experience.
4. One email per recipient per post. Respect the cooldown. Space sends out and cap them per day.
5. The Telethon user account is read-only.
6. Secrets and personal files are gitignored. Never log full email bodies or tokens at INFO level.

---

## 10. Testing

- Add `tests/fixtures/posts/`. I'll paste 5–10 real channel posts there. Also create synthetic ones covering:
  - a single job with an email
  - a single job with a Google Form link
  - a multi-role post with different emails per role
  - obfuscated emails (`hr [at] acme [dot] com`)
  - a hyperlinked "Apply here" (URL only in entities)
  - a shortened link
  - a batch-mismatch post (2024/2025 only)
  - a "registration fee" scam post
  - a repost of an existing job (dedupe)
  - the same recipient across two posts (cooldown)
- Unit test `normalize`, `dedupe`, the `filter` hard rules, and the email validator.
- LLM-dependent tests are marked and run against recorded responses by default.
- Add a `--dry-run` flag: run the full pipeline with no Gmail calls and no Playwright, printing cards to the console.

---

## 11. Hosting

- **Primary:** an always-on process on my laptop or a free VM (Oracle Cloud Always Free or GCP e2-micro), run with `systemd` or `pm2`. Playwright prefill needs a desktop, so link-prefill only works when running locally. On the VM, link jobs fall back to answer sheets automatically.
- **Alternative:** `--poll` mode fetches messages newer than the last stored `tg_message_id`, processes them, and exits. This fits a GitHub Actions cron every 15–30 minutes. In this mode the approval bot must run separately (on the VM), or use a webhook.
- Document both options in the README.

---

## 12. Build order (stop for review after each phase)

1. **Skeleton:** layout, config, DB schema, `.env.example`, README setup steps.
2. **Ingest:** Telethon listener plus `--backfill`, normalize, extractor with guard, tested on fixtures in `--dry-run`.
3. **Decide:** dedupe, filter, resume picker, tested on fixtures.
4. **Email path:** drafter, validator, Gmail draft and send, rate limiter.
5. **Approval bot:** cards, buttons, edit flow, commands. End-to-end with email jobs.
6. **Link path:** classifier, answer sheets, Playwright prefill for Google Forms, then Lever, Greenhouse and Ashby.
7. **Hardening:** error notifications, edited-message handling, `/export`, hosting docs.

## 13. Definition of done

- A new channel post produces an approval card within 60 seconds.
- Email cards contain a validated email under 150 words with the correct resume attached as a Gmail draft. One tap sends it.
- Google Form links open prefilled, with nothing submitted.
- Duplicates and cooldown recipients never produce a second email.
- All fixtures pass in `--dry-run`.
