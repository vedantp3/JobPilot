"""Email deobfuscation, email/URL extraction, short-link resolution."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx

log = logging.getLogger(__name__)

# --- emails -------------------------------------------------------------------
# TLDs accepted for the risky bare-word form "hr at acme dot com".
_BARE_TLDS = (
    "com|in|io|ai|co|org|net|dev|tech|app|us|uk|xyz|me|so|gg|edu|ac|info|biz|"
    "cloud|careers|jobs|work|global|world|sg|de|se|eu|ca|au"
)
_AT_BRACKET = r"\s*(?:\[\s*at\s*\]|\(\s*at\s*\)|\{\s*at\s*\}|<\s*at\s*>)\s*"
_DOT_BRACKET = r"\s*(?:\[\s*dot\s*\]|\(\s*dot\s*\)|\{\s*dot\s*\}|<\s*dot\s*>)\s*"

_EMAIL_RE = re.compile(
    r"(?<![\w.+-])([a-z0-9][a-z0-9._%+-]{0,63})@((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})(?![\w-])",
    re.I,
)

PLACEHOLDER_DOMAINS = {
    "example.com", "example.org", "example.net", "example.in", "domain.com", "email.com",
    "test.com", "yourcompany.com", "company.com", "xyz.com", "abc.com", "sample.com",
}
PLACEHOLDER_LOCALS = {
    "xyz", "abc", "yourname", "your.name", "name", "email", "youremail", "your.email",
    "username", "user", "test", "sample", "firstname.lastname", "firstname",
}

# emoji and other pictographs that channels glue onto addresses
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]"
)


def deobfuscate(text: str) -> str:
    """Rewrite obfuscated email addresses into plain form.

    Handles [at]/(at)/{at}, [dot]/(dot), ' at '/' dot ' (only when the result looks
    like a real domain), and stray spaces around '@' and '.' in addresses.
    """
    t = _EMOJI_RE.sub(" ", text)
    # bracketed forms are unambiguous
    t = re.sub(_AT_BRACKET, "@", t, flags=re.I)
    t = re.sub(_DOT_BRACKET, ".", t, flags=re.I)
    # bare "name at domain dot tld": only with the spelled-out word 'dot' and a known TLD,
    # so ordinary prose like "apply at careers.acme.com" is left alone
    t = re.sub(
        rf"\b([a-z0-9][a-z0-9._%+-]*)\s+at\s+([a-z0-9-]+(?:\s+dot\s+[a-z0-9-]+)*\s+dot\s+(?:{_BARE_TLDS}))\b",
        lambda m: m.group(1) + "@" + re.sub(r"\s+dot\s+", ".", m.group(2), flags=re.I),
        t,
        flags=re.I,
    )
    # "hr @ acme . com" -> "hr@acme.com"
    t = re.sub(r"(?<=[\w.+-])\s*@\s*(?=[a-z0-9])", "@", t, flags=re.I)
    t = re.sub(
        r"@([a-z0-9-]+(?:\s*\.\s*[a-z0-9-]+)+)",
        lambda m: "@" + re.sub(r"\s*\.\s*", ".", m.group(1)),
        t,
        flags=re.I,
    )
    return t


def is_placeholder(email: str) -> bool:
    local, _, domain = email.partition("@")
    return domain in PLACEHOLDER_DOMAINS or local in PLACEHOLDER_LOCALS


def normalize_email(raw: str) -> str | None:
    """Lowercase, strip mailto/trailing punctuation; None if invalid or a placeholder."""
    s = _EMOJI_RE.sub("", raw).strip().lower()
    s = s.removeprefix("mailto:").split("?")[0]
    s = s.strip(" \t\r\n.,;:!?'\"()[]{}<>*_~`|")
    if not _EMAIL_RE.fullmatch(s) or is_placeholder(s):
        return None
    return s


@dataclass(frozen=True)
class Candidate:
    kind: str        # "email" | "link"
    value: str       # normalized email, or cleaned URL (unresolved)
    raw_value: str


def extract_emails(text: str) -> list[Candidate]:
    out: dict[str, Candidate] = {}
    for m in _EMAIL_RE.finditer(deobfuscate(text)):
        raw = m.group(0)
        norm = normalize_email(raw)
        if norm and norm not in out:
            out[norm] = Candidate("email", norm, raw)
    return list(out.values())


# --- URLs ---------------------------------------------------------------------
SHORTENER_HOSTS = {
    "bit.ly", "lnkd.in", "forms.gle", "tinyurl.com", "t.co", "rb.gy", "cutt.ly",
    "shorturl.at", "is.gd", "ow.ly", "buff.ly", "rebrand.ly", "tiny.cc", "s.id",
    "bitly.com", "goo.gl", "surl.li", "linktr.ee",
}
# bare hosts that are commonly posted without a scheme
_BARE_HOSTS = (
    r"forms\.gle|bit\.ly|lnkd\.in|tinyurl\.com|rb\.gy|cutt\.ly|shorturl\.at|t\.me|wa\.me|"
    r"docs\.google\.com|jobs\.lever\.co|boards\.greenhouse\.io|job-boards\.greenhouse\.io|"
    r"jobs\.ashbyhq\.com|(?:www\.)?linkedin\.com|(?:www\.)?naukri\.com|[a-z0-9-]+\.myworkdayjobs\.com"
)
_URL_RE = re.compile(
    rf"(?:https?://|www\.)[^\s<>\"'`]+"
    rf"|\b(?:{_BARE_HOSTS})/[^\s<>\"'`]*"
    # any other bare domain, but only with a path (avoids 'Node.js', and the domain part of emails)
    rf"|(?<![@\w.-])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:{_BARE_TLDS})/[^\s<>\"'`]*",
    re.I,
)
_TRAILING = ".,;:!?'\")]}>*_~|"


def clean_url(raw: str) -> str | None:
    """Trim trailing punctuation/emoji, add a scheme, lowercase the host."""
    s = _EMOJI_RE.sub("", raw).strip()
    # balance a trailing ')' that belongs to surrounding text
    while s and s[-1] in _TRAILING:
        if s[-1] == ")" and s.count("(") >= s.count(")"):
            break
        s = s[:-1]
    if not s:
        return None
    if s.lower().startswith("mailto:"):
        return None
    if not re.match(r"https?://", s, re.I):
        s = "https://" + s
    parts = urlsplit(s)
    if not parts.netloc or "." not in parts.netloc:
        return None
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, parts.fragment))


def url_key(url: str) -> str:
    """Comparison key: no scheme, no www., no trailing slash, lowercase host."""
    p = urlsplit(url if "://" in url else "https://" + url)
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/")
    return f"{host}{path}" + (f"?{p.query}" if p.query else "")


def extract_urls(text: str, entity_urls: list[str] = ()) -> list[Candidate]:
    """URLs from the visible text plus hidden hyperlink URLs from message entities."""
    out: dict[str, Candidate] = {}
    for raw in [m.group(0) for m in _URL_RE.finditer(text)] + list(entity_urls):
        if raw.lower().startswith("mailto:"):
            continue
        url = clean_url(raw)
        if url and url_key(url) not in out:
            out[url_key(url)] = Candidate("link", url, raw)
    return list(out.values())


def extract_candidates(text: str, entity_urls: list[str] = ()) -> list[Candidate]:
    emails = extract_emails(text)
    # mailto: links hidden behind text are emails, not URLs
    for u in entity_urls:
        if u.lower().startswith("mailto:"):
            norm = normalize_email(u)
            if norm and norm not in {c.value for c in emails}:
                emails.append(Candidate("email", norm, u))
    return emails + extract_urls(text, entity_urls)


# --- resolution ---------------------------------------------------------------
def is_shortened(url: str) -> bool:
    return urlsplit(url).netloc.lower().removeprefix("www.") in SHORTENER_HOSTS


async def resolve_url(url: str, client: httpx.AsyncClient | None = None, timeout: float = 10.0) -> str:
    """Follow redirects for known shorteners (HEAD, falling back to GET).

    Returns the final URL, or the input unchanged if it isn't a shortener or
    resolution fails.
    """
    if not is_shortened(url):
        return url
    own = client is None
    client = client or httpx.AsyncClient(
        follow_redirects=True, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (JobPilot link resolver)"},
    )
    try:
        for method in ("HEAD", "GET"):
            try:
                resp = await client.request(method, url, follow_redirects=True, timeout=timeout)
                final = str(resp.url)
                if resp.status_code < 400 and final != url:
                    return clean_url(final) or url
                if resp.status_code < 400 and method == "GET":
                    return url
            except httpx.HTTPError as e:
                log.debug("resolve %s %s failed: %s", method, url, e)
        return url
    finally:
        if own:
            await client.aclose()


async def resolve_all(urls: list[str], client: httpx.AsyncClient | None = None) -> dict[str, str]:
    results = await asyncio.gather(*(resolve_url(u, client) for u in urls))
    return dict(zip(urls, results))
