import httpx
import pytest

from jobpilot.normalize import (
    clean_url, deobfuscate, extract_candidates, extract_emails, extract_urls, is_shortened,
    normalize_email, resolve_url, url_key,
)


def emails(text):
    return [c.value for c in extract_emails(text)]


@pytest.mark.parametrize("text, expected", [
    ("Mail hr@acme.com now", ["hr@acme.com"]),
    ("hr [at] acme [dot] com", ["hr@acme.com"]),
    ("hr(at)acme(dot)co(dot)in", ["hr@acme.co.in"]),
    ("hr {at} acme {dot} io", ["hr@acme.io"]),
    ("careers at zentrix dot ai", ["careers@zentrix.ai"]),
    ("hr @ acme . com", ["hr@acme.com"]),
    ("Priya.Sharma@Acme.AI.", ["priya.sharma@acme.ai"]),
    ("📩hr@acme.com📩", ["hr@acme.com"]),
    ("(hr@acme.com),", ["hr@acme.com"]),
    ("a@x.com and A@X.com", ["a@x.com"]),
])
def test_email_forms(text, expected):
    assert emails(text) == expected


@pytest.mark.parametrize("text", [
    "Apply at Google today",                 # bare 'at' with no domain
    "meet at 5 pm dot com",                  # nonsense
    "write to xyz@company.com",              # placeholder local + domain
    "your.name@example.com",                 # placeholder domain
    "join us at the office",
    "Apply at careers.acme.com/jobs",        # prose 'at' before a real domain is not an email
    "We are hiring at acme.io",
])
def test_no_false_emails(text):
    assert emails(text) == []


def test_bare_at_needs_known_tld():
    assert deobfuscate("we are at home dot thing") == "we are at home dot thing"


def test_normalize_email_rejects_garbage():
    assert normalize_email("mailto:HR@Acme.com?subject=hi") == "hr@acme.com"
    assert normalize_email("not-an-email") is None
    assert normalize_email("xyz@acme.com") is None


def test_urls_plain_bare_and_trailing_punct():
    urls = [c.value for c in extract_urls(
        "Apply: https://jobs.lever.co/acme/123). Form forms.gle/AbC123, www.acme.ai/careers!"
    )]
    assert urls == [
        "https://jobs.lever.co/acme/123",
        "https://forms.gle/AbC123",
        "https://www.acme.ai/careers",
    ]


def test_bare_domain_with_path_but_not_emails_or_js():
    urls = [c.value for c in extract_urls(
        "Apply at careers.acme.com/jobs/42 or mail hr@acme.com/x. Stack: Node.js/React"
    )]
    assert urls == ["https://careers.acme.com/jobs/42"]


def test_url_keeps_balanced_parens():
    assert clean_url("https://en.wikipedia.org/wiki/Foo_(bar)") == "https://en.wikipedia.org/wiki/Foo_(bar)"


def test_entity_urls_and_mailto():
    cands = extract_candidates("👉 Apply here or email us", ["https://jobs.lever.co/k/1", "mailto:HR@k.io"])
    assert [(c.kind, c.value) for c in cands] == [("email", "hr@k.io"), ("link", "https://jobs.lever.co/k/1")]


def test_url_dedupe_by_key():
    cands = extract_urls("https://acme.ai/jobs/ and http://www.acme.ai/jobs")
    assert len(cands) == 1
    assert url_key("https://WWW.acme.ai/jobs/") == url_key("acme.ai/jobs")


def test_is_shortened():
    assert is_shortened("https://bit.ly/x") and is_shortened("https://forms.gle/x")
    assert not is_shortened("https://docs.google.com/forms/d/x")


@pytest.mark.asyncio
async def test_resolve_follows_redirects():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "bit.ly":
            return httpx.Response(301, headers={"Location": "https://docs.google.com/forms/d/e/abc/viewform"})
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as c:
        assert await resolve_url("https://bit.ly/x", c) == "https://docs.google.com/forms/d/e/abc/viewform"
        # non-shortener: untouched, no request made
        assert await resolve_url("https://acme.ai/jobs", c) == "https://acme.ai/jobs"


@pytest.mark.asyncio
async def test_resolve_head_rejected_falls_back_to_get():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "HEAD":
            return httpx.Response(405)
        if req.url.host == "lnkd.in":
            return httpx.Response(302, headers={"Location": "https://www.linkedin.com/jobs/view/1"})
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as c:
        assert await resolve_url("https://lnkd.in/abc", c) == "https://www.linkedin.com/jobs/view/1"


@pytest.mark.asyncio
async def test_resolve_failure_returns_original():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        assert await resolve_url("https://bit.ly/x", c) == "https://bit.ly/x"
