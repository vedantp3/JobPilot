import pytest

from jobpilot.extractor import ExtractionError, extract
from jobpilot.llm import LLMError


class FakeLLM:
    """Returns queued responses; an Exception in the queue is raised."""

    name = "fake"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def complete_json(self, *, system, user, schema, key=None, max_tokens=8000, effort=None):
        self.calls.append(user)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def complete_text(self, **kw):
        raise NotImplementedError


def job(role="ML Intern", contacts=(), **kw):
    return {"company": "Acme", "role": role, "role_category": "ai_ml", "contacts": list(contacts), **kw}


POST = "Acme hiring ML Intern. Mail hr [at] acme [dot] ai or apply https://forms.gle/abc"


@pytest.mark.asyncio
async def test_guard_drops_invented_contacts():
    llm = FakeLLM({"jobs": [job(contacts=[
        {"kind": "email", "value": "HR@acme.ai", "person_name": None},          # real (case differs)
        {"kind": "email", "value": "ceo@acme.ai", "person_name": "Invented"},   # not in post
        {"kind": "link", "value": "forms.gle/abc/", "person_name": None},        # real (format differs)
        {"kind": "link", "value": "https://acme.ai/careers", "person_name": None},  # not in post
    ])]})
    res = await extract(POST, [], llm)
    assert [(c.kind, c.value) for c in res.jobs[0].contacts] == [
        ("email", "hr@acme.ai"), ("link", "https://forms.gle/abc"),
    ]
    assert sorted(res.dropped) == ["ceo@acme.ai", "https://acme.ai/careers"]
    assert res.jobs[0].apply_method == "both"
    # candidates were passed to the model
    assert "hr@acme.ai" in llm.calls[0] and "https://forms.gle/abc" in llm.calls[0]


@pytest.mark.asyncio
async def test_retry_includes_validation_error():
    llm = FakeLLM({"jobs": [{"company": "Acme"}]}, {"jobs": [job()]})
    res = await extract(POST, [], llm)
    assert len(res.jobs) == 1
    assert "failed validation" in llm.calls[1] and "role" in llm.calls[1]


@pytest.mark.asyncio
async def test_fails_after_second_error():
    llm = FakeLLM(LLMError("boom"), {"jobs": "nope"})
    with pytest.raises(ExtractionError):
        await extract(POST, [], llm)
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_no_jobs_is_valid():
    res = await extract("Happy Diwali everyone!", [], FakeLLM({"jobs": []}))
    assert res.jobs == []


@pytest.mark.asyncio
async def test_duplicate_contacts_collapsed_and_kind_mismatch_dropped():
    llm = FakeLLM({"jobs": [job(contacts=[
        {"kind": "email", "value": "hr@acme.ai", "person_name": None},
        {"kind": "email", "value": "hr@acme.ai", "person_name": None},
        {"kind": "link", "value": "hr@acme.ai", "person_name": None},
    ])]})
    res = await extract(POST, [], llm)
    assert [c.value for c in res.jobs[0].contacts] == ["hr@acme.ai"]
