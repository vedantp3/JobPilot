"""Regex candidates + LLM structuring into jobs, with the contact guard."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from jobpilot.config import ROLE_CATEGORIES, get_config
from jobpilot.llm import LLM, LLMError, content_key
from jobpilot.normalize import Candidate, extract_candidates, normalize_email, url_key

log = logging.getLogger(__name__)

RoleCategory = Literal[
    "ai_ml", "software_engineering", "backend", "full_stack", "data", "product", "sap", "devops", "other"
]
assert set(RoleCategory.__args__) == set(ROLE_CATEGORIES)


class ExtractedContact(BaseModel):
    kind: Literal["email", "link"]
    value: str
    person_name: str | None = None


class ExtractedJob(BaseModel):
    company: str | None = None
    role: str
    role_category: RoleCategory = "other"
    batch_eligibility: str | None = None
    location: str | None = None
    work_mode: Literal["remote", "onsite", "hybrid"] | None = None
    stipend_or_ctc: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    jd_summary: str | None = None
    deadline: str | None = None
    contacts: list[ExtractedContact] = Field(default_factory=list)
    post_quality_notes: list[str] = Field(default_factory=list)

    @field_validator("role")
    @classmethod
    def _role_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("role must not be empty")
        return v.strip()

    @field_validator("jd_summary")
    @classmethod
    def _short_summary(cls, v: str | None) -> str | None:
        if v and len(v.split()) > 80:  # prompt asks for <=60; allow slack, then trim
            v = " ".join(v.split()[:60]) + "..."
        return v

    @property
    def apply_method(self) -> str:
        kinds = {c.kind for c in self.contacts}
        if kinds == {"email", "link"}:
            return "both"
        return kinds.pop() if kinds else "unknown"


class Extraction(BaseModel):
    jobs: list[ExtractedJob]


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


_STR = {"type": "string"}
EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["jobs"],
    "properties": {
        "jobs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "company", "role", "role_category", "batch_eligibility", "location", "work_mode",
                    "stipend_or_ctc", "required_skills", "jd_summary", "deadline", "contacts",
                    "post_quality_notes",
                ],
                "properties": {
                    "company": _nullable(_STR),
                    "role": _STR,
                    "role_category": {"type": "string", "enum": list(ROLE_CATEGORIES)},
                    "batch_eligibility": _nullable(_STR),
                    "location": _nullable(_STR),
                    "work_mode": _nullable({"type": "string", "enum": ["remote", "onsite", "hybrid"]}),
                    "stipend_or_ctc": _nullable(_STR),
                    "required_skills": {"type": "array", "items": _STR},
                    "jd_summary": _nullable(_STR),
                    "deadline": _nullable(_STR),
                    "contacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["kind", "value", "person_name"],
                            "properties": {
                                "kind": {"type": "string", "enum": ["email", "link"]},
                                "value": _STR,
                                "person_name": _nullable(_STR),
                            },
                        },
                    },
                    "post_quality_notes": {"type": "array", "items": _STR},
                },
            },
        }
    },
}


class ExtractionError(RuntimeError):
    pass


@dataclass
class ExtractResult:
    jobs: list[ExtractedJob]
    candidates: list[Candidate]
    dropped: list[str] = field(default_factory=list)   # contact values the guard removed


def load_prompt() -> str:
    return (get_config().path("prompts_dir") / "extract.md").read_text(encoding="utf-8")


def build_user_message(text: str, candidates: list[Candidate], error: str | None = None) -> str:
    cand = [{"kind": c.kind, "value": c.value} for c in candidates]
    msg = (
        f"POST:\n<<<\n{text}\n>>>\n\n"
        f"CANDIDATES (the only allowed contact values):\n{json.dumps(cand, indent=1)}"
    )
    if error:
        msg += (
            "\n\nYour previous answer failed validation with this error. "
            f"Return corrected JSON.\nERROR: {error}"
        )
    return msg


def _match_candidate(contact: ExtractedContact, candidates: list[Candidate]) -> Candidate | None:
    """Return the candidate this LLM contact refers to, or None if it was invented."""
    if contact.kind == "email":
        norm = normalize_email(contact.value)
        return next((c for c in candidates if c.kind == "email" and c.value == norm), None)
    key = url_key(contact.value)
    return next((c for c in candidates if c.kind == "link" and url_key(c.value) == key), None)


def apply_guard(extraction: Extraction, candidates: list[Candidate]) -> tuple[list[ExtractedJob], list[str]]:
    """Drop every contact whose value isn't among the regex candidates."""
    dropped: list[str] = []
    for job in extraction.jobs:
        kept: list[ExtractedContact] = []
        seen: set[str] = set()
        for c in job.contacts:
            match = _match_candidate(c, candidates)
            if match is None:
                dropped.append(c.value)
                log.warning("guard: dropped invented contact %r for %s", c.value, job.role)
                continue
            if match.value in seen:
                continue
            seen.add(match.value)
            kept.append(ExtractedContact(kind=match.kind, value=match.value, person_name=c.person_name))
        job.contacts = kept
    return extraction.jobs, dropped


async def extract(text: str, entity_urls: list[str], llm: LLM) -> ExtractResult:
    """Extract jobs from one post. Raises ExtractionError after one failed retry."""
    candidates = extract_candidates(text, entity_urls)
    system = load_prompt()
    key = content_key("extract", text, *sorted(entity_urls))
    error: str | None = None
    for attempt in range(2):
        try:
            raw = await llm.complete_json(
                system=system,
                user=build_user_message(text, candidates, error),
                schema=EXTRACTION_SCHEMA,
                key=key if attempt == 0 else f"{key}.retry",
                effort="low",
            )
            extraction = Extraction.model_validate(raw)
        except (ValidationError, LLMError) as e:
            error = str(e)[:2000]
            log.warning("extraction attempt %d failed: %s", attempt + 1, error.splitlines()[0])
            continue
        jobs, dropped = apply_guard(extraction, candidates)
        return ExtractResult(jobs=jobs, candidates=candidates, dropped=dropped)
    raise ExtractionError(f"extraction failed twice: {error}")
