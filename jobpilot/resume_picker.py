"""Choose a resume track (ai_ml, sde, product, sap, hybrid) with a one-line reason.

Rule-based and deterministic; the approval card lets the user override it.
"""
from __future__ import annotations

import re

_AI = re.compile(
    r"\b(?:ai|ml|ai/ml|machine learning|deep learning|llms?|genai|gen ai|generative|rag|nlp|"
    r"computer vision|pytorch|tensorflow|langchain|langgraph|agents?|agentic|data scien\w*|"
    r"mlops|hugging ?face|transformers?)\b",
    re.I,
)
_PRODUCT = re.compile(
    r"\b(?:product manager|product management|apm|associate product|founder'?s office|"
    r"product analyst|product intern|pm intern|chief of staff)\b",
    re.I,
)
_SAP = re.compile(r"\b(?:sap|btp|s/?4 ?hana|abap|fiori)\b", re.I)
_ENGINEERING = ("backend", "full_stack", "devops")


def pick_resume(job: dict) -> tuple[str, str]:
    """Return (track, reason) for a job row."""
    role = job.get("role") or ""
    skills = " ".join(job.get("required_skills") or [])
    summary = job.get("jd_summary") or ""
    cat = job.get("role_category")

    if cat == "sap" or _SAP.search(role):
        return "sap", "SAP/BTP role"
    if cat == "product" or _PRODUCT.search(role):
        return "product", "product / Founder's Office role"

    ai_in_role = bool(_AI.search(role))
    ai_anywhere = ai_in_role or bool(_AI.search(skills)) or bool(_AI.search(summary))

    if cat in _ENGINEERING:
        if ai_in_role:
            return "hybrid", f"{cat.replace('_', '-')} role with AI work"
        return "sde", f"clearly {cat.replace('_', '-')} scoped"
    if cat == "software_engineering":
        if ai_anywhere:
            return "hybrid", "generalist SWE role that mixes in AI/ML"
        return "sde", "general software engineering role"
    if cat == "ai_ml":
        if _AI.search(role) is None and re.search(r"\b(?:backend|full ?stack|software engineer)\b", role, re.I):
            return "hybrid", "AI role framed as software engineering"
        return "ai_ml", "AI/ML role"
    if cat == "data" and ai_anywhere:
        return "ai_ml", "data role with ML/AI work"
    return "ai_ml", "default track (role isn't clearly SDE, product or SAP)"
