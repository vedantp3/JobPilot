import pytest

from jobpilot.resume_picker import pick_resume


@pytest.mark.parametrize("role, cat, skills, track", [
    ("Machine Learning Intern", "ai_ml", ["PyTorch"], "ai_ml"),
    ("Backend Developer Intern", "backend", ["Node.js"], "sde"),
    ("Full Stack Developer", "full_stack", ["React"], "sde"),
    ("DevOps Intern", "devops", ["AWS"], "sde"),
    ("AI Backend Engineer", "backend", ["Python"], "hybrid"),
    ("Software Engineer Intern", "software_engineering", ["Java"], "sde"),
    ("Software Engineer Intern", "software_engineering", ["Python", "LLMs"], "hybrid"),
    ("Associate Product Manager", "product", [], "product"),
    ("Founder's Office Intern", "other", [], "product"),
    ("SAP BTP Developer (Fresher)", "sap", ["CAP"], "sap"),
    ("ABAP Consultant", "other", [], "sap"),
    ("Data Analyst Intern", "data", ["SQL"], "ai_ml"),
    ("Data Scientist", "data", ["Python"], "ai_ml"),
    ("Growth Intern", "other", [], "ai_ml"),
])
def test_pick_resume(role, cat, skills, track):
    got, reason = pick_resume({"role": role, "role_category": cat, "required_skills": skills})
    assert got == track and reason
