import json

import pytest
from pydantic import ValidationError

from jobpilot.config import PROJECT_ROOT, AppConfig, load_config, load_profile


def test_repo_config_loads():
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    assert cfg.target_batch == 2026
    assert cfg.send_spacing_seconds == (90, 240)
    assert cfg.resume_file("ai_ml").name == "ai_ml.pdf"
    with pytest.raises(ValueError):
        cfg.resume_file("marketing")


def test_config_rejects_unknown_role():
    with pytest.raises(ValidationError):
        AppConfig(accept_roles=["ai_ml", "astronaut"])


def test_config_rejects_bad_spacing():
    with pytest.raises(ValidationError):
        AppConfig(send_spacing_seconds=(240, 90))


def test_example_profile_loads():
    p = load_profile(PROJECT_ROOT / "profile.example.json")
    assert p.first_name == "Vedant"
    assert "RAG Chatbot" in [x.name for x in p.projects]
    assert p.allowed_links == [p.github, p.portfolio, p.linkedin]


def test_missing_profile_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="profile.example.json"):
        load_profile(tmp_path / "profile.json")


def test_profile_requires_core_fields(tmp_path):
    f = tmp_path / "profile.json"
    f.write_text(json.dumps({"name": "X"}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_profile(f)
