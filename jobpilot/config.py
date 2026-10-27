"""Configuration: secrets from .env, settings from config.yaml, facts from profile.json."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# OpenAI (paid, high limits) first; Groq and Gemini free tiers as fallbacks; Anthropic optional.
DEFAULT_LLM_MODELS = {
    "openai": "gpt-6-luna",
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-5-5",
}
LLM_PROVIDERS = tuple(DEFAULT_LLM_MODELS)

ROLE_CATEGORIES = (
    "ai_ml", "software_engineering", "backend", "full_stack",
    "data", "product", "sap", "devops", "other",
)
RESUME_TRACKS = ("ai_ml", "sde", "product", "sap", "hybrid")


class Secrets(BaseSettings):
    """Values from .env (or the process environment)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    tg_api_id: int | None = None
    tg_api_hash: str = ""
    tg_phone: str = ""
    tg_session_string: str = ""
    tg_channel: str = ""
    bot_token: str = ""
    bot_owner_chat_id: int | None = None
    # One provider, or a fallback chain like "openai,groq,gemini": a call that hits a rate limit,
    # quota or outage on one provider moves on to the next. Providers without a key are skipped.
    llm_provider: str = "openai,groq,gemini"
    openai_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = ""          # goes to the provider its name belongs to (gpt-* -> openai, gemini-* -> gemini...)
    openai_model: str = ""
    groq_model: str = ""
    gemini_model: str = ""
    anthropic_model: str = ""
    gmail_credentials_path: str = "credentials.json"
    gmail_token_path: str = "token.json"

    @field_validator("tg_api_id", "bot_owner_chat_id", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return None if v in ("", None) else v

    @field_validator("llm_provider")
    @classmethod
    def _known_providers(cls, v: str) -> str:
        names = [p.strip().lower() for p in (v or "openai,groq,gemini").split(",") if p.strip()]
        unknown = [p for p in names if p not in LLM_PROVIDERS]
        if unknown or not names:
            raise ValueError(f"LLM_PROVIDER entries must be from {LLM_PROVIDERS}, got {unknown or v!r}")
        return ",".join(dict.fromkeys(names))

    @property
    def channels(self) -> list[str]:
        """TG_CHANNEL may list several channels: '@jobs_a, @jobs_b, -1001234567890'."""
        return list(dict.fromkeys(c.strip() for c in re.split(r"[,\s]+", self.tg_channel or "") if c.strip()))

    @property
    def providers(self) -> list[str]:
        return self.llm_provider.split(",")

    @property
    def llm_model_provider(self) -> str | None:
        """Which provider LLM_MODEL belongs to: inferred from its name, else the first provider."""
        if not self.llm_model:
            return None
        m = self.llm_model.lower()
        if m.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai"
        if m.startswith("gemini"):
            return "gemini"
        if m.startswith("claude"):
            return "anthropic"
        return "groq" if "groq" in self.providers and "/" in m or m.startswith(("llama", "qwen")) \
            else self.providers[0]

    def model_for(self, provider: str) -> str:
        specific = getattr(self, f"{provider}_model")
        if specific:
            return specific
        if self.llm_model and self.llm_model_provider == provider:
            return self.llm_model
        return DEFAULT_LLM_MODELS[provider]

    @property
    def model(self) -> str:
        """Model of the primary provider."""
        return self.model_for(self.providers[0])

    @property
    def llm_key_names(self) -> list[str]:
        return [f"{p}_api_key" for p in self.providers]

    def missing(self, *names: str) -> list[str]:
        """Return the env var names (upper-case) among `names` that are unset."""
        return [n.upper() for n in names if getattr(self, n) in ("", None)]


class AppConfig(BaseModel):
    """Values from config.yaml."""

    target_batch: int = 2026
    accept_roles: list[str] = Field(default_factory=lambda: list(ROLE_CATEGORIES[:-1]))
    reject_keywords: list[str] = Field(default_factory=list)
    locations_ok: list[str] = Field(default_factory=list)
    min_stipend_monthly_inr: int = 0
    recipient_cooldown_days: int = 7
    send_spacing_seconds: tuple[int, int] = (90, 240)
    max_emails_per_day: int = 25
    fit_threshold: int = Field(60, ge=0, le=100)
    llm_min_interval_seconds: float = 4.0   # spacing between LLM calls (free-tier RPM limits)
    email_max_words: int = Field(150, ge=60, le=400)   # body + greeting, excluding the signature
    link_prefill: bool = True   # 🖥 Prefill button for Google Forms / Lever / Greenhouse / Ashby (needs a desktop)
    prefill_browser: str = "auto"   # auto (installed Chrome, then Edge, then bundled Chromium) | chrome | msedge | chromium

    db_path: str = "data/jobpilot.db"
    profile_path: str = "profile.json"
    resumes_dir: str = "resumes"
    prompts_dir: str = "jobpilot/prompts"
    browser_profile_dir: str = "browser_profile"

    @field_validator("accept_roles")
    @classmethod
    def _known_roles(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(ROLE_CATEGORIES)
        if unknown:
            raise ValueError(f"unknown role categories: {sorted(unknown)}")
        return v

    @field_validator("send_spacing_seconds")
    @classmethod
    def _ordered(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1] or v[0] < 0:
            raise ValueError("send_spacing_seconds must be [min, max] with 0 <= min <= max")
        return v

    def path(self, attr: str) -> Path:
        """Resolve a path setting against the project root."""
        p = Path(getattr(self, attr))
        return p if p.is_absolute() else PROJECT_ROOT / p

    def resume_file(self, track: str) -> Path:
        if track not in RESUME_TRACKS:
            raise ValueError(f"unknown resume track: {track}")
        return self.path("resumes_dir") / f"{track}.pdf"


class Project(BaseModel):
    name: str
    tags: list[str] = Field(default_factory=list)
    stack: str = ""
    proof: str = ""


class Experience(BaseModel):
    org: str
    role: str
    dates: str = ""
    proof: str = ""


class Profile(BaseModel):
    """profile.json: the only allowed source of facts for generated text."""

    name: str
    email: str
    phone: str
    college: str
    degree: str
    grad_year: int
    cgpa: str = ""
    location: str = ""
    github: str = ""
    portfolio: str = ""
    linkedin: str = ""
    availability: str = ""
    notice_period: str = ""
    expected_ctc: str = ""
    certifications: list[str] = Field(default_factory=list)
    skills: dict[str, list[str]] = Field(default_factory=dict)   # group -> skills, as on the resumes
    projects: list[Project] = Field(default_factory=list)
    featured_projects: list[str] = Field(default_factory=list)   # project names to bring up whenever relevant
    experience: list[Experience] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    standard_answers: dict[str, str] = Field(default_factory=dict)
    never_claim: list[str] = Field(default_factory=list)

    @field_validator("featured_projects")
    @classmethod
    def _featured_exist(cls, v: list[str], info) -> list[str]:
        names = {p.name.lower() for p in info.data.get("projects", [])}
        missing = [n for n in v if n.lower() not in names]
        if missing:
            raise ValueError(f"featured_projects not found in projects: {missing}")
        return v

    @property
    def first_name(self) -> str:
        return self.name.split()[0]

    @property
    def allowed_links(self) -> list[str]:
        return [u for u in (self.github, self.portfolio, self.linkedin) if u]


def load_config(path: Path | None = None) -> AppConfig:
    path = path or PROJECT_ROOT / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return AppConfig.model_validate(data or {})


def load_profile(path: Path | None = None) -> Profile:
    path = path or load_config().path("profile_path")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy profile.example.json to profile.json and fill it in."
        )
    return Profile.model_validate(json.loads(path.read_text(encoding="utf-8")))


@lru_cache
def get_secrets() -> Secrets:
    return Secrets()


@lru_cache
def get_config() -> AppConfig:
    return load_config()
