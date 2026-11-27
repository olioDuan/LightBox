"""Configuration & model catalog loading with default TOML templates (SPEC §3.1, §6.2)."""

from __future__ import annotations

import tomllib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lightbox.paths import AppPaths, get_app_paths

DEFAULT_CONFIG_TOML = """\
# Lightbox general configuration (never store plaintext API keys here)
language = "zh-CN"
sensitivity = "conservative"
model_profile = "balanced"
auto_supervisor = false

[budget]
session_budget_usd = 0.50
monthly_budget_usd = 5.00

[quarantine]
strategy = "outside_library"

[cache]
preview_max_gb = 2.0
"""

DEFAULT_MODELS_TOML = """\
# Lightbox model profiles and pricing catalog (SPEC §6.2)

[profiles.balanced]
embedder = "siglip2-base"
triage_vlm = "claude-haiku-5-5"
agent = "claude-sonnet-5-5"
reviewer = "gemini-2-5-flash"
consolidate = "claude-sonnet-5-5"

[profiles.best]
embedder = "siglip2-base"
triage_vlm = "claude-sonnet-5-5"
agent = "claude-opus-5-5"
reviewer = "gemini-2-5-pro"
consolidate = "claude-sonnet-5-5"

[profiles.local]
embedder = "siglip2-base"
triage_vlm = "mlx-vlm"
agent = "mlx-vlm"
reviewer = "none"
consolidate = "none"

[models.siglip2-base]
provider = "local"
protocol = "local_onnx"
family = "siglip2"
model_id = "google/siglip2-base-patch16-256"
vision = true
tools = false
json_schema = false
prompt_cache = false

[models.claude-haiku-5-5]
provider = "anthropic"
protocol = "anthropic"
family = "anthropic"
model_id = "claude-haiku-5-5"
default_effort = "low"
input_cost_per_mtok = 0.80
output_cost_per_mtok = 4.00
cached_input_cost_per_mtok = 0.08
vision = true
tools = true
json_schema = true
prompt_cache = true

[models.claude-sonnet-5-5]
provider = "anthropic"
protocol = "anthropic"
family = "anthropic"
model_id = "claude-sonnet-5-5"
default_effort = "medium"
input_cost_per_mtok = 3.00
output_cost_per_mtok = 15.00
cached_input_cost_per_mtok = 0.30
vision = true
tools = true
json_schema = true
prompt_cache = true

[models.claude-opus-5-5]
provider = "anthropic"
protocol = "anthropic"
family = "anthropic"
model_id = "claude-opus-5-5"
default_effort = "medium"
input_cost_per_mtok = 15.00
output_cost_per_mtok = 75.00
cached_input_cost_per_mtok = 1.50
vision = true
tools = true
json_schema = true
prompt_cache = true

[models.gemini-2-5-flash]
provider = "gemini"
protocol = "openai_compat"
family = "google"
model_id = "gemini-2.5-flash"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
input_cost_per_mtok = 0.15
output_cost_per_mtok = 0.60
cached_input_cost_per_mtok = 0.0375
vision = true
tools = true
json_schema = true
prompt_cache = true

[models.gemini-2-5-pro]
provider = "gemini"
protocol = "openai_compat"
family = "google"
model_id = "gemini-2.5-pro"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
input_cost_per_mtok = 1.25
output_cost_per_mtok = 10.00
cached_input_cost_per_mtok = 0.3125
vision = true
tools = true
json_schema = true
prompt_cache = true

[models.gpt-4o]
provider = "openai"
protocol = "openai_compat"
family = "openai"
model_id = "gpt-4o"
input_cost_per_mtok = 2.50
output_cost_per_mtok = 10.00
cached_input_cost_per_mtok = 1.25
vision = true
tools = true
json_schema = true
prompt_cache = true

[models.gpt-4o-mini]
provider = "openai"
protocol = "openai_compat"
family = "openai"
model_id = "gpt-4o-mini"
input_cost_per_mtok = 0.15
output_cost_per_mtok = 0.60
cached_input_cost_per_mtok = 0.075
vision = true
tools = true
json_schema = true
prompt_cache = true

[models.mlx-vlm]
provider = "local"
protocol = "openai_compat"
family = "local"
model_id = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
base_url = "http://127.0.0.1:8080/v1"
max_pixels = 1003520
input_cost_per_mtok = 0.0
output_cost_per_mtok = 0.0
cached_input_cost_per_mtok = 0.0
vision = true
tools = true
json_schema = true
prompt_cache = false

[models.ollama-vlm]
provider = "local"
protocol = "openai_compat"
family = "local"
model_id = "qwen2.5vl:7b"
base_url = "http://127.0.0.1:11434/v1"
input_cost_per_mtok = 0.0
output_cost_per_mtok = 0.0
cached_input_cost_per_mtok = 0.0
vision = true
tools = true
json_schema = true
prompt_cache = false
"""


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_budget_usd: float = Field(default=0.50, ge=0.0)
    monthly_budget_usd: float = Field(default=5.00, ge=0.0)


class QuarantineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["outside_library", "sibling", "subfolder", "custom"] = "outside_library"
    custom_dir: str | None = None


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preview_max_gb: float = Field(default=2.0, gt=0.0)


AIProviderName = Literal["anthropic", "openai", "gemini", "ollama", "custom"]


class ActiveAIConfig(BaseModel):
    """Global `[ai]` connection configuration (never stores plaintext API keys)."""

    model_config = ConfigDict(extra="forbid")

    provider: AIProviderName
    agent_model: str = Field(min_length=1)
    triage_model: str = Field(min_length=1)
    base_url: str | None = None
    reviewer: str = "none"
    is_local_free: bool = False
    input_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    output_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    triage_input_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    triage_output_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    max_session_calls: int = Field(default=20, ge=1, le=200)
    max_triage_photos_unpriced: int = Field(default=20, ge=1, le=500)

    @model_validator(mode="after")
    def _validate_paired_costs(self) -> ActiveAIConfig:
        if (self.input_cost_per_mtok is None) != (self.output_cost_per_mtok is None):
            raise ValueError(
                "input_cost_per_mtok and output_cost_per_mtok must both be set or both omitted."
            )
        if (self.triage_input_cost_per_mtok is None) != (self.triage_output_cost_per_mtok is None):
            raise ValueError(
                "triage_input_cost_per_mtok and triage_output_cost_per_mtok must both be set or both omitted."
            )
        return self


class AppConfig(BaseModel):
    """Typed representation of `<APP_DATA_DIR>/config.toml`.

    Uses `extra="forbid"` so plaintext API keys or unknown settings fail fast.
    """

    model_config = ConfigDict(extra="forbid")

    language: Literal["zh-CN", "en-US"] = "zh-CN"
    sensitivity: Literal["conservative", "standard"] = "conservative"
    model_profile: Literal["balanced", "best", "local"] = "balanced"
    auto_supervisor: bool = False
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    quarantine: QuarantineConfig = Field(default_factory=QuarantineConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ai: ActiveAIConfig | None = None


class ProfileRoles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedder: str
    triage_vlm: str
    agent: str
    reviewer: str
    consolidate: str


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "openai", "gemini", "local", "custom"]
    protocol: Literal["anthropic", "openai_compat", "local_onnx"]
    family: str
    model_id: str
    base_url: str | None = None
    default_effort: Literal["low", "medium", "high"] | None = None
    max_pixels: int | None = Field(default=None, gt=0)
    input_cost_per_mtok: float = Field(default=0.0, ge=0.0)
    output_cost_per_mtok: float = Field(default=0.0, ge=0.0)
    cached_input_cost_per_mtok: float = Field(default=0.0, ge=0.0)
    vision: bool = True
    tools: bool = True
    json_schema: bool = True
    prompt_cache: bool = False


class ModelsConfig(BaseModel):
    """Typed representation of `<APP_DATA_DIR>/models.toml`."""

    model_config = ConfigDict(extra="forbid")

    profiles: dict[str, ProfileRoles]
    models: dict[str, ModelSpec]


def load_config(paths: AppPaths | None = None, *, write_default: bool = False) -> AppConfig:
    """Load `config.toml` from `<APP_DATA_DIR>`, falling back to defaults."""
    resolved = paths or get_app_paths()
    if resolved.config_file.exists():
        raw = tomllib.loads(resolved.config_file.read_text(encoding="utf-8"))
        return AppConfig.model_validate(raw)

    if write_default:
        resolved.ensure_dirs()
        resolved.config_file.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")

    return AppConfig.model_validate(tomllib.loads(DEFAULT_CONFIG_TOML))


def load_models_config(
    paths: AppPaths | None = None, *, write_default: bool = False
) -> ModelsConfig:
    """Load `models.toml` from `<APP_DATA_DIR>`, falling back to built-in catalog."""
    resolved = paths or get_app_paths()
    if resolved.models_file.exists():
        raw = tomllib.loads(resolved.models_file.read_text(encoding="utf-8"))
        return ModelsConfig.model_validate(raw)

    if write_default:
        resolved.ensure_dirs()
        resolved.models_file.write_text(DEFAULT_MODELS_TOML, encoding="utf-8")

    return ModelsConfig.model_validate(tomllib.loads(DEFAULT_MODELS_TOML))


__all__ = [
    "AIProviderName",
    "ActiveAIConfig",
    "AppConfig",
    "BudgetConfig",
    "CacheConfig",
    "DEFAULT_CONFIG_TOML",
    "DEFAULT_MODELS_TOML",
    "ModelSpec",
    "ModelsConfig",
    "ProfileRoles",
    "QuarantineConfig",
    "load_config",
    "load_models_config",
]
