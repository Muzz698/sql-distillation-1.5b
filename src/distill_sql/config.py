"""Typed configuration: YAML files validated by pydantic, secrets from the environment / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    cache_dir: Path
    runs_dir: Path

    def resolved(self, root: Path) -> PathsConfig:
        """Return a copy with every relative path anchored at ``root``."""
        return PathsConfig(**{k: (v if v.is_absolute() else root / v) for k, v in self})


class ModelConfig(_Strict):
    base_id: str


class SourceConfig(_Strict):
    """A downloadable archive. ``urls`` are mirrors, tried in order."""

    filename: str
    urls: list[str] = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    extract_to: str  # folder name under paths.raw_dir


class DatasetConfig(_Strict):
    license: str
    source: SourceConfig
    train_splits: list[str] = Field(default_factory=list)
    eval_splits: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _splits_disjoint(self) -> DatasetConfig:
        overlap = set(self.train_splits) & set(self.eval_splits)
        if overlap:
            raise ValueError(f"splits used for both training and eval: {sorted(overlap)}")
        return self


class SchemaConfig(_Strict):
    sample_rows: int = Field(default=3, ge=0, le=20)
    max_value_chars: int = Field(default=60, ge=8)


class ContaminationConfig(_Strict):
    ngram_n: int = Field(default=13, ge=3)
    near_dup_shingle_n: int = Field(default=3, ge=1)
    near_dup_jaccard: float = Field(default=0.8, gt=0.0, le=1.0)
    # Globs (relative to the project root) of JSONL files whose questions will be trained on
    # or shown to a teacher. Each row needs at least `db_id` and `question`.
    train_globs: list[str] = Field(default_factory=list)


class TeacherConfig(_Strict):
    name: str
    provider: Literal["groq", "gemini"]
    model: str
    rpm: int = Field(gt=0)
    tpm: int = Field(gt=0)
    rpd: int = Field(gt=0)


class TrackingConfig(_Strict):
    backend: Literal["mlflow", "none"] = "mlflow"
    mlflow_uri: str = "file:./mlruns"
    experiment: str = "distill-sql"


class EvalRegressionConfig(_Strict):
    predictions_file: Path
    gold_file: Path
    db_dir: Path
    min_execution_accuracy: float = Field(ge=0.0, le=1.0)


class AppConfig(_Strict):
    seed: int = 42
    paths: PathsConfig
    model: ModelConfig
    datasets: dict[str, DatasetConfig]
    teachers: list[TeacherConfig]
    schema_extraction: SchemaConfig = SchemaConfig()
    contamination: ContaminationConfig = ContaminationConfig()
    tracking: TrackingConfig = TrackingConfig()
    eval_regression: EvalRegressionConfig | None = None


class Secrets(BaseSettings):
    """API keys. Read from the environment or a git-ignored ``.env``, never from YAML."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_read_yaml(path.parent / parent), raw)


def load_config(path: str | Path, root: Path = PROJECT_ROOT) -> AppConfig:
    """Load a YAML config (following ``extends:``), validate it, and anchor paths at ``root``."""
    cfg = AppConfig.model_validate(_read_yaml(Path(path)))
    return cfg.model_copy(update={"paths": cfg.paths.resolved(root)})
