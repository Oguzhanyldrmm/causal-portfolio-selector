from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ALGORITHMS: tuple[str, ...] = (
    "PC_discrete",
    "FCI",
    "GES",
    "HC",
    "Tabu",
    "K2",
)


@dataclass(frozen=True)
class FeatureConfig:
    max_pairs: int = 800
    random_seed: int = 42
    association_threshold: float = 0.05
    ci_alpha: float = 0.01


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    n_estimators: int = 300
    min_samples_leaf: int = 2


@dataclass(frozen=True)
class LearnedConfig:
    enabled: bool = False
    device: str = "auto"
    synthetic_graph_count: int = 1000
    n_vars_choices: tuple[int, ...] = (4, 6, 8, 10, 15, 20, 30)
    sample_sizes: tuple[int, ...] = (500, 1000, 3000)
    cardinality_min: int = 2
    cardinality_max: int = 8
    max_indegree: int = 4
    edge_probability_min: float = 0.08
    edge_probability_max: float = 0.25
    embedding_dim: int = 16
    hidden_dim: int = 32
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    max_feature_rows: int = 1000
    synthetic_workers: int = 4
    random_seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = Path(".")
    source_run_dir: Path = Path(
        "/home/oguzhan/Causal-Algorithm-Selection/runs/"
        "benchmark_runs_min_oct_no_pathfinder/20260427T215118Z"
    )
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS
    external_datasets: tuple[str, ...] = ("sachs",)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    learned: LearnedConfig = field(default_factory=LearnedConfig)

    def resolved_root(self) -> Path:
        return self.project_root.expanduser().resolve()


def _coerce_path(value: Any, default: Path) -> Path:
    if value is None:
        return default
    return Path(str(value)).expanduser()


def load_config(path: str | Path | None = None) -> AppConfig:
    payload: dict[str, Any] = {}
    config_path = Path(path) if path else Path("configs/default.yaml")
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text()) or {}

    feature_payload = payload.get("features") or {}
    model_payload = payload.get("model") or {}
    learned_payload = payload.get("learned") or {}
    return AppConfig(
        project_root=_coerce_path(payload.get("project_root"), Path(".")),
        source_run_dir=_coerce_path(
            payload.get("source_run_dir"),
            AppConfig.source_run_dir,
        ),
        algorithms=tuple(payload.get("algorithms") or DEFAULT_ALGORITHMS),
        external_datasets=tuple(payload.get("external_datasets") or ("sachs",)),
        features=FeatureConfig(
            max_pairs=int(feature_payload.get("max_pairs", 800)),
            random_seed=int(feature_payload.get("random_seed", 42)),
            association_threshold=float(feature_payload.get("association_threshold", 0.05)),
            ci_alpha=float(feature_payload.get("ci_alpha", 0.01)),
        ),
        model=ModelConfig(
            random_state=int(model_payload.get("random_state", 42)),
            n_estimators=int(model_payload.get("n_estimators", 300)),
            min_samples_leaf=int(model_payload.get("min_samples_leaf", 2)),
        ),
        learned=LearnedConfig(
            enabled=bool(learned_payload.get("enabled", False)),
            device=str(learned_payload.get("device", "auto")),
            synthetic_graph_count=int(learned_payload.get("synthetic_graph_count", 1000)),
            n_vars_choices=tuple(int(v) for v in learned_payload.get("n_vars_choices", [4, 6, 8, 10, 15, 20, 30])),
            sample_sizes=tuple(int(v) for v in learned_payload.get("sample_sizes", [500, 1000, 3000])),
            cardinality_min=int(learned_payload.get("cardinality_min", 2)),
            cardinality_max=int(learned_payload.get("cardinality_max", 8)),
            max_indegree=int(learned_payload.get("max_indegree", 4)),
            edge_probability_min=float(learned_payload.get("edge_probability_min", 0.08)),
            edge_probability_max=float(learned_payload.get("edge_probability_max", 0.25)),
            embedding_dim=int(learned_payload.get("embedding_dim", 16)),
            hidden_dim=int(learned_payload.get("hidden_dim", 32)),
            epochs=int(learned_payload.get("epochs", 50)),
            batch_size=int(learned_payload.get("batch_size", 16)),
            learning_rate=float(learned_payload.get("learning_rate", 1e-3)),
            validation_fraction=float(learned_payload.get("validation_fraction", 0.2)),
            max_feature_rows=int(learned_payload.get("max_feature_rows", 1000)),
            synthetic_workers=int(learned_payload.get("synthetic_workers", 4)),
            random_seed=int(learned_payload.get("random_seed", 42)),
        ),
    )
