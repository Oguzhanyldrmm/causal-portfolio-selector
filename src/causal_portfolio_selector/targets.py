from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import calibration_table_path, imported_paths, load_import_manifest
from .config import AppConfig
from .features import FEATURE_COLUMNS, build_feature_table


LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "shd",
        "normalized_inverse_shd",
        "adjacency_f1",
        "directed_f1",
        "orientation_accuracy_on_recovered_edges",
        "structural_score",
        "directed_score",
        "combined_score",
        "actual_rank_by_shd",
        "actual_structural_rank",
        "actual_directed_rank",
        "actual_combined_rank",
        "actual_top3_member",
        "actual_top4_member",
        "run_status",
        "source_run_dir",
        "manual_pre_run_score",
        "manual_predicted_rank_by_score",
    }
)


TARGET_COLUMNS: tuple[str, ...] = (
    "dataset_name",
    "algorithm_name",
    "truth_type",
    "split_role",
    "shd",
    "combined_score",
    "quality_rank",
    "oracle_shd",
    "relative_regret",
    "possible_edges",
)


def build_tables(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = config.resolved_root()
    manifest = load_import_manifest(root)
    paths = imported_paths(root)
    dataset_paths = {
        entry["dataset_name"]: root / entry["dataset_path"]
        for entry in manifest["datasets"]
    }
    feature_table = build_feature_table(dataset_paths, config=config.features)

    calibration = pd.read_csv(calibration_table_path(root))
    calibration = calibration[calibration["algorithm_name"].isin(config.algorithms)].copy()
    calibration = calibration[calibration["run_status"] == "success"].copy()
    if calibration.empty:
        raise ValueError("No successful calibration rows were found.")

    targets = calibration.copy()
    for column in ("shd", "combined_score", "n_features"):
        targets[column] = pd.to_numeric(targets[column], errors="coerce")
    targets["possible_edges"] = (
        targets["n_features"] * (targets["n_features"] - 1.0) / 2.0
    ).clip(lower=1.0)
    targets["oracle_shd"] = targets.groupby("dataset_name")["shd"].transform("min")
    targets["relative_regret"] = (
        (targets["shd"] - targets["oracle_shd"]) / targets["possible_edges"]
    )
    targets["split_role"] = targets.apply(
        lambda row: _split_role(row, external_datasets=set(config.external_datasets)),
        axis=1,
    )
    targets = _assign_quality_rank(targets)
    targets = targets[list(TARGET_COLUMNS)].sort_values(
        ["dataset_name", "quality_rank", "algorithm_name"],
        kind="mergesort",
    )

    table_dir = Path(root) / "artifacts" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(table_dir / "features.csv", index=False)
    targets.to_csv(table_dir / "targets.csv", index=False)
    return feature_table, targets


def load_or_build_tables(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = config.resolved_root()
    feature_path = root / "artifacts" / "tables" / "features.csv"
    target_path = root / "artifacts" / "tables" / "targets.csv"
    if feature_path.exists() and target_path.exists():
        return pd.read_csv(feature_path), pd.read_csv(target_path)
    return build_tables(config)


def _truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _split_role(row: pd.Series, *, external_datasets: set[str]) -> str:
    dataset_name = str(row["dataset_name"])
    if dataset_name in external_datasets or str(row.get("truth_type", "")) != "exact_bn_sem":
        return "external"
    if _truthy(row.get("enabled_for_discrete_train")):
        return "train"
    if _truthy(row.get("enabled_for_discrete_eval")):
        return "eval"
    return "calibration"


def _assign_quality_rank(frame: pd.DataFrame) -> pd.DataFrame:
    ranked_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("dataset_name", sort=True):
        ranked = group.sort_values(
            ["shd", "combined_score", "algorithm_name"],
            ascending=[True, False, True],
            kind="mergesort",
        ).copy()
        ranked["quality_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked_parts.append(ranked)
    return pd.concat(ranked_parts, ignore_index=True)


def exact_dataset_names(targets: pd.DataFrame, *, external_datasets: set[str]) -> list[str]:
    exact = targets[
        (targets["truth_type"] == "exact_bn_sem")
        & (~targets["dataset_name"].isin(external_datasets))
    ]
    return sorted(exact["dataset_name"].unique().tolist())


def legacy_train_dataset_names(targets: pd.DataFrame) -> list[str]:
    return sorted(targets.loc[targets["split_role"] == "train", "dataset_name"].unique().tolist())


def legacy_eval_dataset_names(targets: pd.DataFrame) -> list[str]:
    return sorted(targets.loc[targets["split_role"] == "eval", "dataset_name"].unique().tolist())


def external_dataset_names(targets: pd.DataFrame) -> list[str]:
    return sorted(targets.loc[targets["split_role"] == "external", "dataset_name"].unique().tolist())


def feature_matrix(
    feature_table: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    selected_features = tuple(feature_names or FEATURE_COLUMNS)
    columns = ["dataset_name", *selected_features]
    missing = [column for column in columns if column not in feature_table.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    leakage = sorted(set(feature_table.columns) & LEAKAGE_COLUMNS)
    if leakage:
        raise ValueError(f"Feature table contains leakage columns: {leakage}")
    return feature_table[columns].copy()
