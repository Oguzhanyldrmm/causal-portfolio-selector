from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .artifacts import report_path
from .config import AppConfig
from .evaluation import aggregate_metrics, evaluate_prediction, summary_markdown
from .features import FEATURE_COLUMNS
from .models import train_selector
from .targets import exact_dataset_names, load_or_build_tables


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "basic": (
        "n_samples",
        "n_features",
        "sample_to_feature_ratio",
        "continuous_ratio",
        "categorical_ratio",
        "missing_ratio",
        "avg_variance",
        "avg_skewness",
        "avg_kurtosis",
    ),
    "cardinality": (
        "avg_cardinality",
        "max_cardinality",
        "cardinality_entropy",
        "rare_category_ratio",
        "singleton_category_ratio",
        "feature_sparsity_ratio",
    ),
    "association": (
        "mean_nmi",
        "max_nmi",
        "std_nmi",
        "mean_cramers_v",
        "max_cramers_v",
        "std_cramers_v",
        "mean_chi2_pvalue",
        "ci_rejection_rate",
    ),
    "proxy_graph": (
        "proxy_graph_density",
        "proxy_avg_degree",
        "proxy_degree_gini",
        "proxy_avg_clustering",
        "proxy_modularity",
        "proxy_num_components",
    ),
    "all": FEATURE_COLUMNS,
}


def run_phase1_evidence(
    config: AppConfig,
    *,
    random_repeats: int = 200,
) -> dict[str, Path]:
    root = config.resolved_root()
    reports_dir = report_path(root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    feature_table, targets = load_or_build_tables(config)
    dataset_names = exact_dataset_names(
        targets,
        external_datasets=set(config.external_datasets),
    )
    algorithms = tuple(config.algorithms)

    baseline_metrics = _baseline_lodo_metrics(
        targets,
        dataset_names=dataset_names,
        algorithms=algorithms,
        random_repeats=random_repeats,
        random_seed=config.model.random_state,
    )
    ablation_metrics = _ablation_lodo_metrics(
        feature_table,
        targets,
        dataset_names=dataset_names,
        algorithms=algorithms,
        config=config,
    )

    baseline_summary = _aggregate_by_method(baseline_metrics)
    ablation_summary = _aggregate_by_method(ablation_metrics)
    comparison_summary = pd.concat(
        [
            baseline_summary.assign(section="baseline"),
            ablation_summary.assign(section="ablation"),
        ],
        ignore_index=True,
    )

    paths = {
        "baseline_metrics": reports_dir / "baseline_metrics.csv",
        "baseline_summary": reports_dir / "baseline_summary.csv",
        "ablation_metrics": reports_dir / "ablation_metrics.csv",
        "ablation_summary": reports_dir / "ablation_summary.csv",
        "phase1_comparison_summary": reports_dir / "phase1_comparison_summary.csv",
        "phase1_evidence": reports_dir / "phase1_evidence.md",
    }
    baseline_metrics.to_csv(paths["baseline_metrics"], index=False)
    baseline_summary.to_csv(paths["baseline_summary"], index=False)
    ablation_metrics.to_csv(paths["ablation_metrics"], index=False)
    ablation_summary.to_csv(paths["ablation_summary"], index=False)
    comparison_summary.to_csv(paths["phase1_comparison_summary"], index=False)
    paths["phase1_evidence"].write_text(
        _phase1_markdown(baseline_summary, ablation_summary)
    )
    return paths


def run_phase2_evidence(config: AppConfig) -> dict[str, Path]:
    root = config.resolved_root()
    reports_dir = report_path(root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _, targets = load_or_build_tables(config)
    combined_path = root / "artifacts" / "tables" / "features_plus_learned.csv"
    if not combined_path.exists():
        raise FileNotFoundError(
            "Learned feature table not found. Run build-learned-features first."
        )
    feature_table = pd.read_csv(combined_path)
    dataset_names = exact_dataset_names(
        targets,
        external_datasets=set(config.external_datasets),
    )
    algorithms = tuple(config.algorithms)
    learned_columns = tuple(column for column in feature_table.columns if column.startswith("lf_"))
    feature_sets = {
        "handcrafted_all": FEATURE_COLUMNS,
        "learned_only": learned_columns,
        "handcrafted_plus_learned": (*FEATURE_COLUMNS, *learned_columns),
    }
    learned_metrics = _feature_set_lodo_metrics(
        feature_table,
        targets,
        dataset_names=dataset_names,
        algorithms=algorithms,
        config=config,
        feature_sets=feature_sets,
    )
    baseline_metrics = _baseline_lodo_metrics(
        targets,
        dataset_names=dataset_names,
        algorithms=algorithms,
        random_repeats=200,
        random_seed=config.model.random_state,
    )
    learned_summary = _aggregate_by_method(learned_metrics)
    baseline_summary = _aggregate_by_method(baseline_metrics)
    comparison = pd.concat(
        [
            baseline_summary.assign(section="baseline"),
            learned_summary.assign(section="phase2"),
        ],
        ignore_index=True,
    )
    paths = {
        "phase2_metrics": reports_dir / "phase2_metrics.csv",
        "phase2_summary": reports_dir / "phase2_summary.csv",
        "phase2_comparison_summary": reports_dir / "phase2_comparison_summary.csv",
        "phase2_evidence": reports_dir / "phase2_evidence.md",
    }
    learned_metrics.to_csv(paths["phase2_metrics"], index=False)
    learned_summary.to_csv(paths["phase2_summary"], index=False)
    comparison.to_csv(paths["phase2_comparison_summary"], index=False)
    paths["phase2_evidence"].write_text(_phase2_markdown(baseline_summary, learned_summary))
    return paths


def _baseline_lodo_metrics(
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: tuple[str, ...],
    random_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    dataset_names = sorted(dataset_names)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    static_order = tuple(algorithms)

    for held_out in dataset_names:
        train_names = [name for name in dataset_names if name != held_out]
        train_targets = targets[targets["dataset_name"].isin(train_names)].copy()
        method_orders = {
            "static_catalog_order": static_order,
            "train_mean_shd": _rank_by_train_mean_shd(train_targets, algorithms),
            "train_oracle_frequency": _rank_by_oracle_frequency(train_targets, algorithms),
            "train_best_fixed_top3": _rank_by_best_fixed_top3(train_targets, algorithms),
        }
        for method, order in method_orders.items():
            metric = evaluate_prediction(
                dataset_name=held_out,
                prediction=_prediction_from_order(order),
                targets=targets,
            )
            metric.update(
                {
                    "split_name": "lodo",
                    "method": method,
                    "held_out_dataset": held_out,
                    "repeat": 0,
                }
            )
            rows.append(metric)

        for repeat in range(random_repeats):
            order = tuple(rng.permutation(algorithms).tolist())
            metric = evaluate_prediction(
                dataset_name=held_out,
                prediction=_prediction_from_order(order),
                targets=targets,
            )
            metric.update(
                {
                    "split_name": "lodo",
                    "method": "random_top3",
                    "held_out_dataset": held_out,
                    "repeat": repeat,
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows)


def _ablation_lodo_metrics(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: tuple[str, ...],
    config: AppConfig,
) -> pd.DataFrame:
    dataset_names = sorted(dataset_names)
    rows: list[dict[str, object]] = []
    for feature_set_name, feature_names in FEATURE_SETS.items():
        for held_out in dataset_names:
            train_names = [name for name in dataset_names if name != held_out]
            selector = train_selector(
                feature_table,
                targets,
                dataset_names=train_names,
                algorithms=algorithms,
                config=config.model,
                feature_names=feature_names,
            )
            feature_row = feature_table[feature_table["dataset_name"] == held_out].iloc[0]
            prediction = selector.predict_from_features(feature_row.to_dict())
            metric = evaluate_prediction(
                dataset_name=held_out,
                prediction=prediction,
                targets=targets,
            )
            metric.update(
                {
                    "split_name": "lodo",
                    "method": f"learned_{feature_set_name}",
                    "feature_set": feature_set_name,
                    "held_out_dataset": held_out,
                    "repeat": 0,
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows)


def _feature_set_lodo_metrics(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: tuple[str, ...],
    config: AppConfig,
    feature_sets: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    dataset_names = sorted(dataset_names)
    rows: list[dict[str, object]] = []
    for feature_set_name, feature_names in feature_sets.items():
        if not feature_names:
            continue
        for held_out in dataset_names:
            train_names = [name for name in dataset_names if name != held_out]
            selector = train_selector(
                feature_table,
                targets,
                dataset_names=train_names,
                algorithms=algorithms,
                config=config.model,
                feature_names=feature_names,
            )
            feature_row = feature_table[feature_table["dataset_name"] == held_out].iloc[0]
            prediction = selector.predict_from_features(feature_row.to_dict())
            metric = evaluate_prediction(
                dataset_name=held_out,
                prediction=prediction,
                targets=targets,
            )
            metric.update(
                {
                    "split_name": "lodo",
                    "method": f"phase2_{feature_set_name}",
                    "feature_set": feature_set_name,
                    "held_out_dataset": held_out,
                    "repeat": 0,
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows)


def _rank_by_train_mean_shd(
    train_targets: pd.DataFrame,
    algorithms: tuple[str, ...],
) -> tuple[str, ...]:
    grouped = (
        train_targets.groupby("algorithm_name")["shd"]
        .mean()
        .reindex(algorithms)
        .fillna(np.inf)
    )
    return tuple(sorted(algorithms, key=lambda algorithm: (float(grouped[algorithm]), algorithm)))


def _rank_by_oracle_frequency(
    train_targets: pd.DataFrame,
    algorithms: tuple[str, ...],
) -> tuple[str, ...]:
    counts = {algorithm: 0 for algorithm in algorithms}
    for _, group in train_targets.groupby("dataset_name"):
        oracle_shd = float(group["shd"].min())
        oracle_algorithms = set(group.loc[np.isclose(group["shd"], oracle_shd), "algorithm_name"])
        for algorithm in algorithms:
            if algorithm in oracle_algorithms:
                counts[algorithm] += 1
    mean_order = _rank_by_train_mean_shd(train_targets, algorithms)
    mean_rank = {algorithm: rank for rank, algorithm in enumerate(mean_order)}
    return tuple(
        sorted(
            algorithms,
            key=lambda algorithm: (-counts[algorithm], mean_rank[algorithm], algorithm),
        )
    )


def _rank_by_best_fixed_top3(
    train_targets: pd.DataFrame,
    algorithms: tuple[str, ...],
) -> tuple[str, ...]:
    best_key: tuple[float, float, tuple[str, ...]] | None = None
    best_combo: tuple[str, ...] | None = None
    for combo in itertools.combinations(algorithms, 3):
        metrics = _portfolio_train_metrics(train_targets, combo)
        key = (metrics["regret_at_3"], -metrics["top3_hit"], combo)
        if best_key is None or key < best_key:
            best_key = key
            best_combo = combo
    if best_combo is None:
        return algorithms
    remainder = tuple(algorithm for algorithm in _rank_by_train_mean_shd(train_targets, algorithms) if algorithm not in best_combo)
    return (*best_combo, *remainder)


def _portfolio_train_metrics(
    train_targets: pd.DataFrame,
    combo: tuple[str, ...],
) -> dict[str, float]:
    regrets: list[float] = []
    hits: list[float] = []
    for _, group in train_targets.groupby("dataset_name"):
        shd_by_algorithm = {
            str(row["algorithm_name"]): float(row["shd"])
            for _, row in group.iterrows()
        }
        oracle = min(shd_by_algorithm.values())
        best = min(shd_by_algorithm[algorithm] for algorithm in combo)
        regrets.append(best - oracle)
        hits.append(float(np.isclose(best, oracle)))
    return {
        "regret_at_3": float(np.mean(regrets)) if regrets else float("inf"),
        "top3_hit": float(np.mean(hits)) if hits else 0.0,
    }


def _prediction_from_order(order: tuple[str, ...]) -> dict[str, object]:
    return {
        "top_3": list(order[:3]),
        "ranking": {algorithm: rank + 1 for rank, algorithm in enumerate(order)},
        "pairwise_win_score": {
            algorithm: float(len(order) - rank - 1)
            for rank, algorithm in enumerate(order)
        },
        "predicted_relative_regret": {
            algorithm: float(rank)
            for rank, algorithm in enumerate(order)
        },
    }


def _aggregate_by_method(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    numeric_columns = [
        "top1_hit",
        "top3_hit",
        "regret_at_1",
        "regret_at_3",
        "oracle_ratio_at_3",
        "rank_spearman",
        "rank_kendall",
    ]
    return (
        metrics.groupby("method")[numeric_columns]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["top3_hit", "regret_at_3", "top1_hit"], ascending=[False, True, False])
    )


def _phase1_markdown(baseline_summary: pd.DataFrame, ablation_summary: pd.DataFrame) -> str:
    lines = ["# Phase 1 Evidence", ""]
    lines.append("## Baseline Comparison")
    lines.append("")
    lines.append(summary_markdown(baseline_summary.rename(columns={"method": "split_name"})))
    lines.append("## Feature Ablation")
    lines.append("")
    lines.append(summary_markdown(ablation_summary.rename(columns={"method": "split_name"})))
    return "\n".join(lines)


def _phase2_markdown(baseline_summary: pd.DataFrame, learned_summary: pd.DataFrame) -> str:
    lines = ["# Phase 2 Evidence", ""]
    lines.append("## Baseline Comparison")
    lines.append("")
    lines.append(summary_markdown(baseline_summary.rename(columns={"method": "split_name"})))
    lines.append("## Learned Fingerprint Comparison")
    lines.append("")
    lines.append(summary_markdown(learned_summary.rename(columns={"method": "split_name"})))
    return "\n".join(lines)
