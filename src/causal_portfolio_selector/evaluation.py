from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .features import FEATURE_COLUMNS
from .models import PortfolioSelector, train_selector


def evaluate_prediction(
    *,
    dataset_name: str,
    prediction: dict[str, object],
    targets: pd.DataFrame,
) -> dict[str, object]:
    group = targets[targets["dataset_name"] == dataset_name].copy()
    if group.empty:
        raise ValueError(f"No targets found for dataset: {dataset_name}")

    shd_by_algorithm = {
        str(row["algorithm_name"]): float(row["shd"])
        for _, row in group.iterrows()
    }
    rank_by_algorithm = {
        str(row["algorithm_name"]): float(row["quality_rank"])
        for _, row in group.iterrows()
    }
    predicted_ranking = dict(prediction["ranking"])  # type: ignore[arg-type]
    predicted_order = sorted(predicted_ranking, key=lambda name: predicted_ranking[name])
    top3 = predicted_order[:3]
    actual_top3 = (
        group.sort_values(["quality_rank", "shd", "combined_score", "algorithm_name"], ascending=[True, True, False, True])
        ["algorithm_name"]
        .astype(str)
        .head(3)
        .tolist()
    )
    top3_overlap = len(set(top3) & set(actual_top3))

    oracle_shd = min(shd_by_algorithm.values())
    oracle_algorithms = {
        algorithm for algorithm, shd in shd_by_algorithm.items() if np.isclose(shd, oracle_shd)
    }
    pred_best = predicted_order[0]
    best_top3_shd = min(shd_by_algorithm[algorithm] for algorithm in top3)
    regret_at_1 = shd_by_algorithm[pred_best] - oracle_shd
    regret_at_3 = best_top3_shd - oracle_shd

    common_algorithms = [
        algorithm for algorithm in predicted_order if algorithm in rank_by_algorithm
    ]
    pred_ranks = [float(predicted_ranking[algorithm]) for algorithm in common_algorithms]
    actual_ranks = [float(rank_by_algorithm[algorithm]) for algorithm in common_algorithms]
    spearman = _safe_correlation(spearmanr, pred_ranks, actual_ranks)
    kendall = _safe_correlation(kendalltau, pred_ranks, actual_ranks)

    if oracle_shd == 0:
        oracle_ratio_at_3 = 1.0 if best_top3_shd == 0 else np.nan
    else:
        oracle_ratio_at_3 = best_top3_shd / oracle_shd

    return {
        "dataset_name": dataset_name,
        "predicted_top1": pred_best,
        "predicted_top3": ",".join(top3),
        "actual_top3": ",".join(actual_top3),
        "oracle_best": ",".join(sorted(oracle_algorithms)),
        "top1_hit": float(pred_best in oracle_algorithms),
        "top3_hit": float(any(algorithm in oracle_algorithms for algorithm in top3)),
        "oracle_in_top3": float(any(algorithm in oracle_algorithms for algorithm in top3)),
        "top3_overlap": float(top3_overlap),
        "top3_overlap_at_least_2": float(top3_overlap >= 2),
        "oracle_shd": float(oracle_shd),
        "predicted_top1_shd": float(shd_by_algorithm[pred_best]),
        "best_top3_shd": float(best_top3_shd),
        "regret_at_1": float(regret_at_1),
        "regret_at_3": float(regret_at_3),
        "oracle_ratio_at_3": float(oracle_ratio_at_3),
        "rank_spearman": spearman,
        "rank_kendall": kendall,
    }


def _safe_correlation(fn, left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(set(left)) <= 1 or len(set(right)) <= 1:
        return float("nan")
    result = fn(left, right)
    statistic = getattr(result, "statistic", result[0])
    return float(statistic) if np.isfinite(statistic) else float("nan")


def predict_dataset_from_feature_table(
    selector: PortfolioSelector,
    feature_table: pd.DataFrame,
    dataset_name: str,
) -> dict[str, object]:
    row = feature_table[feature_table["dataset_name"] == dataset_name]
    if row.empty:
        raise ValueError(f"Feature row not found for dataset: {dataset_name}")
    features = row.iloc[0][list(FEATURE_COLUMNS)].to_dict()
    return selector.predict_from_features(features)


def evaluate_selector_on_datasets(
    selector: PortfolioSelector,
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for dataset_name in sorted(dataset_names):
        prediction = predict_dataset_from_feature_table(selector, feature_table, dataset_name)
        metrics = evaluate_prediction(
            dataset_name=dataset_name,
            prediction=prediction,
            targets=targets,
        )
        metrics["split_name"] = split_name
        metric_rows.append(metrics)
        for algorithm, rank in dict(prediction["ranking"]).items():  # type: ignore[arg-type]
            prediction_rows.append(
                {
                    "split_name": split_name,
                    "dataset_name": dataset_name,
                    "algorithm_name": algorithm,
                    "predicted_rank": int(rank),
                    "pairwise_win_score": float(
                        dict(prediction["pairwise_win_score"])[algorithm]  # type: ignore[arg-type]
                    ),
                    "predicted_relative_regret": float(
                        dict(prediction["predicted_relative_regret"])[algorithm]  # type: ignore[arg-type]
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def lodo_evaluate(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    model_config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_names = sorted(dataset_names)
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for held_out in dataset_names:
        train_names = [dataset_name for dataset_name in dataset_names if dataset_name != held_out]
        selector = train_selector(
            feature_table,
            targets,
            dataset_names=train_names,
            algorithms=algorithms,
            config=model_config,
        )
        metrics, predictions = evaluate_selector_on_datasets(
            selector,
            feature_table,
            targets,
            dataset_names=[held_out],
            split_name="lodo",
        )
        metrics["held_out_dataset"] = held_out
        predictions["held_out_dataset"] = held_out
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(prediction_frames, ignore_index=True)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "top1_hit",
        "top3_hit",
        "oracle_in_top3",
        "top3_overlap",
        "top3_overlap_at_least_2",
        "regret_at_1",
        "regret_at_3",
        "oracle_ratio_at_3",
        "rank_spearman",
        "rank_kendall",
    ]
    present = [column for column in numeric_columns if column in metrics.columns]
    summary = (
        metrics.groupby("split_name")[present]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("split_name")
    )
    return summary.rename(columns={"top3_overlap": "avg_top3_overlap"})


def summary_markdown(aggregate: pd.DataFrame) -> str:
    lines = ["# Causal Portfolio Selector Evaluation", ""]
    if aggregate.empty:
        lines.append("No metrics available.")
        return "\n".join(lines) + "\n"
    columns = list(aggregate.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in aggregate.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"
