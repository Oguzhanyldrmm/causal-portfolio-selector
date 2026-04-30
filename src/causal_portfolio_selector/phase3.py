from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .artifacts import artifact_path, load_import_manifest, report_path
from .config import AppConfig
from .evaluation import evaluate_prediction
from .experiments import _prediction_from_order
from .features import FEATURE_COLUMNS
from .models import save_selector, train_selector
from .targets import exact_dataset_names, load_or_build_tables


BASE_ALGORITHMS: tuple[str, ...] = (
    "PC_discrete",
    "FCI",
    "GES",
    "HC",
    "Tabu",
    "K2",
)
MISSING_ALGORITHMS: tuple[str, ...] = ("MMHC", "BOSS", "GRaSP")
ALL_ALGORITHMS: tuple[str, ...] = (*BASE_ALGORITHMS, *MISSING_ALGORITHMS)


@dataclass(frozen=True)
class GraphEvaluation:
    status: str
    shd: float | None
    adjacency_f1: float | None
    directed_f1: float | None
    orientation_accuracy_on_recovered_edges: float | None
    graph_type: str | None


def run_phase3_evidence(config: AppConfig) -> dict[str, Path]:
    root = config.resolved_root()
    reports_dir = report_path(root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    table_dir = artifact_path(root, "tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    feature_table, base_targets = load_or_build_tables(config)
    augmented_targets, missing_eval = build_timeout_aware_targets(config, base_targets)
    augmented_target_path = table_dir / "targets_timeout_aware.csv"
    missing_eval_path = reports_dir / "missing_algorithm_evaluation.csv"
    augmented_targets.to_csv(augmented_target_path, index=False)
    missing_eval.to_csv(missing_eval_path, index=False)

    exact_names = exact_dataset_names(
        augmented_targets,
        external_datasets=set(config.external_datasets),
    )
    complete_all9_names = _complete_dataset_names(
        augmented_targets,
        dataset_names=exact_names,
        algorithms=ALL_ALGORITHMS,
    )
    learned_feature_table = _load_phase3_feature_table(root, feature_table)
    learned_columns = tuple(
        column for column in learned_feature_table.columns if column.startswith("lf_")
    )
    feature_sets: dict[str, tuple[str, ...]] = {
        "handcrafted_all": FEATURE_COLUMNS,
    }
    if learned_columns:
        feature_sets["learned_only"] = learned_columns
        feature_sets["handcrafted_plus_learned"] = (*FEATURE_COLUMNS, *learned_columns)

    timeout_aware_metrics = _feature_lodo_metrics(
        learned_feature_table,
        augmented_targets,
        dataset_names=exact_names,
        algorithms=ALL_ALGORITHMS,
        config=config,
        feature_sets=feature_sets,
        split_name="phase3_timeout_aware",
    )
    all9_metrics = _feature_lodo_metrics(
        learned_feature_table,
        augmented_targets,
        dataset_names=complete_all9_names,
        algorithms=ALL_ALGORITHMS,
        config=config,
        feature_sets=feature_sets,
        split_name="phase3_complete_all9",
        train_dataset_names=exact_names,
    )
    baseline_timeout_metrics = _baseline_lodo_metrics(
        augmented_targets,
        dataset_names=exact_names,
        algorithms=ALL_ALGORITHMS,
        random_repeats=200,
        random_seed=config.model.random_state,
        split_name="phase3_timeout_aware",
    )
    baseline_all9_metrics = _baseline_lodo_metrics(
        augmented_targets,
        dataset_names=complete_all9_names,
        algorithms=ALL_ALGORITHMS,
        random_repeats=200,
        random_seed=config.model.random_state,
        split_name="phase3_complete_all9",
        train_dataset_names=exact_names,
    )

    metrics = pd.concat(
        [
            baseline_timeout_metrics,
            timeout_aware_metrics,
            baseline_all9_metrics,
            all9_metrics,
        ],
        ignore_index=True,
    )
    summary = (
        metrics.groupby(["split_name", "method"])[
            [
                "top1_hit",
                "top3_hit",
                "regret_at_1",
                "regret_at_3",
                "oracle_ratio_at_3",
                "rank_spearman",
                "rank_kendall",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["split_name", "top3_hit", "regret_at_3"], ascending=[True, False, True])
    )

    metrics_path = reports_dir / "phase3_metrics.csv"
    summary_path = reports_dir / "phase3_summary.csv"
    evidence_path = reports_dir / "phase3_evidence.md"
    best_feature_set_name, best_feature_names = _best_phase3_feature_set(
        summary,
        feature_sets=feature_sets,
        split_name="phase3_timeout_aware",
    )
    final_selector = train_selector(
        learned_feature_table,
        augmented_targets,
        dataset_names=exact_names,
        algorithms=ALL_ALGORITHMS,
        config=config.model,
        feature_names=best_feature_names,
    )
    final_model_path = artifact_path(root, "models", "selector_phase3_timeout_aware.joblib")
    save_selector(final_selector, final_model_path)

    fixed_order = _rank_by_best_fixed_top3(
        augmented_targets[augmented_targets["dataset_name"].isin(exact_names)].copy(),
        ALL_ALGORITHMS,
    )
    fixed_policy_path = artifact_path(root, "models", "phase3_fixed_policy.json")
    fixed_policy_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_policy_path.write_text(
        json.dumps(
            {
                "description": "Best fixed top-3 portfolio trained on timeout-aware exact datasets.",
                "algorithms": list(ALL_ALGORITHMS),
                "top_3": list(fixed_order[:3]),
                "ranking": {
                    algorithm: rank + 1 for rank, algorithm in enumerate(fixed_order)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    evidence_path.write_text(
        _phase3_markdown(
            summary,
            missing_eval=missing_eval,
            complete_all9_names=complete_all9_names,
            exact_names=exact_names,
            final_model_path=final_model_path,
            best_feature_set_name=best_feature_set_name,
            fixed_policy_path=fixed_policy_path,
            fixed_top3=fixed_order[:3],
        )
    )
    return {
        "targets_timeout_aware": augmented_target_path,
        "missing_algorithm_evaluation": missing_eval_path,
        "phase3_metrics": metrics_path,
        "phase3_summary": summary_path,
        "phase3_evidence": evidence_path,
        "phase3_selector_model": final_model_path,
        "phase3_fixed_policy": fixed_policy_path,
    }


def build_timeout_aware_targets(
    config: AppConfig,
    base_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = config.resolved_root()
    manifest = load_import_manifest(root)
    dataset_meta = {str(entry["dataset_name"]): entry for entry in manifest["datasets"]}
    split_by_dataset = (
        base_targets.drop_duplicates("dataset_name")
        .set_index("dataset_name")["split_role"]
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    records_dir = artifact_path(root, "missing_algorithm_runs", "latest", "records")
    for record_path in sorted(records_dir.glob("*.json")):
        record = json.loads(record_path.read_text())
        dataset_name = str(record["dataset_name"])
        algorithm_name = str(record["algorithm_name"])
        meta = dataset_meta[dataset_name]
        truth = _load_truth(root / str(meta["ground_truth_path"]))
        if record.get("status") == "success" and record.get("graph_result"):
            evaluation = _evaluate_graph(record["graph_result"], truth)
        else:
            evaluation = GraphEvaluation(
                status=str(record.get("status") or "unavailable"),
                shd=None,
                adjacency_f1=None,
                directed_f1=None,
                orientation_accuracy_on_recovered_edges=None,
                graph_type=None,
            )
        eval_row = {
            "dataset_name": dataset_name,
            "algorithm_name": algorithm_name,
            "run_status": record.get("status"),
            "evaluation_status": evaluation.status,
            "runtime_seconds": record.get("runtime_seconds"),
            "timeout_seconds": record.get("timeout_seconds"),
            "n_features": meta.get("n_features"),
            "shd": evaluation.shd,
            "adjacency_f1": evaluation.adjacency_f1,
            "directed_f1": evaluation.directed_f1,
            "orientation_accuracy_on_recovered_edges": evaluation.orientation_accuracy_on_recovered_edges,
            "graph_type": evaluation.graph_type,
            "error": record.get("error"),
        }
        eval_rows.append(eval_row)
        if evaluation.status != "evaluated" or evaluation.shd is None:
            continue
        possible_edges = float(meta["n_features"]) * (float(meta["n_features"]) - 1.0) / 2.0
        rows.append(
            {
                "dataset_name": dataset_name,
                "algorithm_name": algorithm_name,
                "truth_type": meta.get("truth_type"),
                "split_role": split_by_dataset.get(dataset_name, "external"),
                "shd": float(evaluation.shd),
                "combined_score": _combined_score(evaluation, possible_edges=possible_edges),
                "possible_edges": max(1.0, possible_edges),
            }
        )

    base = base_targets.copy()
    base = base[base["algorithm_name"].isin(BASE_ALGORITHMS)].copy()
    combined = pd.concat([base, pd.DataFrame(rows)], ignore_index=True, sort=False)
    for column in ("shd", "combined_score", "possible_edges"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined["oracle_shd"] = combined.groupby("dataset_name")["shd"].transform("min")
    combined["relative_regret"] = (
        (combined["shd"] - combined["oracle_shd"]) / combined["possible_edges"].clip(lower=1.0)
    )
    combined = _assign_quality_rank_by_shd(combined)
    columns = [
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
    ]
    return combined[columns].sort_values(
        ["dataset_name", "quality_rank", "algorithm_name"],
        kind="mergesort",
    ), pd.DataFrame(eval_rows)


def _load_phase3_feature_table(root: Path, fallback: pd.DataFrame) -> pd.DataFrame:
    combined_path = artifact_path(root, "tables", "features_plus_learned.csv")
    if combined_path.exists():
        return pd.read_csv(combined_path)
    return fallback


def _complete_dataset_names(
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: tuple[str, ...],
) -> list[str]:
    required = set(algorithms)
    result: list[str] = []
    for dataset_name in sorted(dataset_names):
        available = set(
            targets.loc[targets["dataset_name"] == dataset_name, "algorithm_name"].astype(str)
        )
        if required.issubset(available):
            result.append(dataset_name)
    return result


def _best_phase3_feature_set(
    summary: pd.DataFrame,
    *,
    feature_sets: dict[str, tuple[str, ...]],
    split_name: str,
) -> tuple[str, tuple[str, ...]]:
    feature_methods = {
        f"{split_name}_{feature_set_name}": feature_set_name
        for feature_set_name in feature_sets
    }
    candidates = summary[
        (summary["split_name"] == split_name)
        & (summary["method"].isin(feature_methods))
    ].copy()
    if candidates.empty:
        fallback = next(iter(feature_sets))
        return fallback, feature_sets[fallback]
    candidates = candidates.sort_values(
        ["top3_hit", "regret_at_3", "top1_hit", "regret_at_1", "method"],
        ascending=[False, True, False, True, True],
        kind="mergesort",
    )
    method = str(candidates.iloc[0]["method"])
    feature_set_name = feature_methods[method]
    return feature_set_name, feature_sets[feature_set_name]


def _feature_lodo_metrics(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: tuple[str, ...],
    config: AppConfig,
    feature_sets: dict[str, tuple[str, ...]],
    split_name: str,
    train_dataset_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    dataset_names = sorted(dataset_names)
    train_pool = sorted(train_dataset_names or dataset_names)
    rows: list[dict[str, Any]] = []
    for feature_set_name, feature_names in feature_sets.items():
        if not feature_names:
            continue
        for held_out in dataset_names:
            train_names = [name for name in train_pool if name != held_out]
            selector = train_selector(
                feature_table,
                targets,
                dataset_names=train_names,
                algorithms=algorithms,
                config=config.model,
                feature_names=feature_names,
            )
            feature_row = feature_table[feature_table["dataset_name"] == held_out].iloc[0]
            raw_prediction = selector.predict_from_features(feature_row.to_dict())
            prediction = _filter_prediction_to_available(raw_prediction, targets, held_out)
            metric = evaluate_prediction(
                dataset_name=held_out,
                prediction=prediction,
                targets=targets,
            )
            metric.update(
                {
                    "split_name": split_name,
                    "method": f"{split_name}_{feature_set_name}",
                    "feature_set": feature_set_name,
                    "held_out_dataset": held_out,
                    "repeat": 0,
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows)


def _baseline_lodo_metrics(
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: tuple[str, ...],
    random_repeats: int,
    random_seed: int,
    split_name: str,
    train_dataset_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    dataset_names = sorted(dataset_names)
    train_pool = sorted(train_dataset_names or dataset_names)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_seed)
    for held_out in dataset_names:
        train_names = [name for name in train_pool if name != held_out]
        train_targets = targets[targets["dataset_name"].isin(train_names)].copy()
        method_orders = {
            f"{split_name}_train_mean_shd": _rank_by_train_mean_shd(train_targets, algorithms),
            f"{split_name}_train_oracle_frequency": _rank_by_oracle_frequency(train_targets, algorithms),
            f"{split_name}_train_best_fixed_top3": _rank_by_best_fixed_top3(train_targets, algorithms),
        }
        for method, order in method_orders.items():
            metric = evaluate_prediction(
                dataset_name=held_out,
                prediction=_filter_order_to_available(order, targets, held_out),
                targets=targets,
            )
            metric.update(
                {
                    "split_name": split_name,
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
                prediction=_filter_order_to_available(order, targets, held_out),
                targets=targets,
            )
            metric.update(
                {
                    "split_name": split_name,
                    "method": f"{split_name}_random_top3",
                    "held_out_dataset": held_out,
                    "repeat": repeat,
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows)


def _available_algorithms(targets: pd.DataFrame, dataset_name: str) -> tuple[str, ...]:
    return tuple(
        targets.loc[targets["dataset_name"] == dataset_name, "algorithm_name"].astype(str).tolist()
    )


def _filter_prediction_to_available(
    prediction: dict[str, Any],
    targets: pd.DataFrame,
    dataset_name: str,
) -> dict[str, Any]:
    ranking = dict(prediction["ranking"])
    order = tuple(sorted(ranking, key=lambda name: ranking[name]))
    return _filter_order_to_available(order, targets, dataset_name)


def _filter_order_to_available(
    order: tuple[str, ...],
    targets: pd.DataFrame,
    dataset_name: str,
) -> dict[str, Any]:
    available = set(_available_algorithms(targets, dataset_name))
    filtered = tuple(algorithm for algorithm in order if algorithm in available)
    missing = tuple(algorithm for algorithm in sorted(available) if algorithm not in filtered)
    return _prediction_from_order((*filtered, *missing))


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
    remainder = tuple(
        algorithm
        for algorithm in _rank_by_train_mean_shd(train_targets, algorithms)
        if algorithm not in best_combo
    )
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
        available_combo = [algorithm for algorithm in combo if algorithm in shd_by_algorithm]
        if not available_combo:
            regrets.append(float("inf"))
            hits.append(0.0)
            continue
        best = min(shd_by_algorithm[algorithm] for algorithm in available_combo)
        regrets.append(best - oracle)
        hits.append(float(np.isclose(best, oracle)))
    finite_regrets = [value for value in regrets if np.isfinite(value)]
    return {
        "regret_at_3": float(np.mean(finite_regrets)) if finite_regrets else float("inf"),
        "top3_hit": float(np.mean(hits)) if hits else 0.0,
    }


def _assign_quality_rank_by_shd(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("dataset_name", sort=True):
        ranked = group.sort_values(
            ["shd", "algorithm_name"],
            ascending=[True, True],
            kind="mergesort",
        ).copy()
        ranked["quality_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        parts.append(ranked)
    return pd.concat(parts, ignore_index=True)


def _combined_score(evaluation: GraphEvaluation, *, possible_edges: float) -> float:
    if evaluation.shd is None:
        return 0.0
    inverse_shd = 1.0 - min(1.0, float(evaluation.shd) / max(1.0, possible_edges))
    adjacency_f1 = float(evaluation.adjacency_f1 or 0.0)
    directed_f1 = float(evaluation.directed_f1 or 0.0)
    orientation = float(evaluation.orientation_accuracy_on_recovered_edges or 0.0)
    structural = 0.6 * inverse_shd + 0.4 * adjacency_f1
    directed = 0.7 * directed_f1 + 0.3 * orientation
    return float(0.7 * structural + 0.3 * directed)


def _load_truth(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "dataset_name": payload.get("dataset_name"),
        "nodes": [str(node) for node in payload.get("nodes", [])],
        "directed_edges": [tuple(map(str, edge)) for edge in payload.get("directed_edges", [])],
        "undirected_edges": [tuple(map(str, edge)) for edge in payload.get("undirected_edges", [])],
        "graph_type": str(payload.get("graph_type") or "dag"),
    }


def _evaluate_graph(graph: dict[str, Any], truth: dict[str, Any]) -> GraphEvaluation:
    nodes = [str(node) for node in graph["nodes"]]
    truth_nodes = [str(node) for node in truth["nodes"]]
    if set(nodes) != set(truth_nodes):
        return GraphEvaluation("node_mismatch", None, None, None, None, graph.get("graph_type"))

    directed_edges, undirected_pairs, adjacency_pairs = _extract_graph_structure(
        np.asarray(graph["adj_matrix"], dtype=float),
        nodes,
    )
    truth_directed = {(str(src), str(dst)) for src, dst in truth["directed_edges"]}
    truth_undirected = {_pair_key(str(left), str(right)) for left, right in truth["undirected_edges"]}
    truth_adjacency = {_pair_key(src, dst) for src, dst in truth_directed} | truth_undirected

    adjacency_tp = len(adjacency_pairs & truth_adjacency)
    adjacency_fp = len(adjacency_pairs - truth_adjacency)
    adjacency_fn = len(truth_adjacency - adjacency_pairs)
    directed_tp = len(directed_edges & truth_directed)
    directed_fp = len(directed_edges - truth_directed)
    directed_fn = len(truth_directed - directed_edges)
    undirected_tp = len(undirected_pairs & truth_undirected)
    shd = _shd(truth_nodes, truth_directed, truth_undirected, directed_edges, undirected_pairs, adjacency_pairs)
    adjacency_precision = _safe_ratio(adjacency_tp, adjacency_tp + adjacency_fp)
    adjacency_recall = _safe_ratio(adjacency_tp, adjacency_tp + adjacency_fn)
    directed_precision = _safe_ratio(directed_tp, directed_tp + directed_fp)
    directed_recall = _safe_ratio(directed_tp, directed_tp + directed_fn)
    adjacency_f1 = _safe_ratio(2.0 * adjacency_precision * adjacency_recall, adjacency_precision + adjacency_recall)
    directed_f1 = _safe_ratio(2.0 * directed_precision * directed_recall, directed_precision + directed_recall)
    orientation = _safe_ratio(directed_tp + undirected_tp, adjacency_tp)
    return GraphEvaluation(
        status="evaluated",
        shd=float(shd),
        adjacency_f1=float(adjacency_f1),
        directed_f1=float(directed_f1),
        orientation_accuracy_on_recovered_edges=float(orientation),
        graph_type=str(graph.get("graph_type") or ""),
    )


def _extract_graph_structure(
    adj: np.ndarray,
    nodes: list[str],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    directed_edges: set[tuple[str, str]] = set()
    undirected_pairs: set[tuple[str, str]] = set()
    adjacency_pairs: set[tuple[str, str]] = set()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            left = adj[i][j]
            right = adj[j][i]
            if left == 0 and right == 0:
                continue
            src = nodes[i]
            dst = nodes[j]
            pair = _pair_key(src, dst)
            adjacency_pairs.add(pair)
            if left == -1 and right == 1:
                directed_edges.add((src, dst))
            elif left == 1 and right == -1:
                directed_edges.add((dst, src))
            elif left == -1 and right == -1:
                undirected_pairs.add(pair)
            elif left > 0 and right > 0:
                undirected_pairs.add(pair)
            elif left > 0 and right <= 0:
                directed_edges.add((src, dst))
            elif right > 0 and left <= 0:
                directed_edges.add((dst, src))
            else:
                undirected_pairs.add(pair)
    return directed_edges, undirected_pairs, adjacency_pairs


def _shd(
    truth_nodes: list[str],
    truth_directed: set[tuple[str, str]],
    truth_undirected: set[tuple[str, str]],
    pred_directed: set[tuple[str, str]],
    pred_undirected: set[tuple[str, str]],
    pred_adjacency: set[tuple[str, str]],
) -> int:
    truth_adjacency = {_pair_key(src, dst) for src, dst in truth_directed} | truth_undirected
    shd = 0
    for i, left in enumerate(truth_nodes):
        for right in truth_nodes[i + 1 :]:
            pair = _pair_key(left, right)
            if _truth_state(pair, truth_directed, truth_undirected, truth_adjacency) != _pred_state(
                pair,
                pred_directed,
                pred_undirected,
                pred_adjacency,
            ):
                shd += 1
    return shd


def _truth_state(
    pair: tuple[str, str],
    directed: set[tuple[str, str]],
    undirected: set[tuple[str, str]],
    adjacency: set[tuple[str, str]],
) -> str:
    if pair not in adjacency:
        return "none"
    if pair in undirected:
        return "undirected"
    left, right = pair
    if (left, right) in directed:
        return "forward"
    if (right, left) in directed:
        return "reverse"
    return "undirected"


def _pred_state(
    pair: tuple[str, str],
    directed: set[tuple[str, str]],
    undirected: set[tuple[str, str]],
    adjacency: set[tuple[str, str]],
) -> str:
    if pair not in adjacency:
        return "none"
    if pair in undirected:
        return "undirected"
    left, right = pair
    if (left, right) in directed:
        return "forward"
    if (right, left) in directed:
        return "reverse"
    return "undirected"


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _phase3_markdown(
    summary: pd.DataFrame,
    *,
    missing_eval: pd.DataFrame,
    complete_all9_names: list[str],
    exact_names: list[str],
    final_model_path: Path,
    best_feature_set_name: str,
    fixed_policy_path: Path,
    fixed_top3: tuple[str, ...],
) -> str:
    status_summary = (
        missing_eval.groupby(["algorithm_name", "run_status"])
        .size()
        .reset_index(name="count")
        .sort_values(["algorithm_name", "run_status"])
    )
    lines = ["# Phase 3 Timeout-Aware Evidence", ""]
    lines.append("## Missing Algorithm Runtime")
    lines.append("")
    lines.append(_table_markdown(status_summary))
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Exact datasets evaluated timeout-aware: {len(exact_names)}")
    lines.append(f"- Exact datasets with complete 9-algorithm coverage: {len(complete_all9_names)}")
    lines.append(f"- Complete 9-algorithm datasets: {', '.join(complete_all9_names) or 'none'}")
    lines.append("")
    lines.append("## Saved Artifacts")
    lines.append("")
    lines.append(f"- Phase 3 selector model: `{final_model_path}`")
    lines.append(f"- Selector feature set: `{best_feature_set_name}`")
    lines.append(f"- Best fixed top-3 policy: `{fixed_policy_path}`")
    lines.append(f"- Best fixed top-3: {', '.join(fixed_top3)}")
    lines.append("")
    for split_name, group in summary.groupby("split_name", sort=True):
        lines.append(f"## {split_name}")
        lines.append("")
        display = group.drop(columns=["split_name"]).rename(columns={"method": "policy"})
        lines.append(_table_markdown(display))
    return "\n".join(lines)


def _table_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows.\n"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)
