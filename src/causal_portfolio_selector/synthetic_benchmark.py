from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import random
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from .artifacts import imported_paths, load_import_manifest
from .config import AppConfig
from .evaluation import evaluate_prediction
from .features import FEATURE_COLUMNS, extract_dataset_features
from .learned.featurize import dataframe_to_learned_inputs
from .learned.fingerprint import extract_fingerprint, learned_feature_columns
from .learned.model import load_biaffine_encoder, train_biaffine_encoder
from .learned.synthetic import SyntheticExample
from .models import (
    load_selector,
    save_selector,
    train_score_selector,
    train_selector,
    train_top3_combination_selector,
    train_top3_membership_selector,
)
from .phase3 import (
    ALL_ALGORITHMS,
    GraphEvaluation,
    _assign_quality_rank_by_shd,
    _combined_score,
    _evaluate_graph,
    _filter_prediction_to_available,
    _rank_by_best_fixed_top3,
    build_timeout_aware_targets,
)
from .targets import exact_dataset_names, load_or_build_tables


GRAPH_FAMILIES: tuple[str, ...] = (
    "erdos_renyi_sparse",
    "erdos_renyi_dense",
    "scale_free",
    "layered_dag",
    "chain_heavy",
    "collider_heavy",
    "hub_spoke",
)
NODE_BUCKETS: tuple[tuple[int, int], ...] = (
    (4, 200),
    (6, 200),
    (8, 250),
    (10, 300),
    (15, 300),
    (20, 300),
    (30, 250),
    (40, 200),
)
SAMPLE_SIZES: tuple[int, ...] = (500, 1000, 3000, 5000)
ALPHA_PROFILES: dict[str, tuple[float, float]] = {
    "sharp": (0.3, 0.8),
    "medium": (1.0, 2.0),
    "smooth": (3.0, 6.0),
}


@dataclass(frozen=True)
class SyntheticGenerateOptions:
    output: Path
    count: int = 2000
    max_nodes: int = 40
    seed: int = 42
    overwrite: bool = False


@dataclass(frozen=True)
class SyntheticRunOptions:
    synthetic_root: Path
    output: Path
    algorithms: tuple[str, ...] = ALL_ALGORITHMS
    timeout_seconds: int = 300
    shard_index: int = 0
    shard_count: int = 1
    resume: bool = True
    overwrite: bool = False
    random_seed: int = 42


def generate_synthetic_bn_suite(options: SyntheticGenerateOptions) -> dict[str, Path]:
    output = options.output.expanduser().resolve()
    datasets_dir = output / "datasets"
    truth_dir = output / "ground_truth"
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not options.overwrite:
        raise FileExistsError(
            f"Synthetic manifest already exists at {manifest_path}. "
            "Pass --overwrite to regenerate."
        )
    datasets_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    n_vars_schedule = _node_schedule(options.count, options.max_nodes)
    for index, n_vars in enumerate(n_vars_schedule):
        seed = int(options.seed + index * 104729)
        rng = np.random.default_rng(seed)
        dataset_name = f"synthetic_{index:06d}"
        graph_family = GRAPH_FAMILIES[index % len(GRAPH_FAMILIES)]
        n_samples = int(SAMPLE_SIZES[(index // len(GRAPH_FAMILIES)) % len(SAMPLE_SIZES)])
        max_indegree = 4
        adjacency = _sample_heterogeneous_dag(
            n_vars,
            graph_family=graph_family,
            max_indegree=max_indegree,
            rng=rng,
        )
        cardinalities = _sample_cardinalities(n_vars, rng)
        alpha_profile = tuple(ALPHA_PROFILES)[index % len(ALPHA_PROFILES)]
        alpha_range = ALPHA_PROFILES[alpha_profile]
        frame = _sample_discrete_bn_with_alpha(
            adjacency,
            cardinalities,
            n_samples,
            alpha_range=alpha_range,
            rng=rng,
        )
        dataset_path = datasets_dir / f"{dataset_name}.csv"
        truth_path = truth_dir / f"{dataset_name}.json"
        frame.to_csv(dataset_path, index=False)

        nodes = [f"X{i}" for i in range(n_vars)]
        directed_edges = [
            [nodes[src], nodes[dst]]
            for src, dst in zip(*np.nonzero(adjacency))
        ]
        density = _dag_density(adjacency)
        truth_payload = {
            "dataset_name": dataset_name,
            "nodes": nodes,
            "graph_type": "dag",
            "directed_edges": directed_edges,
            "undirected_edges": [],
            "dag_edges": directed_edges,
            "truth_type": "synthetic_bn",
            "metadata": {
                "graph_family": graph_family,
                "n_vars": int(n_vars),
                "n_samples": int(n_samples),
                "density": float(density),
                "max_indegree": int(max_indegree),
                "cardinalities": cardinalities.astype(int).tolist(),
                "alpha_profile": alpha_profile,
                "alpha_range": list(alpha_range),
                "seed": seed,
            },
        }
        truth_path.write_text(json.dumps(truth_payload, indent=2, sort_keys=True) + "\n")
        entries.append(
            {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path.relative_to(output)),
                "ground_truth_path": str(truth_path.relative_to(output)),
                "n_samples": int(n_samples),
                "n_features": int(n_vars),
                "truth_type": "synthetic_bn",
                "graph_family": graph_family,
                "density": float(density),
                "max_indegree": int(max_indegree),
                "alpha_profile": alpha_profile,
                "seed": seed,
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(entries),
        "seed": int(options.seed),
        "max_nodes": int(options.max_nodes),
        "datasets": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "synthetic_root": output,
        "datasets_dir": datasets_dir,
        "ground_truth_dir": truth_dir,
        "manifest": manifest_path,
    }


def run_synthetic_algorithm_suite(options: SyntheticRunOptions) -> dict[str, Path]:
    synthetic_root = options.synthetic_root.expanduser().resolve()
    output = options.output.expanduser().resolve()
    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_synthetic_manifest(synthetic_root)
    entries = _shard_entries(
        manifest["datasets"],
        shard_index=options.shard_index,
        shard_count=options.shard_count,
    )
    algorithms = _validate_algorithms(options.algorithms)

    for entry in entries:
        dataset_name = str(entry["dataset_name"])
        dataset_path = synthetic_root / str(entry["dataset_path"])
        for algorithm in algorithms:
            record_path = records_dir / f"{dataset_name}__{algorithm}.json"
            if options.resume and not options.overwrite and record_path.exists():
                continue
            record = _run_one_algorithm_with_timeout(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                algorithm_name=algorithm,
                timeout_seconds=int(options.timeout_seconds),
                random_seed=int(options.random_seed),
            )
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    summary_path = _write_run_summary(output)
    run_manifest_path = output / "manifest.json"
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_root": str(synthetic_root),
        "output": str(output),
        "records_dir": str(records_dir),
        "summary_path": str(summary_path),
        "algorithms": list(algorithms),
        "timeout_seconds": int(options.timeout_seconds),
        "shard_index": int(options.shard_index),
        "shard_count": int(options.shard_count),
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    return {
        "output_dir": output,
        "records_dir": records_dir,
        "summary": summary_path,
        "manifest": run_manifest_path,
    }


def build_synthetic_training_tables(
    config: AppConfig,
    *,
    synthetic_root: str | Path,
    runs: str | Path,
    output: str | Path,
) -> dict[str, Path]:
    synthetic_root = Path(synthetic_root).expanduser().resolve()
    runs = Path(runs).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_synthetic_manifest(synthetic_root)
    split_map = _synthetic_split_map(manifest["datasets"], seed=config.model.random_state)

    feature_rows: list[dict[str, Any]] = []
    for entry in manifest["datasets"]:
        dataset_name = str(entry["dataset_name"])
        dataset_path = synthetic_root / str(entry["dataset_path"])
        feature_rows.append(
            {
                "dataset_name": dataset_name,
                **extract_dataset_features(dataset_path, config=config.features),
            }
        )
    features = pd.DataFrame(feature_rows)

    eval_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    meta_by_name = {str(entry["dataset_name"]): entry for entry in manifest["datasets"]}
    for record_path in sorted((runs / "records").glob("*.json")):
        record = json.loads(record_path.read_text())
        dataset_name = str(record["dataset_name"])
        if dataset_name not in meta_by_name:
            continue
        entry = meta_by_name[dataset_name]
        truth = _load_synthetic_truth(synthetic_root / str(entry["ground_truth_path"]))
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
        eval_rows.append(
            {
                "dataset_name": dataset_name,
                "algorithm_name": str(record["algorithm_name"]),
                "run_status": record.get("status"),
                "evaluation_status": evaluation.status,
                "runtime_seconds": record.get("runtime_seconds"),
                "timeout_seconds": record.get("timeout_seconds"),
                "n_features": entry.get("n_features"),
                "n_samples": entry.get("n_samples"),
                "graph_family": entry.get("graph_family"),
                "shd": evaluation.shd,
                "adjacency_f1": evaluation.adjacency_f1,
                "directed_f1": evaluation.directed_f1,
                "orientation_accuracy_on_recovered_edges": evaluation.orientation_accuracy_on_recovered_edges,
                "graph_type": evaluation.graph_type,
                "error": record.get("error"),
            }
        )
        if evaluation.status != "evaluated" or evaluation.shd is None:
            continue
        possible_edges = _possible_edges(int(entry["n_features"]))
        target_rows.append(
            {
                "dataset_name": dataset_name,
                "algorithm_name": str(record["algorithm_name"]),
                "truth_type": "synthetic_bn",
                "split_role": split_map[dataset_name],
                "shd": float(evaluation.shd),
                "combined_score": _combined_score(evaluation, possible_edges=possible_edges),
                "possible_edges": possible_edges,
            }
        )

    targets = pd.DataFrame(target_rows)
    if targets.empty:
        raise ValueError("No successful synthetic algorithm evaluations found.")
    targets["oracle_shd"] = targets.groupby("dataset_name")["shd"].transform("min")
    targets["relative_regret"] = (
        (targets["shd"] - targets["oracle_shd"]) / targets["possible_edges"].clip(lower=1.0)
    )
    targets = _assign_quality_rank_by_shd(targets)
    targets = targets[
        [
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
    ].sort_values(["dataset_name", "quality_rank", "algorithm_name"], kind="mergesort")

    split_rows = [
        {"dataset_name": name, "split_role": role}
        for name, role in sorted(split_map.items())
    ]
    paths = {
        "features": output / "features.csv",
        "targets": output / "targets.csv",
        "run_evaluations": output / "run_evaluations.csv",
        "splits": output / "splits.csv",
    }
    features.to_csv(paths["features"], index=False)
    targets.to_csv(paths["targets"], index=False)
    pd.DataFrame(eval_rows).to_csv(paths["run_evaluations"], index=False)
    pd.DataFrame(split_rows).to_csv(paths["splits"], index=False)
    return paths


def train_fingerprint_from_synthetic(
    config: AppConfig,
    *,
    synthetic_root: str | Path,
    output: str | Path,
    device: str | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    synthetic_root = Path(synthetic_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    learned_config = config.learned
    updates: dict[str, Any] = {}
    if device is not None:
        updates["device"] = device
    if epochs is not None:
        updates["epochs"] = int(epochs)
    if updates:
        learned_config = replace(learned_config, **updates)
    examples = _synthetic_examples_from_manifest(
        synthetic_root,
        max_rows=learned_config.max_feature_rows,
        random_seed=learned_config.random_seed,
    )
    result = train_biaffine_encoder(examples, learned_config, output_path=output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "synthetic_root": str(synthetic_root),
                "synthetic_graph_count": len(examples),
                "model_path": result["model_path"],
                "device": result["device"],
                "final_metrics": result["final_metrics"],
                "config": asdict(learned_config),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {**result, "manifest_path": str(manifest_path)}


def _handcrafted_feature_sets(feature_table: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    missing = [column for column in FEATURE_COLUMNS if column not in feature_table.columns]
    if missing:
        raise ValueError(f"Feature table is missing handcrafted columns: {missing}")
    knn_columns = tuple(column for column in feature_table.columns if column.startswith("knn_"))
    if knn_columns:
        return {"handcrafted_plus_knn": (*FEATURE_COLUMNS, *knn_columns)}
    return {"handcrafted_all": FEATURE_COLUMNS}


def train_synthetic_selector(
    config: AppConfig,
    *,
    tables: str | Path,
    encoder: str | Path | None,
    output: str | Path,
) -> dict[str, Path]:
    tables = Path(tables).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    feature_table = pd.read_csv(tables / "features.csv")
    targets = pd.read_csv(tables / "targets.csv")
    algorithms = _available_algorithms_from_targets(targets)
    combined = feature_table
    base_feature_sets = _handcrafted_feature_sets(feature_table)
    feature_sets: dict[str, tuple[str, ...]] = dict(base_feature_sets)
    if encoder is not None:
        learned = _build_synthetic_learned_feature_table(
            config,
            tables=tables,
            encoder=Path(encoder).expanduser().resolve(),
        )
        combined = feature_table.merge(learned, on="dataset_name", how="inner", validate="one_to_one")
        learned_cols = tuple(column for column in learned.columns if column.startswith("lf_"))
        feature_sets["learned_only"] = learned_cols
        for base_name, base_columns in base_feature_sets.items():
            feature_sets[f"{base_name}_plus_learned"] = (*base_columns, *learned_cols)
        combined.to_csv(tables / "features_plus_learned.csv", index=False)

    train_names = sorted(targets.loc[targets["split_role"] == "synthetic_train", "dataset_name"].unique())
    val_names = sorted(targets.loc[targets["split_role"] == "synthetic_val", "dataset_name"].unique())
    test_names = sorted(targets.loc[targets["split_role"] == "synthetic_test", "dataset_name"].unique())
    if not train_names:
        raise ValueError("No synthetic_train datasets found in targets.")
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    selectors: dict[str, Any] = {}
    for feature_set_name, feature_names in feature_sets.items():
        selector = train_selector(
            combined,
            targets,
            dataset_names=train_names,
            algorithms=algorithms,
            config=config.model,
            feature_names=feature_names,
        )
        selectors[feature_set_name] = selector
        validation_rows.extend(
            _evaluate_selector_rows(
                selector,
                combined,
                targets,
                dataset_names=val_names,
                split_name="synthetic_val",
                method=feature_set_name,
            )
        )
        test_rows.extend(
            _evaluate_selector_rows(
                selector,
                combined,
                targets,
                dataset_names=test_names,
                split_name="synthetic_test",
                method=feature_set_name,
            )
        )

    validation_metrics = pd.DataFrame(validation_rows)
    test_metrics = pd.DataFrame(test_rows)
    if validation_metrics.empty:
        raise ValueError("No validation metrics were produced.")
    validation_summary = _aggregate_metrics_by_method(validation_metrics)
    best_feature_set = _choose_best_feature_set(validation_summary)
    final_train_names = sorted(set(train_names) | set(val_names))
    best_selector = train_selector(
        combined,
        targets,
        dataset_names=final_train_names,
        algorithms=algorithms,
        config=config.model,
        feature_names=feature_sets[best_feature_set],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_selector(best_selector, output)

    paths = {
        "selector": output,
        "validation_metrics": output.parent / "selector_validation_metrics.csv",
        "validation_summary": output.parent / "selector_validation_summary.csv",
        "test_metrics": output.parent / "selector_test_metrics.csv",
        "test_summary": output.parent / "selector_test_summary.csv",
        "manifest": output.parent / "selector_manifest.json",
    }
    validation_metrics.to_csv(paths["validation_metrics"], index=False)
    validation_summary.to_csv(paths["validation_summary"], index=False)
    test_metrics.to_csv(paths["test_metrics"], index=False)
    _aggregate_metrics_by_method(test_metrics).to_csv(paths["test_summary"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "model_path": str(output),
                "tables": str(tables),
                "encoder": str(encoder) if encoder is not None else None,
                "feature_set": best_feature_set,
                "feature_names": list(feature_sets[best_feature_set]),
                "train_dataset_count": len(train_names),
                "validation_dataset_count": len(val_names),
                "test_dataset_count": len(test_names),
                "algorithms": list(algorithms),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return paths


def train_synthetic_top3_selector(
    config: AppConfig,
    *,
    tables: str | Path,
    encoder: str | Path | None,
    output: str | Path,
) -> dict[str, Path]:
    tables = Path(tables).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    feature_table = pd.read_csv(tables / "features.csv")
    targets = pd.read_csv(tables / "targets.csv")
    algorithms = _available_algorithms_from_targets(targets)
    combined = feature_table
    base_feature_sets = _handcrafted_feature_sets(feature_table)
    feature_sets: dict[str, tuple[str, ...]] = dict(base_feature_sets)
    if encoder is not None:
        learned = _build_synthetic_learned_feature_table(
            config,
            tables=tables,
            encoder=Path(encoder).expanduser().resolve(),
        )
        combined = feature_table.merge(learned, on="dataset_name", how="inner", validate="one_to_one")
        learned_cols = tuple(column for column in learned.columns if column.startswith("lf_"))
        feature_sets["learned_only"] = learned_cols
        for base_name, base_columns in base_feature_sets.items():
            feature_sets[f"{base_name}_plus_learned"] = (*base_columns, *learned_cols)
        combined.to_csv(tables / "features_plus_learned.csv", index=False)

    train_names = sorted(targets.loc[targets["split_role"] == "synthetic_train", "dataset_name"].unique())
    val_names = sorted(targets.loc[targets["split_role"] == "synthetic_val", "dataset_name"].unique())
    test_names = sorted(targets.loc[targets["split_role"] == "synthetic_test", "dataset_name"].unique())
    if not train_names:
        raise ValueError("No synthetic_train datasets found in targets.")
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for feature_set_name, feature_names in feature_sets.items():
        selector = train_top3_membership_selector(
            combined,
            targets,
            dataset_names=train_names,
            algorithms=algorithms,
            config=config.model,
            feature_names=feature_names,
        )
        validation_rows.extend(
            _evaluate_selector_rows(
                selector,
                combined,
                targets,
                dataset_names=val_names,
                split_name="synthetic_val",
                method=feature_set_name,
            )
        )
        test_rows.extend(
            _evaluate_selector_rows(
                selector,
                combined,
                targets,
                dataset_names=test_names,
                split_name="synthetic_test",
                method=feature_set_name,
            )
        )

    validation_metrics = pd.DataFrame(validation_rows)
    test_metrics = pd.DataFrame(test_rows)
    if validation_metrics.empty:
        raise ValueError("No validation metrics were produced.")
    validation_summary = _aggregate_metrics_by_method(validation_metrics)
    best_feature_set = _choose_best_top3_feature_set(validation_summary)
    final_train_names = sorted(set(train_names) | set(val_names))
    best_selector = train_top3_membership_selector(
        combined,
        targets,
        dataset_names=final_train_names,
        algorithms=algorithms,
        config=config.model,
        feature_names=feature_sets[best_feature_set],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_selector(best_selector, output)

    paths = {
        "selector": output,
        "validation_metrics": output.parent / "selector_validation_metrics.csv",
        "validation_summary": output.parent / "selector_validation_summary.csv",
        "test_metrics": output.parent / "selector_test_metrics.csv",
        "test_summary": output.parent / "selector_test_summary.csv",
        "manifest": output.parent / "selector_manifest.json",
    }
    validation_metrics.to_csv(paths["validation_metrics"], index=False)
    validation_summary.to_csv(paths["validation_summary"], index=False)
    test_metrics.to_csv(paths["test_metrics"], index=False)
    _aggregate_metrics_by_method(test_metrics).to_csv(paths["test_summary"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "model_path": str(output),
                "tables": str(tables),
                "encoder": str(encoder) if encoder is not None else None,
                "objective": "top3_membership",
                "feature_set": best_feature_set,
                "feature_names": list(feature_sets[best_feature_set]),
                "train_dataset_count": len(train_names),
                "validation_dataset_count": len(val_names),
                "test_dataset_count": len(test_names),
                "algorithms": list(algorithms),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return paths


def train_synthetic_score_selector(
    config: AppConfig,
    *,
    tables: str | Path,
    encoder: str | Path | None,
    output: str | Path,
) -> dict[str, Path]:
    tables = Path(tables).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    feature_table = pd.read_csv(tables / "features.csv")
    targets = pd.read_csv(tables / "targets.csv")
    algorithms = _available_algorithms_from_targets(targets)
    combined = feature_table
    base_feature_sets = _handcrafted_feature_sets(feature_table)
    feature_sets: dict[str, tuple[str, ...]] = dict(base_feature_sets)
    if encoder is not None:
        learned = _build_synthetic_learned_feature_table(
            config,
            tables=tables,
            encoder=Path(encoder).expanduser().resolve(),
        )
        combined = feature_table.merge(learned, on="dataset_name", how="inner", validate="one_to_one")
        learned_cols = tuple(column for column in learned.columns if column.startswith("lf_"))
        feature_sets["learned_only"] = learned_cols
        for base_name, base_columns in base_feature_sets.items():
            feature_sets[f"{base_name}_plus_learned"] = (*base_columns, *learned_cols)
        combined.to_csv(tables / "features_plus_learned.csv", index=False)

    train_names = sorted(targets.loc[targets["split_role"] == "synthetic_train", "dataset_name"].unique())
    val_names = sorted(targets.loc[targets["split_role"] == "synthetic_val", "dataset_name"].unique())
    test_names = sorted(targets.loc[targets["split_role"] == "synthetic_test", "dataset_name"].unique())
    if not train_names:
        raise ValueError("No synthetic_train datasets found in targets.")
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    score_column = "combined_score"
    for feature_set_name, feature_names in feature_sets.items():
        selector = train_score_selector(
            combined,
            targets,
            dataset_names=train_names,
            algorithms=algorithms,
            score_column=score_column,
            config=config.model,
            feature_names=feature_names,
        )
        validation_rows.extend(
            _evaluate_selector_rows(
                selector,
                combined,
                targets,
                dataset_names=val_names,
                split_name="synthetic_val",
                method=feature_set_name,
            )
        )
        test_rows.extend(
            _evaluate_selector_rows(
                selector,
                combined,
                targets,
                dataset_names=test_names,
                split_name="synthetic_test",
                method=feature_set_name,
            )
        )

    validation_metrics = pd.DataFrame(validation_rows)
    test_metrics = pd.DataFrame(test_rows)
    if validation_metrics.empty:
        raise ValueError("No validation metrics were produced.")
    validation_summary = _aggregate_metrics_by_method(validation_metrics)
    best_feature_set = _choose_best_top3_feature_set(validation_summary)
    final_train_names = sorted(set(train_names) | set(val_names))
    best_selector = train_score_selector(
        combined,
        targets,
        dataset_names=final_train_names,
        algorithms=algorithms,
        score_column=score_column,
        config=config.model,
        feature_names=feature_sets[best_feature_set],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_selector(best_selector, output)

    paths = {
        "selector": output,
        "validation_metrics": output.parent / "selector_validation_metrics.csv",
        "validation_summary": output.parent / "selector_validation_summary.csv",
        "test_metrics": output.parent / "selector_test_metrics.csv",
        "test_summary": output.parent / "selector_test_summary.csv",
        "manifest": output.parent / "selector_manifest.json",
    }
    validation_metrics.to_csv(paths["validation_metrics"], index=False)
    validation_summary.to_csv(paths["validation_summary"], index=False)
    test_metrics.to_csv(paths["test_metrics"], index=False)
    _aggregate_metrics_by_method(test_metrics).to_csv(paths["test_summary"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "model_path": str(output),
                "tables": str(tables),
                "encoder": str(encoder) if encoder is not None else None,
                "objective": "score_regression",
                "score_column": score_column,
                "feature_set": best_feature_set,
                "feature_names": list(feature_sets[best_feature_set]),
                "train_dataset_count": len(train_names),
                "validation_dataset_count": len(val_names),
                "test_dataset_count": len(test_names),
                "algorithms": list(algorithms),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return paths


def train_synthetic_top3_combination_selector(
    config: AppConfig,
    *,
    tables: str | Path,
    encoder: str | Path | None,
    output: str | Path,
    oracle_weight: float = 3.0,
    overlap_weight: float = 1.0,
    overlap_at_least_2_weight: float = 0.0,
    regret_weight: float = 0.25,
) -> dict[str, Path]:
    tables = Path(tables).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    feature_table = pd.read_csv(tables / "features.csv")
    targets = pd.read_csv(tables / "targets.csv")
    algorithms = _available_algorithms_from_targets(targets)
    combined = feature_table
    base_feature_sets = _handcrafted_feature_sets(feature_table)
    feature_sets: dict[str, tuple[str, ...]] = dict(base_feature_sets)
    if encoder is not None:
        learned = _build_synthetic_learned_feature_table(
            config,
            tables=tables,
            encoder=Path(encoder).expanduser().resolve(),
        )
        combined = feature_table.merge(learned, on="dataset_name", how="inner", validate="one_to_one")
        learned_cols = tuple(column for column in learned.columns if column.startswith("lf_"))
        feature_sets["learned_only"] = learned_cols
        for base_name, base_columns in base_feature_sets.items():
            feature_sets[f"{base_name}_plus_learned"] = (*base_columns, *learned_cols)
        combined.to_csv(tables / "features_plus_learned.csv", index=False)

    train_names = sorted(targets.loc[targets["split_role"] == "synthetic_train", "dataset_name"].unique())
    val_names = sorted(targets.loc[targets["split_role"] == "synthetic_val", "dataset_name"].unique())
    test_names = sorted(targets.loc[targets["split_role"] == "synthetic_test", "dataset_name"].unique())
    if not train_names:
        raise ValueError("No synthetic_train datasets found in targets.")
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for feature_set_name, feature_names in feature_sets.items():
        selector = train_top3_combination_selector(
            combined,
            targets,
            dataset_names=train_names,
            algorithms=algorithms,
            config=config.model,
            feature_names=feature_names,
            oracle_weight=oracle_weight,
            overlap_weight=overlap_weight,
            overlap_at_least_2_weight=overlap_at_least_2_weight,
            regret_weight=regret_weight,
        )
        validation_rows.extend(
            _evaluate_selector_rows(
                selector,
                combined,
                targets,
                dataset_names=val_names,
                split_name="synthetic_val",
                method=feature_set_name,
            )
        )
        test_rows.extend(
            _evaluate_selector_rows(
                selector,
                combined,
                targets,
                dataset_names=test_names,
                split_name="synthetic_test",
                method=feature_set_name,
            )
        )

    validation_metrics = pd.DataFrame(validation_rows)
    test_metrics = pd.DataFrame(test_rows)
    if validation_metrics.empty:
        raise ValueError("No validation metrics were produced.")
    validation_summary = _aggregate_metrics_by_method(validation_metrics)
    best_feature_set = _choose_best_top3_feature_set(validation_summary)
    final_train_names = sorted(set(train_names) | set(val_names))
    best_selector = train_top3_combination_selector(
        combined,
        targets,
        dataset_names=final_train_names,
        algorithms=algorithms,
        config=config.model,
        feature_names=feature_sets[best_feature_set],
        oracle_weight=oracle_weight,
        overlap_weight=overlap_weight,
        overlap_at_least_2_weight=overlap_at_least_2_weight,
        regret_weight=regret_weight,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_selector(best_selector, output)

    paths = {
        "selector": output,
        "validation_metrics": output.parent / "selector_validation_metrics.csv",
        "validation_summary": output.parent / "selector_validation_summary.csv",
        "test_metrics": output.parent / "selector_test_metrics.csv",
        "test_summary": output.parent / "selector_test_summary.csv",
        "manifest": output.parent / "selector_manifest.json",
    }
    validation_metrics.to_csv(paths["validation_metrics"], index=False)
    validation_summary.to_csv(paths["validation_summary"], index=False)
    test_metrics.to_csv(paths["test_metrics"], index=False)
    _aggregate_metrics_by_method(test_metrics).to_csv(paths["test_summary"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            {
                "model_path": str(output),
                "tables": str(tables),
                "encoder": str(encoder) if encoder is not None else None,
                "objective": "top3_combination_reward",
                "reward_weights": {
                    "oracle_in_top3": float(oracle_weight),
                    "top3_overlap": float(overlap_weight),
                    "top3_overlap_at_least_2": float(overlap_at_least_2_weight),
                    "regret_at_3": float(regret_weight),
                },
                "feature_set": best_feature_set,
                "feature_names": list(feature_sets[best_feature_set]),
                "train_dataset_count": len(train_names),
                "validation_dataset_count": len(val_names),
                "test_dataset_count": len(test_names),
                "algorithms": list(algorithms),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return paths


def _available_algorithms_from_targets(targets: pd.DataFrame) -> tuple[str, ...]:
    available = {str(name) for name in targets["algorithm_name"].dropna().unique()}
    algorithms = tuple(name for name in ALL_ALGORITHMS if name in available)
    extras = tuple(sorted(available - set(algorithms)))
    result = (*algorithms, *extras)
    if not result:
        raise ValueError("No algorithms were found in targets.")
    return result


def evaluate_synthetic_selector_on_exact(
    config: AppConfig,
    *,
    model: str | Path,
    encoder: str | Path | None,
    output: str | Path,
) -> dict[str, Path]:
    root = config.resolved_root()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selector = load_selector(model)
    base_features, base_targets = load_or_build_tables(config)
    exact_targets_path = root / "artifacts" / "tables" / "targets_timeout_aware.csv"
    if exact_targets_path.exists():
        exact_targets = pd.read_csv(exact_targets_path)
    else:
        exact_targets, _ = build_timeout_aware_targets(config, base_targets)
    exact_names = exact_dataset_names(
        exact_targets,
        external_datasets=set(config.external_datasets),
    )
    feature_table = _exact_feature_table_for_selector(
        config,
        base_features,
        selector_feature_names=tuple(selector.feature_names),
        encoder=Path(encoder).expanduser().resolve() if encoder is not None else None,
    )
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for dataset_name in exact_names:
        feature_row = feature_table[feature_table["dataset_name"] == dataset_name].iloc[0]
        raw_prediction = selector.predict_from_features(feature_row.to_dict())
        prediction = _filter_prediction_to_available(raw_prediction, exact_targets, dataset_name)
        metrics = evaluate_prediction(
            dataset_name=dataset_name,
            prediction=prediction,
            targets=exact_targets,
        )
        metrics.update({"split_name": "exact_final", "method": "synthetic_selector"})
        metric_rows.append(metrics)
        for algorithm, rank in dict(prediction["ranking"]).items():
            prediction_rows.append(
                {
                    "dataset_name": dataset_name,
                    "method": "synthetic_selector",
                    "algorithm_name": algorithm,
                    "predicted_rank": rank,
                }
            )

    fixed_order = _load_fixed_policy_order(root, exact_targets, exact_names)
    for dataset_name in exact_names:
        prediction = _filter_order_to_available(fixed_order, exact_targets, dataset_name)
        metrics = evaluate_prediction(
            dataset_name=dataset_name,
            prediction=prediction,
            targets=exact_targets,
        )
        metrics.update({"split_name": "exact_final", "method": "phase3_fixed_policy"})
        metric_rows.append(metrics)

    metrics = pd.DataFrame(metric_rows)
    summary = _aggregate_metrics_by_method(metrics)
    paths = {
        "metrics": output / "exact_metrics.csv",
        "summary": output / "exact_summary.csv",
        "predictions": output / "exact_predictions.csv",
        "report": output / "exact_report.md",
    }
    metrics.to_csv(paths["metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(prediction_rows).to_csv(paths["predictions"], index=False)
    paths["report"].write_text(_exact_report_markdown(summary, exact_names))
    return paths


def _node_schedule(count: int, max_nodes: int) -> list[int]:
    buckets = [(nodes, weight) for nodes, weight in NODE_BUCKETS if nodes <= max_nodes]
    if not buckets:
        raise ValueError(f"No node buckets available for max_nodes={max_nodes}.")
    weights = np.asarray([weight for _, weight in buckets], dtype=float)
    raw = weights / weights.sum() * int(count)
    counts = np.floor(raw).astype(int)
    remainder = int(count) - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for idx in order[:remainder]:
        counts[idx] += 1
    schedule: list[int] = []
    for (nodes, _), bucket_count in zip(buckets, counts):
        schedule.extend([int(nodes)] * int(bucket_count))
    return schedule[:count]


def _sample_cardinalities(n_vars: int, rng: np.random.Generator) -> np.ndarray:
    high = rng.random(n_vars) >= 0.70
    values = np.empty(n_vars, dtype=np.int16)
    values[~high] = rng.integers(2, 5, size=int((~high).sum()))
    values[high] = rng.integers(5, 9, size=int(high.sum()))
    return values


def _sample_heterogeneous_dag(
    n_vars: int,
    *,
    graph_family: str,
    max_indegree: int,
    rng: np.random.Generator,
) -> np.ndarray:
    order = rng.permutation(n_vars).tolist()
    adjacency = np.zeros((n_vars, n_vars), dtype=np.int8)
    if graph_family.startswith("erdos_renyi"):
        if graph_family.endswith("sparse"):
            p = float(rng.uniform(0.05, 0.12))
        else:
            p = float(rng.uniform(0.22, 0.32))
        _add_ordered_random_edges(adjacency, order, p, rng)
    elif graph_family == "scale_free":
        degrees = np.ones(n_vars, dtype=float)
        for position in range(1, n_vars):
            child = int(order[position])
            candidates = [int(node) for node in order[:position]]
            parent_count = int(rng.integers(0, min(max_indegree, len(candidates)) + 1))
            if parent_count <= 0:
                continue
            weights = np.asarray([degrees[node] for node in candidates], dtype=float)
            weights = weights / weights.sum()
            parents = rng.choice(candidates, size=parent_count, replace=False, p=weights)
            for parent in parents:
                adjacency[int(parent), child] = 1
                degrees[int(parent)] += 1.0
                degrees[child] += 1.0
    elif graph_family == "layered_dag":
        layer_count = int(rng.integers(3, min(6, n_vars) + 1))
        layers = np.array_split(order, layer_count)
        for src_idx, src_layer in enumerate(layers[:-1]):
            for dst_layer in layers[src_idx + 1 :]:
                for parent in src_layer:
                    for child in dst_layer:
                        if rng.random() <= float(rng.uniform(0.10, 0.24)):
                            adjacency[int(parent), int(child)] = 1
    elif graph_family == "chain_heavy":
        for left, right in zip(order[:-1], order[1:]):
            adjacency[int(left), int(right)] = 1
        _add_ordered_random_edges(adjacency, order, float(rng.uniform(0.04, 0.10)), rng)
    elif graph_family == "collider_heavy":
        for position in range(2, n_vars):
            if rng.random() <= 0.45:
                child = int(order[position])
                parents = rng.choice(order[:position], size=2, replace=False)
                for parent in parents:
                    adjacency[int(parent), child] = 1
        _add_ordered_random_edges(adjacency, order, float(rng.uniform(0.04, 0.10)), rng)
    elif graph_family == "hub_spoke":
        hub_count = max(1, min(3, n_vars // 6))
        hubs = [int(node) for node in order[:hub_count]]
        for parent in hubs:
            for child in order[hub_count:]:
                if rng.random() <= float(rng.uniform(0.20, 0.45)):
                    adjacency[parent, int(child)] = 1
        _add_ordered_random_edges(adjacency, order, float(rng.uniform(0.03, 0.08)), rng)
    else:
        raise ValueError(f"Unknown graph family: {graph_family}")
    _cap_indegree(adjacency, max_indegree=max_indegree, rng=rng)
    if adjacency.sum() == 0 and n_vars > 1:
        adjacency[int(order[0]), int(order[1])] = 1
    return adjacency


def _add_ordered_random_edges(
    adjacency: np.ndarray,
    order: list[int],
    probability: float,
    rng: np.random.Generator,
) -> None:
    for left_pos in range(len(order)):
        for right_pos in range(left_pos + 1, len(order)):
            if rng.random() <= probability:
                adjacency[int(order[left_pos]), int(order[right_pos])] = 1


def _cap_indegree(adjacency: np.ndarray, *, max_indegree: int, rng: np.random.Generator) -> None:
    for child in range(adjacency.shape[0]):
        parents = np.flatnonzero(adjacency[:, child])
        if parents.size <= max_indegree:
            continue
        keep = set(rng.choice(parents, size=max_indegree, replace=False).tolist())
        for parent in parents:
            if int(parent) not in keep:
                adjacency[int(parent), child] = 0


def _sample_discrete_bn_with_alpha(
    adjacency: np.ndarray,
    cardinalities: np.ndarray,
    n_samples: int,
    *,
    alpha_range: tuple[float, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    graph = nx.DiGraph(adjacency)
    order = list(nx.topological_sort(graph))
    cpts = []
    for node in range(adjacency.shape[0]):
        parents = np.flatnonzero(adjacency[:, node])
        parent_states = int(np.prod(cardinalities[parents])) if parents.size else 1
        alpha = float(rng.uniform(*alpha_range))
        cpt = rng.dirichlet(
            np.full(int(cardinalities[node]), alpha, dtype=float),
            size=parent_states,
        )
        cpts.append(cpt.astype(np.float32))
    data = np.zeros((n_samples, adjacency.shape[0]), dtype=np.int16)
    for node in order:
        parents = np.flatnonzero(adjacency[:, node])
        cpt = cpts[node]
        if parents.size == 0:
            data[:, node] = _sample_categorical(cpt[0], n_samples, rng)
            continue
        parent_indices = _parent_state_indices(data[:, parents], cardinalities[parents])
        for state_index in np.unique(parent_indices):
            rows = np.flatnonzero(parent_indices == state_index)
            data[rows, node] = _sample_categorical(cpt[int(state_index)], rows.size, rng)
    return pd.DataFrame(data, columns=[f"X{i}" for i in range(adjacency.shape[0])])


def _parent_state_indices(parent_values: np.ndarray, parent_cardinalities: np.ndarray) -> np.ndarray:
    multipliers = np.ones(parent_values.shape[1], dtype=np.int64)
    for idx in range(1, parent_values.shape[1]):
        multipliers[idx] = multipliers[idx - 1] * int(parent_cardinalities[idx - 1])
    return (parent_values.astype(np.int64) * multipliers).sum(axis=1)


def _sample_categorical(
    probabilities: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cumulative = np.cumsum(probabilities)
    return np.searchsorted(cumulative, rng.random(size), side="right").astype(np.int16)


def _dag_density(adjacency: np.ndarray) -> float:
    return float(adjacency.sum() / max(1, adjacency.shape[0] * (adjacency.shape[0] - 1) / 2))


def _load_synthetic_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Synthetic manifest not found: {path}")
    return json.loads(path.read_text())


def _load_synthetic_truth(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "dataset_name": payload.get("dataset_name"),
        "nodes": [str(node) for node in payload.get("nodes", [])],
        "directed_edges": [tuple(map(str, edge)) for edge in payload.get("directed_edges", [])],
        "undirected_edges": [tuple(map(str, edge)) for edge in payload.get("undirected_edges", [])],
        "graph_type": str(payload.get("graph_type") or "dag"),
    }


def _shard_entries(
    entries: Iterable[dict[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    selected = sorted(entries, key=lambda row: (int(row.get("n_features", 0)), str(row["dataset_name"])))
    return [entry for idx, entry in enumerate(selected) if idx % shard_count == shard_index]


def _validate_algorithms(algorithms: Iterable[str]) -> tuple[str, ...]:
    parsed = tuple(str(name) for name in algorithms)
    unknown = sorted(set(parsed) - set(ALL_ALGORITHMS))
    if unknown:
        raise ValueError(f"Unsupported algorithms: {unknown}. Expected subset of {ALL_ALGORITHMS}.")
    return parsed


def _run_one_algorithm_with_timeout(
    *,
    dataset_name: str,
    dataset_path: Path,
    algorithm_name: str,
    timeout_seconds: int,
    random_seed: int,
) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_algorithm_worker,
        kwargs={
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "algorithm_name": algorithm_name,
            "random_seed": random_seed,
            "queue": queue,
        },
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return {
            "dataset_name": dataset_name,
            "algorithm_name": algorithm_name,
            "status": "timeout",
            "runtime_seconds": float(timeout_seconds),
            "timeout_seconds": int(timeout_seconds),
            "graph_result": None,
            "error": f"Timed out after {timeout_seconds} seconds.",
        }
    if queue.empty():
        return {
            "dataset_name": dataset_name,
            "algorithm_name": algorithm_name,
            "status": "no_result",
            "runtime_seconds": None,
            "timeout_seconds": int(timeout_seconds),
            "graph_result": None,
            "error": "Worker exited without returning a result.",
        }
    record = queue.get()
    record["timeout_seconds"] = int(timeout_seconds)
    return record


def _algorithm_worker(
    *,
    dataset_name: str,
    dataset_path: str,
    algorithm_name: str,
    random_seed: int,
    queue: mp.Queue,
) -> None:
    start = time.perf_counter()
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            warnings.filterwarnings("ignore")
            random.seed(random_seed)
            np.random.seed(random_seed)
            try:
                frame = pd.read_csv(dataset_path)
                result = _run_algorithm(algorithm_name, frame)
                queue.put(
                    {
                        "dataset_name": dataset_name,
                        "algorithm_name": algorithm_name,
                        "status": "success",
                        "runtime_seconds": float(time.perf_counter() - start),
                        "n_samples": int(frame.shape[0]),
                        "n_features": int(frame.shape[1]),
                        "graph_result": result,
                        "metrics": _graph_metrics(result["adj_matrix"]),
                        "error": None,
                    }
                )
            except Exception as exc:  # pragma: no cover - optional dependency/env path
                queue.put(
                    {
                        "dataset_name": dataset_name,
                        "algorithm_name": algorithm_name,
                        "status": "run_failed",
                        "runtime_seconds": float(time.perf_counter() - start),
                        "graph_result": None,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                )


def _run_algorithm(algorithm_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    if algorithm_name == "PC_discrete":
        return _run_pc_discrete(frame)
    if algorithm_name == "FCI":
        return _run_fci(frame)
    if algorithm_name == "GES":
        return _run_ges(frame)
    if algorithm_name == "HC":
        return _run_hc(frame)
    if algorithm_name == "Tabu":
        return _run_tabu(frame)
    if algorithm_name == "K2":
        return _run_k2(frame)
    if algorithm_name == "MMHC":
        return _run_mmhc(frame)
    if algorithm_name == "BOSS":
        return _run_boss(frame)
    if algorithm_name == "GRaSP":
        return _run_grasp(frame)
    raise ValueError(f"Unsupported algorithm: {algorithm_name}")


def _run_pc_discrete(frame: pd.DataFrame) -> dict[str, Any]:
    from causallearn.search.ConstraintBased.PC import pc

    data = _encode_categorical_frame(frame)
    cg = pc(
        data.to_numpy(),
        alpha=0.01,
        indep_test="gsq",
        stable=True,
        uc_rule=0,
        show_progress=False,
        node_names=list(frame.columns),
    )
    return {
        "graph_type": "cpdag",
        "nodes": list(frame.columns),
        "adj_matrix": np.asarray(cg.G.graph, dtype=float).tolist(),
        "hyperparams": {"alpha": 0.01, "indep_test": "gsq", "stable": True},
    }


def _run_fci(frame: pd.DataFrame) -> dict[str, Any]:
    from causallearn.search.ConstraintBased.FCI import fci

    data = _encode_categorical_frame(frame)
    graph, edges = fci(
        data.to_numpy(),
        independence_test_method="gsq",
        alpha=0.05,
        depth=4,
        verbose=False,
        show_progress=False,
        node_names=list(frame.columns),
    )
    return {
        "graph_type": "pag",
        "nodes": list(frame.columns),
        "adj_matrix": np.asarray(graph.graph, dtype=float).tolist(),
        "hyperparams": {"alpha": 0.05, "indep_test": "gsq", "depth": 4, "edges": [str(edge) for edge in edges]},
    }


def _run_ges(frame: pd.DataFrame) -> dict[str, Any]:
    try:
        from pgmpy.estimators import BicScore, GES
    except ImportError:  # pragma: no cover
        from pgmpy.estimators import BIC as BicScore
        from pgmpy.estimators import GES

    data = _discretize_dataframe(frame)
    scoring = BicScore(data)
    search = GES(data, use_cache=True)
    model = search.estimate(scoring_method=scoring, min_improvement=0.0001, debug=False)
    return {
        "graph_type": "dag",
        "nodes": list(frame.columns),
        "adj_matrix": _pgmpy_dag_to_adj_matrix(model, list(frame.columns)),
        "hyperparams": {"scoring_method": "bic", "min_improvement": 0.0001, "use_cache": True},
    }


def _run_hc(frame: pd.DataFrame) -> dict[str, Any]:
    model, score = _run_hill_climb(frame, tabu_length=None, score_name="bic", max_indegree=3)
    return {
        "graph_type": "dag",
        "nodes": list(frame.columns),
        "adj_matrix": _pgmpy_dag_to_adj_matrix(model, list(frame.columns)),
        "hyperparams": {"scoring_method": "bic", "max_indegree": 3, "score": score},
    }


def _run_tabu(frame: pd.DataFrame) -> dict[str, Any]:
    model, score = _run_hill_climb(frame, tabu_length=100, score_name="bic", max_indegree=3)
    return {
        "graph_type": "dag",
        "nodes": list(frame.columns),
        "adj_matrix": _pgmpy_dag_to_adj_matrix(model, list(frame.columns)),
        "hyperparams": {"scoring_method": "bic", "tabu_length": 100, "max_indegree": 3, "score": score},
    }


def _run_k2(frame: pd.DataFrame) -> dict[str, Any]:
    try:
        from pgmpy.estimators import HillClimbSearch, K2Score
    except ImportError:  # pragma: no cover
        from pgmpy.estimators import HillClimbSearch, K2 as K2Score

    data = _discretize_dataframe(frame)
    scoring = K2Score(data)
    search = HillClimbSearch(data)
    model = search.estimate(scoring_method=scoring, max_indegree=3, show_progress=False)
    return {
        "graph_type": "dag",
        "nodes": list(frame.columns),
        "adj_matrix": _pgmpy_dag_to_adj_matrix(model, list(frame.columns)),
        "hyperparams": {"max_parents": 3, "score": float(scoring.score(model))},
    }


def _run_mmhc(frame: pd.DataFrame) -> dict[str, Any]:
    try:
        from pgmpy.estimators import BDeuScore, MmhcEstimator
    except ImportError:  # pragma: no cover
        from pgmpy.estimators import BDeu as BDeuScore
        from pgmpy.estimators import MmhcEstimator

    data = _discretize_dataframe(frame)
    scoring = BDeuScore(data)
    search = MmhcEstimator(data)
    model = search.estimate(scoring_method=scoring, tabu_length=10, significance_level=0.01)
    return {
        "graph_type": "dag",
        "nodes": list(frame.columns),
        "adj_matrix": _pgmpy_dag_to_adj_matrix(model, list(frame.columns)),
        "hyperparams": {"scoring_method": "bdeu", "tabu_length": 10, "significance_level": 0.01},
    }


def _run_boss(frame: pd.DataFrame) -> dict[str, Any]:
    from causallearn.search.PermutationBased.BOSS import boss

    data = _encode_categorical_frame(frame)
    graph = boss(
        data.to_numpy(),
        score_func="local_score_BDeu",
        parameters=_bdeu_parameters(data),
        verbose=False,
        node_names=list(frame.columns),
    )
    return {
        "graph_type": "cpdag",
        "nodes": list(frame.columns),
        "adj_matrix": np.asarray(graph.graph, dtype=float).tolist(),
        "hyperparams": {"score_func": "local_score_BDeu", "sample_prior": 1.0, "structure_prior": 1.0},
    }


def _run_grasp(frame: pd.DataFrame) -> dict[str, Any]:
    from causallearn.search.PermutationBased.GRaSP import grasp

    data = _encode_categorical_frame(frame)
    graph = grasp(
        data.to_numpy(),
        score_func="local_score_BDeu",
        depth=3,
        parameters=_bdeu_parameters(data),
        verbose=False,
        node_names=list(frame.columns),
    )
    return {
        "graph_type": "cpdag",
        "nodes": list(frame.columns),
        "adj_matrix": np.asarray(graph.graph, dtype=float).tolist(),
        "hyperparams": {"score_func": "local_score_BDeu", "sample_prior": 1.0, "structure_prior": 1.0, "depth": 3},
    }


def _run_hill_climb(frame: pd.DataFrame, *, tabu_length: int | None, score_name: str, max_indegree: int):
    try:
        from pgmpy.estimators import BicScore, HillClimbSearch
    except ImportError:  # pragma: no cover
        from pgmpy.estimators import BIC as BicScore
        from pgmpy.estimators import HillClimbSearch

    data = _discretize_dataframe(frame)
    scoring = BicScore(data)
    search = HillClimbSearch(data)
    kwargs = {
        "scoring_method": scoring,
        "max_iter": 100000,
        "max_indegree": max_indegree,
        "epsilon": 0.0001,
        "show_progress": False,
    }
    if tabu_length is not None:
        kwargs["tabu_length"] = tabu_length
    model = search.estimate(**kwargs)
    return model, float(scoring.score(model))


def _encode_categorical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        encoded[column] = pd.Categorical(frame[column]).codes
        encoded[column] = encoded[column].fillna(-1).astype(int)
    return encoded


def _discretize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = _encode_categorical_frame(frame)
    for column in encoded.columns:
        encoded[column] = encoded[column].astype("category")
    return encoded


def _bdeu_parameters(data: pd.DataFrame) -> dict[str, Any]:
    return {
        "sample_prior": 1.0,
        "structure_prior": 1.0,
        "r_i_map": {
            index: int(max(1, data.iloc[:, index].nunique(dropna=True)))
            for index in range(data.shape[1])
        },
    }


def _pgmpy_dag_to_adj_matrix(model, nodes: list[str]) -> list[list[float]]:
    adj = np.zeros((len(nodes), len(nodes)), dtype=float)
    node_to_idx = {node: index for index, node in enumerate(nodes)}
    for left, right in model.edges():
        adj[node_to_idx[left], node_to_idx[right]] = 1.0
    return adj.tolist()


def _graph_metrics(adj_matrix: list[list[float]]) -> dict[str, Any]:
    adj = np.asarray(adj_matrix, dtype=float)
    n = int(adj.shape[0])
    edge_count = 0
    possible_pairs = max(1, n * (n - 1) // 2)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] != 0 or adj[j][i] != 0:
                edge_count += 1
    return {
        "node_count": n,
        "edge_count": edge_count,
        "density": float(edge_count / possible_pairs),
    }


def _write_run_summary(output: Path) -> Path:
    records = []
    for record_path in sorted((output / "records").glob("*.json")):
        record = json.loads(record_path.read_text())
        metrics = record.get("metrics") or {}
        graph_result = record.get("graph_result") or {}
        records.append(
            {
                "dataset_name": record.get("dataset_name"),
                "algorithm_name": record.get("algorithm_name"),
                "status": record.get("status"),
                "runtime_seconds": record.get("runtime_seconds"),
                "timeout_seconds": record.get("timeout_seconds"),
                "n_samples": record.get("n_samples"),
                "n_features": record.get("n_features"),
                "graph_type": graph_result.get("graph_type"),
                "node_count": metrics.get("node_count"),
                "edge_count": metrics.get("edge_count"),
                "density": metrics.get("density"),
                "error": record.get("error"),
            }
        )
    summary_path = output / "summary.csv"
    pd.DataFrame(records).to_csv(summary_path, index=False)
    return summary_path


def _possible_edges(n_features: int) -> float:
    return max(1.0, float(n_features) * (float(n_features) - 1.0) / 2.0)


def _synthetic_split_map(entries: list[dict[str, Any]], *, seed: int) -> dict[str, str]:
    names = sorted(str(entry["dataset_name"]) for entry in entries)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(names).tolist()
    train_end = int(round(0.8 * len(shuffled)))
    val_end = train_end + int(round(0.1 * len(shuffled)))
    result: dict[str, str] = {}
    for idx, name in enumerate(shuffled):
        if idx < train_end:
            result[name] = "synthetic_train"
        elif idx < val_end:
            result[name] = "synthetic_val"
        else:
            result[name] = "synthetic_test"
    return result


def _synthetic_examples_from_manifest(
    synthetic_root: Path,
    *,
    max_rows: int | None,
    random_seed: int,
) -> list[SyntheticExample]:
    manifest = _load_synthetic_manifest(synthetic_root)
    examples: list[SyntheticExample] = []
    for index, entry in enumerate(manifest["datasets"]):
        dataset_path = synthetic_root / str(entry["dataset_path"])
        truth = _load_synthetic_truth(synthetic_root / str(entry["ground_truth_path"]))
        variable_features, pair_features = dataframe_to_learned_inputs(
            dataset_path,
            max_rows=max_rows,
            random_seed=random_seed + index,
        )
        node_to_idx = {node: idx for idx, node in enumerate(truth["nodes"])}
        adjacency = np.zeros((len(node_to_idx), len(node_to_idx)), dtype=np.float32)
        for src, dst in truth["directed_edges"]:
            adjacency[node_to_idx[src], node_to_idx[dst]] = 1.0
        examples.append(
            SyntheticExample(
                variable_features=variable_features,
                pair_features=pair_features,
                adjacency=adjacency,
                dataset_name=str(entry["dataset_name"]),
                graph_kind=str(entry.get("graph_family") or "synthetic"),
                n_samples=int(entry["n_samples"]),
            )
        )
    return examples


def _build_synthetic_learned_feature_table(
    config: AppConfig,
    *,
    tables: Path,
    encoder: Path,
) -> pd.DataFrame:
    existing = tables / "learned_features.csv"
    if existing.exists():
        return pd.read_csv(existing)
    synthetic_root = _infer_synthetic_root_from_tables(tables)
    model, payload, device = load_biaffine_encoder(encoder, device=config.learned.device)
    embedding_dim = int(payload["config"]["embedding_dim"])
    manifest = _load_synthetic_manifest(synthetic_root)
    columns = ["dataset_name", *learned_feature_columns(embedding_dim)]
    partial = tables / "learned_features.partial.csv"
    if partial.exists():
        partial_frame = pd.read_csv(partial)
        rows = partial_frame.to_dict("records")
        completed = {str(name) for name in partial_frame["dataset_name"].dropna().unique()}
    else:
        rows = []
        completed = set()
    entries = manifest["datasets"]
    total = len(entries)
    for entry in entries:
        dataset_name = str(entry["dataset_name"])
        if dataset_name in completed:
            continue
        dataset_path = synthetic_root / str(entry["dataset_path"])
        rows.append(
            {
                "dataset_name": dataset_name,
                **extract_fingerprint(
                    dataset_path,
                    model=model,
                    device=device,
                    embedding_dim=embedding_dim,
                    max_rows=config.learned.max_feature_rows,
                    random_seed=config.learned.random_seed,
                ),
            }
        )
        completed.add(dataset_name)
        if len(rows) % 50 == 0 or len(rows) == total:
            pd.DataFrame(rows, columns=columns).to_csv(partial, index=False)
            print(f"learned_features: {len(rows)}/{total}", flush=True)
    learned = pd.DataFrame(rows, columns=columns)
    learned.to_csv(existing, index=False)
    partial.unlink(missing_ok=True)
    return learned


def _infer_synthetic_root_from_tables(tables: Path) -> Path:
    # Default v1 layout: artifacts/synthetic_tables/v1 -> data/synthetic_bn/v1
    root = tables
    if root.name:
        candidate = Path.cwd() / "data" / "synthetic_bn" / root.name
        if (candidate / "manifest.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not infer synthetic root from tables path. "
        "Use the default matching v1 layout or build learned features separately."
    )


def _evaluate_selector_rows(
    selector,
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    split_name: str,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted(dataset_names):
        feature_row = feature_table[feature_table["dataset_name"] == dataset_name]
        if feature_row.empty:
            continue
        raw_prediction = selector.predict_from_features(feature_row.iloc[0].to_dict())
        prediction = _filter_prediction_to_available(raw_prediction, targets, dataset_name)
        metric = evaluate_prediction(
            dataset_name=dataset_name,
            prediction=prediction,
            targets=targets,
        )
        metric.update({"split_name": split_name, "method": method})
        rows.append(metric)
    return rows


def _aggregate_metrics_by_method(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    numeric = [
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
    present = [column for column in numeric if column in metrics.columns]
    summary = (
        metrics.groupby(["split_name", "method"])[present]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary = summary.rename(columns={"top3_overlap": "avg_top3_overlap"})
    return summary.sort_values(
        ["split_name", "avg_top3_overlap", "top3_overlap_at_least_2", "regret_at_3"],
        ascending=[True, False, False, True],
    )


def _choose_best_feature_set(summary: pd.DataFrame) -> str:
    val = summary[summary["split_name"] == "synthetic_val"].copy()
    if val.empty:
        return "handcrafted_all"
    val = val.sort_values(
        ["top3_hit", "regret_at_3", "top1_hit", "regret_at_1", "method"],
        ascending=[False, True, False, True, True],
        kind="mergesort",
    )
    return str(val.iloc[0]["method"])


def _choose_best_top3_feature_set(summary: pd.DataFrame) -> str:
    val = summary[summary["split_name"] == "synthetic_val"].copy()
    if val.empty:
        return "handcrafted_all"
    val = val.sort_values(
        ["avg_top3_overlap", "top3_overlap_at_least_2", "oracle_in_top3", "regret_at_3", "method"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    return str(val.iloc[0]["method"])


def _exact_feature_table_for_selector(
    config: AppConfig,
    base_features: pd.DataFrame,
    *,
    selector_feature_names: tuple[str, ...],
    encoder: Path | None,
) -> pd.DataFrame:
    if not any(name.startswith("lf_") for name in selector_feature_names):
        return base_features
    if encoder is None:
        raise ValueError("Selector expects learned lf_* features, but no encoder path was provided.")
    root = config.resolved_root()
    model, payload, device = load_biaffine_encoder(encoder, device=config.learned.device)
    embedding_dim = int(payload["config"]["embedding_dim"])
    manifest = load_import_manifest(root)
    rows = []
    for entry in manifest["datasets"]:
        dataset_name = str(entry["dataset_name"])
        dataset_path = root / str(entry["dataset_path"])
        rows.append(
            {
                "dataset_name": dataset_name,
                **extract_fingerprint(
                    dataset_path,
                    model=model,
                    device=device,
                    embedding_dim=embedding_dim,
                    max_rows=config.learned.max_feature_rows,
                    random_seed=config.learned.random_seed,
                ),
            }
        )
    learned = pd.DataFrame(rows)
    return base_features.merge(learned, on="dataset_name", how="inner", validate="one_to_one")


def _load_fixed_policy_order(
    root: Path,
    targets: pd.DataFrame,
    exact_names: list[str],
) -> tuple[str, ...]:
    policy_path = root / "artifacts" / "models" / "phase3_fixed_policy.json"
    if policy_path.exists():
        payload = json.loads(policy_path.read_text())
        ranking = dict(payload["ranking"])
        return tuple(sorted(ranking, key=lambda algorithm: int(ranking[algorithm])))
    train_targets = targets[targets["dataset_name"].isin(exact_names)].copy()
    return _rank_by_best_fixed_top3(train_targets, ALL_ALGORITHMS)


def _filter_order_to_available(
    order: tuple[str, ...],
    targets: pd.DataFrame,
    dataset_name: str,
) -> dict[str, Any]:
    available = set(targets.loc[targets["dataset_name"] == dataset_name, "algorithm_name"].astype(str))
    filtered = tuple(algorithm for algorithm in order if algorithm in available)
    missing = tuple(algorithm for algorithm in sorted(available) if algorithm not in filtered)
    ranking_order = (*filtered, *missing)
    return {
        "top_3": list(ranking_order[:3]),
        "ranking": {algorithm: rank + 1 for rank, algorithm in enumerate(ranking_order)},
        "pairwise_win_score": {
            algorithm: float(len(ranking_order) - rank - 1)
            for rank, algorithm in enumerate(ranking_order)
        },
        "predicted_relative_regret": {
            algorithm: float(rank)
            for rank, algorithm in enumerate(ranking_order)
        },
    }


def _exact_report_markdown(summary: pd.DataFrame, exact_names: list[str]) -> str:
    lines = ["# Synthetic-Trained Selector Exact Evaluation", ""]
    lines.append(f"Exact dataset count: {len(exact_names)}")
    lines.append("")
    if summary.empty:
        lines.append("No metrics available.")
        return "\n".join(lines) + "\n"
    lines.append("| split_name | method | top1_hit | top3_hit | regret_at_1 | regret_at_3 | oracle_ratio_at_3 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split_name"]),
                    str(row["method"]),
                    f"{float(row['top1_hit']):.4f}",
                    f"{float(row['top3_hit']):.4f}",
                    f"{float(row['regret_at_1']):.4f}",
                    f"{float(row['regret_at_3']):.4f}",
                    f"{float(row['oracle_ratio_at_3']):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)
