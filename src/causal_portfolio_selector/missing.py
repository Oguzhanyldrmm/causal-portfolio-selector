from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import random
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .artifacts import artifact_path, load_import_manifest
from .config import AppConfig


MISSING_ALGORITHMS: tuple[str, ...] = ("MMHC", "BOSS", "GRaSP")
DEFAULT_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class MissingRunOptions:
    algorithms: tuple[str, ...] = MISSING_ALGORITHMS
    dataset_names: tuple[str, ...] | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_rows: int | None = None
    output_dir: Path | None = None
    resume: bool = True
    overwrite: bool = False
    random_seed: int = 42


def run_missing_algorithm_suite(
    config: AppConfig,
    *,
    options: MissingRunOptions | None = None,
) -> dict[str, Path]:
    options = options or MissingRunOptions()
    root = config.resolved_root()
    manifest = load_import_manifest(root)
    output_dir = options.output_dir or artifact_path(root, "missing_algorithm_runs", "latest")
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    dataset_entries = _select_dataset_entries(
        manifest["datasets"],
        dataset_names=options.dataset_names,
    )
    algorithms = _validate_algorithms(options.algorithms)
    rows: list[dict[str, Any]] = []

    for entry in dataset_entries:
        dataset_name = str(entry["dataset_name"])
        dataset_path = root / str(entry["dataset_path"])
        for algorithm in algorithms:
            record_path = records_dir / f"{dataset_name}__{algorithm}.json"
            if (
                options.resume
                and not options.overwrite
                and record_path.exists()
            ):
                record = json.loads(record_path.read_text())
                rows.append(_summary_row(record))
                continue

            record = _run_one_with_timeout(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                algorithm_name=algorithm,
                timeout_seconds=int(options.timeout_seconds),
                max_rows=options.max_rows,
                random_seed=int(options.random_seed),
            )
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            rows.append(_summary_row(record))

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "summary.csv"
    manifest_path = output_dir / "manifest.json"
    summary.to_csv(summary_path, index=False)
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithms": list(algorithms),
        "dataset_count": len(dataset_entries),
        "record_count": len(rows),
        "timeout_seconds": int(options.timeout_seconds),
        "max_rows": options.max_rows,
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "records_dir": str(records_dir),
    }
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    return {
        "output_dir": output_dir,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _select_dataset_entries(
    entries: Iterable[dict[str, Any]],
    *,
    dataset_names: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    selected = list(entries)
    if dataset_names:
        requested = set(dataset_names)
        selected = [entry for entry in selected if str(entry["dataset_name"]) in requested]
        missing = sorted(requested - {str(entry["dataset_name"]) for entry in selected})
        if missing:
            raise ValueError(f"Unknown imported datasets requested: {missing}")
    return sorted(
        selected,
        key=lambda entry: (float(entry.get("n_features", 0.0)), str(entry["dataset_name"])),
    )


def _validate_algorithms(algorithms: tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted(set(algorithms) - set(MISSING_ALGORITHMS))
    if unknown:
        raise ValueError(
            f"This runner is restricted to missing algorithms {MISSING_ALGORITHMS}; "
            f"got unsupported algorithms: {unknown}"
        )
    return algorithms


def _run_one_with_timeout(
    *,
    dataset_name: str,
    dataset_path: Path,
    algorithm_name: str,
    timeout_seconds: int,
    max_rows: int | None,
    random_seed: int,
) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_worker_run_algorithm,
        kwargs={
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "algorithm_name": algorithm_name,
            "max_rows": max_rows,
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
            "max_rows": max_rows,
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
            "max_rows": max_rows,
            "graph_result": None,
            "error": "Worker exited without returning a result.",
        }
    record = queue.get()
    record["timeout_seconds"] = int(timeout_seconds)
    record["max_rows"] = max_rows
    return record


def _worker_run_algorithm(
    *,
    dataset_name: str,
    dataset_path: str,
    algorithm_name: str,
    max_rows: int | None,
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
                if max_rows is not None:
                    frame = frame.head(int(max_rows)).copy()
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
            except Exception as exc:  # pragma: no cover - exercised by optional deps/env
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
    if algorithm_name == "MMHC":
        return _run_mmhc(frame)
    if algorithm_name == "BOSS":
        return _run_boss(frame)
    if algorithm_name == "GRaSP":
        return _run_grasp(frame)
    raise ValueError(f"Unsupported missing algorithm: {algorithm_name}")


def _run_mmhc(frame: pd.DataFrame) -> dict[str, Any]:
    try:
        from pgmpy.estimators import BDeuScore, MmhcEstimator
    except ImportError:  # pragma: no cover - pgmpy version compatibility
        from pgmpy.estimators import BDeu as BDeuScore
        from pgmpy.estimators import MmhcEstimator

    data = _discretize_dataframe(frame)
    scoring = BDeuScore(data)
    search = MmhcEstimator(data)
    model = search.estimate(
        scoring_method=scoring,
        tabu_length=10,
        significance_level=0.01,
    )
    return {
        "graph_type": "dag",
        "nodes": list(frame.columns),
        "adj_matrix": _pgmpy_dag_to_adj_matrix(model, list(frame.columns)),
        "hyperparams": {
            "scoring_method": "bdeu",
            "tabu_length": 10,
            "significance_level": 0.01,
        },
    }


def _run_boss(frame: pd.DataFrame) -> dict[str, Any]:
    from causallearn.search.PermutationBased.BOSS import boss

    data = _encode_categorical_frame(frame)
    parameters = _bdeu_parameters(data)
    graph = boss(
        data.to_numpy(),
        score_func="local_score_BDeu",
        parameters=parameters,
        verbose=False,
        node_names=list(frame.columns),
    )
    return {
        "graph_type": "cpdag",
        "nodes": list(frame.columns),
        "adj_matrix": np.asarray(graph.graph, dtype=float).tolist(),
        "hyperparams": {
            "score_func": "local_score_BDeu",
            "sample_prior": 1.0,
            "structure_prior": 1.0,
        },
    }


def _run_grasp(frame: pd.DataFrame) -> dict[str, Any]:
    from causallearn.search.PermutationBased.GRaSP import grasp

    data = _encode_categorical_frame(frame)
    parameters = _bdeu_parameters(data)
    graph = grasp(
        data.to_numpy(),
        score_func="local_score_BDeu",
        depth=3,
        parameters=parameters,
        verbose=False,
        node_names=list(frame.columns),
    )
    return {
        "graph_type": "cpdag",
        "nodes": list(frame.columns),
        "adj_matrix": np.asarray(graph.graph, dtype=float).tolist(),
        "hyperparams": {
            "score_func": "local_score_BDeu",
            "sample_prior": 1.0,
            "structure_prior": 1.0,
            "depth": 3,
        },
    }


def _encode_categorical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_float_dtype(series):
            try:
                encoded[column] = pd.qcut(
                    series,
                    q=5,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                encoded[column] = pd.cut(series, bins=5, labels=False)
        else:
            encoded[column] = pd.Categorical(series).codes
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
    possible_pairs = max(1, n * (n - 1) // 2)
    edge_count = 0
    directed_pair_count = 0
    partially_oriented_pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] != 0 or adj[j][i] != 0:
                edge_count += 1
                if adj[i][j] > 0 or adj[j][i] > 0:
                    directed_pair_count += 1
                if adj[i][j] < 0 or adj[j][i] < 0:
                    partially_oriented_pair_count += 1
    return {
        "node_count": n,
        "edge_count": edge_count,
        "density": float(edge_count / possible_pairs),
        "directed_pair_count": directed_pair_count,
        "partially_oriented_pair_count": partially_oriented_pair_count,
    }


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    graph_result = record.get("graph_result") or {}
    return {
        "dataset_name": record.get("dataset_name"),
        "algorithm_name": record.get("algorithm_name"),
        "status": record.get("status"),
        "runtime_seconds": record.get("runtime_seconds"),
        "timeout_seconds": record.get("timeout_seconds"),
        "max_rows": record.get("max_rows"),
        "n_samples": record.get("n_samples"),
        "n_features": record.get("n_features"),
        "graph_type": graph_result.get("graph_type"),
        "node_count": metrics.get("node_count"),
        "edge_count": metrics.get("edge_count"),
        "density": metrics.get("density"),
        "error": record.get("error"),
    }

