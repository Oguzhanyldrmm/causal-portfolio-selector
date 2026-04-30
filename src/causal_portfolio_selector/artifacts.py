from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MAIN_ARTIFACTS: tuple[str, ...] = (
    "aggregated_calibration_table.csv",
    "benchmark_summary_table.csv",
    "output_manifest.json",
    "batch_config.json",
    "algorithm_catalog_snapshot.json",
)


@dataclass(frozen=True)
class ImportedPaths:
    project_root: Path
    imported_root: Path
    datasets_dir: Path
    ground_truth_dir: Path
    benchmark_dir: Path
    manifest_path: Path


def imported_paths(project_root: str | Path) -> ImportedPaths:
    root = Path(project_root).expanduser().resolve()
    imported_root = root / "data" / "imported"
    return ImportedPaths(
        project_root=root,
        imported_root=imported_root,
        datasets_dir=imported_root / "datasets",
        ground_truth_dir=imported_root / "ground_truth",
        benchmark_dir=imported_root / "benchmark_artifacts",
        manifest_path=imported_root / "import_manifest.json",
    )


def artifact_path(project_root: str | Path, *parts: str) -> Path:
    return Path(project_root).expanduser().resolve().joinpath("artifacts", *parts)


def report_path(project_root: str | Path, *parts: str) -> Path:
    return Path(project_root).expanduser().resolve().joinpath("reports", *parts)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required source file does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _infer_old_repo_root(source_run_dir: Path) -> Path:
    source_run_dir = source_run_dir.expanduser().resolve()
    for parent in source_run_dir.parents:
        if (parent / "data").is_dir() and (parent / "config" / "ground_truth").is_dir():
            return parent
    raise ValueError(
        "Could not infer old repository root from source run directory. "
        "Expected parent containing data/ and config/ground_truth/."
    )


def _selected_algorithms(summary_rows: Iterable[dict[str, str]]) -> set[str]:
    algorithms: set[str] = set()
    for row in summary_rows:
        raw = row.get("selected_algorithms") or "[]"
        for algorithm_name in json.loads(raw):
            algorithms.add(str(algorithm_name))
    return algorithms


def import_artifacts(
    *,
    source_run_dir: str | Path,
    project_root: str | Path,
    expected_algorithms: Iterable[str],
) -> dict[str, Any]:
    source = Path(source_run_dir).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source run directory does not exist: {source}")

    old_repo_root = _infer_old_repo_root(source)
    paths = imported_paths(project_root)
    paths.datasets_dir.mkdir(parents=True, exist_ok=True)
    paths.ground_truth_dir.mkdir(parents=True, exist_ok=True)
    paths.benchmark_dir.mkdir(parents=True, exist_ok=True)

    for filename in MAIN_ARTIFACTS:
        _copy_required(source / filename, paths.benchmark_dir / filename)

    summary_rows = _read_csv_rows(source / "benchmark_summary_table.csv")
    calibration_rows = _read_csv_rows(source / "aggregated_calibration_table.csv")
    datasets = sorted({row["dataset_name"] for row in summary_rows})
    algorithms = sorted(_selected_algorithms(summary_rows))
    expected = sorted(str(name) for name in expected_algorithms)
    if algorithms != expected:
        raise ValueError(f"Algorithm mismatch. Found {algorithms}, expected {expected}.")

    successful_rows = [row for row in calibration_rows if row.get("run_status") == "success"]
    expected_row_count = len(datasets) * len(expected)
    if len(successful_rows) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} successful calibration rows, "
            f"found {len(successful_rows)}."
        )

    dataset_entries: list[dict[str, Any]] = []
    for row in summary_rows:
        dataset_name = row["dataset_name"]
        dataset_src = old_repo_root / "data" / f"{dataset_name}.csv"
        truth_src = old_repo_root / "config" / "ground_truth" / f"{dataset_name}.json"
        dataset_dst = paths.datasets_dir / f"{dataset_name}.csv"
        truth_dst = paths.ground_truth_dir / f"{dataset_name}.json"
        _copy_required(dataset_src, dataset_dst)
        _copy_required(truth_src, truth_dst)

        run_dir_raw = row.get("output_dir") or ""
        source_dataset_run = old_repo_root / run_dir_raw
        dataset_artifact_dir = paths.benchmark_dir / "datasets" / dataset_name
        for filename in (
            "run_records.json",
            "comparison_table.csv",
            "metadata_features_table.csv",
            "graph_metrics_table.csv",
            "summary.txt",
        ):
            _copy_required(source_dataset_run / filename, dataset_artifact_dir / filename)

        dataset_entries.append(
            {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_dst.relative_to(paths.project_root)),
                "ground_truth_path": str(truth_dst.relative_to(paths.project_root)),
                "artifact_dir": str(dataset_artifact_dir.relative_to(paths.project_root)),
                "n_samples": float(row["n_samples"]),
                "n_features": float(row["n_features"]),
                "truth_type": _truth_type_for_dataset(calibration_rows, dataset_name),
            }
        )

    manifest = {
        "source_run_dir": str(source),
        "old_repo_root": str(old_repo_root),
        "datasets": dataset_entries,
        "dataset_count": len(datasets),
        "algorithms": expected,
        "algorithm_count": len(expected),
        "successful_row_count": len(successful_rows),
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _truth_type_for_dataset(calibration_rows: list[dict[str, str]], dataset_name: str) -> str:
    for row in calibration_rows:
        if row.get("dataset_name") == dataset_name:
            return row.get("truth_type") or ""
    return ""


def load_import_manifest(project_root: str | Path) -> dict[str, Any]:
    path = imported_paths(project_root).manifest_path
    if not path.exists():
        raise FileNotFoundError(
            f"Import manifest not found at {path}. Run import-artifacts first."
        )
    return json.loads(path.read_text())


def calibration_table_path(project_root: str | Path) -> Path:
    return imported_paths(project_root).benchmark_dir / "aggregated_calibration_table.csv"


def summary_table_path(project_root: str | Path) -> Path:
    return imported_paths(project_root).benchmark_dir / "benchmark_summary_table.csv"
