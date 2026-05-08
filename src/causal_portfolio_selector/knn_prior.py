from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .features import FEATURE_COLUMNS


@dataclass(frozen=True)
class KnnPriorOptions:
    tables: Path
    output: Path
    k: int = 50
    exact_tables: Path | None = None
    metadata_output: Path | None = None


def build_knn_prior_tables(options: KnnPriorOptions) -> dict[str, Path]:
    tables = options.tables.expanduser().resolve()
    output = options.output.expanduser().resolve()
    exact_tables = options.exact_tables.expanduser().resolve() if options.exact_tables else None
    metadata_output = (
        options.metadata_output.expanduser().resolve()
        if options.metadata_output is not None
        else output / "knn_prior_manifest.json"
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(tables / "features.csv")
    targets = pd.read_csv(tables / "targets.csv")
    algorithms = tuple(sorted(targets["algorithm_name"].dropna().astype(str).unique()))
    train_names = sorted(targets.loc[targets["split_role"] == "synthetic_train", "dataset_name"].unique())
    if not train_names:
        raise ValueError("No synthetic_train rows found in targets.csv.")
    if len(train_names) < 2:
        raise ValueError("At least two synthetic_train datasets are required for kNN priors.")
    missing_features = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing_features:
        raise ValueError(f"features.csv is missing handcrafted columns: {missing_features}")

    synthetic_knn = _knn_features_for_queries(
        query_features=features,
        reference_features=features,
        targets=targets,
        train_names=train_names,
        algorithms=algorithms,
        k=options.k,
        drop_self_for_train=True,
    )
    augmented_features = features.merge(synthetic_knn, on="dataset_name", how="inner", validate="one_to_one")
    augmented_features.to_csv(output / "features.csv", index=False)
    synthetic_knn.to_csv(output / "knn_features.csv", index=False)

    for filename in (
        "targets.csv",
        "run_evaluations.csv",
        "splits.csv",
        "selected_train_datasets.csv",
        "selection_manifest.json",
        "learned_features.csv",
    ):
        source = tables / filename
        if source.exists():
            shutil.copy2(source, output / filename)

    paths: dict[str, Path] = {
        "features": output / "features.csv",
        "knn_features": output / "knn_features.csv",
        "manifest": metadata_output,
    }
    exact_dataset_count = 0
    if exact_tables is not None:
        exact_features = pd.read_csv(exact_tables / "features.csv")
        exact_knn = _knn_features_for_queries(
            query_features=exact_features,
            reference_features=features,
            targets=targets,
            train_names=train_names,
            algorithms=algorithms,
            k=options.k,
            drop_self_for_train=False,
        )
        augmented_exact_features = exact_features.merge(
            exact_knn,
            on="dataset_name",
            how="inner",
            validate="one_to_one",
        )
        augmented_exact_features.to_csv(output / "exact_features.csv", index=False)
        exact_knn.to_csv(output / "exact_knn_features.csv", index=False)
        paths["exact_features"] = output / "exact_features.csv"
        paths["exact_knn_features"] = output / "exact_knn_features.csv"
        exact_dataset_count = int(augmented_exact_features.shape[0])

    manifest = {
        "source_tables": str(tables),
        "output_tables": str(output),
        "exact_tables": str(exact_tables) if exact_tables is not None else None,
        "k_requested": int(options.k),
        "train_dataset_count": len(train_names),
        "synthetic_dataset_count": int(features.shape[0]),
        "exact_dataset_count": exact_dataset_count,
        "algorithms": list(algorithms),
        "distance_feature_columns": list(FEATURE_COLUMNS),
        "knn_feature_columns": [column for column in synthetic_knn.columns if column != "dataset_name"],
    }
    metadata_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if metadata_output != output / "knn_prior_manifest.json":
        (output / "knn_prior_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return paths


def _knn_features_for_queries(
    *,
    query_features: pd.DataFrame,
    reference_features: pd.DataFrame,
    targets: pd.DataFrame,
    train_names: list[str],
    algorithms: tuple[str, ...],
    k: int,
    drop_self_for_train: bool,
) -> pd.DataFrame:
    k = max(1, int(k))
    reference_indexed = reference_features.set_index("dataset_name", drop=False)
    missing_train = [name for name in train_names if name not in reference_indexed.index]
    if missing_train:
        raise ValueError(f"Training datasets missing from features.csv: {missing_train[:5]}")

    train_feature_frame = reference_indexed.loc[train_names, list(FEATURE_COLUMNS)]
    train_matrix, query_matrix = _standardized_matrices(train_feature_frame, query_features)
    neighbor_count = min(k + 1 if drop_self_for_train else k, len(train_names))
    nearest = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    nearest.fit(train_matrix)
    distances, indices = nearest.kneighbors(query_matrix)

    top1 = _top1_by_dataset(targets)
    top3 = _top3_by_dataset(targets)
    rows: list[dict[str, Any]] = []
    for row_index, dataset_name in enumerate(query_features["dataset_name"].astype(str)):
        neighbor_names: list[str] = []
        neighbor_distances: list[float] = []
        for distance, neighbor_idx in zip(distances[row_index], indices[row_index]):
            neighbor_name = train_names[int(neighbor_idx)]
            if drop_self_for_train and neighbor_name == dataset_name:
                continue
            neighbor_names.append(neighbor_name)
            neighbor_distances.append(float(distance))
            if len(neighbor_names) >= k:
                break

        if not neighbor_names:
            raise ValueError(f"No neighbors found for dataset {dataset_name}.")
        distance_array = np.asarray(neighbor_distances, dtype=float)
        output_row: dict[str, Any] = {
            "dataset_name": dataset_name,
            "knn_neighbor_count": len(neighbor_names),
            "knn_min_distance": float(distance_array.min()),
            "knn_mean_distance": float(distance_array.mean()),
            "knn_std_distance": float(distance_array.std(ddof=0)),
        }
        for algorithm in algorithms:
            output_row[f"knn_top1_rate_{algorithm}"] = float(
                np.mean([top1.get(name) == algorithm for name in neighbor_names])
            )
            output_row[f"knn_top3_rate_{algorithm}"] = float(
                np.mean([algorithm in top3.get(name, set()) for name in neighbor_names])
            )
        rows.append(output_row)
    return pd.DataFrame(rows)


def _standardized_matrices(
    train_feature_frame: pd.DataFrame,
    query_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_numeric = train_feature_frame.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    query_numeric = query_features.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    means = train_numeric.mean(axis=0)
    stds = train_numeric.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    train_matrix = ((train_numeric.fillna(means) - means) / stds).fillna(0.0).to_numpy(dtype=float)
    query_matrix = ((query_numeric.fillna(means) - means) / stds).fillna(0.0).to_numpy(dtype=float)
    return train_matrix, query_matrix


def _top1_by_dataset(targets: pd.DataFrame) -> dict[str, str]:
    ordered = targets.sort_values(["dataset_name", "quality_rank", "algorithm_name"])
    top_rows = ordered.groupby("dataset_name", as_index=False).first()
    return dict(zip(top_rows["dataset_name"].astype(str), top_rows["algorithm_name"].astype(str)))


def _top3_by_dataset(targets: pd.DataFrame) -> dict[str, set[str]]:
    top3 = targets[targets["quality_rank"] <= 3]
    return {
        str(dataset_name): set(group["algorithm_name"].astype(str))
        for dataset_name, group in top3.groupby("dataset_name")
    }

