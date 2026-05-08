from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import artifact_path, import_artifacts, report_path
from .config import AppConfig
from .evaluation import (
    aggregate_metrics,
    evaluate_selector_on_datasets,
    lodo_evaluate,
    summary_markdown,
)
from .experiments import run_phase1_evidence, run_phase2_evidence
from .learned.fingerprint import build_learned_feature_tables
from .learned.model import train_biaffine_encoder
from .learned.synthetic import generate_synthetic_examples
from .features import extract_dataset_features
from .knn_prior import KnnPriorOptions, build_knn_prior_tables
from .missing import MissingRunOptions, run_missing_algorithm_suite
from .models import load_selector, save_selector, train_selector
from .phase3 import run_phase3_evidence
from .synthetic_benchmark import (
    SyntheticGenerateOptions,
    SyntheticRunOptions,
    build_synthetic_training_tables,
    evaluate_synthetic_selector_on_exact,
    generate_synthetic_bn_suite,
    run_synthetic_algorithm_suite,
    train_fingerprint_from_synthetic,
    train_synthetic_score_selector,
    train_synthetic_selector,
    train_synthetic_top3_combination_selector,
    train_synthetic_top3_selector,
)
from .targets import (
    build_tables,
    exact_dataset_names,
    external_dataset_names,
    legacy_eval_dataset_names,
    legacy_train_dataset_names,
    load_or_build_tables,
)


def run_import_artifacts(config: AppConfig, *, source: str | Path | None = None) -> dict[str, Any]:
    root = config.resolved_root()
    manifest = import_artifacts(
        source_run_dir=source or config.source_run_dir,
        project_root=root,
        expected_algorithms=config.algorithms,
    )
    return manifest


def run_train(config: AppConfig) -> Path:
    root = config.resolved_root()
    feature_table, targets = load_or_build_tables(config)
    train_dataset_names = exact_dataset_names(
        targets,
        external_datasets=set(config.external_datasets),
    )
    selector = train_selector(
        feature_table,
        targets,
        dataset_names=train_dataset_names,
        algorithms=config.algorithms,
        config=config.model,
    )
    model_path = artifact_path(root, "models", "selector.joblib")
    save_selector(selector, model_path)
    return model_path


def run_build_tables(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_tables(config)


def run_evaluate(config: AppConfig) -> dict[str, Path]:
    root = config.resolved_root()
    feature_table, targets = load_or_build_tables(config)
    reports_dir = report_path(root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    exact_names = exact_dataset_names(targets, external_datasets=set(config.external_datasets))
    lodo_metrics, lodo_predictions = lodo_evaluate(
        feature_table,
        targets,
        dataset_names=exact_names,
        algorithms=config.algorithms,
        model_config=config.model,
    )
    lodo_metrics_path = reports_dir / "lodo_metrics.csv"
    lodo_predictions_path = reports_dir / "lodo_predictions.csv"
    lodo_metrics.to_csv(lodo_metrics_path, index=False)
    lodo_predictions.to_csv(lodo_predictions_path, index=False)

    legacy_frames: list[pd.DataFrame] = []
    legacy_prediction_frames: list[pd.DataFrame] = []
    legacy_train = legacy_train_dataset_names(targets)
    if legacy_train:
        legacy_selector = train_selector(
            feature_table,
            targets,
            dataset_names=legacy_train,
            algorithms=config.algorithms,
            config=config.model,
        )
        for split_name, dataset_names in (
            ("legacy_eval", legacy_eval_dataset_names(targets)),
            ("external", external_dataset_names(targets)),
        ):
            if not dataset_names:
                continue
            metrics, predictions = evaluate_selector_on_datasets(
                legacy_selector,
                feature_table,
                targets,
                dataset_names=dataset_names,
                split_name=split_name,
            )
            legacy_frames.append(metrics)
            legacy_prediction_frames.append(predictions)

    legacy_metrics = pd.concat(legacy_frames, ignore_index=True) if legacy_frames else pd.DataFrame()
    legacy_predictions = (
        pd.concat(legacy_prediction_frames, ignore_index=True)
        if legacy_prediction_frames
        else pd.DataFrame()
    )
    legacy_metrics_path = reports_dir / "legacy_split_metrics.csv"
    legacy_predictions_path = reports_dir / "legacy_split_predictions.csv"
    legacy_metrics.to_csv(legacy_metrics_path, index=False)
    legacy_predictions.to_csv(legacy_predictions_path, index=False)

    all_metrics = pd.concat(
        [frame for frame in (lodo_metrics, legacy_metrics) if not frame.empty],
        ignore_index=True,
    )
    all_predictions = pd.concat(
        [frame for frame in (lodo_predictions, legacy_predictions) if not frame.empty],
        ignore_index=True,
    )
    aggregate = aggregate_metrics(all_metrics)
    aggregate_path = reports_dir / "aggregate_metrics.csv"
    predictions_path = reports_dir / "predictions_by_dataset.csv"
    summary_path = reports_dir / "summary.md"
    aggregate.to_csv(aggregate_path, index=False)
    all_predictions.to_csv(predictions_path, index=False)
    summary_path.write_text(summary_markdown(aggregate))

    return {
        "lodo_metrics": lodo_metrics_path,
        "legacy_split_metrics": legacy_metrics_path,
        "aggregate_metrics": aggregate_path,
        "predictions_by_dataset": predictions_path,
        "summary": summary_path,
    }


def run_predict(
    config: AppConfig,
    *,
    dataset_path: str | Path,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    root = config.resolved_root()
    selector = load_selector(model_path or artifact_path(root, "models", "selector.joblib"))
    features = extract_dataset_features(dataset_path, config=config.features)
    prediction = selector.predict_from_features(features)
    prediction["feature_warnings"] = []
    return prediction


def prediction_to_json(prediction: dict[str, Any]) -> str:
    return json.dumps(prediction, indent=2, sort_keys=True) + "\n"


def run_evidence(config: AppConfig, *, random_repeats: int = 200) -> dict[str, Path]:
    return run_phase1_evidence(config, random_repeats=random_repeats)


def run_train_fingerprint(config: AppConfig) -> dict[str, Any]:
    root = config.resolved_root()
    examples = generate_synthetic_examples(config.learned)
    output_path = artifact_path(root, "learned", "biaffine_encoder.pt")
    result = train_biaffine_encoder(examples, config.learned, output_path=output_path)
    manifest_path = artifact_path(root, "learned", "synthetic_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "synthetic_graph_count": len(examples),
        "model_path": result["model_path"],
        "device": result["device"],
        "final_metrics": result["final_metrics"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**result, "manifest_path": str(manifest_path)}


def run_build_learned_features(config: AppConfig, *, model_path: str | Path | None = None) -> dict[str, Path]:
    return build_learned_feature_tables(config, model_path=model_path)


def run_phase2_evidence_report(config: AppConfig) -> dict[str, Path]:
    return run_phase2_evidence(config)


def run_phase3_evidence_report(config: AppConfig) -> dict[str, Path]:
    return run_phase3_evidence(config)


def run_missing_algorithms(
    config: AppConfig,
    *,
    algorithms: tuple[str, ...] | None = None,
    dataset_names: tuple[str, ...] | None = None,
    timeout_seconds: int = 120,
    max_rows: int | None = None,
    output_dir: str | Path | None = None,
    resume: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    return run_missing_algorithm_suite(
        config,
        options=MissingRunOptions(
            algorithms=algorithms or ("MMHC", "BOSS", "GRaSP"),
            dataset_names=dataset_names,
            timeout_seconds=int(timeout_seconds),
            max_rows=max_rows,
            output_dir=Path(output_dir) if output_dir is not None else None,
            resume=resume,
            overwrite=overwrite,
            random_seed=config.model.random_state,
        ),
    )


def run_generate_synthetic_bn(
    *,
    output: str | Path,
    count: int = 2000,
    max_nodes: int = 40,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Path]:
    return generate_synthetic_bn_suite(
        SyntheticGenerateOptions(
            output=Path(output),
            count=int(count),
            max_nodes=int(max_nodes),
            seed=int(seed),
            overwrite=overwrite,
        )
    )


def run_synthetic_algorithms(
    config: AppConfig,
    *,
    synthetic_root: str | Path,
    output: str | Path,
    algorithms: tuple[str, ...] | None = None,
    timeout_seconds: int = 300,
    shard_index: int = 0,
    shard_count: int = 1,
    resume: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    return run_synthetic_algorithm_suite(
        SyntheticRunOptions(
            synthetic_root=Path(synthetic_root),
            output=Path(output),
            algorithms=algorithms or (
                "PC_discrete",
                "FCI",
                "GES",
                "HC",
                "Tabu",
                "K2",
                "MMHC",
                "BOSS",
                "GRaSP",
            ),
            timeout_seconds=int(timeout_seconds),
            shard_index=int(shard_index),
            shard_count=int(shard_count),
            resume=resume,
            overwrite=overwrite,
            random_seed=config.model.random_state,
        )
    )


def run_build_synthetic_training_tables(
    config: AppConfig,
    *,
    synthetic_root: str | Path,
    runs: str | Path,
    output: str | Path,
) -> dict[str, Path]:
    return build_synthetic_training_tables(
        config,
        synthetic_root=synthetic_root,
        runs=runs,
        output=output,
    )


def run_build_knn_prior_features(
    *,
    tables: str | Path,
    output: str | Path,
    exact_tables: str | Path | None = None,
    metadata_output: str | Path | None = None,
    k: int = 50,
) -> dict[str, Path]:
    return build_knn_prior_tables(
        KnnPriorOptions(
            tables=Path(tables),
            output=Path(output),
            exact_tables=Path(exact_tables) if exact_tables is not None else None,
            metadata_output=Path(metadata_output) if metadata_output is not None else None,
            k=int(k),
        )
    )


def run_train_fingerprint_from_synthetic(
    config: AppConfig,
    *,
    synthetic_root: str | Path,
    output: str | Path,
    device: str | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    return train_fingerprint_from_synthetic(
        config,
        synthetic_root=synthetic_root,
        output=output,
        device=device,
        epochs=epochs,
    )


def run_train_synthetic_selector(
    config: AppConfig,
    *,
    tables: str | Path,
    encoder: str | Path | None,
    output: str | Path,
) -> dict[str, Path]:
    return train_synthetic_selector(
        config,
        tables=tables,
        encoder=encoder,
        output=output,
    )


def run_train_synthetic_top3_selector(
    config: AppConfig,
    *,
    tables: str | Path,
    encoder: str | Path | None,
    output: str | Path,
) -> dict[str, Path]:
    return train_synthetic_top3_selector(
        config,
        tables=tables,
        encoder=encoder,
        output=output,
    )


def run_train_synthetic_score_selector(
    config: AppConfig,
    *,
    tables: str | Path,
    encoder: str | Path | None,
    output: str | Path,
) -> dict[str, Path]:
    return train_synthetic_score_selector(
        config,
        tables=tables,
        encoder=encoder,
        output=output,
    )


def run_train_synthetic_top3_combination_selector(
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
    return train_synthetic_top3_combination_selector(
        config,
        tables=tables,
        encoder=encoder,
        output=output,
        oracle_weight=float(oracle_weight),
        overlap_weight=float(overlap_weight),
        overlap_at_least_2_weight=float(overlap_at_least_2_weight),
        regret_weight=float(regret_weight),
    )


def run_evaluate_synthetic_selector_on_exact(
    config: AppConfig,
    *,
    model: str | Path,
    encoder: str | Path | None,
    output: str | Path,
) -> dict[str, Path]:
    return evaluate_synthetic_selector_on_exact(
        config,
        model=model,
        encoder=encoder,
        output=output,
    )
