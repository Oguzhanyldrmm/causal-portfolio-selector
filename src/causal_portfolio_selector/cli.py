from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import AppConfig, load_config
from .pipeline import (
    prediction_to_json,
    run_build_tables,
    run_evaluate,
    run_evidence,
    run_build_learned_features,
    run_build_knn_prior_features,
    run_build_synthetic_training_tables,
    run_evaluate_synthetic_selector_on_exact,
    run_generate_synthetic_bn,
    run_import_artifacts,
    run_missing_algorithms,
    run_phase2_evidence_report,
    run_phase3_evidence_report,
    run_predict,
    run_synthetic_algorithms,
    run_train,
    run_train_fingerprint_from_synthetic,
    run_train_fingerprint,
    run_train_synthetic_score_selector,
    run_train_synthetic_selector,
    run_train_synthetic_top3_combination_selector,
    run_train_synthetic_top3_selector,
)


def _config_with_project_root(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    if args.project_root:
        return AppConfig(
            project_root=Path(args.project_root),
            source_run_dir=config.source_run_dir,
            algorithms=config.algorithms,
            external_datasets=config.external_datasets,
            features=config.features,
            model=config.model,
            learned=config.learned,
        )
    return config


def _with_learned_overrides(args: argparse.Namespace, config: AppConfig) -> AppConfig:
    learned = config.learned
    updates = {}
    for arg_name, field_name in (
        ("synthetic_graph_count", "synthetic_graph_count"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("device", "device"),
        ("synthetic_workers", "synthetic_workers"),
        ("max_feature_rows", "max_feature_rows"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            updates[field_name] = value
    if not updates:
        return config
    return replace(config, learned=replace(learned, **updates))


def _parse_csv_arg(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    parsed = tuple(item.strip() for item in str(value).split(",") if item.strip())
    return parsed or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-portfolio",
        description="Standalone causal discovery algorithm portfolio selector",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--project-root", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-artifacts")
    import_parser.add_argument("--source", default=None)

    subparsers.add_parser("build-tables")
    subparsers.add_parser("train")
    subparsers.add_parser("evaluate")

    evidence_parser = subparsers.add_parser("phase1-evidence")
    evidence_parser.add_argument("--random-repeats", type=int, default=200)

    train_fingerprint_parser = subparsers.add_parser("train-fingerprint")
    train_fingerprint_parser.add_argument("--synthetic-graph-count", type=int, default=None)
    train_fingerprint_parser.add_argument("--epochs", type=int, default=None)
    train_fingerprint_parser.add_argument("--batch-size", type=int, default=None)
    train_fingerprint_parser.add_argument("--device", default=None)
    train_fingerprint_parser.add_argument("--synthetic-workers", type=int, default=None)
    train_fingerprint_parser.add_argument("--max-feature-rows", type=int, default=None)

    build_learned_parser = subparsers.add_parser("build-learned-features")
    build_learned_parser.add_argument("--model", default=None)

    subparsers.add_parser("phase2-evidence")
    subparsers.add_parser(
        "phase3-evidence",
        help="Evaluate the timeout-aware 9-algorithm selector using successful missing runs.",
    )

    missing_parser = subparsers.add_parser(
        "run-missing-algorithms",
        help="Run only MMHC, BOSS, and GRaSP on imported datasets with hard timeouts.",
    )
    missing_parser.add_argument("--datasets", default=None)
    missing_parser.add_argument("--algorithms", default="MMHC,BOSS,GRaSP")
    missing_parser.add_argument("--timeout-seconds", type=int, default=120)
    missing_parser.add_argument("--max-rows", type=int, default=None)
    missing_parser.add_argument("--output-dir", default=None)
    missing_parser.add_argument("--no-resume", action="store_true")
    missing_parser.add_argument("--overwrite", action="store_true")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--dataset", required=True)
    predict_parser.add_argument("--model", default=None)

    synthetic_gen_parser = subparsers.add_parser(
        "generate-synthetic-bn",
        help="Generate heterogeneous synthetic discrete Bayesian-network datasets.",
    )
    synthetic_gen_parser.add_argument("--output", required=True)
    synthetic_gen_parser.add_argument("--count", type=int, default=2000)
    synthetic_gen_parser.add_argument("--max-nodes", type=int, default=40)
    synthetic_gen_parser.add_argument("--seed", type=int, default=42)
    synthetic_gen_parser.add_argument("--overwrite", action="store_true")

    synthetic_run_parser = subparsers.add_parser(
        "run-synthetic-algorithms",
        help="Run the nine-algorithm portfolio on synthetic BN datasets.",
    )
    synthetic_run_parser.add_argument("--synthetic-root", required=True)
    synthetic_run_parser.add_argument("--output", required=True)
    synthetic_run_parser.add_argument(
        "--algorithms",
        default="PC_discrete,FCI,GES,HC,Tabu,K2,MMHC,BOSS,GRaSP",
    )
    synthetic_run_parser.add_argument("--timeout-seconds", type=int, default=300)
    synthetic_run_parser.add_argument("--shard-index", type=int, default=0)
    synthetic_run_parser.add_argument("--shard-count", type=int, default=1)
    synthetic_run_parser.add_argument("--resume", action="store_true", default=True)
    synthetic_run_parser.add_argument("--no-resume", action="store_true")
    synthetic_run_parser.add_argument("--overwrite", action="store_true")

    synthetic_tables_parser = subparsers.add_parser(
        "build-synthetic-training-tables",
        help="Build feature/target tables from synthetic algorithm runs.",
    )
    synthetic_tables_parser.add_argument("--synthetic-root", required=True)
    synthetic_tables_parser.add_argument("--runs", required=True)
    synthetic_tables_parser.add_argument("--output", required=True)

    knn_prior_parser = subparsers.add_parser(
        "build-knn-prior-features",
        help="Build kNN prior features from synthetic training tables.",
    )
    knn_prior_parser.add_argument("--tables", required=True)
    knn_prior_parser.add_argument("--output", required=True)
    knn_prior_parser.add_argument("--exact-tables", default=None)
    knn_prior_parser.add_argument("--metadata-output", default=None)
    knn_prior_parser.add_argument("--k", type=int, default=50)

    synthetic_encoder_parser = subparsers.add_parser(
        "train-fingerprint-from-synthetic",
        help="Train the biaffine fingerprint encoder from generated synthetic BNs.",
    )
    synthetic_encoder_parser.add_argument("--synthetic-root", required=True)
    synthetic_encoder_parser.add_argument("--output", required=True)
    synthetic_encoder_parser.add_argument("--device", default=None)
    synthetic_encoder_parser.add_argument("--epochs", type=int, default=None)

    synthetic_selector_parser = subparsers.add_parser(
        "train-synthetic-selector",
        help="Train a selector using synthetic training tables.",
    )
    synthetic_selector_parser.add_argument("--tables", required=True)
    synthetic_selector_parser.add_argument("--encoder", default=None)
    synthetic_selector_parser.add_argument("--output", required=True)

    synthetic_top3_selector_parser = subparsers.add_parser(
        "train-synthetic-top3-selector",
        help="Train a top-3 membership selector using synthetic training tables.",
    )
    synthetic_top3_selector_parser.add_argument("--tables", required=True)
    synthetic_top3_selector_parser.add_argument("--encoder", default=None)
    synthetic_top3_selector_parser.add_argument("--output", required=True)

    synthetic_score_selector_parser = subparsers.add_parser(
        "train-synthetic-score-selector",
        help="Train a score-regression selector using synthetic training tables.",
    )
    synthetic_score_selector_parser.add_argument("--tables", required=True)
    synthetic_score_selector_parser.add_argument("--encoder", default=None)
    synthetic_score_selector_parser.add_argument("--output", required=True)

    synthetic_top3_combo_parser = subparsers.add_parser(
        "train-synthetic-top3-combination-selector",
        help="Train a top-3 combination reward selector using synthetic training tables.",
    )
    synthetic_top3_combo_parser.add_argument("--tables", required=True)
    synthetic_top3_combo_parser.add_argument("--encoder", default=None)
    synthetic_top3_combo_parser.add_argument("--output", required=True)
    synthetic_top3_combo_parser.add_argument("--oracle-weight", type=float, default=3.0)
    synthetic_top3_combo_parser.add_argument("--overlap-weight", type=float, default=1.0)
    synthetic_top3_combo_parser.add_argument("--overlap-at-least-2-weight", type=float, default=0.0)
    synthetic_top3_combo_parser.add_argument("--regret-weight", type=float, default=0.25)

    synthetic_exact_parser = subparsers.add_parser(
        "evaluate-synthetic-selector-on-exact",
        help="Evaluate a synthetic-trained selector on exact bnlearn datasets.",
    )
    synthetic_exact_parser.add_argument("--model", required=True)
    synthetic_exact_parser.add_argument("--encoder", default=None)
    synthetic_exact_parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _config_with_project_root(args)

    if args.command == "import-artifacts":
        manifest = run_import_artifacts(config, source=args.source)
        print(
            "Imported "
            f"{manifest['dataset_count']} datasets, "
            f"{manifest['algorithm_count']} algorithms, "
            f"{manifest['successful_row_count']} successful rows."
        )
        return 0

    if args.command == "build-tables":
        features, targets = run_build_tables(config)
        print(f"Built feature table {features.shape} and target table {targets.shape}.")
        return 0

    if args.command == "train":
        model_path = run_train(config)
        print(f"Saved selector model to {model_path}.")
        return 0

    if args.command == "evaluate":
        outputs = run_evaluate(config)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "phase1-evidence":
        outputs = run_evidence(config, random_repeats=args.random_repeats)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "train-fingerprint":
        config = _with_learned_overrides(args, config)
        result = run_train_fingerprint(config)
        print(f"model_path: {result['model_path']}")
        print(f"manifest_path: {result['manifest_path']}")
        print(f"device: {result['device']}")
        print(f"final_metrics: {result['final_metrics']}")
        return 0

    if args.command == "build-learned-features":
        outputs = run_build_learned_features(config, model_path=args.model)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "phase2-evidence":
        outputs = run_phase2_evidence_report(config)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "phase3-evidence":
        outputs = run_phase3_evidence_report(config)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "run-missing-algorithms":
        outputs = run_missing_algorithms(
            config,
            algorithms=_parse_csv_arg(args.algorithms),
            dataset_names=_parse_csv_arg(args.datasets),
            timeout_seconds=args.timeout_seconds,
            max_rows=args.max_rows,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            overwrite=args.overwrite,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "predict":
        prediction = run_predict(config, dataset_path=args.dataset, model_path=args.model)
        print(prediction_to_json(prediction), end="")
        return 0

    if args.command == "generate-synthetic-bn":
        outputs = run_generate_synthetic_bn(
            output=args.output,
            count=args.count,
            max_nodes=args.max_nodes,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "run-synthetic-algorithms":
        outputs = run_synthetic_algorithms(
            config,
            synthetic_root=args.synthetic_root,
            output=args.output,
            algorithms=_parse_csv_arg(args.algorithms),
            timeout_seconds=args.timeout_seconds,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            resume=not args.no_resume,
            overwrite=args.overwrite,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "build-synthetic-training-tables":
        outputs = run_build_synthetic_training_tables(
            config,
            synthetic_root=args.synthetic_root,
            runs=args.runs,
            output=args.output,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "build-knn-prior-features":
        outputs = run_build_knn_prior_features(
            tables=args.tables,
            output=args.output,
            exact_tables=args.exact_tables,
            metadata_output=args.metadata_output,
            k=args.k,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "train-fingerprint-from-synthetic":
        outputs = run_train_fingerprint_from_synthetic(
            config,
            synthetic_root=args.synthetic_root,
            output=args.output,
            device=args.device,
            epochs=args.epochs,
        )
        print(f"model_path: {outputs['model_path']}")
        print(f"manifest_path: {outputs['manifest_path']}")
        print(f"device: {outputs['device']}")
        print(f"final_metrics: {outputs['final_metrics']}")
        return 0

    if args.command == "train-synthetic-selector":
        outputs = run_train_synthetic_selector(
            config,
            tables=args.tables,
            encoder=args.encoder,
            output=args.output,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "train-synthetic-top3-selector":
        outputs = run_train_synthetic_top3_selector(
            config,
            tables=args.tables,
            encoder=args.encoder,
            output=args.output,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "train-synthetic-score-selector":
        outputs = run_train_synthetic_score_selector(
            config,
            tables=args.tables,
            encoder=args.encoder,
            output=args.output,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "train-synthetic-top3-combination-selector":
        outputs = run_train_synthetic_top3_combination_selector(
            config,
            tables=args.tables,
            encoder=args.encoder,
            output=args.output,
            oracle_weight=args.oracle_weight,
            overlap_weight=args.overlap_weight,
            overlap_at_least_2_weight=args.overlap_at_least_2_weight,
            regret_weight=args.regret_weight,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "evaluate-synthetic-selector-on-exact":
        outputs = run_evaluate_synthetic_selector_on_exact(
            config,
            model=args.model,
            encoder=args.encoder,
            output=args.output,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
