# Causal Portfolio Selector

Standalone SATzilla-style algorithm portfolio selector for discrete causal discovery.

This repository is intentionally independent from the original
`Causal-Algorithm-Selection` pipeline. The old pipeline is used only as a source
of already-computed CSV/JSON artifacts; no Python modules are imported from it.

## V1 Scope

- Uses existing benchmark outputs for 16 datasets and 6 algorithms.
- Does not rerun the already-completed six causal discovery algorithms.
- Provides a controlled runner for the three missing algorithms:
  `MMHC`, `BOSS`, and `GRaSP`.
- Trains a pre-run selector from dataset meta-features to a top-3 algorithm
  recommendation.
- Reports LODO and legacy split metrics.

## Quick Start

```bash
causal-portfolio import-artifacts \
  --source /home/oguzhan/Causal-Algorithm-Selection/runs/benchmark_runs_min_oct_no_pathfinder/20260427T215118Z

causal-portfolio --config configs/default.yaml train
causal-portfolio --config configs/default.yaml evaluate
causal-portfolio --config configs/default.yaml phase1-evidence

# Phase 2 learned structural fingerprints.
uv run --with torch causal-portfolio --config configs/default.yaml train-fingerprint \
  --synthetic-graph-count 1000 \
  --epochs 50
uv run --with torch causal-portfolio --config configs/default.yaml build-learned-features
causal-portfolio --config configs/default.yaml phase2-evidence

# Run only the three missing algorithms on imported datasets.
# These classical implementations are CPU-bound; use hard timeouts.
uv run --extra benchmark causal-portfolio --config configs/default.yaml run-missing-algorithms \
  --algorithms MMHC,BOSS,GRaSP \
  --timeout-seconds 120

# Phase 3 timeout-aware nine-algorithm evaluation.
causal-portfolio --config configs/default.yaml phase3-evidence

# Synthetic BN benchmark v1. Dataset generation is fast; algorithm runs are CPU-heavy.
uv run causal-portfolio --config configs/default.yaml generate-synthetic-bn \
  --output data/synthetic_bn/v1 \
  --count 2000 \
  --max-nodes 40 \
  --seed 42

uv run --extra benchmark causal-portfolio --config configs/default.yaml run-synthetic-algorithms \
  --synthetic-root data/synthetic_bn/v1 \
  --output artifacts/synthetic_runs/v1 \
  --algorithms PC_discrete,FCI,GES,HC,Tabu,K2,MMHC,BOSS,GRaSP \
  --timeout-seconds 300 \
  --resume

# Recommended for the full 2000 dataset run: use 4 tmux shards.
# If interrupted, rerun the same shard command; --resume skips existing JSON records.
tmux new -s synthetic_algos_0
cd /home/oguzhan/causal-portfolio-selector
uv run --extra benchmark causal-portfolio --config configs/default.yaml run-synthetic-algorithms \
  --synthetic-root data/synthetic_bn/v1 \
  --output artifacts/synthetic_runs/v1 \
  --timeout-seconds 300 \
  --shard-index 0 \
  --shard-count 4 \
  --resume

# Repeat in separate tmux sessions with --shard-index 1, 2, and 3.
# Progress:
find artifacts/synthetic_runs/v1/records -type f | wc -l

causal-portfolio --config configs/default.yaml build-synthetic-training-tables \
  --synthetic-root data/synthetic_bn/v1 \
  --runs artifacts/synthetic_runs/v1 \
  --output artifacts/synthetic_tables/v1

uv run --with torch causal-portfolio --config configs/default.yaml train-fingerprint-from-synthetic \
  --synthetic-root data/synthetic_bn/v1 \
  --output artifacts/synthetic_models/v1/biaffine_encoder.pt \
  --device cuda \
  --epochs 100

causal-portfolio --config configs/default.yaml train-synthetic-selector \
  --tables artifacts/synthetic_tables/v1 \
  --encoder artifacts/synthetic_models/v1/biaffine_encoder.pt \
  --output artifacts/synthetic_models/v1/selector.joblib

causal-portfolio --config configs/default.yaml evaluate-synthetic-selector-on-exact \
  --model artifacts/synthetic_models/v1/selector.joblib \
  --encoder artifacts/synthetic_models/v1/biaffine_encoder.pt \
  --output reports/synthetic_v1

causal-portfolio predict \
  --dataset data/imported/datasets/asia.csv \
  --model artifacts/models/selector.joblib
```

If the package is not installed, use:

```bash
python -m causal_portfolio_selector.cli <command>
```

## Algorithm Pool

V1 is trained on the already-run six-algorithm pool:

- `PC_discrete`
- `FCI`
- `GES`
- `HC`
- `Tabu`
- `K2`

The missing-algorithm runner is restricted to:

- `MMHC`
- `BOSS`
- `GRaSP`

Its default output directory is:

- `artifacts/missing_algorithm_runs/latest`

It writes one JSON record per dataset-algorithm run and a `summary.csv`.
The command is resume-safe by default; existing records are skipped unless
`--overwrite` is passed.

## Outputs

- `artifacts/tables/features.csv`
- `artifacts/tables/targets.csv`
- `artifacts/models/selector.joblib`
- `reports/lodo_metrics.csv`
- `reports/legacy_split_metrics.csv`
- `reports/baseline_summary.csv`
- `reports/ablation_summary.csv`
- `reports/phase1_evidence.md`
- `artifacts/learned/biaffine_encoder.pt`
- `artifacts/tables/learned_features.csv`
- `artifacts/tables/features_plus_learned.csv`
- `reports/phase2_evidence.md`
- `artifacts/missing_algorithm_runs/latest/summary.csv`
- `artifacts/missing_algorithm_runs/latest/manifest.json`
- `reports/predictions_by_dataset.csv`
- `reports/summary.md`
