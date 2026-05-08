# Causal Portfolio Selector

Final 8-algorithm causal discovery portfolio selector for discrete Bayesian-network datasets.

The repository contains the code, configuration, tests, and compact reports. Large datasets, algorithm run records, and trained model binaries are kept outside Git in the final artifact export bundle.

## Final Setup

Algorithm portfolio:

- PC_discrete
- FCI
- GES
- HC
- Tabu
- K2
- BOSS
- GRaSP

Final training data:

- Synthetic BN complete set: 1753 datasets where all 8 algorithms succeeded.
- Balanced synthetic training subset: 985 train datasets, with validation/test tables kept for evaluation.
- Exact evaluation set: 14 held-out exact BN datasets.

Final selectors:

- `oracle4_overlap1_regret025`: top-3 combination reward regressor optimized for oracle coverage and low regret.
- `overlap2_oracle1_regret010`: top-3 combination reward regressor optimized for ground-truth top-3 overlap.
- `balanced_985_ranking`: retained baseline ranking selector.

## Artifact Bundle

Large local artifacts are not committed to Git. The final export bundle is:

```text
/home/oguzhan_yildirim/causal_selection_final_export_20260508
```

It contains:

- final synthetic and exact evaluation datasets;
- ground-truth graph files;
- 8-algorithm synthetic and exact run outputs;
- synthetic/evaluation training tables;
- final model binaries;
- checksums and export manifest.

See `docs/final_artifacts_manifest.md` for the bundle layout and validation counts.

## Repository Layout

```text
configs/     Default configuration.
docs/        Final reports and artifact manifest.
src/         Portfolio selector implementation and CLI.
tests/       Unit tests.
```

Ignored local-only directories include `data/`, `artifacts/`, `experiments/`, and `logs/`.

## Usage

Install dependencies:

```bash
uv sync --extra benchmark --extra learned --extra dev
```

Run tests:

```bash
uv run pytest
```

Train a top-3 combination selector when the synthetic tables are restored locally:

```bash
uv run causal-portfolio --config configs/default.yaml train-synthetic-top3-combination-selector \
  --tables artifacts/synthetic_tables/v1_8alg_balanced_train_985 \
  --output artifacts/synthetic_models/example/selector.joblib
```

Evaluate a saved selector on exact evaluation tables:

```bash
uv run causal-portfolio --config configs/default.yaml evaluate-synthetic-selector-on-exact \
  --model artifacts/synthetic_models/example/selector.joblib \
  --output reports/example_eval
```

## Reports

- `docs/Best_Models_Report.md`
- `docs/Evaluation_report.md`
- `docs/Synthetic_Generation_Groundtruth_Analysis.md`
- `docs/final_artifacts_manifest.md`

## Notes

The final selectors use supervised reward regression over all 56 possible top-3 algorithm combinations. At inference time, the model scores each combination and returns the highest predicted-reward top-3 set.
