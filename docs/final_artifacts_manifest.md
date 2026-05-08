# Final Artifact Manifest

Final external artifact bundle:

```text
/home/oguzhan_yildirim/causal_selection_final_export_20260508
```

This bundle is intentionally kept outside Git because it contains datasets, run records, and large trained model binaries.

## Bundle Layout

```text
datasets/
  synthetic_8alg_complete_1753/
  synthetic_8alg_balanced_train_985/
  evaluation_exact_8alg_no_win95pts/
algorithm_runs/
  synthetic_8alg_complete/
  evaluation_exact_8alg_no_win95pts/
training_tables/
  synthetic_8alg_complete/
  synthetic_8alg_balanced_train_985/
  evaluation_exact_8alg_no_win95pts/
models/
  oracle4_overlap1_regret025/
  overlap2_oracle1_regret010/
  balanced_985_ranking/
reports/
manifest/
```

## Validation Counts

| item | count |
| --- | ---: |
| synthetic complete dataset CSV files | 1753 |
| synthetic complete ground-truth JSON files | 1753 |
| synthetic balanced dataset CSV files | 985 |
| synthetic balanced ground-truth JSON files | 985 |
| exact evaluation dataset CSV files | 14 |
| exact evaluation ground-truth JSON files | 14 |
| synthetic 8-algorithm run JSON records | 14024 |
| exact evaluation BOSS/GRaSP run JSON records | 28 |
| synthetic complete target rows | 14024 |
| balanced table target rows | 10688 |
| exact evaluation target rows | 112 |
| final model binaries | 3 |

## Final Models

| model | role |
| --- | --- |
| oracle4_overlap1_regret025 | best oracle/regret selector |
| overlap2_oracle1_regret010 | best top-3 overlap selector |
| balanced_985_ranking | retained baseline selector |

All three model files were load-tested after export.

## Integrity Files

| file | purpose |
| --- | --- |
| manifest/export_manifest.json | structured export metadata and validation status |
| manifest/checksums.sha256 | SHA256 checksums for exported files |
| manifest/file_list.txt | complete exported file list |
| manifest/file_counts.txt | grouped file counts |
| manifest/directory_sizes.txt | directory size summary |

## Restore Notes

To use the bundle with this repo, copy or symlink the needed bundle folders back into the repo-local ignored paths, for example:

```text
data/synthetic_bn/v1_8alg_complete
data/synthetic_bn/v1_8alg_balanced_train_985
data/evaluation/exact_8alg_no_win95pts
artifacts/synthetic_runs/v1_8alg_complete
artifacts/evaluation_runs/exact_8alg_no_win95pts
artifacts/synthetic_tables/v1_8alg_balanced_train_985
artifacts/evaluation_tables/exact_8alg_no_win95pts
```

These paths are ignored by Git on purpose.
