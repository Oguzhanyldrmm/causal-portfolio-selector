# Evaluation Report

Evaluation set: `artifacts/evaluation_tables/exact_8alg_no_win95pts`

Included saved final models:
- `original_1753_ranking`: `artifacts/synthetic_models/v1_8alg_handcrafted/selector.joblib` (objective=`ranking`, feature_set=`handcrafted_all`, train_dataset_count=`1402`)
- `balanced_985_ranking`: `artifacts/synthetic_models/v1_8alg_balanced_train_985_handcrafted/selector.joblib` (objective=`ranking`, feature_set=`handcrafted_all`, train_dataset_count=`985`)
- `balanced_985_top3_membership`: `artifacts/synthetic_models/v1_8alg_balanced_train_985_top3/selector.joblib` (objective=`top3_membership`, feature_set=`handcrafted_all`, train_dataset_count=`985`)
- `balanced_985_score_regression`: `artifacts/synthetic_models/v1_8alg_balanced_train_985_score/selector.joblib` (objective=`score_regression`, feature_set=`handcrafted_all`, train_dataset_count=`985`)

## Exact Summary

| model | objective | feature_set | train_dataset_count | top1_hit | oracle_in_top3 | top3_overlap_at_least_2 | avg_top3_overlap | avg_regret_at_3 | pred_top1_distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_1753_ranking | ranking | handcrafted_all | 1402 | 3/14 | 7/14 | 6/14 | 1.36 | 4.857 | BOSS:3, HC:8, K2:3 |
| balanced_985_ranking | ranking | handcrafted_all | 985 | 5/14 | 8/14 | 6/14 | 1.36 | 3.786 | BOSS:7, HC:2, K2:5 |
| balanced_985_top3_membership | top3_membership | handcrafted_all | 985 | 4/14 | 6/14 | 3/14 | 1.21 | 4.857 | BOSS:9, K2:5 |
| balanced_985_score_regression | score_regression | handcrafted_all | 985 | 1/14 | 7/14 | 6/14 | 1.14 | 4.143 | BOSS:1, HC:1, K2:10, Tabu:2 |

## Synthetic Val/Test Summary

| model | split | top1_hit | oracle_in_top3 | avg_top3_overlap | top3_overlap_at_least_2 | regret_at_3 |
| --- | --- | --- | --- | --- | --- | --- |
| original_1753_ranking | synthetic_val | 0.634 | 0.909 |  |  | 0.194 |
| original_1753_ranking | synthetic_test | 0.722 | 0.920 |  |  | 0.216 |
| balanced_985_ranking | synthetic_val | 0.600 | 0.891 |  |  | 0.183 |
| balanced_985_ranking | synthetic_test | 0.642 | 0.898 |  |  | 0.324 |
| balanced_985_top3_membership | synthetic_val | 0.560 | 0.886 | 1.806 | 0.697 | 0.206 |
| balanced_985_top3_membership | synthetic_test | 0.557 | 0.903 | 1.898 | 0.767 | 0.216 |
| balanced_985_score_regression | synthetic_val | 0.663 | 0.891 | 1.806 | 0.623 | 0.286 |
| balanced_985_score_regression | synthetic_test | 0.733 | 0.915 | 1.847 | 0.688 | 0.176 |

## Exact Dataset Top-3 Comparison

| dataset_name | ground_truth_top3 | original_1753_ranking_top3 | original_1753_ranking_overlap | original_1753_ranking_oracle_in_top3 | original_1753_ranking_regret_at_3 | balanced_985_ranking_top3 | balanced_985_ranking_overlap | balanced_985_ranking_oracle_in_top3 | balanced_985_ranking_regret_at_3 | balanced_985_top3_membership_top3 | balanced_985_top3_membership_overlap | balanced_985_top3_membership_oracle_in_top3 | balanced_985_top3_membership_regret_at_3 | balanced_985_score_regression_top3 | balanced_985_score_regression_overlap | balanced_985_score_regression_oracle_in_top3 | balanced_985_score_regression_regret_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alarm | PC_discrete, GRaSP, GES | HC, K2, BOSS | 0 | False | 15.0 | BOSS, K2, HC | 0 | False | 15.0 | BOSS, K2, GES | 1 | False | 14.0 | BOSS, GRaSP, PC_discrete | 2 | True | 0.0 |
| asia | HC, Tabu, GES | BOSS, K2, HC | 1 | True | 0.0 | BOSS, K2, HC | 1 | True | 0.0 | BOSS, GES, K2 | 1 | False | 1.0 | Tabu, HC, K2 | 2 | True | 0.0 |
| barley | PC_discrete, FCI, BOSS | HC, BOSS, GRaSP | 1 | False | 15.0 | HC, PC_discrete, BOSS | 2 | True | 0.0 | BOSS, HC, Tabu | 1 | False | 15.0 | K2, HC, Tabu | 0 | False | 15.0 |
| cancer | BOSS, HC, Tabu | BOSS, K2, HC | 2 | True | 0.0 | BOSS, K2, HC | 2 | True | 0.0 | BOSS, FCI, K2 | 1 | True | 0.0 | K2, PC_discrete, HC | 1 | False | 2.0 |
| cat_chain | HC, Tabu, BOSS | K2, HC, BOSS | 2 | True | 0.0 | K2, BOSS, HC | 2 | True | 0.0 | K2, HC, Tabu | 2 | True | 0.0 | K2, HC, Tabu | 2 | True | 0.0 |
| cat_collider | PC_discrete, BOSS, GRaSP | HC, K2, BOSS | 1 | False | 2.0 | K2, BOSS, HC | 1 | False | 2.0 | K2, HC, BOSS | 1 | False | 2.0 | K2, HC, Tabu | 0 | False | 2.0 |
| child | K2, BOSS, FCI | HC, K2, BOSS | 2 | True | 0.0 | K2, HC, BOSS | 2 | True | 0.0 | K2, BOSS, HC | 2 | True | 0.0 | K2, Tabu, HC | 1 | True | 0.0 |
| earthquake | BOSS, GES, GRaSP | HC, K2, BOSS | 1 | True | 0.0 | BOSS, K2, HC | 1 | True | 0.0 | BOSS, K2, Tabu | 1 | True | 0.0 | K2, HC, PC_discrete | 0 | False | 0.0 |
| hailfinder | PC_discrete, HC, Tabu | K2, HC, Tabu | 2 | False | 9.0 | K2, HC, GES | 1 | False | 9.0 | K2, HC, GES | 1 | False | 9.0 | K2, Tabu, HC | 2 | False | 9.0 |
| hepar2 | HC, Tabu, GES | HC, BOSS, K2 | 1 | True | 0.0 | HC, BOSS, GES | 2 | True | 0.0 | BOSS, HC, K2 | 1 | True | 0.0 | Tabu, HC, BOSS | 2 | True | 0.0 |
| insurance | PC_discrete, FCI, BOSS | HC, K2, BOSS | 1 | False | 4.0 | BOSS, K2, HC | 1 | False | 4.0 | BOSS, K2, HC | 1 | False | 4.0 | K2, Tabu, HC | 0 | False | 8.0 |
| mildew | PC_discrete, GES, HC | HC, K2, Tabu | 1 | False | 22.0 | K2, HC, BOSS | 1 | False | 22.0 | K2, BOSS, HC | 1 | False | 22.0 | K2, HC, Tabu | 1 | False | 22.0 |
| survey | PC_discrete, K2, HC | K2, BOSS, HC | 2 | False | 1.0 | BOSS, K2, GES | 1 | False | 1.0 | BOSS, K2, FCI | 1 | False | 1.0 | K2, PC_discrete, Tabu | 2 | True | 0.0 |
| water | BOSS, GRaSP, K2 | BOSS, HC, GRaSP | 2 | True | 0.0 | BOSS, GRaSP, HC | 2 | True | 0.0 | BOSS, GRaSP, GES | 2 | True | 0.0 | HC, BOSS, Tabu | 1 | True | 0.0 |

## kNN Prior Experiment

Experiment path: `experiments/knn_prior_balanced_985_ranking`

Model:
- `balanced_985_knn_prior_ranking`: `experiments/knn_prior_balanced_985_ranking/models/selector.joblib` (objective=`ranking`, feature_set=`handcrafted_plus_knn`, train_dataset_count=`985`, k=`50`)

Exact comparison against current best:

| model | objective | feature_set | train_dataset_count | oracle_in_top3 | top3_overlap_at_least_2 | avg_top3_overlap | avg_regret_at_3 | pred_top1_distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_985_ranking | ranking | handcrafted_all | 985 | 8/14 | 6/14 | 1.357 | 3.786 | BOSS:7, HC:2, K2:5 |
| balanced_985_knn_prior_ranking | ranking | handcrafted_plus_knn | 985 | 8/14 | 5/14 | 1.286 | 3.786 | BOSS:8, HC:2, K2:4 |

Conclusion: kNN prior did not improve the exact evaluation. It preserved `oracle_in_top3` and `avg_regret_at_3`, but reduced `top3_overlap_at_least_2` from `6/14` to `5/14`. The only meaningful dataset-level regression against `balanced_985_ranking` is `hepar2`, where the predicted top-3 changed from `HC, BOSS, GES` to `HC, K2, BOSS`; oracle coverage stayed true, but top-3 overlap dropped from `2` to `1`.

Detailed report: `experiments/knn_prior_balanced_985_ranking/reports/knn_prior_exact_report.md`

## Top-3 Combination Ranker Experiment

Experiment path: `experiments/top3_combination_ranker_balanced_985`

Model:
- `top3_combination_ranker_balanced_985`: `experiments/top3_combination_ranker_balanced_985/models/selector.joblib` (objective=`top3_combination_reward`, feature_set=`handcrafted_all`, train_dataset_count=`985`)

Reward:

```text
reward = 3.0 * oracle_in_top3
       + 1.0 * top3_overlap
       - 0.25 * regret_at_3
```

Exact comparison against current best:

| model | objective | feature_set | train_dataset_count | oracle_in_top3 | top3_overlap_at_least_2 | avg_top3_overlap | avg_regret_at_3 | pred_top1_distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_985_ranking | ranking | handcrafted_all | 985 | 8/14 | 6/14 | 1.357 | 3.786 | BOSS:7, HC:2, K2:5 |
| top3_combination_ranker_balanced_985 | top3_combination_reward | handcrafted_all | 985 | 10/14 | 5/14 | 1.143 | 2.571 | BOSS:1, GRaSP:1, HC:1, K2:11 |

Conclusion: top-3 combination ranker improves the main oracle/regret objective but weakens exact top-3 overlap. It recovers the oracle on `hailfinder`, `mildew`, and `survey`, but loses `barley`. This means it is useful if the operational goal is "include at least one best algorithm in the top 3 with low regret", but it is not yet better for "match at least two of the ground-truth top 3".

Detailed report: `experiments/top3_combination_ranker_balanced_985/reports/top3_combination_exact_report.md`

## Oracle-Regret Top-3 Combination Experiment

Experiment path: `experiments/oracle_regret_top3_combination`

Models:
- `oracle3_overlap1_regret025`: `experiments/oracle_regret_top3_combination/models/oracle3_overlap1_regret025/selector.joblib`
- `oracle3_overlap1_regret050`: `experiments/oracle_regret_top3_combination/models/oracle3_overlap1_regret050/selector.joblib`
- `oracle4_overlap1_regret025`: `experiments/oracle_regret_top3_combination/models/oracle4_overlap1_regret025/selector.joblib`

Exact comparison:

| model | reward | oracle_in_top3 | top3_overlap_at_least_2 | avg_top3_overlap | avg_regret_at_3 | pred_top1_distribution |
| --- | --- | --- | --- | --- | --- | --- |
| balanced_985_ranking | n/a | 8/14 | 6/14 | 1.357 | 3.786 | BOSS:7, HC:2, K2:5 |
| top3_combo_baseline_previous | oracle=3.0, overlap=1.0, regret=0.25 | 10/14 | 5/14 | 1.143 | 2.571 | BOSS:1, GRaSP:1, HC:1, K2:11 |
| oracle3_overlap1_regret025 | oracle=3.0, overlap=1.0, regret=0.25 | 10/14 | 5/14 | 1.143 | 2.571 | BOSS:1, GRaSP:1, HC:1, K2:11 |
| oracle3_overlap1_regret050 | oracle=3.0, overlap=1.0, regret=0.50 | 10/14 | 5/14 | 1.214 | 2.571 | BOSS:1, HC:1, K2:12 |
| oracle4_overlap1_regret025 | oracle=4.0, overlap=1.0, regret=0.25 | 11/14 | 4/14 | 1.143 | 1.357 | GRaSP:1, HC:1, K2:12 |

Conclusion: `oracle4_overlap1_regret025` is the best oracle/regret model so far. It improves `oracle_in_top3` from `8/14` to `11/14` versus `balanced_985_ranking`, and reduces `avg_regret_at_3` from `3.786` to `1.357`. The tradeoff is worse top-3 similarity: `top3_overlap_at_least_2` drops from `6/14` to `4/14`. It gains oracle coverage on `alarm`, `hailfinder`, `insurance`, `mildew`, and `survey`, but loses oracle coverage on `barley` and `cancer`.

Detailed report: `experiments/oracle_regret_top3_combination/reports/oracle_regret_experiment_report.md`

## Top-3 Overlap Combination Optimizer Experiment

Experiment path: `experiments/top3_overlap_combination_optimizer`

Models:
- `overlap2_oracle1_regret010`: `experiments/top3_overlap_combination_optimizer/models/overlap2_oracle1_regret010/selector.joblib`
- `overlap3_oracle1_regret010`: `experiments/top3_overlap_combination_optimizer/models/overlap3_oracle1_regret010/selector.joblib`
- `overlap2_oracle2_regret010`: `experiments/top3_overlap_combination_optimizer/models/overlap2_oracle2_regret010/selector.joblib`

Exact comparison:

| model | reward | oracle_in_top3 | top3_overlap_at_least_2 | avg_top3_overlap | avg_regret_at_3 | pred_top1_distribution |
| --- | --- | --- | --- | --- | --- | --- |
| balanced_985_ranking | n/a | 8/14 | 6/14 | 1.357 | 3.786 | BOSS:7, HC:2, K2:5 |
| oracle4_overlap1_regret025 | oracle=4.0, overlap=1.0, overlap>=2=0.0, regret=0.25 | 11/14 | 4/14 | 1.143 | 1.357 | GRaSP:1, HC:1, K2:12 |
| overlap2_oracle1_regret010 | oracle=1.0, overlap=2.0, overlap>=2=2.0, regret=0.10 | 8/14 | 8/14 | 1.500 | 3.357 | BOSS:3, HC:3, K2:8 |
| overlap3_oracle1_regret010 | oracle=1.0, overlap=3.0, overlap>=2=2.0, regret=0.10 | 9/14 | 8/14 | 1.357 | 3.500 | BOSS:3, HC:3, K2:8 |
| overlap2_oracle2_regret010 | oracle=2.0, overlap=2.0, overlap>=2=2.0, regret=0.10 | 9/14 | 7/14 | 1.357 | 3.214 | BOSS:1, HC:2, K2:11 |

Conclusion: `overlap2_oracle1_regret010` is the best overlap-oriented model so far. It improves `top3_overlap_at_least_2` from `6/14` to `8/14`, improves `avg_top3_overlap` from `1.357` to `1.500`, keeps `oracle_in_top3` at `8/14`, and slightly improves `avg_regret_at_3` from `3.786` to `3.357`. The main regression is `barley`, where oracle coverage is lost and regret increases by `15.0`; the largest gain is `mildew`, where oracle coverage is recovered and regret drops by `22.0`.

Detailed report: `experiments/top3_overlap_combination_optimizer/reports/top3_overlap_experiment_report.md`

## Reward-Sum Top-3 Ensemble Experiment

Experiment path: `experiments/ensemble_reward_sum_top3`

Parent models:
- Oracle/regret parent: `experiments/oracle_regret_top3_combination/models/oracle4_overlap1_regret025/selector.joblib`
- Overlap parent: `experiments/top3_overlap_combination_optimizer/models/overlap2_oracle1_regret010/selector.joblib`

Method:

```text
For each dataset:
  1. Score all 56 top-3 combinations with oracle parent.
  2. Score all 56 top-3 combinations with overlap parent.
  3. Min-max normalize each parent's 56 rewards separately.
  4. ensemble_score = alpha * normalized_oracle_reward + beta * normalized_overlap_reward.
  5. Select the highest-scoring combination.
```

Exact comparison:

| model | alpha | beta | oracle_in_top3 | top3_overlap_at_least_2 | avg_top3_overlap | avg_regret_at_3 |
| --- | --- | --- | --- | --- | --- | --- |
| balanced_985_ranking | n/a | n/a | 8/14 | 6/14 | 1.357 | 3.786 |
| oracle4_overlap1_regret025 | n/a | n/a | 11/14 | 4/14 | 1.143 | 1.357 |
| overlap2_oracle1_regret010 | n/a | n/a | 8/14 | 8/14 | 1.500 | 3.357 |
| ensemble_balanced_100_100 | 1.0 | 1.0 | 9/14 | 6/14 | 1.286 | 3.286 |
| ensemble_oracle_125_100 | 1.25 | 1.0 | 9/14 | 6/14 | 1.286 | 3.214 |
| ensemble_overlap_100_125 | 1.0 | 1.25 | 9/14 | 6/14 | 1.286 | 3.286 |
| ensemble_oracle_150_100 | 1.5 | 1.0 | 9/14 | 6/14 | 1.286 | 3.214 |
| ensemble_overlap_100_150 | 1.0 | 1.5 | 9/14 | 6/14 | 1.286 | 3.286 |

Conclusion: simple normalized reward-sum ensemble did not combine the two parent strengths. It improves over `balanced_985_ranking` on `oracle_in_top3` and `avg_regret_at_3`, but loses the overlap parent's `8/14` top3-overlap-at-least-2 performance and is far worse than the oracle parent on oracle/regret. The main failure mode is that reward-sum suppresses the oracle parent's PC_discrete choices on datasets like `alarm`, `hailfinder`, and `insurance`.

Detailed report: `experiments/ensemble_reward_sum_top3/reports/ensemble_reward_sum_report.md`
