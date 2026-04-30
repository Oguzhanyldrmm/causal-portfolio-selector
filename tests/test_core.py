from __future__ import annotations

import pandas as pd

from causal_portfolio_selector.evaluation import evaluate_prediction
from causal_portfolio_selector.experiments import FEATURE_SETS, _prediction_from_order
from causal_portfolio_selector.features import FEATURE_COLUMNS
from causal_portfolio_selector.models import build_pairwise_rows, build_regression_rows
from causal_portfolio_selector.targets import feature_matrix


def _feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "d1",
                "n_samples": 100.0,
                "n_features": 3.0,
                "sample_to_feature_ratio": 33.3,
                "continuous_ratio": 0.0,
                "categorical_ratio": 1.0,
                "missing_ratio": 0.0,
                "avg_variance": 0.1,
                "avg_skewness": 0.0,
                "avg_kurtosis": 0.0,
                "avg_cardinality": 2.0,
                "max_cardinality": 2.0,
                "cardinality_entropy": 1.0,
                "rare_category_ratio": 0.0,
                "singleton_category_ratio": 0.0,
                "feature_sparsity_ratio": 0.5,
                "mean_nmi": 0.1,
                "max_nmi": 0.2,
                "std_nmi": 0.05,
                "mean_cramers_v": 0.1,
                "max_cramers_v": 0.2,
                "std_cramers_v": 0.05,
                "mean_chi2_pvalue": 0.8,
                "ci_rejection_rate": 0.1,
                "proxy_graph_density": 0.5,
                "proxy_avg_degree": 0.5,
                "proxy_degree_gini": 0.0,
                "proxy_avg_clustering": 0.0,
                "proxy_modularity": 0.0,
                "proxy_num_components": 1.0,
            }
        ]
    )


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "d1",
                "algorithm_name": "A",
                "quality_rank": 1,
                "shd": 1.0,
                "combined_score": 0.9,
                "relative_regret": 0.0,
            },
            {
                "dataset_name": "d1",
                "algorithm_name": "B",
                "quality_rank": 2,
                "shd": 3.0,
                "combined_score": 0.4,
                "relative_regret": 0.2,
            },
            {
                "dataset_name": "d1",
                "algorithm_name": "C",
                "quality_rank": 3,
                "shd": 5.0,
                "combined_score": 0.1,
                "relative_regret": 0.4,
            },
        ]
    )


def test_feature_matrix_rejects_leakage_column() -> None:
    features = _feature_table()
    features["shd"] = 1.0
    try:
        feature_matrix(features)
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("Expected leakage column check to fail")


def test_pairwise_rows_encode_quality_order() -> None:
    rows = build_pairwise_rows(
        _feature_table(),
        _targets(),
        dataset_names=["d1"],
        algorithms=["A", "B", "C"],
    )
    assert rows.shape[0] == 3
    assert rows["label"].tolist() == [1, 1, 1]


def test_regression_rows_include_one_row_per_algorithm() -> None:
    rows = build_regression_rows(
        _feature_table(),
        _targets(),
        dataset_names=["d1"],
        algorithms=["A", "B", "C"],
    )
    assert rows.shape[0] == 3
    assert set(rows["algorithm_name"]) == {"A", "B", "C"}


def test_regret_at_three_not_greater_than_regret_at_one() -> None:
    prediction = {
        "ranking": {"B": 1, "C": 2, "A": 3},
        "pairwise_win_score": {"B": 2.0, "C": 1.0, "A": 0.0},
        "predicted_relative_regret": {"B": 0.2, "C": 0.4, "A": 0.0},
    }
    metrics = evaluate_prediction(dataset_name="d1", prediction=prediction, targets=_targets())
    assert metrics["regret_at_3"] <= metrics["regret_at_1"]
    assert metrics["top3_hit"] == 1.0


def test_feature_sets_are_known_columns() -> None:
    known = set(FEATURE_COLUMNS)
    for columns in FEATURE_SETS.values():
        assert set(columns).issubset(known)
    assert FEATURE_SETS["all"] == FEATURE_COLUMNS


def test_prediction_from_order_returns_distinct_top3() -> None:
    prediction = _prediction_from_order(("A", "B", "C", "D"))
    assert prediction["top_3"] == ["A", "B", "C"]
    assert len(set(prediction["top_3"])) == 3
