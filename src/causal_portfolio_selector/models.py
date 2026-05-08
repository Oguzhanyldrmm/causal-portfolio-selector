from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import ModelConfig
from .features import FEATURE_COLUMNS
from .targets import feature_matrix


@dataclass
class PortfolioSelector:
    algorithms: tuple[str, ...]
    feature_names: tuple[str, ...]
    pairwise_model: Pipeline
    regret_model: Pipeline

    def predict_from_features(self, features: dict[str, float] | pd.Series) -> dict[str, Any]:
        feature_row = pd.DataFrame([{name: float(features.get(name, 0.0)) for name in self.feature_names}])
        pair_rows = []
        for index, algorithm_a in enumerate(self.algorithms):
            for algorithm_b in self.algorithms[index + 1 :]:
                row = feature_row.iloc[0].to_dict()
                row["algorithm_a"] = algorithm_a
                row["algorithm_b"] = algorithm_b
                pair_rows.append(row)

        pair_frame = pd.DataFrame(pair_rows)
        pair_matrix = design_matrix_pairwise(
            pair_frame,
            feature_names=self.feature_names,
            algorithms=self.algorithms,
        )
        probabilities = _positive_class_probability(self.pairwise_model, pair_matrix)
        win_scores = {algorithm: 0.0 for algorithm in self.algorithms}
        for row, probability in zip(pair_rows, probabilities):
            algorithm_a = str(row["algorithm_a"])
            algorithm_b = str(row["algorithm_b"])
            p_a = float(probability)
            win_scores[algorithm_a] += p_a
            win_scores[algorithm_b] += 1.0 - p_a

        regret_rows = []
        for algorithm_name in self.algorithms:
            row = feature_row.iloc[0].to_dict()
            row["algorithm_name"] = algorithm_name
            regret_rows.append(row)
        regret_frame = pd.DataFrame(regret_rows)
        regret_matrix = design_matrix_regression(
            regret_frame,
            feature_names=self.feature_names,
            algorithms=self.algorithms,
        )
        predicted_regrets = self.regret_model.predict(regret_matrix)
        regret_by_algorithm = {
            algorithm: float(max(0.0, regret))
            for algorithm, regret in zip(self.algorithms, predicted_regrets)
        }

        ranked = sorted(
            self.algorithms,
            key=lambda algorithm: (
                -win_scores[algorithm],
                regret_by_algorithm[algorithm],
                algorithm,
            ),
        )
        return {
            "top_3": ranked[:3],
            "ranking": {algorithm: rank + 1 for rank, algorithm in enumerate(ranked)},
            "pairwise_win_score": {algorithm: float(win_scores[algorithm]) for algorithm in ranked},
            "predicted_relative_regret": {
                algorithm: float(regret_by_algorithm[algorithm]) for algorithm in ranked
            },
        }


@dataclass
class Top3MembershipSelector:
    algorithms: tuple[str, ...]
    feature_names: tuple[str, ...]
    membership_model: Pipeline

    def predict_from_features(self, features: dict[str, float] | pd.Series) -> dict[str, Any]:
        feature_row = pd.DataFrame([{name: float(features.get(name, 0.0)) for name in self.feature_names}])
        rows = []
        for algorithm_name in self.algorithms:
            row = feature_row.iloc[0].to_dict()
            row["algorithm_name"] = algorithm_name
            rows.append(row)
        frame = pd.DataFrame(rows)
        matrix = design_matrix_regression(
            frame,
            feature_names=self.feature_names,
            algorithms=self.algorithms,
        )
        probabilities = _positive_class_probability(self.membership_model, matrix)
        probability_by_algorithm = {
            algorithm: float(probability)
            for algorithm, probability in zip(self.algorithms, probabilities)
        }
        ranked = sorted(
            self.algorithms,
            key=lambda algorithm: (
                -probability_by_algorithm[algorithm],
                algorithm,
            ),
        )
        return {
            "top_3": ranked[:3],
            "ranking": {algorithm: rank + 1 for rank, algorithm in enumerate(ranked)},
            "top3_membership_probability": {
                algorithm: float(probability_by_algorithm[algorithm]) for algorithm in ranked
            },
        }


@dataclass
class ScoreRegressionSelector:
    algorithms: tuple[str, ...]
    feature_names: tuple[str, ...]
    score_model: Pipeline
    target_column: str

    def predict_from_features(self, features: dict[str, float] | pd.Series) -> dict[str, Any]:
        feature_row = pd.DataFrame([{name: float(features.get(name, 0.0)) for name in self.feature_names}])
        rows = []
        for algorithm_name in self.algorithms:
            row = feature_row.iloc[0].to_dict()
            row["algorithm_name"] = algorithm_name
            rows.append(row)
        frame = pd.DataFrame(rows)
        matrix = design_matrix_regression(
            frame,
            feature_names=self.feature_names,
            algorithms=self.algorithms,
        )
        predicted_scores = self.score_model.predict(matrix)
        score_by_algorithm = {
            algorithm: float(score)
            for algorithm, score in zip(self.algorithms, predicted_scores)
        }
        ranked = sorted(
            self.algorithms,
            key=lambda algorithm: (
                -score_by_algorithm[algorithm],
                algorithm,
            ),
        )
        return {
            "top_3": ranked[:3],
            "ranking": {algorithm: rank + 1 for rank, algorithm in enumerate(ranked)},
            "predicted_score": {algorithm: float(score_by_algorithm[algorithm]) for algorithm in ranked},
        }


@dataclass
class Top3CombinationSelector:
    algorithms: tuple[str, ...]
    feature_names: tuple[str, ...]
    combo_model: Pipeline
    reward_weights: dict[str, float]

    def predict_from_features(self, features: dict[str, float] | pd.Series) -> dict[str, Any]:
        feature_row = pd.DataFrame([{name: float(features.get(name, 0.0)) for name in self.feature_names}])
        rows = []
        combos = list(itertools.combinations(self.algorithms, 3))
        for combo in combos:
            row = feature_row.iloc[0].to_dict()
            row["top3_combo"] = combo
            rows.append(row)
        frame = pd.DataFrame(rows)
        matrix = design_matrix_top3_combination(
            frame,
            feature_names=self.feature_names,
            algorithms=self.algorithms,
        )
        predicted_rewards = self.combo_model.predict(matrix)
        combo_rewards = {
            combo: float(reward)
            for combo, reward in zip(combos, predicted_rewards)
        }
        best_combo = max(combo_rewards, key=lambda combo: (combo_rewards[combo], tuple(reversed(combo))))
        marginal_rewards = {
            algorithm: float(
                np.mean([reward for combo, reward in combo_rewards.items() if algorithm in combo])
            )
            for algorithm in self.algorithms
        }
        ranked = sorted(
            self.algorithms,
            key=lambda algorithm: (
                algorithm not in best_combo,
                -marginal_rewards[algorithm],
                algorithm,
            ),
        )
        return {
            "top_3": ranked[:3],
            "ranking": {algorithm: rank + 1 for rank, algorithm in enumerate(ranked)},
            "selected_top3_combo": list(best_combo),
            "predicted_combo_reward": {
                ",".join(combo): float(reward)
                for combo, reward in sorted(combo_rewards.items(), key=lambda item: (-item[1], item[0]))
            },
            "algorithm_marginal_combo_reward": {
                algorithm: float(marginal_rewards[algorithm]) for algorithm in ranked
            },
        }


def _positive_class_probability(model: Pipeline, matrix: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    if classes == [1]:
        return np.ones(matrix.shape[0], dtype=float)
    if classes == [0]:
        return np.zeros(matrix.shape[0], dtype=float)
    return probabilities[:, -1]


def design_matrix_pairwise(
    frame: pd.DataFrame,
    *,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
    algorithms: Iterable[str],
) -> pd.DataFrame:
    feature_names = tuple(feature_names)
    algorithms = tuple(algorithms)
    matrix = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").copy()
    for algorithm in algorithms:
        matrix[f"algorithm_a__{algorithm}"] = (frame["algorithm_a"] == algorithm).astype(float)
        matrix[f"algorithm_b__{algorithm}"] = (frame["algorithm_b"] == algorithm).astype(float)
    return matrix


def design_matrix_regression(
    frame: pd.DataFrame,
    *,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
    algorithms: Iterable[str],
) -> pd.DataFrame:
    feature_names = tuple(feature_names)
    algorithms = tuple(algorithms)
    matrix = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").copy()
    for algorithm in algorithms:
        matrix[f"algorithm__{algorithm}"] = (frame["algorithm_name"] == algorithm).astype(float)
    return matrix


def design_matrix_top3_combination(
    frame: pd.DataFrame,
    *,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
    algorithms: Iterable[str],
) -> pd.DataFrame:
    feature_names = tuple(feature_names)
    algorithms = tuple(algorithms)
    matrix = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").copy()
    combos = frame["top3_combo"].apply(lambda value: set(value))
    for algorithm in algorithms:
        matrix[f"combo_has__{algorithm}"] = combos.apply(lambda combo: algorithm in combo).astype(float)
    constraint_based = {"PC_discrete", "FCI"}
    score_based = {"GES", "BOSS", "K2"}
    search_based = set(algorithms) - constraint_based - score_based
    for group_name, group_algorithms in (
        ("constraint_based", constraint_based),
        ("score_based", score_based),
        ("search_based", search_based),
    ):
        available = set(algorithms) & group_algorithms
        matrix[f"combo_has__{group_name}"] = combos.apply(
            lambda combo: bool(combo & available)
        ).astype(float)
        matrix[f"combo_count__{group_name}"] = combos.apply(
            lambda combo: len(combo & available)
        ).astype(float)
    return matrix


def build_pairwise_rows(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    feature_names: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    algorithms = tuple(algorithms)
    feature_names = tuple(feature_names)
    features = feature_matrix(feature_table, feature_names=list(feature_names)).set_index("dataset_name")
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted(dataset_names):
        if dataset_name not in features.index:
            continue
        base = features.loc[dataset_name, list(feature_names)].to_dict()
        group = targets[targets["dataset_name"] == dataset_name].set_index("algorithm_name")
        for index, algorithm_a in enumerate(algorithms):
            for algorithm_b in algorithms[index + 1 :]:
                if algorithm_a not in group.index or algorithm_b not in group.index:
                    continue
                rank_a = float(group.loc[algorithm_a, "quality_rank"])
                rank_b = float(group.loc[algorithm_b, "quality_rank"])
                row = dict(base)
                row.update(
                    {
                        "dataset_name": dataset_name,
                        "algorithm_a": algorithm_a,
                        "algorithm_b": algorithm_b,
                        "label": int(rank_a < rank_b),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def build_regression_rows(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    feature_names: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    algorithms = tuple(algorithms)
    feature_names = tuple(feature_names)
    features = feature_matrix(feature_table, feature_names=list(feature_names)).set_index("dataset_name")
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted(dataset_names):
        if dataset_name not in features.index:
            continue
        base = features.loc[dataset_name, list(feature_names)].to_dict()
        group = targets[
            (targets["dataset_name"] == dataset_name)
            & (targets["algorithm_name"].isin(algorithms))
        ]
        for _, target_row in group.iterrows():
            row = dict(base)
            row.update(
                {
                    "dataset_name": dataset_name,
                    "algorithm_name": target_row["algorithm_name"],
                    "relative_regret": float(target_row["relative_regret"]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_top3_membership_rows(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    feature_names: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    algorithms = tuple(algorithms)
    feature_names = tuple(feature_names)
    features = feature_matrix(feature_table, feature_names=list(feature_names)).set_index("dataset_name")
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted(dataset_names):
        if dataset_name not in features.index:
            continue
        base = features.loc[dataset_name, list(feature_names)].to_dict()
        group = targets[
            (targets["dataset_name"] == dataset_name)
            & (targets["algorithm_name"].isin(algorithms))
        ].set_index("algorithm_name")
        for algorithm_name in algorithms:
            if algorithm_name not in group.index:
                continue
            row = dict(base)
            row.update(
                {
                    "dataset_name": dataset_name,
                    "algorithm_name": algorithm_name,
                    "top3_member": int(float(group.loc[algorithm_name, "quality_rank"]) <= 3.0),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_score_rows(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    score_column: str,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    algorithms = tuple(algorithms)
    feature_names = tuple(feature_names)
    features = feature_matrix(feature_table, feature_names=list(feature_names)).set_index("dataset_name")
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted(dataset_names):
        if dataset_name not in features.index:
            continue
        base = features.loc[dataset_name, list(feature_names)].to_dict()
        group = targets[
            (targets["dataset_name"] == dataset_name)
            & (targets["algorithm_name"].isin(algorithms))
        ]
        for _, target_row in group.iterrows():
            if pd.isna(target_row.get(score_column)):
                continue
            row = dict(base)
            row.update(
                {
                    "dataset_name": dataset_name,
                    "algorithm_name": target_row["algorithm_name"],
                    "score_target": float(target_row[score_column]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_top3_combination_rows(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    feature_names: Iterable[str] = FEATURE_COLUMNS,
    oracle_weight: float = 3.0,
    overlap_weight: float = 1.0,
    overlap_at_least_2_weight: float = 0.0,
    regret_weight: float = 0.25,
) -> pd.DataFrame:
    algorithms = tuple(algorithms)
    feature_names = tuple(feature_names)
    features = feature_matrix(feature_table, feature_names=list(feature_names)).set_index("dataset_name")
    rows: list[dict[str, Any]] = []
    combos = tuple(itertools.combinations(algorithms, 3))
    for dataset_name in sorted(dataset_names):
        if dataset_name not in features.index:
            continue
        base = features.loc[dataset_name, list(feature_names)].to_dict()
        group = targets[
            (targets["dataset_name"] == dataset_name)
            & (targets["algorithm_name"].isin(algorithms))
        ].copy()
        if group.empty:
            continue
        shd_by_algorithm = {
            str(row["algorithm_name"]): float(row["shd"])
            for _, row in group.iterrows()
        }
        oracle_shd = min(shd_by_algorithm.values())
        oracle_algorithms = {
            algorithm
            for algorithm, shd in shd_by_algorithm.items()
            if np.isclose(shd, oracle_shd)
        }
        actual_top3 = set(
            group.sort_values(
                ["quality_rank", "shd", "combined_score", "algorithm_name"],
                ascending=[True, True, False, True],
            )["algorithm_name"]
            .astype(str)
            .head(3)
            .tolist()
        )
        for combo in combos:
            if not all(algorithm in shd_by_algorithm for algorithm in combo):
                continue
            combo_set = set(combo)
            oracle_in_top3 = float(bool(combo_set & oracle_algorithms))
            top3_overlap = float(len(combo_set & actual_top3))
            top3_overlap_at_least_2 = float(top3_overlap >= 2.0)
            regret_at_3 = float(min(shd_by_algorithm[algorithm] for algorithm in combo) - oracle_shd)
            reward = (
                float(oracle_weight) * oracle_in_top3
                + float(overlap_weight) * top3_overlap
                + float(overlap_at_least_2_weight) * top3_overlap_at_least_2
                - float(regret_weight) * regret_at_3
            )
            row = dict(base)
            row.update(
                {
                    "dataset_name": dataset_name,
                    "top3_combo": combo,
                    "oracle_in_top3": oracle_in_top3,
                    "top3_overlap": top3_overlap,
                    "top3_overlap_at_least_2": top3_overlap_at_least_2,
                    "regret_at_3": regret_at_3,
                    "combo_reward": float(reward),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def train_selector(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    config: ModelConfig | None = None,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
) -> PortfolioSelector:
    config = config or ModelConfig()
    algorithms = tuple(algorithms)
    feature_names = tuple(feature_names)

    pairwise_rows = build_pairwise_rows(
        feature_table,
        targets,
        dataset_names=dataset_names,
        algorithms=algorithms,
        feature_names=feature_names,
    )
    regression_rows = build_regression_rows(
        feature_table,
        targets,
        dataset_names=dataset_names,
        algorithms=algorithms,
        feature_names=feature_names,
    )
    if pairwise_rows.empty or regression_rows.empty:
        raise ValueError("Training rows are empty; check dataset split and imported artifacts.")

    pairwise_matrix = design_matrix_pairwise(
        pairwise_rows,
        feature_names=feature_names,
        algorithms=algorithms,
    )
    regression_matrix = design_matrix_regression(
        regression_rows,
        feature_names=feature_names,
        algorithms=algorithms,
    )

    pairwise_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    min_samples_leaf=config.min_samples_leaf,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    regret_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    pairwise_model.fit(pairwise_matrix, pairwise_rows["label"].astype(int))
    regret_model.fit(regression_matrix, regression_rows["relative_regret"].astype(float))
    return PortfolioSelector(
        algorithms=algorithms,
        feature_names=feature_names,
        pairwise_model=pairwise_model,
        regret_model=regret_model,
    )


def train_top3_membership_selector(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    config: ModelConfig | None = None,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
) -> Top3MembershipSelector:
    config = config or ModelConfig()
    algorithms = tuple(algorithms)
    feature_names = tuple(feature_names)
    rows = build_top3_membership_rows(
        feature_table,
        targets,
        dataset_names=dataset_names,
        algorithms=algorithms,
        feature_names=feature_names,
    )
    if rows.empty:
        raise ValueError("Training rows are empty; check dataset split and imported artifacts.")
    matrix = design_matrix_regression(
        rows,
        feature_names=feature_names,
        algorithms=algorithms,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    min_samples_leaf=config.min_samples_leaf,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    model.fit(matrix, rows["top3_member"].astype(int))
    return Top3MembershipSelector(
        algorithms=algorithms,
        feature_names=feature_names,
        membership_model=model,
    )


def train_score_selector(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    score_column: str = "combined_score",
    config: ModelConfig | None = None,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
) -> ScoreRegressionSelector:
    config = config or ModelConfig()
    algorithms = tuple(algorithms)
    feature_names = tuple(feature_names)
    rows = build_score_rows(
        feature_table,
        targets,
        dataset_names=dataset_names,
        algorithms=algorithms,
        score_column=score_column,
        feature_names=feature_names,
    )
    if rows.empty:
        raise ValueError("Training rows are empty; check dataset split and imported artifacts.")
    matrix = design_matrix_regression(
        rows,
        feature_names=feature_names,
        algorithms=algorithms,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    model.fit(matrix, rows["score_target"].astype(float))
    return ScoreRegressionSelector(
        algorithms=algorithms,
        feature_names=feature_names,
        score_model=model,
        target_column=score_column,
    )


def train_top3_combination_selector(
    feature_table: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    dataset_names: Iterable[str],
    algorithms: Iterable[str],
    config: ModelConfig | None = None,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
    oracle_weight: float = 3.0,
    overlap_weight: float = 1.0,
    overlap_at_least_2_weight: float = 0.0,
    regret_weight: float = 0.25,
) -> Top3CombinationSelector:
    config = config or ModelConfig()
    algorithms = tuple(algorithms)
    feature_names = tuple(feature_names)
    rows = build_top3_combination_rows(
        feature_table,
        targets,
        dataset_names=dataset_names,
        algorithms=algorithms,
        feature_names=feature_names,
        oracle_weight=oracle_weight,
        overlap_weight=overlap_weight,
        overlap_at_least_2_weight=overlap_at_least_2_weight,
        regret_weight=regret_weight,
    )
    if rows.empty:
        raise ValueError("Training rows are empty; check dataset split and imported artifacts.")
    matrix = design_matrix_top3_combination(
        rows,
        feature_names=feature_names,
        algorithms=algorithms,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    model.fit(matrix, rows["combo_reward"].astype(float))
    return Top3CombinationSelector(
        algorithms=algorithms,
        feature_names=feature_names,
        combo_model=model,
        reward_weights={
            "oracle_in_top3": float(oracle_weight),
            "top3_overlap": float(overlap_weight),
            "top3_overlap_at_least_2": float(overlap_at_least_2_weight),
            "regret_at_3": float(regret_weight),
        },
    )


def save_selector(
    selector: PortfolioSelector | Top3MembershipSelector | ScoreRegressionSelector | Top3CombinationSelector,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selector, path)


def load_selector(path: str | Path) -> PortfolioSelector:
    return joblib.load(path)
