from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score


VARIABLE_FEATURE_NAMES: tuple[str, ...] = (
    "log_n_samples",
    "log_n_variables",
    "log_cardinality",
    "entropy",
    "mode_ratio",
    "rare_category_ratio",
    "singleton_category_ratio",
    "mean_nmi_to_others",
    "max_nmi_to_others",
    "mean_cramers_to_others",
    "max_cramers_to_others",
    "proxy_degree",
)

PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "nmi",
    "cramers_v",
    "chi2_strength",
    "entropy_abs_diff",
    "log_cardinality_product",
    "same_cardinality",
    "mode_ratio_abs_diff",
    "proxy_edge",
)


def dataframe_to_learned_inputs(
    dataset: str | Path | pd.DataFrame,
    *,
    association_threshold: float = 0.05,
    max_rows: int | None = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(dataset) if not isinstance(dataset, pd.DataFrame) else dataset.copy()
    df.columns = [str(column) for column in df.columns]
    if max_rows is not None and max_rows > 0 and df.shape[0] > max_rows:
        df = df.sample(n=int(max_rows), random_state=int(random_seed)).reset_index(drop=True)
    n_samples, n_variables = df.shape
    columns = list(df.columns)

    cardinalities: list[float] = []
    entropies: list[float] = []
    mode_ratios: list[float] = []
    rare_ratios: list[float] = []
    singleton_ratios: list[float] = []
    for column in columns:
        counts = df[column].dropna().astype("string").value_counts(dropna=True)
        cardinalities.append(float(max(1, counts.shape[0])))
        entropies.append(_normalized_entropy(counts.to_numpy(dtype=float)))
        if counts.empty:
            mode_ratios.append(0.0)
            rare_ratios.append(0.0)
            singleton_ratios.append(0.0)
        else:
            total = float(counts.sum())
            mode_ratios.append(float(counts.max() / max(1.0, total)))
            rare_ratios.append(float((counts <= 5).mean()))
            singleton_ratios.append(float((counts == 1).mean()))

    pair_features = np.zeros((n_variables, n_variables, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    nmi = np.zeros((n_variables, n_variables), dtype=np.float32)
    cramers = np.zeros((n_variables, n_variables), dtype=np.float32)
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            assoc = _pair_association(df.iloc[:, i], df.iloc[:, j])
            nmi_value = assoc["nmi"]
            cramers_value = assoc["cramers_v"]
            chi2_strength = 1.0 - assoc["chi2_pvalue"]
            proxy_edge = float(nmi_value >= association_threshold)
            values = np.asarray(
                [
                    nmi_value,
                    cramers_value,
                    chi2_strength,
                    abs(entropies[i] - entropies[j]),
                    math.log1p(cardinalities[i] * cardinalities[j]) / math.log1p(64.0),
                    float(cardinalities[i] == cardinalities[j]),
                    abs(mode_ratios[i] - mode_ratios[j]),
                    proxy_edge,
                ],
                dtype=np.float32,
            )
            pair_features[i, j, :] = values
            pair_features[j, i, :] = values
            nmi[i, j] = nmi[j, i] = nmi_value
            cramers[i, j] = cramers[j, i] = cramers_value

    proxy_degrees = (nmi >= association_threshold).sum(axis=1) - 1
    variable_features: list[list[float]] = []
    for i in range(n_variables):
        others = [j for j in range(n_variables) if j != i]
        variable_features.append(
            [
                math.log1p(float(n_samples)) / math.log1p(10000.0),
                math.log1p(float(n_variables)) / math.log1p(100.0),
                math.log1p(cardinalities[i]) / math.log1p(16.0),
                entropies[i],
                mode_ratios[i],
                rare_ratios[i],
                singleton_ratios[i],
                float(nmi[i, others].mean()) if others else 0.0,
                float(nmi[i, others].max()) if others else 0.0,
                float(cramers[i, others].mean()) if others else 0.0,
                float(cramers[i, others].max()) if others else 0.0,
                float(proxy_degrees[i] / max(1, n_variables - 1)),
            ]
        )

    return (
        np.asarray(variable_features, dtype=np.float32),
        pair_features.astype(np.float32),
    )


def _normalized_entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    probs = counts / float(counts.sum())
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(max(0.0, min(1.0, entropy / math.log(float(counts.size)))))


def _pair_association(left: pd.Series, right: pd.Series) -> dict[str, float]:
    pair = pd.concat([left.astype("string"), right.astype("string")], axis=1).dropna()
    if pair.shape[0] < 2 or pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return {"nmi": 0.0, "cramers_v": 0.0, "chi2_pvalue": 1.0}
    nmi = float(normalized_mutual_info_score(pair.iloc[:, 0], pair.iloc[:, 1]))
    table = pd.crosstab(pair.iloc[:, 0], pair.iloc[:, 1])
    try:
        chi2, pvalue, _, _ = chi2_contingency(table, correction=False)
    except ValueError:
        return {"nmi": _clip01(nmi), "cramers_v": 0.0, "chi2_pvalue": 1.0}
    denom = float(table.to_numpy().sum()) * max(1, min(table.shape[0] - 1, table.shape[1] - 1))
    cramers_v = math.sqrt(0.0 if denom <= 0 else float(chi2) / denom)
    return {
        "nmi": _clip01(nmi),
        "cramers_v": _clip01(cramers_v),
        "chi2_pvalue": _clip01(float(pvalue)),
    }


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))
