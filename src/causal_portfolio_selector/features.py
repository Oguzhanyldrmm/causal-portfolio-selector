from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score

from .config import FeatureConfig


FEATURE_COLUMNS: tuple[str, ...] = (
    "n_samples",
    "n_features",
    "sample_to_feature_ratio",
    "continuous_ratio",
    "categorical_ratio",
    "missing_ratio",
    "avg_variance",
    "avg_skewness",
    "avg_kurtosis",
    "avg_cardinality",
    "max_cardinality",
    "cardinality_entropy",
    "rare_category_ratio",
    "singleton_category_ratio",
    "feature_sparsity_ratio",
    "mean_nmi",
    "max_nmi",
    "std_nmi",
    "mean_cramers_v",
    "max_cramers_v",
    "std_cramers_v",
    "mean_chi2_pvalue",
    "ci_rejection_rate",
    "proxy_graph_density",
    "proxy_avg_degree",
    "proxy_degree_gini",
    "proxy_avg_clustering",
    "proxy_modularity",
    "proxy_num_components",
)


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _clip01(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _numeric_or_factorized(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(2, int(0.9 * series.dropna().shape[0])):
        return numeric
    codes, _ = pd.factorize(series.astype("string"), sort=True)
    result = pd.Series(codes.astype(float), index=series.index)
    result[result < 0] = np.nan
    return result


def _infer_variable_type(series: pd.Series) -> str:
    observed = series.dropna()
    if observed.empty:
        return "categorical"
    n_unique = observed.nunique(dropna=True)
    if n_unique <= 20:
        return "categorical"
    if pd.api.types.is_float_dtype(observed) or pd.api.types.is_integer_dtype(observed):
        return "continuous"
    return "categorical"


def _normalized_entropy(counts: Iterable[float]) -> float:
    arr = np.asarray([value for value in counts if value > 0], dtype=float)
    if arr.size <= 1:
        return 0.0
    probabilities = arr / float(arr.sum())
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return _clip01(_safe_divide(entropy, math.log(float(arr.size))))


def _gini(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0 or float(arr.sum()) <= 0:
        return 0.0
    arr.sort()
    index = np.arange(1, arr.size + 1, dtype=float)
    return float(np.sum((2 * index - arr.size - 1) * arr) / (arr.size * arr.sum()))


def _pair_association(left: pd.Series, right: pd.Series) -> dict[str, float]:
    pair = pd.concat([left.astype("string"), right.astype("string")], axis=1).dropna()
    if pair.shape[0] < 2:
        return {"nmi": 0.0, "cramers_v": 0.0, "chi2_pvalue": 1.0}
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return {"nmi": 0.0, "cramers_v": 0.0, "chi2_pvalue": 1.0}

    nmi = float(normalized_mutual_info_score(pair.iloc[:, 0], pair.iloc[:, 1]))
    contingency = pd.crosstab(pair.iloc[:, 0], pair.iloc[:, 1])
    try:
        chi2, pvalue, _, _ = chi2_contingency(contingency, correction=False)
    except ValueError:
        return {"nmi": _clip01(nmi), "cramers_v": 0.0, "chi2_pvalue": 1.0}

    n = float(contingency.to_numpy().sum())
    rows, cols = contingency.shape
    denom = n * max(1, min(rows - 1, cols - 1))
    cramers_v = math.sqrt(_safe_divide(float(chi2), denom))
    return {
        "nmi": _clip01(nmi),
        "cramers_v": _clip01(cramers_v),
        "chi2_pvalue": _clip01(float(pvalue)),
    }


def _select_pairs(columns: list[str], config: FeatureConfig) -> list[tuple[str, str]]:
    pairs = list(itertools.combinations(columns, 2))
    if len(pairs) <= config.max_pairs:
        return pairs
    rng = np.random.default_rng(config.random_seed)
    selected = sorted(rng.choice(len(pairs), size=config.max_pairs, replace=False).tolist())
    return [pairs[index] for index in selected]


def _graph_features(
    columns: list[str],
    pair_rows: list[dict[str, float | str]],
    *,
    threshold: float,
) -> dict[str, float]:
    graph = nx.Graph()
    graph.add_nodes_from(columns)
    for row in pair_rows:
        if float(row["nmi"]) >= threshold:
            graph.add_edge(str(row["left"]), str(row["right"]))

    n = len(columns)
    if n <= 1:
        return {
            "proxy_graph_density": 0.0,
            "proxy_avg_degree": 0.0,
            "proxy_degree_gini": 0.0,
            "proxy_avg_clustering": 0.0,
            "proxy_modularity": 0.0,
            "proxy_num_components": float(n),
        }

    degrees = [degree for _, degree in graph.degree()]
    if graph.number_of_edges() == 0:
        modularity = 0.0
    else:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
        modularity = float(nx.algorithms.community.modularity(graph, communities))

    return {
        "proxy_graph_density": float(nx.density(graph)),
        "proxy_avg_degree": _safe_divide(float(np.mean(degrees)), n - 1),
        "proxy_degree_gini": _gini(degrees),
        "proxy_avg_clustering": float(nx.average_clustering(graph)) if n > 1 else 0.0,
        "proxy_modularity": modularity,
        "proxy_num_components": float(nx.number_connected_components(graph)),
    }


def extract_dataset_features(
    dataset_path: str | Path,
    *,
    config: FeatureConfig | None = None,
) -> dict[str, float]:
    config = config or FeatureConfig()
    df = pd.read_csv(dataset_path)
    columns = [str(column) for column in df.columns]
    df.columns = columns
    n_samples = int(df.shape[0])
    n_features = int(df.shape[1])

    variable_types = [_infer_variable_type(df[column]) for column in columns]
    cardinalities: list[float] = []
    rare_ratios: list[float] = []
    singleton_ratios: list[float] = []
    mode_ratios: list[float] = []
    numeric_stats: list[pd.Series] = []
    for column in columns:
        observed = df[column].dropna()
        counts = observed.astype("string").value_counts(dropna=True)
        cardinalities.append(float(max(1, counts.shape[0])))
        if counts.empty:
            rare_ratios.append(0.0)
            singleton_ratios.append(0.0)
            mode_ratios.append(0.0)
        else:
            rare_ratios.append(float((counts <= 5).mean()))
            singleton_ratios.append(float((counts == 1).mean()))
            mode_ratios.append(_safe_divide(float(counts.max()), float(counts.sum())))
        numeric_stats.append(_numeric_or_factorized(df[column]))

    numeric_frame = pd.concat(numeric_stats, axis=1) if numeric_stats else pd.DataFrame()
    variances = numeric_frame.var(skipna=True).replace([np.inf, -np.inf], np.nan)
    skews = numeric_frame.skew(skipna=True).replace([np.inf, -np.inf], np.nan)
    kurtoses = numeric_frame.kurtosis(skipna=True).replace([np.inf, -np.inf], np.nan)

    pair_rows: list[dict[str, float | str]] = []
    for left, right in _select_pairs(columns, config):
        metrics = _pair_association(df[left], df[right])
        pair_rows.append({"left": left, "right": right, **metrics})

    nmi_values = np.asarray([float(row["nmi"]) for row in pair_rows], dtype=float)
    cramers_values = np.asarray([float(row["cramers_v"]) for row in pair_rows], dtype=float)
    pvalues = np.asarray([float(row["chi2_pvalue"]) for row in pair_rows], dtype=float)
    graph_metrics = _graph_features(
        columns,
        pair_rows,
        threshold=config.association_threshold,
    )

    features = {
        "n_samples": float(n_samples),
        "n_features": float(n_features),
        "sample_to_feature_ratio": _safe_divide(float(n_samples), float(n_features)),
        "continuous_ratio": _safe_divide(variable_types.count("continuous"), len(variable_types)),
        "categorical_ratio": _safe_divide(variable_types.count("categorical"), len(variable_types)),
        "missing_ratio": _safe_divide(float(df.isna().sum().sum()), float(max(1, df.size))),
        "avg_variance": float(variances.mean(skipna=True)) if not variances.empty else 0.0,
        "avg_skewness": float(skews.mean(skipna=True)) if not skews.empty else 0.0,
        "avg_kurtosis": float(kurtoses.mean(skipna=True)) if not kurtoses.empty else 0.0,
        "avg_cardinality": float(np.mean(cardinalities)) if cardinalities else 0.0,
        "max_cardinality": float(np.max(cardinalities)) if cardinalities else 0.0,
        "cardinality_entropy": _normalized_entropy(cardinalities),
        "rare_category_ratio": float(np.mean(rare_ratios)) if rare_ratios else 0.0,
        "singleton_category_ratio": float(np.mean(singleton_ratios)) if singleton_ratios else 0.0,
        "feature_sparsity_ratio": float(np.mean(mode_ratios)) if mode_ratios else 0.0,
        "mean_nmi": float(np.mean(nmi_values)) if nmi_values.size else 0.0,
        "max_nmi": float(np.max(nmi_values)) if nmi_values.size else 0.0,
        "std_nmi": float(np.std(nmi_values)) if nmi_values.size else 0.0,
        "mean_cramers_v": float(np.mean(cramers_values)) if cramers_values.size else 0.0,
        "max_cramers_v": float(np.max(cramers_values)) if cramers_values.size else 0.0,
        "std_cramers_v": float(np.std(cramers_values)) if cramers_values.size else 0.0,
        "mean_chi2_pvalue": float(np.mean(pvalues)) if pvalues.size else 1.0,
        "ci_rejection_rate": float((pvalues <= config.ci_alpha).mean()) if pvalues.size else 0.0,
        **graph_metrics,
    }
    return {name: _finite_float(features.get(name, 0.0)) for name in FEATURE_COLUMNS}


def _finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def build_feature_table(
    dataset_paths: dict[str, str | Path],
    *,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for dataset_name, dataset_path in sorted(dataset_paths.items()):
        row: dict[str, float | str] = {"dataset_name": dataset_name}
        row.update(extract_dataset_features(dataset_path, config=config))
        rows.append(row)
    return pd.DataFrame(rows, columns=["dataset_name", *FEATURE_COLUMNS])
