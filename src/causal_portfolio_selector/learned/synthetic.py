from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd

from ..config import LearnedConfig
from .featurize import dataframe_to_learned_inputs


@dataclass(frozen=True)
class SyntheticExample:
    variable_features: np.ndarray
    pair_features: np.ndarray
    adjacency: np.ndarray
    dataset_name: str
    graph_kind: str
    n_samples: int


def generate_synthetic_examples(config: LearnedConfig) -> list[SyntheticExample]:
    worker_count = max(1, int(config.synthetic_workers))
    if worker_count == 1 or config.synthetic_graph_count < 8:
        return [_generate_one(index, config) for index in range(config.synthetic_graph_count)]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_generate_one_from_args, [(index, config) for index in range(config.synthetic_graph_count)]))


def _generate_one_from_args(args: tuple[int, LearnedConfig]) -> SyntheticExample:
    index, config = args
    return _generate_one(index, config)


def _generate_one(index: int, config: LearnedConfig) -> SyntheticExample:
    rng = np.random.default_rng(config.random_seed + index * 9973)
    n_vars = int(rng.choice(config.n_vars_choices))
    sample_size = int(rng.choice(config.sample_sizes))
    graph_kind = "scale_free" if index % 3 == 0 else "erdos_renyi"
    adjacency = sample_dag(n_vars, graph_kind=graph_kind, config=config, rng=rng)
    cardinalities = rng.integers(
        config.cardinality_min,
        config.cardinality_max + 1,
        size=n_vars,
        endpoint=False,
    ).astype(int)
    df = sample_discrete_bn(adjacency, cardinalities, sample_size, rng=rng)
    variable_features, pair_features = dataframe_to_learned_inputs(
        df,
        max_rows=config.max_feature_rows,
        random_seed=config.random_seed + index,
    )
    return SyntheticExample(
        variable_features=variable_features,
        pair_features=pair_features,
        adjacency=adjacency.astype(np.float32),
        dataset_name=f"synthetic_{index:05d}",
        graph_kind=graph_kind,
        n_samples=sample_size,
    )


def sample_dag(
    n_vars: int,
    *,
    graph_kind: str,
    config: LearnedConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    order = rng.permutation(n_vars)
    adjacency = np.zeros((n_vars, n_vars), dtype=np.int8)
    if graph_kind == "scale_free":
        degrees = np.ones(n_vars, dtype=float)
        for position in range(1, n_vars):
            child = int(order[position])
            candidates = [int(node) for node in order[:position]]
            max_parents = min(config.max_indegree, len(candidates))
            if max_parents <= 0:
                continue
            parent_count = int(rng.integers(0, max_parents + 1))
            if parent_count == 0:
                continue
            weights = np.asarray([degrees[node] for node in candidates], dtype=float)
            weights = weights / weights.sum()
            parents = rng.choice(candidates, size=parent_count, replace=False, p=weights)
            for parent in parents:
                adjacency[int(parent), child] = 1
                degrees[int(parent)] += 1.0
                degrees[child] += 1.0
    else:
        edge_probability = float(
            rng.uniform(config.edge_probability_min, config.edge_probability_max)
        )
        for left_position in range(n_vars):
            parent = int(order[left_position])
            for right_position in range(left_position + 1, n_vars):
                child = int(order[right_position])
                if rng.random() <= edge_probability:
                    adjacency[parent, child] = 1

    for child in range(n_vars):
        parents = np.flatnonzero(adjacency[:, child])
        if parents.size > config.max_indegree:
            keep = set(rng.choice(parents, size=config.max_indegree, replace=False).tolist())
            for parent in parents:
                if int(parent) not in keep:
                    adjacency[int(parent), child] = 0
    return adjacency


def sample_discrete_bn(
    adjacency: np.ndarray,
    cardinalities: np.ndarray,
    n_samples: int,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    graph = nx.DiGraph(adjacency)
    order = list(nx.topological_sort(graph))
    cpts = _sample_cpts(adjacency, cardinalities, rng)
    data = np.zeros((n_samples, adjacency.shape[0]), dtype=np.int16)
    for node in order:
        parents = np.flatnonzero(adjacency[:, node])
        cpt = cpts[node]
        if parents.size == 0:
            probs = cpt[0]
            data[:, node] = _sample_categorical(probs, n_samples, rng)
            continue
        parent_indices = _parent_state_indices(data[:, parents], cardinalities[parents])
        for state_index in np.unique(parent_indices):
            rows = np.flatnonzero(parent_indices == state_index)
            probs = cpt[int(state_index)]
            data[rows, node] = _sample_categorical(probs, rows.size, rng)
    return pd.DataFrame(data, columns=[f"X{i}" for i in range(adjacency.shape[0])])


def _sample_cpts(
    adjacency: np.ndarray,
    cardinalities: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    cpts: list[np.ndarray] = []
    for node in range(adjacency.shape[0]):
        parents = np.flatnonzero(adjacency[:, node])
        parent_states = int(np.prod(cardinalities[parents])) if parents.size else 1
        alpha = float(rng.uniform(0.5, 5.0))
        cpt = rng.dirichlet(
            np.full(int(cardinalities[node]), alpha, dtype=float),
            size=parent_states,
        )
        cpts.append(cpt.astype(np.float32))
    return cpts


def _parent_state_indices(parent_values: np.ndarray, parent_cardinalities: np.ndarray) -> np.ndarray:
    multipliers = np.ones(parent_values.shape[1], dtype=np.int64)
    for idx in range(1, parent_values.shape[1]):
        multipliers[idx] = multipliers[idx - 1] * int(parent_cardinalities[idx - 1])
    return (parent_values.astype(np.int64) * multipliers).sum(axis=1)


def _sample_categorical(
    probabilities: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cumulative = np.cumsum(probabilities)
    draws = rng.random(size)
    return np.searchsorted(cumulative, draws, side="right").astype(np.int16)
