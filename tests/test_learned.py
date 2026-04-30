from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd

from causal_portfolio_selector.config import LearnedConfig
from causal_portfolio_selector.learned.featurize import (
    PAIR_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    dataframe_to_learned_inputs,
)
from causal_portfolio_selector.learned.synthetic import sample_dag


def test_synthetic_dag_is_acyclic_and_respects_indegree() -> None:
    config = LearnedConfig(max_indegree=2)
    rng = np.random.default_rng(7)
    adjacency = sample_dag(12, graph_kind="erdos_renyi", config=config, rng=rng)
    assert np.all(np.diag(adjacency) == 0)
    assert int(adjacency.sum(axis=0).max()) <= 2
    assert nx.is_directed_acyclic_graph(nx.DiGraph(adjacency))


def test_learned_input_shapes() -> None:
    df = pd.DataFrame(
        {
            "A": [0, 0, 1, 1, 0, 1],
            "B": [0, 1, 0, 1, 0, 1],
            "C": [1, 1, 0, 0, 1, 0],
        }
    )
    variable_features, pair_features = dataframe_to_learned_inputs(df)
    assert variable_features.shape == (3, len(VARIABLE_FEATURE_NAMES))
    assert pair_features.shape == (3, 3, len(PAIR_FEATURE_NAMES))
    assert np.isfinite(variable_features).all()
    assert np.isfinite(pair_features).all()
