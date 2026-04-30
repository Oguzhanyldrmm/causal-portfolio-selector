from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd

from causal_portfolio_selector.synthetic_benchmark import (
    SyntheticGenerateOptions,
    generate_synthetic_bn_suite,
)


def test_generate_synthetic_bn_suite_writes_manifest_data_and_truth(tmp_path) -> None:
    output = tmp_path / "synthetic_bn"
    paths = generate_synthetic_bn_suite(
        SyntheticGenerateOptions(output=output, count=8, max_nodes=10, seed=3)
    )
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["dataset_count"] == 8
    first = manifest["datasets"][0]
    data = pd.read_csv(output / first["dataset_path"])
    truth = json.loads((output / first["ground_truth_path"]).read_text())
    assert data.shape[0] == int(first["n_samples"])
    assert data.shape[1] == int(first["n_features"])
    assert truth["truth_type"] == "synthetic_bn"
    node_to_idx = {node: idx for idx, node in enumerate(truth["nodes"])}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(node_to_idx)))
    graph.add_edges_from((node_to_idx[src], node_to_idx[dst]) for src, dst in truth["directed_edges"])
    assert nx.is_directed_acyclic_graph(graph)
    assert int(np.asarray(nx.to_numpy_array(graph)).sum(axis=0).max()) <= 4
