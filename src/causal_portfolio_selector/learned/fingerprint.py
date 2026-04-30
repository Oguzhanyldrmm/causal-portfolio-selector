from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import artifact_path, imported_paths, load_import_manifest
from ..config import AppConfig
from .featurize import dataframe_to_learned_inputs
from .model import load_biaffine_encoder, require_torch


STRUCTURAL_FINGERPRINT_COLUMNS: tuple[str, ...] = (
    "lf_prob_mean",
    "lf_prob_std",
    "lf_prob_max",
    "lf_prob_q25",
    "lf_prob_q50",
    "lf_prob_q75",
    "lf_density_t10",
    "lf_density_t30",
    "lf_density_t50",
    "lf_asym_mean",
    "lf_asym_std",
    "lf_asym_max",
    "lf_out_mean",
    "lf_out_std",
    "lf_out_max",
    "lf_in_mean",
    "lf_in_std",
    "lf_in_max",
)


def learned_embedding_columns(embedding_dim: int) -> tuple[str, ...]:
    names: list[str] = []
    for prefix in ("lf_embed_mean", "lf_embed_std", "lf_embed_max"):
        for idx in range(embedding_dim):
            names.append(f"{prefix}_{idx:02d}")
    return tuple(names)


def learned_feature_columns(embedding_dim: int) -> tuple[str, ...]:
    return (*STRUCTURAL_FINGERPRINT_COLUMNS, *learned_embedding_columns(embedding_dim))


def build_learned_feature_tables(
    config: AppConfig,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Path]:
    root = config.resolved_root()
    model_path = Path(model_path or artifact_path(root, "learned", "biaffine_encoder.pt"))
    model, payload, device = load_biaffine_encoder(model_path, device=config.learned.device)
    embedding_dim = int(payload["config"]["embedding_dim"])
    manifest = load_import_manifest(root)
    rows: list[dict[str, Any]] = []
    for entry in manifest["datasets"]:
        dataset_name = str(entry["dataset_name"])
        dataset_path = root / entry["dataset_path"]
        row = {"dataset_name": dataset_name}
        row.update(
            extract_fingerprint(
                dataset_path,
                model=model,
                device=device,
                embedding_dim=embedding_dim,
                max_rows=config.learned.max_feature_rows,
                random_seed=config.learned.random_seed,
            )
        )
        rows.append(row)

    learned_features = pd.DataFrame(rows, columns=["dataset_name", *learned_feature_columns(embedding_dim)])
    table_dir = artifact_path(root, "tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    learned_path = table_dir / "learned_features.csv"
    combined_path = table_dir / "features_plus_learned.csv"
    learned_features.to_csv(learned_path, index=False)

    phase1_features_path = table_dir / "features.csv"
    if not phase1_features_path.exists():
        raise FileNotFoundError("Phase 1 feature table not found. Run build-tables first.")
    phase1 = pd.read_csv(phase1_features_path)
    combined = phase1.merge(learned_features, on="dataset_name", how="inner", validate="one_to_one")
    combined.to_csv(combined_path, index=False)
    return {"learned_features": learned_path, "features_plus_learned": combined_path}


def extract_fingerprint(
    dataset_path: str | Path,
    *,
    model,
    device,
    embedding_dim: int,
    max_rows: int | None = None,
    random_seed: int = 42,
) -> dict[str, float]:
    torch, _ = require_torch()
    variable_features, pair_features = dataframe_to_learned_inputs(
        dataset_path,
        max_rows=max_rows,
        random_seed=random_seed,
    )
    with torch.no_grad():
        variable_tensor = torch.as_tensor(variable_features[None, :, :], device=device)
        pair_tensor = torch.as_tensor(pair_features[None, :, :, :], device=device)
        logits, embeddings = model(variable_tensor, pair_tensor, return_embeddings=True)
        probabilities = torch.sigmoid(logits[0]).detach().cpu().numpy()
        embedding_values = embeddings[0].detach().cpu().numpy()

    n_nodes = probabilities.shape[0]
    mask = ~np.eye(n_nodes, dtype=bool)
    directed_probs = probabilities[mask]
    asymmetry = np.abs(probabilities - probabilities.T)[mask]
    out_strength = (probabilities * mask).sum(axis=1) / max(1, n_nodes - 1)
    in_strength = (probabilities * mask).sum(axis=0) / max(1, n_nodes - 1)

    row = {
        "lf_prob_mean": _safe_stat(np.mean, directed_probs),
        "lf_prob_std": _safe_stat(np.std, directed_probs),
        "lf_prob_max": _safe_stat(np.max, directed_probs),
        "lf_prob_q25": _safe_quantile(directed_probs, 0.25),
        "lf_prob_q50": _safe_quantile(directed_probs, 0.50),
        "lf_prob_q75": _safe_quantile(directed_probs, 0.75),
        "lf_density_t10": float((directed_probs >= 0.10).mean()) if directed_probs.size else 0.0,
        "lf_density_t30": float((directed_probs >= 0.30).mean()) if directed_probs.size else 0.0,
        "lf_density_t50": float((directed_probs >= 0.50).mean()) if directed_probs.size else 0.0,
        "lf_asym_mean": _safe_stat(np.mean, asymmetry),
        "lf_asym_std": _safe_stat(np.std, asymmetry),
        "lf_asym_max": _safe_stat(np.max, asymmetry),
        "lf_out_mean": _safe_stat(np.mean, out_strength),
        "lf_out_std": _safe_stat(np.std, out_strength),
        "lf_out_max": _safe_stat(np.max, out_strength),
        "lf_in_mean": _safe_stat(np.mean, in_strength),
        "lf_in_std": _safe_stat(np.std, in_strength),
        "lf_in_max": _safe_stat(np.max, in_strength),
    }
    means = embedding_values.mean(axis=0)
    stds = embedding_values.std(axis=0)
    maxes = embedding_values.max(axis=0)
    for idx in range(embedding_dim):
        row[f"lf_embed_mean_{idx:02d}"] = float(means[idx])
        row[f"lf_embed_std_{idx:02d}"] = float(stds[idx])
        row[f"lf_embed_max_{idx:02d}"] = float(maxes[idx])
    return row


def _safe_stat(fn, values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    result = float(fn(values))
    return result if np.isfinite(result) else 0.0


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    result = float(np.quantile(values, q))
    return result if np.isfinite(result) else 0.0
