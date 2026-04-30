from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import LearnedConfig
from .featurize import PAIR_FEATURE_NAMES, VARIABLE_FEATURE_NAMES
from .synthetic import SyntheticExample


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Phase 2 learned fingerprints require torch. "
            "Run with `uv run --with torch ...` or install the learned extra."
        ) from exc
    return torch, nn


def resolve_device(device: str):
    torch, _ = require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model(
    *,
    variable_dim: int,
    pair_dim: int,
    hidden_dim: int,
    embedding_dim: int,
):
    torch, nn = require_torch()

    class BiaffineEdgeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.variable_encoder = nn.Sequential(
                nn.Linear(variable_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.cause_projection = nn.Linear(hidden_dim, embedding_dim)
            self.effect_projection = nn.Linear(hidden_dim, embedding_dim)
            self.bilinear = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
            self.pair_bias = nn.Sequential(
                nn.Linear(pair_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.xavier_uniform_(self.bilinear)

        def forward(self, variable_features, pair_features, *, return_embeddings: bool = False):
            hidden = self.variable_encoder(variable_features)
            cause = self.cause_projection(hidden)
            effect = self.effect_projection(hidden)
            bilinear_scores = torch.einsum("bih,hk,bjk->bij", cause, self.bilinear, effect)
            pair_scores = self.pair_bias(pair_features).squeeze(-1)
            scores = bilinear_scores + pair_scores
            if return_embeddings:
                return scores, hidden
            return scores

    return BiaffineEdgeModel()


def train_biaffine_encoder(
    examples: list[SyntheticExample],
    config: LearnedConfig,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("No synthetic examples were generated.")
    torch, nn = require_torch()
    device = resolve_device(config.device)
    rng = np.random.default_rng(config.random_seed)
    indices = rng.permutation(len(examples)).tolist()
    val_count = max(1, int(round(len(indices) * config.validation_fraction)))
    val_indices = set(indices[:val_count])
    train_examples = [example for idx, example in enumerate(examples) if idx not in val_indices]
    val_examples = [example for idx, example in enumerate(examples) if idx in val_indices]
    if not train_examples:
        train_examples, val_examples = examples, examples

    model = build_model(
        variable_dim=len(VARIABLE_FEATURE_NAMES),
        pair_dim=len(PAIR_FEATURE_NAMES),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        shuffled = rng.permutation(len(train_examples)).tolist()
        for start in range(0, len(shuffled), config.batch_size):
            batch = [train_examples[index] for index in shuffled[start : start + config.batch_size]]
            variable_features, pair_features, targets, mask = collate_examples(batch, device=device)
            logits = model(variable_features, pair_features)
            loss = masked_bce_loss(logits, targets, mask, torch=torch, nn=nn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate_biaffine_model(model, val_examples, device=device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                **val_metrics,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "variable_feature_names": VARIABLE_FEATURE_NAMES,
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "history": history,
    }
    torch.save(payload, output_path)
    return {
        "model_path": str(output_path),
        "device": str(device),
        "history": history,
        "final_metrics": history[-1] if history else {},
    }


def load_biaffine_encoder(path: str | Path, *, device: str = "auto"):
    torch, _ = require_torch()
    resolved = resolve_device(device)
    payload = torch.load(path, map_location=resolved, weights_only=False)
    config_payload = payload["config"]
    model = build_model(
        variable_dim=len(payload["variable_feature_names"]),
        pair_dim=len(payload["pair_feature_names"]),
        hidden_dim=int(config_payload["hidden_dim"]),
        embedding_dim=int(config_payload["embedding_dim"]),
    ).to(resolved)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, resolved


def collate_examples(examples: list[SyntheticExample], *, device):
    torch, _ = require_torch()
    max_nodes = max(example.variable_features.shape[0] for example in examples)
    variable_dim = examples[0].variable_features.shape[1]
    pair_dim = examples[0].pair_features.shape[2]
    batch_size = len(examples)
    variable_features = np.zeros((batch_size, max_nodes, variable_dim), dtype=np.float32)
    pair_features = np.zeros((batch_size, max_nodes, max_nodes, pair_dim), dtype=np.float32)
    targets = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros((batch_size, max_nodes, max_nodes), dtype=bool)
    for batch_idx, example in enumerate(examples):
        n_nodes = example.variable_features.shape[0]
        variable_features[batch_idx, :n_nodes, :] = example.variable_features
        pair_features[batch_idx, :n_nodes, :n_nodes, :] = example.pair_features
        targets[batch_idx, :n_nodes, :n_nodes] = example.adjacency
        mask[batch_idx, :n_nodes, :n_nodes] = True
        np.fill_diagonal(mask[batch_idx, :n_nodes, :n_nodes], False)
    return (
        torch.as_tensor(variable_features, device=device),
        torch.as_tensor(pair_features, device=device),
        torch.as_tensor(targets, device=device),
        torch.as_tensor(mask, device=device),
    )


def masked_bce_loss(logits, targets, mask, *, torch, nn):
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    positives = selected_targets.sum()
    negatives = selected_targets.numel() - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), min=1.0, max=50.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(selected_logits, selected_targets)


def evaluate_biaffine_model(model, examples: list[SyntheticExample], *, device) -> dict[str, float]:
    if not examples:
        return {"val_loss": float("nan"), "val_auc": float("nan"), "val_ap": float("nan")}
    torch, nn = require_torch()
    model.eval()
    losses: list[float] = []
    all_scores: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(examples), 32):
            batch = examples[start : start + 32]
            variable_features, pair_features, targets, mask = collate_examples(batch, device=device)
            logits = model(variable_features, pair_features)
            loss = masked_bce_loss(logits, targets, mask, torch=torch, nn=nn)
            losses.append(float(loss.detach().cpu().item()))
            all_scores.append(torch.sigmoid(logits[mask]).detach().cpu().numpy())
            all_targets.append(targets[mask].detach().cpu().numpy())
    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_targets)
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_auc": _safe_auc(y_true, y_score),
        "val_ap": _safe_ap(y_true, y_score),
    }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))
