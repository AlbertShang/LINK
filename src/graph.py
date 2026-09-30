"""Directed graph transformations and deterministic node splits."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def inout_topk_adjacency(adjacency: np.ndarray, top_k: int) -> np.ndarray:
    """Keep strongest directed edges subject to both source and destination Top-K.

    The operation is deliberately directional. An edge retained for ``i -> j`` never
    creates or alters the reverse edge ``j -> i``.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    weights = np.asarray(adjacency, dtype=np.float32).copy()
    weights[~np.isfinite(weights)] = 0
    np.fill_diagonal(weights, 0)
    selected = np.zeros_like(weights, dtype=bool)
    for source, row in enumerate(weights):
        candidates = np.flatnonzero(row > 0)
        order = np.lexsort((candidates, -row[candidates]))
        selected[source, candidates[order[:top_k]]] = True
    for destination in range(weights.shape[1]):
        sources = np.flatnonzero(selected[:, destination])
        if len(sources) > top_k:
            order = np.lexsort((sources, -weights[sources, destination]))
            selected[sources[order[top_k:]], destination] = False
    return np.where(selected, weights, 0).astype(np.float32)


@dataclass(frozen=True)
class NodeSplits:
    train_nodes: np.ndarray
    valid_nodes: np.ndarray
    test_nodes: np.ndarray
    seed: int
    train_ratio: float
    valid_ratio: float


def make_node_splits(
    num_nodes: int, seed: int = 42, train_ratio: float = 0.70, valid_ratio: float = 0.10
) -> NodeSplits:
    """Create a reproducible disjoint train/validation/test node partition."""
    if num_nodes < 3:
        raise ValueError("at least three nodes are required for train/valid/test splits")
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1 or train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio and valid_ratio must be positive and sum to less than one")
    permutation = np.random.default_rng(seed).permutation(num_nodes)
    train_end = int(num_nodes * train_ratio)
    valid_end = int(num_nodes * (train_ratio + valid_ratio))
    if train_end == 0 or valid_end == train_end or valid_end == num_nodes:
        raise ValueError("ratios produce an empty split")
    return NodeSplits(
        train_nodes=np.sort(permutation[:train_end]),
        valid_nodes=np.sort(permutation[train_end:valid_end]),
        test_nodes=np.sort(permutation[valid_end:]),
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )


def save_node_splits(
    path: str | Path, splits: NodeSplits, station_ids: np.ndarray | None = None
) -> None:
    """Save a legacy-compatible split dictionary without introducing torch dependency."""
    payload: dict[str, object] = {
        "train_nodes": splits.train_nodes,
        "valid_nodes": splits.valid_nodes,
        "test_nodes": splits.test_nodes,
        "seed": splits.seed,
        "train_ratio": splits.train_ratio,
        "valid_ratio": splits.valid_ratio,
        "test_ratio": 1 - splits.train_ratio - splits.valid_ratio,
    }
    if station_ids is not None:
        expected_nodes = sum(
            len(part) for part in (splits.train_nodes, splits.valid_nodes, splits.test_nodes)
        )
        if len(station_ids) != expected_nodes:
            raise ValueError("station_ids must align with the full node set")
        payload.update(
            train_station_ids=station_ids[splits.train_nodes],
            valid_station_ids=station_ids[splits.valid_nodes],
            test_station_ids=station_ids[splits.test_nodes],
        )
    with Path(path).open("xb") as handle:
        pickle.dump(payload, handle)
