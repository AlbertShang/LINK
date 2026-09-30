"""Metadata-aware dataset kept separate from STAGANN's legacy dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .datasets import STKrigingDataset
from ..utils import load_pkl


class MetadataSTKrigingDataset(STKrigingDataset):
    """Append aligned static node metadata immediately after traffic flow.

    The on-disk ``data.pkl`` remains compatible with the original baselines.
    Metadata is broadcast over the temporal axis only when a sample is read,
    producing ``[flow, metadata..., TOD, DOW]`` without storing a large repeated
    tensor on disk.  ``STKrigingDataset`` already selects the node subset for
    train/validation/test, so metadata follows exactly the same node ordering.
    Controlled no-metadata runs set ``zero_metadata=True``: this preserves the
    identical input shape and model parameterization while removing all static
    metadata information.
    """

    def __init__(
        self,
        data_file_path: str,
        index_file_path: str,
        node_index_file_path: str,
        mode: str,
        metadata_file_path: str,
        metadata_after_flow: bool = True,
        metadata_layout_version: str = "canonical-node-metadata-v2",
        zero_metadata: bool = False,
    ) -> None:
        super().__init__(
            data_file_path=data_file_path,
            index_file_path=index_file_path,
            node_index_file_path=node_index_file_path,
            mode=mode,
        )
        metadata_path = Path(metadata_file_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

        payload = load_pkl(str(metadata_path))
        if not isinstance(payload, dict) or "node_features" not in payload:
            raise ValueError(
                f"{metadata_path} must contain a 'node_features' matrix"
            )
        actual_layout = payload.get("config", {}).get("layout_version")
        if actual_layout != metadata_layout_version:
            raise ValueError(
                "Metadata layout mismatch: "
                f"{actual_layout!r} != {metadata_layout_version!r}. "
                "Regenerate metadata.pkl from district_*_node_metadata.npz."
            )
        node_features = np.asarray(payload["node_features"], dtype=np.float32)
        if node_features.ndim != 2:
            raise ValueError(
                "node_features must have shape [node, feature], got "
                f"{node_features.shape}"
            )
        split_nodes = np.concatenate([
            np.asarray(self.nodes[key], dtype=np.int64)
            for key in ("train_nodes", "valid_nodes", "test_nodes")
        ])
        total_nodes = len(split_nodes)
        if node_features.shape[0] != total_nodes:
            raise ValueError(
                "Metadata and node split are not aligned: "
                f"{node_features.shape[0]} != {total_nodes}"
            )
        if (
            np.unique(split_nodes).size != total_nodes
            or split_nodes.min() < 0
            or split_nodes.max() >= total_nodes
        ):
            raise ValueError(
                "train/valid/test nodes must be a non-overlapping partition of metadata rows"
            )
        if not np.isfinite(node_features).all():
            raise ValueError("node_features contains NaN or Inf")

        selected_metadata = torch.from_numpy(node_features[self.nodes_index]).float()
        self.zero_metadata = bool(zero_metadata)
        self.metadata = (
            torch.zeros_like(selected_metadata)
            if self.zero_metadata
            else selected_metadata
        )
        self.metadata_dim = int(self.metadata.shape[1])
        self.metadata_after_flow = bool(metadata_after_flow)

    def __getitem__(self, index: int) -> torch.Tensor:
        flow_data = super().__getitem__(index)
        if flow_data.ndim != 3:
            raise ValueError(
                "Expected a window with shape [time, node, feature], got "
                f"{tuple(flow_data.shape)}"
            )
        if flow_data.shape[1] != self.metadata.shape[0]:
            raise ValueError(
                "Flow and metadata node axes differ: "
                f"{flow_data.shape[1]} != {self.metadata.shape[0]}"
            )
        metadata = self.metadata.unsqueeze(0).expand(flow_data.shape[0], -1, -1)
        if self.metadata_after_flow:
            return torch.cat((flow_data[..., :1], metadata, flow_data[..., 1:]), dim=-1)
        return torch.cat((flow_data, metadata), dim=-1)
