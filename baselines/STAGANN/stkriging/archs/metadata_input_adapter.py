"""Shared static-metadata input adapter for kriging baselines."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn


APPENDED_METADATA_LAYOUT_VERSION = "flow-metadata-calendar-v2"


class MetadataInputAdapter(nn.Module):
    """Fuse metadata appended immediately after each traffic-flow value.

    New metadata datasets provide ``[flow, metadata..., TOD, DOW]``.  Legacy
    baselines generally consume only channel zero, so this adapter projects the
    concatenated flow/metadata vector into a residual update of the flow channel,
    restores ``[flow, TOD, DOW]``, and calls the unchanged model.  The optional
    legacy layouts keep earlier refactor configurations usable.
    """

    def __init__(
        self,
        base_architecture: type[nn.Module],
        base_parameters: dict[str, Any],
        metadata_dim: int,
        dynamic_feature_dim: int,
        metadata_embedding_dim: int = 20,
        fusion_hidden_dim: int = 32,
        dropout: float = 0.3,
        split_adjacency: bool = False,
        metadata_after_flow: bool = True,
    ) -> None:
        super().__init__()
        if metadata_dim <= 0:
            raise ValueError("metadata_dim must be positive")
        if dynamic_feature_dim <= 0:
            raise ValueError("dynamic_feature_dim must be positive")

        self.metadata_dim = int(metadata_dim)
        self.dynamic_feature_dim = int(dynamic_feature_dim)
        self.split_adjacency = bool(split_adjacency)
        self.metadata_after_flow = bool(metadata_after_flow)
        self.base_model = base_architecture(**deepcopy(base_parameters))

        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(self.metadata_dim),
            nn.Linear(self.metadata_dim, metadata_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(metadata_embedding_dim, metadata_embedding_dim),
            nn.ReLU(),
        )
        self.input_fusion = nn.Sequential(
            nn.Linear(self.dynamic_feature_dim + metadata_embedding_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def _node_metadata(self, location: Any, x: torch.Tensor) -> torch.Tensor:
        if location is None:
            raise ValueError("MetadataInputAdapter requires DATASET.LOCATION metadata")

        # The legacy runner uses [location_grid, location_id].  The refactored
        # dataset stores the same aligned node feature tensor in both slots.
        metadata = location[-1] if isinstance(location, (list, tuple)) else location
        metadata = torch.as_tensor(metadata, dtype=x.dtype, device=x.device)
        if metadata.ndim == 2:
            metadata = metadata.unsqueeze(0)
        if metadata.ndim != 3:
            raise ValueError(
                "Node metadata must have shape [node, feature] or "
                f"[batch, node, feature], got {tuple(metadata.shape)}"
            )
        if metadata.shape[-1] != self.metadata_dim:
            raise ValueError(
                f"Expected metadata_dim={self.metadata_dim}, got {metadata.shape[-1]}"
            )
        if metadata.shape[1] != x.shape[2]:
            raise ValueError(
                "Metadata node count must match the input node axis: "
                f"{metadata.shape[1]} != {x.shape[2]}"
            )
        if metadata.shape[0] == 1 and x.shape[0] != 1:
            metadata = metadata.expand(x.shape[0], -1, -1)
        elif metadata.shape[0] != x.shape[0]:
            raise ValueError(
                "Metadata batch size must be one or match the input batch size: "
                f"{metadata.shape[0]} != {x.shape[0]}"
            )
        return metadata

    def _split_features(
        self, x: torch.Tensor, location: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected X with shape [B, L, N, C], got {tuple(x.shape)}")
        expected_appended_dim = self.dynamic_feature_dim + self.metadata_dim
        if x.shape[-1] == expected_appended_dim:
            if self.metadata_after_flow:
                metadata_end = 1 + self.metadata_dim
                metadata = x[..., 1:metadata_end]
                dynamic = torch.cat((x[..., :1], x[..., metadata_end:]), dim=-1)
            else:
                dynamic = x[..., :self.dynamic_feature_dim]
                metadata = x[..., self.dynamic_feature_dim:]
            if dynamic.shape[-1] != self.dynamic_feature_dim:
                raise ValueError(
                    "Appended metadata layout is inconsistent with the configured "
                    f"dynamic_feature_dim={self.dynamic_feature_dim}"
                )
            return dynamic, metadata
        if x.shape[-1] != self.dynamic_feature_dim:
            raise ValueError(
                "Expected either dynamic-only input for the location fallback or "
                "metadata appended after the dynamic channels: "
                f"C in {{{self.dynamic_feature_dim}, {expected_appended_dim}}}, "
                f"got {x.shape[-1]}"
            )

        metadata = self._node_metadata(location, x)
        metadata = metadata.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        return x, metadata

    def _fuse_metadata(self, x: torch.Tensor, location: Any) -> torch.Tensor:
        dynamic, metadata_by_step = self._split_features(x, location)
        metadata = metadata_by_step[:, 0]
        metadata_embedding = self.metadata_encoder(metadata)
        metadata_by_step = metadata_embedding.unsqueeze(1).expand(
            -1, dynamic.shape[1], -1, -1
        )
        augmented_input = torch.cat((dynamic, metadata_by_step), dim=-1)

        # Preserve TOD/DOW and any other auxiliary channels.  All legacy
        # baselines consume the signal in channel zero, so the residual fusion
        # makes the appended metadata observable to every wrapped model.
        fused = dynamic.clone()
        fused[..., :1] = dynamic[..., :1] + self.input_fusion(augmented_input)
        return fused

    @staticmethod
    def _mask_unknown_columns(
        adjacency: torch.Tensor, unknown_nodes: Any, batch_size: int
    ) -> torch.Tensor:
        """Build SATCN's masked adjacency without changing runner settings."""
        adjacency = torch.as_tensor(adjacency)
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1).clone()
        elif adjacency.ndim == 3:
            if adjacency.shape[0] == 1 and batch_size != 1:
                adjacency = adjacency.expand(batch_size, -1, -1).clone()
            elif adjacency.shape[0] != batch_size:
                raise ValueError(
                    "Adjacency batch size must be one or match X: "
                    f"{adjacency.shape[0]} != {batch_size}"
                )
            else:
                adjacency = adjacency.clone()
        else:
            raise ValueError(
                f"Expected adjacency with two or three dimensions, got {adjacency.ndim}"
            )

        indices = torch.as_tensor(
            unknown_nodes, dtype=torch.long, device=adjacency.device
        )
        indices = indices.reshape(indices.shape[0], -1) if indices.ndim > 1 else indices[None]
        if indices.shape[0] == 1 and batch_size != 1:
            indices = indices.expand(batch_size, -1)
        elif indices.shape[0] != batch_size:
            raise ValueError(
                "Unknown-node batch size must be one or match X: "
                f"{indices.shape[0]} != {batch_size}"
            )
        scatter_index = indices.unsqueeze(1).expand(-1, adjacency.shape[1], -1)
        return adjacency.scatter(2, scatter_index, 0)

    def forward(self, X: torch.Tensor, location=None, **kwargs):
        fused_input = self._fuse_metadata(X, location)

        if self.split_adjacency:
            adjacency = kwargs.pop("adj")
            if isinstance(adjacency, (list, tuple)):
                if len(adjacency) != 2:
                    raise ValueError(
                        "Split adjacency must contain [adjacency, masked_adjacency]"
                    )
                adjacency, masked_adjacency = adjacency
            else:
                masked_adjacency = self._mask_unknown_columns(
                    adjacency, kwargs["unknown_nodes"], fused_input.shape[0]
                )
            return self.base_model(
                X=fused_input,
                adj=adjacency,
                adj_mask=masked_adjacency,
                location=location,
                **kwargs,
            )

        return self.base_model(X=fused_input, location=location, **kwargs)
