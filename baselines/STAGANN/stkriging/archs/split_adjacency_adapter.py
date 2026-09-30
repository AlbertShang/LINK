"""Compatibility adapter for baselines that require two adjacency matrices."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn


class SplitAdjacencyAdapter(nn.Module):
    """Call an unchanged backbone with its regular and masked adjacency.

    SATCN expects ``adj`` and ``adj_mask`` as separate arguments, whereas the
    shared kriging runner exposes a single adjacency argument.  This adapter
    only bridges that calling convention; it does not read, encode, or fuse
    node metadata.
    """

    def __init__(
        self,
        base_architecture: type[nn.Module],
        base_parameters: dict[str, Any],
    ) -> None:
        super().__init__()
        self.base_model = base_architecture(**deepcopy(base_parameters))

    @staticmethod
    def _mask_unknown_columns(
        adjacency: torch.Tensor, unknown_nodes: Any, batch_size: int
    ) -> torch.Tensor:
        adjacency = torch.as_tensor(adjacency)
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1).clone()
        elif adjacency.ndim == 3:
            if adjacency.shape[0] == 1 and batch_size != 1:
                adjacency = adjacency.expand(batch_size, -1, -1).clone()
            elif adjacency.shape[0] != batch_size:
                raise ValueError(
                    f"Adjacency batch {adjacency.shape[0]} != input batch {batch_size}"
                )
            else:
                adjacency = adjacency.clone()
        else:
            raise ValueError(
                f"Adjacency must have shape [N,N] or [B,N,N], got {adjacency.shape}"
            )

        indices = torch.as_tensor(
            unknown_nodes, dtype=torch.long, device=adjacency.device
        )
        indices = indices.reshape(indices.shape[0], -1) if indices.ndim > 1 else indices[None]
        if indices.shape[0] == 1 and batch_size != 1:
            indices = indices.expand(batch_size, -1)
        elif indices.shape[0] != batch_size:
            raise ValueError(
                f"Unknown-node batch {indices.shape[0]} != input batch {batch_size}"
            )
        scatter_index = indices.unsqueeze(1).expand(-1, adjacency.shape[1], -1)
        return adjacency.scatter(2, scatter_index, 0)

    def forward(self, X: torch.Tensor, adj, unknown_nodes, **kwargs):
        if isinstance(adj, (list, tuple)):
            if len(adj) != 2:
                raise ValueError("Expected [adjacency, masked_adjacency]")
            adjacency, masked_adjacency = adj
        else:
            adjacency = adj
            masked_adjacency = self._mask_unknown_columns(
                adjacency, unknown_nodes, X.shape[0]
            )
        return self.base_model(
            X=X,
            adj=adjacency,
            adj_mask=masked_adjacency,
            unknown_nodes=unknown_nodes,
            **kwargs,
        )
