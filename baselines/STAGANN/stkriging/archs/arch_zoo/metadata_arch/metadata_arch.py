"""Architecture-level metadata fusion for the STAGANN baseline zoo.

The original architectures are intentionally not edited.  Each class below is
a distinct metadata architecture which owns two trainable parts: a
``TrafficMetadataFusion`` input block and the corresponding legacy backbone.
Metadata is therefore learned inside the model and checkpointed with it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from ..dualstn_arch import DualSTN
from ..gcn_arch import GCN
from ..gin_arch import GIN
from ..grin_arch import GRIN
from ..iagcn_arch import IAGCN
from ..ignnk_arch import IGNNK
from ..increase_arch import INCREASE
from ..okriging_arch import Okriging
from ..satcn_arch import SATCN
from ..stagann_arch import STAGANN


METADATA_ARCHITECTURE_VERSION = "canonical-node-metadata-v2"


class TrafficMetadataFusion(nn.Module):
    """Learn a flow representation from ``[traffic flow, node metadata]``.

    Input layout is ``[flow, metadata..., TOD, DOW]``.  Only flow and metadata
    enter this block; calendar channels are restored unchanged for backbones
    such as INCREASE and STAGANN which use their original channel positions.
    The residual output layer starts at zero, so every metadata architecture
    begins from the exact legacy flow input and learns metadata contributions
    during training.
    """

    def __init__(
        self,
        metadata_dim: int,
        dynamic_feature_dim: int = 3,
        metadata_embedding_dim: int = 20,
        fusion_hidden_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if metadata_dim <= 0:
            raise ValueError("metadata_dim must be positive")
        if dynamic_feature_dim < 1:
            raise ValueError("dynamic_feature_dim must include traffic flow")
        self.metadata_dim = int(metadata_dim)
        self.dynamic_feature_dim = int(dynamic_feature_dim)
        self.expected_input_dim = self.dynamic_feature_dim + self.metadata_dim

        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(self.metadata_dim),
            nn.Linear(self.metadata_dim, metadata_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(metadata_embedding_dim, metadata_embedding_dim),
            nn.ReLU(),
        )
        self.flow_fusion = nn.Sequential(
            nn.Linear(1 + metadata_embedding_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )
        nn.init.zeros_(self.flow_fusion[-1].weight)
        nn.init.zeros_(self.flow_fusion[-1].bias)

    def split_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected X [B, L, N, C], got {tuple(x.shape)}")
        if x.shape[-1] != self.expected_input_dim:
            raise ValueError(
                "Metadata architecture expects [flow, metadata..., calendar...] "
                f"with C={self.expected_input_dim}, got C={x.shape[-1]}. "
                "Regenerate metadata.pkl from district_*_node_metadata.npz and "
                "use MetadataSTKrigingDataset."
            )
        metadata_end = 1 + self.metadata_dim
        flow = x[..., :1]
        metadata = x[..., 1:metadata_end]
        calendar = x[..., metadata_end:]
        if calendar.shape[-1] != self.dynamic_feature_dim - 1:
            raise ValueError(
                "Calendar feature count does not match dynamic_feature_dim: "
                f"{calendar.shape[-1]} != {self.dynamic_feature_dim - 1}"
            )
        return flow, metadata, calendar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flow, metadata_by_step, calendar = self.split_input(x)
        # Static metadata is repeated by the Dataset.  Encode it once per node
        # and broadcast the learned representation over the window.
        metadata_embedding = self.metadata_encoder(metadata_by_step[:, 0])
        metadata_embedding = metadata_embedding.unsqueeze(1).expand(
            -1, x.shape[1], -1, -1
        )
        flow_with_metadata = torch.cat((flow, metadata_embedding), dim=-1)
        learned_flow = flow + self.flow_fusion(flow_with_metadata)
        return torch.cat((learned_flow, calendar), dim=-1)


class MetadataBaselineArchitecture(nn.Module):
    """Base for explicit metadata variants; not used directly by configs."""

    BACKBONE: type[nn.Module]
    REQUIRES_MASKED_ADJACENCY = False

    def __init__(
        self,
        backbone_parameters: dict[str, Any],
        metadata_dim: int,
        dynamic_feature_dim: int = 3,
        metadata_embedding_dim: int = 20,
        fusion_hidden_dim: int = 32,
        metadata_dropout: float = 0.3,
        split_adjacency: bool = False,
    ) -> None:
        super().__init__()
        self.metadata_fusion = TrafficMetadataFusion(
            metadata_dim=metadata_dim,
            dynamic_feature_dim=dynamic_feature_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=metadata_dropout,
        )
        self.backbone = self.BACKBONE(**deepcopy(backbone_parameters))
        self.split_adjacency = bool(
            split_adjacency or self.REQUIRES_MASKED_ADJACENCY
        )

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
                    f"Adjacency batch {adjacency.shape[0]} != flow batch {batch_size}"
                )
            else:
                adjacency = adjacency.clone()
        else:
            raise ValueError(f"Adjacency must be [N,N] or [B,N,N], got {adjacency.shape}")

        indices = torch.as_tensor(
            unknown_nodes, dtype=torch.long, device=adjacency.device
        )
        indices = indices.reshape(indices.shape[0], -1) if indices.ndim > 1 else indices[None]
        if indices.shape[0] == 1 and batch_size != 1:
            indices = indices.expand(batch_size, -1)
        elif indices.shape[0] != batch_size:
            raise ValueError(
                f"Unknown-node batch {indices.shape[0]} != flow batch {batch_size}"
            )
        scatter_index = indices.unsqueeze(1).expand(-1, adjacency.shape[1], -1)
        return adjacency.scatter(2, scatter_index, 0)

    def forward(self, X: torch.Tensor, location=None, **kwargs):
        learned_input = self.metadata_fusion(X)
        if self.split_adjacency:
            adjacency = kwargs.pop("adj")
            if isinstance(adjacency, (list, tuple)):
                if len(adjacency) != 2:
                    raise ValueError("Expected [adjacency, masked_adjacency]")
                adjacency, masked_adjacency = adjacency
            else:
                masked_adjacency = self._mask_unknown_columns(
                    adjacency, kwargs["unknown_nodes"], learned_input.shape[0]
                )
            return self.backbone(
                X=learned_input,
                adj=adjacency,
                adj_mask=masked_adjacency,
                location=location,
                **kwargs,
            )
        return self.backbone(X=learned_input, location=location, **kwargs)


class MetadataDualSTN(MetadataBaselineArchitecture):
    BACKBONE = DualSTN


class MetadataGCN(MetadataBaselineArchitecture):
    BACKBONE = GCN


class MetadataGIN(MetadataBaselineArchitecture):
    BACKBONE = GIN


class MetadataGRIN(MetadataBaselineArchitecture):
    BACKBONE = GRIN


class MetadataIAGCN(MetadataBaselineArchitecture):
    BACKBONE = IAGCN


class MetadataIGNNK(MetadataBaselineArchitecture):
    BACKBONE = IGNNK


class MetadataINCREASE(MetadataBaselineArchitecture):
    BACKBONE = INCREASE


class MetadataOkriging(MetadataBaselineArchitecture):
    BACKBONE = Okriging


class MetadataSATCN(MetadataBaselineArchitecture):
    BACKBONE = SATCN
    REQUIRES_MASKED_ADJACENCY = True


class MetadataSTAGANN(MetadataBaselineArchitecture):
    BACKBONE = STAGANN


METADATA_ARCHITECTURES = {
    DualSTN: MetadataDualSTN,
    GCN: MetadataGCN,
    GIN: MetadataGIN,
    GRIN: MetadataGRIN,
    IAGCN: MetadataIAGCN,
    IGNNK: MetadataIGNNK,
    INCREASE: MetadataINCREASE,
    Okriging: MetadataOkriging,
    SATCN: MetadataSATCN,
    STAGANN: MetadataSTAGANN,
}
