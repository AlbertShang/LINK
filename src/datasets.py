"""Dataset contracts and safe on-disk I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistrictDataset:
    """A PEMS district dataset whose three node axes are aligned."""

    flow: np.ndarray
    metadata: pd.DataFrame
    adjacency: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.flow.shape[1]


def district_paths(directory: Path, district: int) -> tuple[Path, Path, Path]:
    """Return conventional flow, metadata and adjacency paths for a district."""
    prefix = f"district_{district}"
    return (
        directory / f"{prefix}_flow.npy",
        directory / f"{prefix}_meta_feature.csv",
        directory / f"{prefix}_adj.npy",
    )


def validate_dataset(dataset: DistrictDataset) -> None:
    """Raise a clear error when flow, metadata and adjacency are misaligned."""
    flow, metadata, adjacency = dataset.flow, dataset.metadata, dataset.adjacency
    if flow.ndim != 3:
        raise ValueError(f"flow must have shape [time, node, feature], got {flow.shape}")
    if flow.shape[0] == 0 or flow.shape[1] == 0:
        raise ValueError("flow must contain at least one time step and node")
    nodes = flow.shape[1]
    if len(metadata) != nodes:
        raise ValueError(f"metadata has {len(metadata)} rows but flow has {nodes} nodes")
    if adjacency.shape != (nodes, nodes):
        raise ValueError(
            f"adjacency must have shape ({nodes}, {nodes}), got {adjacency.shape}"
        )


def load_district_dataset(directory: str | Path, district: int) -> DistrictDataset:
    """Load the conventional files for one district and validate alignment."""
    flow_path, metadata_path, adjacency_path = district_paths(Path(directory), district)
    missing = [str(p) for p in (flow_path, metadata_path, adjacency_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing district files: " + ", ".join(missing))

    dataset = DistrictDataset(
        flow=np.load(flow_path),
        metadata=pd.read_csv(metadata_path, sep="\t"),
        adjacency=np.load(adjacency_path),
    )
    validate_dataset(dataset)
    return dataset


def save_district_dataset(directory: str | Path, district: int, dataset: DistrictDataset) -> None:
    """Save a validated dataset using the repository's stable naming convention."""
    validate_dataset(dataset)
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_path, metadata_path, adjacency_path = district_paths(output_dir, district)
    np.save(flow_path, dataset.flow)
    dataset.metadata.to_csv(metadata_path, sep="\t", index=False)
    np.save(adjacency_path, dataset.adjacency)
