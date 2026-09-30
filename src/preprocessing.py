"""Pure functions implementing the archive -> district -> pure-data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import DistrictDataset, validate_dataset


def merge_monthly_flow_files(
    month_paths: list[str | Path], max_missing_rate: float = 0.30
) -> tuple[np.ndarray, np.ndarray]:
    """Merge raw monthly arrays and return valid flow plus its original-node mask.

    Raw PEMS arrays are expected to be ``[node, time, feature]``. Only feature 0 is
    retained, matching the original experiment. The returned flow is ``[time,node,1]``.
    """
    if not month_paths:
        raise ValueError("at least one monthly file is required")
    if not 0 <= max_missing_rate <= 1:
        raise ValueError("max_missing_rate must be in [0, 1]")

    months: list[np.ndarray] = []
    nodes: int | None = None
    for raw_path in month_paths:
        array = np.load(raw_path)
        if array.ndim != 3 or array.shape[2] < 1:
            raise ValueError(f"{raw_path} must have shape [node, time, feature], got {array.shape}")
        if nodes is None:
            nodes = array.shape[0]
        elif array.shape[0] != nodes:
            raise ValueError("all monthly arrays must have the same node count")
        months.append(array[:, :, :1])

    flow = np.concatenate(months, axis=1).transpose(1, 0, 2)
    valid_mask = np.isnan(flow[:, :, 0]).mean(axis=0) < max_missing_rate
    return flow[:, valid_mask, :], valid_mask


def align_metadata(
    node_order: np.ndarray,
    metadata: pd.DataFrame,
    valid_mask: np.ndarray,
    station_id: str = "station_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order metadata by node order and apply the flow-derived validity mask."""
    if station_id not in metadata:
        raise KeyError(f"metadata has no {station_id!r} column")
    if len(node_order) != len(valid_mask):
        raise ValueError("node_order and valid_mask must describe the same number of nodes")
    if metadata[station_id].duplicated().any():
        raise ValueError(f"{station_id!r} values must be unique")

    indexed = metadata.set_index(station_id)
    missing = pd.Index(node_order).difference(indexed.index)
    if not missing.empty:
        raise KeyError(f"{len(missing)} node IDs are absent from metadata; e.g. {missing[0]!r}")
    sorted_metadata = indexed.loc[node_order].reset_index()
    return sorted_metadata, sorted_metadata.loc[valid_mask].reset_index(drop=True)


def split_districts(
    flow: np.ndarray,
    metadata: pd.DataFrame,
    adjacency: np.ndarray,
    districts: list[int],
    district_column: str = "District",
    valid_mask: np.ndarray | None = None,
    adjacency_kind: str = "weights",
) -> dict[int, DistrictDataset]:
    """Create aligned datasets for each requested district.

    ``adjacency`` may be global; provide ``valid_mask`` to apply the same original-node
    filtering that produced ``flow`` and ``metadata``.
    """
    if district_column not in metadata:
        raise KeyError(f"metadata has no {district_column!r} column")
    if adjacency_kind not in {"weights", "distance"}:
        raise ValueError("adjacency_kind must be 'weights' or 'distance'")
    if valid_mask is not None:
        if adjacency.shape != (len(valid_mask), len(valid_mask)):
            raise ValueError("global adjacency shape does not match valid_mask")
        original_node_indices = np.flatnonzero(valid_mask)
    else:
        if adjacency.shape != (flow.shape[1], flow.shape[1]):
            raise ValueError("adjacency shape does not match the valid node set")
        original_node_indices = np.arange(flow.shape[1])
    if flow.ndim != 3 or flow.shape[1] != len(metadata):
        raise ValueError("flow and metadata node axes are not aligned")

    result: dict[int, DistrictDataset] = {}
    district_values = pd.to_numeric(metadata[district_column], errors="raise").to_numpy()
    for district in districts:
        mask = district_values == district
        if not mask.any():
            raise ValueError(f"district {district} has no nodes")
        district_source_indices = original_node_indices[mask]
        district_adjacency = adjacency[
            np.ix_(district_source_indices, district_source_indices)
        ]
        if adjacency_kind == "distance":
            district_adjacency = distance_to_adjacency(district_adjacency)
        selected = DistrictDataset(
            flow=flow[:, mask, :],
            metadata=metadata.loc[mask].reset_index(drop=True),
            adjacency=district_adjacency,
        )
        validate_dataset(selected)
        result[district] = selected
    return result


def distance_to_adjacency(distance: np.ndarray) -> np.ndarray:
    """Convert directed distances to weights where a shorter edge is stronger.

    The raw matrix uses ``-1`` for missing edges and ``0`` on the diagonal. Reciprocal
    distance preserves its directed topology and makes Top-K select nearest road links.
    """
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance matrix must be square")
    weights = np.zeros(distance.shape, dtype=np.float32)
    valid = np.isfinite(distance) & (distance > 0)
    np.divide(1.0, distance, out=weights, where=valid, casting="unsafe")
    np.fill_diagonal(weights, 0)
    return weights


@dataclass(frozen=True)
class CoreSelectionConfig:
    core_radius_km: float = 3.0
    min_local_neighbors: int = 10
    min_2d_score: float = 0.06
    min_unique_freeways: int = 2
    expand_radius_km: float = 5.0
    allow_upstream_boundary: bool = True


def _validate_selection_columns(metadata: pd.DataFrame) -> None:
    missing = {"Lat", "Lng", "Fwy"}.difference(metadata.columns)
    if missing:
        raise KeyError("core selection requires metadata columns: " + ", ".join(sorted(missing)))


def select_spatial_core(
    dataset: DistrictDataset, config: CoreSelectionConfig | None = None
) -> tuple[DistrictDataset, np.ndarray]:
    """Select a spatially dense, weakly connected directed-road core.

    This is the parameterized counterpart of the legacy ``sensor_selection.py``. It
    deliberately does not write maps; visualization belongs in a separate experiment.
    """
    config = config or CoreSelectionConfig()
    validate_dataset(dataset)
    _validate_selection_columns(dataset.metadata)
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from sklearn.neighbors import BallTree
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError("install scipy and scikit-learn to select a pure spatial core") from error

    metadata = dataset.metadata
    lat = pd.to_numeric(metadata["Lat"], errors="coerce")
    lng = pd.to_numeric(metadata["Lng"], errors="coerce")
    valid_geo = (
        lat.notna()
        & lng.notna()
        & np.isfinite(lat)
        & np.isfinite(lng)
        & lat.between(-90, 90)
        & lng.between(-180, 180)
    ).to_numpy()
    valid_indices = np.flatnonzero(valid_geo)
    if not len(valid_indices):
        raise ValueError("no valid latitude/longitude pairs")

    earth_radius_km = 6371.0088
    coordinates = np.column_stack((lat.iloc[valid_indices], lng.iloc[valid_indices]))
    coordinates_rad = np.radians(coordinates)
    tree = BallTree(coordinates_rad, metric="haversine")
    local = tree.query_radius(coordinates_rad, r=config.core_radius_km / earth_radius_km)

    lat_rad, lng_rad = np.radians(coordinates[:, 0]), np.radians(coordinates[:, 1])
    mean_lat, mean_lng = lat_rad.mean(), lng_rad.mean()
    planar = np.column_stack(
        (
            earth_radius_km * (lng_rad - mean_lng) * np.cos(mean_lat),
            earth_radius_km * (lat_rad - mean_lat),
        )
    )
    neighbor_count = np.zeros(dataset.num_nodes, dtype=np.int32)
    score_2d = np.zeros(dataset.num_nodes, dtype=np.float64)
    freeway_count = np.zeros(dataset.num_nodes, dtype=np.int32)
    for valid_position, local_positions in enumerate(local):
        node = valid_indices[valid_position]
        neighbor_count[node] = max(len(local_positions) - 1, 0)
        if len(local_positions) >= 3:
            eigenvalues = np.linalg.eigvalsh(np.cov(planar[local_positions], rowvar=False))
            if eigenvalues[-1] > 1e-12:
                score_2d[node] = float(eigenvalues[0] / eigenvalues[-1])
        local_nodes = valid_indices[local_positions]
        freeway_count[node] = metadata.iloc[local_nodes]["Fwy"].dropna().astype(str).nunique()

    seed_mask = (
        valid_geo
        & (neighbor_count >= config.min_local_neighbors)
        & (score_2d >= config.min_2d_score)
        & (freeway_count >= config.min_unique_freeways)
    )
    seed_positions = np.flatnonzero(seed_mask)
    if not len(seed_positions):
        raise ValueError("no core seeds; relax core-selection thresholds")
    global_to_valid = np.full(dataset.num_nodes, -1, dtype=int)
    global_to_valid[valid_indices] = np.arange(len(valid_indices))
    seed_tree = BallTree(coordinates_rad[global_to_valid[seed_positions]], metric="haversine")
    distance, _ = seed_tree.query(coordinates_rad, k=1)
    expanded_mask = np.zeros(dataset.num_nodes, dtype=bool)
    expanded_mask[valid_indices[distance[:, 0] * earth_radius_km <= config.expand_radius_km]] = True

    directed = np.isfinite(dataset.adjacency) & (dataset.adjacency != 0)
    np.fill_diagonal(directed, False)
    weak = directed | directed.T

    def largest_component(mask: np.ndarray) -> np.ndarray:
        selected_indices = np.flatnonzero(mask)
        result = np.zeros(len(mask), dtype=bool)
        if not len(selected_indices):
            return result
        _, labels = connected_components(
            csr_matrix(weak[np.ix_(selected_indices, selected_indices)]), directed=False
        )
        largest_label = np.bincount(labels).argmax()
        result[selected_indices[labels == largest_label]] = True
        return result

    selected = largest_component(expanded_mask)
    outside = ~expanded_mask
    out_to_outside = directed[:, outside].any(axis=1)
    in_from_outside = directed[outside, :].any(axis=0)
    for _ in range(20):
        previous = selected.copy()
        while selected.any():
            selected_indices = np.flatnonzero(selected)
            subgraph = directed[np.ix_(selected_indices, selected_indices)]
            in_degree, out_degree = subgraph.sum(axis=0), subgraph.sum(axis=1)
            valid = (in_degree > 0) & (out_degree > 0)
            valid |= (
                (in_degree > 0)
                & (out_degree == 0)
                & out_to_outside[selected_indices]
            )
            if config.allow_upstream_boundary:
                valid |= (
                    (in_degree == 0)
                    & (out_degree > 0)
                    & in_from_outside[selected_indices]
                )
            if valid.all():
                break
            selected[selected_indices[~valid]] = False
        selected = largest_component(selected)
        if np.array_equal(previous, selected):
            break
    if not selected.any():
        raise ValueError("selection became empty; relax graph-boundary constraints")

    indices = np.flatnonzero(selected)
    pure = DistrictDataset(
        flow=dataset.flow[:, indices, :],
        metadata=metadata.iloc[indices].reset_index(drop=True),
        adjacency=dataset.adjacency[np.ix_(indices, indices)],
    )
    validate_dataset(pure)
    return pure, indices
