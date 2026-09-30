"""Adaptive boundary subgraphs with metadata attached during construction."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
import torch

from .metadata import DIRECTION_CATEGORIES, NodeMetadata


def _neighbor_lists(adjacency: np.ndarray) -> tuple[list[list[tuple[int, float]]], ...]:
    nodes = adjacency.shape[0]
    incoming: list[list[tuple[int, float]]] = [[] for _ in range(nodes)]
    outgoing: list[list[tuple[int, float]]] = [[] for _ in range(nodes)]
    sources, destinations = np.where(adjacency > 0)
    for source, destination in zip(sources, destinations, strict=True):
        if source == destination:
            continue
        weight = float(adjacency[source, destination])
        outgoing[int(source)].append((int(destination), weight))
        incoming[int(destination)].append((int(source), weight))
    return incoming, outgoing


def _expand_branch(
    center: int,
    neighbors: list[list[tuple[int, float]]],
    train_nodes: set[int],
    *,
    upstream: bool,
) -> tuple[dict[int, int], set[int], list[tuple[int, int, float, int]]]:
    """Search unknown nodes until train-node boundaries and return propagation edges."""
    hops = {center: 0}
    boundaries: set[int] = set()
    traversed: list[tuple[int, int, float]] = []
    queue: deque[int] = deque([center])
    while queue:
        current = queue.popleft()
        for neighbor, weight in neighbors[current]:
            original_source, original_destination = (
                (neighbor, current) if upstream else (current, neighbor)
            )
            traversed.append((original_source, original_destination, weight))
            if neighbor in hops:
                continue
            hops[neighbor] = hops[current] + 1
            if neighbor in train_nodes and neighbor != center:
                boundaries.add(neighbor)
            else:
                queue.append(neighbor)

    propagation: list[tuple[int, int, float, int]] = []
    for original_source, original_destination, weight in traversed:
        if upstream:
            source, destination = original_source, original_destination
            outer, inner = original_source, original_destination
        else:
            source, destination = original_destination, original_source
            outer, inner = original_destination, original_source
        if outer in hops and inner in hops and hops[outer] == hops[inner] + 1:
            propagation.append((source, destination, weight, hops[outer]))
    return hops, boundaries, propagation


def _edge_metadata(
    node_ids: np.ndarray,
    edge_index: np.ndarray,
    metadata: NodeMetadata,
) -> np.ndarray:
    """Build propagation-edge features from already prepared node metadata."""
    edge_count = edge_index.shape[1]
    feature_dim = (
        metadata.type_feature.shape[1] * 2
        + metadata.direction_feature.shape[1] * 2
        + 2
    )
    features = np.zeros((edge_count, feature_dim), dtype=np.float32)
    other_direction = DIRECTION_CATEGORIES.index("other")
    for edge in range(edge_count):
        source = node_ids[edge_index[0, edge]]
        destination = node_ids[edge_index[1, edge]]
        same_freeway = float(
            metadata.freeway[source] != ""
            and metadata.freeway[source] == metadata.freeway[destination]
        )
        same_direction = float(
            metadata.direction_code[source] != other_direction
            and metadata.direction_code[source] == metadata.direction_code[destination]
        )
        features[edge] = np.concatenate(
            (
                metadata.type_feature[source],
                metadata.type_feature[destination],
                metadata.direction_feature[source],
                metadata.direction_feature[destination],
                np.asarray((same_freeway, same_direction), dtype=np.float32),
            )
        )
    return features


def build_adaptive_subgraphs(
    adjacency: np.ndarray,
    train_node_ids: Iterable[int],
    metadata: NodeMetadata,
) -> list[dict[str, object]]:
    """Build every center graph with node and edge metadata in one pass."""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if metadata.num_nodes != adjacency.shape[0]:
        raise ValueError("metadata and adjacency node counts differ")
    train_nodes = {int(node) for node in train_node_ids}
    incoming, outgoing = _neighbor_lists(adjacency)
    subgraphs: list[dict[str, object]] = []
    for center in range(adjacency.shape[0]):
        up_hops, up_boundaries, up_edges = _expand_branch(
            center, incoming, train_nodes, upstream=True
        )
        down_hops, down_boundaries, down_edges = _expand_branch(
            center, outgoing, train_nodes, upstream=False
        )
        node_ids = np.asarray(
            [center] + sorted((set(up_hops) | set(down_hops)) - {center}), dtype=np.int64
        )
        local_index = {int(node): index for index, node in enumerate(node_ids)}
        propagation = up_edges + down_edges
        if propagation:
            local_edges = [
                (local_index[source], local_index[destination])
                for source, destination, _, _ in propagation
            ]
            edge_index = np.asarray(
                local_edges, dtype=np.int64
            ).T
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.asarray([edge[2] for edge in propagation], dtype=np.float32)
        edge_direction = np.asarray(
            [1.0] * len(up_edges) + [-1.0] * len(down_edges), dtype=np.float32
        )
        edge_hop = np.asarray([edge[3] for edge in propagation], dtype=np.int64)
        all_boundaries = up_boundaries | down_boundaries
        boundary_mask = np.asarray([node in all_boundaries for node in node_ids])
        upstream_mask = np.asarray([node in up_boundaries for node in node_ids])
        downstream_mask = np.asarray([node in down_boundaries for node in node_ids])
        boundary_mask[0] = upstream_mask[0] = downstream_mask[0] = False
        subgraphs.append(
            {
                "center_id": center,
                "center_index": 0,
                "node_ids": torch.from_numpy(node_ids),
                "type_code": torch.from_numpy(metadata.type_code[node_ids]),
                "direction_code": torch.from_numpy(metadata.direction_code[node_ids]),
                "type_feature": torch.from_numpy(metadata.type_feature[node_ids]),
                "direction_feature": torch.from_numpy(metadata.direction_feature[node_ids]),
                "boundary_mask": torch.from_numpy(boundary_mask),
                "upstream_boundary_mask": torch.from_numpy(upstream_mask),
                "downstream_boundary_mask": torch.from_numpy(downstream_mask),
                "up_hop": torch.tensor([up_hops.get(int(node), -1) for node in node_ids]),
                "down_hop": torch.tensor([down_hops.get(int(node), -1) for node in node_ids]),
                "propagation_edge_index": torch.from_numpy(edge_index),
                "propagation_edge_weight": torch.from_numpy(edge_weight),
                "propagation_direction": torch.from_numpy(edge_direction),
                "propagation_hop": torch.from_numpy(edge_hop),
                "propagation_edge_metadata": torch.from_numpy(
                    _edge_metadata(node_ids, edge_index, metadata)
                ),
            }
        )
    return subgraphs


def build_disconnected_supergraph(
    subgraphs: list[dict[str, object]], center_ids: Iterable[int]
) -> dict[str, torch.Tensor | int]:
    """Batch independent center graphs without creating cross-component edges."""
    centers = [int(center) for center in center_ids]
    if not centers:
        raise ValueError("at least one center is required")
    node_ids: list[torch.Tensor] = []
    center_indices: list[int] = []
    edge_indices: list[torch.Tensor] = []
    edge_weights: list[torch.Tensor] = []
    edge_directions: list[torch.Tensor] = []
    edge_hops: list[torch.Tensor] = []
    edge_metadata: list[torch.Tensor] = []
    boundaries: list[torch.Tensor] = []
    upstream_boundaries: list[torch.Tensor] = []
    downstream_boundaries: list[torch.Tensor] = []
    up_hops: list[torch.Tensor] = []
    down_hops: list[torch.Tensor] = []
    node_components: list[torch.Tensor] = []
    offset = 0
    for component, center in enumerate(centers):
        graph = subgraphs[center]
        local_nodes = graph["node_ids"]
        assert isinstance(local_nodes, torch.Tensor)
        node_ids.append(local_nodes.long())
        center_indices.append(offset + int(graph["center_index"]))
        boundaries.append(graph["boundary_mask"].bool())
        upstream_boundaries.append(graph["upstream_boundary_mask"].bool())
        downstream_boundaries.append(graph["downstream_boundary_mask"].bool())
        up_hops.append(graph["up_hop"].long())
        down_hops.append(graph["down_hop"].long())
        node_components.append(
            torch.full((len(local_nodes),), component, dtype=torch.long)
        )
        local_edges = graph["propagation_edge_index"]
        assert isinstance(local_edges, torch.Tensor)
        if local_edges.shape[1]:
            edge_indices.append(local_edges.long() + offset)
            edge_weights.append(graph["propagation_edge_weight"].float())
            edge_directions.append(graph["propagation_direction"].float())
            edge_hops.append(graph["propagation_hop"].long())
            edge_metadata.append(graph["propagation_edge_metadata"].float())
        offset += len(local_nodes)
    metadata_dim = subgraphs[centers[0]]["propagation_edge_metadata"].shape[1]
    return {
        "center_global_ids": torch.tensor(centers),
        "center_indices": torch.tensor(center_indices),
        "node_ids_global": torch.cat(node_ids),
        "boundary_mask": torch.cat(boundaries),
        "upstream_boundary_mask": torch.cat(upstream_boundaries),
        "downstream_boundary_mask": torch.cat(downstream_boundaries),
        "up_hop": torch.cat(up_hops),
        "down_hop": torch.cat(down_hops),
        # Maps every duplicated local node to its independent focal-subgraph
        # component.  Dynamic frequency selection uses it to keep one common
        # frequency coordinate system within each subgraph.
        "node_component": torch.cat(node_components),
        "edge_index": (
            torch.cat(edge_indices, dim=1)
            if edge_indices
            else torch.empty(2, 0, dtype=torch.long)
        ),
        "edge_weight": torch.cat(edge_weights) if edge_weights else torch.empty(0),
        "edge_direction": torch.cat(edge_directions) if edge_directions else torch.empty(0),
        "edge_hop": torch.cat(edge_hops) if edge_hops else torch.empty(0, dtype=torch.long),
        "edge_metadata": (
            torch.cat(edge_metadata) if edge_metadata else torch.empty(0, metadata_dim)
        ),
        "num_components": len(centers),
        "num_nodes": offset,
    }


def _full_graph_propagation_depth(
    node_count: int,
    anchors: torch.Tensor,
    edge_index: torch.Tensor,
    edge_direction: torch.Tensor,
) -> int:
    """Return the rounds needed to exhaust both directional BFS frontiers."""
    maximum_depth = 0
    anchor_ids = torch.nonzero(anchors, as_tuple=False).flatten().tolist()
    for direction in (1, -1):
        outgoing: list[list[int]] = [[] for _ in range(node_count)]
        direction_mask = edge_direction == direction
        sources = edge_index[0, direction_mask].tolist()
        destinations = edge_index[1, direction_mask].tolist()
        for source, destination in zip(sources, destinations, strict=True):
            outgoing[source].append(destination)
        distance = [-1] * node_count
        queue: deque[int] = deque()
        for anchor in anchor_ids:
            distance[anchor] = 0
            queue.append(anchor)
        while queue:
            source = queue.popleft()
            for destination in outgoing[source]:
                if distance[destination] >= 0:
                    continue
                distance[destination] = distance[source] + 1
                maximum_depth = max(maximum_depth, distance[destination])
                queue.append(destination)
    return max(1, maximum_depth)


def build_full_propagation_topology(
    subgraphs: list[dict[str, object]],
) -> dict[str, torch.Tensor | int]:
    """Recover the query-independent directed road topology once.

    The union recovers both propagation orientations already stored in the
    preprocessed subgraphs. For every original road edge, its destination's
    upstream subgraph contains the forward orientation and its source's
    downstream subgraph contains the reverse orientation.
    """
    node_count = len(subgraphs)
    edge_records: dict[
        tuple[int, int, int], tuple[float, torch.Tensor]
    ] = {}
    metadata_dim: int | None = None
    for graph in subgraphs:
        node_ids = graph["node_ids"]
        edge_index = graph["propagation_edge_index"]
        edge_weight = graph["propagation_edge_weight"]
        edge_direction = graph["propagation_direction"]
        edge_metadata = graph["propagation_edge_metadata"]
        assert isinstance(node_ids, torch.Tensor)
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(edge_weight, torch.Tensor)
        assert isinstance(edge_direction, torch.Tensor)
        assert isinstance(edge_metadata, torch.Tensor)
        metadata_dim = int(edge_metadata.shape[1])
        for edge in range(edge_index.shape[1]):
            source = int(node_ids[int(edge_index[0, edge])].item())
            destination = int(node_ids[int(edge_index[1, edge])].item())
            direction = 1 if float(edge_direction[edge].item()) > 0 else -1
            key = (direction, source, destination)
            if key not in edge_records:
                edge_records[key] = (
                    float(edge_weight[edge].item()),
                    edge_metadata[edge].float().clone(),
                )

    if metadata_dim is None:
        raise ValueError("subgraphs cannot be empty")
    ordered = sorted(edge_records.items())
    if ordered:
        edge_index = torch.tensor(
            [[key[1], key[2]] for key, _ in ordered], dtype=torch.long
        ).T.contiguous()
        edge_weight = torch.tensor(
            [record[0] for _, record in ordered], dtype=torch.float32
        )
        edge_direction = torch.tensor(
            [float(key[0]) for key, _ in ordered], dtype=torch.float32
        )
        edge_metadata = torch.stack([record[1] for _, record in ordered])
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float32)
        edge_direction = torch.empty(0, dtype=torch.float32)
        edge_metadata = torch.empty(0, metadata_dim, dtype=torch.float32)
    return {
        "node_ids_global": torch.arange(node_count, dtype=torch.long),
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_direction": edge_direction,
        # Full-graph GNN layers use the original edges directly rather than
        # target-specific hop labels.
        "edge_hop": torch.ones(edge_weight.shape[0], dtype=torch.long),
        "edge_metadata": edge_metadata,
        "num_nodes": node_count,
        "full_graph": 1,
    }


def build_full_propagation_graph(
    subgraphs: list[dict[str, object]],
    center_ids: Iterable[int],
    observed_node_ids: Iterable[int],
    *,
    topology: dict[str, torch.Tensor | int] | None = None,
) -> dict[str, torch.Tensor | int]:
    """Attach focal targets and visible anchors to the complete road graph."""
    centers = [int(center) for center in center_ids]
    if not centers:
        raise ValueError("at least one center is required")
    if topology is None:
        topology = build_full_propagation_topology(subgraphs)
    node_count = int(topology["num_nodes"])
    if node_count != len(subgraphs):
        raise ValueError("full topology and subgraph node counts differ")
    if any(center < 0 or center >= node_count for center in centers):
        raise IndexError("center ID is outside the full road graph")

    anchors = torch.zeros(node_count, dtype=torch.bool)
    observed_ids = torch.as_tensor(list(observed_node_ids), dtype=torch.long)
    if observed_ids.numel():
        if bool(((observed_ids < 0) | (observed_ids >= node_count)).any()):
            raise IndexError("observed node ID is outside the full road graph")
    anchors[observed_ids] = True
    anchors[torch.tensor(centers, dtype=torch.long)] = False
    edge_index = topology["edge_index"]
    edge_direction = topology["edge_direction"]
    assert isinstance(edge_index, torch.Tensor)
    assert isinstance(edge_direction, torch.Tensor)
    propagation_depth = _full_graph_propagation_depth(
        node_count, anchors, edge_index, edge_direction
    )
    centers_tensor = torch.tensor(centers, dtype=torch.long)
    return {
        **topology,
        "center_global_ids": centers_tensor,
        "center_indices": centers_tensor.clone(),
        "boundary_mask": anchors,
        "upstream_boundary_mask": anchors.clone(),
        "downstream_boundary_mask": anchors.clone(),
        "up_hop": torch.full((node_count,), -1, dtype=torch.long),
        "down_hop": torch.full((node_count,), -1, dtype=torch.long),
        "num_components": 1,
        "propagation_depth": propagation_depth,
    }
