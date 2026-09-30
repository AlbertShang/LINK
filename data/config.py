"""District-specific preprocessing configuration.

Edit :func:`get_district_config` when a District needs a different spatial scale,
graph sparsity, or node split. The executable script also accepts temporary command-
line overrides, but these values are the reproducible experiment defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistrictPreprocessConfig:
    """All parameters that are allowed to vary between Districts."""

    core_radius_km: float
    min_local_neighbors: int
    min_2d_score: float
    min_unique_freeways: int
    expand_radius_km: float
    allow_upstream_boundary: bool
    top_k: int
    expected_pure_nodes: int | None = None
    split_seed: int = 42
    train_ratio: float = 0.70
    valid_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.core_radius_km <= 0 or self.expand_radius_km <= 0:
            raise ValueError("spatial radii must be positive")
        if self.min_local_neighbors < 0 or self.min_unique_freeways < 0:
            raise ValueError("minimum counts cannot be negative")
        if not 0 <= self.min_2d_score <= 1:
            raise ValueError("min_2d_score must be in [0, 1]")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.expected_pure_nodes is not None and self.expected_pure_nodes <= 0:
            raise ValueError("expected_pure_nodes must be positive or None")
        if not 0 < self.train_ratio < 1 or not 0 < self.valid_ratio < 1:
            raise ValueError("split ratios must be in (0, 1)")
        if self.train_ratio + self.valid_ratio >= 1:
            raise ValueError("train_ratio + valid_ratio must be less than one")


def get_district_config(district: int) -> DistrictPreprocessConfig:
    """Return the preprocessing defaults for one PEMS District.

    The spatial radii reproduce the released TraffiKrig node sets exactly.
    Graph Top-K changes adjacency sparsification only and must not change these
    selected node sets.
    """
    configs = {
        3: DistrictPreprocessConfig(
            core_radius_km=3.0,
            min_local_neighbors=10,
            min_2d_score=0.06,
            min_unique_freeways=2,
            expand_radius_km=5.0,
            allow_upstream_boundary=True,
            top_k=3,
            expected_pure_nodes=381,
        ),
        4: DistrictPreprocessConfig(
            core_radius_km=2.0,
            min_local_neighbors=10,
            min_2d_score=0.06,
            min_unique_freeways=2,
            expand_radius_km=2.0,
            allow_upstream_boundary=True,
            top_k=3,
            expected_pure_nodes=487,
        ),
        7: DistrictPreprocessConfig(
            core_radius_km=2.0,
            min_local_neighbors=10,
            min_2d_score=0.06,
            min_unique_freeways=2,
            expand_radius_km=2.2,
            allow_upstream_boundary=True,
            top_k=3,
            expected_pure_nodes=1151,
        ),
        8: DistrictPreprocessConfig(
            core_radius_km=3.0,
            min_local_neighbors=10,
            min_2d_score=0.06,
            min_unique_freeways=2,
            expand_radius_km=5.0,
            allow_upstream_boundary=True,
            top_k=3,
            expected_pure_nodes=512,
        ),
    }
    try:
        return configs[district]
    except KeyError as error:
        supported = ", ".join(str(value) for value in sorted(configs))
        raise ValueError(
            f"no preprocessing config for District {district}; choose {supported}"
        ) from error
