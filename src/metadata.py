"""Metadata normalization and serialisation shared by every experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TYPE_CATEGORIES = ("mainline", "onramp", "offramp", "hov", "other")
DIRECTION_CATEGORIES = ("n", "s", "e", "w", "ne", "nw", "se", "sw", "other")


def normalize_type(value: object) -> str:
    if pd.isna(value):
        return "other"
    text = str(value).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if text == "mainline":
        return "mainline"
    if text in {"onramp", "onrmp", "entranceramp"}:
        return "onramp"
    if text in {"offramp", "offrmp", "exitramp"}:
        return "offramp"
    return "hov" if text == "hov" else "other"


def normalize_direction(value: object) -> str:
    if pd.isna(value):
        return "other"
    text = str(value).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "north": "n", "northbound": "n", "nb": "n",
        "south": "s", "southbound": "s", "sb": "s",
        "east": "e", "eastbound": "e", "eb": "e",
        "west": "w", "westbound": "w", "wb": "w",
        "northeast": "ne", "northeastbound": "ne", "neb": "ne",
        "northwest": "nw", "northwestbound": "nw", "nwb": "nw",
        "southeast": "se", "southeastbound": "se", "seb": "se",
        "southwest": "sw", "southwestbound": "sw", "swb": "sw",
    }
    text = aliases.get(text, text)
    return text if text in DIRECTION_CATEGORIES else "other"


def normalize_freeway(value: object) -> str:
    """Give equivalent numeric freeway IDs one stable text representation."""
    if pd.isna(value):
        return ""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


@dataclass(frozen=True)
class NodeMetadata:
    """Leakage-safe static metadata prepared before any graph is constructed.

    ``type_feature`` and ``direction_feature`` have fixed, public vocabularies. The
    canonical freeway string is intentionally retained rather than globally one-hot
    encoded: model training must derive its freeway vocabulary from train nodes only.
    """

    type_code: np.ndarray
    direction_code: np.ndarray
    type_feature: np.ndarray
    direction_feature: np.ndarray
    freeway: np.ndarray

    @property
    def num_nodes(self) -> int:
        return len(self.type_code)


def build_node_metadata(metadata: pd.DataFrame) -> NodeMetadata:
    """Normalize standard PEMS metadata and build fixed-vocabulary node features."""
    required = {"Type", "Direction", "Fwy"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError("metadata is missing required columns: " + ", ".join(sorted(missing)))
    type_index = {name: index for index, name in enumerate(TYPE_CATEGORIES)}
    direction_index = {name: index for index, name in enumerate(DIRECTION_CATEGORIES)}
    type_code = np.asarray(
        [type_index[normalize_type(x)] for x in metadata["Type"]], dtype=np.int64
    )
    direction_code = np.asarray(
        [direction_index[normalize_direction(x)] for x in metadata["Direction"]], dtype=np.int64
    )
    type_feature = np.eye(len(TYPE_CATEGORIES), dtype=np.float32)[type_code]
    direction_feature = np.eye(len(DIRECTION_CATEGORIES), dtype=np.float32)[direction_code]
    freeway = np.asarray([normalize_freeway(x) for x in metadata["Fwy"]], dtype=str)
    return NodeMetadata(type_code, direction_code, type_feature, direction_feature, freeway)


def metadata_path(directory: str | Path, district: int) -> Path:
    return Path(directory) / f"district_{district}_node_metadata.npz"


def save_node_metadata(directory: str | Path, district: int, node_metadata: NodeMetadata) -> Path:
    """Persist an aligned metadata artifact alongside a district dataset."""
    output = metadata_path(directory, district)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        type_code=node_metadata.type_code,
        direction_code=node_metadata.direction_code,
        type_feature=node_metadata.type_feature,
        direction_feature=node_metadata.direction_feature,
        freeway=node_metadata.freeway,
        type_categories=np.asarray(TYPE_CATEGORIES),
        direction_categories=np.asarray(DIRECTION_CATEGORIES),
    )
    return output


def load_node_metadata(
    directory: str | Path, district: int, expected_nodes: int | None = None
) -> NodeMetadata:
    """Load the standard artifact and ensure it remains aligned with the dataset."""
    path = metadata_path(directory, district)
    if not path.exists():
        raise FileNotFoundError(f"missing node metadata artifact: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = NodeMetadata(
            type_code=archive["type_code"],
            direction_code=archive["direction_code"],
            type_feature=archive["type_feature"],
            direction_feature=archive["direction_feature"],
            freeway=archive["freeway"],
        )
    if expected_nodes is not None and result.num_nodes != expected_nodes:
        raise ValueError(f"node metadata has {result.num_nodes} rows, expected {expected_nodes}")
    return result
