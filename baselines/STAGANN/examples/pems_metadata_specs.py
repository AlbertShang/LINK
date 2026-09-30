"""Paper-aligned baseline settings shared by all pure PEMS districts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from stkriging.archs import (
    DualSTN,
    GCN,
    GIN,
    GRIN,
    IAGCN,
    IGNNK,
    INCREASE,
    Okriging,
    SATCN,
    STAGANN,
)
from stkriging.losses import masked_binary, masked_mae

from examples.pems03_metadata_common import build_metadata_config


MODEL_NAMES = (
    "DualSTN",
    "GCN",
    "GIN",
    "GRIN",
    "IAGCN",
    "IGNNK",
    "INCREASE",
    "Okriging",
    "SATCN",
    "STAGANN",
)
SUPPORTED_DISTRICTS = (3, 4, 7, 8)
UNIFIED_EPOCHS = 50
UNIFIED_EARLY_STOPPING_PATIENCE = 10
PAPER_BATCH_SIZE = 64
PAPER_METADATA_EMBEDDING_DIM = 20

_ARCHITECTURES = {
    "DualSTN": DualSTN,
    "GCN": GCN,
    "GIN": GIN,
    "GRIN": GRIN,
    "IAGCN": IAGCN,
    "IGNNK": IGNNK,
    "INCREASE": INCREASE,
    "Okriging": Okriging,
    "SATCN": SATCN,
    "STAGANN": STAGANN,
}

_BASE_SPECS: dict[str, dict[str, Any]] = {
    "DualSTN": {
        "model_parameters": {
            "in_dim": 1,
            "hyperGCN_param": {"dims_hyper": [-1, 16], "depth_GCN": 1},
            "TGN_param": {
                "layer1": {"dims_TGN": [-1, 8], "depth_TGN": 3},
                "layer2": {"dims_TGN": [8, 16], "depth_TGN": 3},
                "layer3": {"dims_TGN": [16, 32], "depth_TGN": 3},
            },
            "hidden_size": 100,
            "length": 24,
            "dropout": 0.3,
        },
        "learning_rate": 0.003,
        "weight_decay": 1.0e-6,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 1,
        "min_delta": 0.5,
    },
    "GCN": {
        "model_parameters": {
            "time_dimension": 24,
            "hidden_dimnesion": 100,
            "order": 2,
        },
        "learning_rate": 0.003,
        "weight_decay": 1.0e-6,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 1,
        "min_delta": 0.5,
    },
    "GIN": {
        "model_parameters": {
            "time_dimension": 24,
            "hidden_dimnesion": 100,
            "order": 2,
        },
        "learning_rate": 0.003,
        "weight_decay": 1.0e-6,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 1,
        "min_delta": 0.5,
    },
    "GRIN": {
        "model_parameters": {
            "d_in": 1,
            "d_hidden": 16,
            "d_ff": 10,
            "ff_dropout": 0.3,
        },
        "learning_rate": 0.01,
        "weight_decay": 1.0e-6,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 1,
        "min_delta": 0.5,
    },
    "IAGCN": {
        "model_parameters": {
            "T": 24,
            "dropout": 0.3,
            "residual_channels": 100,
            "dilation_channels": 100,
            "skip_channels": 100,
            "end_channels": 100,
        },
        "learning_rate": 0.003,
        "weight_decay": 1.0e-6,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 1,
        "min_delta": 0.5,
    },
    "IGNNK": {
        "model_parameters": {
            "time_dimension": 24,
            "hidden_dimnesion": 100,
            "order": 1,
        },
        "learning_rate": 0.003,
        "weight_decay": 1.0e-5,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 2,
        "min_delta": 0.5,
    },
    "INCREASE": {
        "model_parameters": {"K": 4, "hidden_dim": 16},
        # Match the original per-dataset INCREASE configurations.  A 1e-2
        # learning rate can destabilize its recurrent exponential gates on
        # the converted district graphs during the first epoch.
        "learning_rate": 0.001,
        "weight_decay": 1.0e-6,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 1,
        "min_delta": 0.5,
    },
    "Okriging": {
        "model_parameters": {},
        "learning_rate": 0.003,
        "weight_decay": 1.0e-6,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 2,
        "min_delta": 0.5,
    },
    "SATCN": {
        "model_parameters": {
            "in_variables": 1,
            "layers": 1,
            "channels": 16,
            "neighbor": 2,
            "dropout": 0.3,
        },
        "learning_rate": 0.003,
        "weight_decay": 1.0e-6,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 2,
        "min_delta": 0.5,
        "mask_matrix": False,
        "split_adjacency": True,
        "matrix_transform": "random_walk",
    },
    "STAGANN": {
        "model_parameters": {
            "time_dimension": 24,
            "hidden_dimnesion": 100,
            "order": 2,
            "meta_data": 0,
            "mask": 50,
            "t_of_d": True,
            "d_of_w": True,
            "m_of_y": False,
            "metadata_embedding_dim": 20,
            "label_embedding_dim": 12,
            "adversarial_rounds": 5,
            "dropout": 0.3,
        },
        "learning_rate": 0.003,
        "weight_decay": 1.0e-5,
        "epochs": UNIFIED_EPOCHS,
        "batch_size": PAPER_BATCH_SIZE,
        "workers": 2,
        "min_delta": 0.5,
        "losses": {"masked_mae": masked_mae, "masked_binary": masked_binary},
    },
}

# Only implementation details not fixed by the paper's shared parameter table
# remain district-specific. In particular, learning rate, batch size, dropout,
# and temporal embedding dimensions must not be overridden here.
_DISTRICT_OVERRIDES: dict[tuple[int, str], dict[str, Any]] = {
    # Mirror the non-budget settings used by the existing PEMS03 metadata
    # configurations so registered zero-metadata runs remain exactly matched.
    (3, "GCN"): {"min_delta": 0.0},
    (3, "GIN"): {"plot_imputation_epoch": 50},
    (3, "IAGCN"): {"model_parameters": {
        **_BASE_SPECS["IAGCN"]["model_parameters"],
        "gcn_layer": 1,
    }},
    (3, "IGNNK"): {"optimizer_eps": None},
    (3, "SATCN"): {"min_delta": 0.0},
    (3, "STAGANN"): {
        "model_parameters": {
            **_BASE_SPECS["STAGANN"]["model_parameters"],
            "dropout": 0.3,
        },
        "min_delta": 0.0,
        "optimizer_eps": None,
        "batch_select_node_random": None,
        "batch_matrix_random": None,
        "batch_random": True,
    },
    (7, "IGNNK"): {"weight_decay": 1.0e-6},
    (8, "IGNNK"): {"weight_decay": 1.0e-6},
    (7, "STAGANN"): {"weight_decay": 1.0e-6},
    (8, "STAGANN"): {
        "weight_decay": 1.0e-6,
    },
}


def registered_epochs(model_name: str, district: int) -> int:
    """Return the configured maximum epoch count without loading a dataset."""
    if model_name not in _ARCHITECTURES:
        raise KeyError(f"Unknown baseline: {model_name}")
    if district not in SUPPORTED_DISTRICTS:
        raise ValueError(f"Unsupported PEMS district: {district}")
    return UNIFIED_EPOCHS


def registered_baseline_spec(
    model_name: str, district: int
) -> tuple[type, dict[str, Any]]:
    """Return an original baseline architecture with paper-aligned settings."""
    if model_name not in _ARCHITECTURES:
        raise KeyError(f"Unknown baseline: {model_name}")
    if district not in SUPPORTED_DISTRICTS:
        raise ValueError(f"Unsupported PEMS district: {district}")
    spec = deepcopy(_BASE_SPECS[model_name])
    spec.update(deepcopy(_DISTRICT_OVERRIDES.get((district, model_name), {})))
    # Use one common optimization budget for every model and district. This is
    # also shared by metadata-on and zero-metadata configurations.
    spec["epochs"] = UNIFIED_EPOCHS
    spec["batch_size"] = PAPER_BATCH_SIZE
    spec["early_stopping_patience"] = UNIFIED_EARLY_STOPPING_PATIENCE
    spec["metadata_embedding_dim"] = PAPER_METADATA_EMBEDDING_DIM
    return _ARCHITECTURES[model_name], spec


def build_registered_metadata_config(
    model_name: str, district: int, *, top_k: int = 3
):
    """Build one paper-aligned metadata configuration for a selected graph."""
    architecture, spec = registered_baseline_spec(model_name, district)
    return build_metadata_config(
        district=district,
        top_k=top_k,
        model_name=model_name,
        architecture=architecture,
        **spec,
    )


def build_registered_no_metadata_config(
    model_name: str, district: int, *, top_k: int = 3
):
    """Build the matching configuration without static node metadata."""
    from examples.pems_no_metadata_common import build_no_metadata_config

    architecture, spec = registered_baseline_spec(model_name, district)
    return build_no_metadata_config(
        district=district,
        top_k=top_k,
        model_name=model_name,
        architecture=architecture,
        **spec,
    )
