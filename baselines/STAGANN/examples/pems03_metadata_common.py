"""Configuration builder for metadata variants of pure PEMS baselines."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from easydict import EasyDict

from stkriging.archs.arch_zoo.metadata_arch import (
    METADATA_ARCHITECTURES,
    METADATA_ARCHITECTURE_VERSION,
)
from stkriging.data.metadata_dataset import MetadataSTKrigingDataset
from stkriging.losses import masked_mae
from stkriging.runners import MetadataSpatiotemporalKrigingRunner
from stkriging.utils import load_adj


STAGANN_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_DISTRICTS = (3, 4, 7, 8)
DYNAMIC_FEATURE_DIM = 3  # normalized flow, time of day, day of week


def _loader(directory: Path, batch_size: int, workers: int, *, shuffle: bool) -> EasyDict:
    return EasyDict(
        DIR=str(directory),
        BATCH_SIZE=batch_size,
        PREFETCH=False,
        SHUFFLE=shuffle,
        NUM_WORKERS=workers,
        PIN_MEMORY=False,
    )


def build_metadata_config(
    *,
    district: int = 3,
    top_k: int = 3,
    model_name: str,
    architecture: type,
    model_parameters: dict[str, Any],
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    workers: int,
    min_delta: float,
    losses: dict[str, Any] | None = None,
    optimizer_eps: float | None = 1.0e-8,
    early_stopping_patience: int = 10,
    batch_select_node_random: bool | None = True,
    batch_matrix_random: bool | None = False,
    mask_matrix: bool = False,
    batch_random: bool | None = None,
    plot_imputation_epoch: int | None = None,
    split_adjacency: bool = False,
    matrix_transform: str = "original",
    metadata_embedding_dim: int = 20,
    metadata_fusion_hidden_dim: int = 32,
    zero_metadata: bool = False,
) -> EasyDict:
    """Build matched metadata-on or zero-metadata baseline configurations."""
    if district not in SUPPORTED_DISTRICTS:
        raise ValueError(
            f"Unsupported PEMS district {district}; expected one of {SUPPORTED_DISTRICTS}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    dataset_name = f"pems{district:02d}_metadata_topk{top_k}_71_24"
    dataset_dir = STAGANN_DIR / "datasets" / dataset_name
    metadata_path = dataset_dir / "metadata.pkl"
    adjacency, train_nodes, valid_nodes, test_nodes = load_adj(
        str(dataset_dir / "adj_mx.pkl"), str(dataset_dir / "adj_index.pkl")
    )
    with metadata_path.open("rb") as file:
        metadata = pickle.load(file)
    with (dataset_dir / "dataset_info.json").open("r", encoding="utf-8") as file:
        dataset_info = json.load(file)
    processed_shape = dataset_info.get("processed_data_shape", [])
    if len(processed_shape) != 3 or processed_shape[-1] != DYNAMIC_FEATURE_DIM:
        raise ValueError(
            "Metadata configurations require [flow, TOD, DOW] before metadata; "
            f"dataset reports processed_data_shape={processed_shape}"
        )
    node_features = metadata.get("node_features")
    metadata_config = metadata.get("config", {})
    if metadata_config.get("layout_version") != METADATA_ARCHITECTURE_VERSION:
        raise ValueError(
            "metadata.pkl is from the obsolete CSV/Lat/Lng encoder. Regenerate it "
            "from pure_dataset district_*_node_metadata.npz with "
            "generate_training_data.py --overwrite."
        )
    metadata_dim = int(metadata_config.get("node_feature_dim", 0))
    if node_features is None or getattr(node_features, "ndim", None) != 2:
        raise ValueError("metadata.pkl must contain node_features with shape [node, feature]")
    if node_features.shape != (adjacency.shape[0], metadata_dim):
        raise ValueError(
            "Metadata must align with the adjacency matrix and metadata config: "
            f"{node_features.shape} != ({adjacency.shape[0]}, {metadata_dim})"
        )

    cfg = EasyDict()
    metadata_mode = "no_metadata" if zero_metadata else "metadata"
    cfg.DESCRIPTION = (
        f"{model_name} {metadata_mode} variant on converted pure PEMS{district:02d}"
    )
    cfg.RUNNER = MetadataSpatiotemporalKrigingRunner
    cfg.DATASET_CLS = MetadataSTKrigingDataset
    cfg.DATASET_ARGS = {
        "metadata_file_path": str(metadata_path),
        "metadata_after_flow": True,
        "metadata_layout_version": METADATA_ARCHITECTURE_VERSION,
        "zero_metadata": zero_metadata,
    }
    cfg.DATASET_TRAINRATIO = 7
    cfg.DATASET_VALRATIO = 1
    cfg.DATASET_LEN = 24
    cfg.DATASET_NAME = dataset_name
    cfg.DATASET_TYPE = "Traffic flow"
    cfg.GPU_NUM = 1
    cfg.ENV = EasyDict(SEED=42, CUDNN=EasyDict(ENABLED=True))

    cfg.DATASET = EasyDict(
        MATRIX=adjacency,
        MATRIX_TRANSFORM=matrix_transform,
        TRANSFORM="standard_transform",
        TRAININDEX=train_nodes,
        VALIDINDEX=valid_nodes,
        TESTINDEX=test_nodes,
    )
    metadata_architecture = METADATA_ARCHITECTURES.get(architecture)
    if metadata_architecture is None:
        raise KeyError(f"No metadata architecture registered for {architecture.__name__}")
    cfg.MODEL = EasyDict(
        NAME=model_name,
        ARCH=metadata_architecture,
        PARAM={
            "backbone_parameters": model_parameters,
            "metadata_dim": metadata_dim,
            "dynamic_feature_dim": DYNAMIC_FEATURE_DIM,
            "metadata_embedding_dim": metadata_embedding_dim,
            "fusion_hidden_dim": metadata_fusion_hidden_dim,
            "split_adjacency": split_adjacency,
        },
        FORWARD_FEATURES=list(range(DYNAMIC_FEATURE_DIM + metadata_dim)),
        TARGET_FEATURES=[0],
    )

    optimizer_parameters = {"lr": learning_rate, "weight_decay": weight_decay}
    if optimizer_eps is not None:
        optimizer_parameters["eps"] = optimizer_eps
    cfg.TRAIN = EasyDict(
        LOSS=losses or {"masked_mae": masked_mae},
        OPTIM=EasyDict(TYPE="Adam", PARAM=optimizer_parameters),
        LR_SCHEDULER=EasyDict(
            TYPE="MultiStepLR",
            PARAM={"milestones": [30, 70, 120, 160], "gamma": 0.5},
        ),
        CLIP_GRAD_PARAM={"max_norm": 5.0},
        NUM_EPOCHS=epochs,
        NULL_VAL=0.0,
        RATIO=0.7,
        DATA=_loader(dataset_dir, batch_size, workers, shuffle=True),
        EARLY_STOPPING=EasyDict(
            ENABLED=True,
            PATIENCE=early_stopping_patience,
            MIN_DELTA=min_delta,
            METRIC="val_MAE",
            GREATER_BEST=False,
        ),
        ADVERSARIAL_ROUNDS=5,
    )
    if batch_select_node_random is not None:
        cfg.TRAIN.BATCH_SELECT_NODE_RANDOM = batch_select_node_random
    if batch_matrix_random is not None:
        cfg.TRAIN.BATCH_MATRIX_RANDOM = batch_matrix_random
    if batch_random is not None:
        cfg.TRAIN.BATCH_RANDOM = batch_random
    if plot_imputation_epoch is not None:
        cfg.TRAIN.PLOT_IMPUTATION_EPOCH = plot_imputation_epoch
    cfg.TRAIN.Mask_Matrix = mask_matrix
    cfg.TRAIN.CKPT_SAVE_DIR = str(
        STAGANN_DIR
        / "checkpoints"
        / dataset_name
        / metadata_mode
        / f"{model_name}_{epochs}"
    )

    cfg.VAL = EasyDict(
        INTERVAL=1,
        DATA=_loader(dataset_dir, batch_size, workers, shuffle=False),
    )
    cfg.TEST = EasyDict(
        INTERVAL=1,
        DATA=_loader(dataset_dir, batch_size, workers, shuffle=False),
    )
    return cfg
