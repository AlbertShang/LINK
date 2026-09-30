"""Shared configuration builder for baselines on the pure PEMS03 dataset."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from easydict import EasyDict

from stkriging.archs import MetadataInputAdapter
from stkriging.data import STKrigingDataset
from stkriging.losses import masked_mae
from stkriging.runners import SimpleSpatiotemporalKrigingRunner
from stkriging.utils import load_adj


STAGANN_DIR = Path(__file__).resolve().parents[1]
DATASET_NAME = "pems03_refactor_71_24"


def _loader(directory: Path, batch_size: int, workers: int, *, shuffle: bool) -> EasyDict:
    return EasyDict(
        DIR=str(directory),
        BATCH_SIZE=batch_size,
        PREFETCH=False,
        SHUFFLE=shuffle,
        NUM_WORKERS=workers,
        PIN_MEMORY=False,
    )


def build_config(
    *,
    model_name: str,
    architecture: type,
    model_parameters: dict[str, Any],
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    workers: int,
    min_delta: float,
    runner: type = SimpleSpatiotemporalKrigingRunner,
    metadata_embedding_dim: int = 16,
    metadata_fusion_hidden_dim: int = 32,
    split_adjacency: bool = False,
) -> EasyDict:
    """Build a metadata-aware config against one canonical converted dataset."""
    dataset_dir = STAGANN_DIR / "datasets" / DATASET_NAME
    adjacency, train_nodes, valid_nodes, test_nodes = load_adj(
        str(dataset_dir / "adj_mx.pkl"), str(dataset_dir / "adj_index.pkl")
    )
    with (dataset_dir / "metadata.pkl").open("rb") as file:
        metadata = pickle.load(file)
    node_metadata = torch.as_tensor(metadata["node_features"], dtype=torch.float32)
    metadata_dim = int(metadata["config"]["node_feature_dim"])
    if node_metadata.ndim != 2 or node_metadata.shape[0] != adjacency.shape[0]:
        raise ValueError(
            "Metadata must be a [node, feature] matrix aligned with the adjacency matrix"
        )
    if node_metadata.shape[1] != metadata_dim:
        raise ValueError("metadata config does not match metadata node feature width")

    cfg = EasyDict()
    cfg.DESCRIPTION = f"{model_name} on the converted pure PEMS03 dataset"
    cfg.RUNNER = runner
    cfg.DATASET_CLS = STKrigingDataset
    cfg.DATASET_TRAINRATIO = 7
    cfg.DATASET_VALRATIO = 1
    cfg.DATASET_LEN = 24
    cfg.DATASET_NAME = DATASET_NAME
    cfg.DATASET_TYPE = "Traffic flow"
    cfg.GPU_NUM = 1
    cfg.ENV = EasyDict(SEED=42, CUDNN=EasyDict(ENABLED=True))

    cfg.DATASET = EasyDict(
        MATRIX=adjacency,
        MATRIX_TRANSFORM="original",
        TRANSFORM="standard_transform",
        TRAININDEX=train_nodes,
        VALIDINDEX=valid_nodes,
        TESTINDEX=test_nodes,
        LOCATION=[node_metadata, node_metadata],
    )
    cfg.MODEL = EasyDict(
        NAME=model_name,
        ARCH=MetadataInputAdapter,
        PARAM={
            "base_architecture": architecture,
            "base_parameters": model_parameters,
            "metadata_dim": metadata_dim,
            "dynamic_feature_dim": 3,
            "metadata_embedding_dim": metadata_embedding_dim,
            "fusion_hidden_dim": metadata_fusion_hidden_dim,
            "split_adjacency": split_adjacency,
        },
        FORWARD_FEATURES=[0, 1, 2],
        TARGET_FEATURES=[0],
    )

    cfg.TRAIN = EasyDict(
        LOSS={"masked_mae": masked_mae},
        OPTIM=EasyDict(
            TYPE="Adam",
            PARAM={"lr": learning_rate, "weight_decay": weight_decay, "eps": 1.0e-8},
        ),
        LR_SCHEDULER=EasyDict(
            TYPE="MultiStepLR",
            PARAM={"milestones": [30, 70, 120, 160], "gamma": 0.5},
        ),
        CLIP_GRAD_PARAM={"max_norm": 5.0},
        NUM_EPOCHS=epochs,
        BATCH_SELECT_NODE_RANDOM=True,
        BATCH_MATRIX_RANDOM=False,
        Mask_Matrix=split_adjacency,
        NULL_VAL=0.0,
        RATIO=0.7,
        DATA=_loader(dataset_dir, batch_size, workers, shuffle=True),
        EARLY_STOPPING=EasyDict(
            ENABLED=True,
            PATIENCE=10,
            MIN_DELTA=min_delta,
            METRIC="val_MAE",
            GREATER_BEST=False,
        ),
    )
    cfg.TRAIN.CKPT_SAVE_DIR = str(
        STAGANN_DIR / "checkpoints" / DATASET_NAME / f"{model_name}_{epochs}"
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
