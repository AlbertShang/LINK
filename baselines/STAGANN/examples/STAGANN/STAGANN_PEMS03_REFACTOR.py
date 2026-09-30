"""STAGANN configuration for the converted pure PEMS03 dataset."""

import sys
import pickle
from pathlib import Path

import torch
from easydict import EasyDict


STAGANN_DIR = Path(__file__).resolve().parents[2]
if str(STAGANN_DIR) not in sys.path:
    sys.path.insert(0, str(STAGANN_DIR))

from stkriging.archs import STAGANN
from stkriging.data import STKrigingDataset
from stkriging.losses import masked_binary, masked_mae
from stkriging.runners import SimpleSpatiotemporalKrigingRunner
from stkriging.utils import load_adj


CFG = EasyDict()
CFG.DESCRIPTION = "STAGANN on the converted pure PEMS03 dataset"
CFG.RUNNER = SimpleSpatiotemporalKrigingRunner
CFG.DATASET_CLS = STKrigingDataset
CFG.DATASET_TRAINRATIO = 7
CFG.DATASET_VALRATIO = 1
CFG.DATASET_LEN = 24
CFG.DATASET_NAME = "pems03_refactor_71_24"
CFG.DATASET_TYPE = "Traffic flow"
CFG.GPU_NUM = 1

CFG.ENV = EasyDict(SEED=42, CUDNN=EasyDict(ENABLED=True))

dataset_dir = STAGANN_DIR / "datasets" / CFG.DATASET_NAME
adj_mx, train_index, valid_index, test_index = load_adj(
    str(dataset_dir / "adj_mx.pkl"), str(dataset_dir / "adj_index.pkl")
)
with (dataset_dir / "metadata.pkl").open("rb") as file:
    metadata = pickle.load(file)

node_metadata = torch.from_numpy(metadata["node_features"]).float()
metadata_config = metadata["config"]
if node_metadata.shape[0] != adj_mx.shape[0]:
    raise ValueError("Metadata node count must match the adjacency matrix.")

CFG.DATASET = EasyDict(
    MATRIX=adj_mx,
    MATRIX_TRANSFORM="original",
    TRANSFORM="standard_transform",
    TRAININDEX=train_index,
    VALIDINDEX=valid_index,
    TESTINDEX=test_index,
    # The runner subsets and permutes both tensors alongside the graph.  The
    # STAGANN metadata branch consumes the first tensor; the second preserves
    # the runner's two-location interface without duplicating data on disk.
    LOCATION=[node_metadata, node_metadata],
)

CFG.MODEL = EasyDict(
    NAME="STAGANN",
    ARCH=STAGANN,
    PARAM={
        "time_dimension": 24,
        "hidden_dimnesion": 100,
        "order": 2,
        "meta_data": 1,
        "metadata_dim": int(metadata_config["node_feature_dim"]),
        "metadata_layout": metadata_config,
        "mask": 50,
        "t_of_d": True,
        "d_of_w": True,
        "m_of_y": False,
        "dropout": 0.3,
    },
    FORWARD_FEATURES=[0, 1, 2],
    TARGET_FEATURES=[0],
)

CFG.TRAIN = EasyDict(
    LOSS={"masked_mae": masked_mae, "masked_binary": masked_binary},
    OPTIM=EasyDict(TYPE="Adam", PARAM={"lr": 0.0003, "weight_decay": 0.00001}),
    LR_SCHEDULER=EasyDict(
        TYPE="MultiStepLR", PARAM={"milestones": [30, 70, 120, 160], "gamma": 0.5}
    ),
    CLIP_GRAD_PARAM={"max_norm": 5.0},
    NUM_EPOCHS=50,
    BATCH_RANDOM=True,
    NULL_VAL=0.0,
    RATIO=0.7,
    DATA=EasyDict(
        DIR=str(dataset_dir), BATCH_SIZE=64, PREFETCH=False, SHUFFLE=True,
        NUM_WORKERS=2, PIN_MEMORY=False,
    ),
    EARLY_STOPPING=EasyDict(
        ENABLED=True, PATIENCE=50, MIN_DELTA=0.0,
        METRIC="val_MAE", GREATER_BEST=False,
    ),
)
CFG.TRAIN.CKPT_SAVE_DIR = str(
    STAGANN_DIR
    / "checkpoints"
    / CFG.DATASET_NAME
    / "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)

CFG.VAL = EasyDict(
    INTERVAL=1,
    DATA=EasyDict(
        DIR=str(dataset_dir), BATCH_SIZE=64, PREFETCH=False, SHUFFLE=False,
        NUM_WORKERS=2, PIN_MEMORY=False,
    ),
)
CFG.TEST = EasyDict(
    INTERVAL=1,
    DATA=EasyDict(
        DIR=str(dataset_dir), BATCH_SIZE=64, PREFETCH=False, SHUFFLE=False,
        NUM_WORKERS=2, PIN_MEMORY=False,
    ),
)
