"""STAGANN metadata-input configuration with legacy training hyperparameters."""

import sys
from pathlib import Path

STAGANN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGANN_DIR))

from examples.pems03_metadata_common import build_metadata_config
from stkriging.archs import STAGANN
from stkriging.losses import masked_binary, masked_mae


CFG = build_metadata_config(
    model_name="STAGANN",
    architecture=STAGANN,
    model_parameters={
        "time_dimension": 24,
        "hidden_dimnesion": 100,
        "order": 2,
        # Metadata is learned by MetadataSTAGANN's flow-fusion input block, so
        # the legacy architecture's unrelated location branch stays off.
        "meta_data": 0,
        "mask": 50,
        "t_of_d": True,
        "d_of_w": True,
        "m_of_y": False,
        "dropout": 0.3,
    },
    learning_rate=0.0003,
    weight_decay=0.00001,
    epochs=50,
    batch_size=64,
    workers=2,
    min_delta=0.0,
    losses={"masked_mae": masked_mae, "masked_binary": masked_binary},
    optimizer_eps=None,
    early_stopping_patience=10,
    batch_select_node_random=None,
    batch_matrix_random=None,
    batch_random=True,
)
