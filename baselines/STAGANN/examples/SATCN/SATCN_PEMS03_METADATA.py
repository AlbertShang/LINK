"""SATCN metadata configuration; legacy training hyperparameters are retained."""

import sys
from pathlib import Path

STAGANN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGANN_DIR))

from examples.pems03_metadata_common import build_metadata_config
from stkriging.archs import SATCN


CFG = build_metadata_config(
    model_name="SATCN",
    architecture=SATCN,
    model_parameters={"in_variables": 1, "layers": 1, "channels": 64, "neighbor": 2},
    learning_rate=0.0003,
    weight_decay=1.0e-6,
    epochs=50,
    batch_size=64,
    workers=2,
    min_delta=0.0,
    # MetadataSATCN supplies SATCN's second adjacency internally while
    # retaining the legacy Mask_Matrix=False training setting.
    mask_matrix=False,
    split_adjacency=True,
)
