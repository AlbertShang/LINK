"""GRIN metadata configuration; legacy training hyperparameters are retained."""

import sys
from pathlib import Path

STAGANN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGANN_DIR))

from examples.pems03_metadata_common import build_metadata_config
from stkriging.archs import GRIN


CFG = build_metadata_config(
    model_name="GRIN",
    architecture=GRIN,
    model_parameters={"d_in": 1, "d_hidden": 16, "d_ff": 10},
    learning_rate=0.001,
    weight_decay=1.0e-6,
    epochs=50,
    batch_size=64,
    workers=1,
    min_delta=0.5,
)
