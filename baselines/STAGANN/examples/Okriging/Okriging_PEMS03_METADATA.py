"""Ordinary-kriging metadata configuration; legacy settings are retained."""

import sys
from pathlib import Path

STAGANN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGANN_DIR))

from examples.pems03_metadata_common import build_metadata_config
from stkriging.archs import Okriging


CFG = build_metadata_config(
    model_name="Okriging",
    architecture=Okriging,
    model_parameters={},
    learning_rate=0.0000002,
    weight_decay=1.0e-6,
    epochs=50,
    batch_size=64,
    workers=2,
    min_delta=0.5,
)
