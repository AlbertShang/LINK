"""IGNNK configuration for the converted pure PEMS03 dataset."""

import sys
from pathlib import Path

STAGANN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGANN_DIR))

from examples.pems03_refactor_common import build_config
from stkriging.archs import IGNNK

CFG = build_config(
    model_name="IGNNK",
    architecture=IGNNK,
    model_parameters={"time_dimension": 24, "hidden_dimnesion": 100, "order": 1},
    learning_rate=0.0003,
    weight_decay=0.00001,
    epochs=50,
    batch_size=64,
    workers=2,
    min_delta=0.5,
)
