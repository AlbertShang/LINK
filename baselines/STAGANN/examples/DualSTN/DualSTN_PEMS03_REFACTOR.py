"""DualSTN configuration for the converted pure PEMS03 dataset."""

import sys
from pathlib import Path

STAGANN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGANN_DIR))

from examples.pems03_refactor_common import build_config
from stkriging.archs import DualSTN

CFG = build_config(
    model_name="DualSTN",
    architecture=DualSTN,
    model_parameters={
        "in_dim": 1,
        "hyperGCN_param": {"dims_hyper": [-1, 16], "depth_GCN": 1},
        "TGN_param": {
            "layer1": {"dims_TGN": [-1, 8], "depth_TGN": 3},
            "layer2": {"dims_TGN": [8, 16], "depth_TGN": 3},
            "layer3": {"dims_TGN": [16, 32], "depth_TGN": 3},
        },
        "hidden_size": 16,
        "length": 24,
    },
    learning_rate=0.001,
    weight_decay=1.0e-6,
    epochs=50,
    batch_size=64,
    workers=1,
    min_delta=0.5,
)
