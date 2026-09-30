"""Launch one matched baseline with its static metadata channels zeroed."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


STAGANN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGANN_DIR))
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from examples.pems_metadata_specs import (  # noqa: E402
    MODEL_NAMES,
    SUPPORTED_DISTRICTS,
    build_registered_no_metadata_config,
)
from stkriging import launch_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--district", required=True, type=int, choices=SUPPORTED_DISTRICTS
    )
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--gpus", default="0", help="GPU IDs passed to EasyTorch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    cfg = build_registered_no_metadata_config(
        args.model, args.district, top_k=args.top_k
    )
    launch_training(cfg, args.gpus)


if __name__ == "__main__":
    main()
