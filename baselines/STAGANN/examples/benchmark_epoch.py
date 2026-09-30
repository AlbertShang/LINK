"""Time one train-only epoch of a registered PEMS03 STAGANN baseline."""

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
    build_registered_metadata_config,
    build_registered_no_metadata_config,
)
from stkriging import launch_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--mode", choices=("metadata", "no_metadata"), required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k <= 0 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error("Top-K and batch size must be positive")
    builder = (
        build_registered_metadata_config
        if args.mode == "metadata" else build_registered_no_metadata_config
    )
    cfg = builder(args.model, 3, top_k=args.top_k)
    cfg.TRAIN.NUM_EPOCHS = 1
    cfg.TRAIN.BENCHMARK_RESULT_PATH = str(args.result.resolve())
    if args.batch_size is not None:
        cfg.TRAIN.DATA.BATCH_SIZE = args.batch_size
        cfg.VAL.DATA.BATCH_SIZE = args.batch_size
        cfg.TEST.DATA.BATCH_SIZE = args.batch_size
    launch_training(cfg, args.gpus)


if __name__ == "__main__":
    main()
