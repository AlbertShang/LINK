"""Fail-fast check for the appended dataset and architecture-level fusion."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import sys
from pathlib import Path

# This script is also useful standalone on high-core training servers.  NumExpr
# reads these variables during indirect EasyTorch/Pandas imports.
_numexpr_limit = 64
try:
    _requested_numexpr_threads = int(
        os.environ.get("NUMEXPR_NUM_THREADS", os.cpu_count() or 1)
    )
except ValueError:
    _requested_numexpr_threads = os.cpu_count() or 1
os.environ["NUMEXPR_MAX_THREADS"] = str(_numexpr_limit)
os.environ["NUMEXPR_NUM_THREADS"] = str(
    min(max(_requested_numexpr_threads, 1), _numexpr_limit)
)

import numpy as np
import torch


STAGANN_DIR = Path(__file__).resolve().parents[1]
# Match examples/run.py's package roots.  EasyTorch's registry derives the
# import name ``STAGANN.data`` while importing stkriging, so both the STAGANN
# directory and its parent (which exposes the STAGANN namespace) are needed.
sys.path.insert(0, str(STAGANN_DIR))
if str(STAGANN_DIR.parent) not in sys.path:
    sys.path.append(str(STAGANN_DIR.parent))

from stkriging.archs.arch_zoo.metadata_arch import (
    METADATA_ARCHITECTURE_VERSION,
    TrafficMetadataFusion,
)
from stkriging.data import SCALER_REGISTRY
from stkriging.data.datasets import STKrigingDataset
from stkriging.data.metadata_dataset import MetadataSTKrigingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=STAGANN_DIR / "datasets" / "pems03_metadata_topk3_71_24",
    )
    return parser.parse_args()


def main() -> None:
    directory = parse_args().dataset_dir.resolve()
    with (directory / "scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)
    scaler_name = scaler.get("func")
    if not scaler_name:
        raise AssertionError("scaler.pkl does not declare a scaler function")
    SCALER_REGISTRY.get(scaler_name)
    scaler_mean = float(scaler.get("args", {}).get("mean", np.nan))
    scaler_std = float(scaler.get("args", {}).get("std", np.nan))
    if not np.isfinite(scaler_mean) or not np.isfinite(scaler_std) or scaler_std <= 0:
        raise AssertionError(
            f"Invalid standard scaler: mean={scaler_mean}, std={scaler_std}"
        )

    with (directory / "dataset_info.json").open("r", encoding="utf-8") as file:
        dataset_info = json.load(file)
    top_k = int(dataset_info.get("adjacency_top_k", 0))
    adjacency_file = str(dataset_info.get("adjacency_file", ""))
    with (directory / "adj_mx.pkl").open("rb") as file:
        adjacency = np.asarray(pickle.load(file))
    nonzero = adjacency != 0
    max_out_degree = int(nonzero.sum(axis=1).max())
    max_in_degree = int(nonzero.sum(axis=0).max())
    if (
        top_k <= 0
        or "adj_inout_topk" not in adjacency_file
        or max_out_degree > top_k
        or max_in_degree > top_k
    ):
        raise AssertionError(
            "Metadata dataset must use the configured directed in/out Top-K graph: "
            f"file={adjacency_file!r}, top_k={top_k}, "
            f"max_out={max_out_degree}, max_in={max_in_degree}. Regenerate it."
        )

    metadata_path = directory / "metadata.pkl"
    with metadata_path.open("rb") as file:
        metadata_payload = pickle.load(file)
    metadata_config = metadata_payload["config"]
    if metadata_config.get("layout_version") != METADATA_ARCHITECTURE_VERSION:
        raise AssertionError(
            "metadata.pkl is obsolete. Run generate_training_data.py --overwrite "
            "to rebuild it from district_*_node_metadata.npz."
        )
    metadata_dim = int(metadata_config["node_feature_dim"])

    dataset = MetadataSTKrigingDataset(
        data_file_path=str(directory / "data.pkl"),
        index_file_path=str(directory / "index.pkl"),
        node_index_file_path=str(directory / "adj_index.pkl"),
        metadata_file_path=str(metadata_path),
        metadata_after_flow=True,
        metadata_layout_version=METADATA_ARCHITECTURE_VERSION,
        mode="train",
    )
    sample = dataset[0]
    original = STKrigingDataset.__getitem__(dataset, 0)
    expected_features = original.shape[-1] + metadata_dim
    if sample.shape[-1] != expected_features:
        raise AssertionError(
            f"Expected {expected_features} concatenated features, got {sample.shape[-1]}"
        )
    torch.testing.assert_close(sample[..., :1], original[..., :1])
    torch.testing.assert_close(sample[..., -2:], original[..., 1:])
    torch.testing.assert_close(
        sample[..., 1:1 + metadata_dim],
        dataset.metadata.unsqueeze(0).expand(sample.shape[0], -1, -1),
    )

    fusion = TrafficMetadataFusion(
        metadata_dim=metadata_dim,
        dynamic_feature_dim=original.shape[-1],
        metadata_embedding_dim=20,
        fusion_hidden_dim=32,
        dropout=0.0,
    ).eval()
    # Make one explicit metadata-to-flow path deterministic so this check does
    # not depend on a lucky random initialization or active ReLU units.
    with torch.no_grad():
        for parameter in fusion.parameters():
            parameter.zero_()
        fusion.metadata_encoder[0].weight.fill_(1.0)
        fusion.metadata_encoder[1].weight[0, 0] = 1.0
        fusion.metadata_encoder[1].bias.fill_(2.0)
        fusion.metadata_encoder[4].weight[0, 0] = 1.0
        fusion.metadata_encoder[4].bias.fill_(2.0)
        fusion.flow_fusion[0].weight[0, 1] = 1.0
        fusion.flow_fusion[0].bias.fill_(2.0)
        fusion.flow_fusion[3].weight[0, 0] = 1.0
    fused = fusion(sample.unsqueeze(0))
    if fused.shape[-1] != original.shape[-1]:
        raise AssertionError(
            f"Legacy baseline must receive {original.shape[-1]} features, got {fused.shape[-1]}"
        )
    torch.testing.assert_close(fused[..., 1:], sample.unsqueeze(0)[..., -2:])

    changed = sample.unsqueeze(0).clone()
    changed[..., 1] += 10.0
    changed_fused = fusion(changed)
    if torch.equal(fused[..., 0], changed_fused[..., 0]):
        raise AssertionError("Changing metadata did not change the traffic-flow model input")

    architecture_path = inspect.getsourcefile(TrafficMetadataFusion)
    print(f"architecture:     {architecture_path}")
    print(
        f"scaler registry:  {scaler_name} [OK], "
        f"mean={scaler_mean:.6f}, std={scaler_std:.6f}"
    )
    print(
        f"adjacency:        {adjacency_file}, in/out Top-{top_k} "
        f"(max_in={max_in_degree}, max_out={max_out_degree})"
    )
    print(f"layout version:   {METADATA_ARCHITECTURE_VERSION}")
    print(f"original sample:  {tuple(original.shape)} [flow, TOD, DOW]")
    print(f"metadata:         {tuple(dataset.metadata.shape)}")
    print(f"concatenated:     {tuple(sample.shape)} [flow, metadata..., TOD, DOW]")
    print(f"baseline input:   {tuple(fused.shape)} [metadata-fused flow, TOD, DOW]")
    print("metadata input pipeline: OK")


if __name__ == "__main__":
    main()
