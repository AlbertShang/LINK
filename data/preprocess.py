"""Build the PEMS03/04/07/08 districts of TraffiKrig from the 2023 archive.

Run from the repository root:

    python data/preprocess.py

The default ``all`` stage performs the complete archive-to-model-input pipeline:
four-month flow cleaning, metadata alignment, district/core selection, directed
Top-K construction, deterministic node splitting, and adaptive subgraph creation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import (  # noqa: E402
    DistrictPreprocessConfig,
    get_district_config,
)
from src.datasets import DistrictDataset, district_paths, load_district_dataset  # noqa: E402
from src.graph import NodeSplits, inout_topk_adjacency, make_node_splits  # noqa: E402
from src.metadata import (  # noqa: E402
    DIRECTION_CATEGORIES,
    TYPE_CATEGORIES,
    build_node_metadata,
    load_node_metadata,
    save_node_metadata,
)
from src.preprocessing import (  # noqa: E402
    CoreSelectionConfig,
    align_metadata,
    distance_to_adjacency,
    select_spatial_core,
)

DEFAULT_DISTRICTS = (3, 4, 7, 8)
DEFAULT_MONTHS = (1, 2, 3, 4)


def month_paths(archive_dir: Path, months: list[int]) -> list[Path]:
    """Resolve the nested LargeST 2023 monthly files and validate their presence."""
    paths = [archive_dir / "year_2023" / "year_2023" / f"2023_p{month:02d}.npy" for month in months]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing monthly flow files: " + ", ".join(missing))
    return paths


def calculate_valid_mask(
    paths: list[Path], max_missing_rate: float, node_chunk_size: int
) -> tuple[np.ndarray, int]:
    """Calculate the legacy node mask without loading all four months into RAM."""
    if not 0 <= max_missing_rate <= 1:
        raise ValueError("max_missing_rate must be in [0, 1]")
    if node_chunk_size <= 0:
        raise ValueError("node_chunk_size must be positive")

    arrays = [np.load(path, mmap_mode="r") for path in paths]
    first = arrays[0]
    if first.ndim != 3 or first.shape[2] < 1:
        raise ValueError(f"{paths[0]} must have shape [node, time, feature]")
    node_count = first.shape[0]
    for path, array in zip(paths, arrays, strict=True):
        if array.ndim != 3 or array.shape[0] != node_count or array.shape[2] < 1:
            raise ValueError(f"incompatible monthly array: {path} has shape {array.shape}")

    total_steps = sum(array.shape[1] for array in arrays)
    missing_count = np.zeros(node_count, dtype=np.int64)
    for start in range(0, node_count, node_chunk_size):
        end = min(start + node_chunk_size, node_count)
        for array in arrays:
            missing_count[start:end] += np.isnan(array[start:end, :, 0]).sum(axis=1)
    return missing_count / total_steps < max_missing_rate, total_steps


def _dataset_output_paths(output_dir: Path, district: int) -> list[Path]:
    flow, metadata, adjacency = district_paths(output_dir, district)
    return [
        flow,
        metadata,
        adjacency,
        output_dir / f"district_{district}_selected_index.npy",
        output_dir / f"district_{district}_archive_index.npy",
        output_dir / f"district_{district}_node_metadata.npz",
        output_dir / "preprocess_config.json",
    ]


def _graph_output_paths(output_dir: Path, district: int, top_k: int) -> list[Path]:
    return [
        output_dir / f"district_{district}_adj_inout_topk{top_k}.npy",
        output_dir / "adj_index.pkl",
        output_dir / f"adaptive_boundary_subgraphs_inout_topk{top_k}.pt",
    ]


def _output_paths(
    output_dir: Path, district: int, config: DistrictPreprocessConfig
) -> list[Path]:
    return _dataset_output_paths(output_dir, district) + _graph_output_paths(
        output_dir, district, config.top_k
    )


def resolve_district_configs(
    args: argparse.Namespace,
) -> dict[int, DistrictPreprocessConfig]:
    """Load function defaults and apply optional run-specific CLI overrides."""
    overrides = {
        name: value
        for name, value in {
            "core_radius_km": args.core_radius_km,
            "min_local_neighbors": args.min_local_neighbors,
            "min_2d_score": args.min_2d_score,
            "min_unique_freeways": args.min_unique_freeways,
            "expand_radius_km": args.expand_radius_km,
            "top_k": args.top_k,
            "split_seed": args.seed,
            "train_ratio": args.train_ratio,
            "valid_ratio": args.valid_ratio,
        }.items()
        if value is not None
    }
    if args.disallow_upstream_boundary:
        overrides["allow_upstream_boundary"] = False
    configs = {}
    for district in args.districts:
        config = get_district_config(district)
        saved_path = args.output_root / f"pems0{district}" / "preprocess_config.json"
        if args.stage == "graph" and saved_path.is_file():
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            saved.pop("district", None)
            config = replace(config, **saved)
        configs[district] = replace(config, **overrides)
    return configs


def _check_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "outputs already exist; use --overwrite to replace them: " + ", ".join(existing)
        )


def _write_selected_flow(
    paths: list[Path], archive_indices: np.ndarray, output_path: Path, total_steps: int
) -> None:
    """Stream selected nodes from node-major raw arrays into time-major pure flow."""
    first = np.load(paths[0], mmap_mode="r")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.npy")
    if temporary.exists():
        temporary.unlink()
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=first.dtype,
        shape=(total_steps, len(archive_indices), 1),
    )
    offset = 0
    for path in paths:
        month = np.load(path, mmap_mode="r")
        steps = month.shape[1]
        output[offset : offset + steps, :, 0] = month[archive_indices, :, 0].T
        offset += steps
    output.flush()
    del output
    temporary.replace(output_path)


def _save_dataset_stage(
    args: argparse.Namespace,
    paths: list[Path],
    valid_mask: np.ndarray,
    total_steps: int,
    valid_metadata: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Select and write the four aligned pure district datasets."""
    global_valid_indices = np.flatnonzero(valid_mask)
    adjacency_source = np.load(args.archive_dir / args.adjacency_file, mmap_mode="r")
    if adjacency_source.shape != (len(valid_mask), len(valid_mask)):
        raise ValueError(f"{args.adjacency_file} and monthly flow node counts differ")

    district_values = pd.to_numeric(valid_metadata[args.district_column], errors="raise").to_numpy()
    summary: dict[str, dict[str, object]] = {}
    for district in args.districts:
        district_config = args.district_configs[district]
        selection_config = CoreSelectionConfig(
            core_radius_km=district_config.core_radius_km,
            min_local_neighbors=district_config.min_local_neighbors,
            min_2d_score=district_config.min_2d_score,
            min_unique_freeways=district_config.min_unique_freeways,
            expand_radius_km=district_config.expand_radius_km,
            allow_upstream_boundary=district_config.allow_upstream_boundary,
        )
        district_mask = district_values == district
        if not district_mask.any():
            raise ValueError(f"district {district} has no nodes after flow cleaning")
        district_archive_indices = global_valid_indices[district_mask]
        district_metadata = valid_metadata.loc[district_mask].reset_index(drop=True)
        district_source = np.asarray(
            adjacency_source[np.ix_(district_archive_indices, district_archive_indices)]
        )
        if args.adjacency_kind == "distance":
            district_adjacency = distance_to_adjacency(district_source)
        else:
            district_adjacency = district_source.astype(np.float32, copy=True)
            district_adjacency[~np.isfinite(district_adjacency)] = 0
            district_adjacency[district_adjacency <= 0] = 0
            np.fill_diagonal(district_adjacency, 0)
        selection_input = DistrictDataset(
            flow=np.empty((1, len(district_metadata), 1), dtype=np.float32),
            metadata=district_metadata,
            adjacency=district_adjacency,
        )
        pure, selected_indices = select_spatial_core(selection_input, selection_config)
        expected_nodes = district_config.expected_pure_nodes
        if (
            expected_nodes is not None
            and len(selected_indices) != expected_nodes
            and not args.allow_node_count_change
        ):
            raise RuntimeError(
                f"PEMS0{district} produced {len(selected_indices)} nodes, but the "
                f"TraffiKrig specification requires {expected_nodes}. Check the "
                "spatial-selection parameters and source archive, or pass "
                "--allow-node-count-change for an intentional dataset variant."
            )
        pure_archive_indices = district_archive_indices[selected_indices]
        output_dir = args.output_root / f"pems0{district}"
        district_outputs = _output_paths(output_dir, district, district_config)
        _check_outputs(district_outputs, args.overwrite)
        if args.overwrite:
            stale_graph_outputs = list(
                output_dir.glob(f"district_{district}_adj_inout_topk*.npy")
            )
            stale_graph_outputs += list(
                output_dir.glob("adaptive_boundary_subgraphs_inout_topk*.pt")
            )
            stale_graph_outputs += [output_dir / "adj_index.pkl"]
            for stale_graph_output in stale_graph_outputs:
                if stale_graph_output.exists():
                    stale_graph_output.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)
        flow_path, metadata_path, adjacency_path = district_paths(output_dir, district)
        _write_selected_flow(paths, pure_archive_indices, flow_path, total_steps)
        pure.metadata.to_csv(metadata_path, sep="\t", index=False)
        np.save(adjacency_path, pure.adjacency.astype(np.float32, copy=False))
        np.save(output_dir / f"district_{district}_selected_index.npy", selected_indices)
        np.save(output_dir / f"district_{district}_archive_index.npy", pure_archive_indices)
        save_node_metadata(output_dir, district, build_node_metadata(pure.metadata))
        (output_dir / "preprocess_config.json").write_text(
            json.dumps(
                {"district": district, **asdict(district_config)},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        summary[f"pems0{district}"] = {
            "valid_district_nodes": int(len(district_metadata)),
            "pure_nodes": int(len(selected_indices)),
            "time_steps": int(total_steps),
            "config": asdict(district_config),
        }
        print(
            f"PEMS0{district}: {len(district_metadata)} cleaned nodes -> "
            f"{len(selected_indices)} pure nodes | core={district_config.core_radius_km:g}km | "
            f"expand={district_config.expand_radius_km:g}km | Top-{district_config.top_k}"
        )
    return summary


def _save_splits(path: Path, splits: NodeSplits, station_ids: np.ndarray) -> None:
    payload = {
        **asdict(splits),
        "test_ratio": 1 - splits.train_ratio - splits.valid_ratio,
        "train_station_ids": station_ids[splits.train_nodes],
        "valid_station_ids": station_ids[splits.valid_nodes],
        "test_station_ids": station_ids[splits.test_nodes],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _save_graph_stage(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is required; install requirements.txt first") from error

    from src.subgraphs import build_adaptive_subgraphs

    all_graph_outputs = [
        path
        for district in args.districts
        for path in _graph_output_paths(
            args.output_root / f"pems0{district}",
            district,
            args.district_configs[district].top_k,
        )
    ]
    _check_outputs(all_graph_outputs, args.overwrite)
    for district in args.districts:
        district_config = args.district_configs[district]
        directory = args.output_root / f"pems0{district}"
        dataset = load_district_dataset(directory, district)
        graph_outputs = _graph_output_paths(directory, district, district_config.top_k)
        _check_outputs(graph_outputs, args.overwrite)
        metadata = load_node_metadata(directory, district, dataset.num_nodes)
        topk_adjacency = inout_topk_adjacency(dataset.adjacency, district_config.top_k)
        topk_path, split_path, subgraph_path = graph_outputs
        np.save(topk_path, topk_adjacency)

        splits = make_node_splits(
            dataset.num_nodes,
            seed=district_config.split_seed,
            train_ratio=district_config.train_ratio,
            valid_ratio=district_config.valid_ratio,
        )
        station_ids = dataset.metadata[args.station_id].to_numpy()
        _save_splits(split_path, splits, station_ids)
        subgraphs = build_adaptive_subgraphs(topk_adjacency, splits.train_nodes, metadata)
        torch.save(
            {
                "district": district,
                "top_k": district_config.top_k,
                "train_nodes": splits.train_nodes,
                "valid_nodes": splits.valid_nodes,
                "test_nodes": splits.test_nodes,
                "metadata_config": {
                    "type_categories": TYPE_CATEGORIES,
                    "direction_categories": DIRECTION_CATEGORIES,
                    "edge_metadata_dim": int(
                        metadata.type_feature.shape[1] * 2
                        + metadata.direction_feature.shape[1] * 2
                        + 2
                    ),
                },
                "preprocess_config": asdict(district_config),
                "subgraphs": subgraphs,
            },
            subgraph_path,
        )
        (directory / "preprocess_config.json").write_text(
            json.dumps(
                {"district": district, **asdict(district_config)},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"PEMS0{district}: Top-{district_config.top_k}, split "
            f"{len(splits.train_nodes)}/{len(splits.valid_nodes)}/{len(splits.test_nodes)}, "
            f"{len(subgraphs)} adaptive subgraphs"
        )
    summary_path = args.output_root / "preprocessing_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("district_configs", {})
        for district, config in args.district_configs.items():
            summary["district_configs"][str(district)] = asdict(config)
            dataset_summary = summary.get("datasets", {}).get(f"pems0{district}")
            if dataset_summary is not None:
                dataset_summary["config"] = asdict(config)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "dataset", "graph"), default="all")
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "archive")
    parser.add_argument("--output-root", type=Path, default=ROOT / "pure_dataset_topk")
    parser.add_argument("--months", nargs="+", type=int, default=list(DEFAULT_MONTHS))
    parser.add_argument("--districts", nargs="+", type=int, default=list(DEFAULT_DISTRICTS))
    parser.add_argument("--max-missing-rate", type=float, default=0.30)
    parser.add_argument("--node-chunk-size", type=int, default=512)
    parser.add_argument("--adjacency-file", default="adj_matrix.npy")
    parser.add_argument("--adjacency-kind", choices=("weights", "distance"), default="weights")
    parser.add_argument("--district-column", default="District")
    parser.add_argument("--station-id", default="station_id")
    parser.add_argument("--top-k", type=int, help="override every selected District")
    parser.add_argument("--seed", type=int, help="override every selected District")
    parser.add_argument("--train-ratio", type=float, help="override every selected District")
    parser.add_argument("--valid-ratio", type=float, help="override every selected District")
    parser.add_argument("--core-radius-km", type=float, help="override every selected District")
    parser.add_argument("--min-local-neighbors", type=int, help="override every selected District")
    parser.add_argument("--min-2d-score", type=float, help="override every selected District")
    parser.add_argument("--min-unique-freeways", type=int, help="override every selected District")
    parser.add_argument("--expand-radius-km", type=float, help="override every selected District")
    parser.add_argument("--disallow-upstream-boundary", action="store_true")
    parser.add_argument(
        "--allow-node-count-change",
        action="store_true",
        help="permit an intentional node-selection variant instead of enforcing TraffiKrig counts",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.archive_dir = args.archive_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.district_configs = resolve_district_configs(args)
    if args.stage in {"all", "dataset"}:
        all_dataset_outputs = [
            path
            for district in args.districts
            for path in _output_paths(
                args.output_root / f"pems0{district}",
                district,
                args.district_configs[district],
            )
        ]
        _check_outputs(all_dataset_outputs, args.overwrite)
        paths = month_paths(args.archive_dir, args.months)
        valid_mask, total_steps = calculate_valid_mask(
            paths, args.max_missing_rate, args.node_chunk_size
        )
        node_order = np.load(args.archive_dir / "node_order.npy")
        raw_metadata = pd.read_csv(args.archive_dir / "sensor_meta_feature.csv", sep="\t")
        _, valid_metadata = align_metadata(node_order, raw_metadata, valid_mask, args.station_id)
        summary = _save_dataset_stage(args, paths, valid_mask, total_steps, valid_metadata)
        args.output_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset": "TraffiKrig",
            "year": 2023,
            "months": args.months,
            "districts": args.districts,
            "archive_nodes": int(len(valid_mask)),
            "valid_nodes": int(valid_mask.sum()),
            "max_missing_rate": args.max_missing_rate,
            "adjacency_file": args.adjacency_file,
            "adjacency_kind": args.adjacency_kind,
            "district_configs": {
                str(district): asdict(config)
                for district, config in args.district_configs.items()
            },
            "datasets": summary,
        }
        (args.output_root / "preprocessing_summary.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"cleaned nodes: {valid_mask.sum()}/{len(valid_mask)}")
    if args.stage in {"all", "graph"}:
        _save_graph_stage(args)


if __name__ == "__main__":
    main()
