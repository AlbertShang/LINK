"""Plot node-level reconstruction curves for IGNNK and LINK.

The script evaluates IGNNK without metadata from its best validation-MAE
checkpoint, aligns the prediction with LINK's completed flow,
and selects one On Ramp and one Mainline test node by the full-test-period MAE
gap (IGNNK MAE minus LINK MAE).  The displayed day is selected only
by ground-truth variability after the nodes have been fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STAGANN_DIR = ROOT / "baselines" / "STAGANN"
if str(STAGANN_DIR) not in sys.path:
    sys.path.insert(0, str(STAGANN_DIR))

from stkriging.archs.arch_zoo.metadata_arch import MetadataIGNNK  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--district", type=int, default=3, choices=(3, 4, 7, 8))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--pure-data-root", type=Path, default=ROOT / "pure_dataset_topk")
    parser.add_argument(
        "--link-completion",
        type=Path,
        default=ROOT / "outputs" / "pems03" / "completed_flow.npy",
        help="metadata-aware LINK completed_flow.npy",
    )
    parser.add_argument(
        "--ignnk-checkpoint",
        type=Path,
        help="IGNNK no-metadata best-val-MAE checkpoint; auto-detected if omitted",
    )
    parser.add_argument(
        "--ignnk-prediction",
        type=Path,
        help="reuse an aligned [time,node] IGNNK prediction instead of inference",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "oversmoothing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-points", type=int, default=288)
    parser.add_argument(
        "--min-window-valid-ratio",
        type=float,
        default=0.8,
        help="minimum ratio of points valid for both selected nodes",
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--min-mae-gap", type=float, default=0.0)
    args = parser.parse_args()
    if args.top_k <= 0 or args.batch_size <= 0 or args.window_points <= 1:
        parser.error("--top-k, --batch-size and --window-points must be positive")
    if not 0 < args.min_window_valid_ratio <= 1:
        parser.error("--min-window-valid-ratio must be in (0, 1]")
    return args


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def squeeze_flow(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [time,node] or [time,node,1], got {values.shape}")
    return values.astype(np.float32, copy=False)


def checkpoint_candidates(district: int, top_k: int) -> list[Path]:
    root = (
        STAGANN_DIR / "checkpoints"
        / f"pems{district:02d}_metadata_topk{top_k}_71_24"
        / "no_metadata"
    )
    candidates = list(root.glob("IGNNK_*/**/IGNNK_best_val_MAE.pt"))
    if not candidates:
        candidates = list(root.glob("IGNNK_*/**/*best*MAE*.pt"))
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.ignnk_checkpoint is not None:
        checkpoint = args.ignnk_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"IGNNK checkpoint not found: {checkpoint}")
        return checkpoint
    candidates = checkpoint_candidates(args.district, args.top_k)
    if not candidates:
        raise FileNotFoundError(
            "Cannot find an IGNNK no-metadata best-val-MAE checkpoint. "
            "Pass it explicitly with --ignnk-checkpoint."
        )
    return candidates[0]


def state_dict_from_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], int | None]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected checkpoint payload: {type(payload).__name__}")
    state = payload.get("model_state_dict", payload.get("model", payload))
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint has no model state dictionary: {path}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    epoch = payload.get("epoch")
    return state, int(epoch) if epoch is not None else None


def infer_ignnk(
    dataset_dir: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    data_payload = load_pickle(dataset_dir / "data.pkl")
    processed = np.asarray(data_payload["processed_data"], dtype=np.float32)
    splits = load_pickle(dataset_dir / "adj_index.pkl")
    indices = load_pickle(dataset_dir / "index.pkl")
    scaler = load_pickle(dataset_dir / "scaler.pkl")
    metadata = load_pickle(dataset_dir / "metadata.pkl")
    adjacency = np.asarray(load_pickle(dataset_dir / "adj_mx.pkl"), dtype=np.float32)

    train_nodes = np.asarray(splits["train_nodes"], dtype=np.int64)
    test_nodes = np.asarray(splits["test_nodes"], dtype=np.int64)
    node_order = np.concatenate((train_nodes, test_nodes))
    unknown_local = np.arange(len(train_nodes), len(node_order), dtype=np.int64)
    adjacency = adjacency[np.ix_(node_order, node_order)]
    metadata_dim = int(metadata["config"]["node_feature_dim"])

    model = MetadataIGNNK(
        backbone_parameters={"time_dimension": 24, "hidden_dimnesion": 100, "order": 1},
        metadata_dim=metadata_dim,
        dynamic_feature_dim=3,
        metadata_embedding_dim=20,
        fusion_hidden_dim=32,
        split_adjacency=False,
    ).to(device)
    state, epoch = state_dict_from_checkpoint(checkpoint, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    windows = [tuple(map(int, item)) for item in indices["test"]]
    if not windows or any(len(item) != 2 or item[1] - item[0] != 24 for item in windows):
        raise ValueError("IGNNK export expects 24-point (start,end) test windows")
    predictions = np.full((processed.shape[0], processed.shape[1]), np.nan, dtype=np.float32)
    sums = np.zeros((processed.shape[0], len(test_nodes)), dtype=np.float64)
    counts = np.zeros((processed.shape[0], len(test_nodes)), dtype=np.int32)
    adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    mean = float(np.asarray(scaler["args"]["mean"]).reshape(-1)[0])
    std = float(np.asarray(scaler["args"]["std"]).reshape(-1)[0])

    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            batch_windows = windows[offset:offset + batch_size]
            dynamic = np.stack(
                [processed[start:end, node_order, :] for start, end in batch_windows], axis=0
            )
            zeros = np.zeros(
                (*dynamic.shape[:-1], metadata_dim), dtype=np.float32
            )
            model_input = np.concatenate(
                (dynamic[..., :1], zeros, dynamic[..., 1:]), axis=-1
            )
            model_input[..., unknown_local, 0] = 0.0
            tensor = torch.from_numpy(model_input).to(device)
            batch_unknown = np.broadcast_to(
                unknown_local[None, :], (len(batch_windows), len(unknown_local))
            )
            output = model(
                X=tensor,
                adj=adjacency_tensor,
                unknown_nodes=batch_unknown,
                location=None,
                batch_seen=None,
                epoch=1,
                train=False,
            )[0]
            output = output[..., unknown_local].detach().cpu().numpy() * std + mean
            for row, (start, end) in enumerate(batch_windows):
                sums[start:end] += output[row]
                counts[start:end] += 1

    valid = counts > 0
    test_prediction = np.full_like(sums, np.nan, dtype=np.float32)
    test_prediction[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    predictions[:, test_nodes] = test_prediction
    return predictions, {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": epoch,
        "test_window_count": len(windows),
        "test_nodes": int(len(test_nodes)),
    }


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalized_type(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def node_metrics(
    truth: np.ndarray,
    ignnk: np.ndarray,
    link: np.ndarray,
    node: int,
) -> dict[str, float | int]:
    valid = (
        np.isfinite(truth[:, node])
        & (truth[:, node] > 0)
        & np.isfinite(ignnk[:, node])
        & np.isfinite(link[:, node])
    )
    if not valid.any():
        return {"count": 0, "ignnk_mae": np.nan, "link_mae": np.nan, "mae_gap": np.nan}
    ignnk_mae = float(np.mean(np.abs(ignnk[valid, node] - truth[valid, node])))
    link_mae = float(np.mean(np.abs(link[valid, node] - truth[valid, node])))
    return {
        "count": int(valid.sum()),
        "ignnk_mae": ignnk_mae,
        "link_mae": link_mae,
        "mae_gap": ignnk_mae - link_mae,
    }


def choose_nodes(
    truth: np.ndarray,
    ignnk: np.ndarray,
    link: np.ndarray,
    test_nodes: np.ndarray,
    metadata: list[dict[str, str]],
    min_gap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    wanted = {"onramp", "mainline"}
    for node in test_nodes:
        node_type = normalized_type(metadata[int(node)]["Type"])
        if node_type not in wanted:
            continue
        row = {
            "node_index": int(node),
            "station_id": metadata[int(node)].get("station_id", str(node)),
            "type": metadata[int(node)]["Type"],
            **node_metrics(truth, ignnk, link, int(node)),
        }
        if int(row["count"]) > 0 and np.isfinite(float(row["mae_gap"])):
            candidates.append(row)
    candidates.sort(key=lambda row: float(row["mae_gap"]), reverse=True)
    selected: list[dict[str, Any]] = []
    for node_type in ("onramp", "mainline"):
        typed = [row for row in candidates if normalized_type(str(row["type"])) == node_type]
        if not typed:
            raise RuntimeError(f"No valid test node with Type={node_type}")
        choice = typed[0]
        if float(choice["mae_gap"]) < min_gap:
            raise RuntimeError(
                f"Largest {node_type} MAE gap is {choice['mae_gap']:.4f}, "
                f"below --min-mae-gap={min_gap:.4f}"
            )
        selected.append(choice)
    return selected, candidates


def choose_common_window(
    truth: np.ndarray,
    ignnk: np.ndarray,
    link: np.ndarray,
    nodes: list[int],
    width: int,
    min_valid_ratio: float,
) -> tuple[int, int, float]:
    eligible: list[tuple[float, float, int, int]] = []
    fallback: list[tuple[float, float, int, int]] = []
    for start in range(0, truth.shape[0] - width + 1, width):
        end = start + width
        arrays = [truth[start:end, nodes], ignnk[start:end, nodes], link[start:end, nodes]]
        valid = (
            np.isfinite(arrays[0])
            & (arrays[0] > 0)
            & np.isfinite(arrays[1])
            & np.isfinite(arrays[2])
        )
        common = valid.all(axis=1)
        common_ratio = float(common.mean())
        if not common.any():
            continue
        variability = float(np.mean([
            np.std(arrays[0][:, column][valid[:, column]])
            for column in range(len(nodes))
        ]))
        item = (variability, common_ratio, start, end)
        fallback.append(item)
        if common_ratio >= min_valid_ratio:
            eligible.append(item)
    if eligible:
        # Window selection remains independent of either model's error: among
        # sufficiently complete windows, show the most variable true curves.
        best = max(eligible, key=lambda item: item[0])
    elif fallback:
        # Sparse missing observations can affect every full day. Fall back to
        # the best-covered day instead of requiring all 288 points to exist.
        best = max(fallback, key=lambda item: (item[1], item[0]))
        print(
            f"Warning: no window reached valid ratio {min_valid_ratio:.1%}; "
            f"using best available ratio {best[1]:.1%}.",
            file=sys.stderr,
        )
    else:
        raise RuntimeError(
            f"No {width}-point window contains any common valid prediction "
            "for both selected nodes"
        )
    return best[2], best[3], best[1]


def curve_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size < 2 or not np.isfinite(values).all():
        return {"std": None, "total_variation": None}
    return {
        "std": float(np.std(values)),
        "total_variation": float(np.mean(np.abs(np.diff(values)))),
    }


def safe_corr(first: np.ndarray, second: np.ndarray) -> float | None:
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def linear_interpolate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly fill internal gaps and use the nearest value at both edges."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not valid.any():
        raise ValueError("Cannot interpolate a curve without any valid values")
    positions = np.arange(values.size, dtype=np.float64)
    return np.interp(positions, positions[valid], values[valid]).astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(
    output_dir: Path,
    truth: np.ndarray,
    ignnk: np.ndarray,
    link: np.ndarray,
    selected: list[dict[str, Any]],
    start: int,
    end: int,
    start_date: datetime,
) -> dict[str, Any]:
    nodes = [int(row["node_index"]) for row in selected]
    timestamps = [start_date + timedelta(minutes=5 * index) for index in range(start, end)]
    colors = {"truth": "#222222", "ignnk": "#D55E00", "link": "#0072B2"}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True, constrained_layout=True)
    window_rows: list[dict[str, Any]] = []
    raw_curve_rows: list[dict[str, Any]] = []
    interpolated_curve_rows: list[dict[str, Any]] = []
    interpolated_truth: list[np.ndarray] = []
    interpolated_ignnk: list[np.ndarray] = []
    interpolated_link: list[np.ndarray] = []
    for axis, row, node in zip(axes, selected, nodes):
        target = truth[start:end, node].copy()
        baseline = ignnk[start:end, node].copy()
        ours = link[start:end, node].copy()
        target_valid = np.isfinite(target) & (target > 0)
        baseline_valid = np.isfinite(baseline)
        link_valid = np.isfinite(ours)
        valid = (
            target_valid & baseline_valid & link_valid
        )
        ignnk_window_mae = float(np.mean(np.abs(baseline[valid] - target[valid])))
        link_window_mae = float(np.mean(np.abs(ours[valid] - target[valid])))
        target_stats = curve_stats(target[valid])
        ignnk_stats = curve_stats(baseline[valid])
        link_stats = curve_stats(ours[valid])
        window_rows.append({
            **row,
            "window_start": timestamps[0].isoformat(),
            "window_end": timestamps[-1].isoformat(),
            "window_ignnk_mae": ignnk_window_mae,
            "window_link_mae": link_window_mae,
            "window_valid_points": int(valid.sum()),
            "window_valid_ratio": float(valid.mean()),
            "truth_std": target_stats["std"],
            "ignnk_std": ignnk_stats["std"],
            "link_std": link_stats["std"],
            "truth_total_variation": target_stats["total_variation"],
            "ignnk_total_variation": ignnk_stats["total_variation"],
            "link_total_variation": link_stats["total_variation"],
        })
        target_raw = target.copy()
        baseline_raw = baseline.copy()
        link_raw = ours.copy()
        target_raw[~target_valid] = np.nan
        baseline_raw[~baseline_valid] = np.nan
        link_raw[~link_valid] = np.nan
        target_plot = linear_interpolate(target_raw, target_valid)
        baseline_plot = linear_interpolate(baseline_raw, baseline_valid)
        link_plot = linear_interpolate(link_raw, link_valid)
        interpolated_truth.append(target_plot)
        interpolated_ignnk.append(baseline_plot)
        interpolated_link.append(link_plot)
        for offset, timestamp in enumerate(timestamps):
            identity = {
                "time_index": start + offset,
                "timestamp": timestamp.isoformat(),
                "node_index": node,
                "station_id": row["station_id"],
                "type": row["type"],
            }
            raw_curve_rows.append({
                **identity,
                "ground_truth": float(target_raw[offset]) if np.isfinite(target_raw[offset]) else "",
                "ignnk_no_metadata": (
                    float(baseline_raw[offset]) if np.isfinite(baseline_raw[offset]) else ""
                ),
                "link_metadata": (
                    float(link_raw[offset]) if np.isfinite(link_raw[offset]) else ""
                ),
                "common_valid": int(valid[offset]),
            })
            interpolated_curve_rows.append({
                **identity,
                "ground_truth": float(target_plot[offset]),
                "ignnk_no_metadata": float(baseline_plot[offset]),
                "link_metadata": float(link_plot[offset]),
                "ground_truth_interpolated": int(not target_valid[offset]),
                "ignnk_interpolated": int(not baseline_valid[offset]),
                "link_interpolated": int(not link_valid[offset]),
            })
        axis.plot(timestamps, target_plot, color=colors["truth"], lw=2.0, label="Ground truth")
        axis.plot(
            timestamps, baseline_plot, color=colors["ignnk"], lw=1.6,
            label=f"IGNNK w/o metadata (node MAE={row['ignnk_mae']:.2f})",
        )
        axis.plot(
            timestamps, link_plot, color=colors["link"], lw=1.7,
            label=f"LINK with metadata (node MAE={row['link_mae']:.2f})",
        )
        axis.set_title(
            f"{row['type']} · sensor {row['station_id']} · "
            f"MAE gap={row['mae_gap']:.2f}", loc="left", fontsize=11
        )
        axis.set_ylabel("Traffic flow")
        axis.grid(axis="y", color="#D9D9D9", lw=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper right")

    pair_valid = (
        np.isfinite(truth[start:end, nodes]).all(axis=1)
        & (truth[start:end, nodes] > 0).all(axis=1)
        & np.isfinite(ignnk[start:end, nodes]).all(axis=1)
        & np.isfinite(link[start:end, nodes]).all(axis=1)
    )
    truth_pair = truth[start:end, nodes][pair_valid]
    ignnk_pair = ignnk[start:end, nodes][pair_valid]
    link_pair = link[start:end, nodes][pair_valid]
    pair = {
        "common_valid_points": int(pair_valid.sum()),
        "common_valid_ratio": float(pair_valid.mean()),
        "truth_pair_correlation": safe_corr(truth_pair[:, 0], truth_pair[:, 1]),
        "ignnk_pair_correlation": safe_corr(ignnk_pair[:, 0], ignnk_pair[:, 1]),
        "link_pair_correlation": safe_corr(link_pair[:, 0], link_pair[:, 1]),
        "truth_pair_separation": float(np.mean(np.abs(truth_pair[:, 0] - truth_pair[:, 1]))),
        "ignnk_pair_separation": float(np.mean(np.abs(ignnk_pair[:, 0] - ignnk_pair[:, 1]))),
        "link_pair_separation": float(np.mean(np.abs(link_pair[:, 0] - link_pair[:, 1]))),
    }
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=4))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel(timestamps[0].strftime("%Y-%m-%d") + " (time of day)")
    fig.suptitle("Node-level traffic reconstruction comparison", fontsize=14, fontweight="bold")
    figure_stem = output_dir / "ignnk_vs_link_curves"
    fig.savefig(figure_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(figure_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_csv(output_dir / "selected_nodes.csv", window_rows)
    raw_csv = output_dir / "selected_curves_raw.csv"
    interpolated_csv = output_dir / "selected_curves_interpolated.csv"
    interpolated_wide_csv = output_dir / "selected_curves_interpolated_wide.csv"
    interpolated_npz = output_dir / "selected_curves_interpolated.npz"
    write_csv(raw_csv, raw_curve_rows)
    write_csv(interpolated_csv, interpolated_curve_rows)
    wide_rows: list[dict[str, Any]] = []
    for offset, timestamp in enumerate(timestamps):
        wide_row: dict[str, Any] = {
            "time_index": start + offset,
            "timestamp": timestamp.isoformat(),
        }
        for column, row in enumerate(selected):
            prefix = f"{normalized_type(str(row['type']))}_{row['station_id']}"
            wide_row[f"{prefix}_ground_truth"] = float(interpolated_truth[column][offset])
            wide_row[f"{prefix}_ignnk_no_metadata"] = float(interpolated_ignnk[column][offset])
            wide_row[f"{prefix}_link_metadata"] = float(interpolated_link[column][offset])
        wide_rows.append(wide_row)
    write_csv(interpolated_wide_csv, wide_rows)
    np.savez_compressed(
        interpolated_npz,
        time_index=np.arange(start, end, dtype=np.int64),
        timestamp=np.asarray([value.isoformat() for value in timestamps]),
        node_index=np.asarray(nodes, dtype=np.int64),
        station_id=np.asarray([str(row["station_id"]) for row in selected]),
        node_type=np.asarray([str(row["type"]) for row in selected]),
        ground_truth=np.stack(interpolated_truth, axis=1),
        ignnk_no_metadata=np.stack(interpolated_ignnk, axis=1),
        link_metadata=np.stack(interpolated_link, axis=1),
    )
    return {
        "window_nodes": window_rows,
        "pair_comparison": pair,
        "curve_exports": {
            "raw_csv": str(raw_csv),
            "interpolated_csv": str(interpolated_csv),
            "interpolated_wide_csv": str(interpolated_wide_csv),
            "interpolated_npz": str(interpolated_npz),
        },
    }


def main() -> None:
    args = parse_args()
    args.pure_data_root = args.pure_data_root.resolve()
    args.output_dir = args.output_dir.resolve() / f"pems{args.district:02d}_topk{args.top_k}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pure_dir = args.pure_data_root / f"pems{args.district:02d}"
    dataset_dir = (
        STAGANN_DIR / "datasets"
        / f"pems{args.district:02d}_metadata_topk{args.top_k}_71_24"
    )
    required = [
        pure_dir / f"district_{args.district}_flow.npy",
        pure_dir / f"district_{args.district}_meta_feature.csv",
        pure_dir / "adj_index.pkl",
        args.link_completion,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing curve-analysis inputs: " + ", ".join(missing))

    truth = squeeze_flow(
        np.load(pure_dir / f"district_{args.district}_flow.npy"), "ground truth"
    )
    link = squeeze_flow(np.load(args.link_completion), "LINK completion")
    if link.shape != truth.shape:
        raise ValueError(f"LINK completion shape {link.shape} != truth {truth.shape}")

    inference_info: dict[str, Any]
    if args.ignnk_prediction is not None:
        ignnk = squeeze_flow(np.load(args.ignnk_prediction), "IGNNK prediction")
        inference_info = {"prediction_source": str(args.ignnk_prediction.resolve())}
    else:
        checkpoint = resolve_checkpoint(args)
        device = torch.device(args.device)
        ignnk, inference_info = infer_ignnk(
            dataset_dir, checkpoint, device, args.batch_size
        )
        prediction_path = args.output_dir / "ignnk_no_metadata_predictions.npy"
        np.save(prediction_path, ignnk)
        inference_info["prediction_source"] = str(prediction_path)
    if ignnk.shape != truth.shape:
        raise ValueError(f"IGNNK prediction shape {ignnk.shape} != truth {truth.shape}")

    split = load_pickle(pure_dir / "adj_index.pkl")
    test_nodes = np.asarray(split["test_nodes"], dtype=np.int64)
    metadata = read_metadata(pure_dir / f"district_{args.district}_meta_feature.csv")
    selected, candidates = choose_nodes(
        truth, ignnk, link, test_nodes, metadata, args.min_mae_gap
    )
    write_csv(args.output_dir / "candidate_node_ranking.csv", candidates)
    nodes = [int(row["node_index"]) for row in selected]
    start, end, window_valid_ratio = choose_common_window(
        truth, ignnk, link, nodes, args.window_points,
        args.min_window_valid_ratio,
    )
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    plot_info = plot_curves(
        args.output_dir, truth, ignnk, link, selected, start, end, start_date
    )
    summary = {
        "district": args.district,
        "top_k": args.top_k,
        "selection_rule": "largest full-test-period IGNNK_MAE_minus_LINK_MAE within each Type",
        "window_rule": (
            "highest-ground-truth-variability window meeting the common-valid-ratio "
            "threshold; best-covered window used as fallback"
        ),
        "window_indices": [start, end],
        "window_min_node_valid_ratio": window_valid_ratio,
        "link_completion": str(args.link_completion.resolve()),
        **inference_info,
        **plot_info,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"PNG: {args.output_dir / 'ignnk_vs_link_curves.png'}")
    print(f"PDF: {args.output_dir / 'ignnk_vs_link_curves.pdf'}")


if __name__ == "__main__":
    main()
