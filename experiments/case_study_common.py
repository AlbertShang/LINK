"""Shared prediction export and selection utilities for node-level case studies.

The public scripts are :mod:`case_study_day` and :mod:`case_study_scale`.
This module keeps every prediction on the original ``[time, node]`` axis so
curves produced by different baseline runners can be compared point by point.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STAGANN_DIR = ROOT / "baselines" / "STAGANN"
KITS_DIR = ROOT / "baselines" / "KITS"
BASELINE_NAMES = {
    "dualstn": "DualSTN",
    "gcn": "GCN",
    "gin": "GIN",
    "grin": "GRIN",
    "iagcn": "IAGCN",
    "ignnk": "IGNNK",
    "increase": "INCREASE",
    "okriging": "Okriging",
    "satcn": "SATCN",
    "stagann": "STAGANN",
}
SPECIAL_NAMES = {"link": "LINK", "kits": "KITS"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    mode: str

    @property
    def key(self) -> str:
        return f"{self.name}:{self.mode}"

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", self.key).strip("_").lower()

    @property
    def label(self) -> str:
        pretty = "STA-GANN" if self.name == "STAGANN" else self.name
        suffix = "Meta" if self.mode == "metadata" else "NoMeta"
        return f"{pretty} ({suffix})"


@dataclass
class StudyContext:
    district: int
    top_k: int
    truth: np.ndarray
    adjacency: np.ndarray
    train_nodes: np.ndarray
    valid_nodes: np.ndarray
    test_nodes: np.ndarray
    metadata_rows: list[dict[str, str]]
    metadata_features: np.ndarray
    pure_dir: Path
    converted_dir: Path


def parse_model_spec(value: str) -> ModelSpec:
    raw_name, separator, raw_mode = value.partition(":")
    normalized = raw_name.strip().lower().replace("-", "").replace("_", "")
    if normalized in BASELINE_NAMES:
        name = BASELINE_NAMES[normalized]
    elif normalized in SPECIAL_NAMES:
        name = SPECIAL_NAMES[normalized]
    else:
        supported = ", ".join((*SPECIAL_NAMES.values(), *BASELINE_NAMES.values()))
        raise argparse.ArgumentTypeError(f"Unknown model {raw_name!r}; choose from {supported}")
    mode = raw_mode.strip().lower().replace("-", "_") if separator else "metadata"
    if mode not in {"metadata", "no_metadata"}:
        raise argparse.ArgumentTypeError(
            f"Invalid mode {mode!r} in {value!r}; use metadata or no_metadata"
        )
    return ModelSpec(name, mode)


def parse_prediction_override(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction overrides use MODEL:MODE=/path/file.npy")
    token, path = value.split("=", 1)
    return parse_model_spec(token).key, Path(path).expanduser()


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--district", type=int, default=3, choices=(3, 4, 7, 8))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--pure-data-root", type=Path, default=ROOT / "pure_dataset_topk")
    parser.add_argument(
        "--converted-data-root", type=Path,
        default=STAGANN_DIR / "datasets",
        help="root containing pemsXX_metadata_topkK_71_24",
    )
    parser.add_argument(
        "--prediction-cache", type=Path,
        default=ROOT / "outputs" / "case_study_predictions",
    )
    parser.add_argument(
        "--prediction", action="append", default=[], type=parse_prediction_override,
        metavar="MODEL:MODE=FILE",
        help="reuse a precomputed aligned [time,node] prediction; repeat as needed",
    )
    parser.add_argument(
        "--checkpoint", action="append", default=[], type=parse_prediction_override,
        metavar="MODEL:MODE=FILE",
        help="override a model checkpoint; repeat as needed",
    )
    parser.add_argument("--refresh-predictions", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-date", default="2023-01-01")


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


def _metadata_csv(pure_dir: Path, district: int) -> Path:
    candidates = [
        pure_dir / f"district_{district}_metadata.csv",
        pure_dir / f"district_{district}_node_metadata.csv",
    ]
    candidates.extend(sorted(pure_dir.glob("*.csv")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find node metadata CSV in {pure_dir}")


def read_metadata_rows(path: Path, node_count: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if len(rows) != node_count:
        raise ValueError(f"{path} has {len(rows)} rows but flow has {node_count} nodes")
    return rows


def load_context(args: argparse.Namespace) -> StudyContext:
    pure_dir = Path(args.pure_data_root) / f"pems{args.district:02d}"
    truth = squeeze_flow(
        np.load(pure_dir / f"district_{args.district}_flow.npy"), "ground truth"
    )
    adjacency = np.load(
        pure_dir / f"district_{args.district}_adj_inout_topk{args.top_k}.npy"
    ).astype(np.float32, copy=False)
    split = load_pickle(pure_dir / "adj_index.pkl")
    converted_dir = (
        Path(args.converted_data_root)
        / f"pems{args.district:02d}_metadata_topk{args.top_k}_71_{args.window}"
    )
    metadata_payload = load_pickle(converted_dir / "metadata.pkl")
    metadata_features = np.asarray(metadata_payload["node_features"], dtype=np.float32)
    rows = read_metadata_rows(_metadata_csv(pure_dir, args.district), truth.shape[1])
    if adjacency.shape != (truth.shape[1], truth.shape[1]):
        raise ValueError(f"Adjacency {adjacency.shape} does not match truth {truth.shape}")
    if metadata_features.shape[0] != truth.shape[1]:
        raise ValueError("Converted metadata features are not aligned with the pure flow")
    return StudyContext(
        district=args.district,
        top_k=args.top_k,
        truth=truth,
        adjacency=adjacency,
        train_nodes=np.asarray(split["train_nodes"], dtype=np.int64),
        valid_nodes=np.asarray(split["valid_nodes"], dtype=np.int64),
        test_nodes=np.asarray(split["test_nodes"], dtype=np.int64),
        metadata_rows=rows,
        metadata_features=metadata_features,
        pure_dir=pure_dir,
        converted_dir=converted_dir,
    )


def _load_state(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected checkpoint payload in {path}: {type(payload).__name__}")
    state = payload.get("model_state_dict", payload.get("model", payload))
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint has no model state dictionary: {path}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state, payload


def _latest(paths: Iterable[Path], description: str) -> Path:
    candidates = [path for path in paths if path.is_file()]
    if not candidates:
        raise FileNotFoundError(description)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint(spec: ModelSpec, context: StudyContext, override: Path | None) -> Path:
    if override is not None:
        path = override.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint override does not exist: {path}")
        return path
    if spec.name == "KITS":
        preferred = KITS_DIR / "results" / f"pems{context.district:02d}_topk{context.top_k}" / spec.mode / "best.pt"
        candidates = [preferred]
        candidates.extend(KITS_DIR.glob(
            f"**/pems{context.district:02d}_topk{context.top_k}/{spec.mode}/best.pt"
        ))
        return _latest(candidates, f"Cannot find KITS {spec.mode} checkpoint; pass --checkpoint")
    root = (
        STAGANN_DIR / "checkpoints"
        / f"pems{context.district:02d}_metadata_topk{context.top_k}_71_24"
        / spec.mode
    )
    candidates = list(root.glob(f"{spec.name}_*/**/{spec.name}_best_val_MAE.pt"))
    candidates.extend(root.glob(f"{spec.name}_*/**/*best*MAE*.pt"))
    return _latest(
        candidates,
        f"Cannot find {spec.key} best-val-MAE checkpoint under {root}; pass --checkpoint",
    )


def _normalize_output_shape(output: torch.Tensor, window: int, nodes: int) -> torch.Tensor:
    if output.ndim == 4 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 3:
        raise ValueError(f"Expected prediction [batch,time,node], got {tuple(output.shape)}")
    if output.shape[1:] == (nodes, window):
        output = output.transpose(1, 2)
    if output.shape[1:] != (window, nodes):
        raise ValueError(
            f"Prediction shape {tuple(output.shape)} does not match window={window}, nodes={nodes}"
        )
    return output


def infer_stagann_family(
    spec: ModelSpec,
    context: StudyContext,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if str(STAGANN_DIR) not in sys.path:
        sys.path.insert(0, str(STAGANN_DIR))
    from examples.pems_metadata_specs import registered_baseline_spec
    from stkriging.archs.arch_zoo.metadata_arch import METADATA_ARCHITECTURES
    from stkriging.utils import adj_transform

    architecture, model_spec = registered_baseline_spec(spec.name, context.district)
    wrapper = METADATA_ARCHITECTURES[architecture]
    model = wrapper(
        backbone_parameters=model_spec["model_parameters"],
        metadata_dim=context.metadata_features.shape[1],
        dynamic_feature_dim=3,
        metadata_embedding_dim=model_spec.get("metadata_embedding_dim", 20),
        fusion_hidden_dim=32,
        split_adjacency=model_spec.get("split_adjacency", False),
    ).to(device)
    state, payload = _load_state(checkpoint, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    data_payload = load_pickle(context.converted_dir / "data.pkl")
    processed = np.asarray(data_payload["processed_data"], dtype=np.float32)
    indices = load_pickle(context.converted_dir / "index.pkl")
    scaler = load_pickle(context.converted_dir / "scaler.pkl")
    node_order = np.concatenate((context.train_nodes, context.test_nodes))
    unknown = np.arange(len(context.train_nodes), len(node_order), dtype=np.int64)
    raw_adjacency = context.adjacency[np.ix_(node_order, node_order)]
    transformed, _ = adj_transform(raw_adjacency, model_spec.get("matrix_transform", "original"))
    adjacency = torch.as_tensor(
        np.asarray(transformed[0], dtype=np.float32), device=device
    )
    static = context.metadata_features[node_order]
    if spec.mode == "no_metadata":
        static = np.zeros_like(static)
    windows = [tuple(map(int, item)) for item in indices["test"]]
    predictions = np.full(context.truth.shape, np.nan, dtype=np.float32)
    sums = np.zeros((context.truth.shape[0], len(context.test_nodes)), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int32)
    mean = float(np.asarray(scaler["args"]["mean"]).reshape(-1)[0])
    std = float(np.asarray(scaler["args"]["std"]).reshape(-1)[0])

    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            batch_windows = windows[offset:offset + batch_size]
            # STA-GANN historically used a bare squeeze() while constructing
            # dynamic graphs.  Keep a defensive two-sample batch for old
            # checkouts whose model file has not yet received squeeze(-1).
            inference_windows = batch_windows
            if spec.name == "STAGANN" and len(batch_windows) == 1:
                inference_windows = [batch_windows[0], batch_windows[0]]
            dynamic = np.stack(
                [processed[start:end, node_order] for start, end in inference_windows], axis=0
            )
            repeated_static = np.broadcast_to(
                static[None, None, :, :],
                (len(inference_windows), dynamic.shape[1], len(node_order), static.shape[1]),
            )
            model_input = np.concatenate(
                (dynamic[..., :1], repeated_static, dynamic[..., 1:]), axis=-1
            ).astype(np.float32, copy=False)
            model_input[..., unknown, 0] = 0.0
            batch_unknown = np.broadcast_to(
                unknown[None], (len(inference_windows), len(unknown))
            ).copy()
            result = model(
                X=torch.from_numpy(model_input).to(device),
                adj=adjacency,
                unknown_nodes=batch_unknown,
                location=None,
                batch_seen=None,
                epoch=1,
                train=False,
            )
            output = result[0] if isinstance(result, (list, tuple)) else result
            output = _normalize_output_shape(output, dynamic.shape[1], len(node_order))
            output = output[..., unknown].detach().cpu().numpy() * std + mean
            for row, (start, end) in enumerate(batch_windows):
                sums[start:end] += output[row]
                counts[start:end] += 1

    valid = counts > 0
    heldout = np.full_like(sums, np.nan, dtype=np.float32)
    heldout[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    predictions[:, context.test_nodes] = heldout
    return predictions, {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "test_windows": len(windows),
    }


def _fill_nonfinite(flow: np.ndarray) -> np.ndarray:
    if np.isfinite(flow).all():
        return flow
    flow = flow.copy()
    time = np.arange(flow.shape[0])
    for node in range(flow.shape[1]):
        valid = np.isfinite(flow[:, node])
        if not valid.any():
            raise ValueError(f"Node {node} has no finite flow values")
        flow[~valid, node] = np.interp(time[~valid], time[valid], flow[valid, node])
    return flow


def infer_kits(
    spec: ModelSpec,
    context: StudyContext,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    window: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if str(KITS_DIR) not in sys.path:
        sys.path.insert(0, str(KITS_DIR))
    from lib.nn.models import KITS

    flow = _fill_nonfinite(context.truth.astype(np.float32, copy=True))
    count = len(flow) - window + 1
    train_count = round(count * 0.7)
    training_end = train_count - 1 + window
    mean = float(flow[:training_end].mean())
    std = float(flow[:training_end].std())
    values = ((flow - mean) / std).astype(np.float32)
    node_order = np.concatenate((context.train_nodes, context.test_nodes))
    observed_count = len(context.train_nodes)
    adjacency = context.adjacency[np.ix_(node_order, node_order)]
    state, payload = _load_state(checkpoint, device)
    has_metadata_projection = any(
        str(name).startswith("metadata_projection.") for name in state
    )
    checkpoint_metadata_dim = int(payload.get(
        "metadata_dim",
        context.metadata_features.shape[1] if has_metadata_projection else 0,
    ))
    model_args = SimpleNamespace(
        dataset_name=f"pems{context.district:02d}_pure",
        use_adj_drop=False,
        use_init=False,
        metadata_dim=checkpoint_metadata_dim,
        metadata_embedding_dim=(
            int(state["metadata_projection.0.weight"].shape[0])
            if has_metadata_projection else 20
        ),
    )
    hidden_dim = int(state["fc_1.weight"].shape[0])
    model = KITS(context.adjacency, d_in=1, d_hidden=hidden_dim, args=model_args).to(device)
    model.load_state_dict(state, strict=True)
    model.adj = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    model.eval()
    metadata = None
    if checkpoint_metadata_dim:
        static = context.metadata_features[node_order]
        if spec.mode == "no_metadata":
            static = np.zeros_like(static)
        if static.shape[1] != checkpoint_metadata_dim:
            raise ValueError(
                f"KITS checkpoint metadata_dim={checkpoint_metadata_dim}, data has {static.shape[1]}"
            )
        metadata = torch.from_numpy(static).to(device)

    starts = list(range(train_count, count))
    predictions = np.full(context.truth.shape, np.nan, dtype=np.float32)
    sums = np.zeros((context.truth.shape[0], len(context.test_nodes)), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int32)
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            batch = np.stack(
                [values[start:start + window, node_order] for start in batch_starts], axis=0
            )[..., None]
            tensor = torch.from_numpy(batch).to(device)
            mask = torch.zeros_like(tensor, dtype=torch.bool)
            observed = tensor[:, :, :observed_count] * std + mean
            mask[:, :, :observed_count] = observed.abs() > 5e-5
            output = model(torch.where(mask, tensor, 0.0), mask=mask, metadata=metadata)
            output = output[:, :, observed_count:, 0].cpu().numpy() * std + mean
            for row, start in enumerate(batch_starts):
                end = start + window
                sums[start:end] += output[row]
                counts[start:end] += 1
    valid = counts > 0
    heldout = np.full_like(sums, np.nan, dtype=np.float32)
    heldout[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    predictions[:, context.test_nodes] = heldout
    return predictions, {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "evaluation_windows": len(starts),
    }


def _default_link_prediction(spec: ModelSpec, context: StudyContext) -> Path:
    if spec.mode == "metadata":
        candidates = [
            ROOT / "outputs" / f"pems{context.district:02d}" / "completed_flow.npy",
            ROOT / "outputs" / f"PEMS{context.district:02d}" / "completed_flow.npy",
        ]
    else:
        candidates = [
            ROOT / "outputs" / "ablation_no_metadata" / f"pems{context.district:02d}" / "completed_flow.npy",
            ROOT / "outputs" / "no_metadata" / f"pems{context.district:02d}" / "completed_flow.npy",
        ]
    return _latest(candidates, f"Cannot find {spec.key} completed_flow.npy; pass --prediction")


def load_predictions(
    specs: list[ModelSpec], args: argparse.Namespace, context: StudyContext
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("--models contains a duplicate MODEL:MODE entry")
    prediction_overrides = dict(args.prediction)
    checkpoint_overrides = dict(args.checkpoint)
    cache_dir = (
        Path(args.prediction_cache)
        / f"pems{context.district:02d}_topk{context.top_k}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    predictions: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for spec in specs:
        override = prediction_overrides.get(spec.key)
        cache = cache_dir / f"{spec.slug}.npy"
        source: Path | None = None
        details: dict[str, Any] = {}
        if override is not None:
            source = override.resolve()
        elif cache.is_file() and not args.refresh_predictions:
            source = cache
        elif spec.name == "LINK":
            source = _default_link_prediction(spec, context)
        if source is not None:
            if not source.is_file():
                raise FileNotFoundError(f"Prediction file does not exist: {source}")
            prediction = squeeze_flow(np.load(source), f"{spec.key} prediction")
            details = {"prediction_source": str(source)}
        else:
            checkpoint = resolve_checkpoint(
                spec, context, checkpoint_overrides.get(spec.key)
            )
            print(f"Exporting aligned prediction for {spec.key} from {checkpoint}", flush=True)
            if spec.name == "KITS":
                prediction, details = infer_kits(
                    spec, context, checkpoint, device,
                    args.inference_batch_size, args.window,
                )
            else:
                prediction, details = infer_stagann_family(
                    spec, context, checkpoint, device, args.inference_batch_size
                )
            np.save(cache, prediction)
            details["prediction_cache"] = str(cache)
        if prediction.shape != context.truth.shape:
            raise ValueError(
                f"{spec.key} prediction {prediction.shape} != truth {context.truth.shape}"
            )
        predictions[spec.key] = prediction
        provenance[spec.key] = details
    return predictions, provenance


def canonical_type(row: dict[str, str]) -> str:
    value = next((row.get(key, "") for key in ("Type", "type", "TYPE") if key in row), "")
    return re.sub(r"[^a-z]", "", value.lower())


def row_value(row: dict[str, str], *keys: str) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for key in keys:
        if key.lower() in lowered and str(lowered[key.lower()]).strip():
            return str(lowered[key.lower()]).strip()
    return ""


def station_id(row: dict[str, str], node: int) -> str:
    return row_value(row, "station_id", "station", "sensor_id", "id") or str(node)


def latitude_longitude(row: dict[str, str]) -> tuple[float, float]:
    try:
        return float(row_value(row, "lat", "latitude")), float(row_value(row, "lng", "lon", "longitude"))
    except ValueError:
        return float("nan"), float("nan")


def common_valid_mask(
    truth: np.ndarray, predictions: dict[str, np.ndarray], nodes: Iterable[int]
) -> np.ndarray:
    nodes = np.asarray(list(nodes), dtype=np.int64)
    valid = np.isfinite(truth[:, nodes]) & (np.abs(truth[:, nodes]) > 5e-5)
    for prediction in predictions.values():
        valid &= np.isfinite(prediction[:, nodes])
    return valid


def interpolate_series(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    keep = np.isfinite(result)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool)
    x = np.arange(len(result))
    if not keep.any():
        return np.full_like(result, np.nan)
    result[~keep] = np.interp(x[~keep], x[keep], result[keep])
    return result


def choose_day_window(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    nodes: list[int],
    width: int,
    min_valid_ratio: float,
    day: str | None,
    start_date: str,
) -> tuple[int, int, str]:
    if day:
        origin = datetime.fromisoformat(start_date)
        selected = datetime.fromisoformat(day)
        start = int((selected - origin).days * width)
        end = start + width
        if start < 0 or end > truth.shape[0]:
            raise ValueError(f"Requested day {day} is outside the available time axis")
        return start, end, "explicit"
    common = common_valid_mask(truth, predictions, nodes)
    best: tuple[float, int, int] | None = None
    for start in range(0, truth.shape[0] - width + 1, width):
        end = start + width
        ratio = float(common[start:end].mean())
        if ratio < min_valid_ratio:
            continue
        variability = float(np.nanmean([
            np.nanstd(truth[start:end, node][common[start:end, idx]])
            for idx, node in enumerate(nodes)
        ]))
        score = variability + 1e-6 * ratio
        if best is None or score > best[0]:
            best = (score, start, end)
    if best is None:
        maximum = 0.0
        for start in range(0, truth.shape[0] - width + 1, width):
            maximum = max(maximum, float(common[start:start + width].mean()))
        raise RuntimeError(
            f"No common {width}-point day reaches valid ratio {min_valid_ratio:.2f}; "
            f"best available ratio is {maximum:.3f}. Lower --min-day-valid-ratio or pass --day."
        )
    return best[1], best[2], "truth_variability"


def timestamps(start: int, end: int, start_date: str, minutes: int = 5) -> list[datetime]:
    origin = datetime.fromisoformat(start_date)
    return [origin + timedelta(minutes=minutes * index) for index in range(start, end)]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
