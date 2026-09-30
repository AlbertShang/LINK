"""Train and evaluate KITS on the shared pure PEMS Top-K graph datasets.

Run from any directory, for example::

    python baselines/KITS/run_pure.py --districts 3 4 7 8 --top-k 3

The temporal windows and node splits match the STAGANN baseline adapter:
24-step windows, the first 70% of windows for training, and the remaining
windows for both validation and test on disjoint held-out node sets. By
default, each district is run once without and once with node metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from lib.nn.models import KITS


KITS_DIR = Path(__file__).resolve().parent
PURE_ROOT = KITS_DIR.parents[1] / "pure_dataset_topk"
DISTRICTS = (3, 4, 7, 8)
MODES = ("no_metadata", "metadata")
TYPE_CATEGORIES = ("mainline", "onramp", "offramp", "hov", "other")
DIRECTION_CATEGORIES = ("n", "s", "e", "w", "ne", "nw", "se", "sw", "other")


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, nodes: np.ndarray, starts: range, window: int):
        self.values = torch.from_numpy(values)
        self.nodes = torch.as_tensor(nodes, dtype=torch.long)
        self.starts = starts
        self.window = window

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = self.starts[index]
        return self.values[start:start + self.window].index_select(1, self.nodes)


def load_pure_dataset(root: Path, district: int, top_k: int):
    directory = root / f"pems{district:02d}"
    flow_path = directory / f"district_{district}_flow.npy"
    adj_path = directory / f"district_{district}_adj_inout_topk{top_k}.npy"
    split_path = directory / "adj_index.pkl"
    for path in (flow_path, adj_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing pure dataset input: {path}")

    flow = np.load(flow_path)
    if flow.ndim == 3 and flow.shape[-1] == 1:
        flow = flow[..., 0]
    if flow.ndim != 2:
        raise ValueError(f"Expected single-channel [time, node] flow, got {flow.shape}")
    flow = flow.astype(np.float32, copy=False)
    # The STAGANN adapter interpolates each node along time before scaling.
    if not np.isfinite(flow).all():
        flow = flow.copy()
        time = np.arange(flow.shape[0])
        for node in range(flow.shape[1]):
            valid = np.isfinite(flow[:, node])
            if not valid.any():
                raise ValueError(f"Node {node} has no finite flow values")
            if not valid.all():
                flow[~valid, node] = np.interp(time[~valid], time[valid], flow[valid, node])

    adj = np.load(adj_path).astype(np.float32, copy=False)
    if adj.shape != (flow.shape[1], flow.shape[1]):
        raise ValueError(f"Adjacency {adj.shape} does not match {flow.shape[1]} nodes")
    if not np.isfinite(adj).all() or (adj < 0).any():
        raise ValueError("Adjacency must contain finite, nonnegative weights")
    adj = adj.copy()
    np.fill_diagonal(adj, 0.0)

    with split_path.open("rb") as file:
        split = pickle.load(file)
    keys = ("train_nodes", "valid_nodes", "test_nodes")
    if not isinstance(split, dict) or any(key not in split for key in keys):
        raise ValueError(f"{split_path} must contain {keys}")
    nodes = {key: np.asarray(split[key], dtype=np.int64) for key in keys}
    for key, values in nodes.items():
        if values.ndim != 1 or not len(values):
            raise ValueError(f"{key} must be a nonempty one-dimensional array")
    partition = np.concatenate(list(nodes.values()))
    if (len(partition) != flow.shape[1] or np.unique(partition).size != flow.shape[1]
            or partition.min() < 0 or partition.max() >= flow.shape[1]):
        raise ValueError("Node split must partition the flow columns exactly once")
    return flow, adj, nodes


def normalize_freeway(value) -> str:
    """Use the same canonical freeway IDs as the STAGANN pure-data adapter."""
    if value is None or not str(value).strip():
        return ""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def load_metadata_features(root: Path, district: int, node_count: int,
                           train_nodes: np.ndarray) -> np.ndarray:
    """Encode canonical type, direction and train-vocabulary freeway features."""
    path = root / f"pems{district:02d}" / f"district_{district}_node_metadata.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing pure node metadata: {path}")
    required = {
        "type_code", "direction_code", "type_feature", "direction_feature",
        "freeway", "type_categories", "direction_categories",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"Node metadata is missing {sorted(missing)}: {path}")
        type_code = archive["type_code"]
        direction_code = archive["direction_code"]
        type_feature = archive["type_feature"].astype(np.float32, copy=False)
        direction_feature = archive["direction_feature"].astype(np.float32, copy=False)
        freeways = [normalize_freeway(value) for value in archive["freeway"]]
        type_categories = tuple(str(value) for value in archive["type_categories"])
        direction_categories = tuple(str(value) for value in archive["direction_categories"])
    if type_categories != TYPE_CATEGORIES or direction_categories != DIRECTION_CATEGORIES:
        raise ValueError(f"Unexpected canonical metadata vocabulary in {path}")
    if (type_code.shape != (node_count,) or direction_code.shape != (node_count,)
            or type_feature.shape != (node_count, len(TYPE_CATEGORIES))
            or direction_feature.shape != (node_count, len(DIRECTION_CATEGORIES))
            or len(freeways) != node_count):
        raise ValueError(f"Node metadata does not align with {node_count} flow nodes")
    if (not np.array_equal(type_feature.argmax(axis=1), type_code)
            or not np.array_equal(direction_feature.argmax(axis=1), direction_code)
            or not np.allclose(type_feature.sum(axis=1), 1.0)
            or not np.allclose(direction_feature.sum(axis=1), 1.0)):
        raise ValueError(f"Invalid one-hot type/direction features in {path}")

    # The held-out node freeways must not enter the categorical vocabulary.
    train_freeways = sorted({freeways[index] for index in train_nodes if freeways[index]})
    freeway_to_index = {value: index for index, value in enumerate(train_freeways)}
    freeway_feature = np.zeros((node_count, len(train_freeways) + 1), dtype=np.float32)
    unknown = len(train_freeways)
    for node, freeway in enumerate(freeways):
        freeway_feature[node, freeway_to_index.get(freeway, unknown)] = 1.0
    features = np.concatenate(
        (type_feature, direction_feature, freeway_feature), axis=1,
    ).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite node metadata in {path}")
    return features


def make_window_split(time_steps: int, window: int):
    count = time_steps - window + 1
    if count < 2:
        raise ValueError("Not enough time steps for both training and evaluation windows")
    train_count = round(count * 0.7)
    if not 0 < train_count < count:
        raise ValueError("70/30 temporal split produced an empty partition")
    return range(train_count), range(train_count, count)


def make_loader(values, nodes, starts, window, batch_size, samples_per_epoch=0):
    dataset = WindowDataset(values, nodes, starts, window)
    if samples_per_epoch:
        sampler = RandomSampler(dataset, replacement=True, num_samples=samples_per_epoch)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def training_step(model, batch, train_nodes, total_nodes, mean, std,
                  train_metadata=None):
    """KITS's observed-node and virtual-node reconstruction/cycle losses."""
    batch = batch.unsqueeze(-1)
    b, s, n, d = batch.shape
    ratio = n / total_nodes
    dynamic_ratio = ratio + 0.2 * np.random.random()
    virtual_count = int(n / dynamic_ratio) - n
    if virtual_count <= 0:
        raise ValueError("KITS needs at least one virtual node; check the node split")
    virtual = batch.new_zeros((b, s, virtual_count, d))
    valid_observations = (batch * std + mean).abs() > 5e-5
    x = torch.cat((torch.where(valid_observations, batch, 0.0), virtual), dim=2)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :, :n] = valid_observations
    metadata = None
    if train_metadata is not None:
        metadata = torch.cat((
            train_metadata,
            train_metadata.new_zeros((virtual_count, train_metadata.size(1))),
        ), dim=0)
    imputation, imputation_cyc, target_cyc = model(
        x, mask=mask, known_set=train_nodes.tolist(),
        sub_entry_num=virtual_count, reset=True, metadata=metadata,
    )
    if not valid_observations.any():
        raise ValueError("Training batch has no nonzero observed flow values")
    reconstruction = F.l1_loss(
        imputation[:, :, :n][valid_observations], batch[valid_observations],
    )
    return reconstruction + F.l1_loss(imputation_cyc, target_cyc)


def train_one_epoch(model, train_loader, optimizer, train_nodes, total_nodes,
                    mean, std, train_metadata, device, epoch):
    """Run exactly the training updates used by the regular KITS runner."""
    model.train()
    train_loss = 0.0
    steps = 0
    for batch in train_loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = training_step(
            model, batch, train_nodes, total_nodes, mean, std, train_metadata,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite KITS training loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        steps += 1
    if not steps:
        raise RuntimeError("KITS training epoch has no optimizer steps")
    return train_loss / steps, steps


@torch.no_grad()
def evaluate(model, loader, observed_count, mean, std, device, metadata=None):
    model.eval()
    absolute = squared = percentage = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True).unsqueeze(-1)
        mask = torch.zeros_like(batch, dtype=torch.bool)
        mask[:, :, :observed_count] = (
            (batch[:, :, :observed_count] * std + mean).abs() > 5e-5
        )
        predictions = model(torch.where(mask, batch, 0.0), mask=mask,
                            metadata=metadata)
        target = batch[:, :, observed_count:] * std + mean
        predictions = predictions[:, :, observed_count:] * std + mean
        valid = torch.isfinite(target) & (target.abs() > 5e-5)
        error = (predictions - target)[valid]
        reference = target[valid]
        absolute += error.abs().double().sum().item()
        squared += error.square().double().sum().item()
        percentage += (error.abs() / reference.abs()).double().sum().item()
        count += int(valid.sum().item())
    if count == 0:
        raise ValueError("No nonzero held-out targets available for evaluation")
    scores = {
        "MAE": absolute / count,
        "RMSE": (squared / count) ** 0.5,
        "MAPE": 100.0 * percentage / count,
        "count": count,
    }
    if not all(np.isfinite(scores[key]) for key in ("MAE", "RMSE", "MAPE")):
        raise FloatingPointError("KITS produced non-finite held-out metrics")
    return scores


@torch.inference_mode()
def benchmark_inference(model, loader, observed_count, mean, std, device,
                        metadata=None):
    """Time the complete test-loader forward loop without metric reduction."""
    model.eval()
    steps = samples = 0
    target_count_tensor = torch.zeros((), dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        samples += int(batch.shape[0])
        batch = batch.to(device, non_blocking=True).unsqueeze(-1)
        mask = torch.zeros_like(batch, dtype=torch.bool)
        mask[:, :, :observed_count] = (
            (batch[:, :, :observed_count] * std + mean).abs() > 5e-5
        )
        predictions = model(
            torch.where(mask, batch, 0.0), mask=mask, metadata=metadata
        )
        target = batch[:, :, observed_count:] * std + mean
        target_count_tensor += (
            torch.isfinite(target) & (target.abs() > 5e-5)
        ).sum()
        # Materialize the held-out output so all indexed inference work is
        # included before the synchronization boundary.
        _ = predictions[:, :, observed_count:]
        steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    target_count = int(target_count_tensor.item())
    if not steps or target_count <= 0:
        raise RuntimeError("KITS inference benchmark has no valid test targets")
    return {
        "seconds": seconds,
        "steps": steps,
        "samples": samples,
        "target_count": target_count,
        "batch_size": loader.batch_size,
        "scope": "complete test-split forward loop",
    }


def run_district(args, district: int, mode: str):
    if mode not in MODES:
        raise ValueError(f"Unknown KITS mode: {mode}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    flow, full_adj, nodes = load_pure_dataset(args.pure_data_root, district, args.top_k)
    train_starts, evaluation_starts = make_window_split(len(flow), args.window)
    # Match the other baselines' global scaler, including the last point of
    # the final training window and without looking at future time steps.
    training_end = train_starts[-1] + args.window
    mean = float(flow[:training_end].mean())
    std = float(flow[:training_end].std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training flow has invalid standard deviation")
    values = ((flow - mean) / std).astype(np.float32)

    train_nodes = nodes["train_nodes"]
    valid_nodes = np.concatenate((train_nodes, nodes["valid_nodes"]))
    test_nodes = np.concatenate((train_nodes, nodes["test_nodes"]))
    node_metadata = load_metadata_features(
        args.pure_data_root, district, flow.shape[1], train_nodes,
    )
    if mode == "no_metadata":
        # Match the other baselines' zero-metadata ablation: same parameters,
        # same input dimensions, and no static information in any node.
        node_metadata = np.zeros_like(node_metadata)
    train_loader = make_loader(
        values, train_nodes, train_starts, args.window, args.batch_size,
        args.samples_per_epoch,
    )
    valid_loader = make_loader(
        values, valid_nodes, evaluation_starts, args.window, args.eval_batch_size,
    )
    test_loader = make_loader(
        values, test_nodes, evaluation_starts, args.window, args.eval_batch_size,
    )
    device = torch.device(args.device)
    output_dir = args.output_dir / f"pems{district:02d}_topk{args.top_k}" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or output_dir / "best.pt"
    state = torch.load(checkpoint, map_location=device) if args.test_only else None
    if state is not None and "model" not in state:
        raise ValueError(f"Checkpoint has no KITS model state: {checkpoint}")
    if state is not None and state.get("mode", mode) != mode:
        raise ValueError(f"Checkpoint mode {state['mode']} does not match {mode}")
    checkpoint_has_metadata = state is not None and any(
        name.startswith("metadata_projection.") for name in state["model"]
    )
    if mode == "metadata" and state is not None and not checkpoint_has_metadata:
        raise ValueError("Metadata evaluation requires a metadata-trained checkpoint")
    legacy_no_metadata = (
        mode == "no_metadata" and state is not None and not checkpoint_has_metadata
    )
    model_args = SimpleNamespace(
        dataset_name=f"pems{district:02d}_pure", use_adj_drop=False,
        use_init=False, metadata_dim=0 if legacy_no_metadata else node_metadata.shape[1],
        metadata_embedding_dim=args.metadata_embedding_dim,
    )
    model = KITS(full_adj, d_in=1, d_hidden=args.hidden_dim, args=model_args).to(device)
    train_metadata = valid_metadata = test_metadata = None
    if not legacy_no_metadata:
        train_metadata = torch.from_numpy(node_metadata[train_nodes]).to(device)
        valid_metadata = torch.from_numpy(node_metadata[valid_nodes]).to(device)
        test_metadata = torch.from_numpy(node_metadata[test_nodes]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=min(args.lr, 0.0001),
    )

    if args.benchmark_result is not None:
        model.adj = torch.as_tensor(full_adj, device=device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        buffer_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in model.buffers()
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train_loss, steps = train_one_epoch(
            model, train_loader, optimizer, train_nodes, flow.shape[1],
            mean, std, train_metadata, device, 1,
        )
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        training_peak_allocated = (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None
        )
        training_peak_reserved = (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" else None
        )
        model.adj = torch.as_tensor(
            full_adj[np.ix_(test_nodes, test_nodes)], device=device
        )
        inference = benchmark_inference(
            model,
            test_loader,
            len(train_nodes),
            mean,
            std,
            device,
            test_metadata,
        )
        result = {
            "dataset": "TraffiKrig",
            "seconds": seconds,
            "steps": steps,
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "buffer_bytes": buffer_bytes,
            "peak_gpu_allocated_bytes": training_peak_allocated,
            "peak_gpu_reserved_bytes": training_peak_reserved,
            "batch_size": args.batch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "train_loss": train_loss,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "torch_version": torch.__version__,
            "inference": inference,
        }
        args.benchmark_result.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_result.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )
        print(f"KITS one-epoch benchmark: {result}", flush=True)
        return result

    best_mae = float("inf")
    best_epoch = 0
    best_validation = None
    stale_epochs = 0
    epochs_ran = 0
    early_stopped = False
    training_total_seconds = None
    epochs = range(0) if args.test_only else range(1, args.epochs + 1)
    if not args.test_only:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_started = time.perf_counter()
    for epoch in epochs:
        epochs_ran = epoch
        model.adj = torch.as_tensor(full_adj, device=device)
        train_loss, steps = train_one_epoch(
            model, train_loader, optimizer, train_nodes, flow.shape[1],
            mean, std, train_metadata, device, epoch,
        )
        scheduler.step()

        model.adj = torch.as_tensor(full_adj[np.ix_(valid_nodes, valid_nodes)], device=device)
        validation = evaluate(
            model, valid_loader, len(train_nodes), mean, std, device, valid_metadata,
        )
        print(
            f"PEMS{district:02d} {mode} epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} "
            f"val_MAE={validation['MAE']:.4f} val_RMSE={validation['RMSE']:.4f} "
            f"val_MAPE={validation['MAPE']:.4f}%", flush=True,
        )
        if validation["MAE"] < best_mae:
            best_mae = validation["MAE"]
            best_epoch = epoch
            best_validation = validation
            stale_epochs = 0
            model.adj = torch.as_tensor(full_adj, device=device)
            torch.save({
                "model": model.state_dict(), "epoch": epoch, "val_MAE": best_mae,
                "mode": mode, "metadata_dim": model_args.metadata_dim,
            }, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                early_stopped = True
                print(
                    f"PEMS{district:02d} {mode} early stopping at epoch {epoch}: "
                    f"val_MAE has not improved for {args.patience} epochs; "
                    f"best epoch={best_epoch}, best val_MAE={best_mae:.4f}",
                    flush=True,
                )
                break

    if not args.test_only:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_total_seconds = time.perf_counter() - training_started
        print(
            f"PEMS{district:02d} {mode} training completed: "
            f"epochs_ran={epochs_ran}, early_stopped={early_stopped}, "
            f"total_time={training_total_seconds:.2f}s",
            flush=True,
        )

    if state is None:
        state = torch.load(checkpoint, map_location=device)
    # The last validation pass uses the smaller train+valid graph. Checkpoints
    # contain the full-graph adjacency, so restore its shape before loading.
    model.adj = torch.as_tensor(full_adj, device=device)
    model.load_state_dict(state["model"])
    if args.test_only:
        best_epoch = int(state.get("epoch", 0))
        model.adj = torch.as_tensor(full_adj[np.ix_(valid_nodes, valid_nodes)], device=device)
        best_validation = evaluate(
            model, valid_loader, len(train_nodes), mean, std, device, valid_metadata,
        )
    model.adj = torch.as_tensor(full_adj[np.ix_(test_nodes, test_nodes)], device=device)
    testing = evaluate(
        model, test_loader, len(train_nodes), mean, std, device, test_metadata,
    )
    result = {
        "model": "KITS", "mode": mode, "dataset": "TraffiKrig",
        "district": district,
        "top_k": args.top_k,
        "metadata_dim": model_args.metadata_dim,
        "metadata_embedding_dim": args.metadata_embedding_dim,
        "epoch": best_epoch, "train_nodes": len(train_nodes),
        "epochs_ran": epochs_ran,
        "early_stopped": early_stopped,
        "training_total_seconds": training_total_seconds,
        "valid_nodes": len(nodes["valid_nodes"]), "test_nodes": len(nodes["test_nodes"]),
        "train_windows": len(train_starts), "evaluation_windows": len(evaluation_starts),
        "val_MAE": best_validation["MAE"], "val_RMSE": best_validation["RMSE"],
        "val_MAPE": best_validation["MAPE"],
        "test_MAE": testing["MAE"], "test_RMSE": testing["RMSE"],
        "test_MAPE": testing["MAPE"], "test_count": testing["count"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"PEMS{district:02d} {mode} best epoch {best_epoch}: "
        f"test_MAE={testing['MAE']:.4f} "
        f"test_RMSE={testing['RMSE']:.4f} test_MAPE={testing['MAPE']:.4f}%",
        flush=True,
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--districts", nargs="+", type=int, choices=DISTRICTS, default=list(DISTRICTS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--pure-data-root", type=Path, default=PURE_ROOT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=KITS_DIR / "results")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=5120)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--metadata-embedding-dim", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-only", action="store_true",
                        help="evaluate a saved best.pt without training")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="checkpoint to use with --test-only; default: this mode's best.pt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark-result", type=Path, default=None,
                        help="time one training epoch, write JSON, and skip validation/checkpoints")
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates")
    for name in ("top_k", "window", "epochs", "patience", "batch_size", "eval_batch_size",
                 "hidden_dim", "metadata_embedding_dim"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.samples_per_epoch < 0 or (0 < args.samples_per_epoch < args.batch_size):
        parser.error("--samples-per-epoch must be zero or at least --batch-size")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.checkpoint is not None and not args.test_only:
        parser.error("--checkpoint requires --test-only")
    if args.checkpoint is not None and (len(args.districts) != 1 or len(args.modes) != 1):
        parser.error("--checkpoint requires exactly one district and one mode")
    if args.benchmark_result is not None and (
        args.test_only or len(args.districts) != 1 or len(args.modes) != 1
    ):
        parser.error("--benchmark-result needs exactly one district and mode, without --test-only")
    return args


def main():
    args = parse_args()
    if args.dry_run:
        for district in args.districts:
            directory = args.pure_data_root / f"pems{district:02d}"
            for mode in args.modes:
                print(f"PEMS{district:02d} {mode}: "
                      f"{directory / f'district_{district}_flow.npy'}; "
                      f"{directory / f'district_{district}_adj_inout_topk{args.top_k}.npy'}; "
                      f"{directory / 'adj_index.pkl'}; "
                      f"{directory / f'district_{district}_node_metadata.npz'}")
        return
    if args.benchmark_result is not None:
        run_district(args, args.districts[0], args.modes[0])
        return
    results = [
        run_district(args, district, mode)
        for district in args.districts for mode in args.modes
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = (f"topk{args.top_k}_pure_results.csv" if set(args.modes) == set(MODES)
                else f"topk{args.top_k}_{'_'.join(args.modes)}_results.csv")
    path = args.output_dir / filename
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results: {path}")


if __name__ == "__main__":
    main()
