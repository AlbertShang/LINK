"""Time one training epoch in each SPN stage of LINK on TraffiKrig PEMS03."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import get_experiment_config  # noqa: E402
from experiments.main import _device, _load_data  # noqa: E402
from src.engine import LINKExperiment  # noqa: E402


def timed_epoch(experiment, optimizer, scheduler, residual_stage, epoch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    blocks = experiment.sample_training_blocks(epoch)
    loss, steps = experiment._train_epoch(
        experiment.training_graphs,
        blocks,
        optimizer,
        scheduler,
        residual_stage,
        experiment.training_config.seed + epoch,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not steps:
        raise RuntimeError("LINK training epoch has no optimizer steps")
    result = {"seconds": time.perf_counter() - started, "steps": steps, "loss": loss}
    if device.type == "cuda":
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return result


def timed_inference(experiment, device):
    """Time one complete test-split forward pass after the benchmark epoch."""
    experiment.model.eval()
    graph = experiment.graphs["test"]
    ranges = experiment._evaluation_ranges()
    target_count_tensor = torch.zeros((), dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start, end in ranges:
            _, information = experiment._run_model(
                graph,
                start,
                end,
                experiment.model_config.use_residual_refinement,
            )
            target = experiment._targets(graph, start, end)
            valid = experiment._valid_mask(target, information["prediction_valid"])
            target_count_tensor += valid.sum()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    target_count = int(target_count_tensor.item())
    if not ranges or target_count <= 0:
        raise RuntimeError("LINK inference benchmark has no valid test targets")
    centers = graph["center_global_ids"]
    assert isinstance(centers, torch.Tensor)
    return {
        "seconds": seconds,
        "steps": len(ranges),
        "samples": experiment.sequence_length,
        "target_count": target_count,
        "batch_size": int(centers.numel()),
        "scope": "complete test-split forward loop",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metadata", "no_metadata"), required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--pure-data-root", type=Path, default=ROOT / "pure_dataset_topk")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--center-batch-size", type=int, default=None)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k <= 0 or (args.center_batch_size is not None and args.center_batch_size <= 0):
        parser.error("Top-K and center batch size must be positive")
    device = _device(args.device)
    flow, _, subgraphs, splits, edge_metadata_dim, _ = _load_data(
        args.pure_data_root, 3, args.top_k, None,
    )
    config = get_experiment_config(3)
    config = replace(
        config,
        model=replace(config.model, use_metadata=args.mode == "metadata"),
        training=replace(
            config.training,
            center_batch_size=args.center_batch_size or config.training.center_batch_size,
        ),
    )
    experiment = LINKExperiment(
        flow, subgraphs, splits, edge_metadata_dim, device, config,
    )
    parameter_count = sum(parameter.numel() for parameter in experiment.model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in experiment.model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in experiment.model.buffers()
    )
    windows_per_day = (
        config.model.day_points // config.model.window_size
        if config.model.time_window_impute else 1
    )
    steps_per_epoch = (
        len(experiment.training_graphs)
        * config.training.train_days_per_epoch
        * windows_per_day
    )

    trend_optimizer = experiment._build_optimizer(experiment.model.trend_mlp.parameters())
    trend_scheduler = experiment._build_scheduler(
        trend_optimizer, steps_per_epoch * config.training.trend_epochs,
    )
    trend = timed_epoch(experiment, trend_optimizer, trend_scheduler, False, 1, device)

    stages = {"trend": trend}
    if config.model.use_residual_refinement:
        for parameter in experiment.model.trend_mlp.parameters():
            parameter.requires_grad = False
        residual_optimizer = experiment._build_optimizer(
            experiment.model.residual_mlp.parameters()
        )
        residual_scheduler = experiment._build_scheduler(
            residual_optimizer, steps_per_epoch * config.training.residual_epochs,
        )
        stages["residual"] = timed_epoch(
            experiment,
            residual_optimizer,
            residual_scheduler,
            True,
            config.training.trend_epochs + 1,
            device,
        )

    inference = timed_inference(experiment, device)

    total_seconds = sum(stage["seconds"] for stage in stages.values())
    total_steps = sum(stage["steps"] for stage in stages.values())
    stage_count = len(stages)
    stage_epoch_weights = {"trend": config.training.trend_epochs}
    if "residual" in stages:
        stage_epoch_weights["residual"] = config.training.residual_epochs
    scheduled_epochs = sum(stage_epoch_weights.values())
    result = {
        "method": "LINK",
        "model": "SPN",
        "dataset": "TraffiKrig",
        "seconds": sum(
            stages[name]["seconds"] * count
            for name, count in stage_epoch_weights.items()
        ) / scheduled_epochs,
        "steps": sum(
            stages[name]["steps"] * count
            for name, count in stage_epoch_weights.items()
        ) / scheduled_epochs,
        "total_seconds": total_seconds,
        "total_steps": total_steps,
        "stage_count": stage_count,
        "stage_epoch_weights": stage_epoch_weights,
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "peak_gpu_allocated_bytes": max(
            (stage.get("peak_gpu_allocated_bytes", 0) for stage in stages.values()),
            default=0,
        ) if device.type == "cuda" else None,
        "peak_gpu_reserved_bytes": max(
            (stage.get("peak_gpu_reserved_bytes", 0) for stage in stages.values()),
            default=0,
        ) if device.type == "cuda" else None,
        "center_batch_size": config.training.center_batch_size,
        "train_days_per_epoch": config.training.train_days_per_epoch,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch_version": torch.__version__,
        "stages": stages,
        "inference": inference,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"LINK one-epoch-per-SPN-stage benchmark: {result}", flush=True)


if __name__ == "__main__":
    main()
