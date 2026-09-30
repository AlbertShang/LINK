"""Run TraffiKrig PEMS03 Top-K=3 training and inference benchmarks serially.

Each method uses its existing training updates. Data conversion, model setup,
validation, testing, and checkpointing are outside the timed interval. LINK
has two SPN training stages; one epoch of each is reported separately and
their sum is included for convenience. After that epoch, every worker also
times one complete test-split forward loop using the in-memory model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGANN_DIR = ROOT / "baselines" / "STAGANN"
BASELINES = (
    "DualSTN", "GCN", "GIN", "GRIN", "IAGCN", "IGNNK",
    "INCREASE", "Okriging", "SATCN", "STAGANN", "KITS", "LINK",
)
MODES = ("metadata", "no_metadata")
STAGANN_ARTIFACTS = (
    "data.pkl", "scaler.pkl", "index.pkl", "adj_index.pkl",
    "adj_mx.pkl", "metadata.pkl", "dataset_info.json",
)
FIELDS = (
    "model", "mode", "dataset", "district", "top_k", "configuration", "status",
    "seconds", "steps",
    "stage_count", "total_seconds", "total_steps", "seconds_per_step",
    "trend_seconds", "trend_steps", "residual_seconds",
    "residual_steps", "batch_size", "center_batch_size",
    "train_days_per_epoch", "samples_per_epoch",
    "parameter_count", "parameter_mib", "buffer_mib", "model_state_mib",
    "peak_gpu_allocated_mib", "peak_gpu_reserved_mib",
    "trend_peak_gpu_allocated_mib", "residual_peak_gpu_allocated_mib",
    "inference_seconds", "inference_steps", "inference_samples",
    "inference_target_count", "inference_batch_size",
    "inference_seconds_per_step", "inference_ms_per_sample",
    "inference_targets_per_second", "inference_scope",
    "device", "gpu_name", "torch_version",
    "log", "error",
)

MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-data-root", type=Path, default=ROOT / "pure_dataset_topk")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "efficiency_pems03_topk3")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--models", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--gpus", default="0", help="GPU IDs for the STAGANN launcher")
    parser.add_argument("--device", default="cuda", help="device for KITS and LINK")
    parser.add_argument("--stagann-python", default=sys.executable)
    parser.add_argument("--kits-python", default=sys.executable)
    parser.add_argument("--link-python", default=sys.executable)
    parser.add_argument("--stagann-batch-size", type=int)
    parser.add_argument("--kits-batch-size", type=int)
    parser.add_argument("--kits-samples-per-epoch", type=int)
    parser.add_argument("--center-batch-size", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-prepare-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models) or len(set(args.modes)) != len(args.modes):
        parser.error("--models and --modes must not contain duplicates")
    for name in ("stagann_batch_size", "kits_batch_size", "center_batch_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.kits_samples_per_epoch is not None and args.kits_samples_per_epoch < 0:
        parser.error("--kits-samples-per-epoch must be nonnegative")
    return args


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    try:
        requested = int(env.get("NUMEXPR_NUM_THREADS", os.cpu_count() or 1))
    except ValueError:
        requested = os.cpu_count() or 1
    env["NUMEXPR_MAX_THREADS"] = "64"
    env["NUMEXPR_NUM_THREADS"] = str(max(1, min(requested, 64)))
    # GPU work is synchronized at the measurement boundaries in every worker.
    env["CUDA_LAUNCH_BLOCKING"] = "0"
    return env


def run_logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("Command: " + " ".join(command) + "\n\n")
        stream.flush()
        completed = subprocess.run(
            command, cwd=ROOT, env=subprocess_env(), stdout=stream,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"exit code {completed.returncode}; see {log}")


def ensure_stagann_data(args: argparse.Namespace) -> None:
    dataset_dir = STAGANN_DIR / "datasets" / "pems03_metadata_topk3_71_24"
    pure_dir = args.pure_data_root / "pems03"
    pure_inputs = (
        pure_dir / "district_3_flow.npy",
        pure_dir / "district_3_adj_inout_topk3.npy",
        pure_dir / "district_3_node_metadata.npz",
        pure_dir / "adj_index.pkl",
    )
    missing_pure = [str(path) for path in pure_inputs if not path.is_file()]
    if missing_pure:
        raise FileNotFoundError(f"Missing TraffiKrig PEMS03 inputs: {missing_pure}")
    ready = all((dataset_dir / name).is_file() for name in STAGANN_ARTIFACTS)
    current = False
    if ready:
        try:
            info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
            current = (
                info.get("district") == 3
                and Path(info.get("source_dir", "")).resolve() == pure_dir
                and info.get("adjacency_file") == "district_3_adj_inout_topk3.npy"
                and info.get("adjacency_top_k") == 3
                and info.get("metadata", {}).get("layout_version")
                == "canonical-node-metadata-v2"
                and max(path.stat().st_mtime for path in pure_inputs)
                <= min((dataset_dir / name).stat().st_mtime for name in STAGANN_ARTIFACTS)
            )
        except (OSError, ValueError, TypeError):
            current = False
    if current:
        return
    if args.no_prepare_data:
        raise FileNotFoundError(f"Missing or stale converted STAGANN PEMS03 dataset: {dataset_dir}")
    command = [
        args.stagann_python,
        str(STAGANN_DIR / "data_preparation" / "pems03_refactor" / "generate_training_data.py"),
        "--district", "3", "--top-k", "3", "--pure-dir", str(pure_dir),
        "--output-dir", str(dataset_dir),
    ]
    if ready or dataset_dir.exists():
        command.append("--overwrite")
    print("Preparing STAGANN data", flush=True)
    run_logged(command, args.output_dir / "prepare_stagann_data.log")
    missing = [name for name in STAGANN_ARTIFACTS if not (dataset_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"STAGANN conversion did not produce {missing}")


def check_pure_inputs(args: argparse.Namespace) -> None:
    directory = args.pure_data_root / "pems03"
    required = set()
    if "LINK" in args.models:
        required.update((
            "district_3_flow.npy", "district_3_meta_feature.csv",
            "adj_index.pkl", "adaptive_boundary_subgraphs_inout_topk3.pt",
        ))
    if "KITS" in args.models:
        required.update((
            "district_3_flow.npy", "district_3_adj_inout_topk3.npy",
            "adj_index.pkl", "district_3_node_metadata.npz",
        ))
    missing = [str(directory / name) for name in sorted(required)
               if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing TraffiKrig PEMS03 inputs: {missing}")


def command_for(args: argparse.Namespace, model: str, mode: str, result: Path) -> list[str]:
    if model == "LINK":
        command = [
            args.link_python, str(ROOT / "experiments" / "benchmark_epoch.py"),
            "--mode", mode, "--top-k", "3", "--pure-data-root", str(args.pure_data_root),
            "--device", args.device, "--result", str(result),
        ]
        if args.center_batch_size is not None:
            command += ["--center-batch-size", str(args.center_batch_size)]
        return command
    if model == "KITS":
        command = [
            args.kits_python, str(ROOT / "baselines" / "KITS" / "run_pure.py"),
            "--districts", "3", "--modes", mode, "--top-k", "3",
            "--pure-data-root", str(args.pure_data_root),
            "--device", args.device, "--benchmark-result", str(result),
            "--output-dir", str(args.output_dir / "kits_work"),
        ]
        if args.kits_batch_size is not None:
            command += ["--batch-size", str(args.kits_batch_size)]
        if args.kits_samples_per_epoch is not None:
            command += ["--samples-per-epoch", str(args.kits_samples_per_epoch)]
        return command
    command = [
        args.stagann_python,
        str(STAGANN_DIR / "examples" / "benchmark_epoch.py"),
        "--model", model, "--mode", mode, "--top-k", "3",
        "--gpus", args.gpus, "--result", str(result),
    ]
    if args.stagann_batch_size is not None:
        command += ["--batch-size", str(args.stagann_batch_size)]
    return command


def result_row(model: str, mode: str, result: Path, log: Path) -> dict[str, object]:
    payload = json.loads(result.read_text(encoding="utf-8"))
    stages = payload.get("stages", {})
    trend = stages.get("trend", {})
    residual = stages.get("residual", {})
    if stages:
        # Recompute from the stages so --skip-existing also handles JSON files
        # produced before the top-level one-epoch average was introduced.
        stage_count = len(stages)
        total_seconds = sum(float(stage["seconds"]) for stage in stages.values())
        total_steps = sum(int(stage["steps"]) for stage in stages.values())
        weights = payload.get("stage_epoch_weights") or {
            name: 1 for name in stages
        }
        if set(weights) != set(stages) or any(
            not isinstance(count, int) or count <= 0 for count in weights.values()
        ):
            raise ValueError(f"Invalid stage epoch weights: {weights}")
        scheduled_epochs = sum(weights.values())
        seconds = sum(
            float(stage["seconds"]) * weights[name]
            for name, stage in stages.items()
        ) / scheduled_epochs
        steps = sum(
            int(stage["steps"]) * weights[name]
            for name, stage in stages.items()
        ) / scheduled_epochs
    else:
        stage_count = 1
        total_seconds = float(payload["seconds"])
        total_steps = int(payload["steps"])
        seconds = total_seconds
        steps = total_steps
    if not math.isfinite(seconds) or seconds <= 0 or total_steps <= 0:
        raise ValueError(f"Invalid benchmark measurement: {payload}")
    inference = payload.get("inference")
    if not isinstance(inference, dict):
        raise ValueError(f"Missing inference benchmark measurement: {payload}")
    inference_seconds = float(inference["seconds"])
    inference_steps = int(inference["steps"])
    inference_samples = int(inference["samples"])
    inference_target_count = int(inference["target_count"])
    if (
        not math.isfinite(inference_seconds)
        or inference_seconds <= 0
        or inference_steps <= 0
        or inference_samples <= 0
        or inference_target_count <= 0
    ):
        raise ValueError(f"Invalid inference benchmark measurement: {inference}")
    if model == "LINK":
        default_config = (
            payload.get("center_batch_size") == 7
            and payload.get("train_days_per_epoch") == 7
        )
    elif model == "KITS":
        default_config = (
            payload.get("batch_size") == 4
            and payload.get("samples_per_epoch") == 5120
            and inference.get("batch_size") == 2
        )
    else:
        default_config = (
            payload.get("batch_size") == 64
            and inference.get("batch_size") == 64
        )

    def as_mib(value: object) -> float | str:
        return "" if value is None else int(value) / MIB

    parameter_bytes = payload.get("parameter_bytes")
    buffer_bytes = payload.get("buffer_bytes")
    return {
        "model": model, "mode": mode, "dataset": "TraffiKrig",
        "district": "PEMS03", "top_k": 3,
        "configuration": "default" if default_config else "custom_or_unknown",
        "status": "ok", "seconds": seconds, "steps": steps,
        "stage_count": stage_count, "total_seconds": total_seconds,
        "total_steps": total_steps, "seconds_per_step": total_seconds / total_steps,
        "trend_seconds": trend.get("seconds", ""), "trend_steps": trend.get("steps", ""),
        "residual_seconds": residual.get("seconds", ""),
        "residual_steps": residual.get("steps", ""),
        "batch_size": payload.get("batch_size", ""),
        "center_batch_size": payload.get("center_batch_size", ""),
        "train_days_per_epoch": payload.get("train_days_per_epoch", ""),
        "samples_per_epoch": payload.get("samples_per_epoch", ""),
        "parameter_count": payload.get("parameter_count", ""),
        "parameter_mib": as_mib(parameter_bytes),
        "buffer_mib": as_mib(buffer_bytes),
        "model_state_mib": (
            (int(parameter_bytes) + int(buffer_bytes)) / MIB
            if parameter_bytes is not None and buffer_bytes is not None else ""
        ),
        "peak_gpu_allocated_mib": as_mib(payload.get("peak_gpu_allocated_bytes")),
        "peak_gpu_reserved_mib": as_mib(payload.get("peak_gpu_reserved_bytes")),
        "trend_peak_gpu_allocated_mib": as_mib(
            trend.get("peak_gpu_allocated_bytes")
        ),
        "residual_peak_gpu_allocated_mib": as_mib(
            residual.get("peak_gpu_allocated_bytes")
        ),
        "inference_seconds": inference_seconds,
        "inference_steps": inference_steps,
        "inference_samples": inference_samples,
        "inference_target_count": inference_target_count,
        "inference_batch_size": inference.get("batch_size", ""),
        "inference_seconds_per_step": inference_seconds / inference_steps,
        "inference_ms_per_sample": 1000.0 * inference_seconds / inference_samples,
        "inference_targets_per_second": inference_target_count / inference_seconds,
        "inference_scope": inference.get("scope", ""),
        "device": payload.get("device", ""),
        "gpu_name": payload.get("gpu_name", ""),
        "torch_version": payload.get("torch_version", ""),
        "log": str(log), "error": "",
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def has_complete_metrics(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return all(key in payload for key in (
        "parameter_count", "parameter_bytes", "buffer_bytes",
        "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes",
    )) and isinstance(payload.get("inference"), dict) and all(
        key in payload["inference"]
        for key in ("seconds", "steps", "samples", "target_count")
    )


def main() -> None:
    args = parse_args()
    args.pure_data_root = args.pure_data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    jobs = [(model, mode) for mode in args.modes for model in args.models]
    if args.dry_run:
        print("TraffiKrig PEMS03 Top-K=3; one training epoch plus test-split inference per job")
        for model, mode in jobs:
            result = args.output_dir / f"{model}_{mode}.json"
            print(" ".join(command_for(args, model, mode, result)))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_pure_inputs(args)
    if any(model not in ("KITS", "LINK") for model in args.models):
        ensure_stagann_data(args)
    rows: list[dict[str, object]] = []
    summary = args.output_dir / "epoch_times.csv"
    for index, (model, mode) in enumerate(jobs, 1):
        name = f"{model}_{mode}"
        result = args.output_dir / f"{name}.json"
        log = args.output_dir / f"{name}.log"
        print(f"[{index}/{len(jobs)}] {name}", flush=True)
        try:
            if not (args.skip_existing and result.is_file() and has_complete_metrics(result)):
                result.unlink(missing_ok=True)
                run_logged(command_for(args, model, mode, result), log)
            if not result.is_file():
                raise RuntimeError(f"Worker finished without writing {result}; see {log}")
            row = result_row(model, mode, result, log)
            print(
                f"  {row['seconds']:.3f} s/epoch, {row['steps']:g} steps/epoch; "
                f"inference={row['inference_seconds']:.3f} s/test split",
                flush=True,
            )
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"  FAILED: {exc}", flush=True)
            row = {
                "model": model, "mode": mode, "dataset": "TraffiKrig",
                "district": "PEMS03", "top_k": 3,
                "status": "failed", "log": str(log), "error": str(exc),
            }
        rows.append(row)
        write_summary(summary, rows)
    print(f"Results: {summary}")
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        raise SystemExit(f"{len(failed)} of {len(jobs)} efficiency jobs failed; see {summary}")


if __name__ == "__main__":
    main()
