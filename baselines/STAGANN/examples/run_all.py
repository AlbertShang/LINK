"""Prepare pure PEMS03 and run every kriging baseline sequentially.

Run from any working directory with::

    python baselines/STAGANN/examples/run_all.py --gpus 0
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


STAGANN_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = STAGANN_DIR.parents[1]
DATASET_DIR = STAGANN_DIR / "datasets" / "pems03_refactor_71_24"
PURE_DATASET_DIR = REPOSITORY_ROOT / "pure_dataset" / "pems03"
DATA_GENERATOR = (
    STAGANN_DIR / "data_preparation" / "pems03_refactor" / "generate_training_data.py"
)
RUNNER = STAGANN_DIR / "examples" / "run.py"
RESULT_FILE = STAGANN_DIR / "checkpoints" / DATASET_DIR.name / "baseline_results.csv"

CONFIGS = {
    "IGNNK": "examples/IGNNK/IGNNK_PEMS03_REFACTOR.py",
    "INCREASE": "examples/INCREASE/INCREASE_PEMS03_REFACTOR.py",
    "Okriging": "examples/Okriging/Okriging_PEMS03_REFACTOR.py",
    "SATCN": "examples/SATCN/SATCN_PEMS03_REFACTOR.py",
    "STAGANN": "examples/STAGANN/STAGANN_PEMS03_REFACTOR.py",
}

DATASET_ARTIFACTS = (
    "data.pkl",
    "scaler.pkl",
    "index.pkl",
    "adj_index.pkl",
    "adj_mx.pkl",
    "metadata.pkl",
    "dataset_info.json",
)
PURE_INPUTS = (
    PURE_DATASET_DIR / "district_3_flow.npy",
    PURE_DATASET_DIR / "district_3_adj.npy",
    PURE_DATASET_DIR / "district_3_meta_feature.csv",
    PURE_DATASET_DIR / "adj_index.pkl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all baselines on the converted pure PEMS03 dataset."
    )
    parser.add_argument("--gpus", default="0", help="visible GPU IDs passed to EasyTorch")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(CONFIGS),
        default=None,
        help="optional subset of models; the default runs all models",
    )
    parser.add_argument(
        "-c",
        "--cfg",
        type=Path,
        help="run one explicit config instead of the registered refactor configs",
    )
    parser.add_argument(
        "--no-prepare-data",
        action="store_true",
        help="fail when converted dataset artifacts are absent instead of generating them",
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="regenerate and replace converted dataset artifacts before training",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="stop after the first failed baseline"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print preparation/training commands only"
    )
    return parser.parse_args()


def dataset_is_ready() -> bool:
    return all((DATASET_DIR / artifact).is_file() for artifact in DATASET_ARTIFACTS)


def pure_dataset_is_newer() -> bool:
    existing_inputs = [path for path in PURE_INPUTS if path.is_file()]
    if existing_inputs and len(existing_inputs) != len(PURE_INPUTS):
        missing = [str(path) for path in PURE_INPUTS if not path.is_file()]
        raise FileNotFoundError("Pure PEMS03 dataset is incomplete; missing=" + str(missing))
    if not existing_inputs:
        return False
    oldest_artifact = min((DATASET_DIR / name).stat().st_mtime for name in DATASET_ARTIFACTS)
    newest_input = max(path.stat().st_mtime for path in existing_inputs)
    return newest_input > oldest_artifact


def prepare_dataset(args: argparse.Namespace) -> None:
    ready = dataset_is_ready()
    if ready and args.no_prepare_data:
        print(f"Dataset ready: {DATASET_DIR}")
        return
    if not ready and args.no_prepare_data:
        missing = [
            name for name in DATASET_ARTIFACTS if not (DATASET_DIR / name).is_file()
        ]
        raise FileNotFoundError(
            f"Converted dataset is incomplete at {DATASET_DIR}; missing={missing}"
        )
    stale = ready and pure_dataset_is_newer()
    if ready and not stale and not args.overwrite_data:
        print(f"Dataset ready: {DATASET_DIR}")
        return
    command = [sys.executable, str(DATA_GENERATOR), "--district", "3"]
    if ready or args.overwrite_data:
        command.append("--overwrite")
    print("Prepare dataset:", " ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=STAGANN_DIR, check=True)


def training_command(config: Path | str, gpus: str) -> list[str]:
    return [sys.executable, str(RUNNER), "-c", str(config), "--gpus", gpus]


def _metrics(text: str, prefix: str) -> dict[str, float]:
    pattern = rf"{prefix}_(MAE|RMSE|MAPE|R2|ACC):\s*([-+0-9.eE]+)"
    return {name.lower(): float(value) for name, value in re.findall(pattern, text)}


def best_validation_result(
    model_name: str, epochs: int, not_before: float
) -> dict[str, float | int] | None:
    log_root = STAGANN_DIR / "checkpoints" / DATASET_DIR.name / f"{model_name}_{epochs}"
    logs = list(log_root.rglob("training_log_*.log")) if log_root.is_dir() else []
    logs = [path for path in logs if path.stat().st_mtime >= not_before - 2]
    if not logs:
        return None
    lines = max(logs, key=lambda path: path.stat().st_mtime).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    epoch: int | None = None
    validation: dict[str, float] | None = None
    records: list[dict[str, float | int]] = []
    for line in lines:
        epoch_match = re.search(r"Epoch\s+(\d+)\s*/", line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
        if "Result <val>:" in line:
            validation = _metrics(line, "val")
        elif "Result <test>:" in line and epoch is not None and validation:
            record: dict[str, float | int] = {"epoch": epoch}
            record.update({f"val_{key}": value for key, value in validation.items()})
            record.update({f"test_{key}": value for key, value in _metrics(line, "test").items()})
            records.append(record)
            validation = None
    candidates = [record for record in records if "val_mae" in record]
    return min(candidates, key=lambda record: float(record["val_mae"])) if candidates else None


def write_results(rows: list[dict[str, object]]) -> None:
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model",
        "status",
        "elapsed_minutes",
        "best_epoch",
        "val_mae",
        "test_mae",
        "test_rmse",
        "test_mape",
        "test_r2",
    )
    with RESULT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    args = parse_args()
    if args.cfg is not None:
        config = args.cfg if args.cfg.is_absolute() else STAGANN_DIR / args.cfg
        if not config.is_file():
            raise FileNotFoundError(f"Configuration does not exist: {config}")
        command = training_command(config, args.gpus)
        print("Run:", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=STAGANN_DIR, check=True)
        return 0

    prepare_dataset(args)
    model_names = args.models or list(CONFIGS)
    configs = [(name, STAGANN_DIR / CONFIGS[name]) for name in model_names]
    missing_configs = [str(path) for _, path in configs if not path.is_file()]
    if missing_configs:
        raise FileNotFoundError("Missing baseline configs: " + ", ".join(missing_configs))

    print(f"Run {len(configs)} baselines on {DATASET_DIR.name}: {', '.join(model_names)}")
    results: list[dict[str, object]] = []
    for index, (name, config) in enumerate(configs, start=1):
        command = training_command(config.relative_to(STAGANN_DIR), args.gpus)
        print(f"\n[{index}/{len(configs)}] {name}")
        print("Command:", " ".join(command))
        if args.dry_run:
            results.append({"model": name, "status": "DRY-RUN", "elapsed_minutes": 0.0})
            continue
        started_at = time.time()
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=STAGANN_DIR, check=False)
        elapsed = time.perf_counter() - started
        status = "OK" if completed.returncode == 0 else f"FAILED({completed.returncode})"
        epochs = 15 if name == "IAGCN" else 1 if name == "Okriging" else 50
        metrics = best_validation_result(name, epochs, started_at)
        row: dict[str, object] = {
            "model": name,
            "status": status,
            "elapsed_minutes": round(elapsed / 60, 4),
        }
        if metrics:
            row.update(metrics)
            row["best_epoch"] = row.pop("epoch")
        results.append(row)
        if completed.returncode != 0 and args.fail_fast:
            break

    print("\nRun summary")
    print(f"  {'Model':<10} {'Status':<12} {'Best':>5} {'Val MAE':>10} {'Test MAE':>10}")
    for row in results:
        print(
            f"  {row['model']:<10} {row['status']:<12} "
            f"{str(row.get('best_epoch', '-')):>5} "
            f"{str(row.get('val_mae', '-')):>10} {str(row.get('test_mae', '-')):>10}"
        )
    if not args.dry_run:
        write_results(results)
        print(f"Detailed comparison: {RESULT_FILE}")
    failures = [
        str(row["model"]) for row in results if str(row["status"]).startswith("FAILED")
    ]
    if failures:
        print("Failed baselines:", ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
