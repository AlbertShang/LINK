"""Prepare and run metadata-aware baselines on pure PEMS03/04/07/08.

Examples from the repository root::

    python baselines/STAGANN/examples/run_all_metadata.py --districts 3 4 7 8 --top-k 3 --gpus 0
    python baselines/STAGANN/examples/run_all_metadata.py --districts 3 4 7 8 --top-k 3 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


STAGANN_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = STAGANN_DIR.parents[1]
DATA_GENERATOR = (
    STAGANN_DIR / "data_preparation" / "pems03_refactor" / "generate_training_data.py"
)
REGISTERED_RUNNER = STAGANN_DIR / "examples" / "run_metadata.py"
INPUT_CHECKER = STAGANN_DIR / "examples" / "check_metadata_input.py"
DEFAULT_PURE_DATA_ROOT = REPOSITORY_ROOT / "pure_dataset_topk"
SUPPORTED_DISTRICTS = (3, 4, 7, 8)
MODELS = (
    "DualSTN", "GCN", "GIN", "GRIN", "IAGCN", "IGNNK",
    "INCREASE", "Okriging", "SATCN", "STAGANN",
)
DATASET_ARTIFACTS = (
    "data.pkl", "scaler.pkl", "index.pkl", "adj_index.pkl",
    "adj_mx.pkl", "metadata.pkl", "dataset_info.json",
)


@dataclass(frozen=True)
class DatasetContext:
    district: int
    top_k: int
    pure_dir: Path
    dataset_dir: Path
    adjacency: Path
    pure_inputs: tuple[Path, ...]

    @property
    def label(self) -> str:
        return f"PEMS{self.district:02d}"

    @property
    def result_file(self) -> Path:
        return (
            STAGANN_DIR / "checkpoints" / self.dataset_dir.name / "metadata"
            / "baseline_metadata_results.csv"
        )


def dataset_context(
    district: int, pure_data_root: Path, top_k: int
) -> DatasetContext:
    pure_dir = pure_data_root / f"pems{district:02d}"
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    adjacency = pure_dir / f"district_{district}_adj_inout_topk{top_k}.npy"
    dataset_dir = (
        STAGANN_DIR / "datasets"
        / f"pems{district:02d}_metadata_topk{top_k}_71_24"
    )
    pure_inputs = (
        pure_dir / f"district_{district}_flow.npy",
        adjacency,
        pure_dir / f"district_{district}_node_metadata.npz",
        pure_dir / "adj_index.pkl",
    )
    return DatasetContext(
        district, top_k, pure_dir, dataset_dir, adjacency, pure_inputs,
    )


def subprocess_environment() -> dict[str, str]:
    """Clamp NumExpr threads without changing model training settings."""
    environment = os.environ.copy()
    max_threads = 64
    requested = environment.get("NUMEXPR_NUM_THREADS")
    try:
        requested_threads = (
            int(requested) if requested is not None else (os.cpu_count() or 1)
        )
    except ValueError:
        requested_threads = os.cpu_count() or 1
    environment["NUMEXPR_MAX_THREADS"] = str(max_threads)
    environment["NUMEXPR_NUM_THREADS"] = str(
        min(max(requested_threads, 1), max_threads)
    )
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--districts", nargs="+", type=int, choices=SUPPORTED_DISTRICTS,
        default=[3], help="pure PEMS districts to run; default: 3",
    )
    parser.add_argument(
        "--pure-data-root", type=Path, default=DEFAULT_PURE_DATA_ROOT,
        help="directory containing the non-nested pems03/pems04/pems07/pems08 folders",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="input graph Top-K to select from --pure-data-root; default: 3",
    )
    parser.add_argument("--gpus", default="0", help="GPU IDs passed to EasyTorch")
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=None,
        help="optional model subset; default: all ten baselines",
    )
    parser.add_argument(
        "--no-prepare-data", action="store_true",
        help="fail if converted dataset artifacts are absent",
    )
    parser.add_argument(
        "--overwrite-data", action="store_true",
        help="regenerate converted artifacts before training",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="stop after the first failure",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without training",
    )
    return parser.parse_args()


def dataset_is_ready(context: DatasetContext) -> bool:
    return all((context.dataset_dir / name).is_file() for name in DATASET_ARTIFACTS)


def dataset_layout_is_current(context: DatasetContext) -> bool:
    try:
        info = json.loads(
            (context.dataset_dir / "dataset_info.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return False
    return (
        info.get("district") == context.district
        and Path(info.get("source_dir", "")).resolve()
        == context.pure_dir.resolve()
        and info.get("metadata", {}).get("layout_version")
        == "canonical-node-metadata-v2"
        and info.get("adjacency_file") == context.adjacency.name
        and info.get("adjacency_top_k") == context.top_k
    )


def pure_dataset_is_newer(context: DatasetContext) -> bool:
    existing = [path for path in context.pure_inputs if path.is_file()]
    if existing and len(existing) != len(context.pure_inputs):
        missing = [str(path) for path in context.pure_inputs if not path.is_file()]
        raise FileNotFoundError(
            f"Pure {context.label} dataset is incomplete; missing={missing}"
        )
    if not existing:
        return False
    oldest_artifact = min(
        (context.dataset_dir / name).stat().st_mtime for name in DATASET_ARTIFACTS
    )
    return max(path.stat().st_mtime for path in existing) > oldest_artifact


def prepare_dataset(context: DatasetContext, args: argparse.Namespace) -> None:
    ready = dataset_is_ready(context)
    if not ready and args.no_prepare_data:
        missing = [name for name in DATASET_ARTIFACTS
                   if not (context.dataset_dir / name).is_file()]
        raise FileNotFoundError(
            f"Converted dataset is incomplete at {context.dataset_dir}; missing={missing}"
        )
    stale = ready and (
        not dataset_layout_is_current(context) or pure_dataset_is_newer(context)
    )
    if ready and not stale and not args.overwrite_data:
        print(f"Dataset ready: {context.dataset_dir}")
        return

    command = [
        sys.executable,
        str(DATA_GENERATOR),
        "--district", str(context.district),
        "--top-k", str(context.top_k),
        "--pure-dir", str(context.pure_dir),
        "--output-dir", str(context.dataset_dir),
    ]
    if ready or args.overwrite_data:
        command.append("--overwrite")
    print("Prepare dataset:", " ".join(command))
    if not args.dry_run:
        subprocess.run(
            command, cwd=STAGANN_DIR, env=subprocess_environment(), check=True
        )


def check_metadata_input(context: DatasetContext, dry_run: bool) -> None:
    command = [
        sys.executable, str(INPUT_CHECKER),
        "--dataset-dir", str(context.dataset_dir),
    ]
    print("Check metadata input:", " ".join(command))
    if not dry_run:
        subprocess.run(
            command, cwd=STAGANN_DIR, env=subprocess_environment(), check=True
        )


def training_command(
    model: str, district: int, top_k: int, gpus: str
) -> list[str]:
    return [
        sys.executable,
        str(REGISTERED_RUNNER),
        "--model", model,
        "--district", str(district),
        "--top-k", str(top_k),
        "--gpus", gpus,
    ]


def _metrics(text: str, prefix: str) -> dict[str, float]:
    pattern = rf"{prefix}_(MAE|RMSE|MAPE|R2|ACC):\s*([-+0-9.eE]+)"
    return {name.lower(): float(value) for name, value in re.findall(pattern, text)}


def best_validation_result(
    context: DatasetContext, model_name: str, not_before: float
) -> dict[str, float | int] | None:
    log_root = STAGANN_DIR / "checkpoints" / context.dataset_dir.name / "metadata"
    logs = list(log_root.glob(f"{model_name}_*/**/training_log_*.log"))
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
            record.update(
                {f"test_{key}": value for key, value in _metrics(line, "test").items()}
            )
            records.append(record)
            validation = None
    candidates = [record for record in records if "val_mae" in record]
    return (
        min(candidates, key=lambda record: float(record["val_mae"]))
        if candidates else None
    )


def write_results(context: DatasetContext, rows: list[dict[str, object]]) -> None:
    context.result_file.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "district", "model", "status", "elapsed_minutes", "best_epoch",
        "val_mae", "test_mae", "test_rmse", "test_mape", "test_r2",
    )
    with context.result_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def run_registered_configs(
    context: DatasetContext, args: argparse.Namespace
) -> list[dict[str, object]]:
    model_names = args.models or list(MODELS)
    print(
        f"Run {len(model_names)} metadata baselines on {context.dataset_dir.name}: "
        + ", ".join(model_names)
    )
    results: list[dict[str, object]] = []
    for index, name in enumerate(model_names, start=1):
        command = training_command(
            name, context.district, context.top_k, args.gpus
        )
        print(f"\n[{context.label} {index}/{len(model_names)}] {name}")
        print("Command:", " ".join(command))
        if args.dry_run:
            results.append({
                "district": context.district, "model": name,
                "status": "DRY-RUN", "elapsed_minutes": 0.0,
            })
            continue

        started_at = time.time()
        started = time.perf_counter()
        completed = subprocess.run(
            command, cwd=STAGANN_DIR, env=subprocess_environment(), check=False
        )
        elapsed = time.perf_counter() - started
        status = "OK" if completed.returncode == 0 else f"FAILED({completed.returncode})"
        metrics = best_validation_result(context, name, started_at)
        row: dict[str, object] = {
            "district": context.district,
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
    return results


def print_summary(context: DatasetContext, rows: list[dict[str, object]]) -> None:
    print(f"\n{context.label} run summary")
    print(f"  {'Model':<10} {'Status':<12} {'Best':>5} {'Val MAE':>10} {'Test MAE':>10}")
    for row in rows:
        print(
            f"  {row['model']:<10} {row['status']:<12} "
            f"{str(row.get('best_epoch', '-')):>5} "
            f"{str(row.get('val_mae', '-')):>10} "
            f"{str(row.get('test_mae', '-')):>10}"
        )


def main() -> int:
    args = parse_args()
    args.pure_data_root = args.pure_data_root.resolve()
    if len(set(args.districts)) != len(args.districts):
        raise ValueError("--districts cannot contain duplicates")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    any_failure = False
    for district in args.districts:
        context = dataset_context(district, args.pure_data_root, args.top_k)
        print(f"\n{'=' * 80}\n{context.label} | pure={context.pure_dir}")
        prepare_dataset(context, args)
        check_metadata_input(context, args.dry_run)

        rows = run_registered_configs(context, args)
        print_summary(context, rows)
        if not args.dry_run:
            write_results(context, rows)
            print(f"Detailed comparison: {context.result_file}")
        failures = [
            str(row["model"]) for row in rows
            if str(row["status"]).startswith("FAILED")
        ]
        if failures:
            any_failure = True
            print(
                f"Failed {context.label} baselines: {', '.join(failures)}",
                file=sys.stderr,
            )
            if args.fail_fast:
                break
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
