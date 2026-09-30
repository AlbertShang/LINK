"""Train, evaluate, and complete TraffiKrig datasets with LINK and SPN.

Examples from the repository root:

    python experiments/main.py --districts 3
    python experiments/main.py --mode evaluate --districts 3
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from dataclasses import asdict, replace
from pathlib import Path

# Some hosted environments export NUMEXPR_NUM_THREADS as the full CPU count
# while NumExpr is capped at 64.  Clamp it before importing pandas; this warning
# is unrelated to model training, but otherwise appears as a misleading error.
try:
    _NUMEXPR_MAX_THREADS = int(os.environ.get("NUMEXPR_MAX_THREADS", "64"))
except ValueError:
    _NUMEXPR_MAX_THREADS = 64
try:
    _NUMEXPR_THREADS = int(
        os.environ.get("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 1))
    )
except ValueError:
    _NUMEXPR_THREADS = _NUMEXPR_MAX_THREADS
os.environ["NUMEXPR_MAX_THREADS"] = str(_NUMEXPR_MAX_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(
    max(1, min(_NUMEXPR_THREADS, _NUMEXPR_MAX_THREADS))
)

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import (  # noqa: E402
    EvaluationConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    SUPPORTED_LOSSES,
    SUPPORTED_METRICS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_SCHEDULERS,
    SUPPORTED_TREND_LOSS_DOMAINS,
    TrainingConfig,
    experiment_config_from_dict,
    get_experiment_config,
)
from src.engine import LINKExperiment, safe_torch_load  # noqa: E402
from src.metrics import format_metrics  # noqa: E402


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _load_data(
    data_root: Path,
    district: int,
    top_k_override: int | None,
    max_time_steps: int | None,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    list[dict[str, object]],
    dict[str, np.ndarray],
    int,
    int,
]:
    directory = data_root / f"pems0{district}"
    config_path = directory / "preprocess_config.json"
    if top_k_override is not None:
        top_k = top_k_override
    elif config_path.is_file():
        top_k = int(json.loads(config_path.read_text(encoding="utf-8"))["top_k"])
    else:
        top_k = 3
    flow_path = directory / f"district_{district}_flow.npy"
    metadata_path = directory / f"district_{district}_meta_feature.csv"
    split_path = directory / "adj_index.pkl"
    subgraph_path = directory / f"adaptive_boundary_subgraphs_inout_topk{top_k}.pt"
    missing = [
        str(path)
        for path in (flow_path, metadata_path, split_path, subgraph_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "preprocessed inputs are missing; run data/preprocess.py first: " + ", ".join(missing)
        )
    flow = np.load(flow_path)
    if max_time_steps is not None:
        flow = flow[:max_time_steps]
    metadata = pd.read_csv(metadata_path, sep="\t")
    with split_path.open("rb") as handle:
        saved_splits = pickle.load(handle)
    split = {
        "train": np.asarray(saved_splits["train_nodes"], dtype=np.int64),
        "valid": np.asarray(saved_splits["valid_nodes"], dtype=np.int64),
        "test": np.asarray(saved_splits["test_nodes"], dtype=np.int64),
    }
    payload = safe_torch_load(subgraph_path)
    if int(payload.get("top_k", top_k)) != top_k:
        raise ValueError("saved subgraph payload and District Top-K config differ")
    for name, key in (("train", "train_nodes"), ("valid", "valid_nodes"), ("test", "test_nodes")):
        if not np.array_equal(split[name], np.asarray(payload[key])):
            raise ValueError(
                f"{name} split differs between adj_index.pkl and subgraph file"
            )
    subgraphs = payload["subgraphs"]
    edge_metadata_dim = int(payload["metadata_config"]["edge_metadata_dim"])
    if len(metadata) != flow.shape[1] or len(subgraphs) != flow.shape[1]:
        raise ValueError("flow, metadata, and subgraphs are not node-aligned")
    return flow, metadata, subgraphs, split, edge_metadata_dim, top_k


def _checkpoint_path(args: argparse.Namespace, district: int) -> Path:
    if args.checkpoint is not None:
        if len(args.districts) != 1:
            raise ValueError("--checkpoint can only be used with one district")
        return args.checkpoint.resolve()
    configured = get_experiment_config(district)
    directory = args.output_root / f"pems0{district}"
    if args.time_window_impute or configured.model.time_window_impute:
        directory = directory / "time_window_impute"
    loss_domain = args.trend_loss_domain
    if loss_domain is None:
        loss_domain = configured.loss.trend_loss_domain
    if loss_domain != "time":
        directory = directory / "with_frequency_domain_loss"
    if (
        args.subgraph_frequency_topk
        or configured.model.use_subgraph_frequency_topk
    ):
        directory = directory / "ablation_subgraph_frequency_topk"
    if (
        args.disable_adaptive_subgraph
        or not configured.model.use_adaptive_subgraphs
    ):
        directory = directory / "ablation_spectral_gnn"
    strict_no_fft = (
        args.disable_fft_encoding or not configured.model.use_fft_decomposition
    )
    temporal_spectral_ablation = (
        args.disable_spectral_propagation
        or (
            not configured.model.use_fft_encoding
            and configured.model.use_fft_decomposition
        )
    )
    if strict_no_fft:
        directory = directory / "ablation_no_fft"
    elif temporal_spectral_ablation:
        directory = directory / "ablation_no_spectral_propagation"
    if (
        not strict_no_fft
        and (
            args.disable_residual_refinement
            or not configured.model.use_residual_refinement
        )
    ):
        directory = directory / "ablation_no_residual"
    return directory / "best_model.pt"


def _checkpoint_config(
    checkpoint: dict[str, object], district: int
) -> ExperimentConfig:
    """Read the new nested config and remain compatible with existing checkpoints."""
    serialized = checkpoint.get("experiment_config")
    if isinstance(serialized, dict):
        return experiment_config_from_dict(serialized)

    defaults = get_experiment_config(district)
    model_values = {**asdict(defaults.model), **dict(checkpoint["model_config"])}
    training_values = {
        **asdict(defaults.training),
        **dict(checkpoint["training_config"]),
    }
    loss_values = {**asdict(defaults.loss), **dict(checkpoint.get("loss_config", {}))}
    evaluation_values = {
        **asdict(defaults.evaluation),
        **dict(checkpoint.get("evaluation_config", {})),
    }
    evaluation_values["metrics"] = tuple(evaluation_values["metrics"])
    return ExperimentConfig(
        model=ModelConfig(**model_values),
        training=TrainingConfig(**training_values),
        loss=LossConfig(**loss_values),
        evaluation=EvaluationConfig(**evaluation_values),
    )


def _configured_values(args: argparse.Namespace, names: tuple[str, ...]) -> dict[str, object]:
    return {
        name: getattr(args, name)
        for name in names
        if getattr(args, name) is not None
    }


def _apply_cli_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
    *,
    training_run: bool,
) -> ExperimentConfig:
    """Apply explicit CLI values; omitted arguments keep the Python config unchanged."""
    model = config.model
    training = config.training
    loss = config.loss
    if training_run:
        model_overrides = _configured_values(
            args,
            (
                "window_size",
                "trend_frequency_bins",
                "hidden_dim",
                "dropout",
                "day_points",
                "start_weekday",
                "branch_aggregation",
                "max_propagation_hop",
                "gnn_layers",
            ),
        )
        if args.disable_time_features:
            model_overrides["use_time_features"] = False
        if args.disable_fft_encoding:
            model_overrides["use_fft_encoding"] = False
            model_overrides["use_fft_decomposition"] = False
            model_overrides["use_residual_refinement"] = False
        elif args.disable_spectral_propagation:
            model_overrides["use_fft_encoding"] = False
        if args.subgraph_frequency_topk:
            model_overrides["use_subgraph_frequency_topk"] = True
        if args.time_window_impute:
            model_overrides["time_window_impute"] = True
        if args.disable_residual_refinement:
            model_overrides["use_residual_refinement"] = False
        if args.disable_metadata:
            model_overrides["use_metadata"] = False
        if args.metadata_only:
            model_overrides["metadata_only"] = True
        if args.disable_adaptive_subgraph:
            model_overrides["use_adaptive_subgraphs"] = False
        model = replace(
            model,
            **model_overrides,
        )
        training = replace(
            training,
            **_configured_values(
                args,
                (
                    "seed",
                    "trend_epochs",
                    "residual_epochs",
                    "early_stopping_patience",
                    "early_stopping_min_delta",
                    "optimizer",
                    "scheduler",
                    "learning_rate",
                    "weight_decay",
                    "optimizer_eps",
                    "gradient_clip",
                    "train_days_per_epoch",
                    "center_batch_size",
                    "validation_days_per_weekday",
                    "evaluate_every",
                    "evaluation_chunk",
                ),
            ),
        )
        loss = replace(
            loss,
            **_configured_values(
                args,
                (
                    "trend_loss",
                    "trend_loss_domain",
                    "trend_time_weight",
                    "trend_amplitude_weight",
                    "trend_phase_weight",
                    "residual_loss",
                    "final_loss",
                    "final_loss_weight",
                ),
            ),
        )
    elif args.evaluation_chunk is not None:
        training = replace(training, evaluation_chunk=args.evaluation_chunk)

    metrics = config.evaluation.metrics
    if args.eval_metrics is not None:
        metrics = tuple(args.eval_metrics)
    checkpoint_metric = args.checkpoint_metric
    if checkpoint_metric is None:
        checkpoint_metric = config.evaluation.checkpoint_metric
        if checkpoint_metric not in metrics:
            checkpoint_metric = metrics[0]
    evaluation = replace(
        config.evaluation,
        metrics=metrics,
        checkpoint_metric=checkpoint_metric,
        **_configured_values(args, ("checkpoint_mode",)),
    )
    return ExperimentConfig(
        model=model,
        training=training,
        loss=loss,
        evaluation=evaluation,
    )


def run_district(args: argparse.Namespace, district: int, device: torch.device) -> None:
    flow, metadata, subgraphs, split, edge_metadata_dim, top_k = _load_data(
        args.data_root, district, args.top_k, args.max_time_steps
    )
    checkpoint_path = _checkpoint_path(args, district)
    checkpoint = None
    if args.mode == "evaluate":
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = safe_torch_load(checkpoint_path, device)
        if int(checkpoint.get("district", district)) != district:
            raise ValueError("checkpoint district does not match requested district")
        if int(checkpoint.get("top_k", top_k)) != top_k:
            raise ValueError("checkpoint Top-K does not match the requested subgraphs")
        config = _apply_cli_overrides(
            _checkpoint_config(checkpoint, district), args, training_run=False
        )
    else:
        config = _apply_cli_overrides(
            get_experiment_config(district), args, training_run=True
        )
    experiment = LINKExperiment(
        flow,
        subgraphs,
        split,
        edge_metadata_dim,
        device,
        config,
    )
    output_dir = args.output_root / f"pems0{district}"
    if config.model.time_window_impute:
        output_dir = output_dir / "time_window_impute"
    if config.loss.trend_loss_domain != "time":
        output_dir = output_dir / "with_frequency_domain_loss"
    if config.model.use_subgraph_frequency_topk:
        output_dir = output_dir / "ablation_subgraph_frequency_topk"
    if not config.model.use_adaptive_subgraphs:
        output_dir = output_dir / "ablation_spectral_gnn"
    if not config.model.use_fft_decomposition:
        output_dir = output_dir / "ablation_no_fft"
    elif not config.model.use_fft_encoding:
        output_dir = output_dir / "ablation_no_spectral_propagation"
    if (
        config.model.use_fft_decomposition
        and not config.model.use_residual_refinement
    ):
        output_dir = output_dir / "ablation_no_residual"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_experiment_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("=" * 80)
    print(
        f"PEMS0{district} | device={device} | flow={tuple(flow.shape)} | Top-{top_k}"
    )
    split_sizes = "/".join(str(len(split[name])) for name in ("train", "valid", "test"))
    total_parameters = sum(
        parameter.numel() for parameter in experiment.model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in experiment.model.parameters()
        if parameter.requires_grad
    )
    print(
        f"nodes train/valid/test={split_sizes} | "
        f"parameters trainable/total={trainable_parameters:,}/{total_parameters:,}"
    )
    if config.model.metadata_only:
        print(
            "features input=focal Type/Direction + "
            f"{'TOD/DOW' if config.model.use_time_features else 'no calendar'} | "
            "traffic=off | topology=off | target-decomposition="
            f"{'FFT-lowpass' if config.model.use_fft_decomposition else 'none/raw-flow'}"
        )
    else:
        if not config.model.use_fft_decomposition:
            frequency_selection = "n/a"
        elif config.model.use_subgraph_frequency_topk:
            frequency_selection = "subgraph-boundary-topk"
        else:
            frequency_selection = "fixed-low-k"
        graph_variant = (
            "adaptive-subgraph + shared-edge-MLP"
            if config.model.use_adaptive_subgraphs
            else "full-graph spectral-GNN"
        )
        print(
            "features "
            f"propagation-domain="
            f"{'spectral-coefficients' if config.model.use_fft_encoding else 'temporal-windows'} | "
            f"decomposition="
            f"{'FFT-lowpass' if config.model.use_fft_decomposition else 'none/raw-flow'} | "
            f"frequency-selection="
            f"{frequency_selection} | "
            f"graph={graph_variant} | "
            f"metadata={'on' if config.model.use_metadata else 'off'} | "
            f"TOD/DOW={'on' if config.model.use_time_features else 'off'} | "
            f"residual={'on' if config.model.use_residual_refinement else 'off'} | "
            f"trend-loss-domain={config.loss.trend_loss_domain}"
        )
        if config.model.time_window_impute:
            print(
                "imputation=whole-window | "
                f"window={config.model.window_size} | "
                f"frequency bins={config.model.trend_frequency_bins} | "
                "stride=window_size | supervision=all window points"
            )
        if config.loss.trend_loss_domain != "time":
            print(
                "trend loss weights "
                f"time={config.loss.trend_time_weight:g} | "
                f"log-amplitude={config.loss.trend_amplitude_weight:g} | "
                f"circular-phase={config.loss.trend_phase_weight:g}"
            )
    if not config.model.use_adaptive_subgraphs and not config.model.metadata_only:
        print(
            f"spectral GNN layers={config.model.gnn_layers} | "
            "weighted directed adjacency + self transformation"
        )
    if args.mode == "train":
        windows_per_day = (
            config.model.day_points // config.model.window_size
            if config.model.time_window_impute
            else 1
        )
        steps_per_epoch = (
            len(experiment.training_graphs)
            * config.training.train_days_per_epoch
            * windows_per_day
        )
        scheduler_label = (
            f"{config.training.scheduler}/optimizer-step"
            if config.training.scheduler != "none"
            else "none"
        )
        print(
            f"train center batch<={config.training.center_batch_size} | "
            f"center batches={len(experiment.training_graphs)} | "
            f"days/epoch={config.training.train_days_per_epoch} | "
            f"optimizer steps/epoch={steps_per_epoch} | scheduler={scheduler_label}"
        )
    if args.mode == "train":
        checkpoint = experiment.train()
        checkpoint.update(
            {
                "method": "LINK",
                "model": "SPN",
                "dataset": "TraffiKrig",
                "district": district,
                "top_k": top_k,
                "station_ids": metadata[args.station_id].to_numpy(),
                "train_nodes": split["train"],
                "valid_nodes": split["valid"],
                "test_nodes": split["test"],
            }
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"checkpoint: {checkpoint_path}")
    else:
        assert checkpoint is not None
        experiment.restore(checkpoint)

    metrics = experiment.metrics()
    completed, completion_valid = experiment.completion()
    np.save(output_dir / "completed_flow.npy", completed)
    np.save(output_dir / "completion_valid_mask.npy", completion_valid)
    result = {
        "method": "LINK",
        "model": "SPN",
        "dataset": "TraffiKrig",
        "district": district,
        "mode": args.mode,
        "checkpoint": str(checkpoint_path),
        "flow_shape": list(flow.shape),
        "completion_shape": list(completed.shape),
        "completion_semantics": (
            "original observations on train nodes; model predictions on validation/test nodes"
        ),
        "graph_variant": (
            "adaptive_subgraph_shared_mlp"
            if config.model.use_adaptive_subgraphs
            else "full_graph_spectral_gnn"
        ),
        "trend_loss_domain": config.loss.trend_loss_domain,
        "trend_loss_weights": {
            "time": config.loss.trend_time_weight,
            "amplitude": config.loss.trend_amplitude_weight,
            "phase": config.loss.trend_phase_weight,
        },
        **metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"validation {format_metrics(metrics['validation'])}")
    print(f"test {format_metrics(metrics['test'])}")
    print(f"completion: {output_dir / 'completed_flow.npy'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--districts", nargs="+", type=int, default=[3, 4, 7, 8])
    parser.add_argument("--data-root", type=Path, default=ROOT / "pure_dataset_topk")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, help="override the saved District config")
    parser.add_argument("--station-id", default="station_id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--trend-frequency-bins", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--disable-fft-encoding",
        action="store_true",
        help=(
            "strict w/o FFT ablation: skip FFT decomposition, propagate raw "
            "causal flow windows in the time domain, supervise the final "
            "window point, and disable residual refinement"
        ),
    )
    parser.add_argument(
        "--disable-spectral-propagation",
        action="store_true",
        help=(
            "keep FFT low-pass decomposition and residual refinement, but "
            "reconstruct and propagate complete time-domain trend windows"
        ),
    )
    parser.add_argument(
        "--time-window-impute",
        "--time_window_impute",
        dest="time_window_impute",
        action="store_true",
        help=(
            "enable window-level spectral imputation: propagate one set of "
            "low-frequency coefficients per non-overlapping temporal window, "
            "decode the complete focal window, and supervise all window points"
        ),
    )
    parser.add_argument(
        "--subgraph-frequency-topk",
        "--subgraph_frequency_topk",
        dest="subgraph_frequency_topk",
        action="store_true",
        help=(
            "select one shared set of K Fourier bins per adaptive subgraph and "
            "causal window from observed boundary amplitudes; retain DC and "
            "choose the remaining bins by mean boundary amplitude"
        ),
    )
    parser.add_argument(
        "--disable-time-features",
        action="store_true",
        help=(
            "ablate TOD/DOW during training by replacing their four input "
            "features with zeros while keeping the model shape unchanged"
        ),
    )
    parser.add_argument(
        "--disable-metadata",
        action="store_true",
        help=(
            "ablate static road/sensor metadata by replacing Type, Direction, "
            "same-freeway, and same-direction edge features with zeros; keep "
            "subgraphs, graph features, model shape, and parameter count unchanged"
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "train a direct focal-node metadata baseline using only Type, "
            "Direction, and (unless disabled) TOD/DOW; traffic values, edge "
            "metadata, adjacency, subgraphs, weights, directions, and degrees "
            "are not read by the predictor"
        ),
    )
    parser.add_argument(
        "--disable-adaptive-subgraph",
        "--spectral-gnn-ablation",
        dest="disable_adaptive_subgraph",
        action="store_true",
        help=(
            "replace query-centered subgraphs and recurrent edge MLP propagation "
            "with a weighted GNN over the complete directed road graph; keep "
            "the FFT features, metadata, targets, splits, and residual stage"
        ),
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        help="number of message-passing layers in the full-graph spectral GNN",
    )
    parser.add_argument(
        "--disable-residual-refinement",
        action="store_true",
        help=(
            "ablate the complete residual stage: train only the trend branch and "
            "use ReLU(trend) directly for validation, testing, and completion"
        ),
    )
    parser.add_argument("--day-points", type=int)
    parser.add_argument("--start-weekday", type=int)
    parser.add_argument("--branch-aggregation", choices=("mean", "sum"))
    parser.add_argument("--max-propagation-hop", type=int)
    parser.add_argument("--trend-epochs", type=int)
    parser.add_argument("--residual-epochs", type=int)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help=(
            "stop a training stage after this many consecutive validation "
            "evaluations without checkpoint-metric improvement"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        help=(
            "minimum absolute checkpoint-metric improvement required to reset "
            "early-stopping patience"
        ),
    )
    parser.add_argument("--optimizer", choices=tuple(sorted(SUPPORTED_OPTIMIZERS)))
    parser.add_argument("--scheduler", choices=tuple(sorted(SUPPORTED_SCHEDULERS)))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--optimizer-eps", type=float)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--train-days-per-epoch", type=int)
    parser.add_argument(
        "--center-batch-size",
        type=int,
        help="maximum pseudo-focal train centers in each optimizer step",
    )
    parser.add_argument("--validation-days-per-weekday", type=int)
    parser.add_argument("--evaluate-every", type=int)
    parser.add_argument("--evaluation-chunk", type=int)
    parser.add_argument("--trend-loss", choices=tuple(sorted(SUPPORTED_LOSSES)))
    parser.add_argument(
        "--trend-loss-domain",
        choices=tuple(sorted(SUPPORTED_TREND_LOSS_DOMAINS)),
        help=(
            "supervise the decoded trend in time, retained amplitude/phase in "
            "frequency, or both with hybrid"
        ),
    )
    parser.add_argument("--trend-time-weight", type=float)
    parser.add_argument("--trend-amplitude-weight", type=float)
    parser.add_argument("--trend-phase-weight", type=float)
    parser.add_argument("--residual-loss", choices=tuple(sorted(SUPPORTED_LOSSES)))
    parser.add_argument("--final-loss", choices=tuple(sorted(SUPPORTED_LOSSES)))
    parser.add_argument("--final-loss-weight", type=float)
    parser.add_argument(
        "--eval-metrics",
        nargs="+",
        choices=tuple(sorted(SUPPORTED_METRICS)),
    )
    parser.add_argument(
        "--checkpoint-metric", choices=tuple(sorted(SUPPORTED_METRICS))
    )
    parser.add_argument("--checkpoint-mode", choices=("min", "max"))
    parser.add_argument(
        "--max-time-steps",
        type=int,
        help="optional debugging limit; normal experiments should use the full four months",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    device = _device(args.device)
    for district in args.districts:
        run_district(args, district, device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
