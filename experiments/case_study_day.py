"""One-day reconstruction case study with a selectable set of models.

When no node is supplied, the script ranks held-out nodes by the reference
model's MAE improvement over the strongest selected baseline on their common
raw test points.  It then chooses the displayed day using ground-truth
variability only.  Interpolated values are exported for publication plotting;
all reported errors continue to use raw valid observations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from case_study_common import (
    ModelSpec,
    ROOT,
    add_shared_arguments,
    canonical_type,
    choose_day_window,
    common_valid_mask,
    interpolate_series,
    load_context,
    load_predictions,
    parse_model_spec,
    station_id,
    timestamps,
    write_json,
)


DEFAULT_MODELS = [
    ModelSpec("LINK", "metadata"),
    ModelSpec("STAGANN", "metadata"),
    ModelSpec("KITS", "metadata"),
    ModelSpec("IGNNK", "metadata"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.add_argument(
        "--models", nargs="+", type=parse_model_spec, default=DEFAULT_MODELS,
        metavar="MODEL:MODE",
        help="models to compare, e.g. LINK:metadata STAGANN:metadata",
    )
    parser.add_argument(
        "--reference", type=parse_model_spec, default=ModelSpec("LINK", "metadata")
    )
    parser.add_argument("--node-index", type=int)
    parser.add_argument("--station-id")
    parser.add_argument(
        "--node-type", default="any",
        help="optional auto-selection filter, e.g. Mainline or On Ramp",
    )
    parser.add_argument("--day", help="explicit YYYY-MM-DD; otherwise selected from truth variability")
    parser.add_argument("--day-points", type=int, default=288)
    parser.add_argument("--min-day-valid-ratio", type=float, default=0.8)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "case_study" / "one_day",
    )
    args = parser.parse_args()
    if args.reference.key not in {model.key for model in args.models}:
        parser.error("--reference must also be present in --models")
    if len(args.models) < 2:
        parser.error("case study requires at least two models")
    if args.node_index is not None and args.station_id is not None:
        parser.error("use only one of --node-index and --station-id")
    if args.day_points <= 1 or args.inference_batch_size <= 0:
        parser.error("--day-points and --inference-batch-size must be positive")
    if not 0 < args.min_day_valid_ratio <= 1:
        parser.error("--min-day-valid-ratio must be in (0,1]")
    return args


def resolve_requested_node(args, context) -> int | None:
    if args.node_index is not None:
        if args.node_index not in set(context.test_nodes.tolist()):
            raise ValueError(f"Node {args.node_index} is not a held-out test node")
        return args.node_index
    if args.station_id is not None:
        matches = [
            node for node in context.test_nodes
            if station_id(context.metadata_rows[int(node)], int(node)) == str(args.station_id)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Station ID {args.station_id!r} matched {len(matches)} held-out test nodes"
            )
        return int(matches[0])
    return None


def rank_nodes(args, context, predictions) -> list[dict[str, object]]:
    baseline_keys = [key for key in predictions if key != args.reference.key]
    requested_type = "" if args.node_type.lower() == "any" else "".join(
        character for character in args.node_type.lower() if character.isalpha()
    )
    ranking: list[dict[str, object]] = []
    for node_value in context.test_nodes:
        node = int(node_value)
        row = context.metadata_rows[node]
        if requested_type and canonical_type(row) != requested_type:
            continue
        valid = common_valid_mask(context.truth, predictions, [node])[:, 0]
        if valid.sum() < args.day_points:
            continue
        truth = context.truth[valid, node]
        maes = {
            key: float(np.mean(np.abs(prediction[valid, node] - truth)))
            for key, prediction in predictions.items()
        }
        strongest_baseline = min(baseline_keys, key=lambda key: maes[key])
        ranking.append({
            "node_index": node,
            "station_id": station_id(row, node),
            "node_type": canonical_type(row),
            "valid_points": int(valid.sum()),
            "reference_mae": maes[args.reference.key],
            "strongest_baseline": strongest_baseline,
            "strongest_baseline_mae": maes[strongest_baseline],
            "reference_advantage": maes[strongest_baseline] - maes[args.reference.key],
            **{f"mae::{key}": value for key, value in maes.items()},
        })
    ranking.sort(key=lambda item: float(item["reference_advantage"]), reverse=True)
    if not ranking:
        raise RuntimeError("No held-out node satisfies the requested type and validity constraints")
    return ranking


def write_ranking(path: Path, ranking: list[dict[str, object]]) -> None:
    keys = list(ranking[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(ranking)


def export_curves(args, context, predictions, node: int, start: int, end: int) -> dict:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dates = timestamps(start, end, args.start_date)
    raw_truth = context.truth[start:end, node]
    raw_valid = np.isfinite(raw_truth) & (np.abs(raw_truth) > 5e-5)
    raw = {key: prediction[start:end, node] for key, prediction in predictions.items()}
    interpolated = {"truth": interpolate_series(raw_truth, raw_valid)}
    interpolated.update({key: interpolate_series(values) for key, values in raw.items()})

    columns = ["time_index", "timestamp", "truth_valid", "truth", *predictions]
    for filename, use_interpolated in (
        ("selected_curves_raw.csv", False),
        ("selected_curves_interpolated.csv", True),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for offset, timestamp in enumerate(dates):
                values = interpolated if use_interpolated else {"truth": raw_truth, **raw}
                writer.writerow({
                    "time_index": start + offset,
                    "timestamp": timestamp.isoformat(),
                    "truth_valid": int(raw_valid[offset]),
                    **{key: float(series[offset]) for key, series in values.items()},
                })
    np.savez_compressed(
        output / "selected_curves_interpolated.npz",
        time_index=np.arange(start, end),
        timestamp=np.asarray([value.isoformat() for value in dates]),
        truth_valid=raw_valid,
        truth=interpolated["truth"],
        **{spec.slug: interpolated[spec.key] for spec in args.models},
    )

    metrics = {}
    for key, values in raw.items():
        valid = raw_valid & np.isfinite(values)
        error = values[valid] - raw_truth[valid]
        metrics[key] = {
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(np.square(error)))),
            "valid_points": int(valid.sum()),
        }

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(12, 4.5))
        axis.plot(dates, interpolated["truth"], color="black", linewidth=2.4, label="Ground truth")
        for spec in args.models:
            axis.plot(dates, interpolated[spec.key], linewidth=1.5, label=spec.label)
        axis.set_ylabel("Traffic flow")
        axis.set_xlabel("Time")
        axis.grid(alpha=0.2)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.legend(ncol=min(3, len(args.models) + 1), frameon=False)
        figure.tight_layout()
        figure.savefig(output / "selected_curves_preview.png", dpi=220, bbox_inches="tight")
        plt.close(figure)
    except ImportError:
        print("matplotlib is unavailable; CSV/NPZ exports were still written", flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    context = load_context(args)
    predictions, provenance = load_predictions(args.models, args, context)
    ranking = rank_nodes(args, context, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_ranking(args.output_dir / "candidate_nodes.csv", ranking)
    requested = resolve_requested_node(args, context)
    if requested is None:
        selected = ranking[0]
    else:
        selected = next((item for item in ranking if item["node_index"] == requested), None)
        if selected is None:
            raise RuntimeError(f"Requested node {requested} lacks enough common valid predictions")
    node = int(selected["node_index"])
    start, end, day_rule = choose_day_window(
        context.truth, predictions, [node], args.day_points,
        args.min_day_valid_ratio, args.day, args.start_date,
    )
    metrics = export_curves(args, context, predictions, node, start, end)
    payload = {
        "district": context.district,
        "top_k": context.top_k,
        "models": [spec.key for spec in args.models],
        "reference": args.reference.key,
        "node_selection": "explicit" if requested is not None else "largest_reference_advantage_over_strongest_baseline",
        "selected_node": selected,
        "window": {"start": start, "end": end, "day": timestamps(start, start + 1, args.start_date)[0].date().isoformat(), "selection": day_rule},
        "raw_window_metrics": metrics,
        "prediction_provenance": provenance,
        "note": "Interpolation is used only in exported display curves; ranking and errors use raw valid values.",
    }
    write_json(args.output_dir / "selection.json", payload)
    print(
        f"Selected node={node} station={selected['station_id']} type={selected['node_type']} "
        f"day={payload['window']['day']} output={args.output_dir}", flush=True,
    )


if __name__ == "__main__":
    main()
