"""Find two nearby On Ramp nodes and one Mainline node for a scale case study.

The selected triplet must be geographically close and have different
ground-truth traffic scales.  Among valid triplets, selection favors cases
where the designated smoothing-prone baselines retain less of the true
between-node scale range.  Results include candidate rankings, raw metrics,
interpolated curve values, and a preview with one panel per method.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
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
    latitude_longitude,
    load_context,
    load_predictions,
    parse_model_spec,
    station_id,
    timestamps,
    write_json,
)


DEFAULT_MODELS = [
    ModelSpec("LINK", "metadata"),
    ModelSpec("IGNNK", "no_metadata"),
    ModelSpec("STAGANN", "no_metadata"),
    ModelSpec("KITS", "no_metadata"),
]
DEFAULT_SMOOTHING_MODELS = [
    ModelSpec("IGNNK", "no_metadata"),
    ModelSpec("STAGANN", "no_metadata"),
    ModelSpec("KITS", "no_metadata"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.add_argument("--models", nargs="+", type=parse_model_spec, default=DEFAULT_MODELS)
    parser.add_argument(
        "--reference", type=parse_model_spec, default=ModelSpec("LINK", "metadata"),
        help="method expected to preserve the true between-node scale distinction",
    )
    parser.add_argument(
        "--smoothing-models", nargs="+", type=parse_model_spec,
        default=DEFAULT_SMOOTHING_MODELS,
        help="selected models whose between-node scale compression enters triplet ranking",
    )
    parser.add_argument(
        "--nodes", nargs=3, type=int, metavar=("NODE1", "NODE2", "NODE3"),
        help="explicit held-out nodes; must contain two On Ramp and one Mainline",
    )
    parser.add_argument("--max-distance-km", type=float, default=8.0)
    parser.add_argument("--min-truth-scale-ratio", type=float, default=2.0)
    parser.add_argument("--min-pairwise-relative-gap", type=float, default=0.2)
    parser.add_argument("--min-reference-retention", type=float, default=0.7)
    parser.add_argument("--max-median-smoothing-retention", type=float, default=0.7)
    parser.add_argument("--scale-statistic", choices=("median", "mean"), default="median")
    parser.add_argument("--day", help="explicit YYYY-MM-DD; otherwise selected from truth variability")
    parser.add_argument("--day-points", type=int, default=288)
    parser.add_argument("--min-day-valid-ratio", type=float, default=0.8)
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "case_study" / "scale_compression",
    )
    args = parser.parse_args()
    model_keys = {model.key for model in args.models}
    if args.reference.key not in model_keys:
        parser.error("--reference must also be present in --models")
    missing = [model.key for model in args.smoothing_models if model.key not in model_keys]
    if missing:
        parser.error(f"--smoothing-models must be included in --models; missing {missing}")
    if args.max_distance_km <= 0 or args.min_truth_scale_ratio <= 1:
        parser.error("--max-distance-km must be positive and --min-truth-scale-ratio > 1")
    if not 0 <= args.min_pairwise_relative_gap < 1:
        parser.error("--min-pairwise-relative-gap must be in [0,1)")
    if args.min_reference_retention < 0 or args.max_median_smoothing_retention < 0:
        parser.error("scale-retention thresholds must be nonnegative")
    if not 0 < args.min_day_valid_ratio <= 1:
        parser.error("--min-day-valid-ratio must be in (0,1]")
    return args


def haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def scale_value(values: np.ndarray, statistic: str) -> float:
    return float(np.mean(values) if statistic == "mean" else np.median(values))


def node_scale_table(args, context, predictions) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for raw_node in context.test_nodes:
        node = int(raw_node)
        valid = common_valid_mask(context.truth, predictions, [node])[:, 0]
        if valid.sum() < args.day_points:
            continue
        truth_values = context.truth[valid, node]
        scales = {
            "truth": scale_value(truth_values, args.scale_statistic),
            "valid_points": float(valid.sum()),
        }
        scales.update({
            key: scale_value(prediction[valid, node], args.scale_statistic)
            for key, prediction in predictions.items()
        })
        result[node] = scales
    return result


def triplet_record(args, context, scales, nodes: tuple[int, int, int]) -> dict[str, object] | None:
    coordinates = [latitude_longitude(context.metadata_rows[node]) for node in nodes]
    if not all(np.isfinite(value) for pair in coordinates for value in pair):
        return None
    distances = [
        haversine_km(coordinates[first], coordinates[second])
        for first, second in itertools.combinations(range(3), 2)
    ]
    truth = np.asarray([scales[node]["truth"] for node in nodes], dtype=float)
    if not np.all(np.isfinite(truth)) or np.min(truth) <= 0:
        return None
    truth_range = float(np.ptp(truth))
    ratio = float(np.max(truth) / np.min(truth))
    pairwise_gap = min(
        abs(truth[first] - truth[second]) / max(truth[first], truth[second])
        for first, second in itertools.combinations(range(3), 2)
    )
    retention = {}
    for model in args.smoothing_models:
        predicted = np.asarray([scales[node][model.key] for node in nodes], dtype=float)
        retention[model.key] = float(np.ptp(predicted) / max(truth_range, 1e-8))
    median_retention = float(np.median(list(retention.values())))
    reference_values = np.asarray(
        [scales[node][args.reference.key] for node in nodes], dtype=float
    )
    reference_retention = float(np.ptp(reference_values) / max(truth_range, 1e-8))
    max_distance = float(max(distances))
    passes = (
        max_distance <= args.max_distance_km
        and ratio >= args.min_truth_scale_ratio
        and pairwise_gap >= args.min_pairwise_relative_gap
        and reference_retention >= args.min_reference_retention
        and median_retention <= args.max_median_smoothing_retention
    )
    score = (
        math.log(ratio)
        + min(reference_retention, 2.0)
        - median_retention
        - max_distance / args.max_distance_km
        + pairwise_gap
    )
    record: dict[str, object] = {
        "nodes": nodes,
        "onramp_1": nodes[0],
        "onramp_2": nodes[1],
        "mainline": nodes[2],
        "max_distance_km": max_distance,
        "truth_scale_ratio": ratio,
        "min_pairwise_relative_gap": float(pairwise_gap),
        "median_scale_retention": median_retention,
        "reference_scale_retention": reference_retention,
        "passes_constraints": passes,
        "selection_score": float(score),
    }
    record.update({f"retention::{key}": value for key, value in retention.items()})
    return record


def enumerate_triplets(args, context, scales) -> list[dict[str, object]]:
    onramps = [
        int(node) for node in context.test_nodes
        if int(node) in scales and canonical_type(context.metadata_rows[int(node)]) == "onramp"
    ]
    mainlines = [
        int(node) for node in context.test_nodes
        if int(node) in scales and canonical_type(context.metadata_rows[int(node)]) == "mainline"
    ]
    if len(onramps) < 2 or not mainlines:
        raise RuntimeError(
            f"Need at least two valid held-out On Ramp nodes and one Mainline; found {len(onramps)} and {len(mainlines)}"
        )
    records = []
    for first, second in itertools.combinations(onramps, 2):
        for mainline in mainlines:
            record = triplet_record(args, context, scales, (first, second, mainline))
            if record is not None:
                records.append(record)
    records.sort(
        key=lambda item: (bool(item["passes_constraints"]), float(item["selection_score"])),
        reverse=True,
    )
    if not records:
        raise RuntimeError("No candidate triplet has usable latitude/longitude and scale values")
    return records


def validate_explicit_nodes(args, context, scales) -> tuple[int, int, int]:
    nodes = tuple(args.nodes)
    if len(set(nodes)) != 3 or any(node not in set(context.test_nodes.tolist()) for node in nodes):
        raise ValueError("--nodes must be three distinct held-out test-node indices")
    if any(node not in scales for node in nodes):
        raise ValueError("At least one explicit node lacks enough common valid predictions")
    onramps = [node for node in nodes if canonical_type(context.metadata_rows[node]) == "onramp"]
    mainlines = [node for node in nodes if canonical_type(context.metadata_rows[node]) == "mainline"]
    if len(onramps) != 2 or len(mainlines) != 1:
        raise ValueError("--nodes must contain exactly two On Ramp nodes and one Mainline node")
    return onramps[0], onramps[1], mainlines[0]


def flatten_record(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "nodes"}


def export_candidate_table(path: Path, records: list[dict[str, object]], limit: int) -> None:
    rows = [flatten_record(record) for record in records[:limit]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_scale_summary(path: Path, args, context, predictions, nodes, scales) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = ["truth", *predictions]
    truth_values = np.asarray([scales[node]["truth"] for node in nodes])
    truth_range = float(np.ptp(truth_values))
    for method in methods:
        values = np.asarray([scales[node][method] for node in nodes])
        ratio = float(np.max(values) / max(np.min(values), 1e-8))
        retention = 1.0 if method == "truth" else float(np.ptp(values) / max(truth_range, 1e-8))
        for node, value in zip(nodes, values):
            row = context.metadata_rows[node]
            rows.append({
                "method": method,
                "node_index": node,
                "station_id": station_id(row, node),
                "node_type": canonical_type(row),
                "scale_statistic": args.scale_statistic,
                "scale_value": float(value),
                "triplet_scale_ratio": ratio,
                "scale_range_retention": retention,
            })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def export_curves(args, context, predictions, nodes, start, end) -> dict[str, object]:
    output = args.output_dir
    dates = timestamps(start, end, args.start_date)
    methods = {"truth": context.truth, **predictions}
    interpolated: dict[str, np.ndarray] = {}
    for method, matrix in methods.items():
        series = []
        for node in nodes:
            raw = matrix[start:end, node]
            valid = None
            if method == "truth":
                valid = np.isfinite(raw) & (np.abs(raw) > 5e-5)
            series.append(interpolate_series(raw, valid))
        interpolated[method] = np.stack(series, axis=1)

    fields = [
        "time_index", "timestamp", "node_index", "station_id", "node_type",
        "truth_valid", *methods,
    ]
    for filename, use_interpolated in (
        ("selected_curves_raw.csv", False),
        ("selected_curves_interpolated.csv", True),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for offset, timestamp in enumerate(dates):
                for column, node in enumerate(nodes):
                    row = context.metadata_rows[node]
                    truth_raw = context.truth[start + offset, node]
                    source = interpolated if use_interpolated else {
                        method: matrix[start:end, nodes] for method, matrix in methods.items()
                    }
                    writer.writerow({
                        "time_index": start + offset,
                        "timestamp": timestamp.isoformat(),
                        "node_index": node,
                        "station_id": station_id(row, node),
                        "node_type": canonical_type(row),
                        "truth_valid": int(np.isfinite(truth_raw) and abs(truth_raw) > 5e-5),
                        **{method: float(values[offset, column]) for method, values in source.items()},
                    })
    np.savez_compressed(
        output / "selected_curves_interpolated.npz",
        time_index=np.arange(start, end),
        timestamp=np.asarray([value.isoformat() for value in dates]),
        nodes=np.asarray(nodes),
        truth=interpolated["truth"],
        **{spec.slug: interpolated[spec.key] for spec in args.models},
    )

    window_summary: dict[str, object] = {}
    truth_scale = np.asarray([
        scale_value(interpolated["truth"][:, column], args.scale_statistic)
        for column in range(3)
    ])
    truth_range = float(np.ptp(truth_scale))
    for method, values in interpolated.items():
        method_scale = np.asarray([
            scale_value(values[:, column], args.scale_statistic) for column in range(3)
        ])
        window_summary[method] = {
            "node_scales": method_scale.tolist(),
            "scale_ratio": float(np.max(method_scale) / max(np.min(method_scale), 1e-8)),
            "scale_range_retention": (
                1.0 if method == "truth" else float(np.ptp(method_scale) / max(truth_range, 1e-8))
            ),
        }

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            len(methods), 1, figsize=(12, 2.65 * len(methods)), sharex=True, sharey=True,
        )
        axes = np.atleast_1d(axes)
        colors = ("#2ca02c", "#d62728", "#1f77b4")
        for axis, (method, values) in zip(axes, interpolated.items()):
            title = "Ground truth" if method == "truth" else next(
                spec.label for spec in args.models if spec.key == method
            )
            for column, (node, color) in enumerate(zip(nodes, colors)):
                row = context.metadata_rows[node]
                axis.plot(
                    dates, values[:, column], color=color, linewidth=1.5,
                    label=f"{canonical_type(row)} {station_id(row, node)}",
                )
            axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
            axis.set_ylabel("Flow")
            axis.grid(alpha=0.2)
            axis.legend(ncol=3, frameon=False, fontsize=8)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[-1].set_xlabel("Time")
        figure.tight_layout()
        figure.savefig(output / "scale_compression_preview.png", dpi=220, bbox_inches="tight")
        plt.close(figure)
    except ImportError:
        print("matplotlib is unavailable; CSV/NPZ exports were still written", flush=True)
    return window_summary


def main() -> None:
    args = parse_args()
    context = load_context(args)
    predictions, provenance = load_predictions(args.models, args, context)
    scales = node_scale_table(args, context, predictions)
    records = enumerate_triplets(args, context, scales)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_candidate_table(args.output_dir / "triplet_candidates.csv", records, args.candidate_limit)
    if args.nodes:
        nodes = validate_explicit_nodes(args, context, scales)
        selected = triplet_record(args, context, scales, nodes)
        if selected is None:
            raise ValueError("Explicit nodes lack valid latitude/longitude or positive scale values")
        selection_rule = "explicit"
    else:
        selected = records[0]
        nodes = tuple(selected["nodes"])
        selection_rule = "constraints_then_truth_scale_and_baseline_compression"
        if not selected["passes_constraints"]:
            print(
                "Warning: no triplet met every distance/scale constraint; using the highest-scoring fallback. "
                "See triplet_candidates.csv and selection.json.", flush=True,
            )
    scale_rows = export_scale_summary(
        args.output_dir / "full_period_scale_summary.csv",
        args, context, predictions, nodes, scales,
    )
    start, end, day_rule = choose_day_window(
        context.truth, predictions, list(nodes), args.day_points,
        args.min_day_valid_ratio, args.day, args.start_date,
    )
    window_summary = export_curves(args, context, predictions, nodes, start, end)
    selected_nodes = []
    for node in nodes:
        row = context.metadata_rows[node]
        lat, lng = latitude_longitude(row)
        selected_nodes.append({
            "node_index": node,
            "station_id": station_id(row, node),
            "node_type": canonical_type(row),
            "latitude": lat,
            "longitude": lng,
            "truth_scale": scales[node]["truth"],
        })
    payload = {
        "district": context.district,
        "top_k": context.top_k,
        "models": [model.key for model in args.models],
        "reference": args.reference.key,
        "smoothing_models": [model.key for model in args.smoothing_models],
        "selection_rule": selection_rule,
        "constraints": {
            "max_distance_km": args.max_distance_km,
            "min_truth_scale_ratio": args.min_truth_scale_ratio,
            "min_pairwise_relative_gap": args.min_pairwise_relative_gap,
            "min_reference_retention": args.min_reference_retention,
            "max_median_smoothing_retention": args.max_median_smoothing_retention,
            "scale_statistic": args.scale_statistic,
        },
        "selected_triplet": flatten_record(selected),
        "selected_nodes": selected_nodes,
        "window": {
            "start": start,
            "end": end,
            "day": timestamps(start, start + 1, args.start_date)[0].date().isoformat(),
            "selection": day_rule,
        },
        "full_period_scale_rows": scale_rows,
        "window_scale_summary": window_summary,
        "prediction_provenance": provenance,
        "note": "Triplet selection is an illustrative case study. Interpolation is used only for exported curves; full-period selection statistics use raw common valid points.",
    }
    write_json(args.output_dir / "selection.json", payload)
    print(
        f"Selected nodes={nodes} day={payload['window']['day']} "
        f"truth_scale_ratio={selected['truth_scale_ratio']:.3f} "
        f"median_retention={selected['median_scale_retention']:.3f} "
        f"output={args.output_dir}", flush=True,
    )


if __name__ == "__main__":
    main()
