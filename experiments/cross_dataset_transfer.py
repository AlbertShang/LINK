"""Zero-shot transfer of a frozen PEMS03 LINK checkpoint to PEMS04/07/08."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the training entry point's environment setup and data contracts.
from experiments.main import _checkpoint_config, _device, _load_data
import numpy as np
import torch
from src.engine import LINKExperiment, safe_torch_load
from src.metrics import format_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/pems03/best_model.pt')
    parser.add_argument('--target-districts', nargs='+', type=int, choices=(4, 7, 8), default=[4, 7, 8])
    parser.add_argument('--data-root', type=Path, default=ROOT / 'pure_dataset_topk')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/ablation_cross_dataset/pems03')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--evaluation-chunk', type=int)
    parser.add_argument('--max-time-steps', type=int, help='debugging only; omit for reported results')
    parser.add_argument('--include-source', action='store_true', help='also evaluate PEMS03 as an in-domain reference')
    parser.add_argument('--save-completion', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if len(set(args.target_districts)) != len(args.target_districts):
        raise ValueError('target districts must be unique')
    if args.max_time_steps is not None and args.max_time_steps <= 0:
        raise ValueError('--max-time-steps must be positive')
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'PEMS03 checkpoint not found: {checkpoint_path}; train with experiments/main.py --districts 3 first')
    checkpoint = safe_torch_load(checkpoint_path)
    if checkpoint.get('district') != 3:
        raise ValueError('transfer requires a checkpoint explicitly tagged district=3')
    config = _checkpoint_config(checkpoint, 3)
    if args.evaluation_chunk is not None:
        config = replace(config, training=replace(config.training, evaluation_chunk=args.evaluation_chunk))
    scale = float(checkpoint['flow_scale'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('checkpoint has an invalid flow scale')
    if 'top_k' not in checkpoint or int(checkpoint['top_k']) <= 0:
        raise ValueError('checkpoint must record a positive top_k')
    top_k = int(checkpoint['top_k'])
    device = _device(args.device)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results, rows = [], []
    districts = ([3] if args.include_source else []) + args.target_districts
    for district in districts:
        flow, _, subgraphs, split, edge_dim, _ = _load_data(
            args.data_root.resolve(), district, top_k, args.max_time_steps
        )
        experiment = LINKExperiment(flow, subgraphs, split, edge_dim, device, config)
        # Strict weight loading checks feature compatibility; restore replaces
        # the constructor's observed-node scale with the source checkpoint's.
        experiment.restore(checkpoint)
        experiment.model.requires_grad_(False)
        experiment.model.eval()
        with torch.inference_mode():
            metrics = experiment.metrics()
        result = {
            'experiment': 'cross_dataset_zero_shot',
            'source_district': 3,
            'target_district': district,
            'protocol': 'in_domain_reference' if district == 3 else 'zero_shot',
            'checkpoint': str(checkpoint_path),
            'target_training': False,
            'normalization': 'source_checkpoint',
            'flow_scale': experiment.flow_scale,
            'top_k': top_k,
            'flow_shape': list(flow.shape),
            'node_split_sizes': {key: len(value) for key, value in split.items()},
            'max_time_steps': args.max_time_steps,
            'experiment_config': asdict(config),
            **metrics,
        }
        directory = output_root / f'pems{district:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        if args.save_completion:
            with torch.inference_mode():
                completed, valid = experiment.completion()
            np.save(directory / 'completed_flow.npy', completed)
            np.save(directory / 'completion_valid_mask.npy', valid)
            del completed, valid
        (directory / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        results.append(result)
        for name in ('validation', 'test'):
            rows.append({'source': 'PEMS03', 'target': f'PEMS{district:02d}', 'protocol': result['protocol'], 'split': name, **metrics[name]})
            print(f'PEMS03 -> PEMS{district:02d} | {name} | {format_metrics(metrics[name])}', flush=True)
        del experiment, flow, subgraphs
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (output_root / 'summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    with (output_root / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Results: {output_root}')


if __name__ == '__main__':
    main()
