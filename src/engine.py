"""Training engine for LINK and its controlled ablation variants."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs import ExperimentConfig

from .metrics import MetricAccumulator, compute_loss, format_metrics
from .models import SPN, make_time_features
from .models.spn import (
    interpolate_temporal_window,
    lowpass_window,
    make_temporal_windows,
    selected_trend_window,
    trend_current,
)
from .subgraphs import (
    build_disconnected_supergraph,
    build_full_propagation_graph,
    build_full_propagation_topology,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load repository-owned torch data across old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=map_location)


def graph_to_device(
    graph: dict[str, torch.Tensor | int], device: torch.device
) -> dict[str, torch.Tensor | int]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in graph.items()
    }


def partition_centers(
    center_ids: np.ndarray, maximum_batch_size: int, seed: int
) -> list[list[int]]:
    """Create reproducible, balanced center batches no larger than the limit."""
    if maximum_batch_size <= 0:
        raise ValueError("maximum_batch_size must be positive")
    centers = [int(center) for center in center_ids]
    if not centers:
        raise ValueError("training split must contain at least one center")
    random.Random(seed).shuffle(centers)
    batch_count = (len(centers) + maximum_batch_size - 1) // maximum_batch_size
    base_size, larger_batches = divmod(len(centers), batch_count)
    batches: list[list[int]] = []
    offset = 0
    for index in range(batch_count):
        size = base_size + int(index < larger_batches)
        batches.append(centers[offset : offset + size])
        offset += size
    return batches


class LINKExperiment:
    """Train and evaluate SPN within the LINK traffic-kriging framework."""

    def __init__(
        self,
        flow: np.ndarray,
        subgraphs: list[dict[str, object]],
        splits: dict[str, np.ndarray],
        edge_metadata_dim: int,
        device: torch.device,
        config: ExperimentConfig,
    ) -> None:
        if flow.ndim == 3:
            if flow.shape[2] != 1:
                raise ValueError(f"flow must have one feature, got {flow.shape}")
            flow = flow[:, :, 0]
        if flow.ndim != 2:
            raise ValueError(f"flow must have shape [time, node], got {flow.shape}")
        if len(subgraphs) != flow.shape[1]:
            raise ValueError("subgraph count and flow node count differ")

        set_random_seed(config.training.seed)
        self.device = device
        self.experiment_config = config
        self.model_config = config.model
        self.training_config = config.training
        self.loss_config = config.loss
        self.evaluation_config = config.evaluation
        self.flow = torch.from_numpy(np.asarray(flow, dtype=np.float32)).to(device)
        self.splits = {key: np.asarray(value, dtype=np.int64) for key, value in splits.items()}
        center_batches = partition_centers(
            self.splits["train"],
            self.training_config.center_batch_size,
            self.training_config.seed,
        )
        if config.model.metadata_only:
            def prepare_metadata_only_graph(
                centers: list[int] | np.ndarray,
            ) -> dict[str, torch.Tensor | int]:
                center_ids = [int(center) for center in centers]
                type_feature = torch.stack(
                    [
                        subgraphs[center]["type_feature"][
                            int(subgraphs[center].get("center_index", 0))
                        ].float()
                        for center in center_ids
                    ]
                )
                direction_feature = torch.stack(
                    [
                        subgraphs[center]["direction_feature"][
                            int(subgraphs[center].get("center_index", 0))
                        ].float()
                        for center in center_ids
                    ]
                )
                center_tensor = torch.tensor(center_ids, dtype=torch.long)
                return graph_to_device(
                    {
                        "center_global_ids": center_tensor,
                        "center_indices": torch.arange(len(center_ids)),
                        "node_ids_global": center_tensor.clone(),
                        "boundary_mask": torch.zeros(
                            len(center_ids), dtype=torch.bool
                        ),
                        "type_feature": type_feature,
                        "direction_feature": direction_feature,
                        "num_nodes": len(center_ids),
                    },
                    device,
                )

            self.training_graphs = [
                prepare_metadata_only_graph(centers) for centers in center_batches
            ]
            self.graphs = {
                key: prepare_metadata_only_graph(value)
                for key, value in self.splits.items()
                if key != "train"
            }
        elif config.model.use_adaptive_subgraphs:
            self.training_graphs = [
                graph_to_device(
                    build_disconnected_supergraph(subgraphs, centers), device
                )
                for centers in center_batches
            ]
            self.graphs = {
                key: graph_to_device(
                    build_disconnected_supergraph(subgraphs, value), device
                )
                for key, value in self.splits.items()
                if key != "train"
            }
        else:
            observed_nodes = self.splits["train"]
            topology = build_full_propagation_topology(subgraphs)
            topology_on_device = graph_to_device(topology, device)

            def prepare_full_graph(
                centers: list[int] | np.ndarray,
            ) -> dict[str, torch.Tensor | int]:
                graph = build_full_propagation_graph(
                    subgraphs,
                    centers,
                    observed_nodes,
                    topology=topology,
                )
                target_fields = {
                    key: value for key, value in graph.items() if key not in topology
                }
                return {
                    **topology_on_device,
                    **graph_to_device(target_fields, device),
                }

            self.training_graphs = [
                prepare_full_graph(centers) for centers in center_batches
            ]
            self.graphs = {
                key: prepare_full_graph(value)
                for key, value in self.splits.items()
                if key != "train"
            }
        train_indices = torch.as_tensor(self.splits["train"], device=device)
        train_flow = self.flow[:, train_indices]
        positive = train_flow[torch.isfinite(train_flow) & (train_flow > 0)]
        if not positive.numel():
            raise ValueError("training nodes contain no positive finite observations")
        self.flow_scale = float(positive.mean().item())
        node_metadata_dim = int(subgraphs[0]["type_feature"].shape[1]) + int(
            subgraphs[0]["direction_feature"].shape[1]
        )
        self.model = SPN(
            edge_metadata_dim=edge_metadata_dim,
            window_size=config.model.window_size,
            trend_frequency_bins=config.model.trend_frequency_bins,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            time_feature_dim=config.model.time_feature_dim,
            branch_aggregation=config.model.branch_aggregation,
            max_propagation_hop=config.model.max_propagation_hop,
            node_metadata_dim=node_metadata_dim,
            use_fft_encoding=config.model.use_fft_encoding,
            use_fft_decomposition=config.model.use_fft_decomposition,
            use_metadata=config.model.use_metadata,
            metadata_only=config.model.metadata_only,
            time_window_impute=config.model.time_window_impute,
            use_subgraph_frequency_topk=config.model.use_subgraph_frequency_topk,
            use_graph_network=not config.model.use_adaptive_subgraphs,
            gnn_layers=config.model.gnn_layers,
        ).to(device)
        if not config.model.use_residual_refinement:
            # Keep the module in the state dict for simple architecture and
            # checkpoint handling, but make the ablated parameters inactive.
            for parameter in self.model.residual_mlp.parameters():
                parameter.requires_grad = False

    @property
    def sequence_length(self) -> int:
        return self.flow.shape[0]

    def _day_groups(self) -> dict[int, list[int]]:
        day_points = self.model_config.day_points
        groups = {weekday: [] for weekday in range(7)}
        for day in range(1, self.sequence_length // day_points):
            weekday = (self.model_config.start_weekday + day) % 7
            groups[weekday].append(day)
        return groups

    def sample_training_blocks(self, epoch: int) -> list[tuple[int, int]]:
        groups = self._day_groups()
        available = [weekday for weekday, days in groups.items() if days]
        count = self.training_config.train_days_per_epoch
        if count > len(available):
            raise ValueError(
                f"train_days_per_epoch={count}, but only {len(available)} weekdays are available"
            )
        rng = random.Random(self.training_config.seed + epoch)
        blocks = []
        for weekday in rng.sample(available, count):
            day = rng.choice(groups[weekday])
            start = day * self.model_config.day_points
            blocks.append((start, start + self.model_config.day_points))
        return blocks

    def validation_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        count = self.training_config.validation_days_per_weekday
        for days in self._day_groups().values():
            if not days:
                continue
            if count == 1:
                indices = np.asarray([len(days) // 2])
            else:
                indices = np.linspace(0, len(days) - 1, min(count, len(days))).round().astype(int)
            for index in np.unique(indices):
                day = days[int(index)]
                start = day * self.model_config.day_points
                blocks.append((start, start + self.model_config.day_points))
        if not blocks:
            raise ValueError("the sequence is too short to create validation day blocks")
        return blocks

    def _context(self, graph: dict[str, torch.Tensor | int]) -> int:
        if self.model_config.metadata_only or self.model_config.time_window_impute:
            return 0
        if bool(graph.get("full_graph", 0)):
            # The full-graph GNN FFT-encodes each node window once and then
            # applies ordinary graph layers. It therefore needs one causal
            # window of history, not one new window per propagation hop.
            return self.model_config.window_size - 1
        edge_hop = graph["edge_hop"]
        assert isinstance(edge_hop, torch.Tensor)
        hop = int(edge_hop.max().item()) if edge_hop.numel() else 1
        if self.model_config.max_propagation_hop is not None:
            hop = min(hop, self.model_config.max_propagation_hop)
        return max(1, hop) * (self.model_config.window_size - 1)

    def _run_model(
        self,
        graph: dict[str, torch.Tensor | int],
        start: int,
        end: int,
        use_residual: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.model_config.time_window_impute:
            length = end - start
            if not 0 < length <= self.model_config.window_size:
                raise ValueError(
                    "window imputation model calls must contain at most "
                    "window_size points"
                )
            node_ids = graph["node_ids_global"]
            boundary = graph["boundary_mask"]
            assert isinstance(node_ids, torch.Tensor)
            assert isinstance(boundary, torch.Tensor)
            raw = self.flow[start:end][:, node_ids]
            observed = torch.isfinite(raw) & (raw > 0) & boundary.unsqueeze(0)
            model_input = (
                torch.where(observed, raw, torch.zeros_like(raw)) / self.flow_scale
            )
            padding = self.model_config.window_size - length
            if padding:
                model_input = torch.cat(
                    (
                        model_input,
                        model_input.new_zeros(padding, node_ids.numel()),
                    ),
                    dim=0,
                )
                observed = torch.cat(
                    (
                        observed,
                        torch.zeros(
                            padding,
                            node_ids.numel(),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    ),
                    dim=0,
                )
            if self.model_config.use_time_features:
                time_features = make_time_features(
                    start,
                    start + self.model_config.window_size,
                    self.device,
                    day_points=self.model_config.day_points,
                    start_weekday=self.model_config.start_weekday,
                    dtype=model_input.dtype,
                )
            else:
                time_features = torch.zeros(
                    self.model_config.window_size,
                    self.model_config.time_feature_dim,
                    device=self.device,
                    dtype=model_input.dtype,
                )
            prediction, information = self.model(
                model_input,
                observed,
                graph,
                time_features,
                use_residual=use_residual,
            )
            if not self.model_config.use_residual_refinement:
                prediction = torch.relu(information["trend_prediction"])
            return prediction[:length], {
                key: value[:length] for key, value in information.items()
            }

        context_start = max(0, start - self._context(graph))
        node_ids = graph["node_ids_global"]
        boundary = graph["boundary_mask"]
        assert isinstance(node_ids, torch.Tensor)
        assert isinstance(boundary, torch.Tensor)
        if self.model_config.metadata_only:
            # Do not even gather boundary flow in the strict metadata baseline.
            # These tensors provide only the common model-call shape.
            model_input = self.flow.new_zeros(end - context_start, node_ids.numel())
            observed = torch.zeros_like(model_input, dtype=torch.bool)
        else:
            raw = self.flow[context_start:end][:, node_ids]
            observed = torch.isfinite(raw) & (raw > 0) & boundary.unsqueeze(0)
            model_input = (
                torch.where(observed, raw, torch.zeros_like(raw)) / self.flow_scale
            )
        if self.model_config.use_time_features:
            time_features = make_time_features(
                context_start,
                end,
                self.device,
                day_points=self.model_config.day_points,
                start_weekday=self.model_config.start_weekday,
                dtype=model_input.dtype,
            )
        else:
            # Keep the input dimension and parameter count identical for a clean
            # TOD/DOW ablation; only the information carried by these columns is
            # removed.
            time_features = torch.zeros(
                end - context_start,
                self.model_config.time_feature_dim,
                device=self.device,
                dtype=model_input.dtype,
            )
        prediction, information = self.model(
            model_input, observed, graph, time_features, use_residual=use_residual
        )
        if not self.model_config.use_residual_refinement:
            prediction = torch.relu(information["trend_prediction"])
        offset = start - context_start
        selection = slice(offset, offset + end - start)
        return prediction[selection], {
            key: value[selection] for key, value in information.items()
        }

    def _targets(
        self, graph: dict[str, torch.Tensor | int], start: int, end: int
    ) -> torch.Tensor:
        centers = graph["center_global_ids"]
        assert isinstance(centers, torch.Tensor)
        return self.flow[start:end][:, centers]

    def _target_components(
        self,
        graph: dict[str, torch.Tensor | int],
        start: int,
        end: int,
        frequency_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model_config.time_window_impute:
            length = end - start
            centers = graph["center_global_ids"]
            assert isinstance(centers, torch.Tensor)
            raw = self.flow[start:end][:, centers] / self.flow_scale
            valid = torch.isfinite(raw) & (raw > 0)
            padding = self.model_config.window_size - length
            if padding < 0:
                raise ValueError("target block exceeds window_size")
            if padding:
                raw = torch.cat(
                    (raw, raw.new_zeros(padding, centers.numel())), dim=0
                )
                valid = torch.cat(
                    (
                        valid,
                        torch.zeros(
                            padding,
                            centers.numel(),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    ),
                    dim=0,
                )
            if frequency_indices is None:
                trend = lowpass_window(
                    raw,
                    valid,
                    self.model_config.window_size,
                    self.model_config.trend_frequency_bins,
                )
            else:
                trend = selected_trend_window(
                    raw,
                    valid,
                    self.model_config.window_size,
                    frequency_indices[0],
                )
            safe = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            return (
                safe[:length],
                trend[:length],
                (safe - trend)[:length],
            )

        context_start = max(0, start - (self.model_config.window_size - 1))
        centers = graph["center_global_ids"]
        assert isinstance(centers, torch.Tensor)
        raw = self.flow[context_start:end][:, centers] / self.flow_scale
        valid = torch.isfinite(raw) & (raw > 0)
        safe = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        offset = start - context_start
        selection = slice(offset, offset + end - start)
        if frequency_indices is None:
            trend = trend_current(
                raw,
                valid,
                self.model_config.window_size,
                self.model_config.trend_frequency_bins,
                self.model_config.use_fft_decomposition,
            )[selection]
        else:
            windows = make_temporal_windows(raw, self.model_config.window_size)
            valid_windows = (
                make_temporal_windows(valid.float(), self.model_config.window_size) > 0
            )
            windows = interpolate_temporal_window(windows, valid_windows)
            spectrum = torch.fft.rfft(
                windows,
                n=self.model_config.window_size,
                dim=-1,
                norm="ortho",
            )[selection]
            coefficients = torch.gather(
                spectrum, -1, frequency_indices.long()
            )
            frequency_count = self.model_config.window_size // 2 + 1
            selected_spectrum = coefficients.new_zeros(
                *coefficients.shape[:-1], frequency_count
            )
            selected_spectrum.scatter_(
                -1, frequency_indices.long(), coefficients
            )
            trend = torch.fft.irfft(
                selected_spectrum,
                n=self.model_config.window_size,
                dim=-1,
                norm="ortho",
            )[..., -1]
        selected_safe = safe[selection]
        return selected_safe, trend, selected_safe - trend

    def _target_spectrum(
        self,
        graph: dict[str, torch.Tensor | int],
        start: int,
        end: int,
        frequency_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return retained focal coefficients and their window-level validity."""
        if not self.model_config.use_fft_encoding:
            raise ValueError("spectral targets require FFT encoding")
        centers = graph["center_global_ids"]
        assert isinstance(centers, torch.Tensor)
        width = self.model_config.window_size
        bins = self.model_config.trend_frequency_bins

        if self.model_config.time_window_impute:
            raw = self.flow[start:end][:, centers] / self.flow_scale
            valid = torch.isfinite(raw) & (raw > 0)
            padding = width - (end - start)
            if padding < 0:
                raise ValueError("spectral target block exceeds window_size")
            if padding:
                raw = torch.cat(
                    (raw, raw.new_zeros(padding, centers.numel())), dim=0
                )
                valid = torch.cat(
                    (
                        valid,
                        torch.zeros(
                            padding,
                            centers.numel(),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    ),
                    dim=0,
                )
            node_valid = valid.transpose(0, 1)
            node_windows = interpolate_temporal_window(
                raw.transpose(0, 1), node_valid
            )
            spectrum = torch.fft.rfft(
                node_windows,
                n=width,
                dim=-1,
                norm="forward",
            )
            if frequency_indices is None:
                spectrum = spectrum[..., :bins]
            else:
                spectrum = torch.gather(
                    spectrum, -1, frequency_indices[0].long()
                )
            return spectrum.unsqueeze(0), node_valid.any(dim=-1).unsqueeze(0)

        context_start = max(0, start - (width - 1))
        raw = self.flow[context_start:end][:, centers] / self.flow_scale
        valid = torch.isfinite(raw) & (raw > 0)
        windows = make_temporal_windows(raw, width)
        valid_windows = make_temporal_windows(valid.float(), width) > 0
        windows = interpolate_temporal_window(windows, valid_windows)
        spectrum = torch.fft.rfft(
            windows,
            n=width,
            dim=-1,
            norm="ortho",
        )
        offset = start - context_start
        selection = slice(offset, offset + end - start)
        spectrum = spectrum[selection]
        if frequency_indices is None:
            spectrum = spectrum[..., :bins]
        else:
            spectrum = torch.gather(
                spectrum, -1, frequency_indices.long()
            )
        return spectrum, valid_windows[selection].any(dim=-1)

    def _frequency_domain_trend_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        frequency_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-amplitude and amplitude-weighted circular phase losses."""
        if not valid.any():
            raise RuntimeError("frequency-domain trend loss has no valid targets")
        predicted_amplitude = torch.abs(prediction)
        target_amplitude = torch.abs(target)
        amplitude_loss = compute_loss(
            self.loss_config.trend_loss,
            torch.log1p(predicted_amplitude[valid]),
            torch.log1p(target_amplitude[valid]),
        )

        # DC and Nyquist are real-valued and have no useful circular phase.
        # For all other selected bins, use 1-cos(delta) rather than a wrapped
        # angle MSE and downweight bins whose target energy is negligible.
        if frequency_indices is None:
            frequency_indices = torch.arange(
                prediction.shape[-1], device=prediction.device
            ).view(*([1] * (prediction.ndim - 1)), -1).expand_as(prediction.real)
        phase_mask = frequency_indices != 0
        if self.model_config.window_size % 2 == 0:
            phase_mask &= frequency_indices != self.model_config.window_size // 2
        if not phase_mask.any():
            return amplitude_loss, amplitude_loss.new_zeros(())
        predicted = prediction
        expected = target
        predicted_scale = torch.abs(predicted).clamp_min(1e-6)
        expected_scale = torch.abs(expected).clamp_min(1e-6)
        cosine = (
            predicted.real * expected.real + predicted.imag * expected.imag
        ) / (predicted_scale * expected_scale)
        circular_error = 1.0 - cosine.clamp(min=-1.0, max=1.0)
        phase_weights = (
            torch.abs(expected)
            * valid.unsqueeze(-1).to(expected.real.dtype)
            * phase_mask.to(expected.real.dtype)
        )
        phase_loss = (circular_error * phase_weights).sum() / phase_weights.sum().clamp_min(
            1e-8
        )
        return amplitude_loss, phase_loss

    @staticmethod
    def _valid_mask(target: torch.Tensor, prediction_valid: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(target) & (target > 0) & prediction_valid.bool()

    def _expand_window_blocks(
        self, blocks: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Split day blocks into non-overlapping whole-window training jobs."""
        if not self.model_config.time_window_impute:
            return blocks
        width = self.model_config.window_size
        expanded: list[tuple[int, int]] = []
        for start, end in blocks:
            if (end - start) % width:
                raise ValueError(
                    "time_window_impute requires each sampled block length to "
                    "be divisible by window_size"
                )
            expanded.extend(
                (window_start, window_start + width)
                for window_start in range(start, end, width)
            )
        return expanded

    def _evaluation_ranges(self) -> list[tuple[int, int]]:
        width = (
            self.model_config.window_size
            if self.model_config.time_window_impute
            else self.training_config.evaluation_chunk
        )
        return [
            (start, min(start + width, self.sequence_length))
            for start in range(0, self.sequence_length, width)
        ]

    def _train_epoch(
        self,
        graphs: list[dict[str, torch.Tensor | int]],
        blocks: list[tuple[int, int]],
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        residual_stage: bool,
        shuffle_seed: int,
    ) -> tuple[float, int]:
        self.model.train()
        if residual_stage:
            self.model.trend_mlp.eval()
        loss_sum, count_sum = 0.0, 0
        blocks = self._expand_window_blocks(blocks)
        jobs = [
            (start, end, graph_index)
            for start, end in blocks
            for graph_index in range(len(graphs))
        ]
        random.Random(shuffle_seed).shuffle(jobs)
        optimizer_steps = 0
        for start, end, graph_index in jobs:
            graph = graphs[graph_index]
            optimizer.zero_grad(set_to_none=True)
            prediction, information = self._run_model(graph, start, end, residual_stage)
            frequency_indices = information.get("spectral_frequency_indices")
            target_scaled, target_trend, target_residual = self._target_components(
                graph, start, end, frequency_indices
            )
            target = self._targets(graph, start, end)
            valid = self._valid_mask(target, information["prediction_valid"])
            if not valid.any():
                continue
            frequency_targets: torch.Tensor | None = None
            if residual_stage:
                residual_loss = compute_loss(
                    self.loss_config.residual_loss,
                    information["residual_prediction"][valid], target_residual[valid]
                )
                final_loss = compute_loss(
                    self.loss_config.final_loss,
                    prediction[valid],
                    target_scaled[valid],
                )
                loss = residual_loss + self.loss_config.final_loss_weight * final_loss
                parameters = self.model.residual_mlp.parameters()
            else:
                time_domain_loss = compute_loss(
                    self.loss_config.trend_loss,
                    information["trend_prediction"][valid], target_trend[valid]
                )
                loss_domain = self.loss_config.trend_loss_domain
                if loss_domain == "time":
                    loss = self.loss_config.trend_time_weight * time_domain_loss
                else:
                    if "spectral_prediction" not in information:
                        raise RuntimeError(
                            "frequency-domain loss requires spectral model outputs"
                        )
                    frequency_targets, frequency_target_valid = self._target_spectrum(
                        graph, start, end, frequency_indices
                    )
                    frequency_valid = (
                        information["spectral_prediction_valid"].bool()
                        & frequency_target_valid
                    )
                    amplitude_loss, phase_loss = self._frequency_domain_trend_loss(
                        information["spectral_prediction"],
                        frequency_targets,
                        frequency_valid,
                        frequency_indices,
                    )
                    frequency_loss = (
                        self.loss_config.trend_amplitude_weight * amplitude_loss
                        + self.loss_config.trend_phase_weight * phase_loss
                    )
                    if loss_domain == "frequency":
                        loss = frequency_loss
                    else:
                        loss = (
                            self.loss_config.trend_time_weight * time_domain_loss
                            + frequency_loss
                        )
                parameters = self.model.trend_mlp.parameters()
            if not torch.isfinite(loss):
                diagnostic_tensors = {
                    "prediction": prediction,
                    "trend_prediction": information["trend_prediction"],
                    "residual_prediction": information["residual_prediction"],
                    "target_scaled": target_scaled,
                    "target_trend": target_trend,
                    "target_residual": target_residual,
                }
                if not residual_stage and frequency_targets is not None:
                    diagnostic_tensors.update(
                        {
                            "spectral_prediction": information[
                                "spectral_prediction"
                            ],
                            "spectral_target": frequency_targets,
                        }
                    )
                invalid = [
                    name
                    for name, tensor in diagnostic_tensors.items()
                    if not torch.isfinite(tensor).all()
                ]
                source = ", ".join(invalid) if invalid else "loss computation"
                raise RuntimeError(
                    f"non-finite training loss for block [{start}, {end}); "
                    f"invalid tensor(s): {source}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                self.training_config.gradient_clip,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(gradient_norm):
                optimizer.zero_grad(set_to_none=True)
                stage = "residual" if residual_stage else "trend"
                raise RuntimeError(
                    f"non-finite {stage} gradient for block [{start}, {end}); "
                    "the optimizer step was skipped so model parameters remain intact"
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer_steps += 1
            count = int(valid.sum().item())
            loss_sum += loss.item() * count
            count_sum += count
        if count_sum == 0:
            raise RuntimeError("training epoch had no valid prediction targets")
        return loss_sum / count_sum, optimizer_steps

    @torch.no_grad()
    def evaluate_blocks(
        self,
        graph: dict[str, torch.Tensor | int],
        blocks: list[tuple[int, int]],
        use_residual: bool,
        final_prediction: bool = False,
    ) -> dict[str, float | int]:
        self.model.eval()
        metrics = MetricAccumulator(self.evaluation_config.metrics)
        for start, end in self._expand_window_blocks(blocks):
            prediction, information = self._run_model(graph, start, end, use_residual)
            target = self._targets(graph, start, end)
            valid = self._valid_mask(target, information["prediction_valid"])
            if use_residual or final_prediction:
                metrics.update(prediction * self.flow_scale, target, valid)
            else:
                _, target_trend, _ = self._target_components(
                    graph,
                    start,
                    end,
                    information.get("spectral_frequency_indices"),
                )
                metrics.update(
                    information["trend_prediction"] * self.flow_scale,
                    target_trend * self.flow_scale,
                    valid,
                )
        return metrics.compute()

    @torch.no_grad()
    def evaluate_full(
        self, graph: dict[str, torch.Tensor | int]
    ) -> dict[str, float | int]:
        self.model.eval()
        metrics = MetricAccumulator(self.evaluation_config.metrics)
        use_residual = self.model_config.use_residual_refinement
        for start, end in self._evaluation_ranges():
            prediction, information = self._run_model(
                graph, start, end, use_residual
            )
            target = self._targets(graph, start, end)
            valid = self._valid_mask(target, information["prediction_valid"])
            metrics.update(prediction * self.flow_scale, target, valid)
        return metrics.compute()

    def _build_optimizer(
        self, parameters: Any
    ) -> torch.optim.Optimizer:
        options = {
            "lr": self.training_config.learning_rate,
            "weight_decay": self.training_config.weight_decay,
            "eps": self.training_config.optimizer_eps,
        }
        if self.training_config.optimizer == "adam":
            return torch.optim.Adam(parameters, **options)
        if self.training_config.optimizer == "adamw":
            return torch.optim.AdamW(parameters, **options)
        raise ValueError(f"unsupported optimizer: {self.training_config.optimizer}")

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, total_optimizer_steps: int
    ) -> Any | None:
        if self.training_config.scheduler == "none":
            return None
        if self.training_config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_optimizer_steps
            )
        raise ValueError(f"unsupported scheduler: {self.training_config.scheduler}")

    def _initial_best_score(self) -> float:
        return float("inf") if self.evaluation_config.checkpoint_mode == "min" else -float("inf")

    def _is_better(self, score: float, best_score: float) -> bool:
        min_delta = self.training_config.early_stopping_min_delta
        if self.evaluation_config.checkpoint_mode == "min":
            return score < best_score - min_delta
        return score > best_score + min_delta

    def _checkpoint_score(self, metrics: dict[str, float | int]) -> float:
        return float(metrics[self.evaluation_config.checkpoint_metric])

    def train(self) -> dict[str, Any]:
        """Train with spatial center mini-batches and step-based LR scheduling."""
        set_random_seed(self.training_config.seed)
        validation_blocks = self.validation_blocks()
        valid_graph = self.graphs["valid"]
        windows_per_day = (
            self.model_config.day_points // self.model_config.window_size
            if self.model_config.time_window_impute
            else 1
        )
        optimizer_steps_per_epoch = (
            len(self.training_graphs)
            * self.training_config.train_days_per_epoch
            * windows_per_day
        )

        trend_optimizer = self._build_optimizer(self.model.trend_mlp.parameters())
        planned_trend_steps = (
            optimizer_steps_per_epoch * self.training_config.trend_epochs
        )
        trend_scheduler = self._build_scheduler(
            trend_optimizer, planned_trend_steps
        )
        best_trend_score = self._initial_best_score()
        best_trend_epoch, best_trend_state = 0, None
        best_trend_metrics: dict[str, float | int] | None = None
        trend_no_improvement = 0
        trend_epochs_ran = 0
        trend_optimizer_steps = 0
        trend_early_stopped = False
        for epoch in range(1, self.training_config.trend_epochs + 1):
            trend_epochs_ran = epoch
            started = time.perf_counter()
            blocks = self.sample_training_blocks(epoch)
            loss, epoch_steps = self._train_epoch(
                self.training_graphs,
                blocks,
                trend_optimizer,
                trend_scheduler,
                False,
                self.training_config.seed + epoch,
            )
            trend_optimizer_steps += epoch_steps
            should_evaluate = (
                epoch == 1
                or epoch % self.training_config.evaluate_every == 0
                or epoch == self.training_config.trend_epochs
            )
            if should_evaluate:
                validation = self.evaluate_blocks(
                    valid_graph,
                    validation_blocks,
                    False,
                    final_prediction=not self.model_config.use_residual_refinement,
                )
                score = self._checkpoint_score(validation)
                if self._is_better(score, best_trend_score):
                    best_trend_score = score
                    best_trend_epoch = epoch
                    best_trend_state = copy.deepcopy(self.model.trend_mlp.state_dict())
                    best_trend_metrics = copy.deepcopy(validation)
                    trend_no_improvement = 0
                else:
                    trend_no_improvement += 1
                print(
                    f"Trend {epoch:03d}/{self.training_config.trend_epochs} | "
                    f"loss={loss:.6f} | val {format_metrics(validation)} | "
                    f"best {self.evaluation_config.checkpoint_metric.upper()}="
                    f"{best_trend_score:.4f} | patience="
                    f"{trend_no_improvement}/"
                    f"{self.training_config.early_stopping_patience} | "
                    f"min_delta={self.training_config.early_stopping_min_delta:g} | "
                    f"steps={epoch_steps}/{optimizer_steps_per_epoch} "
                    f"total={trend_optimizer_steps} | "
                    f"lr={trend_optimizer.param_groups[0]['lr']:.6g} | "
                    f"{time.perf_counter() - started:.1f}s"
                )
            else:
                print(
                    f"Trend {epoch:03d}/{self.training_config.trend_epochs} | "
                    f"loss={loss:.6f} | steps={epoch_steps}/"
                    f"{optimizer_steps_per_epoch} total={trend_optimizer_steps} | "
                    f"lr={trend_optimizer.param_groups[0]['lr']:.6g} | "
                    f"{time.perf_counter() - started:.1f}s"
                )
            if (
                should_evaluate
                and trend_no_improvement
                >= self.training_config.early_stopping_patience
            ):
                trend_early_stopped = True
                print(
                    "Trend early stopping: validation "
                    f"{self.evaluation_config.checkpoint_metric.upper()} did not "
                    f"improve for {self.training_config.early_stopping_patience} "
                    f"consecutive evaluations; best epoch={best_trend_epoch}."
                )
                break
        if best_trend_state is None or best_trend_metrics is None:
            raise RuntimeError("no valid trend checkpoint was produced")
        self.model.trend_mlp.load_state_dict(best_trend_state)
        for parameter in self.model.trend_mlp.parameters():
            parameter.requires_grad = False

        residual_enabled = self.model_config.use_residual_refinement
        if residual_enabled:
            residual_optimizer = self._build_optimizer(
                self.model.residual_mlp.parameters()
            )
            planned_residual_steps = (
                optimizer_steps_per_epoch * self.training_config.residual_epochs
            )
            residual_scheduler = self._build_scheduler(
                residual_optimizer, planned_residual_steps
            )
            best_score = self._initial_best_score()
            best_epoch, best_state, best_quick_metrics = 0, None, None
            residual_epochs = range(1, self.training_config.residual_epochs + 1)
        else:
            residual_optimizer = None
            residual_scheduler = None
            planned_residual_steps = 0
            best_score = best_trend_score
            best_epoch = 0
            best_state = copy.deepcopy(self.model.state_dict())
            best_quick_metrics = copy.deepcopy(best_trend_metrics)
            residual_epochs = ()
            print(
                "Residual refinement disabled: using ReLU(trend) as the final "
                f"prediction; best trend epoch={best_trend_epoch}."
            )
        residual_no_improvement = 0
        residual_epochs_ran = 0
        residual_optimizer_steps = 0
        residual_early_stopped = False
        for epoch in residual_epochs:
            assert residual_optimizer is not None
            residual_epochs_ran = epoch
            started = time.perf_counter()
            blocks = self.sample_training_blocks(epoch + self.training_config.trend_epochs)
            loss, epoch_steps = self._train_epoch(
                self.training_graphs,
                blocks,
                residual_optimizer,
                residual_scheduler,
                True,
                self.training_config.seed
                + self.training_config.trend_epochs
                + epoch,
            )
            residual_optimizer_steps += epoch_steps
            should_evaluate = (
                epoch == 1
                or epoch % self.training_config.evaluate_every == 0
                or epoch == self.training_config.residual_epochs
            )
            if should_evaluate:
                validation = self.evaluate_blocks(valid_graph, validation_blocks, True)
                score = self._checkpoint_score(validation)
                if self._is_better(score, best_score):
                    best_score = score
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_quick_metrics = copy.deepcopy(validation)
                    residual_no_improvement = 0
                else:
                    residual_no_improvement += 1
                print(
                    f"Residual {epoch:03d}/{self.training_config.residual_epochs} | "
                    f"loss={loss:.6f} | val {format_metrics(validation)} | "
                    f"best {self.evaluation_config.checkpoint_metric.upper()}="
                    f"{best_score:.4f} | "
                    f"patience={residual_no_improvement}/"
                    f"{self.training_config.early_stopping_patience} | "
                    f"min_delta={self.training_config.early_stopping_min_delta:g} | "
                    f"steps={epoch_steps}/{optimizer_steps_per_epoch} "
                    f"total={residual_optimizer_steps} | "
                    f"lr={residual_optimizer.param_groups[0]['lr']:.6g} | "
                    f"{time.perf_counter() - started:.1f}s"
                )
            else:
                print(
                    f"Residual {epoch:03d}/{self.training_config.residual_epochs} | "
                    f"loss={loss:.6f} | steps={epoch_steps}/"
                    f"{optimizer_steps_per_epoch} total={residual_optimizer_steps} | "
                    f"lr={residual_optimizer.param_groups[0]['lr']:.6g} | "
                    f"{time.perf_counter() - started:.1f}s"
                )
            if (
                should_evaluate
                and residual_no_improvement
                >= self.training_config.early_stopping_patience
            ):
                residual_early_stopped = True
                print(
                    "Residual early stopping: validation "
                    f"{self.evaluation_config.checkpoint_metric.upper()} did not "
                    f"improve for {self.training_config.early_stopping_patience} "
                    f"consecutive evaluations; best epoch={best_epoch}."
                )
                break
        if best_state is None or best_quick_metrics is None:
            raise RuntimeError("no valid final checkpoint was produced")
        self.model.load_state_dict(best_state)
        checkpoint = {
            "model_state_dict": best_state,
            "experiment_config": asdict(self.experiment_config),
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "loss_config": asdict(self.loss_config),
            "evaluation_config": asdict(self.evaluation_config),
            "flow_scale": self.flow_scale,
            "scheduler_step_unit": (
                "optimizer_step" if self.training_config.scheduler != "none" else None
            ),
            "train_center_batch_count": len(self.training_graphs),
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "planned_trend_optimizer_steps": planned_trend_steps,
            "trend_optimizer_steps": trend_optimizer_steps,
            "best_trend_epoch": best_trend_epoch,
            "trend_epochs_ran": trend_epochs_ran,
            "trend_early_stopped": trend_early_stopped,
            "best_trend_validation_score": best_trend_score,
            "trend_validation_target": (
                "trend"
                if self.model_config.use_residual_refinement
                else "flow"
            ),
            "residual_refinement_enabled": residual_enabled,
            "best_residual_epoch": best_epoch,
            "planned_residual_optimizer_steps": planned_residual_steps,
            "residual_optimizer_steps": residual_optimizer_steps,
            "residual_epochs_ran": residual_epochs_ran,
            "residual_early_stopped": residual_early_stopped,
            "early_stopping_patience": self.training_config.early_stopping_patience,
            "early_stopping_min_delta": self.training_config.early_stopping_min_delta,
            "checkpoint_metric": self.evaluation_config.checkpoint_metric,
            "checkpoint_mode": self.evaluation_config.checkpoint_mode,
            "best_validation_score": best_score,
            "best_quick_validation_metrics": best_quick_metrics,
        }
        if self.evaluation_config.checkpoint_metric == "mae":
            checkpoint["best_trend_validation_mae"] = best_trend_score
        return checkpoint

    def restore(self, checkpoint: dict[str, Any]) -> None:
        self.model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_scale = float(checkpoint["flow_scale"])
        if not np.isfinite(checkpoint_scale) or checkpoint_scale <= 0:
            raise ValueError("checkpoint has an invalid flow scale")
        self.flow_scale = checkpoint_scale

    @torch.no_grad()
    def predict(
        self, graph: dict[str, torch.Tensor | int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict every center and return real-scale values plus validity mask."""
        self.model.eval()
        centers = graph["center_global_ids"]
        assert isinstance(centers, torch.Tensor)
        prediction = np.full((self.sequence_length, len(centers)), np.nan, dtype=np.float32)
        valid = np.zeros(prediction.shape, dtype=bool)
        for start, end in self._evaluation_ranges():
            values, information = self._run_model(
                graph,
                start,
                end,
                self.model_config.use_residual_refinement,
            )
            prediction[start:end] = (values * self.flow_scale).cpu().numpy()
            valid[start:end] = information["prediction_valid"].cpu().numpy()
        prediction[~valid] = np.nan
        return prediction, valid

    def metrics(self) -> dict[str, dict[str, float | int]]:
        return {
            "validation": self.evaluate_full(self.graphs["valid"]),
            "test": self.evaluate_full(self.graphs["test"]),
        }

    def completion(self) -> tuple[np.ndarray, np.ndarray]:
        """Complete held-out validation/test sensors while retaining train observations."""
        completed = np.full(self.flow.shape, np.nan, dtype=np.float32)
        completion_valid = np.zeros(self.flow.shape, dtype=bool)
        train = torch.as_tensor(self.splits["train"], device=self.device)
        train_values = self.flow[:, train].cpu().numpy()
        completed[:, self.splits["train"]] = train_values
        completion_valid[:, self.splits["train"]] = np.isfinite(train_values) & (train_values > 0)
        for split_name in ("valid", "test"):
            prediction, valid = self.predict(self.graphs[split_name])
            indices = self.splits[split_name]
            completed[:, indices] = prediction
            completion_valid[:, indices] = valid
        return completed, completion_valid
