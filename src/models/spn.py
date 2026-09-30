"""Spectral Propagation Network (SPN) used by LINK."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_FULL_GRAPH_EDGE_CHUNK_SIZE = 128


def make_time_features(
    start: int,
    end: int,
    device: torch.device | str,
    *,
    day_points: int = 288,
    start_weekday: int = 6,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create absolute time-of-day and day-of-week circular features."""
    time = torch.arange(start, end, device=device, dtype=dtype)
    tod = torch.remainder(time, day_points) / day_points
    dow = torch.remainder(torch.floor(time / day_points) + start_weekday, 7.0) / 7.0
    return torch.stack(
        (
            torch.sin(2 * torch.pi * tod),
            torch.cos(2 * torch.pi * tod),
            torch.sin(2 * torch.pi * dow),
            torch.cos(2 * torch.pi * dow),
        ),
        dim=-1,
    )


def make_temporal_windows(values: torch.Tensor, window_size: int) -> torch.Tensor:
    """Convert ``[time,node]`` to causal ``[time,node,window]`` windows."""
    padded = F.pad(values.transpose(0, 1), (window_size - 1, 0))
    return padded.unfold(1, window_size, 1).permute(1, 0, 2)


def interpolate_temporal_window(window: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Interpolate missing values and use nearest valid values at both boundaries."""
    valid = valid.bool() & torch.isfinite(window)
    values = torch.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
    width = values.shape[-1]
    shape = [1] * (values.ndim - 1) + [width]
    index = torch.arange(width, device=values.device).view(*shape).expand_as(values)
    left = torch.cummax(torch.where(valid, index, torch.full_like(index, -1)), dim=-1).values
    right_source = torch.where(valid, index, torch.full_like(index, width))
    right = torch.flip(
        torch.cummin(torch.flip(right_source, dims=[-1]), dim=-1).values,
        dims=[-1],
    )
    has_left, has_right = left >= 0, right < width
    left_value = torch.gather(values, -1, left.clamp(0, width - 1))
    right_value = torch.gather(values, -1, right.clamp(0, width - 1))
    alpha = (index - left).to(values.dtype) / (right - left).clamp_min(1).to(values.dtype)
    interpolated = left_value + alpha * (right_value - left_value)
    filled = torch.where(has_left & has_right, interpolated, torch.zeros_like(values))
    filled = torch.where(has_left & ~has_right, left_value, filled)
    filled = torch.where(~has_left & has_right, right_value, filled)
    return torch.where(valid, values, filled)


def lowpass_causal_windows(
    values: torch.Tensor, valid: torch.Tensor, window_size: int, frequency_bins: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """FFT-decompose causal windows and reconstruct their retained-bin trend."""
    windows = make_temporal_windows(values, window_size)
    valid_windows = make_temporal_windows(valid.float(), window_size) > 0
    windows = interpolate_temporal_window(windows, valid_windows)
    spectrum = torch.fft.rfft(windows, n=window_size, dim=-1, norm="ortho")
    spectrum[..., frequency_bins:] = 0
    trend_windows = torch.fft.irfft(
        spectrum, n=window_size, dim=-1, norm="ortho"
    )
    return trend_windows, valid_windows


def lowpass_current(
    values: torch.Tensor, valid: torch.Tensor, window_size: int, frequency_bins: int
) -> torch.Tensor:
    trend_windows, _ = lowpass_causal_windows(
        values, valid, window_size, frequency_bins
    )
    return trend_windows[..., -1]


def decode_spectral_current(
    coefficients: torch.Tensor,
    window_size: int,
    frequency_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode retained bins only when a time-domain value is required."""
    frequency_bins = window_size // 2 + 1
    if coefficients.shape[-1] > frequency_bins:
        raise ValueError("spectral state has more bins than the target rFFT grid")
    if frequency_indices is None:
        padding = torch.zeros(
            *coefficients.shape[:-1],
            frequency_bins - coefficients.shape[-1],
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        spectrum = torch.cat((coefficients, padding), dim=-1)
    else:
        if frequency_indices.shape != coefficients.shape:
            raise ValueError("frequency indices must match retained coefficients")
        if bool(((frequency_indices < 0) | (frequency_indices >= frequency_bins)).any()):
            raise ValueError("frequency index is outside the target rFFT grid")
        spectrum = coefficients.new_zeros(*coefficients.shape[:-1], frequency_bins)
        spectrum.scatter_(-1, frequency_indices.long(), coefficients)
    return torch.fft.irfft(
        spectrum, n=window_size, dim=-1, norm="ortho"
    )[..., -1]


def decode_spectral_window(
    coefficients: torch.Tensor,
    window_size: int,
    frequency_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero-pad retained coefficients and reconstruct the complete window.

    Window-level propagation uses ``norm="forward"`` so that the DC component
    is the window mean rather than growing with ``sqrt(window_size)``.  This is
    important because the same spectral messenger is recursively applied at
    every graph hop.
    """
    frequency_bins = window_size // 2 + 1
    if coefficients.shape[-1] > frequency_bins:
        raise ValueError("spectral state has more bins than the target rFFT grid")
    if frequency_indices is None:
        padding = torch.zeros(
            *coefficients.shape[:-1],
            frequency_bins - coefficients.shape[-1],
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        spectrum = torch.cat((coefficients, padding), dim=-1)
    else:
        if frequency_indices.shape != coefficients.shape:
            raise ValueError("frequency indices must match retained coefficients")
        if bool(((frequency_indices < 0) | (frequency_indices >= frequency_bins)).any()):
            raise ValueError("frequency index is outside the target rFFT grid")
        spectrum = coefficients.new_zeros(*coefficients.shape[:-1], frequency_bins)
        spectrum.scatter_(-1, frequency_indices.long(), coefficients)
    return torch.fft.irfft(spectrum, n=window_size, dim=-1, norm="forward")


def selected_trend_current(
    values: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
    frequency_indices: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the current value from explicitly selected causal FFT bins."""
    windows = make_temporal_windows(values, window_size)
    valid_windows = make_temporal_windows(valid.float(), window_size) > 0
    windows = interpolate_temporal_window(windows, valid_windows)
    spectrum = torch.fft.rfft(windows, n=window_size, dim=-1, norm="ortho")
    coefficients = torch.gather(spectrum, -1, frequency_indices.long())
    return decode_spectral_current(coefficients, window_size, frequency_indices)


def selected_trend_window(
    values: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
    frequency_indices: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a complete window from explicitly selected FFT bins."""
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values and valid must have matching [window,node] shapes")
    if values.shape[0] != window_size:
        raise ValueError(
            f"expected a {window_size}-point window, got {values.shape[0]}"
        )
    node_valid = valid.transpose(0, 1).bool()
    node_windows = interpolate_temporal_window(values.transpose(0, 1), node_valid)
    spectrum = torch.fft.rfft(
        node_windows, n=window_size, dim=-1, norm="forward"
    )
    coefficients = torch.gather(spectrum, -1, frequency_indices.long())
    return decode_spectral_window(
        coefficients, window_size, frequency_indices
    ).transpose(0, 1)


def lowpass_window(
    values: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
    frequency_bins: int,
) -> torch.Tensor:
    """Return the complete low-pass reconstruction of ``[window,node]``."""
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values and valid must have matching [window,node] shapes")
    if values.shape[0] != window_size:
        raise ValueError(
            f"expected a {window_size}-point window, got {values.shape[0]}"
        )
    node_windows = values.transpose(0, 1)
    node_valid = valid.transpose(0, 1).bool()
    node_windows = interpolate_temporal_window(node_windows, node_valid)
    spectrum = torch.fft.rfft(
        node_windows, n=window_size, dim=-1, norm="forward"
    )
    spectrum[..., frequency_bins:] = 0
    return torch.fft.irfft(
        spectrum, n=window_size, dim=-1, norm="forward"
    ).transpose(0, 1)


def temporal_trend_current(
    values: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
    frequency_bins: int = 4,
) -> torch.Tensor:
    """Return the FFT low-pass target shared by both propagation domains."""
    return lowpass_current(values, valid, window_size, frequency_bins)


def trend_current(
    values: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
    frequency_bins: int,
    use_fft_decomposition: bool,
) -> torch.Tensor:
    """Build an FFT trend target or the raw-flow target for strict w/o FFT."""
    if use_fft_decomposition:
        return temporal_trend_current(values, valid, window_size, frequency_bins)
    return torch.where(
        valid.bool(),
        torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(values),
    )


def _fuse_branches(
    upstream: torch.Tensor,
    upstream_valid: torch.Tensor,
    downstream: torch.Tensor,
    downstream_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros_like(upstream)
    both = upstream_valid & downstream_valid
    output[both] = 0.5 * (upstream[both] + downstream[both])
    output[upstream_valid & ~downstream_valid] = upstream[upstream_valid & ~downstream_valid]
    output[downstream_valid & ~upstream_valid] = downstream[downstream_valid & ~upstream_valid]
    return output, upstream_valid | downstream_valid


class SpectralTrendMLP(nn.Module):
    """Trend messenger with spectral and full-window temporal variants."""

    def __init__(
        self,
        edge_metadata_dim: int,
        window_size: int,
        trend_frequency_bins: int,
        hidden_dim: int,
        dropout: float,
        time_feature_dim: int,
        use_fft_encoding: bool = True,
        stabilize_recursive_spectrum: bool = False,
        use_frequency_indices: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.trend_bins = trend_frequency_bins
        self.use_fft_encoding = use_fft_encoding
        self.stabilize_recursive_spectrum = stabilize_recursive_spectrum
        self.use_frequency_indices = use_frequency_indices
        if use_frequency_indices and not use_fft_encoding:
            raise ValueError("frequency-index features require FFT encoding")
        traffic_feature_dim = self.trend_bins * 3 if use_fft_encoding else window_size
        if use_frequency_indices:
            traffic_feature_dim += self.trend_bins
        output_dim = self.trend_bins * 3 if use_fft_encoding else window_size
        input_dim = (
            window_size
            + traffic_feature_dim
            + 4
            + edge_metadata_dim
            + time_feature_dim
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        source_state: torch.Tensor,
        source_valid: torch.Tensor,
        edge_weight: torch.Tensor,
        direction: torch.Tensor,
        source_out_degree: torch.Tensor,
        destination_in_degree: torch.Tensor,
        edge_metadata: torch.Tensor,
        time_feature: torch.Tensor,
        frequency_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_fft_encoding:
            if not source_state.is_complex():
                raise TypeError("spectral trend propagation requires complex coefficients")
            if source_state.shape[-1] != self.trend_bins:
                raise ValueError(
                    "spectral state width must equal trend_frequency_bins"
                )
            # sin(angle(z)) and cos(angle(z)) are respectively imag/|z| and
            # real/|z|.  Express them directly with a clamped denominator:
            # autograd through torch.angle is singular for a zero (or nearly
            # cancelled) complex coefficient, which can poison Adam after a
            # single recursive propagation step.
            amplitude = torch.abs(source_state)
            safe_amplitude = amplitude.clamp_min(1e-4)
            traffic_features = torch.cat(
                (
                    amplitude,
                    source_state.imag / safe_amplitude,
                    source_state.real / safe_amplitude,
                ),
                dim=-1,
            )
            if self.use_frequency_indices:
                if frequency_indices is None:
                    raise ValueError(
                        "dynamic spectral propagation requires frequency indices"
                    )
                if frequency_indices.shape != source_state.shape:
                    raise ValueError(
                        "frequency indices must match the spectral state shape"
                    )
                maximum_index = max(1, self.window_size // 2)
                traffic_features = torch.cat(
                    (
                        traffic_features,
                        frequency_indices.to(source_state.real.dtype) / maximum_index,
                    ),
                    dim=-1,
                )
        else:
            # The controlled counterpart remains fully in the time domain
            # between hops: one complete low-pass or raw-flow window enters and
            # one complete predicted window leaves the shared MLP. Boundary
            # interpolation has already happened while constructing the state,
            # so do not alter that state a second time.
            traffic_features = source_state
        features = torch.cat(
            (
                source_valid.float(),
                traffic_features,
                edge_weight.unsqueeze(-1),
                direction.unsqueeze(-1),
                source_out_degree.unsqueeze(-1),
                destination_in_degree.unsqueeze(-1),
                edge_metadata,
                time_feature,
            ),
            dim=-1,
        )
        output = self.network(features)
        if not self.use_fft_encoding:
            return output
        if self.stabilize_recursive_spectrum:
            # A complete window is represented by one recurrent spectral
            # state.  Bounding its logits prevents a long subgraph path from
            # exponentially increasing the coefficient scale.  NaNs are not
            # replaced here, so genuine invalid inputs still fail loudly.
            output = output.clamp(min=-20.0, max=20.0)
        bins = self.trend_bins
        amplitude = F.softplus(output[..., :bins])
        phase_sin, phase_cos = output[..., bins : 2 * bins], output[..., 2 * bins :]
        norm = torch.sqrt(phase_sin.square() + phase_cos.square() + 1e-8)
        phase_sin, phase_cos = phase_sin / norm, phase_cos / norm
        phase_sin = torch.cat((torch.zeros_like(phase_sin[..., :1]), phase_sin[..., 1:]), -1)
        phase_cos = torch.cat((torch.ones_like(phase_cos[..., :1]), phase_cos[..., 1:]), -1)
        coefficients = torch.complex(amplitude * phase_cos, amplitude * phase_sin)
        return coefficients


class ResidualMLP(nn.Module):
    """Propagate only the time-domain residual plus structural context."""

    def __init__(
        self,
        edge_metadata_dim: int,
        window_size: int,
        hidden_dim: int,
        dropout: float,
        time_feature_dim: int,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        # residual window + validity window.  The complete trend window is
        # deliberately excluded: [residual, trend] could be summed inside the
        # MLP to recover the raw traffic window and bypass the spectral branch.
        input_dim = window_size * 2 + 4 + edge_metadata_dim + time_feature_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        residual_window: torch.Tensor,
        source_valid: torch.Tensor,
        edge_weight: torch.Tensor,
        direction: torch.Tensor,
        source_out_degree: torch.Tensor,
        destination_in_degree: torch.Tensor,
        edge_metadata: torch.Tensor,
        time_feature: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                residual_window,
                source_valid.float(),
                edge_weight.unsqueeze(-1),
                direction.unsqueeze(-1),
                source_out_degree.unsqueeze(-1),
                destination_in_degree.unsqueeze(-1),
                edge_metadata,
                time_feature,
            ),
            dim=-1,
        )
        output = self.network(features)
        return output.squeeze(-1) if self.output_dim == 1 else output


class WeightedDirectedGraphLayer(nn.Module):
    """One normalized, edge-aware directed graph-convolution layer."""

    def __init__(
        self, input_dim: int, output_dim: int, edge_context_dim: int
    ) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.edge_linear = nn.Linear(edge_context_dim, output_dim, bias=False)

    def forward(
        self,
        node_features: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_context: torch.Tensor,
    ) -> torch.Tensor:
        """Apply weighted GCN aggregation to ``[time, node, feature]``."""
        output = self.self_linear(node_features)
        if not source.numel():
            return output

        node_count = node_features.shape[1]
        magnitude = edge_weight.abs()
        out_degree = edge_weight.new_zeros(node_count).index_add_(
            0, source, magnitude
        )
        in_degree = edge_weight.new_zeros(node_count).index_add_(
            0, destination, magnitude
        )
        normalization = edge_weight / torch.sqrt(
            out_degree[source].clamp_min(1e-8)
            * in_degree[destination].clamp_min(1e-8)
        )
        # Sparse multiplication avoids materializing a
        # [time, edge, hidden] tensor on the large PEMS07 graph. Linear graph
        # convolution commutes with the weighted sum, so aggregate node
        # features first and project them afterwards.
        adjacency = torch.sparse_coo_tensor(
            torch.stack((destination, source)),
            normalization,
            (node_count, node_count),
            device=node_features.device,
        ).coalesce()
        time_steps, _, input_dim = node_features.shape
        flattened = node_features.permute(1, 0, 2).reshape(
            node_count, time_steps * input_dim
        )
        neighbor_features = torch.sparse.mm(adjacency, flattened)
        neighbor_features = neighbor_features.reshape(
            node_count, time_steps, input_dim
        ).permute(1, 0, 2)
        aggregated = self.neighbor_linear(neighbor_features)

        edge_bias = self.edge_linear(edge_context) * normalization.unsqueeze(-1)
        aggregated_edge_bias = edge_bias.new_zeros(
            node_count, edge_bias.shape[-1]
        ).index_add_(0, destination, edge_bias)
        return output + aggregated + aggregated_edge_bias.unsqueeze(0)


class _FullGraphGNN(nn.Module):
    """Shared full-graph backbone used by spectral trend and residual heads."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_metadata_dim: int,
        hidden_dim: int,
        dropout: float,
        layers: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # The extra scalar tells one shared network which directed branch is
        # being evaluated; edge metadata otherwise keeps its original width.
        edge_context_dim = edge_metadata_dim + 1
        self.layers = nn.ModuleList(
            WeightedDirectedGraphLayer(
                hidden_dim, hidden_dim, edge_context_dim
            )
            for _ in range(layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_metadata: torch.Tensor,
        direction: int,
    ) -> torch.Tensor:
        branch = edge_metadata.new_full(
            (edge_metadata.shape[0], 1), float(direction)
        )
        edge_context = torch.cat((edge_metadata, branch), dim=-1)
        hidden = self.dropout(F.relu(self.input_projection(node_features)))
        for layer in self.layers:
            hidden = self.dropout(
                F.relu(
                    layer(
                        hidden,
                        source,
                        destination,
                        edge_weight,
                        edge_context,
                    )
                )
            )
        return self.output_projection(hidden)


class SpectralTrendGNN(nn.Module):
    """Full-road-graph GNN over retained FFT amplitude and phase features."""

    def __init__(
        self,
        edge_metadata_dim: int,
        window_size: int,
        trend_frequency_bins: int,
        hidden_dim: int,
        dropout: float,
        time_feature_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.trend_bins = trend_frequency_bins
        self.network = _FullGraphGNN(
            input_dim=window_size + trend_frequency_bins * 3 + time_feature_dim,
            output_dim=trend_frequency_bins * 3,
            edge_metadata_dim=edge_metadata_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            layers=layers,
        )

    def forward(
        self,
        source_state: torch.Tensor,
        source_valid: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_metadata: torch.Tensor,
        time_feature: torch.Tensor,
        direction: int,
    ) -> torch.Tensor:
        if not source_state.is_complex():
            raise TypeError("SpectralTrendGNN requires complex FFT coefficients")
        amplitude = torch.abs(source_state)
        safe_amplitude = amplitude.clamp_min(1e-4)
        spectral_features = torch.cat(
            (
                amplitude,
                source_state.imag / safe_amplitude,
                source_state.real / safe_amplitude,
            ),
            dim=-1,
        )
        node_time = time_feature.unsqueeze(1).expand(
            -1, source_state.shape[1], -1
        )
        features = torch.cat(
            (source_valid.float(), spectral_features, node_time), dim=-1
        )
        output = self.network(
            features,
            source,
            destination,
            edge_weight,
            edge_metadata,
            direction,
        )
        bins = self.trend_bins
        coefficient_amplitude = F.softplus(output[..., :bins])
        phase_sin = output[..., bins : 2 * bins]
        phase_cos = output[..., 2 * bins :]
        norm = torch.sqrt(phase_sin.square() + phase_cos.square() + 1e-8)
        phase_sin, phase_cos = phase_sin / norm, phase_cos / norm
        phase_sin = torch.cat(
            (torch.zeros_like(phase_sin[..., :1]), phase_sin[..., 1:]), -1
        )
        phase_cos = torch.cat(
            (torch.ones_like(phase_cos[..., :1]), phase_cos[..., 1:]), -1
        )
        return torch.complex(
            coefficient_amplitude * phase_cos,
            coefficient_amplitude * phase_sin,
        )


class ResidualGNN(nn.Module):
    """Full-road-graph residual counterpart with the same graph operator."""

    def __init__(
        self,
        edge_metadata_dim: int,
        window_size: int,
        hidden_dim: int,
        dropout: float,
        time_feature_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.network = _FullGraphGNN(
            input_dim=window_size * 2 + time_feature_dim,
            output_dim=1,
            edge_metadata_dim=edge_metadata_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            layers=layers,
        )

    def forward(
        self,
        residual_window: torch.Tensor,
        source_valid: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_metadata: torch.Tensor,
        time_feature: torch.Tensor,
        direction: int,
    ) -> torch.Tensor:
        node_time = time_feature.unsqueeze(1).expand(
            -1, residual_window.shape[1], -1
        )
        features = torch.cat(
            (residual_window, source_valid.float(), node_time), dim=-1
        )
        return self.network(
            features,
            source,
            destination,
            edge_weight,
            edge_metadata,
            direction,
        ).squeeze(-1)


class MetadataOnlyTrendMLP(nn.Module):
    """Predict the scalar trend from focal-node metadata and calendar time."""

    def __init__(
        self,
        node_metadata_dim: int,
        hidden_dim: int,
        dropout: float,
        time_feature_dim: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(node_metadata_dim + time_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, node_metadata: torch.Tensor, time_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((node_metadata, time_feature), dim=-1)).squeeze(-1)


class MetadataOnlyResidualMLP(nn.Module):
    """Refine the metadata-only trend without traffic or topology inputs."""

    def __init__(
        self,
        node_metadata_dim: int,
        hidden_dim: int,
        dropout: float,
        time_feature_dim: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(node_metadata_dim + time_feature_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_metadata: torch.Tensor,
        time_feature: torch.Tensor,
        trend: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((node_metadata, time_feature, trend.unsqueeze(-1)), dim=-1)
        return self.network(features).squeeze(-1)


class SPN(nn.Module):
    """Spectral Propagation Network with controlled ablation variants."""

    def __init__(
        self,
        edge_metadata_dim: int,
        window_size: int,
        trend_frequency_bins: int,
        hidden_dim: int,
        dropout: float,
        time_feature_dim: int,
        branch_aggregation: str,
        max_propagation_hop: int | None,
        node_metadata_dim: int = 14,
        use_fft_encoding: bool = True,
        use_fft_decomposition: bool = True,
        use_metadata: bool = True,
        metadata_only: bool = False,
        time_window_impute: bool = False,
        use_subgraph_frequency_topk: bool = False,
        use_graph_network: bool = False,
        gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.trend_frequency_bins = trend_frequency_bins
        self.branch_aggregation = branch_aggregation
        self.max_propagation_hop = max_propagation_hop
        self.use_fft_encoding = use_fft_encoding
        self.use_fft_decomposition = use_fft_decomposition
        self.use_metadata = use_metadata
        self.metadata_only = metadata_only
        self.time_window_impute = time_window_impute
        self.use_subgraph_frequency_topk = use_subgraph_frequency_topk
        self.use_graph_network = use_graph_network
        if metadata_only and not use_metadata:
            raise ValueError("metadata_only requires use_metadata=True")
        if use_fft_encoding and not use_fft_decomposition:
            raise ValueError("spectral propagation requires FFT decomposition")
        if time_window_impute and not use_fft_encoding:
            raise ValueError("time_window_impute requires FFT encoding")
        if time_window_impute and metadata_only:
            raise ValueError("time_window_impute is incompatible with metadata_only")
        if use_subgraph_frequency_topk and not use_fft_encoding:
            raise ValueError("subgraph frequency Top-K requires FFT encoding")
        if use_subgraph_frequency_topk and metadata_only:
            raise ValueError(
                "subgraph frequency Top-K is incompatible with metadata_only"
            )
        if use_graph_network and not use_fft_encoding:
            raise ValueError("the full-graph GNN requires FFT spectral features")
        if use_graph_network and time_window_impute:
            raise ValueError(
                "the full-graph GNN currently supports point imputation only"
            )
        if use_graph_network and use_subgraph_frequency_topk:
            raise ValueError(
                "the full-graph GNN is incompatible with subgraph frequency Top-K"
            )
        if gnn_layers <= 0:
            raise ValueError("gnn_layers must be positive")
        if metadata_only:
            self.trend_mlp = MetadataOnlyTrendMLP(
                node_metadata_dim, hidden_dim, dropout, time_feature_dim
            )
            self.residual_mlp = MetadataOnlyResidualMLP(
                node_metadata_dim, hidden_dim, dropout, time_feature_dim
            )
        elif use_graph_network:
            self.trend_mlp = SpectralTrendGNN(
                edge_metadata_dim,
                window_size,
                trend_frequency_bins,
                hidden_dim,
                dropout,
                time_feature_dim,
                gnn_layers,
            )
            self.residual_mlp = ResidualGNN(
                edge_metadata_dim,
                window_size,
                hidden_dim,
                dropout,
                time_feature_dim,
                gnn_layers,
            )
        else:
            self.trend_mlp = SpectralTrendMLP(
                edge_metadata_dim,
                window_size,
                trend_frequency_bins,
                hidden_dim,
                dropout,
                time_feature_dim,
                use_fft_encoding,
                stabilize_recursive_spectrum=time_window_impute,
                use_frequency_indices=use_subgraph_frequency_topk,
            )
            self.residual_mlp = ResidualMLP(
                edge_metadata_dim,
                window_size,
                hidden_dim,
                dropout,
                time_feature_dim,
                output_dim=window_size if time_window_impute else 1,
            )

    def _eligible_selection_boundary(
        self, graph: dict[str, torch.Tensor | int]
    ) -> torch.Tensor:
        """Return the visible boundary nodes allowed by the hop limit."""
        boundary = graph["boundary_mask"]
        assert isinstance(boundary, torch.Tensor)
        if self.max_propagation_hop is None:
            return boundary.bool()
        upstream = graph["upstream_boundary_mask"]
        downstream = graph["downstream_boundary_mask"]
        up_hop = graph["up_hop"]
        down_hop = graph["down_hop"]
        assert isinstance(upstream, torch.Tensor)
        assert isinstance(downstream, torch.Tensor)
        assert isinstance(up_hop, torch.Tensor)
        assert isinstance(down_hop, torch.Tensor)
        return (
            upstream.bool()
            & (up_hop > 0)
            & (up_hop <= self.max_propagation_hop)
        ) | (
            downstream.bool()
            & (down_hop > 0)
            & (down_hop <= self.max_propagation_hop)
        )

    def _top_frequency_indices(
        self, amplitude: torch.Tensor, available_count: torch.Tensor
    ) -> torch.Tensor:
        """Keep DC and rank the remaining shared bins by boundary amplitude."""
        retained = self.trend_frequency_bins
        frequency_count = self.window_size // 2 + 1
        fallback = torch.arange(retained, device=amplitude.device).view(1, 1, -1)
        fallback = fallback.expand(*amplitude.shape[:-1], -1)
        if retained == frequency_count:
            return fallback
        if retained == 1:
            selected = fallback
        else:
            strongest = torch.topk(
                amplitude[..., 1:], retained - 1, dim=-1
            ).indices + 1
            selected = torch.cat(
                (torch.zeros_like(strongest[..., :1]), strongest), dim=-1
            )
            selected = torch.sort(selected, dim=-1).values
        return torch.where(
            (available_count > 0).unsqueeze(-1), selected, fallback
        ).detach()

    def _select_subgraph_frequency_indices(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
    ) -> torch.Tensor | None:
        """Select a shared Top-K spectrum using visible boundary signals only.

        Point imputation returns ``[time, component, K]``; whole-window
        imputation returns ``[1, component, K]``.  Every node duplicated inside
        the same query-centered subgraph therefore uses identical frequency
        coordinates at a given causal window.
        """
        if not self.use_subgraph_frequency_topk:
            return None
        if bool(graph.get("full_graph", 0)):
            raise ValueError(
                "subgraph frequency Top-K requires query-centered subgraphs"
            )
        node_component = graph["node_component"]
        assert isinstance(node_component, torch.Tensor)
        component_count = int(graph["num_components"])
        boundary = self._eligible_selection_boundary(graph)
        boundary_nodes = torch.nonzero(boundary, as_tuple=False).flatten()
        frequency_count = self.window_size // 2 + 1

        if self.time_window_impute:
            boundary_valid = observed[:, boundary_nodes].transpose(0, 1).bool()
            boundary_values = interpolate_temporal_window(
                input_flow[:, boundary_nodes].transpose(0, 1), boundary_valid
            )
            spectrum = torch.fft.rfft(
                boundary_values,
                n=self.window_size,
                dim=-1,
                norm="forward",
            ).unsqueeze(0)
            available = boundary_valid.any(dim=-1).unsqueeze(0)
        else:
            boundary_observed = observed[:, boundary_nodes].bool()
            boundary_windows = make_temporal_windows(
                input_flow[:, boundary_nodes], self.window_size
            )
            boundary_valid_windows = make_temporal_windows(
                boundary_observed.float(), self.window_size
            ) > 0
            boundary_windows = interpolate_temporal_window(
                boundary_windows, boundary_valid_windows
            )
            spectrum = torch.fft.rfft(
                boundary_windows,
                n=self.window_size,
                dim=-1,
                norm="ortho",
            )
            available = boundary_valid_windows.any(dim=-1)

        sample_count = spectrum.shape[0]
        amplitude = input_flow.new_zeros(
            sample_count, component_count, frequency_count
        )
        count = input_flow.new_zeros(sample_count, component_count)
        if boundary_nodes.numel():
            components = node_component[boundary_nodes]
            amplitude.index_add_(
                1,
                components,
                torch.abs(spectrum) * available.unsqueeze(-1).to(input_flow.dtype),
            )
            count.index_add_(1, components, available.to(input_flow.dtype))
            amplitude = amplitude / count.clamp_min(1).unsqueeze(-1)
        return self._top_frequency_indices(amplitude, count)

    def _direction_graph(
        self, graph: dict[str, torch.Tensor | int], direction: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...] | None:
        edge_direction = graph["edge_direction"]
        edge_hop = graph["edge_hop"]
        assert isinstance(edge_direction, torch.Tensor)
        assert isinstance(edge_hop, torch.Tensor)
        mask = edge_direction == direction
        if self.max_propagation_hop is not None:
            mask &= edge_hop <= self.max_propagation_hop
        if not mask.any():
            return None
        edge_index = graph["edge_index"]
        edge_weight = graph["edge_weight"]
        edge_metadata = graph["edge_metadata"]
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(edge_weight, torch.Tensor)
        assert isinstance(edge_metadata, torch.Tensor)
        if not self.use_metadata:
            # Controlled metadata ablation: preserve the input width, parameter
            # count, topology, weights, propagation direction, and node degrees.
            edge_metadata = torch.zeros_like(edge_metadata)
        edges = edge_index[:, mask]
        source, destination = edges[0], edges[1]
        node_count = int(graph["num_nodes"])
        out_degree = torch.zeros(node_count, dtype=dtype, device=source.device)
        in_degree = torch.zeros_like(out_degree)
        ones = torch.ones(source.numel(), dtype=dtype, device=source.device)
        out_degree.index_add_(0, source, ones)
        in_degree.index_add_(0, destination, ones)
        return (
            source,
            destination,
            edge_weight[mask],
            edge_hop[mask],
            edge_metadata[mask],
            out_degree,
            in_degree,
        )

    def _aggregate(
        self, candidate: torch.Tensor, valid: torch.Tensor, destination: torch.Tensor, nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps = candidate.shape[0]
        output = candidate.new_zeros(time_steps, nodes).index_add_(
            1, destination, candidate * valid.float()
        )
        count = candidate.new_zeros(time_steps, nodes).index_add_(1, destination, valid.float())
        if self.branch_aggregation == "mean":
            output = output / count.clamp_min(1)
        return output, count > 0

    def _aggregate_spectral(
        self,
        candidate: torch.Tensor,
        valid: torch.Tensor,
        destination: torch.Tensor,
        nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate complex coefficients without decoding intermediate nodes."""
        time_steps, _, bins = candidate.shape
        output = candidate.new_zeros(time_steps, nodes, bins).index_add_(
            1, destination, candidate * valid.unsqueeze(-1).to(candidate.dtype)
        )
        count = candidate.real.new_zeros(time_steps, nodes).index_add_(
            1, destination, valid.float()
        )
        if self.branch_aggregation == "mean":
            output = output / count.clamp_min(1).unsqueeze(-1)
        return output, count > 0

    def _aggregate_temporal_windows(
        self,
        candidate: torch.Tensor,
        valid: torch.Tensor,
        destination: torch.Tensor,
        nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate one complete time-domain trend window per edge."""
        time_steps = candidate.shape[0]
        output = candidate.new_zeros(time_steps, nodes, self.window_size).index_add_(
            1,
            destination,
            candidate * valid.unsqueeze(-1).to(candidate.dtype),
        )
        count = candidate.new_zeros(time_steps, nodes).index_add_(
            1, destination, valid.float()
        )
        if self.branch_aggregation == "mean":
            output = output / count.clamp_min(1).unsqueeze(-1)
        return output, count > 0

    def _propagate_temporal_trend_windows(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Propagate complete low-pass or raw-flow windows in the time domain.

        With FFT decomposition enabled, FFT/iFFT construct boundary trend
        windows. Strict w/o FFT instead starts from interpolated raw windows.
        The shared MLP maps ``W -> W`` at every hop, and only the last point of
        the final focal-node window is used for supervision.
        """
        time_steps, nodes = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        node_hop = graph["up_hop" if direction == 1 else "down_hop"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(node_hop, torch.Tensor)
        assert isinstance(centers, torch.Tensor)
        if self.max_propagation_hop is not None:
            boundary = boundary & (node_hop > 0) & (
                node_hop <= self.max_propagation_hop
            )

        state = input_flow.new_zeros(time_steps, nodes, self.window_size)
        valid = torch.zeros(time_steps, nodes, dtype=torch.bool, device=input_flow.device)
        boundary_nodes = torch.nonzero(boundary, as_tuple=False).flatten()
        if boundary_nodes.numel():
            boundary_values = input_flow[:, boundary_nodes]
            boundary_observed = observed[:, boundary_nodes].bool()
            if self.use_fft_decomposition:
                boundary_windows, boundary_valid_windows = lowpass_causal_windows(
                    boundary_values,
                    boundary_observed,
                    self.window_size,
                    self.trend_frequency_bins,
                )
            else:
                boundary_windows = make_temporal_windows(
                    boundary_values, self.window_size
                )
                boundary_valid_windows = make_temporal_windows(
                    boundary_observed.float(), self.window_size
                ) > 0
                boundary_windows = interpolate_temporal_window(
                    boundary_windows, boundary_valid_windows
                )
            boundary_valid = boundary_valid_windows.any(dim=-1)
            state[:, boundary_nodes] = torch.where(
                boundary_valid.unsqueeze(-1),
                boundary_windows,
                torch.zeros_like(boundary_windows),
            )
            valid[:, boundary_nodes] = boundary_valid

        direction_graph = self._direction_graph(graph, direction, input_flow.dtype)
        if direction_graph is not None:
            source_all, destination_all, weights, hops, metadata, out_degree, in_degree = (
                direction_graph
            )
            for hop in range(int(hops.max()), 0, -1):
                hop_mask = hops == hop
                if not hop_mask.any():
                    continue
                source = source_all[hop_mask]
                destination = destination_all[hop_mask]
                source_valid = valid[:, source]
                if not source_valid.any():
                    continue

                edge_count = source.numel()
                source_valid_windows = torch.ones(
                    time_steps,
                    edge_count,
                    self.window_size,
                    dtype=torch.bool,
                    device=input_flow.device,
                )
                boundary_sources = boundary[source]
                if boundary_sources.any():
                    source_valid_windows[:, boundary_sources] = make_temporal_windows(
                        observed[:, source[boundary_sources]].float(),
                        self.window_size,
                    ) > 0
                common = (
                    weights[hop_mask].unsqueeze(0).expand(time_steps, edge_count),
                    torch.full(
                        (time_steps, edge_count),
                        float(direction),
                        dtype=input_flow.dtype,
                        device=input_flow.device,
                    ),
                    out_degree[source].unsqueeze(0).expand(time_steps, edge_count),
                    in_degree[destination].unsqueeze(0).expand(time_steps, edge_count),
                    metadata[hop_mask].unsqueeze(0).expand(time_steps, edge_count, -1),
                    time_feature.unsqueeze(1).expand(time_steps, edge_count, -1),
                )
                candidate = self.trend_mlp(
                    state[:, source], source_valid_windows, *common
                )
                aggregated, new_valid = self._aggregate_temporal_windows(
                    candidate, source_valid, destination, nodes
                )
                writable = new_valid & ~boundary.unsqueeze(0)
                state = torch.where(writable.unsqueeze(-1), aggregated, state)
                valid |= new_valid

        decoded_state = state[..., -1]
        return decoded_state[:, centers], valid[:, centers], decoded_state, None

    def _propagate_spectral_trend(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
        frequency_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagate low-frequency coefficients and decode only after the last hop."""
        time_steps, nodes = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        node_hop = graph["up_hop" if direction == 1 else "down_hop"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(node_hop, torch.Tensor)
        assert isinstance(centers, torch.Tensor)
        node_frequency_indices: torch.Tensor | None = None
        if frequency_indices is not None:
            node_component = graph["node_component"]
            assert isinstance(node_component, torch.Tensor)
            node_frequency_indices = frequency_indices[:, node_component]
        if self.max_propagation_hop is not None:
            boundary = boundary & (node_hop > 0) & (
                node_hop <= self.max_propagation_hop
            )

        direction_graph = self._direction_graph(graph, direction, input_flow.dtype)
        state = torch.zeros(
            time_steps,
            nodes,
            self.trend_frequency_bins,
            dtype=torch.complex64,
            device=input_flow.device,
        )
        valid = torch.zeros(time_steps, nodes, dtype=torch.bool, device=input_flow.device)

        boundary_nodes = torch.nonzero(boundary, as_tuple=False).flatten()
        if boundary_nodes.numel():
            boundary_observed = observed[:, boundary_nodes].bool()
            boundary_windows = make_temporal_windows(
                input_flow[:, boundary_nodes], self.window_size
            )
            boundary_valid_windows = make_temporal_windows(
                boundary_observed.float(), self.window_size
            ) > 0
            boundary_windows = interpolate_temporal_window(
                boundary_windows, boundary_valid_windows
            )
            boundary_spectrum = torch.fft.rfft(
                boundary_windows,
                n=self.window_size,
                dim=-1,
                norm="ortho",
            )
            if node_frequency_indices is None:
                boundary_spectrum = boundary_spectrum[..., : self.trend_frequency_bins]
            else:
                boundary_spectrum = torch.gather(
                    boundary_spectrum,
                    -1,
                    node_frequency_indices[:, boundary_nodes],
                )
            boundary_valid = boundary_valid_windows.any(dim=-1)
            state[:, boundary_nodes] = torch.where(
                boundary_valid.unsqueeze(-1),
                boundary_spectrum,
                torch.zeros_like(boundary_spectrum),
            )
            valid[:, boundary_nodes] = boundary_valid

        if direction_graph is not None:
            source_all, destination_all, weights, hops, metadata, out_degree, in_degree = (
                direction_graph
            )
            for hop in range(int(hops.max()), 0, -1):
                hop_mask = hops == hop
                if not hop_mask.any():
                    continue
                source, destination = source_all[hop_mask], destination_all[hop_mask]
                source_valid = valid[:, source]
                if not source_valid.any():
                    continue

                edge_count = source.numel()
                source_valid_windows = torch.ones(
                    time_steps,
                    edge_count,
                    self.window_size,
                    dtype=torch.bool,
                    device=input_flow.device,
                )
                boundary_sources = boundary[source]
                if boundary_sources.any():
                    source_valid_windows[:, boundary_sources] = make_temporal_windows(
                        observed[:, source[boundary_sources]].float(), self.window_size
                    ) > 0

                common = (
                    weights[hop_mask].unsqueeze(0).expand(time_steps, edge_count),
                    torch.full(
                        (time_steps, edge_count),
                        float(direction),
                        dtype=input_flow.dtype,
                        device=input_flow.device,
                    ),
                    out_degree[source].unsqueeze(0).expand(time_steps, edge_count),
                    in_degree[destination].unsqueeze(0).expand(time_steps, edge_count),
                    metadata[hop_mask].unsqueeze(0).expand(time_steps, edge_count, -1),
                    time_feature.unsqueeze(1).expand(time_steps, edge_count, -1),
                )
                if node_frequency_indices is None:
                    candidate = self.trend_mlp(
                        state[:, source], source_valid_windows, *common
                    )
                else:
                    candidate = self.trend_mlp(
                        state[:, source],
                        source_valid_windows,
                        *common,
                        node_frequency_indices[:, source],
                    )
                aggregated, new_valid = self._aggregate_spectral(
                    candidate, source_valid, destination, nodes
                )
                writable = new_valid & ~boundary.unsqueeze(0)
                state = torch.where(writable.unsqueeze(-1), aggregated, state)
                valid |= new_valid

        decoded_state = decode_spectral_current(
            state, self.window_size, node_frequency_indices
        )
        return (
            decoded_state[:, centers],
            valid[:, centers],
            decoded_state,
            state[:, centers],
        )

    def _full_graph_rounds(self, graph: dict[str, torch.Tensor | int]) -> int:
        rounds = int(graph["propagation_depth"])
        if self.max_propagation_hop is not None:
            rounds = min(rounds, self.max_propagation_hop)
        return max(1, rounds)

    def _graph_network_edges(
        self,
        graph: dict[str, torch.Tensor | int],
        direction: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one directed full-graph edge set for the GNN ablation."""
        direction_graph = self._direction_graph(graph, direction, dtype)
        if direction_graph is not None:
            source, destination, weights, _, metadata, _, _ = direction_graph
            return source, destination, weights, metadata
        edge_index = graph["edge_index"]
        edge_weight = graph["edge_weight"]
        edge_metadata = graph["edge_metadata"]
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(edge_weight, torch.Tensor)
        assert isinstance(edge_metadata, torch.Tensor)
        return (
            edge_index.new_empty(0),
            edge_index.new_empty(0),
            edge_weight.new_empty(0),
            edge_metadata.new_empty((0, edge_metadata.shape[-1])),
        )

    def _propagate_graph_network_spectral(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict retained spectra with an ordinary full-graph GNN."""
        time_steps, nodes = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(centers, torch.Tensor)

        valid_points = observed.bool() & boundary.unsqueeze(0)
        valid_windows = make_temporal_windows(
            valid_points.float(), self.window_size
        ) > 0
        node_windows = make_temporal_windows(input_flow, self.window_size)
        node_windows = interpolate_temporal_window(node_windows, valid_windows)
        spectrum = torch.fft.rfft(
            node_windows, n=self.window_size, dim=-1, norm="ortho"
        )[..., : self.trend_frequency_bins]
        available = valid_windows.any(dim=-1)
        spectrum = torch.where(
            available.unsqueeze(-1), spectrum, torch.zeros_like(spectrum)
        )

        source, destination, weights, metadata = self._graph_network_edges(
            graph, direction, input_flow.dtype
        )
        predicted_spectrum = self.trend_mlp(
            spectrum,
            valid_windows,
            source,
            destination,
            weights,
            metadata,
            time_feature,
            direction,
        )
        decoded = decode_spectral_current(predicted_spectrum, self.window_size)
        prediction_valid = torch.ones(
            time_steps,
            centers.numel(),
            dtype=torch.bool,
            device=input_flow.device,
        )
        return (
            decoded[:, centers],
            prediction_valid,
            decoded,
            predicted_spectrum[:, centers],
        )

    def _propagate_graph_network_residual(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Predict the same residual target using a matched full-graph GNN."""
        time_steps, _ = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(centers, torch.Tensor)

        valid_points = observed.bool() & boundary.unsqueeze(0)
        boundary_trend = trend_current(
            input_flow,
            valid_points,
            self.window_size,
            self.trend_frequency_bins,
            self.use_fft_decomposition,
        )
        residual = torch.where(
            valid_points,
            input_flow - boundary_trend,
            torch.zeros_like(input_flow),
        )
        residual_windows = make_temporal_windows(residual, self.window_size)
        valid_windows = make_temporal_windows(
            valid_points.float(), self.window_size
        ) > 0
        source, destination, weights, metadata = self._graph_network_edges(
            graph, direction, input_flow.dtype
        )
        predicted_residual = self.residual_mlp(
            residual_windows,
            valid_windows,
            source,
            destination,
            weights,
            metadata,
            time_feature,
            direction,
        )
        prediction_valid = torch.ones(
            time_steps,
            centers.numel(),
            dtype=torch.bool,
            device=input_flow.device,
        )
        return (
            predicted_residual[:, centers],
            prediction_valid,
            predicted_residual,
            None,
        )

    def _propagate_full_graph_spectral(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synchronously diffuse spectral states over the complete road graph."""
        time_steps, nodes = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(centers, torch.Tensor)
        direction_graph = self._direction_graph(graph, direction, input_flow.dtype)

        state = torch.zeros(
            time_steps,
            nodes,
            self.trend_frequency_bins,
            dtype=torch.complex64,
            device=input_flow.device,
        )
        valid = torch.zeros(time_steps, nodes, dtype=torch.bool, device=input_flow.device)
        boundary_nodes = torch.nonzero(boundary, as_tuple=False).flatten()
        if boundary_nodes.numel():
            boundary_observed = observed[:, boundary_nodes].bool()
            boundary_windows = make_temporal_windows(
                input_flow[:, boundary_nodes], self.window_size
            )
            boundary_valid_windows = make_temporal_windows(
                boundary_observed.float(), self.window_size
            ) > 0
            boundary_windows = interpolate_temporal_window(
                boundary_windows, boundary_valid_windows
            )
            boundary_spectrum = torch.fft.rfft(
                boundary_windows,
                n=self.window_size,
                dim=-1,
                norm="ortho",
            )[..., : self.trend_frequency_bins]
            boundary_valid = boundary_valid_windows.any(dim=-1)
            state[:, boundary_nodes] = torch.where(
                boundary_valid.unsqueeze(-1),
                boundary_spectrum,
                torch.zeros_like(boundary_spectrum),
            )
            valid[:, boundary_nodes] = boundary_valid

        frontier = valid.clone()
        if direction_graph is not None:
            source_all, destination_all, weights, _, metadata, out_degree, in_degree = (
                direction_graph
            )
            for _ in range(self._full_graph_rounds(graph)):
                if not frontier.any():
                    break
                active_edges = torch.nonzero(
                    frontier[:, source_all].any(dim=0), as_tuple=False
                ).flatten()
                if not active_edges.numel():
                    break
                aggregated = state.new_zeros(
                    time_steps, nodes, self.trend_frequency_bins
                )
                count = input_flow.new_zeros(time_steps, nodes)
                for edge_chunk in active_edges.split(_FULL_GRAPH_EDGE_CHUNK_SIZE):
                    source = source_all[edge_chunk]
                    destination = destination_all[edge_chunk]
                    source_valid = frontier[:, source]
                    edge_count = source.numel()
                    source_valid_windows = torch.ones(
                        time_steps,
                        edge_count,
                        self.window_size,
                        dtype=torch.bool,
                        device=input_flow.device,
                    )
                    boundary_sources = boundary[source]
                    if boundary_sources.any():
                        source_valid_windows[:, boundary_sources] = make_temporal_windows(
                            observed[:, source[boundary_sources]].float(),
                            self.window_size,
                        ) > 0
                    common = (
                        weights[edge_chunk].unsqueeze(0).expand(time_steps, edge_count),
                        torch.full(
                            (time_steps, edge_count),
                            float(direction),
                            dtype=input_flow.dtype,
                            device=input_flow.device,
                        ),
                        out_degree[source].unsqueeze(0).expand(time_steps, edge_count),
                        in_degree[destination].unsqueeze(0).expand(time_steps, edge_count),
                        metadata[edge_chunk]
                        .unsqueeze(0)
                        .expand(time_steps, edge_count, -1),
                        time_feature.unsqueeze(1).expand(time_steps, edge_count, -1),
                    )
                    candidate = self.trend_mlp(
                        state[:, source], source_valid_windows, *common
                    )
                    aggregated.index_add_(
                        1,
                        destination,
                        candidate * source_valid.unsqueeze(-1).to(candidate.dtype),
                    )
                    count.index_add_(1, destination, source_valid.float())
                if self.branch_aggregation == "mean":
                    aggregated = aggregated / count.clamp_min(1).unsqueeze(-1)
                received = count > 0
                frontier = received & ~valid & ~boundary.unsqueeze(0)
                state = torch.where(frontier.unsqueeze(-1), aggregated, state)
                valid |= frontier

        decoded_state = decode_spectral_current(state, self.window_size)
        return (
            decoded_state[:, centers],
            valid[:, centers],
            decoded_state,
            state[:, centers],
        )

    def _propagate_full_graph_temporal_trend_windows(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Synchronously diffuse complete low-pass or raw windows over the graph."""
        time_steps, nodes = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(centers, torch.Tensor)
        direction_graph = self._direction_graph(graph, direction, input_flow.dtype)

        state = input_flow.new_zeros(time_steps, nodes, self.window_size)
        valid = torch.zeros(time_steps, nodes, dtype=torch.bool, device=input_flow.device)
        boundary_nodes = torch.nonzero(boundary, as_tuple=False).flatten()
        if boundary_nodes.numel():
            boundary_values = input_flow[:, boundary_nodes]
            boundary_observed = observed[:, boundary_nodes].bool()
            if self.use_fft_decomposition:
                boundary_windows, boundary_valid_windows = lowpass_causal_windows(
                    boundary_values,
                    boundary_observed,
                    self.window_size,
                    self.trend_frequency_bins,
                )
            else:
                boundary_windows = make_temporal_windows(
                    boundary_values, self.window_size
                )
                boundary_valid_windows = make_temporal_windows(
                    boundary_observed.float(), self.window_size
                ) > 0
                boundary_windows = interpolate_temporal_window(
                    boundary_windows, boundary_valid_windows
                )
            boundary_valid = boundary_valid_windows.any(dim=-1)
            state[:, boundary_nodes] = torch.where(
                boundary_valid.unsqueeze(-1),
                boundary_windows,
                torch.zeros_like(boundary_windows),
            )
            valid[:, boundary_nodes] = boundary_valid

        frontier = valid.clone()
        if direction_graph is not None:
            source_all, destination_all, weights, _, metadata, out_degree, in_degree = (
                direction_graph
            )
            for _ in range(self._full_graph_rounds(graph)):
                if not frontier.any():
                    break
                active_edges = torch.nonzero(
                    frontier[:, source_all].any(dim=0), as_tuple=False
                ).flatten()
                if not active_edges.numel():
                    break
                aggregated = input_flow.new_zeros(
                    time_steps, nodes, self.window_size
                )
                count = input_flow.new_zeros(time_steps, nodes)
                for edge_chunk in active_edges.split(_FULL_GRAPH_EDGE_CHUNK_SIZE):
                    source = source_all[edge_chunk]
                    destination = destination_all[edge_chunk]
                    source_valid = frontier[:, source]
                    edge_count = source.numel()
                    source_valid_windows = torch.ones(
                        time_steps,
                        edge_count,
                        self.window_size,
                        dtype=torch.bool,
                        device=input_flow.device,
                    )
                    boundary_sources = boundary[source]
                    if boundary_sources.any():
                        source_valid_windows[:, boundary_sources] = make_temporal_windows(
                            observed[:, source[boundary_sources]].float(),
                            self.window_size,
                        ) > 0
                    common = (
                        weights[edge_chunk].unsqueeze(0).expand(time_steps, edge_count),
                        torch.full(
                            (time_steps, edge_count),
                            float(direction),
                            dtype=input_flow.dtype,
                            device=input_flow.device,
                        ),
                        out_degree[source].unsqueeze(0).expand(time_steps, edge_count),
                        in_degree[destination].unsqueeze(0).expand(time_steps, edge_count),
                        metadata[edge_chunk]
                        .unsqueeze(0)
                        .expand(time_steps, edge_count, -1),
                        time_feature.unsqueeze(1).expand(time_steps, edge_count, -1),
                    )
                    candidate = self.trend_mlp(
                        state[:, source], source_valid_windows, *common
                    )
                    aggregated.index_add_(
                        1,
                        destination,
                        candidate * source_valid.unsqueeze(-1).to(candidate.dtype),
                    )
                    count.index_add_(1, destination, source_valid.float())
                if self.branch_aggregation == "mean":
                    aggregated = aggregated / count.clamp_min(1).unsqueeze(-1)
                received = count > 0
                frontier = received & ~valid & ~boundary.unsqueeze(0)
                state = torch.where(frontier.unsqueeze(-1), aggregated, state)
                valid |= frontier

        decoded_state = state[..., -1]
        return decoded_state[:, centers], valid[:, centers], decoded_state, None

    def _propagate_full_graph_temporal(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
        residual: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Synchronously diffuse scalar residual states over the full graph."""
        time_steps, nodes = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(centers, torch.Tensor)
        direction_graph = self._direction_graph(graph, direction, input_flow.dtype)
        valid = observed.bool() & boundary.unsqueeze(0)
        boundary_trend = trend_current(
            input_flow,
            valid,
            self.window_size,
            self.trend_frequency_bins,
            self.use_fft_decomposition,
        )
        if residual:
            state = torch.where(
                valid, input_flow - boundary_trend, torch.zeros_like(input_flow)
            )
            propagated_state = state
        else:
            state = torch.where(valid, input_flow, torch.zeros_like(input_flow))
            propagated_state = torch.where(
                valid, boundary_trend, torch.zeros_like(input_flow)
            )

        frontier = torch.zeros_like(valid)
        boundary_nodes = torch.nonzero(boundary, as_tuple=False).flatten()
        if boundary_nodes.numel():
            frontier[:, boundary_nodes] = make_temporal_windows(
                valid[:, boundary_nodes].float(), self.window_size
            ).bool().any(dim=-1)

        if direction_graph is not None:
            source_all, destination_all, weights, _, metadata, out_degree, in_degree = (
                direction_graph
            )
            for _ in range(self._full_graph_rounds(graph)):
                if not frontier.any():
                    break
                active_edges = torch.nonzero(
                    frontier[:, source_all].any(dim=0), as_tuple=False
                ).flatten()
                if not active_edges.numel():
                    break
                aggregated = input_flow.new_zeros(time_steps, nodes)
                count = input_flow.new_zeros(time_steps, nodes)
                for edge_chunk in active_edges.split(_FULL_GRAPH_EDGE_CHUNK_SIZE):
                    source = source_all[edge_chunk]
                    destination = destination_all[edge_chunk]
                    edge_count = source.numel()
                    valid_window = make_temporal_windows(
                        valid[:, source].float(), self.window_size
                    ) > 0
                    source_valid = frontier[:, source] & valid_window.any(dim=-1)
                    state_window = make_temporal_windows(
                        state[:, source], self.window_size
                    )
                    common = (
                        weights[edge_chunk].unsqueeze(0).expand(time_steps, edge_count),
                        torch.full(
                            (time_steps, edge_count),
                            float(direction),
                            dtype=input_flow.dtype,
                            device=input_flow.device,
                        ),
                        out_degree[source].unsqueeze(0).expand(time_steps, edge_count),
                        in_degree[destination].unsqueeze(0).expand(time_steps, edge_count),
                        metadata[edge_chunk]
                        .unsqueeze(0)
                        .expand(time_steps, edge_count, -1),
                        time_feature.unsqueeze(1).expand(time_steps, edge_count, -1),
                    )
                    if residual:
                        candidate = self.residual_mlp(
                            state_window, valid_window, *common
                        )
                    else:
                        candidate = self.trend_mlp(
                            state_window, valid_window, *common
                        )
                    aggregated.index_add_(
                        1, destination, candidate * source_valid.float()
                    )
                    count.index_add_(1, destination, source_valid.float())
                if self.branch_aggregation == "mean":
                    aggregated = aggregated / count.clamp_min(1)
                received = count > 0
                frontier = received & ~valid & ~boundary.unsqueeze(0)
                state = torch.where(frontier, aggregated, state)
                propagated_state = torch.where(
                    frontier, aggregated, propagated_state
                )
                valid |= frontier

        return propagated_state[:, centers], valid[:, centers], propagated_state, None

    def _window_edge_context(
        self,
        weights: torch.Tensor,
        direction: int,
        source: torch.Tensor,
        destination: torch.Tensor,
        metadata: torch.Tensor,
        out_degree: torch.Tensor,
        in_degree: torch.Tensor,
        time_feature: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        """Build one context vector per edge for a complete temporal window."""
        edge_count = source.numel()
        # For a day-aligned 288-point window, the spectral phase represents the
        # within-day position. Averaging retains the window-level DOW signal
        # while avoiding an arbitrary choice of one TOD endpoint.
        window_time = time_feature.mean(dim=0, keepdim=True).expand(edge_count, -1)
        return (
            weights,
            torch.full(
                (edge_count,),
                float(direction),
                dtype=dtype,
                device=source.device,
            ),
            out_degree[source],
            in_degree[destination],
            metadata,
            window_time,
        )

    def _aggregate_window_spectral(
        self,
        candidate: torch.Tensor,
        valid: torch.Tensor,
        destination: torch.Tensor,
        nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate one complex spectral message per active edge."""
        bins = candidate.shape[-1]
        output = candidate.new_zeros(nodes, bins).index_add_(
            0, destination, candidate * valid.unsqueeze(-1).to(candidate.dtype)
        )
        count = candidate.real.new_zeros(nodes).index_add_(
            0, destination, valid.float()
        )
        if self.branch_aggregation == "mean":
            output = output / count.clamp_min(1).unsqueeze(-1)
        return output, count > 0

    def _aggregate_window_values(
        self,
        candidate: torch.Tensor,
        valid: torch.Tensor,
        destination: torch.Tensor,
        nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate complete ``[window]`` residual messages by destination."""
        output = candidate.new_zeros(nodes, self.window_size).index_add_(
            0, destination, candidate * valid.unsqueeze(-1).float()
        )
        count = candidate.new_zeros(nodes).index_add_(
            0, destination, valid.float()
        )
        if self.branch_aggregation == "mean":
            output = output / count.clamp_min(1).unsqueeze(-1)
        return output.transpose(0, 1), count > 0

    def _propagate_window_spectral_trend(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
        frequency_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagate one spectrum and decode one complete focal window."""
        if input_flow.shape[0] != self.window_size:
            raise ValueError(
                "window imputation requires exactly window_size input points"
            )
        _, nodes = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        node_hop = graph["up_hop" if direction == 1 else "down_hop"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(node_hop, torch.Tensor)
        assert isinstance(centers, torch.Tensor)
        node_frequency_indices: torch.Tensor | None = None
        if frequency_indices is not None:
            node_component = graph["node_component"]
            assert isinstance(node_component, torch.Tensor)
            node_frequency_indices = frequency_indices[0, node_component]
        if self.max_propagation_hop is not None:
            boundary = boundary & (node_hop > 0) & (
                node_hop <= self.max_propagation_hop
            )

        state = torch.zeros(
            nodes,
            self.trend_frequency_bins,
            dtype=torch.complex64,
            device=input_flow.device,
        )
        node_valid = torch.zeros(nodes, dtype=torch.bool, device=input_flow.device)
        boundary_nodes = torch.nonzero(boundary, as_tuple=False).flatten()
        if boundary_nodes.numel():
            boundary_valid = observed[:, boundary_nodes].transpose(0, 1).bool()
            boundary_values = interpolate_temporal_window(
                input_flow[:, boundary_nodes].transpose(0, 1), boundary_valid
            )
            boundary_spectrum = torch.fft.rfft(
                boundary_values,
                n=self.window_size,
                dim=-1,
                norm="forward",
            )
            if node_frequency_indices is None:
                boundary_spectrum = boundary_spectrum[..., : self.trend_frequency_bins]
            else:
                boundary_spectrum = torch.gather(
                    boundary_spectrum,
                    -1,
                    node_frequency_indices[boundary_nodes],
                )
            available = boundary_valid.any(dim=-1)
            state[boundary_nodes] = torch.where(
                available.unsqueeze(-1),
                boundary_spectrum,
                torch.zeros_like(boundary_spectrum),
            )
            node_valid[boundary_nodes] = available

        direction_graph = self._direction_graph(graph, direction, input_flow.dtype)
        if direction_graph is not None:
            source_all, destination_all, weights, hops, metadata, out_degree, in_degree = (
                direction_graph
            )
            for hop in range(int(hops.max()), 0, -1):
                hop_mask = hops == hop
                if not hop_mask.any():
                    continue
                source = source_all[hop_mask]
                destination = destination_all[hop_mask]
                source_valid = node_valid[source]
                if not source_valid.any():
                    continue
                source_valid_windows = torch.ones(
                    source.numel(),
                    self.window_size,
                    dtype=torch.bool,
                    device=input_flow.device,
                )
                boundary_sources = boundary[source]
                if boundary_sources.any():
                    source_valid_windows[boundary_sources] = observed[
                        :, source[boundary_sources]
                    ].transpose(0, 1)
                common = self._window_edge_context(
                    weights[hop_mask],
                    direction,
                    source,
                    destination,
                    metadata[hop_mask],
                    out_degree,
                    in_degree,
                    time_feature,
                    input_flow.dtype,
                )
                if node_frequency_indices is None:
                    candidate = self.trend_mlp(
                        state[source], source_valid_windows, *common
                    )
                else:
                    candidate = self.trend_mlp(
                        state[source],
                        source_valid_windows,
                        *common,
                        node_frequency_indices[source],
                    )
                aggregated, new_valid = self._aggregate_window_spectral(
                    candidate, source_valid, destination, nodes
                )
                writable = new_valid & ~boundary
                state = torch.where(writable.unsqueeze(-1), aggregated, state)
                node_valid |= writable

        decoded_state = decode_spectral_window(
            state, self.window_size, node_frequency_indices
        ).transpose(0, 1)
        prediction_valid = node_valid[centers].unsqueeze(0).expand(
            self.window_size, -1
        )
        return (
            decoded_state[:, centers],
            prediction_valid,
            decoded_state,
            state[centers].unsqueeze(0),
        )

    def _propagate_window_residual(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
        frequency_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Propagate one complete residual window at every graph hop."""
        if input_flow.shape[0] != self.window_size:
            raise ValueError(
                "window imputation requires exactly window_size input points"
            )
        _, nodes = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        node_hop = graph["up_hop" if direction == 1 else "down_hop"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(node_hop, torch.Tensor)
        assert isinstance(centers, torch.Tensor)
        if self.max_propagation_hop is not None:
            boundary = boundary & (node_hop > 0) & (
                node_hop <= self.max_propagation_hop
            )

        value_valid = observed.bool() & boundary.unsqueeze(0)
        filled = interpolate_temporal_window(
            input_flow.transpose(0, 1), value_valid.transpose(0, 1)
        ).transpose(0, 1)
        if frequency_indices is None:
            boundary_trend = lowpass_window(
                input_flow,
                value_valid,
                self.window_size,
                self.trend_frequency_bins,
            )
        else:
            node_component = graph["node_component"]
            assert isinstance(node_component, torch.Tensor)
            boundary_trend = selected_trend_window(
                input_flow,
                value_valid,
                self.window_size,
                frequency_indices[0, node_component],
            )
        state = torch.where(
            boundary.unsqueeze(0),
            filled - boundary_trend,
            torch.zeros_like(input_flow),
        )
        node_valid = value_valid.any(dim=0)
        direction_graph = self._direction_graph(graph, direction, input_flow.dtype)
        if direction_graph is not None:
            source_all, destination_all, weights, hops, metadata, out_degree, in_degree = (
                direction_graph
            )
            for hop in range(int(hops.max()), 0, -1):
                hop_mask = hops == hop
                if not hop_mask.any():
                    continue
                source = source_all[hop_mask]
                destination = destination_all[hop_mask]
                source_valid = node_valid[source]
                if not source_valid.any():
                    continue
                source_valid_windows = value_valid[:, source].transpose(0, 1)
                common = self._window_edge_context(
                    weights[hop_mask],
                    direction,
                    source,
                    destination,
                    metadata[hop_mask],
                    out_degree,
                    in_degree,
                    time_feature,
                    input_flow.dtype,
                )
                candidate = self.residual_mlp(
                    state[:, source].transpose(0, 1),
                    source_valid_windows,
                    *common,
                )
                # The window head is recursively reused at every hop.  Keep a
                # finite normalized-flow range so one unusually large message
                # cannot be amplified by all remaining hops.  NaNs remain NaN
                # and are reported by the engine diagnostics.
                candidate = candidate.clamp(min=-20.0, max=20.0)
                aggregated, new_valid = self._aggregate_window_values(
                    candidate, source_valid, destination, nodes
                )
                writable = new_valid & ~boundary
                state = torch.where(writable.unsqueeze(0), aggregated, state)
                value_valid |= writable.unsqueeze(0)
                node_valid |= writable

        prediction_valid = node_valid[centers].unsqueeze(0).expand(
            self.window_size, -1
        )
        return state[:, centers], prediction_valid, state, None

    def _propagate(
        self,
        input_flow: torch.Tensor,
        observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        direction: int,
        residual: bool,
        frequency_indices: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if self.time_window_impute:
            if bool(graph.get("full_graph", 0)):
                raise ValueError(
                    "time_window_impute currently supports adaptive subgraphs only"
                )
            if residual:
                return self._propagate_window_residual(
                    input_flow,
                    observed,
                    graph,
                    time_feature,
                    direction,
                    frequency_indices,
                )
            return self._propagate_window_spectral_trend(
                input_flow,
                observed,
                graph,
                time_feature,
                direction,
                frequency_indices,
            )
        if bool(graph.get("full_graph", 0)):
            if self.use_graph_network:
                if residual:
                    return self._propagate_graph_network_residual(
                        input_flow, observed, graph, time_feature, direction
                    )
                return self._propagate_graph_network_spectral(
                    input_flow, observed, graph, time_feature, direction
                )
            if not residual:
                if self.use_fft_encoding:
                    return self._propagate_full_graph_spectral(
                        input_flow, observed, graph, time_feature, direction
                    )
                return self._propagate_full_graph_temporal_trend_windows(
                    input_flow, observed, graph, time_feature, direction
                )
            return self._propagate_full_graph_temporal(
                input_flow,
                observed,
                graph,
                time_feature,
                direction,
                True,
            )
        if not residual:
            if self.use_fft_encoding:
                return self._propagate_spectral_trend(
                    input_flow,
                    observed,
                    graph,
                    time_feature,
                    direction,
                    frequency_indices,
                )
            return self._propagate_temporal_trend_windows(
                input_flow, observed, graph, time_feature, direction
            )
        time_steps, nodes = input_flow.shape
        prefix = "upstream" if direction == 1 else "downstream"
        boundary = graph[f"{prefix}_boundary_mask"]
        node_hop = graph["up_hop" if direction == 1 else "down_hop"]
        centers = graph["center_indices"]
        assert isinstance(boundary, torch.Tensor)
        assert isinstance(node_hop, torch.Tensor)
        assert isinstance(centers, torch.Tensor)
        if self.max_propagation_hop is not None:
            boundary = boundary & (node_hop > 0) & (
                node_hop <= self.max_propagation_hop
            )
        direction_graph = self._direction_graph(graph, direction, input_flow.dtype)
        valid = observed.bool() & boundary.unsqueeze(0)
        if frequency_indices is None:
            boundary_trend = trend_current(
                input_flow,
                valid,
                self.window_size,
                self.trend_frequency_bins,
                self.use_fft_decomposition,
            )
        else:
            node_component = graph["node_component"]
            assert isinstance(node_component, torch.Tensor)
            boundary_trend = selected_trend_current(
                input_flow,
                valid,
                self.window_size,
                frequency_indices[:, node_component],
            )
        if residual:
            state = torch.where(valid, input_flow - boundary_trend, torch.zeros_like(input_flow))
            propagated_state = state
        else:
            raise AssertionError("trend propagation must use a domain-specific path")
        if direction_graph is None:
            zeros = input_flow.new_zeros(time_steps, len(centers))
            return zeros, zeros.bool(), propagated_state, None
        source_all, destination_all, weights, hops, metadata, out_degree, in_degree = (
            direction_graph
        )
        for hop in range(int(hops.max()), 0, -1):
            hop_mask = hops == hop
            if not hop_mask.any():
                continue
            source, destination = source_all[hop_mask], destination_all[hop_mask]
            valid_window = make_temporal_windows(
                valid[:, source].float(), self.window_size
            ) > 0
            source_valid = valid_window.any(dim=-1)
            if not source_valid.any():
                continue
            edge_count = source.numel()
            common = (
                weights[hop_mask].unsqueeze(0).expand(time_steps, edge_count),
                torch.full(
                    (time_steps, edge_count),
                    float(direction),
                    dtype=input_flow.dtype,
                    device=input_flow.device,
                ),
                out_degree[source].unsqueeze(0).expand(time_steps, edge_count),
                in_degree[destination].unsqueeze(0).expand(time_steps, edge_count),
                metadata[hop_mask].unsqueeze(0).expand(time_steps, edge_count, -1),
                time_feature.unsqueeze(1).expand(time_steps, edge_count, -1),
            )
            state_window = make_temporal_windows(state[:, source], self.window_size)
            candidate = self.residual_mlp(state_window, valid_window, *common)
            aggregated, new_valid = self._aggregate(
                candidate, source_valid, destination, nodes
            )
            writable = new_valid & ~boundary.unsqueeze(0)
            state = torch.where(writable, aggregated, state)
            valid |= new_valid
        propagated_state = state
        return propagated_state[:, centers], valid[:, centers], propagated_state, None

    def _forward_metadata_only(
        self,
        input_flow: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        use_residual: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict focal values without reading traffic or road connectivity."""
        centers = graph["center_indices"]
        type_feature = graph["type_feature"]
        direction_feature = graph["direction_feature"]
        assert isinstance(centers, torch.Tensor)
        assert isinstance(type_feature, torch.Tensor)
        assert isinstance(direction_feature, torch.Tensor)
        center_metadata = torch.cat(
            (type_feature[centers], direction_feature[centers]), dim=-1
        ).to(input_flow.dtype)
        time_steps, center_count = input_flow.shape[0], centers.numel()
        metadata_input = center_metadata.unsqueeze(0).expand(
            time_steps, center_count, -1
        )
        time_input = time_feature.unsqueeze(1).expand(time_steps, center_count, -1)
        trend = self.trend_mlp(metadata_input, time_input)
        prediction_valid = torch.ones_like(trend, dtype=torch.bool)
        if not use_residual:
            return trend, {
                "prediction_valid": prediction_valid,
                "trend_prediction": trend,
                "residual_prediction": torch.zeros_like(trend),
            }
        residual = self.residual_mlp(metadata_input, time_input, trend)
        prediction = F.relu(trend + residual)
        return prediction, {
            "prediction_valid": prediction_valid,
            "trend_prediction": trend,
            "residual_prediction": residual,
        }

    def forward(
        self,
        input_flow: torch.Tensor,
        input_observed: torch.Tensor,
        graph: dict[str, torch.Tensor | int],
        time_feature: torch.Tensor,
        *,
        use_residual: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if use_residual and not self.use_fft_decomposition:
            raise ValueError(
                "residual refinement requires FFT trend/residual decomposition"
            )
        if self.metadata_only:
            return self._forward_metadata_only(
                input_flow, graph, time_feature, use_residual
            )
        frequency_indices = self._select_subgraph_frequency_indices(
            input_flow, input_observed, graph
        )
        up_trend, up_valid, _, up_spectrum = self._propagate(
            input_flow,
            input_observed,
            graph,
            time_feature,
            1,
            False,
            frequency_indices,
        )
        down_trend, down_valid, _, down_spectrum = self._propagate(
            input_flow,
            input_observed,
            graph,
            time_feature,
            -1,
            False,
            frequency_indices,
        )
        trend, trend_valid = _fuse_branches(
            up_trend, up_valid, down_trend, down_valid
        )
        trend_information = {
            "prediction_valid": trend_valid,
            "trend_prediction": trend,
            "residual_prediction": torch.zeros_like(trend),
        }
        if self.use_fft_encoding:
            assert up_spectrum is not None
            assert down_spectrum is not None
            spectral_up_valid = up_valid[:1] if self.time_window_impute else up_valid
            spectral_down_valid = (
                down_valid[:1] if self.time_window_impute else down_valid
            )
            spectrum, spectrum_valid = _fuse_branches(
                up_spectrum,
                spectral_up_valid,
                down_spectrum,
                spectral_down_valid,
            )
            trend_information["spectral_prediction"] = spectrum
            trend_information["spectral_prediction_valid"] = spectrum_valid
            if frequency_indices is not None:
                trend_information["spectral_frequency_indices"] = frequency_indices
        if not use_residual:
            return trend, trend_information
        up_residual, up_residual_valid, _, _ = self._propagate(
            input_flow,
            input_observed,
            graph,
            time_feature,
            1,
            True,
            frequency_indices,
        )
        down_residual, down_residual_valid, _, _ = self._propagate(
            input_flow,
            input_observed,
            graph,
            time_feature,
            -1,
            True,
            frequency_indices,
        )
        residual, residual_valid = _fuse_branches(
            up_residual, up_residual_valid, down_residual, down_residual_valid
        )
        prediction = F.relu(trend + residual)
        trend_information.update(
            {
                "prediction_valid": trend_valid & residual_valid,
                "residual_prediction": residual,
            }
        )
        return prediction, trend_information
