"""Configurable backpropagation losses and streaming evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_loss(name: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate one configured differentiable loss."""
    if name == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    if name == "mae":
        return F.l1_loss(prediction, target)
    if name == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"unsupported backpropagation loss: {name}")


class MetricAccumulator:
    """Accumulate exact dataset-level metrics without retaining predictions."""

    def __init__(self, metric_names: tuple[str, ...]) -> None:
        self.metric_names = metric_names
        self.absolute = 0.0
        self.squared = 0.0
        self.percentage = 0.0
        self.percentage_count = 0
        self.count = 0

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> None:
        if not valid.any():
            return
        prediction, target = prediction[valid], target[valid]
        error = prediction - target
        absolute = error.abs()
        self.absolute += float(absolute.sum().item())
        self.squared += float(error.square().sum().item())
        if "mape" in self.metric_names:
            percentage_valid = torch.isfinite(target) & (target > 0)
            if percentage_valid.any():
                self.percentage += float(
                    (absolute[percentage_valid] / target[percentage_valid]).sum().item()
                )
                self.percentage_count += int(percentage_valid.sum().item())
        self.count += int(valid.sum().item())

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            return {**{name: float("nan") for name in self.metric_names}, "count": 0}
        values: dict[str, float | int] = {"count": self.count}
        if "mae" in self.metric_names:
            values["mae"] = self.absolute / self.count
        if "mse" in self.metric_names:
            values["mse"] = self.squared / self.count
        if "rmse" in self.metric_names:
            values["rmse"] = float(np.sqrt(self.squared / self.count))
        if "mape" in self.metric_names:
            values["mape"] = (
                self.percentage / self.percentage_count * 100.0
                if self.percentage_count
                else float("nan")
            )
        return values


def format_metrics(metrics: dict[str, float | int]) -> str:
    """Format configured metrics consistently for epoch logs."""
    parts = []
    for name, value in metrics.items():
        if name == "count":
            continue
        suffix = "%" if name == "mape" else ""
        parts.append(f"{name.upper()}={float(value):.4f}{suffix}")
    return " | ".join(parts)
