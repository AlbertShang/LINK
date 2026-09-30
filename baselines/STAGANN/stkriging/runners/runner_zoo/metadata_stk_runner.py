"""Runner used only by metadata baselines for metric-unit compatibility."""

import torch

from ...metrics import masked_mape
from .simple_stk_runner import SimpleSpatiotemporalKrigingRunner


def masked_mape_percent(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    null_val: float = 0.0,
) -> torch.Tensor:
    """Return MAPE in percent, matching ``src.metrics.MetricAccumulator``."""
    return masked_mape(predictions, targets, null_val=null_val) * 100.0


class PercentMAPESpatiotemporalKrigingRunner(SimpleSpatiotemporalKrigingRunner):
    """Keep legacy training behavior and report MAPE with a percent unit."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.metrics = dict(self.metrics)
        self.metrics["MAPE"] = masked_mape_percent


class MetadataSpatiotemporalKrigingRunner(
    PercentMAPESpatiotemporalKrigingRunner
):
    """Backward-compatible name used by metadata-aware configurations."""
