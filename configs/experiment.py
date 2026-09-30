"""Editable model, optimization, loss, and evaluation configuration.

The model implementation in ``src/models`` contains no experiment defaults. Edit
``get_experiment_config`` to define reproducible experiments; command-line arguments
in ``experiments/main.py`` are intended only for temporary overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUPPORTED_LOSSES = {"smooth_l1", "mae", "mse"}
SUPPORTED_METRICS = {"mae", "mse", "rmse", "mape"}
SUPPORTED_OPTIMIZERS = {"adam", "adamw"}
SUPPORTED_SCHEDULERS = {"cosine", "none"}
SUPPORTED_TREND_LOSS_DOMAINS = {"time", "frequency", "hybrid"}


@dataclass(frozen=True)
class ModelConfig:
    """Parameters that define SPN and its temporal features."""

    window_size: int = 24
    trend_frequency_bins: int = 4
    hidden_dim: int = 64
    dropout: float = 0.3
    # The full model keeps the first K coefficients in the Fourier domain across
    # every propagation hop and decodes only after reaching the focal node.
    # Disabling this flag changes only the propagation representation to a
    # complete time-domain window; decomposition is controlled separately.
    use_fft_encoding: bool = True
    # Keep the FFT low-pass trend/residual split. Setting this to False defines
    # the strict w/o FFT ablation: propagate raw flow windows and disable the
    # residual stage because no frequency-derived residual target exists.
    use_fft_decomposition: bool = True
    # Ablation/variant: choose one shared set of K Fourier bins for every
    # query-centered subgraph from its visible boundary signals.  The default
    # model instead retains the fixed low-frequency bins 0, ..., K-1.
    use_subgraph_frequency_topk: bool = False
    # Optional window-level imputation experiment. Each non-overlapping window
    # is encoded once, propagated as K complex coefficients, decoded once at
    # the focal node, and supervised over every point in the window.
    time_window_impute: bool = False
    # Controlled w/o Residual Refinement ablation. When disabled, only the
    # primary spectral/temporal trend stage is trained and its
    # non-negative prediction is used for validation, testing, and completion.
    use_residual_refinement: bool = True
    # Static road/sensor metadata consists of source/destination Type and
    # Direction plus same-freeway and same-direction edge indicators. Disabling
    # it zeros these inputs without changing the graph or model dimensions.
    use_metadata: bool = True
    # Direct diagnostic baseline: predict from focal-node Type/Direction and,
    # unless separately disabled, TOD/DOW. It never reads traffic or topology.
    metadata_only: bool = False
    use_time_features: bool = True
    # Disable for the structural/model ablation: replace query-centered
    # subgraphs and recurrent edge MLP propagation with a weighted full-road
    # graph GNN whose trend node features remain in the Fourier domain.
    use_adaptive_subgraphs: bool = True
    gnn_layers: int = 2
    time_feature_dim: int = 4
    day_points: int = 288
    start_weekday: int = 6  # 2023-01-01, Monday=0
    branch_aggregation: str = "mean"
    max_propagation_hop: int | None = None

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.hidden_dim <= 0:
            raise ValueError("window_size and hidden_dim must be positive")
        if not 0 < self.trend_frequency_bins <= self.window_size // 2 + 1:
            raise ValueError("trend_frequency_bins is incompatible with window_size")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.time_feature_dim != 4:
            raise ValueError("the current absolute TOD/DOW encoder produces 4 features")
        if self.day_points <= 0:
            raise ValueError("day_points must be positive")
        if not 0 <= self.start_weekday <= 6:
            raise ValueError("start_weekday must be in [0, 6]")
        if self.branch_aggregation not in {"mean", "sum"}:
            raise ValueError("branch_aggregation must be 'mean' or 'sum'")
        if self.max_propagation_hop is not None and self.max_propagation_hop <= 0:
            raise ValueError("max_propagation_hop must be positive or None")
        if self.gnn_layers <= 0:
            raise ValueError("gnn_layers must be positive")
        if self.metadata_only and not self.use_metadata:
            raise ValueError("metadata_only requires use_metadata=True")
        if self.metadata_only and not self.use_adaptive_subgraphs:
            raise ValueError(
                "metadata_only is incompatible with the full-graph GNN ablation"
            )
        if self.use_fft_encoding and not self.use_fft_decomposition:
            raise ValueError("spectral propagation requires FFT decomposition")
        if not self.use_fft_decomposition and self.use_residual_refinement:
            raise ValueError(
                "residual refinement requires FFT trend/residual decomposition"
            )
        if self.use_subgraph_frequency_topk and not self.use_fft_encoding:
            raise ValueError(
                "subgraph frequency Top-K requires use_fft_encoding=True"
            )
        if self.use_subgraph_frequency_topk and not self.use_adaptive_subgraphs:
            raise ValueError(
                "subgraph frequency Top-K requires adaptive subgraphs"
            )
        if self.use_subgraph_frequency_topk and self.metadata_only:
            raise ValueError(
                "subgraph frequency Top-K is incompatible with metadata_only"
            )
        if self.time_window_impute and not self.use_fft_encoding:
            raise ValueError("time_window_impute requires use_fft_encoding=True")
        if self.time_window_impute and self.metadata_only:
            raise ValueError("time_window_impute is incompatible with metadata_only")
        if self.time_window_impute and not self.use_adaptive_subgraphs:
            raise ValueError(
                "time_window_impute currently requires adaptive subgraphs"
            )
        if (
            not self.use_adaptive_subgraphs
            and not self.metadata_only
            and not self.use_fft_encoding
        ):
            raise ValueError(
                "the full-graph GNN ablation requires spectral FFT features"
            )
        if self.time_window_impute and self.day_points % self.window_size != 0:
            raise ValueError(
                "time_window_impute requires window_size to divide day_points"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Sampling, optimizer, scheduler, and runtime training parameters."""

    seed: int = 42
    trend_epochs: int = 50
    residual_epochs: int = 50
    # Count validation evaluations, resetting to zero whenever the configured
    # checkpoint metric improves by more than early_stopping_min_delta. With
    # evaluate_every=1 this equals epochs.
    early_stopping_patience: int = 10
    # Minimum absolute checkpoint-metric change required to reset patience.
    # Validation metrics are evaluated in their reported (unscaled) units.
    early_stopping_min_delta: float = 0.0
    optimizer: str = "adam"
    scheduler: str = "cosine"
    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    optimizer_eps: float = 1e-8
    gradient_clip: float = 5.0
    train_days_per_epoch: int = 7
    center_batch_size: int = 7
    validation_days_per_weekday: int = 1
    evaluate_every: int = 1
    evaluation_chunk: int = 4096

    def __post_init__(self) -> None:
        positive = {
            "trend_epochs": self.trend_epochs,
            "residual_epochs": self.residual_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "learning_rate": self.learning_rate,
            "optimizer_eps": self.optimizer_eps,
            "gradient_clip": self.gradient_clip,
            "train_days_per_epoch": self.train_days_per_epoch,
            "center_batch_size": self.center_batch_size,
            "validation_days_per_weekday": self.validation_days_per_weekday,
            "evaluate_every": self.evaluate_every,
            "evaluation_chunk": self.evaluation_chunk,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError("training values must be positive: " + ", ".join(invalid))
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"unsupported optimizer: {self.optimizer}")
        if self.scheduler not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"unsupported scheduler: {self.scheduler}")


@dataclass(frozen=True)
class LossConfig:
    """Backpropagation objectives for the two training stages."""

    trend_loss: str = "smooth_l1"
    # ``frequency`` supervises the retained coefficients through log-amplitude
    # and circular phase losses. ``hybrid`` combines these with the decoded
    # time-domain trend loss.
    trend_loss_domain: str = "time"
    trend_time_weight: float = 1.0
    trend_amplitude_weight: float = 1.0
    trend_phase_weight: float = 1.0
    residual_loss: str = "smooth_l1"
    final_loss: str = "smooth_l1"
    final_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in (self.trend_loss, self.residual_loss, self.final_loss):
            if name not in SUPPORTED_LOSSES:
                raise ValueError(f"unsupported backpropagation loss: {name}")
        if self.trend_loss_domain not in SUPPORTED_TREND_LOSS_DOMAINS:
            raise ValueError(
                "trend_loss_domain must be 'time', 'frequency', or 'hybrid'"
            )
        weights = {
            "trend_time_weight": self.trend_time_weight,
            "trend_amplitude_weight": self.trend_amplitude_weight,
            "trend_phase_weight": self.trend_phase_weight,
        }
        negative = [name for name, value in weights.items() if value < 0]
        if negative:
            raise ValueError("loss weights cannot be negative: " + ", ".join(negative))
        if self.trend_loss_domain in {"time", "hybrid"} and not self.trend_time_weight:
            raise ValueError("time or hybrid trend loss requires trend_time_weight > 0")
        if (
            self.trend_loss_domain in {"frequency", "hybrid"}
            and self.trend_amplitude_weight + self.trend_phase_weight <= 0
        ):
            raise ValueError(
                "frequency or hybrid trend loss requires a positive amplitude "
                "or phase weight"
            )
        if self.final_loss_weight < 0:
            raise ValueError("final_loss_weight cannot be negative")


@dataclass(frozen=True)
class EvaluationConfig:
    """Metrics reported during evaluation and used for checkpoint selection."""

    metrics: tuple[str, ...] = ("mae", "rmse", "mape")
    checkpoint_metric: str = "mae"
    checkpoint_mode: str = "min"

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("at least one evaluation metric is required")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("evaluation metrics cannot contain duplicates")
        unsupported = set(self.metrics).difference(SUPPORTED_METRICS)
        if unsupported:
            raise ValueError("unsupported evaluation metrics: " + ", ".join(sorted(unsupported)))
        if self.checkpoint_metric not in self.metrics:
            raise ValueError("checkpoint_metric must be included in metrics")
        if self.checkpoint_mode not in {"min", "max"}:
            raise ValueError("checkpoint_mode must be 'min' or 'max'")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if self.loss.trend_loss_domain == "time":
            return
        if not self.model.use_fft_encoding:
            raise ValueError("frequency-domain loss requires use_fft_encoding=True")
        if self.model.metadata_only:
            raise ValueError("frequency-domain loss is incompatible with metadata_only")


def get_experiment_config(district: int) -> ExperimentConfig:
    """Return the editable experiment configuration for one District.

    All Districts currently reproduce the same final-model settings. Branch on
    ``district`` here when a model, loss, or evaluation protocol should differ.
    """
    if district not in {3, 4, 7, 8}:
        raise ValueError(f"no experiment config for District {district}")
    return ExperimentConfig(
        model=ModelConfig(
            window_size=24,
            trend_frequency_bins=4,
            hidden_dim=64,
            dropout=0.3,
            use_fft_encoding=True,
            use_fft_decomposition=True,
            use_subgraph_frequency_topk=False,
            time_window_impute=False,
            use_residual_refinement=True,
            use_metadata=True,
            metadata_only=False,
            use_time_features=True,
            use_adaptive_subgraphs=True,
            gnn_layers=2,
            day_points=288,
            start_weekday=6,
            branch_aggregation="mean",
            max_propagation_hop=None,
        ),
        training=TrainingConfig(
            seed=42,
            trend_epochs=50,
            residual_epochs=50,
            early_stopping_patience=10,
            early_stopping_min_delta=0.0,
            optimizer="adam",
            scheduler="cosine",
            learning_rate=0.003,
            weight_decay=1e-4,
            gradient_clip=5.0,
            train_days_per_epoch=7,
            center_batch_size=7,
            validation_days_per_weekday=1,
            evaluate_every=1,
            evaluation_chunk=4096,
        ),
        loss=LossConfig(
            trend_loss="smooth_l1",
            trend_loss_domain="time",
            trend_time_weight=1.0,
            trend_amplitude_weight=1.0,
            trend_phase_weight=1.0,
            residual_loss="smooth_l1",
            final_loss="smooth_l1",
            final_loss_weight=0.5,
        ),
        evaluation=EvaluationConfig(
            metrics=("mae", "rmse", "mape"),
            checkpoint_metric="mae",
            checkpoint_mode="min",
        ),
    )


def experiment_config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    """Rebuild nested dataclasses from a serialized checkpoint configuration."""
    return ExperimentConfig(
        model=ModelConfig(**values["model"]),
        training=TrainingConfig(**values["training"]),
        loss=LossConfig(**values["loss"]),
        evaluation=EvaluationConfig(
            **{
                **values["evaluation"],
                "metrics": tuple(values["evaluation"]["metrics"]),
            }
        ),
    )
