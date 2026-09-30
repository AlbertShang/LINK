"""Controlled pure-PEMS baseline configurations without metadata information."""

from __future__ import annotations

from typing import Any

from easydict import EasyDict

from examples.pems03_metadata_common import build_metadata_config


SUPPORTED_DISTRICTS = (3, 4, 7, 8)


def build_no_metadata_config(
    *,
    district: int,
    top_k: int = 3,
    model_name: str,
    architecture: type,
    model_parameters: dict[str, Any],
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    workers: int,
    min_delta: float,
    losses: dict[str, Any] | None = None,
    optimizer_eps: float | None = 1.0e-8,
    early_stopping_patience: int = 10,
    batch_select_node_random: bool | None = True,
    batch_matrix_random: bool | None = False,
    mask_matrix: bool = False,
    batch_random: bool | None = None,
    plot_imputation_epoch: int | None = None,
    split_adjacency: bool = False,
    matrix_transform: str = "original",
    metadata_embedding_dim: int = 20,
    metadata_fusion_hidden_dim: int = 32,
) -> EasyDict:
    """Build the exact metadata architecture with its metadata tensor zeroed.

    Both sides of the ablation therefore share the same backbone, metadata
    fusion modules, parameter count, initialization scheme, optimizer, loader,
    early stopping, and dataset split. Only the information in the static
    metadata channels differs.
    """
    if district not in SUPPORTED_DISTRICTS:
        raise ValueError(
            f"Unsupported PEMS district {district}; expected one of "
            f"{SUPPORTED_DISTRICTS}"
        )
    return build_metadata_config(
        district=district,
        top_k=top_k,
        model_name=model_name,
        architecture=architecture,
        model_parameters=model_parameters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        min_delta=min_delta,
        losses=losses,
        optimizer_eps=optimizer_eps,
        early_stopping_patience=early_stopping_patience,
        batch_select_node_random=batch_select_node_random,
        batch_matrix_random=batch_matrix_random,
        mask_matrix=mask_matrix,
        batch_random=batch_random,
        plot_imputation_epoch=plot_imputation_epoch,
        split_adjacency=split_adjacency,
        matrix_transform=matrix_transform,
        metadata_embedding_dim=metadata_embedding_dim,
        metadata_fusion_hidden_dim=metadata_fusion_hidden_dim,
        zero_metadata=True,
    )
