"""Metadata-learning variants of every legacy kriging architecture."""

from .metadata_arch import (
    METADATA_ARCHITECTURE_VERSION,
    METADATA_ARCHITECTURES,
    MetadataDualSTN,
    MetadataGCN,
    MetadataGIN,
    MetadataGRIN,
    MetadataIAGCN,
    MetadataIGNNK,
    MetadataINCREASE,
    MetadataOkriging,
    MetadataSATCN,
    MetadataSTAGANN,
    TrafficMetadataFusion,
)

__all__ = [
    "METADATA_ARCHITECTURE_VERSION",
    "TrafficMetadataFusion",
    "MetadataDualSTN",
    "MetadataGCN",
    "MetadataGIN",
    "MetadataGRIN",
    "MetadataIAGCN",
    "MetadataIGNNK",
    "MetadataINCREASE",
    "MetadataOkriging",
    "MetadataSATCN",
    "MetadataSTAGANN",
    "METADATA_ARCHITECTURES",
]
