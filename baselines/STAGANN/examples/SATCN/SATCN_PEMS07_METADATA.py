"""SATCN metadata configuration for pure PEMS07."""

import sys
from pathlib import Path

STAGANN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGANN_DIR))

from examples.pems_metadata_specs import build_registered_metadata_config


CFG = build_registered_metadata_config("SATCN", 7)

