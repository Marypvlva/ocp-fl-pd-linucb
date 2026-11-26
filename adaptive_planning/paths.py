"""Shared paths for source-checkout experiments and their artifacts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
