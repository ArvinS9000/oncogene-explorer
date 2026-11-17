"""Project paths, configuration loading and logging setup."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("ONCOGENE_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


@dataclass(frozen=True)
class Paths:
    """All filesystem locations used by the pipeline, relative to the project root."""

    root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def metadata(self) -> Path:
        return self.root / "data" / "metadata"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def models(self) -> Path:
        return self.artifacts / "models"

    @property
    def metrics(self) -> Path:
        return self.artifacts / "metrics"

    @property
    def figures(self) -> Path:
        return self.artifacts / "figures"

    @property
    def reports(self) -> Path:
        return self.artifacts / "reports"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.metadata, self.models,
                  self.metrics, self.figures, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML experiment configuration."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {path} did not parse to a mapping")
    return cfg


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging to console (and optionally a file). Idempotent."""
    logger = logging.getLogger("oncogene")
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
