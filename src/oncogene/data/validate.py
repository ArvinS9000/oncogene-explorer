"""Dataset validation checks.

Checks never modify data. Each returns a :class:`Check` with status ``pass``, ``warn`` or
``fail``; the preprocessing stage aborts on any ``fail`` and records the full report.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("oncogene.data.validate")


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail
    detail: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(d["value"], (np.integer, np.floating)):
            d["value"] = d["value"].item()
        return d


def check_sample_counts(meta: pd.DataFrame, expected_samples: int,
                        expected_patients: int) -> list[Check]:
    n, p = len(meta), meta["patient_id"].nunique()
    return [
        Check("sample_count", "pass" if n == expected_samples else "fail",
              f"{n} samples (expected {expected_samples})", n),
        Check("patient_count", "pass" if p == expected_patients else "fail",
              f"{p} unique patients (expected {expected_patients})", p),
    ]


def check_duplicate_ids(meta: pd.DataFrame, X: pd.DataFrame) -> list[Check]:
    ds = int(meta.index.duplicated().sum())
    dp = int(X.columns.duplicated().sum())  # X is samples x probes
    return [
        Check("duplicate_sample_ids", "pass" if ds == 0 else "fail", f"{ds} duplicated sample IDs", ds),
        Check("duplicate_probe_ids", "pass" if dp == 0 else "fail", f"{dp} duplicated probe IDs", dp),
    ]


def check_pairs(meta: pd.DataFrame) -> list[Check]:
    """Every patient must contribute exactly one tumor and one adjacent-normal sample."""
    missing_pid = int(meta["patient_id"].isna().sum() + (meta["patient_id"].astype(str) == "").sum())
    counts = meta.groupby("patient_id")["label"].agg(["count", "sum"])
    bad = counts[(counts["count"] != 2) | (counts["sum"] != 1)]
    return [
        Check("missing_patient_ids", "pass" if missing_pid == 0 else "fail",
              f"{missing_pid} samples without a patient identifier", missing_pid),
        Check("matched_pairs", "pass" if bad.empty else "fail",
              "all patients have exactly one tumor + one adjacent-normal sample" if bad.empty
              else f"patients with incomplete/invalid pairs: {bad.index.tolist()}",
              int(len(counts) - len(bad))),
    ]


def check_labels(meta: pd.DataFrame) -> list[Check]:
    ok_values = set(meta["label"].unique()) <= {0, 1}
    counts = meta["label"].value_counts().to_dict()
    balanced = counts.get(0, 0) == counts.get(1, 0)
    return [
        Check("label_values", "pass" if ok_values else "fail",
              f"label values {sorted(meta['label'].unique().tolist())}"),
        Check("label_balance", "pass" if balanced else "warn",
              f"tumor={counts.get(1, 0)}, adjacent normal={counts.get(0, 0)}"),
    ]


def check_expression_values(X: pd.DataFrame, near_constant_sd: float,
                            log_scale_max: float) -> list[Check]:
    """X is samples x probes."""
    numeric = X.apply(pd.to_numeric, errors="coerce")
    non_numeric = int((numeric.isna() & X.notna()).sum().sum())
    missing = int(X.isna().sum().sum())
    sd = numeric.std(axis=0, ddof=1)
    n_const = int((sd == 0).sum())
    n_near = int((sd < near_constant_sd).sum())
    vmin, vmax = float(np.nanmin(numeric.values)), float(np.nanmax(numeric.values))
    log_like = vmax < log_scale_max and vmin >= 0
    med = numeric.median(axis=1)
    iqr = numeric.quantile(0.75, axis=1) - numeric.quantile(0.25, axis=1)
    med_spread = float(med.max() - med.min())
    return [
        Check("non_numeric_values", "pass" if non_numeric == 0 else "fail",
              f"{non_numeric} non-numeric entries", non_numeric),
        Check("missing_values", "pass" if missing == 0 else "warn",
              f"{missing} missing expression values", missing),
        Check("constant_features", "pass" if n_const == 0 else "warn",
              f"{n_const} probes with zero variance", n_const),
        Check("near_constant_features", "pass" if n_near == 0 else "warn",
              f"{n_near} probes with SD < {near_constant_sd} (reported, not removed)", n_near),
        Check("expression_scale", "pass" if log_like else "warn",
              f"range [{vmin:.2f}, {vmax:.2f}] -> "
              + ("consistent with log2 RMA; no further log transform" if log_like
                 else "may not be log-transformed; inspect before modeling"),
              {"min": vmin, "max": vmax, "log2_like": log_like}),
        Check("per_sample_scale_consistency", "pass" if med_spread < 0.5 else "warn",
              f"per-sample medians span {med_spread:.3f} log2 units; "
              f"IQR range {iqr.min():.2f}-{iqr.max():.2f}",
              med_spread),
    ]


def check_platform(meta: pd.DataFrame, expected_platform: str) -> list[Check]:
    plats = set(meta["platform_id"].dropna().unique())
    return [Check("platform", "pass" if plats == {expected_platform} else "fail",
                  f"platform(s): {sorted(plats)}")]


def run_all(X: pd.DataFrame, meta: pd.DataFrame, cfg: dict[str, Any]) -> list[Check]:
    """Run all validation checks. ``X`` is samples x probes, aligned with ``meta``."""
    ds, pp = cfg["dataset"], cfg["preprocessing"]
    checks: list[Check] = []
    checks += check_sample_counts(meta, ds["expected_samples"], ds["expected_patients"])
    checks += check_duplicate_ids(meta, X)
    checks += check_pairs(meta)
    checks += check_labels(meta)
    checks += check_platform(meta, ds["platform"])
    checks += check_expression_values(X, pp["near_constant_sd"], pp["log_scale_max_threshold"])
    if list(X.index) != list(meta.index):
        checks.append(Check("matrix_metadata_alignment", "fail",
                            "expression matrix rows are not aligned with metadata"))
    else:
        checks.append(Check("matrix_metadata_alignment", "pass", "rows aligned with metadata"))
    for c in checks:
        getattr(log, {"pass": "info", "warn": "warning", "fail": "error"}[c.status])(
            "[%s] %s: %s", c.status.upper(), c.name, c.detail)
    return checks
