"""Build the standardized, validated dataset from the raw GEO downloads.

Only label-free, deterministic operations happen here (they cannot leak information
between CV folds). Every decision is written to ``artifacts/reports/preprocessing_report.json``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from oncogene.config import Paths, load_config
from oncogene.data import download as dl
from oncogene.data.load import (build_sample_metadata, parse_platform_table,
                                parse_series_matrix)
from oncogene.data.validate import run_all
from oncogene.preprocessing.transforms import log2_if_needed

log = logging.getLogger("oncogene.preprocessing")


def normalize_symbol(sym: Any) -> str | None:
    """Upper-case, strip whitespace; return None for empty / placeholder symbols."""
    if not isinstance(sym, str):
        return None
    s = sym.strip().upper()
    return s or None


def to_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce to float; non-numeric cells become NaN (and are reported by validation)."""
    return X.apply(pd.to_numeric, errors="coerce").astype(float)


def summarize(X: pd.DataFrame) -> dict[str, Any]:
    v = X.values
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "missing_values": int(np.isnan(v).sum()),
        "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
        "mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
    }


def build_dataset(config: dict[str, Any] | None = None, paths: Paths | None = None) -> dict[str, Any]:
    """Parse raw files, validate, apply documented label-free filters, write processed data."""
    cfg = config or load_config()
    paths = (paths or Paths()).ensure()
    pp = cfg["preprocessing"]

    sm_path = paths.raw / dl.SERIES_MATRIX_FILE
    gpl_path = paths.raw / dl.PLATFORM_TABLE_FILE
    for p in (sm_path, gpl_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run `oncogene download` first")

    log.info("Parsing series matrix %s", sm_path.name)
    sm = parse_series_matrix(sm_path)
    meta = build_sample_metadata(sm.sample_meta_raw)
    X = to_numeric_matrix(sm.expression.T)       # samples x probes
    X.index.name = "sample_id"
    X = X.loc[meta.index]
    log.info("Raw matrix: %d samples x %d probes", *X.shape)

    log.info("Parsing platform annotation %s", gpl_path.name)
    ann = parse_platform_table(gpl_path)
    ann["gene_symbol"] = ann["gene_symbol"].map(normalize_symbol)

    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accession": cfg["dataset"]["accession"],
        "series_title": (sm.series_meta.get("title") or [""])[0],
        "data_processing": sorted(set(meta["data_processing"].dropna())),
        "raw_summary": summarize(X),
        "filters": [],
    }

    checks = run_all(X, meta, cfg)
    report["validation"] = [c.to_dict() for c in checks]
    failures = [c for c in checks if c.status == "fail"]
    if failures:
        (paths.reports / "validation_report.json").write_text(json.dumps(report["validation"], indent=2))
        raise ValueError("Validation failed: " + "; ".join(f"{c.name}: {c.detail}" for c in failures))

    # --- scale: RMA output is already log2. Apply log2 only if the data are linear-scale.
    X, transformed = log2_if_needed(X, pp["log_scale_max_threshold"])
    report["log2_transform_applied"] = transformed
    log.info("log2 transform applied: %s (RMA values are already log2)", transformed)

    # --- missing values: none expected for RMA; if present they are imputed in-pipeline, not here.
    report["missing_value_policy"] = (
        "No global imputation. Probes with any missing value would be retained and imputed "
        "inside model pipelines; GSE109169 contains none.")

    # --- annotation alignment
    unannotated = X.columns.difference(ann.index)
    report["filters"].append({"step": "probes_absent_from_platform_table",
                              "removed": int(len(unannotated)),
                              "rationale": "cannot be categorised or mapped to genes"})
    X = X.drop(columns=unannotated)
    ann = ann.reindex(X.columns)

    # --- design category (label-free): keep 'main' transcript clusters, drop controls
    keep = set(pp["keep_categories"])
    cat_counts = ann["category"].value_counts(dropna=False).to_dict()
    drop_cat = ann.index[~ann["category"].isin(keep)]
    report["filters"].append({"step": "design_category", "kept_categories": sorted(keep),
                              "category_counts": {str(k): int(v) for k, v in cat_counts.items()},
                              "removed": int(len(drop_cat)),
                              "rationale": "control / background probesets do not measure genes"})
    X = X.drop(columns=drop_cat)

    # --- exactly-constant probes (label-free, deterministic)
    if pp.get("drop_exactly_constant", True):
        sd = X.std(axis=0, ddof=1)
        const = sd.index[sd == 0]
        report["filters"].append({"step": "exactly_constant", "removed": int(len(const)),
                                  "rationale": "zero variance carries no information"})
        X = X.drop(columns=const)

    near = int((X.std(axis=0, ddof=1) < pp["near_constant_sd"]).sum())
    report["near_constant_retained"] = {
        "threshold_sd": pp["near_constant_sd"], "count": near,
        "note": "Retained globally; variance filtering is learned inside each training fold."}

    ann = ann.loc[X.columns]
    n_sym = ann["gene_symbol"].notna().sum()
    dup_sym = ann["gene_symbol"].dropna().duplicated(keep=False)
    report["annotation"] = {
        "probes_with_symbol": int(n_sym),
        "probes_without_symbol": int(len(ann) - n_sym),
        "unique_symbols": int(ann["gene_symbol"].nunique()),
        "probes_sharing_a_symbol": int(dup_sym.sum()),
        "multi_gene_probes": int((ann["n_gene_symbols"] > 1).sum()),
        "policy": "Features are kept at probe (transcript-cluster) level; the first gene "
                  "assignment is used as the display symbol. Gene-level collapsing is only "
                  "done for pathway analysis.",
    }
    report["final_summary"] = summarize(X)
    log.info("Final matrix: %d samples x %d probes (%d unique symbols)",
             X.shape[0], X.shape[1], report["annotation"]["unique_symbols"])

    # --- write outputs
    X.to_parquet(paths.processed / "expression_matrix.parquet")
    meta.drop(columns=["data_processing"]).to_csv(paths.metadata / "samples.csv")
    ann.to_csv(paths.metadata / "probe_annotation.csv")
    (paths.metadata / "series_metadata.json").write_text(json.dumps(sm.series_meta, indent=2))
    sm.sample_meta_raw.to_csv(paths.metadata / "samples_raw_geo_fields.csv", index=False)
    (paths.reports / "validation_report.json").write_text(json.dumps(report["validation"], indent=2))
    (paths.reports / "preprocessing_report.json").write_text(json.dumps(report, indent=2))
    log.info("Wrote processed data to %s and %s", paths.processed, paths.metadata)
    return report
