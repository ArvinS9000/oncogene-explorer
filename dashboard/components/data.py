"""Cached, read-only loaders for pipeline artifacts.

The dashboard never trains models and never unpickles model files: it only reads CSV, JSON
and Parquet artifacts produced by the validated pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

METRICS = ROOT / "artifacts" / "metrics"
REPORTS = ROOT / "artifacts" / "reports"
MODELS = ROOT / "artifacts" / "models"
PROCESSED = ROOT / "data" / "processed"
META = ROOT / "data" / "metadata"


class MissingArtifact(FileNotFoundError):
    pass


def _need(path: Path) -> Path:
    if not path.exists():
        try:
            shown = path.relative_to(ROOT).as_posix()
        except ValueError:
            shown = str(path)
        raise MissingArtifact(shown)
    return path


@st.cache_data(show_spinner="Loading expression matrix…")
def expression() -> pd.DataFrame:
    return pd.read_parquet(_need(PROCESSED / "expression_matrix.parquet"))


@st.cache_data
def samples() -> pd.DataFrame:
    return pd.read_csv(_need(META / "samples.csv"), index_col="sample_id", dtype={"patient_id": str})


@st.cache_data
def annotation() -> pd.DataFrame:
    a = pd.read_csv(_need(META / "probe_annotation.csv"), index_col="probe_id",
                    dtype={"probe_id": str, "entrez_id": str})
    a.index = a.index.astype(str)
    return a


@st.cache_data
def metric_csv(name: str, index_col: str | int | None = None, **kw) -> pd.DataFrame:
    dtype = {"probe_id": str, "patient_id": str, **kw.pop("dtype", {})}
    df = pd.read_csv(_need(METRICS / name), index_col=index_col, dtype=dtype, **kw)
    if df.index.name == "probe_id":
        df.index = df.index.astype(str)
    return df


@st.cache_data
def metric_json(name: str):
    return json.loads(_need(METRICS / name).read_text())


@st.cache_data
def report_json(name: str):
    return json.loads(_need(REPORTS / name).read_text())


@st.cache_data
def report_text(name: str) -> str:
    return _need(REPORTS / name).read_text(encoding="utf-8")


@st.cache_data
def model_cards() -> dict:
    cards = {}
    for p in sorted(MODELS.glob("*.model_card.json")):
        cards[p.name.replace(".model_card.json", "")] = json.loads(p.read_text())
    return cards


def probe_label_map() -> pd.Series:
    ann = annotation()
    return ann["gene_symbol"].fillna(pd.Series(ann.index, index=ann.index))


@st.cache_data
def gene_search_options() -> pd.DataFrame:
    """Searchable 'SYMBOL - probe' list, ordered by DE significance when available."""
    ann = annotation()
    try:
        order = metric_csv("de_results.csv", index_col="probe_id").index
        ann = ann.reindex(order.intersection(ann.index).append(ann.index.difference(order)))
    except MissingArtifact:
        pass
    lab = ann["gene_symbol"].fillna("(no symbol)") + "  ·  probe " + ann.index
    return pd.DataFrame({"label": lab.values, "probe_id": ann.index}, index=ann.index)
