"""Parsers for GEO series-matrix / platform files and loaders for processed data."""

from __future__ import annotations

import gzip
import io
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oncogene.config import Paths

log = logging.getLogger("oncogene.data.load")

TUMOR, NORMAL = 1, 0
LABEL_NAMES = {NORMAL: "Adjacent normal", TUMOR: "Tumor"}

# Exact characteristic values used by the submitters (verified against GSE109169).
_TISSUE_MAP = {
    "breast cancer tissue": TUMOR,
    "breast tumor adjacent normal tissue": NORMAL,
}


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _unquote(s: str) -> str:
    s = s.strip()
    return s[1:-1] if len(s) >= 2 and s[0] == s[-1] == '"' else s


@dataclass
class SeriesMatrix:
    expression: pd.DataFrame          # probes x samples (as distributed by GEO)
    sample_meta_raw: pd.DataFrame     # every !Sample_* field, one row per sample
    series_meta: dict[str, list[str]]  # every !Series_* field


def parse_series_matrix(path: Path) -> SeriesMatrix:
    """Parse a GEO ``*_series_matrix.txt(.gz)`` file without external GEO libraries."""
    series: dict[str, list[str]] = defaultdict(list)
    sample_fields: dict[str, list[list[str]]] = defaultdict(list)
    table_lines: list[str] = []
    in_table = False
    with _open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                table_lines.append(line)
            elif line.startswith("!Series_"):
                key, *vals = line.split("\t")
                series[key[len("!Series_"):]].extend(_unquote(v) for v in vals)
            elif line.startswith("!Sample_"):
                key, *vals = line.split("\t")
                sample_fields[key[len("!Sample_"):]].append([_unquote(v) for v in vals])

    if not table_lines:
        raise ValueError(f"No expression table found in {path}")

    expr = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", index_col=0, quotechar='"')
    expr.index = expr.index.astype(str)
    expr.index.name = "probe_id"

    meta: dict[str, list[str]] = {}
    for key, rows in sample_fields.items():
        if key.startswith("characteristics"):
            # Characteristics rows are "name: value"; split them into named columns.
            for row in rows:
                for i, cell in enumerate(row):
                    if ":" in cell:
                        name, value = cell.split(":", 1)
                        col = "char_" + re.sub(r"\W+", "_", name.strip().lower()).strip("_")
                        meta.setdefault(col, [""] * len(row))[i] = value.strip()
        else:
            for j, row in enumerate(rows):
                meta[key if j == 0 else f"{key}_{j}"] = row
    sample_meta = pd.DataFrame(meta)
    return SeriesMatrix(expression=expr, sample_meta_raw=sample_meta, series_meta=dict(series))


def build_sample_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive tidy sample metadata (labels, patient IDs) from raw GEO fields.

    Labels come from the ``tissue`` characteristic; the sample title (e.g. ``2102-Tumor``) is
    used as an independent cross-check. Any disagreement raises instead of being guessed.
    """
    df = pd.DataFrame({
        "sample_id": raw["geo_accession"],
        "title": raw["title"],
        "tissue_raw": raw["char_tissue"],
        "patient_id": raw["char_patient_identifier"].astype(str),
        "age_years": pd.to_numeric(raw["char_age"].str.extract(r"(\d+)")[0], errors="coerce"),
        "cancer_stage": raw.get("char_cancer_stage"),
        "sex": raw.get("char_sex"),
        "platform_id": raw.get("platform_id"),
        "data_processing": raw.get("data_processing"),
    })
    unknown = set(df["tissue_raw"]) - set(_TISSUE_MAP)
    if unknown:
        raise ValueError(f"Unrecognised tissue annotations: {sorted(unknown)}")
    df["label"] = df["tissue_raw"].map(_TISSUE_MAP).astype(int)
    df["tissue_type"] = df["label"].map({TUMOR: "Tumor", NORMAL: "Adjacent normal"})

    # Cross-check against titles of the form "<patient>-<Tumor|Normal>".
    m = df["title"].str.extract(r"^(?P<pid>[^-]+)-(?P<kind>Tumor|Normal)$")
    title_label = m["kind"].map({"Tumor": TUMOR, "Normal": NORMAL})
    if title_label.isna().any():
        log.warning("Some titles do not follow '<patient>-<Tumor|Normal>'; cross-check partial")
    bad_label = title_label.notna() & (title_label != df["label"])
    bad_pid = m["pid"].notna() & (m["pid"] != df["patient_id"])
    if bad_label.any() or bad_pid.any():
        raise ValueError(
            "Title / characteristic disagreement for samples: "
            f"{df.loc[bad_label | bad_pid, 'sample_id'].tolist()}")
    return df.set_index("sample_id")


def parse_platform_table(path: Path) -> pd.DataFrame:
    """Parse the GPL5175 platform table and extract the primary gene assignment per probe.

    ``gene_assignment`` looks like ``acc // SYMBOL // description // cytoband // entrez /// ...``.
    The first assignment is taken as primary; the number of distinct symbols is retained so
    that multi-gene transcript clusters can be flagged.
    """
    with _open_text(path) as fh:
        text = fh.read()
    start = text.find("!platform_table_begin")
    if start < 0:
        raise ValueError(f"No platform table in {path}")
    body = text[text.index("\n", start) + 1:]
    end = body.find("!platform_table_end")
    if end >= 0:
        body = body[:end]
    usecols = ["ID", "seqname", "RANGE_STRAND", "RANGE_START", "RANGE_STOP",
               "total_probes", "gene_assignment", "category"]
    tab = pd.read_csv(io.StringIO(body), sep="\t", usecols=usecols, dtype=str, low_memory=False)

    def first_assignment(s: str) -> tuple[str | None, str | None, str | None, str | None, int]:
        if not isinstance(s, str) or s.strip() in ("", "---"):
            return None, None, None, None, 0
        entries = [e.split(" // ") for e in s.split(" /// ")]
        symbols = {e[1].strip() for e in entries if len(e) > 1 and e[1].strip() != "---"}
        e = entries[0] + [None] * 5
        clean = [x.strip() if isinstance(x, str) and x.strip() != "---" else None for x in e[:5]]
        return clean[1], clean[2], clean[3], clean[4], len(symbols)

    parsed = tab["gene_assignment"].map(first_assignment)
    ann = pd.DataFrame(parsed.tolist(), columns=["gene_symbol", "gene_description", "cytoband",
                                                 "entrez_id", "n_gene_symbols"])
    ann.insert(0, "probe_id", tab["ID"].astype(str))
    ann["chromosome"] = tab["seqname"]
    ann["strand"] = tab["RANGE_STRAND"]
    ann["start"] = pd.to_numeric(tab["RANGE_START"], errors="coerce")
    ann["stop"] = pd.to_numeric(tab["RANGE_STOP"], errors="coerce")
    ann["total_probes"] = pd.to_numeric(tab["total_probes"], errors="coerce")
    ann["category"] = tab["category"]
    return ann.set_index("probe_id")


# --------------------------------------------------------------------------- processed data

@dataclass
class Dataset:
    X: pd.DataFrame            # samples x probes, log2 RMA
    meta: pd.DataFrame         # indexed by sample_id, aligned with X
    annotation: pd.DataFrame   # indexed by probe_id, aligned with X.columns

    @property
    def y(self) -> pd.Series:
        return self.meta["label"]

    @property
    def groups(self) -> pd.Series:
        return self.meta["patient_id"]


def load_processed(paths: Paths | None = None) -> Dataset:
    """Load the standardized feature matrix written by the preprocessing stage."""
    paths = paths or Paths()
    xf = paths.processed / "expression_matrix.parquet"
    mf = paths.metadata / "samples.csv"
    af = paths.metadata / "probe_annotation.csv"
    missing = [str(p) for p in (xf, mf, af) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed data missing; run `oncogene preprocess` first. Missing: " + ", ".join(missing))
    X = pd.read_parquet(xf)
    meta = pd.read_csv(mf, index_col="sample_id", dtype={"patient_id": str})
    ann = pd.read_csv(af, index_col="probe_id", dtype={"probe_id": str, "entrez_id": str})
    ann.index = ann.index.astype(str)
    meta = meta.loc[X.index]
    ann = ann.reindex(X.columns)
    return Dataset(X=X, meta=meta, annotation=ann)
