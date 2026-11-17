"""Exploratory over-representation analysis (ORA).

* Gene sets: Reactome 2022 and GO Biological Process 2023, as distributed by Enrichr
  (downloaded once, checksummed in provenance.json). KEGG is deliberately not used by
  default because of its licensing terms.
* Background universe: every gene symbol measured on the array after preprocessing, NOT the
  whole genome. Gene sets are intersected with this universe before testing.
* Test: one-sided hypergeometric test, run separately for up- and down-regulated genes,
  Benjamini-Hochberg corrected within each library x direction.
* Probe-to-gene: a gene is "hit" if any of its probes passes the DE thresholds; probes with
  conflicting directions for the same gene are excluded from both lists and reported.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from oncogene.analysis.differential_expression import benjamini_hochberg

log = logging.getLogger("oncogene.analysis.pathway")


def read_gmt(path: Path) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genes = {g.split(",")[0].strip().upper() for g in parts[2:] if g.strip()}
            sets[parts[0].strip()] = genes
    if not sets:
        raise ValueError(f"No gene sets parsed from {path}")
    return sets


def gene_hit_lists(de: pd.DataFrame) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return (universe, up_genes, down_genes, conflicting_genes) from a probe-level DE table."""
    sym = de["gene_symbol"].dropna()
    universe = set(sym.unique())
    sig = de[de["significant"] & de["gene_symbol"].notna()]
    up = set(sig.loc[sig["log2_fold_change"] > 0, "gene_symbol"])
    down = set(sig.loc[sig["log2_fold_change"] < 0, "gene_symbol"])
    conflict = up & down
    return universe, up - conflict, down - conflict, conflict


def ora(hits: set[str], universe: set[str], gene_sets: dict[str, set[str]],
        min_size: int = 10, max_size: int = 500) -> pd.DataFrame:
    """Hypergeometric over-representation of ``hits`` among ``gene_sets`` within ``universe``."""
    hits = hits & universe
    N, n = len(universe), len(hits)
    rows = []
    for name, genes in gene_sets.items():
        gs = genes & universe
        K = len(gs)
        if K < min_size or K > max_size:
            continue
        overlap = hits & gs
        k = len(overlap)
        p = stats.hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0
        rows.append({"term": name, "set_size_in_universe": K, "overlap": k,
                     "expected": K * n / N if N else np.nan,
                     "fold_enrichment": (k / n) / (K / N) if n and K else np.nan,
                     "gene_ratio": k / n if n else np.nan,
                     "p_value": p, "genes": ";".join(sorted(overlap))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_adj"] = benjamini_hochberg(df["p_value"].values)
    return df.sort_values(["p_adj", "p_value"]).reset_index(drop=True)


def _split_term(term: str) -> tuple[str, str | None]:
    """Separate the identifier Enrichr appends to term names (R-HSA-..., GO:...)."""
    import re
    m = re.search(r"\s(R-HSA-\d+)$", term)
    if m:
        return term[:m.start()].strip(), m.group(1)
    m = re.search(r"\((GO:\d+)\)$", term)
    if m:
        return term[:m.start()].strip(), m.group(1)
    return term, None


def term_url(term_id: str | None) -> str | None:
    if not term_id:
        return None
    if term_id.startswith("R-HSA-"):
        return f"https://reactome.org/content/detail/{term_id}"
    if term_id.startswith("GO:"):
        return f"https://amigo.geneontology.org/amigo/term/{term_id}"
    return None


def run_pathway_analysis(de: pd.DataFrame, gmt_paths: dict[str, Path], min_size: int = 10,
                         max_size: int = 500) -> tuple[pd.DataFrame, dict]:
    universe, up, down, conflict = gene_hit_lists(de)
    summary = {"universe_size": len(universe), "n_up_genes": len(up), "n_down_genes": len(down),
               "n_conflicting_genes": len(conflict), "libraries": {}}
    if len(up) + len(down) < 5:
        summary["limitation"] = "Too few significant genes for meaningful ORA."
        log.warning(summary["limitation"])
    frames = []
    for lib, path in gmt_paths.items():
        if not path.exists():
            summary["libraries"][lib] = "missing"
            log.warning("Gene-set library %s not found at %s; skipping", lib, path)
            continue
        sets = read_gmt(path)
        covered = np.mean([len(g & universe) / len(g) for g in sets.values() if g])
        summary["libraries"][lib] = {"n_sets": len(sets),
                                     "mean_fraction_of_set_genes_on_array": round(float(covered), 3)}
        for direction, hits in (("up in tumor", up), ("down in tumor", down)):
            r = ora(hits, universe, sets, min_size, max_size)
            if r.empty:
                continue
            r.insert(0, "direction", direction)
            r.insert(0, "library", lib)
            frames.append(r)
    res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not res.empty:
        split = res["term"].map(_split_term)
        res["term_name"] = [s[0] for s in split]
        res["term_id"] = [s[1] for s in split]
        res["term_url"] = res["term_id"].map(term_url)
        res["neg_log10_p_adj"] = -np.log10(res["p_adj"].clip(lower=1e-300))
    log.info("ORA: universe=%d genes, %d up / %d down; %d terms tested, %d with FDR<0.05",
             len(universe), len(up), len(down), len(res),
             int((res["p_adj"] < 0.05).sum()) if not res.empty else 0)
    return res, summary
