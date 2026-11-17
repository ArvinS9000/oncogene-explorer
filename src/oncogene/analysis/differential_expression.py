"""Paired differential expression: tumor vs. matched adjacent-normal tissue.

Method
------
For every probe and patient we form d = tumor - normal on the log2 RMA scale. Because the
values are already log2, mean(d) is a valid log2 fold change. The primary test is a paired
t-test (one-sample t-test on d); a Wilcoxon signed-rank test is reported as a
distribution-free sensitivity check. P-values are corrected with Benjamini-Hochberg FDR.

A moderated (limma-style) t-test would borrow variance information across genes; it is not
implemented here and is documented as a limitation / future improvement.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("oncogene.analysis.de")


def benjamini_hochberg(p: np.ndarray | pd.Series) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs preserved)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def paired_differences(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Patients x probes matrix of (tumor - normal) log2 differences."""
    tumor = meta[meta["label"] == 1].reset_index().set_index("patient_id")["sample_id"]
    normal = meta[meta["label"] == 0].reset_index().set_index("patient_id")["sample_id"]
    pids = sorted(set(tumor.index) & set(normal.index))
    if len(pids) != meta["patient_id"].nunique():
        raise ValueError("Not every patient has both a tumor and a normal sample")
    D = X.loc[tumor[pids].values].values - X.loc[normal[pids].values].values
    return pd.DataFrame(D, index=pd.Index(pids, name="patient_id"), columns=X.columns)


def paired_de(X: pd.DataFrame, meta: pd.DataFrame, annotation: pd.DataFrame | None = None,
              fdr_alpha: float = 0.05, min_abs_log2fc: float = 1.0,
              wilcoxon: bool = True) -> pd.DataFrame:
    """Run paired DE for every probe and return a ranked results table."""
    D = paired_differences(X, meta)
    n = D.shape[0]
    mean_d = D.mean(axis=0)
    sd_d = D.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean_d / (sd_d / np.sqrt(n))
    p_t = 2 * stats.t.sf(np.abs(t), df=n - 1)
    p_t = np.where(sd_d == 0, 1.0, p_t)

    res = pd.DataFrame({
        "log2_fold_change": mean_d,
        "fold_change": np.power(2.0, mean_d),
        "mean_tumor": X[meta["label"].values == 1].mean(axis=0),
        "mean_normal": X[meta["label"].values == 0].mean(axis=0),
        "sd_paired_diff": sd_d,
        "cohens_dz": mean_d / sd_d.replace(0, np.nan),
        "t_statistic": t,
        "p_value": p_t,
        "n_pairs": n,
        "frac_pairs_up": (D > 0).mean(axis=0),
    }, index=X.columns)
    res["p_adj"] = benjamini_hochberg(res["p_value"].values)

    if wilcoxon:
        # Vectorised across probes; zero differences dropped (Wilcoxon default 'wilcox').
        w = stats.wilcoxon(D.values, axis=0, zero_method="wilcox", nan_policy="omit")
        res["wilcoxon_p"] = w.pvalue
        res["wilcoxon_p_adj"] = benjamini_hochberg(w.pvalue)

    res["direction"] = np.where(res["log2_fold_change"] > 0, "up in tumor", "down in tumor")
    res["significant"] = (res["p_adj"] < fdr_alpha) & (res["log2_fold_change"].abs() >= min_abs_log2fc)
    res["neg_log10_p_adj"] = -np.log10(res["p_adj"].clip(lower=1e-300))

    if annotation is not None:
        cols = [c for c in ("gene_symbol", "gene_description", "entrez_id", "chromosome",
                            "cytoband") if c in annotation.columns]
        res = annotation[cols].reindex(res.index).join(res)
    res.index.name = "probe_id"
    res = res.sort_values(["p_adj", "p_value"])
    log.info("DE: %d probes tested, %d FDR<%.2f, %d also |log2FC|>=%.1f (%d up / %d down)",
             len(res), int((res["p_adj"] < fdr_alpha).sum()), fdr_alpha,
             int(res["significant"].sum()), min_abs_log2fc,
             int((res["significant"] & (res["log2_fold_change"] > 0)).sum()),
             int((res["significant"] & (res["log2_fold_change"] < 0)).sum()))
    return res
