"""Exploratory analysis.

Everything here is DESCRIPTIVE and uses all 50 samples. It is never used as model input;
cross-validated models refit their own transforms inside each training fold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


def dataset_overview(X: pd.DataFrame, meta: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_samples": int(len(meta)),
        "n_patients": int(meta["patient_id"].nunique()),
        "n_tumor": int((meta["label"] == 1).sum()),
        "n_normal": int((meta["label"] == 0).sum()),
        "n_features": int(X.shape[1]),
        "missing_values": int(X.isna().sum().sum()),
        "expression_min": float(X.values.min()),
        "expression_max": float(X.values.max()),
        "expression_median": float(np.median(X.values)),
        "age_median": float(meta.drop_duplicates("patient_id")["age_years"].median()),
        "stage_counts": meta.drop_duplicates("patient_id")["cancer_stage"].value_counts().to_dict(),
    }


def per_sample_summary(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    q = X.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    q.columns = ["q05", "q25", "median", "q75", "q95"]
    out = pd.concat([meta[["patient_id", "tissue_type"]], q,
                     X.mean(axis=1).rename("mean"), X.std(axis=1).rename("sd")], axis=1)
    return out


def exploratory_pca(X: pd.DataFrame, meta: pd.DataFrame, n_components: int = 5,
                    top_variable: int | None = 2000) -> tuple[pd.DataFrame, np.ndarray]:
    """Unsupervised PCA on (optionally) the most variable probes, centred per probe."""
    Xs = X
    if top_variable and top_variable < X.shape[1]:
        Xs = X[X.var().sort_values(ascending=False).index[:top_variable]]
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(Xs - Xs.mean())
    df = pd.DataFrame(scores, index=X.index, columns=[f"PC{i+1}" for i in range(n_components)])
    df = df.join(meta[["patient_id", "tissue_type", "label", "cancer_stage", "age_years"]])
    return df, pca.explained_variance_ratio_


def sample_correlation(X: pd.DataFrame, top_variable: int | None = 2000) -> pd.DataFrame:
    Xs = X
    if top_variable and top_variable < X.shape[1]:
        Xs = X[X.var().sort_values(ascending=False).index[:top_variable]]
    return Xs.T.corr(method="pearson")


def cluster_order(corr: pd.DataFrame, method: str = "average") -> list[str]:
    dist = 1 - corr.values
    np.fill_diagonal(dist, 0)
    Z = hierarchy.linkage(squareform(dist, checks=False), method=method)
    return [corr.index[i] for i in hierarchy.leaves_list(Z)]


def detect_outliers(corr: pd.DataFrame, meta: pd.DataFrame, z_thresh: float = -2.5) -> pd.DataFrame:
    """Flag samples whose mean correlation to same-tissue samples is unusually low.

    Uses a robust z-score (median / MAD) within each tissue group. Flags are informational:
    no sample is removed automatically.
    """
    rows = []
    for sid in corr.index:
        same = meta.index[(meta["label"] == meta.loc[sid, "label"]) & (meta.index != sid)]
        rows.append((sid, corr.loc[sid, same].mean()))
    df = pd.DataFrame(rows, columns=["sample_id", "mean_within_group_corr"]).set_index("sample_id")
    df = df.join(meta[["patient_id", "tissue_type"]])
    z = []
    for _, g in df.groupby("tissue_type"):
        med = g["mean_within_group_corr"].median()
        mad = (g["mean_within_group_corr"] - med).abs().median() * 1.4826 or 1e-9
        z.append((g["mean_within_group_corr"] - med) / mad)
    df["robust_z"] = pd.concat(z)
    df["flagged"] = df["robust_z"] < z_thresh
    return df


def marker_scores(X: pd.DataFrame, meta: pd.DataFrame, annotation: pd.DataFrame,
                  panels: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]]]:
    """Per-sample score = mean z-score of a fixed marker panel's probes.

    Panels are fixed literature lists (not derived from these labels). Scores describe tissue
    composition, e.g. how adipose-rich a sample is, and help interpret what separates classes.
    """
    sym = annotation["gene_symbol"]
    Z = (X - X.mean()) / X.std().replace(0, 1)
    out = meta[["patient_id", "tissue_type"]].copy()
    coverage = {}
    for name, genes in panels.items():
        genes_u = [g.upper() for g in genes]
        probes = sym.index[sym.isin(genes_u)]
        found = sorted(set(sym.loc[probes]))
        coverage[name] = {"found": found, "missing": sorted(set(genes_u) - set(found))}
        out[f"{name}_score"] = Z[probes].mean(axis=1) if len(probes) else np.nan
    return out, coverage


def pair_distances(X: pd.DataFrame, meta: pd.DataFrame, top_variable: int | None = 2000) -> pd.DataFrame:
    """Per-patient tumor-vs-normal distance (1 - Pearson r) and mean |log2 difference|."""
    Xs = X
    if top_variable and top_variable < X.shape[1]:
        Xs = X[X.var().sort_values(ascending=False).index[:top_variable]]
    rows = []
    for pid, g in meta.groupby("patient_id"):
        t = Xs.loc[g.index[g["label"] == 1][0]]
        n = Xs.loc[g.index[g["label"] == 0][0]]
        rows.append({"patient_id": pid,
                     "tumor_sample": t.name, "normal_sample": n.name,
                     "correlation": float(np.corrcoef(t, n)[0, 1]),
                     "mean_abs_log2_diff": float((t - n).abs().mean()),
                     "cancer_stage": g["cancer_stage"].iloc[0],
                     "age_years": g["age_years"].iloc[0]})
    return pd.DataFrame(rows).sort_values("correlation")
