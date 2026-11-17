"""Model interpretation.

* Logistic-regression / linear-SVM coefficients: inputs are standardized inside the pipeline,
  so coefficients are on a per-SD scale. With thousands of correlated genes and L2
  regularization, weight is spread across correlated features; magnitudes are not causal.
* Random-forest impurity importance: biased toward high-variance features and split
  arbitrarily among correlated features. Reported, but secondary to permutation importance.
* Permutation importance: computed on each OUTER validation fold (held-out patients only)
  for the features the fold's pipeline actually used, then averaged across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def _scores(clf, Z: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        try:
            return clf.predict_proba(Z)[:, 1]
        except AttributeError:
            pass
    return clf.decision_function(Z)


def _margin(clf, Z: np.ndarray, y: np.ndarray) -> float:
    """Mean signed margin toward the correct class on held-out data.

    Uses decision_function (log-odds for LR, signed distance for SVM) when available,
    otherwise P(tumor) - 0.5 (random forest). Larger = more confidently correct.
    """
    s = clf.decision_function(Z) if hasattr(clf, "decision_function") else clf.predict_proba(Z)[:, 1] - 0.5
    return float(np.mean((2 * np.asarray(y) - 1) * s))


def fold_permutation_importance(pipe: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
                                selected: np.ndarray, n_repeats: int = 20,
                                seed: int = 0) -> pd.DataFrame:
    """Permutation importance on one held-out fold, for the features the fold's model used.

    Two scores are recorded per feature:

    * ``auc_drop``: drop in held-out ROC-AUC. With highly redundant expression features this
      is typically 0 for every single gene (no one gene is necessary) - itself a finding.
    * ``importance_mean``: relative drop in the mean signed margin (see ``_margin``), i.e. how
      much less confidently correct the held-out predictions become. This is the ranking score.

    Every pre-classifier step is a per-column operation (impute, select, scale), so permuting
    an input column is equivalent to permuting its transformed column; we permute in the
    reduced space for speed. Features never used by a tree model provably have zero
    permutation importance and are skipped (recorded as 0).
    """
    rng = np.random.default_rng(seed)
    Z = pipe[:-1].transform(X_test)
    clf = pipe[-1]
    base_auc = roc_auc_score(y_test, _scores(clf, Z))
    base_margin = _margin(clf, Z, y_test)
    denom = abs(base_margin) if base_margin != 0 else 1.0
    used = np.ones(Z.shape[1], dtype=bool)
    if hasattr(clf, "feature_importances_"):
        used = clf.feature_importances_ > 0
    means, sds, auc_drop = np.zeros(Z.shape[1]), np.zeros(Z.shape[1]), np.zeros(Z.shape[1])
    for j in np.flatnonzero(used):
        drops, adrops = np.empty(n_repeats), np.empty(n_repeats)
        col = Z[:, j].copy()
        for r in range(n_repeats):
            Z[:, j] = rng.permutation(col)
            drops[r] = (base_margin - _margin(clf, Z, y_test)) / denom
            adrops[r] = base_auc - roc_auc_score(y_test, _scores(clf, Z))
        Z[:, j] = col
        means[j], sds[j], auc_drop[j] = drops.mean(), drops.std(ddof=1), adrops.mean()
    return pd.DataFrame({"feature_index": selected, "importance_mean": means,
                         "importance_sd": sds, "auc_drop": auc_drop,
                         "baseline_auc": base_auc, "baseline_margin": base_margin})


def linear_coefficients(pipe: Pipeline, feature_names: list[str] | np.ndarray) -> pd.DataFrame | None:
    """Standardized coefficients for linear models (LR, linear SVM); None otherwise."""
    from oncogene.models.train import selected_feature_indices
    clf = pipe[-1]
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef.toarray() if hasattr(coef, "toarray") else coef).ravel()
    idx = selected_feature_indices(pipe)
    df = pd.DataFrame({"probe_id": np.asarray(feature_names)[idx], "coefficient": coef})
    df["abs_coefficient"] = df["coefficient"].abs()
    df["direction"] = np.where(df["coefficient"] > 0, "higher in tumor-predicted", "higher in normal-predicted")
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def tree_importance(pipe: Pipeline, feature_names: list[str] | np.ndarray) -> pd.DataFrame | None:
    from oncogene.models.train import selected_feature_indices
    clf = pipe[-1]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        return None
    idx = selected_feature_indices(pipe)
    df = pd.DataFrame({"probe_id": np.asarray(feature_names)[idx], "impurity_importance": imp})
    return df.sort_values("impurity_importance", ascending=False).reset_index(drop=True)


def aggregate_permutation(perm: pd.DataFrame, n_folds_by_model: dict[str, int]) -> pd.DataFrame:
    """Average fold-level permutation importance; folds where a probe was unused count as 0."""
    if perm.empty:
        return perm
    g = perm.groupby(["model", "probe_id"]).agg(sum=("importance_mean", "sum"),
                                                folds_used=("importance_mean", "count"),
                                                auc_sum=("auc_drop", "sum")).reset_index()
    g["n_folds_total"] = g["model"].map(n_folds_by_model)
    g["permutation_importance"] = g["sum"] / g["n_folds_total"]
    g["mean_auc_drop"] = g["auc_sum"] / g["n_folds_total"]
    g = g.drop(columns=["sum", "auc_sum"])
    return g.sort_values(["model", "permutation_importance"], ascending=[True, False])


def build_gene_ranking(annotation: pd.DataFrame, de: pd.DataFrame, lr_coef: pd.DataFrame | None,
                       rf_imp: pd.DataFrame | None, perm_agg: pd.DataFrame,
                       selection: pd.DataFrame) -> pd.DataFrame:
    """One long table: (probe, method) -> importance score, joined with DE statistics."""
    frames = []
    if lr_coef is not None:
        frames.append(lr_coef.assign(method="LR standardized coefficient (final model)",
                                     importance=lr_coef["abs_coefficient"],
                                     signed_value=lr_coef["coefficient"])[
            ["probe_id", "method", "importance", "signed_value"]])
    if rf_imp is not None:
        frames.append(rf_imp.assign(method="RF impurity importance (final model)",
                                    importance=rf_imp["impurity_importance"], signed_value=np.nan)[
            ["probe_id", "method", "importance", "signed_value"]])
    for model, g in perm_agg.groupby("model"):
        frames.append(g.assign(method=f"Permutation importance, held-out margin drop ({model})",
                               importance=g["permutation_importance"], signed_value=np.nan)[
            ["probe_id", "method", "importance", "signed_value"]])
    for model, g in selection.groupby("model"):
        frames.append(g.assign(method=f"Outer-fold selection frequency ({model})",
                               importance=g["selection_frequency"], signed_value=np.nan)[
            ["probe_id", "method", "importance", "signed_value"]])
    long = pd.concat(frames, ignore_index=True)
    long["rank_within_method"] = long.groupby("method")["importance"].rank(ascending=False, method="min")
    ann_cols = ["gene_symbol", "gene_description", "entrez_id", "cytoband"]
    de_cols = ["log2_fold_change", "p_adj", "significant", "direction"]
    long = long.merge(annotation[ann_cols], left_on="probe_id", right_index=True, how="left")
    long = long.merge(de[de_cols].rename(columns={"direction": "de_direction"}),
                      left_on="probe_id", right_index=True, how="left")
    return long.sort_values(["method", "rank_within_method"]).reset_index(drop=True)
