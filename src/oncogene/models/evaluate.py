"""Metrics and nested patient-grouped cross-validation.

Outer loop  : RepeatedPatientKFold -> unbiased-ish generalization estimate.
Inner loop  : CheckedGroupKFold inside GridSearchCV -> hyperparameter selection.
Every learned transform (imputation, variance ranking, scaling, k-best) is refit per fold.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from oncogene.models.interpret import fold_permutation_importance
from oncogene.models.train import (best_params, build_estimator, decision_scores,
                                   selected_feature_indices, unwrap)
from oncogene.models.tune import RepeatedPatientKFold, assert_no_patient_leakage

log = logging.getLogger("oncogene.models.evaluate")

METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "roc_auc"]


def compute_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Binary metrics with class 1 = tumor. ROC-AUC uses continuous scores, never hard labels."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def patient_bootstrap_ci(oof: pd.DataFrame, metric: str, n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI from resampling *patients* (both samples together) with replacement."""
    rng = np.random.default_rng(seed)
    pids = oof["patient_id"].unique()
    by_pid = {p: g for p, g in oof.groupby("patient_id")}
    vals = []
    for _ in range(n_boot):
        g = pd.concat([by_pid[p] for p in rng.choice(pids, size=len(pids), replace=True)])
        if g["y_true"].nunique() < 2:
            continue
        m = compute_metrics(g["y_true"], g["y_pred"], g["score"])
        vals.append(m[metric])
    if not vals:
        return (np.nan, np.nan)
    return (float(np.nanpercentile(vals, 100 * alpha / 2)),
            float(np.nanpercentile(vals, 100 * (1 - alpha / 2))))


@dataclass
class NestedCVResult:
    fold_metrics: pd.DataFrame
    oof: pd.DataFrame
    params: pd.DataFrame
    permutation: pd.DataFrame
    selection: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


def run_nested_cv(model_name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                  cfg: dict[str, Any], compute_permutation: bool = True) -> NestedCVResult:
    m = cfg["modeling"]
    seed = cfg["seed"]
    outer = RepeatedPatientKFold(m["outer_folds"], m["outer_repeats"], random_state=seed)
    Xv, yv, gv = X.values, y.values, groups.astype(str).values
    feat = np.asarray(X.columns)
    fold_rows, oof_rows, param_rows, perm_frames, sel_rows = [], [], [], [], []

    for split in outer.iter_splits(gv):
        tr, te = split.train_idx, split.test_idx
        assert_no_patient_leakage(gv, tr, te)
        est = build_estimator(model_name, cfg, seed=seed + split.repeat)
        t0 = time.perf_counter()
        if hasattr(est, "param_grid"):  # GridSearchCV: inner CV must see patient groups
            est.fit(Xv[tr], yv[tr], groups=gv[tr])
        else:
            est.fit(Xv[tr], yv[tr])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        score = decision_scores(est, Xv[te])
        pred = est.predict(Xv[te])
        pred_s = time.perf_counter() - t0

        met = compute_metrics(yv[te], pred, score)
        fold_rows.append({"model": model_name, "repeat": split.repeat, "fold": split.fold,
                          "n_train": len(tr), "n_test": len(te),
                          "n_test_patients": len(set(gv[te])),
                          "fit_seconds": fit_s, "predict_seconds": pred_s, **met})
        for i, idx in enumerate(te):
            oof_rows.append({"model": model_name, "repeat": split.repeat, "fold": split.fold,
                             "sample_id": X.index[idx], "patient_id": gv[idx],
                             "y_true": int(yv[idx]), "y_pred": int(pred[i]), "score": float(score[i])})
        bp = best_params(est)
        pipe = unwrap(est)
        sel = selected_feature_indices(pipe)
        n_tied = np.nan
        if hasattr(est, "cv_results_"):
            scores = np.asarray(est.cv_results_["mean_test_score"])
            n_tied = int(np.isclose(scores, est.best_score_).sum())
        param_rows.append({"model": model_name, "repeat": split.repeat, "fold": split.fold,
                           "inner_best_score": getattr(est, "best_score_", np.nan),
                           "n_grid_points": len(est.cv_results_["params"]) if hasattr(est, "cv_results_") else 1,
                           "n_tied_at_best": n_tied,
                           "n_features_used": len(sel),
                           **{k: str(v) for k, v in bp.items()}})
        if model_name != "dummy":
            sel_rows.append(pd.Series(1, index=feat[sel], name=f"r{split.repeat}f{split.fold}"))
            if compute_permutation:
                pi = fold_permutation_importance(
                    pipe, Xv[te], yv[te], sel, n_repeats=cfg["interpretation"]["permutation_repeats"],
                    seed=seed + 1000 * split.repeat + split.fold)
                pi.insert(0, "probe_id", feat[pi.pop("feature_index").values])
                pi.insert(0, "fold", split.fold)
                pi.insert(0, "repeat", split.repeat)
                pi.insert(0, "model", model_name)
                perm_frames.append(pi)
        log.info("%-20s repeat %d fold %d | AUC=%.3f BA=%.3f | k=%d | %s | %.1fs",
                 model_name, split.repeat, split.fold, met["roc_auc"], met["balanced_accuracy"],
                 len(sel), bp, fit_s)

    fm = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    n_folds = len(fm)
    selection = (pd.concat(sel_rows, axis=1).fillna(0).sum(axis=1) / n_folds
                 if sel_rows else pd.Series(dtype=float))
    selection = selection.rename("selection_frequency").rename_axis("probe_id").reset_index()
    selection.insert(0, "model", model_name)
    res = NestedCVResult(fm, oof, pd.DataFrame(param_rows),
                         pd.concat(perm_frames, ignore_index=True) if perm_frames else pd.DataFrame(),
                         selection)
    res.summary = summarize(model_name, fm, oof, cfg)
    return res


def summarize(model_name: str, fm: pd.DataFrame, oof: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Fold-level mean/SD/min/max plus pooled out-of-fold metrics with patient-bootstrap CIs."""
    out: dict[str, Any] = {"model": model_name, "n_outer_folds": int(len(fm)),
                           "fit_seconds_mean": float(fm["fit_seconds"].mean()),
                           "predict_seconds_mean": float(fm["predict_seconds"].mean()),
                           "fold": {}, "pooled": {}}
    for mname in METRICS:
        v = fm[mname]
        out["fold"][mname] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                              "min": float(v.min()), "max": float(v.max())}
    # Pool OOF predictions per repeat (each sample predicted exactly once per repeat).
    pooled = []
    for r, g in oof.groupby("repeat"):
        pooled.append(compute_metrics(g["y_true"], g["y_pred"], g["score"]))
    pooled_df = pd.DataFrame(pooled)
    rep0 = oof[oof["repeat"] == oof["repeat"].min()]
    nb = cfg["interpretation"]["bootstrap_iterations"]
    for mname in METRICS:
        lo, hi = patient_bootstrap_ci(rep0, mname, n_boot=nb, seed=cfg["seed"])
        out["pooled"][mname] = {"mean_over_repeats": float(pooled_df[mname].mean()),
                                "repeat0": float(pooled_df[mname].iloc[0]),
                                "ci95_patient_bootstrap_repeat0": [lo, hi]}
    cm = pooled_df[["tn", "fp", "fn", "tp"]].sum().astype(int).to_dict()
    out["confusion_matrix_summed_over_repeats"] = cm
    out["confusion_matrix_repeat0"] = {k: int(pooled_df[k].iloc[0]) for k in ("tn", "fp", "fn", "tp")}
    out["patient_level_exact_ci"] = patient_level_exact_ci(rep0)
    return out


def run_learning_curve(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """Held-out performance vs. number of training *patients* (fixed, untuned hyperparameters).

    For each training size n, sample n patients (both samples each) for training and evaluate on
    all remaining patients; repeat with different seeds. Shows how quickly the signal is learned
    and how variable performance is with very few patients.
    """
    from oncogene.models.train import build_pipeline
    lc = cfg["learning_curve"]
    rng = np.random.default_rng(cfg["seed"])
    gv = groups.astype(str).values
    pids = np.array(sorted(set(gv)))
    rows = []
    for n in lc["train_patients"]:
        if n >= len(pids):
            continue
        for r in range(lc["repeats"]):
            train_p = rng.choice(pids, size=n, replace=False)
            tr = np.flatnonzero(np.isin(gv, train_p))
            te = np.flatnonzero(~np.isin(gv, train_p))
            assert_no_patient_leakage(gv, tr, te)
            pipe = build_pipeline(lc["model"], cfg).set_params(**lc["params"])
            pipe.fit(X.values[tr], y.values[tr])
            s = decision_scores(pipe, X.values[te])
            met = compute_metrics(y.values[te], pipe.predict(X.values[te]), s)
            rows.append({"model": lc["model"], "n_train_patients": n, "repeat": r,
                         "n_test_patients": len(set(gv[te])), **met})
    df = pd.DataFrame(rows)
    log.info("Learning curve: %s", df.groupby("n_train_patients")[["roc_auc", "balanced_accuracy"]]
             .mean().round(3).to_dict())
    return df


def patient_level_exact_ci(oof: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Clopper-Pearson CI treating each *patient* as one Bernoulli trial.

    A patient counts as a success only if both of their samples are classified correctly.
    This is conservative and, unlike a bootstrap, stays informative when every prediction
    is correct (0/25 errors still allows an error rate up to ~14%).
    """
    from scipy.stats import binomtest
    ok = (oof["y_true"] == oof["y_pred"]).groupby(oof["patient_id"]).all()
    k, n = int(ok.sum()), int(len(ok))
    ci = binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")
    return {"patients_fully_correct": k, "n_patients": n,
            "proportion": k / n if n else np.nan,
            "ci95": [float(ci.low), float(ci.high)],
            "max_plausible_patient_error_rate": float(1 - ci.low)}
