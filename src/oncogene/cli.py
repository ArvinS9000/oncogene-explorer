"""Command-line entry point: ``oncogene <stage>`` or ``python -m oncogene.cli <stage>``.

Stages (run in order by ``all``):
  download    fetch GEO series matrix, platform annotation, gene-set libraries
  preprocess  parse, validate, apply label-free filters, write processed data
  analyze     exploratory analysis, paired DE, pathway ORA
  train       nested patient-grouped CV for all models, final models, interpretation
  figures     static PNG figures for README / report
  report      Markdown research summary
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from oncogene.config import Paths, load_config, setup_logging

log = logging.getLogger("oncogene.cli")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    log.info("Wrote %s", path)


# ----------------------------------------------------------------------------- stages
def stage_download(cfg, paths, force=False):
    from oncogene.data.download import download_all
    download_all(cfg, paths, force=force)


def stage_preprocess(cfg, paths):
    from oncogene.preprocessing.cleaning import build_dataset
    build_dataset(cfg, paths)


def stage_analyze(cfg, paths):
    from oncogene.analysis import exploratory as eda
    from oncogene.analysis.differential_expression import paired_de
    from oncogene.analysis.pathway_analysis import run_pathway_analysis
    from oncogene.data.load import load_processed

    ds = load_processed(paths)
    X, meta, ann = ds.X, ds.meta, ds.annotation
    mdir = paths.metrics

    write_json(mdir / "eda_overview.json", eda.dataset_overview(X, meta))
    eda.per_sample_summary(X, meta).to_csv(mdir / "per_sample_summary.csv")
    pca_df, evr = eda.exploratory_pca(X, meta)
    pca_df.to_csv(mdir / "pca_scores.csv")
    write_json(mdir / "pca_variance.json", {"explained_variance_ratio": evr.tolist(),
                                            "note": "Exploratory PCA on top-2000 variable probes, all samples."})
    corr = eda.sample_correlation(X)
    corr.to_csv(mdir / "sample_correlation.csv")
    write_json(mdir / "sample_cluster_order.json", eda.cluster_order(corr))
    out = eda.detect_outliers(corr, meta)
    out.to_csv(mdir / "outlier_screen.csv")
    log.info("Outlier screen flagged %d sample(s): %s", int(out["flagged"].sum()),
             out.index[out["flagged"]].tolist())
    eda.pair_distances(X, meta).to_csv(mdir / "pair_distances.csv", index=False)
    scores, coverage = eda.marker_scores(X, meta, ann, cfg.get("marker_panels", {}))
    scores.to_csv(mdir / "marker_scores.csv")
    write_json(mdir / "marker_panel_coverage.json", coverage)

    dcfg = cfg["differential_expression"]
    de = paired_de(X, meta, ann, fdr_alpha=dcfg["fdr_alpha"], min_abs_log2fc=dcfg["min_abs_log2fc"])
    de.to_csv(mdir / "de_results.csv")

    pcfg = cfg["pathway"]
    gmts = {lib: paths.raw / f"{lib}.gmt" for lib in cfg["dataset"]["gene_set_libraries"]}
    pw, pw_summary = run_pathway_analysis(de, gmts, pcfg["min_set_size"], pcfg["max_set_size"])
    pw.to_csv(mdir / "pathway_results.csv", index=False)
    write_json(mdir / "pathway_summary.json", pw_summary)


def stage_train(cfg, paths, models: list[str] | None = None, fast: bool = False):
    from oncogene.data.load import load_processed
    from oncogene.models import interpret
    from oncogene.models.evaluate import run_nested_cv
    from oncogene.models.train import (MODEL_NAMES, build_estimator, save_model, unwrap)

    if fast:
        cfg = json.loads(json.dumps(cfg))
        cfg["modeling"]["outer_repeats"] = 1
        cfg["interpretation"]["permutation_repeats"] = 5
        cfg["interpretation"]["bootstrap_iterations"] = 200
        log.warning("FAST mode: 1 outer repeat, reduced permutation/bootstrap iterations")

    ds = load_processed(paths)
    X, y, groups = ds.X, ds.y, ds.groups
    models = models or MODEL_NAMES
    mdir = paths.metrics

    fms, oofs, params, perms, sels, summaries = [], [], [], [], [], {}
    t_all = time.perf_counter()
    for name in models:
        log.info("=== Nested CV: %s ===", name)
        res = run_nested_cv(name, X, y, groups, cfg)
        fms.append(res.fold_metrics); oofs.append(res.oof); params.append(res.params)
        perms.append(res.permutation); sels.append(res.selection)
        summaries[name] = res.summary
        f = res.summary["fold"]
        log.info("%s: AUC %.3f ± %.3f | balanced acc %.3f ± %.3f", name, f["roc_auc"]["mean"],
                 f["roc_auc"]["sd"], f["balanced_accuracy"]["mean"], f["balanced_accuracy"]["sd"])

    fm = pd.concat(fms, ignore_index=True); fm.to_csv(mdir / "fold_metrics.csv", index=False)
    oof = pd.concat(oofs, ignore_index=True); oof.to_csv(mdir / "oof_predictions.csv", index=False)
    pd.concat(params, ignore_index=True).to_csv(mdir / "fold_best_params.csv", index=False)
    perm = pd.concat([p for p in perms if not p.empty], ignore_index=True) if any(not p.empty for p in perms) else pd.DataFrame()
    perm.to_csv(mdir / "permutation_importance_folds.csv", index=False)
    sel = pd.concat([s for s in sels if not s.empty], ignore_index=True)
    sel.to_csv(mdir / "selection_frequency.csv", index=False)
    if "learning_curve" in cfg:
        from oncogene.models.evaluate import run_learning_curve
        lc_cfg = cfg if not fast else {**cfg, "learning_curve": {**cfg["learning_curve"], "repeats": 5}}
        run_learning_curve(X, y, groups, lc_cfg).to_csv(mdir / "learning_curve.csv", index=False)
    n_folds = fm.groupby("model").size().to_dict()
    perm_agg = interpret.aggregate_permutation(perm, n_folds)
    perm_agg.to_csv(mdir / "permutation_importance.csv", index=False)

    # Final models: tuned with patient-grouped CV on ALL samples, for interpretation only.
    finals = {}
    for name in models:
        est = build_estimator(name, cfg)
        if hasattr(est, "param_grid"):
            est.fit(X.values, y.values, groups=groups.astype(str).values)
        else:
            est.fit(X.values, y.values)
        finals[name] = est
        save_model(est, name, paths.models, cfg, list(X.columns),
                   extra={"nested_cv_summary": {k: v["mean"] for k, v in summaries[name]["fold"].items()}})

    feat = list(X.columns)
    lr_coef = interpret.linear_coefficients(unwrap(finals["logistic_regression"]), feat) if "logistic_regression" in finals else None
    rf_imp = interpret.tree_importance(unwrap(finals["random_forest"]), feat) if "random_forest" in finals else None
    svm_coef = interpret.linear_coefficients(unwrap(finals["svm"]), feat) if "svm" in finals else None
    ann = ds.annotation
    for df, fname in ((lr_coef, "final_lr_coefficients.csv"), (rf_imp, "final_rf_importance.csv"),
                      (svm_coef, "final_svm_coefficients.csv")):
        if df is not None:
            df.merge(ann[["gene_symbol", "gene_description"]], left_on="probe_id",
                     right_index=True, how="left").to_csv(mdir / fname, index=False)

    de_path = mdir / "de_results.csv"
    de = pd.read_csv(de_path, index_col="probe_id", dtype={"probe_id": str}) if de_path.exists() else pd.DataFrame(
        columns=["log2_fold_change", "p_adj", "significant", "direction"])
    de.index = de.index.astype(str)
    ranking = interpret.build_gene_ranking(ann, de, lr_coef, rf_imp, perm_agg, sel)
    ranking.to_csv(mdir / "gene_ranking.csv", index=False)

    summary = {
        "experiment_name": cfg["experiment_name"], "model_version": cfg["model_version"],
        "seed": cfg["seed"], "n_samples": int(len(X)), "n_patients": int(groups.nunique()),
        "n_features": int(X.shape[1]), "fast_mode": fast,
        "cv": {k: cfg["modeling"][k] for k in ("outer_folds", "outer_repeats", "inner_folds", "scoring")},
        "total_seconds": time.perf_counter() - t_all,
        "models": summaries,
    }
    write_json(mdir / "model_summary.json", summary)
    shutil.copy(paths.root / "configs" / "experiment.yaml", paths.reports / "experiment_config_used.yaml")


def stage_figures(cfg, paths):
    from oncogene.data.load import load_processed
    from oncogene.visualization.plots import export_static_figures
    ds = load_processed(paths)
    m = paths.metrics
    pca_df = pd.read_csv(m / "pca_scores.csv", index_col=0, dtype={"patient_id": str})
    evr = np.array(json.loads((m / "pca_variance.json").read_text())["explained_variance_ratio"])
    de = pd.read_csv(m / "de_results.csv", index_col="probe_id", dtype={"probe_id": str})
    oof = pd.read_csv(m / "oof_predictions.csv")
    fm = pd.read_csv(m / "fold_metrics.csv")
    top = de[de["significant"]].assign(s=lambda d: d["log2_fold_change"].abs()).nlargest(40, "s")
    labels = [s if isinstance(s, str) else p for p, s in zip(top.index, top["gene_symbol"])]
    dc = cfg["differential_expression"]
    for p in export_static_figures(paths.figures, X=ds.X, meta=ds.meta, pca_df=pca_df, evr=evr, de=de,
                                   oof=oof, fm=fm, top_probes=list(top.index), probe_labels=labels,
                                   fdr=dc["fdr_alpha"], min_lfc=dc["min_abs_log2fc"]):
        log.info("Figure: %s", p)


def stage_report(cfg, paths):
    from oncogene.reporting import write_report
    p = write_report(cfg, paths)
    log.info("Report: %s", p)


STAGES = ["download", "preprocess", "analyze", "train", "figures", "report"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oncogene", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", default=None, help="path to experiment YAML")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models for `train`")
    ap.add_argument("--fast", action="store_true", help="quick smoke run (1 outer repeat)")
    args = ap.parse_args(argv)

    paths = Paths().ensure()
    setup_logging(paths.reports / "pipeline.log")
    cfg = load_config(args.config)
    np.random.seed(cfg["seed"])
    stages = STAGES if args.stage == "all" else [args.stage]
    try:
        for s in stages:
            log.info(">>> stage: %s", s)
            t0 = time.perf_counter()
            if s == "download":
                stage_download(cfg, paths, args.force_download)
            elif s == "preprocess":
                stage_preprocess(cfg, paths)
            elif s == "analyze":
                stage_analyze(cfg, paths)
            elif s == "train":
                stage_train(cfg, paths, args.models, args.fast)
            elif s == "figures":
                stage_figures(cfg, paths)
            elif s == "report":
                stage_report(cfg, paths)
            log.info("<<< stage %s done in %.1fs", s, time.perf_counter() - t0)
    except Exception:
        log.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
