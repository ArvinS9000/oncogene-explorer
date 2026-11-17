"""Generate a Markdown research summary from pipeline artifacts (Advanced Feature E)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from oncogene.config import Paths

MODEL_LABELS = {"dummy": "Dummy (prior)", "logistic_regression": "Logistic regression",
                "random_forest": "Random forest", "svm": "SVM"}


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci and ci[0] is not None else "n/a"


def model_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for m, s in summary["models"].items():
        f = s["fold"]
        ex = s["patient_level_exact_ci"]
        rows.append({
            "Model": MODEL_LABELS.get(m, m),
            "ROC-AUC (fold mean ± SD)": f"{f['roc_auc']['mean']:.3f} ± {f['roc_auc']['sd']:.3f}",
            "Balanced acc.": f"{f['balanced_accuracy']['mean']:.3f} ± {f['balanced_accuracy']['sd']:.3f}",
            "Recall (sens.)": f"{f['recall']['mean']:.3f}",
            "Specificity": f"{f['specificity']['mean']:.3f}",
            "F1": f"{f['f1']['mean']:.3f}",
            "Patients fully correct (exact 95% CI)": f"{ex['patients_fully_correct']}/{ex['n_patients']} {_fmt_ci(ex['ci95'])}",
            "Fit s/fold": f"{s['fit_seconds_mean']:.1f}",
        })
    return pd.DataFrame(rows)


def write_report(cfg: dict[str, Any], paths: Paths | None = None) -> Path:
    paths = paths or Paths()
    m = paths.metrics
    ov = json.loads((m / "eda_overview.json").read_text())
    pre = json.loads((paths.reports / "preprocessing_report.json").read_text())
    summ = json.loads((m / "model_summary.json").read_text())
    de = pd.read_csv(m / "de_results.csv", dtype={"probe_id": str})
    pw = pd.read_csv(m / "pathway_results.csv")
    pws = json.loads((m / "pathway_summary.json").read_text())
    lc = pd.read_csv(m / "learning_curve.csv") if (m / "learning_curve.csv").exists() else None
    dc = cfg["differential_expression"]

    sig = de[de["significant"]]
    up = sig[sig["log2_fold_change"] > 0].nlargest(10, "log2_fold_change")
    dn = sig[sig["log2_fold_change"] < 0].nsmallest(10, "log2_fold_change")
    gl = lambda d: ", ".join(f"{s} ({v:+.2f})" for s, v in zip(d["gene_symbol"].fillna(d["probe_id"]), d["log2_fold_change"]))

    lines = [
        "# OncoGene Explorer - research summary",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from pipeline artifacts. "
        "Research and education only; not a diagnostic tool._",
        "",
        "## Dataset",
        f"- GEO **{cfg['dataset']['accession']}** ({cfg['dataset']['platform']}, Affymetrix Human Exon 1.0 ST, "
        "core transcript clusters, RMA log2).",
        f"- {ov['n_samples']} samples from {ov['n_patients']} patients: {ov['n_tumor']} tumor, "
        f"{ov['n_normal']} paired **adjacent-normal** (not healthy-donor) tissue.",
        f"- {ov['n_features']:,} probes after label-free filtering; {ov['missing_values']} missing values; "
        f"log2 transform applied: {pre['log2_transform_applied']}.",
        f"- Median patient age {ov['age_median']:.0f}; stages: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(ov["stage_counts"].items())) + ".",
        "",
        "## Differential expression (paired t-test, BH-FDR)",
        f"- {int((de['p_adj'] < dc['fdr_alpha']).sum()):,} probes at FDR < {dc['fdr_alpha']}; "
        f"{len(sig):,} also |log2FC| ≥ {dc['min_abs_log2fc']} "
        f"({(sig['log2_fold_change'] > 0).sum()} up, {(sig['log2_fold_change'] < 0).sum()} down in tumor).",
        f"- Largest increases in tumor: {gl(up)}.",
        f"- Largest decreases in tumor: {gl(dn)}.",
        "",
        "## Pathway over-representation (exploratory)",
        f"- Universe: {pws['universe_size']:,} array genes; {pws['n_up_genes']} up / {pws['n_down_genes']} down genes tested.",
    ]
    for direction in ("up in tumor", "down in tumor"):
        top = pw[(pw["direction"] == direction)].nsmallest(5, "p_adj")
        lines.append(f"- Top terms, genes {direction}: " + "; ".join(
            f"{t} ({lib.split('_')[0]}, FDR {p:.1e})" for t, lib, p in zip(top["term_name"], top["library"], top["p_adj"])))
    lines += ["", "## Classification (nested, patient-grouped CV)",
              f"Outer: {summ['cv']['outer_folds']}-fold × {summ['cv']['outer_repeats']} repeats over patients; "
              f"inner: {summ['cv']['inner_folds']}-fold grouped grid search on {summ['cv']['scoring']}.", "",
              model_table(summ).to_markdown(index=False), ""]
    if lc is not None:
        g = lc.groupby("n_train_patients")[["roc_auc", "balanced_accuracy"]].agg(["mean", "std"]).round(3)
        g.columns = [f"{m.replace('_', ' ')} ({s})" for m, s in g.columns]
        g.index.name = "training patients"
        lines += ["### Learning curve (logistic regression, fixed hyperparameters)", "",
                  g.to_markdown(), ""]
    lines += [
        "## Interpretation and limitations",
        "- The task (tumor vs. **adjacent** normal from the same breast) is close to trivially separable; "
        "near-perfect metrics reflect large, consistent tissue-level differences - notably adipocyte/stromal "
        "content vs. proliferation - and do **not** imply diagnostic utility.",
        "- 25 independent patients: even zero observed errors are compatible with a substantial true error rate "
        "(see exact CIs above).",
        "- Single cohort, single platform, no external validation; adjacent-normal tissue can carry field "
        "effects and differs from healthy-donor breast tissue.",
        "- Feature importance and DE identify statistical associations, not causal mechanisms.",
    ]
    out = paths.reports / "research_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
