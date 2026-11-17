import streamlit as st

from components import data
from components.ui import DISCLAIMER, guarded, note, page_setup

page_setup("Findings & limitations", "What this analysis does — and does not — show",
           "A summary of results with the caveats needed to interpret them responsibly.")

with guarded("the findings summary"):
    ov = data.metric_json("eda_overview.json")
    de = data.metric_csv("de_results.csv", index_col="probe_id")
    summ = data.metric_json("model_summary.json")
    pws = data.metric_json("pathway_summary.json")
    sig = de[de["significant"]]

    st.subheader("1 · Descriptive observations")
    st.markdown(f"""
- {ov['n_samples']} samples from {ov['n_patients']} patients. Every patient contributes exactly one tumor and one
  adjacent-normal sample, and all labels and patient IDs were cross-checked against two independent GEO fields.
- RMA log2 values with no missing data and tightly aligned per-sample distributions.
- In exploratory PCA, the first component largely separates tumor from adjacent normal tissue, and each patient's two
  samples are usually far apart. Tissue type dominates over patient identity.
""")
    st.subheader("2 · Differential expression")
    st.markdown(f"""
- **{len(sig):,} probes** pass FDR < 0.05 and |log2FC| ≥ 1 (**{(sig.log2_fold_change > 0).sum()} up**,
  **{(sig.log2_fold_change < 0).sum()} down** in tumor); {(de.p_adj < 0.05).sum():,} probes pass FDR alone.
- The largest decreases are adipocyte and stromal genes (FABP4, ADIPOQ, CD36, LYVE1). The largest increases include
  proliferation (TOP2A, TPX2, CKS2), matrix remodeling (MMP11) and immune chemokines (CXCL9, CXCL10).
- Enrichment ({pws['n_up_genes']} up / {pws['n_down_genes']} down genes) shows mitotic cell cycle among genes up in
  tumor and adipogenesis/PPAR signaling, angiogenesis and ECM organization among genes down in tumor.
""")
    st.subheader("3 · Predictive performance")
    lines = []
    for m, s in summ["models"].items():
        ex = s["patient_level_exact_ci"]
        lines.append(f"- **{m.replace('_', ' ')}** — ROC-AUC {s['fold']['roc_auc']['mean']:.3f} ± "
                     f"{s['fold']['roc_auc']['sd']:.3f}, balanced accuracy {s['fold']['balanced_accuracy']['mean']:.3f}; "
                     f"{ex['patients_fully_correct']}/{ex['n_patients']} patients fully correct "
                     f"(exact 95% CI {ex['ci95'][0]:.2f}–{ex['ci95'][1]:.2f}).")
    st.markdown("\n".join(lines))
    note("All three real classifiers perform similarly; the differences between them are within fold-to-fold noise. "
         "This dataset cannot tell us which algorithm is 'better', and the dummy baseline (AUC 0.5) confirms the "
         "evaluation is not trivially inflated by class imbalance.")

st.subheader("4 · Candidate biological associations")
st.markdown("""
**Most consistently used by the models** (high across LR coefficients, held-out permutation importance and selection
stability): *INHBA* (activin A; up in tumor, associated with cancer-associated fibroblasts), *C2orf40* / *ECRG4*
(a proposed tumor suppressor; down), *DMD*, *FIGF* / *VEGFD* (lymphangiogenic growth factor; down), *TFPI*,
*SDPR* / *CAVIN2* (caveolae; down) and *ADH1B* (down; abundant in adipose and stroma). The models favor genes
with very large, very consistent within-patient changes, and many of them mark stromal or adipose tissue.

**Largest fold changes** come from proliferation (*TOP2A*, *CKS2*, *TPX2*), stromal remodeling (*MMP11*,
*COL11A1*, *CST1*), immune chemokines (*CXCL9*, *CXCL10*) and adipocyte genes (*FABP4*, *ADIPOQ*, *CD36*).

These are **hypotheses and context, not discoveries**: their ranking shows association with tissue type in this
cohort. Many have prior literature in breast cancer, which is consistent with them, but they have not been
validated here.
""")

st.subheader("5 · Limitations")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
**Small sample size.** 25 independent patients. Fold-level metrics rest on 5 patients each, and exact patient-level
intervals stay wide even with zero errors.

**Overfitting risk.** ~18,000 features vs. 50 samples. We mitigate it with nested CV, in-fold feature selection
and regularization, but hyperparameter choices are often ties, and importance rankings are unstable among
correlated genes.

**Adjacent-normal comparison.** "Normal" tissue is taken next to a tumor. It can carry field-cancerization effects
and differs from healthy-donor tissue. Results do **not** generalize to screening or to healthy populations.
""")
with c2:
    st.markdown("""
**Tissue composition confounding.** Adjacent-normal breast tissue is fat-rich; tumors are epithelium- and cell-dense.
Much of the signal is composition (adipocytes vs. proliferating cells), not necessarily malignant transformation.

**Other confounders.** Single center, single platform and a single ancestry-specific cohort (Taiwan). There is no
subtype, receptor-status or tumor-purity information, and batch/processing order is unknown.

**Annotation.** Probe → gene mapping uses the platform's first gene assignment (some transcript clusters map to
several genes), and some symbols are outdated aliases (e.g. *FIGF* = *VEGFD*).
""")

st.subheader("6 · Future directions")
st.markdown("""
- **External validation** on independent cohorts (e.g. other GEO breast tumor/normal series, TCGA-BRCA), with
  platform-aware gene alignment and batch correction fitted without test labels.
- **Harder, clinically meaningful tasks**: molecular subtype, receptor status or prognosis instead of tumor vs. normal.
- **Composition-aware analysis**: cell-type deconvolution, or adjusting DE for adipocyte/stromal content.
- **Moderated statistics** (limma-style empirical Bayes) and gene-set tests that account for inter-gene correlation.
- Explanation methods such as SHAP, with explicit handling of correlated features.
""")
st.info(DISCLAIMER, icon="ℹ️")
