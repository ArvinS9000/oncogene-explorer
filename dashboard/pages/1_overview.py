import plotly.graph_objects as go
import streamlit as st

from components import data
from components.ui import DISCLAIMER, guarded, note, page_setup, tissue_legend
from oncogene.visualization.plots import TISSUE_COLORS, _style, fig_class_distribution

page_setup(
    "Overview",
    "OncoGene Explorer",
    "An end-to-end, leakage-safe analysis of breast-cancer gene expression: how do breast tumors "
    "differ from adjacent normal tissue taken from the same patient, and which genes carry that signal?",
)
st.info(DISCLAIMER, icon="ℹ️")

with guarded("the dataset summary"):
    ov = data.metric_json("eda_overview.json")
    c = st.columns(5)
    c[0].metric("Samples", ov["n_samples"])
    c[1].metric("Patients (matched pairs)", ov["n_patients"])
    c[2].metric("Tumor samples", ov["n_tumor"])
    c[3].metric("Adjacent-normal samples", ov["n_normal"])
    c[4].metric("Expression features", f"{ov['n_features']:,}")

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Research question")
    st.markdown(
        "Using 25 patient-matched pairs of **breast tumor** and **adjacent normal** tissue, we ask:\n\n"
        "1. Which genes are consistently expressed differently between a patient's tumor and their own "
        "adjacent normal tissue? *(paired differential expression)*\n"
        "2. Can supervised models distinguish the two tissue types for **patients they have never seen**? "
        "*(nested, patient-grouped cross-validation)*\n"
        "3. Which genes and pathways drive these differences, and what do they suggest biologically? "
        "*(interpretation + pathway enrichment)*")
    st.subheader("Dataset")
    st.markdown(
        "- **Source:** NCBI GEO [GSE109169](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109169) "
        "(Chang et al., Taiwan; PMID [29706651](https://pubmed.ncbi.nlm.nih.gov/29706651/))\n"
        "- **Platform:** GPL5175, Affymetrix Human Exon 1.0 ST, core transcript clusters\n"
        "- **Values:** RMA-normalized, already log2 — no further log transform applied\n"
        "- **Comparison:** tumor vs. *adjacent* normal from the same patient — **not** healthy-donor tissue")
with right:
    with guarded("the class distribution"):
        st.plotly_chart(fig_class_distribution(data.samples()), use_container_width=True)

st.divider()
st.subheader("Major findings at a glance")
with guarded("the findings summary"):
    de = data.metric_csv("de_results.csv", index_col="probe_id")
    summ = data.metric_json("model_summary.json")
    sig = de[de["significant"]]
    lr = summ["models"].get("logistic_regression", {})
    ex = lr.get("patient_level_exact_ci", {})
    c = st.columns(3)
    c[0].metric("Probes differentially expressed", f"{len(sig):,}",
                help="BH-FDR < 0.05 and |log2 fold change| ≥ 1, paired t-test")
    c[1].metric("Up / down in tumor", f"{(sig.log2_fold_change > 0).sum()} / {(sig.log2_fold_change < 0).sum()}")
    if lr:
        c[2].metric("Logistic regression held-out ROC-AUC",
                    f"{lr['fold']['roc_auc']['mean']:.3f}",
                    help="Mean over outer folds of nested, patient-grouped CV")
    st.markdown(
        "- **Proliferation up, adipose/stroma down.** Genes higher in tumors are dominated by cell-cycle and "
        "mitosis genes (e.g. *TOP2A*, *TPX2*, *CKS2*); genes lower in tumors are dominated by adipocyte and "
        "stromal genes (e.g. *FABP4*, *ADIPOQ*, *CD36*).\n"
        "- **The classes are almost perfectly separable**, even with very few training patients. That reflects "
        "large tissue-composition differences, not a diagnostic test (see *Findings & limitations*).\n"
        "- **Evidence ≠ causation.** Rankings show association with tissue type, not that a gene causes cancer.")

with guarded("the composition scatter"):
    ms = data.metric_csv("marker_scores.csv", index_col=0)
    st.subheader("What separates the tissues? Composition, mostly")
    st.markdown(tissue_legend(), unsafe_allow_html=True)
    fig = go.Figure()
    for tissue, color in TISSUE_COLORS.items():
        g = ms[ms["tissue_type"] == tissue]
        fig.add_trace(go.Scatter(x=g["adipocyte_score"], y=g["proliferation_score"], mode="markers",
                                 name=tissue, marker=dict(color=color, size=11, line=dict(color="#fcfcfb", width=2)),
                                 customdata=g[["patient_id"]].assign(s=g.index).values,
                                 hovertemplate="%{customdata[1]} · patient %{customdata[0]}<br>adipocyte %{x:.2f}"
                                               "<br>proliferation %{y:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Adipocyte marker score (mean z: ADIPOQ, PLIN1, LEP, FABP4, …)",
                      yaxis_title="Proliferation marker score (mean z: MKI67, TOP2A, CCNB1, …)")
    st.plotly_chart(_style(fig, None, 420), use_container_width=True)
    note("Marker panels are fixed literature lists, not chosen from these labels. Adjacent-normal breast is "
         "fat-rich, and tumors are cell-dense and proliferative, so two simple composition scores already "
         "separate most samples. Any classifier trained on this contrast will exploit that.")
