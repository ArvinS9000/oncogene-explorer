import json

import pandas as pd
import streamlit as st

from components import data
from components.ui import download_csv, guarded, note, page_setup
from oncogene.visualization.plots import (fig_correlation_heatmap, fig_expression_histogram,
                                          fig_pca, fig_sample_boxplots)

page_setup("Dataset explorer", "Samples, patients and expression values",
           "Browse the validated sample metadata and expression matrix, filter by tissue or patient, "
           "and export what you see.")

with guarded("the dataset explorer"):
    meta = data.samples()
    X = data.expression()
    ann = data.annotation()

    f1, f2, f3 = st.columns([1, 1.4, 1.4])
    tissues = f1.multiselect("Tissue type", ["Adjacent normal", "Tumor"], default=["Adjacent normal", "Tumor"])
    patients = f2.multiselect("Patients", sorted(meta["patient_id"].unique()),
                              placeholder="All patients (select to filter)")
    stages = f3.multiselect("Cancer stage", sorted(meta["cancer_stage"].dropna().unique()),
                            placeholder="All stages")
    m = meta[meta["tissue_type"].isin(tissues)]
    if patients:
        m = m[m["patient_id"].isin(patients)]
    if stages:
        m = m[m["cancer_stage"].isin(stages)]
    if m.empty:
        st.warning("No samples match these filters.")
        st.stop()

    tab_meta, tab_expr, tab_qc, tab_struct, tab_val = st.tabs(
        ["Sample metadata", "Expression matrix", "Distributions", "Sample structure", "Validation & provenance"])

    with tab_meta:
        q = st.text_input("Search metadata", placeholder="e.g. GSM2934490, 2102, IIIA")
        mv = m.reset_index()
        if q:
            mv = mv[mv.astype(str).apply(lambda r: r.str.contains(q, case=False, regex=False)).any(axis=1)]
        st.dataframe(mv, width="stretch", hide_index=True,
                     column_config={"age_years": st.column_config.NumberColumn("age", format="%d")})
        st.caption(f"{len(mv)} of {len(meta)} samples")
        download_csv(mv.set_index("sample_id"), "samples_filtered.csv")

    with tab_expr:
        c1, c2 = st.columns([2, 1])
        gq = c1.text_input("Filter probes by gene symbol (comma-separated, blank = first N probes)",
                           placeholder="ESR1, ERBB2, MKI67")
        n = c2.slider("Probes shown", 10, 500, 50, step=10)
        if gq.strip():
            syms = [s.strip().upper() for s in gq.split(",") if s.strip()]
            probes = ann.index[ann["gene_symbol"].isin(syms)].intersection(X.columns)
            if len(probes) == 0:
                st.info("No probes match those symbols.")
        else:
            probes = X.columns[:n]
        if len(probes):
            sub = X.loc[m.index, probes]
            sub.columns = [f"{ann.loc[p, 'gene_symbol'] or '—'} ({p})" for p in probes]
            st.dataframe(sub.round(3), width="stretch", height=420)
            st.caption(f"{sub.shape[0]} samples × {sub.shape[1]} probes · log2 RMA")
            download_csv(sub, "expression_filtered.csv")

    with tab_qc:
        rep = data.report_json("preprocessing_report.json")
        c = st.columns(4)
        c[0].metric("Missing values", rep["final_summary"]["missing_values"])
        c[1].metric("Value range (log2)", f"{rep['final_summary']['min']:.1f} – {rep['final_summary']['max']:.1f}")
        c[2].metric("Probes removed (label-free)", sum(f["removed"] for f in rep["filters"]))
        c[3].metric("Near-constant probes kept", rep["near_constant_retained"]["count"])
        st.plotly_chart(fig_expression_histogram(X.loc[m.index], m), width="stretch")
        summ = data.metric_csv("per_sample_summary.csv", index_col=0)
        st.plotly_chart(fig_sample_boxplots(summ.loc[m.index]), width="stretch")
        note("Per-sample distributions are tightly aligned, as expected after RMA quantile normalization. "
             "No sample shows a shifted or compressed scale.")

    with tab_struct:
        pca = data.metric_csv("pca_scores.csv", index_col=0)
        evr = data.metric_json("pca_variance.json")["explained_variance_ratio"]
        c1, c2, c3 = st.columns([1, 1, 2])
        pcx = c1.selectbox("X axis", [f"PC{i}" for i in range(1, 6)], index=0)
        pcy = c2.selectbox("Y axis", [f"PC{i}" for i in range(1, 6)], index=1)
        pairs = c3.toggle("Connect each patient's tumor–normal pair", value=True)
        st.plotly_chart(fig_pca(pca.loc[m.index], evr, pairs, pcx, pcy), width="stretch")
        note("<b>Exploratory only.</b> This PCA uses all 50 samples (top 2,000 most variable probes) to "
             "visualize structure. It is never used as model input; classifiers refit every transform inside "
             "each training fold.")
        corr = data.metric_csv("sample_correlation.csv", index_col=0)
        order = [s for s in data.metric_json("sample_cluster_order.json") if s in m.index]
        st.plotly_chart(fig_correlation_heatmap(corr, order, meta), width="stretch")
        st.markdown("**Outlier screen** — robust z-score of each sample's mean correlation with its own tissue group")
        out = data.metric_csv("outlier_screen.csv", index_col=0).loc[lambda d: d.index.isin(m.index)]
        st.dataframe(out.sort_values("robust_z").round(3), width="stretch", height=260)
        note("Flags are informational; <b>no sample is removed</b>. Flagged samples are mostly tumors, which "
             "are biologically heterogeneous (subtype, tumor purity), plus normals with unusual composition.", warn=True)
        pdist = data.metric_csv("pair_distances.csv")
        st.markdown("**Tumor–normal distance per patient** (1 − Pearson r on top-variable probes)")
        st.dataframe(pdist.round(3), width="stretch", hide_index=True, height=260)

    with tab_val:
        val = pd.DataFrame(data.report_json("validation_report.json"))
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
        val.insert(0, "", val["status"].map(icon))
        st.dataframe(val[["", "name", "status", "detail"]], width="stretch", hide_index=True)
        st.markdown("**Label-free filtering decisions**")
        st.json(rep["filters"], expanded=False)
        st.markdown("**Annotation**")
        st.json(rep["annotation"], expanded=False)
        try:
            prov = json.loads((data.ROOT / "data" / "raw" / "provenance.json").read_text())
            st.markdown("**Download provenance** (source URLs and SHA-256 checksums)")
            st.dataframe(pd.DataFrame(prov).T[["url", "sha256", "bytes", "retrieved_utc"]], width="stretch")
        except FileNotFoundError:
            st.caption("Provenance file not present (raw data not downloaded on this machine).")
