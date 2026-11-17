import numpy as np
import pandas as pd
import streamlit as st

from components import data
from components.ui import (download_csv, fmt_p, gene_links, guarded, note, page_setup,
                           render_links)
from oncogene.visualization.plots import (fig_expression_heatmap, fig_gene_paired,
                                          fig_volcano)

page_setup("Gene expression", "Explore individual genes and differential expression",
           "Search any gene, compare each patient's tumor with their own adjacent normal tissue, and "
           "browse the paired differential-expression results.")

with guarded("the gene explorer"):
    X, meta, ann = data.expression(), data.samples(), data.annotation()
    de = data.metric_csv("de_results.csv", index_col="probe_id")
    opts = data.gene_search_options()

    tab_gene, tab_volcano, tab_table, tab_heat = st.tabs(
        ["Gene lookup", "Volcano plot", "DE results table", "Heatmap"])

    with tab_gene:
        default = int(np.flatnonzero(opts["label"].str.startswith("TOP2A"))[0]) if opts["label"].str.startswith("TOP2A").any() else 0
        choice = st.selectbox("Search by gene symbol or probe ID (type to filter)", opts["label"].tolist(),
                              index=default)
        probe = opts.index[opts["label"] == choice][0]
        a, r = ann.loc[probe], de.loc[probe] if probe in de.index else None
        sym = a["gene_symbol"] if isinstance(a["gene_symbol"], str) else None
        st.markdown(f"### {sym or probe}")
        st.caption(f"{a.get('gene_description') or 'No description in platform annotation'} · probe {probe} · "
                   f"{a.get('cytoband') or ''} · annotation: GEO GPL5175 (Affymetrix)")
        render_links(gene_links(sym, a.get("entrez_id")))
        if r is not None:
            c = st.columns(5)
            c[0].metric("log2 fold change", f"{r['log2_fold_change']:+.2f}", help="mean(tumor − paired normal)")
            c[1].metric("Fold change", f"{r['fold_change']:.2f}×")
            c[2].metric("FDR (paired t)", fmt_p(r["p_adj"]))
            c[3].metric("FDR (Wilcoxon)", fmt_p(r.get("wilcoxon_p_adj", np.nan)))
            c[4].metric("Pairs higher in tumor", f"{r['frac_pairs_up']:.0%}")
        st.plotly_chart(fig_gene_paired(X, meta, probe, sym), use_container_width=True)
        with st.expander("Gene-level summary statistics"):
            s = X[probe].groupby(meta["tissue_type"]).describe().T.round(3)
            st.dataframe(s, use_container_width=True)
            other = ann.index[(ann["gene_symbol"] == sym) & (ann.index != probe)] if sym else []
            if len(other):
                st.caption(f"Other probes mapped to {sym}: {', '.join(other)}")

    with tab_volcano:
        c1, c2 = st.columns(2)
        fdr = c1.select_slider("FDR threshold", [0.001, 0.01, 0.05, 0.1], value=0.05)
        lfc = c2.slider("|log2 fold change| threshold", 0.0, 3.0, 1.0, 0.25)
        st.plotly_chart(fig_volcano(de, fdr, lfc), use_container_width=True)
        n_up = ((de.p_adj < fdr) & (de.log2_fold_change >= lfc)).sum()
        n_dn = ((de.p_adj < fdr) & (de.log2_fold_change <= -lfc)).sum()
        st.caption(f"{n_up} probes up and {n_dn} down in tumor at FDR < {fdr} and |log2FC| ≥ {lfc}.")
        note("Effect size is the mean within-patient log2 difference; because RMA values are already log2, this is a "
             "valid log2 fold change. Points are probes (transcript clusters); a gene can appear more than once.")

    with tab_table:
        c1, c2, c3 = st.columns([2, 1, 1])
        q = c1.text_input("Filter by symbol or description", placeholder="e.g. KRT, collagen, ESR1")
        only_sig = c2.toggle("Significant only", value=True)
        direction = c3.selectbox("Direction", ["Both", "up in tumor", "down in tumor"])
        t = de.copy()
        if only_sig:
            t = t[t["significant"]]
        if direction != "Both":
            t = t[t["direction"] == direction]
        if q:
            mask = t["gene_symbol"].fillna("").str.contains(q, case=False, regex=False) | \
                   t["gene_description"].fillna("").str.contains(q, case=False, regex=False)
            t = t[mask]
        cols = ["gene_symbol", "gene_description", "log2_fold_change", "direction", "cohens_dz", "p_value",
                "p_adj", "wilcoxon_p_adj", "significant", "mean_tumor", "mean_normal", "frac_pairs_up"]
        st.dataframe(t[cols], use_container_width=True, height=480,
                     column_config={"p_value": st.column_config.NumberColumn(format="%.2e"),
                                    "p_adj": st.column_config.NumberColumn("FDR", format="%.2e"),
                                    "wilcoxon_p_adj": st.column_config.NumberColumn("Wilcoxon FDR", format="%.2e"),
                                    "log2_fold_change": st.column_config.NumberColumn("log2FC", format="%.2f"),
                                    "cohens_dz": st.column_config.NumberColumn("Cohen's dz", format="%.2f"),
                                    "frac_pairs_up": st.column_config.ProgressColumn("pairs up", min_value=0, max_value=1)})
        st.caption(f"{len(t):,} probes")
        download_csv(t[cols], "differential_expression.csv")
        note("Method: paired t-test on within-patient differences, Benjamini–Hochberg FDR across all probes; the "
             "Wilcoxon signed-rank test is a distribution-free sensitivity check (its smallest attainable p-value is "
             "limited with 25 pairs, so many top genes tie).")

    with tab_heat:
        c1, c2, c3 = st.columns([1, 1, 2])
        n = c1.slider("Genes shown", 10, 100, 40, 5)
        how = c2.radio("Gene selection", ["Top DE (|log2FC|)", "Most variable", "Custom list"])
        custom = c3.text_input("Custom symbols", placeholder="ESR1, PGR, ERBB2, KRT5, MKI67",
                               disabled=how != "Custom list")
        if how == "Top DE (|log2FC|)":
            probes = de[de["significant"]].assign(a=lambda d: d.log2_fold_change.abs()).nlargest(n, "a").index
        elif how == "Most variable":
            probes = X.var().nlargest(n).index
        else:
            syms = [s.strip().upper() for s in custom.split(",") if s.strip()]
            probes = ann.index[ann["gene_symbol"].isin(syms)].intersection(X.columns)
        if len(probes) < 2:
            st.info("Select at least two genes.")
        else:
            labels = [ann.loc[p, "gene_symbol"] if isinstance(ann.loc[p, "gene_symbol"], str) else p for p in probes]
            st.plotly_chart(fig_expression_heatmap(X, meta, list(probes), labels), use_container_width=True)
            note("Rows are z-scored per gene; rows and columns are hierarchically clustered (average linkage, "
                 "correlation distance). <b>Descriptive only:</b> 'Top DE' genes were chosen using all samples' "
                 "labels, so clean separation here is expected and is <i>not</i> evidence of predictive generalization.",
                 warn=how == "Top DE (|log2FC|)")
