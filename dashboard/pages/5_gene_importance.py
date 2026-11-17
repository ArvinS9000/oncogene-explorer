import numpy as np
import pandas as pd
import streamlit as st

from components import data
from components.ui import (download_csv, fmt_p, gene_links, guarded, note, page_setup,
                           render_links)
from oncogene.visualization.plots import (MODEL_COLORS, fig_gene_paired, fig_importance_bars,
                                          fig_signed_bars)

page_setup("Interpretability", "Which genes do the models rely on?",
           "Several complementary importance measures, each with its own caveats. Read them together, "
           "and as associations — not as evidence that a gene causes cancer.")

with guarded("gene importance"):
    rank = data.metric_csv("gene_ranking.csv")
    X, meta, ann = data.expression(), data.samples(), data.annotation()
    de = data.metric_csv("de_results.csv", index_col="probe_id")
    rank["label"] = rank["gene_symbol"].fillna(rank["probe_id"])

    methods = list(rank["method"].unique())
    default = next((i for i, m in enumerate(methods) if m.startswith("LR standardized")), 0)
    method = st.selectbox("Importance method", methods, index=default)
    sub = rank[rank["method"] == method].copy()
    n = st.slider("Genes shown", 10, 60, 25, 5)

    if method.startswith("LR standardized"):
        st.plotly_chart(fig_signed_bars(sub, "signed_value", "label",
                                        "Logistic regression: standardized coefficients", n),
                        width="stretch")
        note("Inputs are standardized inside the pipeline, so coefficients are per-SD effects on the log-odds of "
             "'tumor'. L2 regularization spreads weight across correlated genes, so a small coefficient does not "
             "mean a gene is unimportant, and the sign reflects the model's use of that gene given all others.")
    elif method.startswith("RF impurity"):
        st.plotly_chart(fig_importance_bars(sub, "importance", "label", "Random forest: impurity importance", n,
                                            MODEL_COLORS["random_forest"]), width="stretch")
        note("Impurity (Gini) importance is computed on training data, is biased toward high-variance features, "
             "and is split arbitrarily between correlated genes. Treat it as secondary to held-out measures.", warn=True)
    elif method.startswith("Permutation"):
        st.plotly_chart(fig_importance_bars(sub, "importance", "label",
                                            "Permutation importance on held-out patients (relative margin drop)", n),
                        width="stretch")
        note("Computed separately in every outer validation fold, for the genes that fold's model used, then averaged "
             "(folds where a gene was not selected count as 0). Score = relative drop in the mean signed margin "
             "toward the correct class. The drop in held-out <b>ROC-AUC</b> was 0 for every single gene: the signal is "
             "so redundant that no one gene is necessary for correct ranking.")
    else:
        st.plotly_chart(fig_importance_bars(sub, "importance", "label",
                                            "Fraction of outer folds in which the gene was selected", n),
                        width="stretch")
        note("Selection stability: how often the in-fold ANOVA k-best step picked the gene across outer training sets. "
             "Stable selection suggests a robust univariate difference, not causal relevance.")

    st.subheader("Ranked table")
    q = st.text_input("Search genes in this ranking", placeholder="e.g. ADIPOQ")
    show = sub if not q else sub[sub["label"].str.contains(q, case=False, regex=False)]
    cols = ["rank_within_method", "label", "probe_id", "gene_description", "importance", "signed_value",
            "log2_fold_change", "p_adj", "de_direction", "significant"]
    st.dataframe(show[cols].rename(columns={"label": "gene", "p_adj": "DE FDR"}), width="stretch",
                 hide_index=True, height=380,
                 column_config={"DE FDR": st.column_config.NumberColumn(format="%.2e"),
                                "importance": st.column_config.NumberColumn(format="%.4f"),
                                "signed_value": st.column_config.NumberColumn("signed coef.", format="%.4f"),
                                "log2_fold_change": st.column_config.NumberColumn("log2FC", format="%.2f")})
    download_csv(show[cols].set_index("probe_id"), "gene_ranking.csv")

    st.divider()
    st.subheader("Gene detail")
    top_labels = sub.nsmallest(60, "rank_within_method")
    choice = st.selectbox("Choose a ranked gene", top_labels["label"] + "  ·  probe " + top_labels["probe_id"])
    probe = choice.split("probe ")[-1]
    a = ann.loc[probe]
    sym = a["gene_symbol"] if isinstance(a["gene_symbol"], str) else None
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown(f"### {sym or probe}")
        st.caption(f"{a.get('gene_description') or 'No description available'}  \n"
                   f"Cytoband {a.get('cytoband') or 'n/a'} · probe {probe} · source: GPL5175 annotation (GEO)")
        render_links(gene_links(sym, a.get("entrez_id")))
        if probe in de.index:
            r = de.loc[probe]
            st.markdown(f"**Differential expression:** log2FC {r['log2_fold_change']:+.2f}, "
                        f"FDR {fmt_p(r['p_adj'])}, higher in tumor in {r['frac_pairs_up']:.0%} of pairs.")
        all_m = rank[rank["probe_id"] == probe][["method", "importance", "rank_within_method"]]
        st.markdown("**Across importance methods**")
        st.dataframe(all_m, hide_index=True, width="stretch")
    with c2:
        st.plotly_chart(fig_gene_paired(X, meta, probe, sym), width="stretch")
    note("High model importance means a gene's expression helps <i>distinguish tumor from adjacent normal tissue in "
         "this cohort</i>. Many top genes reflect tissue composition (fat, stroma, proliferating cells). Establishing a "
         "causal role in cancer requires experimental evidence beyond expression data.", warn=True)
