import streamlit as st

from components import data
from components.ui import download_csv, guarded, note, page_setup
from oncogene.visualization.plots import fig_pathway_bar, fig_pathway_dot

page_setup("Pathways", "Exploratory pathway enrichment",
           "Over-representation analysis of differentially expressed genes in Reactome pathways and Gene "
           "Ontology biological processes, tested against the genes actually measured on the array.")

with guarded("pathway results"):
    pw = data.metric_csv("pathway_results.csv")
    s = data.metric_json("pathway_summary.json")
    c = st.columns(4)
    c[0].metric("Background universe", f"{s['universe_size']:,} genes")
    c[1].metric("DE genes up in tumor", s["n_up_genes"])
    c[2].metric("DE genes down in tumor", s["n_down_genes"])
    c[3].metric("Terms with FDR < 0.05", int((pw["p_adj"] < 0.05).sum()))
    if s.get("limitation"):
        st.warning(s["limitation"])

    f1, f2, f3, f4 = st.columns([1.3, 1.1, 1, 1])
    libs = f1.multiselect("Library", sorted(pw["library"].unique()), default=sorted(pw["library"].unique()),
                          format_func=lambda s: s.replace("_", " "))
    direction = f2.radio("Direction", ["up in tumor", "down in tumor", "both"], horizontal=True)
    fdr = f3.select_slider("FDR cut-off", [0.001, 0.01, 0.05, 0.1, 0.25], value=0.05)
    n = f4.slider("Terms shown", 5, 40, 15)
    d = pw[pw["library"].isin(libs) & (pw["p_adj"] < fdr)]
    if direction != "both":
        d = d[d["direction"] == direction]
    if d.empty:
        st.info("No terms pass these filters.")
        st.stop()

    t1, t2, t3 = st.tabs(["Bar chart", "Dot plot", "Table & genes"])
    with t1:
        st.plotly_chart(fig_pathway_bar(d, n), use_container_width=True)
    with t2:
        st.plotly_chart(fig_pathway_dot(d, n), use_container_width=True)
    with t3:
        q = st.text_input("Search terms or genes", placeholder="e.g. cell cycle, PPARG")
        show = d if not q else d[d["term_name"].str.contains(q, case=False, regex=False) |
                                 d["genes"].str.contains(q.upper(), regex=False)]
        st.dataframe(show[["library", "direction", "term_name", "term_id", "overlap", "set_size_in_universe",
                           "expected", "fold_enrichment", "p_value", "p_adj", "genes", "term_url"]],
                     use_container_width=True, hide_index=True, height=420,
                     column_config={"p_value": st.column_config.NumberColumn(format="%.2e"),
                                    "p_adj": st.column_config.NumberColumn("FDR", format="%.2e"),
                                    "expected": st.column_config.NumberColumn(format="%.1f"),
                                    "fold_enrichment": st.column_config.NumberColumn(format="%.1f"),
                                    "term_url": st.column_config.LinkColumn("reference", display_text="open")})
        download_csv(show.set_index("term"), "pathway_enrichment.csv")
        term = st.selectbox("Gene–pathway associations for term", show["term_name"].head(100))
        row = show[show["term_name"] == term].iloc[0]
        st.markdown(f"**{term}** ({row['library']}, {row['direction']}) — {row['overlap']} of "
                    f"{row['set_size_in_universe']} measured genes in the set are DE.")
        st.code(", ".join(row["genes"].split(";")), language=None)

st.subheader("Interpretation notes")
note("""<b>Genes up in tumor</b> are strongly enriched for mitotic cell-cycle processes (chromatid segregation, spindle
checkpoint, Reactome 'Cell Cycle, Mitotic'), which is consistent with proliferating tumor cells.
<b>Genes down in tumor</b> point to adipocyte biology (white-adipocyte differentiation, PPAR/nuclear-receptor
signaling, thermogenesis), angiogenesis and extracellular-matrix organization, which reflects the fat- and stroma-rich
composition of adjacent normal breast tissue.""")
note("""<b>Method & caveats.</b> One-sided hypergeometric test per term, run separately for up- and down-regulated
genes, with Benjamini–Hochberg correction per library and direction. Gene sets come from Enrichr's Reactome 2022 and
GO Biological Process 2023 libraries, intersected with the array universe (terms with 10–500 measured genes).
KEGG is not used because of its licensing terms. ORA ignores gene–gene correlation and depends on arbitrary DE
thresholds; overlapping GO terms are not independent. Probes were collapsed to gene symbols, and genes with probes
changing in opposite directions were excluded.""", warn=True)
