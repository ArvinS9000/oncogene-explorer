"""OncoGene Explorer - Streamlit entry point.

Run from the project root:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="OncoGene Explorer", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

pages = [
    st.Page("pages/1_overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("pages/2_dataset_explorer.py", url_path="dataset_explorer", title="Dataset explorer", icon=":material/table:"),
    st.Page("pages/3_gene_expression.py", url_path="gene_expression", title="Gene expression", icon=":material/genetics:"),
    st.Page("pages/4_models.py", url_path="models", title="Machine learning models", icon=":material/model_training:"),
    st.Page("pages/5_gene_importance.py", url_path="gene_importance", title="Gene importance", icon=":material/leaderboard:"),
    st.Page("pages/6_pathways.py", url_path="pathways", title="Biological pathways", icon=":material/hub:"),
    st.Page("pages/7_findings.py", url_path="findings", title="Findings & limitations", icon=":material/fact_check:"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🧬 OncoGene Explorer")
    st.caption("Breast tumor vs. paired adjacent-normal tissue · GEO GSE109169")
    st.divider()
    st.caption("Research & education only — not a diagnostic tool.")

nav.run()
