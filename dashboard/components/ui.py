"""Shared UI helpers: styling, disclaimers, metric tiles, downloads, gene links."""

from __future__ import annotations

from contextlib import contextmanager
from urllib.parse import quote

import pandas as pd
import streamlit as st

from components.data import MissingArtifact

CSS = """
<style>
  :root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --surface:#fcfcfb;
          --normal:#2a78d6; --tumor:#eb6834; }
  .block-container { padding-top: 2.2rem; max-width: 1280px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  .og-eyebrow { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted);
                margin-bottom: .1rem; font-weight: 600; }
  .og-lede { color: var(--ink2); font-size: 1.02rem; max-width: 70ch; margin-top: -.4rem; }
  .og-note { border-left: 3px solid var(--line); padding: .5rem .9rem; color: var(--ink2);
             font-size: .92rem; background: var(--surface); border-radius: 0 6px 6px 0; margin: .4rem 0 1rem; }
  .og-warn { border-left-color: #eda100; }
  .og-chip { display:inline-block; padding: 0 .5rem; border-radius: 999px; font-size: .8rem;
             border: 1px solid var(--line); color: var(--ink2); margin-right: .3rem; }
  .og-dot { display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.35rem;
            vertical-align: -1px; }
  div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line);
             border-radius: 10px; padding: .7rem .9rem; }
  div[data-testid="stMetricLabel"] p { color: var(--ink2); font-size: .82rem; }
</style>
"""

DISCLAIMER = ("Research and education only. Results describe one public cohort (25 patients) and "
              "are not a clinically validated diagnostic, screening or treatment tool.")


def page_setup(eyebrow: str, title: str, lede: str | None = None) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<div class='og-eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.title(title)
    if lede:
        st.markdown(f"<p class='og-lede'>{lede}</p>", unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f"<div class='og-note{' og-warn' if warn else ''}'>{text}</div>", unsafe_allow_html=True)


def tissue_legend() -> str:
    return ("<span class='og-dot' style='background:#2a78d6'></span>Adjacent normal&nbsp;&nbsp;"
            "<span class='og-dot' style='background:#eb6834'></span>Tumor")


@contextmanager
def guarded(what: str = "this section"):
    """Render a helpful message instead of a stack trace when artifacts are missing."""
    try:
        yield
    except MissingArtifact as e:
        st.warning(f"Required artifact `{e}` was not found, so {what} cannot be shown. "
                   "Generate it with `python -m oncogene.cli all` (see README).", icon="⚠️")
    except Exception as e:  # noqa: BLE001 - surface the error but keep the page alive
        st.error(f"Could not render {what}: {type(e).__name__}: {e}")


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=True).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key or f"dl_{filename}")


def gene_links(symbol: str | None, entrez: str | None) -> dict[str, str]:
    links = {}
    if entrez and str(entrez) not in ("nan", "None", ""):
        links["NCBI Gene"] = f"https://www.ncbi.nlm.nih.gov/gene/{quote(str(entrez).split('.')[0])}"
    if symbol:
        s = quote(symbol)
        links["HGNC"] = f"https://www.genenames.org/tools/search/#!/?query={s}"
        links["UniProt"] = f"https://www.uniprot.org/uniprotkb?query=gene_exact:{s}+AND+organism_id:9606"
        links["Human Protein Atlas"] = f"https://www.proteinatlas.org/search/{s}"
        links["Ensembl"] = f"https://www.ensembl.org/Homo_sapiens/Search/Results?q={s}"
        links["PubMed (+ breast cancer)"] = f"https://pubmed.ncbi.nlm.nih.gov/?term={s}%5BTitle%2FAbstract%5D+AND+breast+cancer"
    return links


def render_links(links: dict[str, str]) -> None:
    if links:
        st.markdown(" · ".join(f"[{k}]({v})" for k, v in links.items()))


def fmt_p(p: float) -> str:
    return "n/a" if pd.isna(p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")
