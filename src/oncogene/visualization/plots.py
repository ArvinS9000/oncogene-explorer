"""Plotly figure builders (dashboard) and matplotlib exports (static report figures).

Color roles (validated for color-vision deficiency; see docs/methodology.md):
  tissue    : adjacent normal = blue, tumor = orange
  models    : LR = blue, RF = orange, SVM = aqua, dummy = gray dashed baseline
  diverging : blue <-> red with a neutral gray midpoint (z-scores, signed coefficients)
  sequential: single-hue blue ramp (correlations, magnitudes)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

# ----------------------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NORMAL_C, TUMOR_C = "#2a78d6", "#eb6834"
TISSUE_COLORS = {"Adjacent normal": NORMAL_C, "Tumor": TUMOR_C}
MODEL_COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834",
                "svm": "#1baf7a", "dummy": MUTED}
MODEL_LABELS = {"dummy": "Dummy (prior)", "logistic_regression": "Logistic regression",
                "random_forest": "Random forest", "svm": "SVM"}
NS_COLOR = "#c3c2b7"
DIVERGING = [[0.0, "#104281"], [0.25, "#5598e7"], [0.5, "#f0efec"], [0.75, "#e66767"], [1.0, "#a3201f"]]
SEQUENTIAL = [[0.0, "#cde2fb"], [0.33, "#6da7ec"], [0.66, "#256abf"], [1.0, "#0d366b"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _style(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", font=dict(family=FONT, size=13, color=INK_2),
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left") if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font_family=FONT), height=height,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, title_font=dict(color=INK_2))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, title_font=dict(color=INK_2))
    return fig


# ----------------------------------------------------------------------------- dataset / EDA
def fig_class_distribution(meta: pd.DataFrame) -> go.Figure:
    c = meta["tissue_type"].value_counts().reindex(["Adjacent normal", "Tumor"]).reset_index()
    c.columns = ["tissue_type", "samples"]
    fig = px.bar(c, x="tissue_type", y="samples", color="tissue_type", text="samples",
                 color_discrete_map=TISSUE_COLORS)
    fig.update_traces(marker_line_width=0, textposition="outside", width=0.5)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Samples")
    return _style(fig, "Tissue class distribution", 320)


def fig_expression_histogram(X: pd.DataFrame, meta: pd.DataFrame, bins: int = 80) -> go.Figure:
    fig = go.Figure()
    for tissue, color in TISSUE_COLORS.items():
        vals = X.loc[meta.index[meta["tissue_type"] == tissue]].values.ravel()
        fig.add_trace(go.Histogram(x=vals, name=tissue, marker_color=color, opacity=0.55,
                                   nbinsx=bins, histnorm="probability density"))
    fig.update_layout(barmode="overlay", xaxis_title="log2 expression (RMA)", yaxis_title="Density")
    return _style(fig, "Expression value distribution (all probes)", 360)


def fig_sample_boxplots(summary: pd.DataFrame) -> go.Figure:
    """Per-sample distribution from precomputed quantiles (5/25/50/75/95)."""
    s = summary.sort_values(["tissue_type", "patient_id"])
    fig = go.Figure()
    for tissue, color in TISSUE_COLORS.items():
        g = s[s["tissue_type"] == tissue]
        fig.add_trace(go.Box(x=g.index, q1=g["q25"], median=g["median"], q3=g["q75"],
                             lowerfence=g["q05"], upperfence=g["q95"], name=tissue,
                             marker_color=color, line_width=1.5, fillcolor=color, opacity=0.8,
                             customdata=g[["patient_id"]].values,
                             hovertemplate="%{x}<br>patient %{customdata[0]}<extra></extra>"))
    fig.update_layout(xaxis_title="Sample (whiskers = 5th-95th percentile)",
                      yaxis_title="log2 expression", xaxis_tickangle=-60)
    return _style(fig, "Per-sample expression distributions", 420)


def fig_pca(pca_df: pd.DataFrame, evr: np.ndarray, connect_pairs: bool = True,
            x: str = "PC1", y: str = "PC2") -> go.Figure:
    fig = go.Figure()
    if connect_pairs:
        xs, ys = [], []
        for _, g in pca_df.groupby("patient_id"):
            xs += list(g[x]) + [None]
            ys += list(g[y]) + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=AXIS, width=1),
                                 hoverinfo="skip", name="Patient pair", showlegend=True))
    for tissue, color in TISSUE_COLORS.items():
        g = pca_df[pca_df["tissue_type"] == tissue]
        fig.add_trace(go.Scatter(
            x=g[x], y=g[y], mode="markers", name=tissue,
            marker=dict(color=color, size=11, line=dict(color=SURFACE, width=2)),
            customdata=np.c_[g.index, g["patient_id"], g["cancer_stage"].fillna("")],
            hovertemplate="<b>%{customdata[0]}</b><br>patient %{customdata[1]}<br>"
                          "stage %{customdata[2]}<br>" + x + "=%{x:.1f}, " + y + "=%{y:.1f}<extra>" + tissue + "</extra>"))
    i, j = int(x[2:]) - 1, int(y[2:]) - 1
    fig.update_layout(xaxis_title=f"{x} ({evr[i]:.1%} variance)", yaxis_title=f"{y} ({evr[j]:.1%} variance)")
    return _style(fig, "Exploratory PCA (all samples, unsupervised)", 480)


def fig_expression_heatmap(X: pd.DataFrame, meta: pd.DataFrame, probes: list[str],
                           labels: list[str] | None = None, cluster: bool = True) -> go.Figure:
    """Row-z-scored heatmap of selected probes; samples and genes optionally clustered."""
    from scipy.cluster import hierarchy
    M = X[probes].T
    Z = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1).replace(0, 1), axis=0)
    row_order, col_order = list(range(Z.shape[0])), list(range(Z.shape[1]))
    if cluster and Z.shape[0] > 2:
        row_order = hierarchy.leaves_list(hierarchy.linkage(Z.values, "average", metric="correlation"))
        col_order = hierarchy.leaves_list(hierarchy.linkage(Z.values.T, "average", metric="correlation"))
    Z = Z.iloc[row_order, col_order]
    ylab = [labels[i] for i in row_order] if labels else list(Z.index)
    samples = Z.columns
    tissue = meta.loc[samples, "tissue_type"]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Z.values.clip(-3, 3), x=list(samples), y=ylab, colorscale=DIVERGING,
                             zmid=0, zmin=-3, zmax=3, colorbar=dict(title="z-score", thickness=12),
                             customdata=np.tile(tissue.values, (Z.shape[0], 1)),
                             hovertemplate="%{y}<br>%{x} (%{customdata})<br>z=%{z:.2f}<extra></extra>",
                             yaxis="y"))
    fig.add_trace(go.Heatmap(z=[[1 if t == "Tumor" else 0 for t in tissue]], x=list(samples),
                             y=["Tissue"], colorscale=[[0, NORMAL_C], [1, TUMOR_C]], showscale=False,
                             customdata=[list(tissue)], hovertemplate="%{x}: %{customdata}<extra></extra>",
                             yaxis="y2", xgap=1))
    n = Z.shape[0]
    fig.update_layout(yaxis=dict(domain=[0, 0.93], tickfont=dict(size=9 if n > 40 else 11)),
                      yaxis2=dict(domain=[0.95, 1.0], showticklabels=True),
                      xaxis=dict(showticklabels=False, title="Samples (tumor = orange bar, normal = blue bar)"))
    return _style(fig, None, max(420, 14 * n + 120))


def fig_correlation_heatmap(corr: pd.DataFrame, order: list[str], meta: pd.DataFrame) -> go.Figure:
    C = corr.loc[order, order]
    lab = [f"{s} · {meta.loc[s, 'tissue_type'][0]} · {meta.loc[s, 'patient_id']}" for s in order]
    fig = go.Figure(go.Heatmap(z=C.values, x=lab, y=lab, colorscale=SEQUENTIAL,
                               colorbar=dict(title="Pearson r", thickness=12),
                               hovertemplate="%{y}<br>%{x}<br>r=%{z:.3f}<extra></extra>"))
    fig.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(tickfont=dict(size=8), autorange="reversed"))
    return _style(fig, "Sample-sample correlation (hierarchically clustered)", 640)


# ----------------------------------------------------------------------------- DE
def fig_volcano(de: pd.DataFrame, fdr: float = 0.05, min_lfc: float = 1.0, label_top: int = 12) -> go.Figure:
    d = de.copy()
    d["cls"] = np.where((d["p_adj"] < fdr) & (d["log2_fold_change"] >= min_lfc), "Up in tumor",
                np.where((d["p_adj"] < fdr) & (d["log2_fold_change"] <= -min_lfc), "Down in tumor",
                         "Not significant"))
    d["label"] = d["gene_symbol"].fillna(d.index.to_series())
    colors = {"Not significant": NS_COLOR, "Down in tumor": NORMAL_C, "Up in tumor": TUMOR_C}
    fig = go.Figure()
    for cls in ["Not significant", "Down in tumor", "Up in tumor"]:
        g = d[d["cls"] == cls]
        fig.add_trace(go.Scattergl(
            x=g["log2_fold_change"], y=g["neg_log10_p_adj"], mode="markers", name=f"{cls} ({len(g):,})",
            marker=dict(color=colors[cls], size=5 if cls == "Not significant" else 7,
                        opacity=0.5 if cls == "Not significant" else 0.85),
            customdata=np.c_[g["label"], g.index, g["p_value"], g["p_adj"]],
            hovertemplate="<b>%{customdata[0]}</b> (probe %{customdata[1]})<br>log2FC=%{x:.2f}<br>"
                          "p=%{customdata[2]:.2e}<br>FDR=%{customdata[3]:.2e}<extra></extra>"))
    top = d[d["cls"] != "Not significant"].assign(score=lambda t: t["neg_log10_p_adj"] * t["log2_fold_change"].abs())
    for _, r in top.nlargest(label_top, "score").iterrows():
        fig.add_annotation(x=r["log2_fold_change"], y=r["neg_log10_p_adj"], text=r["label"],
                           showarrow=False, yshift=10, font=dict(size=10, color=INK))
    fig.add_hline(y=-np.log10(fdr), line=dict(color=MUTED, dash="dot", width=1))
    for v in (-min_lfc, min_lfc):
        fig.add_vline(x=v, line=dict(color=MUTED, dash="dot", width=1))
    fig.update_layout(xaxis_title="log2 fold change (tumor − paired normal)",
                      yaxis_title="−log10 FDR-adjusted p")
    return _style(fig, "Volcano plot: paired tumor vs. adjacent-normal", 520)


def fig_gene_paired(X: pd.DataFrame, meta: pd.DataFrame, probe: str, label: str | None = None) -> go.Figure:
    """Box + slopegraph: each patient's normal->tumor change for one probe."""
    df = meta[["patient_id", "tissue_type"]].assign(expr=X[probe].values)
    fig = go.Figure()
    for pid, g in df.groupby("patient_id"):
        g = g.set_index("tissue_type").reindex(["Adjacent normal", "Tumor"])
        up = g.loc["Tumor", "expr"] > g.loc["Adjacent normal", "expr"]
        fig.add_trace(go.Scatter(x=["Adjacent normal", "Tumor"], y=g["expr"], mode="lines",
                                 line=dict(color=TUMOR_C if up else NORMAL_C, width=1), opacity=0.35,
                                 showlegend=False, hovertemplate=f"patient {pid}<br>%{{y:.2f}}<extra></extra>"))
    for tissue, color in TISSUE_COLORS.items():
        g = df[df["tissue_type"] == tissue]
        fig.add_trace(go.Box(x=[tissue] * len(g), y=g["expr"], name=tissue, marker_color=color,
                             boxpoints="all", jitter=0, pointpos=0, width=0.35, fillcolor="rgba(0,0,0,0)",
                             line_width=2, marker=dict(size=7, line=dict(color=SURFACE, width=1.5)),
                             customdata=g[["patient_id"]].values,
                             hovertemplate="patient %{customdata[0]}<br>%{y:.2f}<extra></extra>"))
    fig.update_layout(showlegend=False, yaxis_title="log2 expression (RMA)", xaxis_title=None)
    return _style(fig, f"{label or probe}: paired expression (lines connect each patient)", 440)


# ----------------------------------------------------------------------------- models
def fig_metric_distribution(fm: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure()
    for m in [m for m in MODEL_LABELS if m in fm["model"].unique()]:
        g = fm[fm["model"] == m]
        fig.add_trace(go.Box(y=g[metric], name=MODEL_LABELS[m], marker_color=MODEL_COLORS[m],
                             boxpoints="all", jitter=0.4, pointpos=0, line_width=2,
                             fillcolor="rgba(0,0,0,0)", marker=dict(size=7, opacity=0.8),
                             customdata=g[["repeat", "fold"]].values,
                             hovertemplate="repeat %{customdata[0]}, fold %{customdata[1]}<br>"
                                           + metric + "=%{y:.3f}<extra>" + MODEL_LABELS[m] + "</extra>"))
    fig.update_layout(showlegend=False, yaxis_title=metric.replace("_", " "),
                      yaxis_range=[-0.03, 1.05])
    return _style(fig, f"Outer-fold {metric.replace('_', ' ')} (each dot = one held-out fold)", 400)


def fig_roc(oof: pd.DataFrame, models: list[str], repeat: int | None = None) -> go.Figure:
    """ROC from pooled out-of-fold predictions (held-out patients only)."""
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=AXIS, dash="dot", width=1))
    for m in models:
        g = oof[oof["model"] == m]
        reps = [repeat] if repeat is not None else sorted(g["repeat"].unique())
        for i, r in enumerate(reps):
            gr = g[g["repeat"] == r]
            fpr, tpr, _ = roc_curve(gr["y_true"], gr["score"])
            from sklearn.metrics import auc
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", line_shape="hv",
                line=dict(color=MODEL_COLORS[m], width=2, dash="dash" if m == "dummy" else "solid"),
                opacity=1.0 if i == 0 else 0.45, legendgroup=m, showlegend=i == 0,
                name=f"{MODEL_LABELS[m]}" + (f" (AUC {auc(fpr, tpr):.3f})" if len(reps) == 1 else ""),
                hovertemplate=f"{MODEL_LABELS[m]} · repeat {r}<br>FPR=%{{x:.2f}} TPR=%{{y:.2f}}<extra></extra>"))
    fig.update_layout(xaxis_title="False positive rate (1 − specificity)", yaxis_title="True positive rate (recall)",
                      xaxis_range=[-0.02, 1.02], yaxis_range=[-0.02, 1.02])
    return _style(fig, "ROC curves from out-of-fold predictions", 460)


def fig_confusion(cm: dict[str, int], title: str) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Pred. normal", "Pred. tumor"], y=["True normal", "True tumor"],
                               colorscale=SEQUENTIAL, showscale=False, text=z, texttemplate="%{text}",
                               textfont=dict(size=18), hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
                               xgap=2, ygap=2))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _style(fig, title, 300)


def fig_runtime(fm: pd.DataFrame) -> go.Figure:
    r = fm.groupby("model")[["fit_seconds", "predict_seconds"]].mean().reindex(
        [m for m in MODEL_LABELS if m in fm["model"].unique()]).reset_index()
    r["label"] = r["model"].map(MODEL_LABELS)
    fig = px.bar(r, x="fit_seconds", y="label", orientation="h", color="model",
                 color_discrete_map=MODEL_COLORS, text=r["fit_seconds"].map("{:.1f}s".format))
    fig.update_traces(marker_line_width=0, textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Mean seconds per outer fold (incl. inner grid search)",
                      yaxis_title=None)
    return _style(fig, "Training time per outer fold", 280)


# ----------------------------------------------------------------------------- interpretation
def fig_signed_bars(df: pd.DataFrame, value: str, label: str, title: str, n: int = 25) -> go.Figure:
    d = df.reindex(df[value].abs().sort_values(ascending=False).index).head(n).iloc[::-1]
    colors = np.where(d[value] > 0, TUMOR_C, NORMAL_C)
    fig = go.Figure(go.Bar(x=d[value], y=d[label], orientation="h", marker_color=colors,
                           marker_line_width=0,
                           hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
    fig.update_layout(xaxis_title=f"{value.replace('_', ' ')}  (orange → tumor, blue → normal)",
                      yaxis_title=None, bargap=0.25)
    return _style(fig, title, max(360, 20 * len(d) + 100))


def fig_importance_bars(df: pd.DataFrame, value: str, label: str, title: str, n: int = 25,
                        color: str = NORMAL_C) -> go.Figure:
    d = df.nlargest(n, value).iloc[::-1]
    fig = go.Figure(go.Bar(x=d[value], y=d[label], orientation="h", marker_color=color,
                           marker_line_width=0, hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
    fig.update_layout(xaxis_title=value.replace("_", " "), yaxis_title=None, bargap=0.25)
    return _style(fig, title, max(360, 20 * len(d) + 100))


# ----------------------------------------------------------------------------- pathways
def fig_pathway_bar(pw: pd.DataFrame, n: int = 15) -> go.Figure:
    d = pw.nsmallest(n, "p_adj").iloc[::-1]
    colors = np.where(d["direction"] == "up in tumor", TUMOR_C, NORMAL_C)
    fig = go.Figure(go.Bar(x=d["neg_log10_p_adj"], y=d["term_name"].str.slice(0, 70), orientation="h",
                           marker_color=colors, marker_line_width=0,
                           customdata=np.c_[d["overlap"], d["set_size_in_universe"], d["p_adj"], d["direction"]],
                           hovertemplate="%{y}<br>%{customdata[3]}<br>overlap %{customdata[0]}/%{customdata[1]}"
                                         "<br>FDR=%{customdata[2]:.2e}<extra></extra>"))
    fig.update_layout(xaxis_title="−log10 FDR  (orange = genes up in tumor, blue = down)",
                      yaxis_title=None, bargap=0.25)
    return _style(fig, "Top enriched terms", max(380, 26 * len(d) + 100))


def fig_pathway_dot(pw: pd.DataFrame, n: int = 20) -> go.Figure:
    d = pw.nsmallest(n, "p_adj").iloc[::-1]
    fig = go.Figure(go.Scatter(
        x=d["gene_ratio"], y=d["term_name"].str.slice(0, 70), mode="markers",
        marker=dict(size=8 + 22 * d["overlap"] / max(d["overlap"].max(), 1), color=d["neg_log10_p_adj"],
                    colorscale=SEQUENTIAL, colorbar=dict(title="−log10 FDR", thickness=12),
                    line=dict(color=SURFACE, width=1.5)),
        customdata=np.c_[d["overlap"], d["p_adj"], d["direction"]],
        hovertemplate="%{y}<br>%{customdata[2]}<br>gene ratio %{x:.3f}<br>overlap %{customdata[0]}"
                      "<br>FDR=%{customdata[1]:.2e}<extra></extra>"))
    fig.update_layout(xaxis_title="Gene ratio (overlap / DE genes in direction); size = overlap",
                      yaxis_title=None)
    return _style(fig, "Enrichment dot plot", max(400, 26 * len(d) + 100))


# ----------------------------------------------------------------------------- static exports
def export_static_figures(out: Path, *, X, meta, pca_df, evr, de, oof, fm, top_probes,
                          probe_labels, fdr=0.05, min_lfc=1.0) -> list[Path]:
    """Matplotlib/seaborn PNGs for the README and the report (no browser required)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import auc

    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for _, g in pca_df.groupby("patient_id"):
        ax.plot(g["PC1"], g["PC2"], color=AXIS, lw=0.8, zorder=1)
    for t, c in TISSUE_COLORS.items():
        g = pca_df[pca_df["tissue_type"] == t]
        ax.scatter(g["PC1"], g["PC2"], c=c, s=55, edgecolor=SURFACE, lw=1.5, label=t, zorder=2)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})"); ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("Exploratory PCA (lines join each patient's pair)", loc="left", color=INK)
    ax.legend(frameon=False)
    p = out / "pca.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sig_up = (de["p_adj"] < fdr) & (de["log2_fold_change"] >= min_lfc)
    sig_dn = (de["p_adj"] < fdr) & (de["log2_fold_change"] <= -min_lfc)
    ns = ~(sig_up | sig_dn)
    ax.scatter(de.loc[ns, "log2_fold_change"], de.loc[ns, "neg_log10_p_adj"], s=4, c=NS_COLOR, alpha=0.5,
               label="Not significant", rasterized=True)
    ax.scatter(de.loc[sig_dn, "log2_fold_change"], de.loc[sig_dn, "neg_log10_p_adj"], s=8, c=NORMAL_C,
               label=f"Down in tumor ({sig_dn.sum()})")
    ax.scatter(de.loc[sig_up, "log2_fold_change"], de.loc[sig_up, "neg_log10_p_adj"], s=8, c=TUMOR_C,
               label=f"Up in tumor ({sig_up.sum()})")
    # Label the 5 largest effects in each direction, stacked in a column beside the cloud.
    for mask, side in ((sig_up, 1), (sig_dn, -1)):
        top = de[mask].reindex(de.loc[mask, "log2_fold_change"].abs().sort_values(ascending=False).index).head(5)
        top = top.sort_values("neg_log10_p_adj", ascending=False)
        ymax = de["neg_log10_p_adj"].max()
        for i, (pid, r) in enumerate(top.iterrows()):
            tx = (de["log2_fold_change"].max() + 0.6) if side > 0 else (de["log2_fold_change"].min() - 0.6)
            ty = ymax * (0.95 - 0.09 * i)
            ax.annotate(r["gene_symbol"] if isinstance(r["gene_symbol"], str) else pid,
                        (r["log2_fold_change"], r["neg_log10_p_adj"]), xytext=(tx, ty),
                        fontsize=7.5, color=INK, ha="left" if side > 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    lo, hi = de["log2_fold_change"].min(), de["log2_fold_change"].max()
    ax.set_xlim(lo - 2.2, hi + 2.2)
    ax.axhline(-np.log10(fdr), color=MUTED, ls=":", lw=1)
    for v in (-min_lfc, min_lfc):
        ax.axvline(v, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("log2 fold change (tumor − paired normal)"); ax.set_ylabel("−log10 FDR")
    ax.set_title("Paired differential expression", loc="left", color=INK); ax.legend(frameon=False, fontsize=8, loc="lower right")
    p = out / "volcano.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=AXIS, ls=":", lw=1)
    for m in [m for m in MODEL_LABELS if m in oof["model"].unique()]:
        g = oof[(oof["model"] == m) & (oof["repeat"] == 0)]
        fpr, tpr, _ = roc_curve(g["y_true"], g["score"])
        ax.step(fpr, tpr, where="post", color=MODEL_COLORS[m], lw=2, ls="--" if m == "dummy" else "-",
                label=f"{MODEL_LABELS[m]} (AUC {auc(fpr, tpr):.3f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC, pooled out-of-fold (repeat 0)", loc="left", color=INK); ax.legend(frameon=False, fontsize=8)
    p = out / "roc.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = [m for m in MODEL_LABELS if m in fm["model"].unique()]
    sns.stripplot(data=fm, x="model", y="balanced_accuracy", order=order, hue="model",
                  palette=MODEL_COLORS, ax=ax, size=6, jitter=0.2, legend=False)
    sns.boxplot(data=fm, x="model", y="balanced_accuracy", order=order, ax=ax, fill=False,
                color=INK_2, width=0.4, showfliers=False, linewidth=1)
    ax.set_xticks(range(len(order)), [MODEL_LABELS[m] for m in order])
    ax.set_xlabel(""); ax.set_ylabel("Balanced accuracy"); ax.set_ylim(-0.03, 1.05)
    ax.set_title("Outer-fold balanced accuracy (nested, patient-grouped CV)", loc="left", color=INK)
    p = out / "cv_balanced_accuracy.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    M = X[top_probes].T
    M.index = probe_labels
    Zs = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1), axis=0).clip(-3, 3)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [c for _, c in DIVERGING])
    cg = sns.clustermap(Zs, cmap=cmap, center=0, vmin=-3, vmax=3, method="average", metric="correlation",
                        col_colors=meta.loc[Zs.columns, "tissue_type"].map(TISSUE_COLORS).values,
                        xticklabels=False, yticklabels=True, figsize=(9, 9.6),
                        cbar_pos=(0.03, 0.84, 0.012, 0.07),
                        dendrogram_ratio=(0.12, 0.12))
    cg.ax_heatmap.tick_params(axis="y", labelsize=7)
    cg.ax_cbar.set_title("row z", fontsize=8, color=INK_2)
    cg.ax_cbar.tick_params(labelsize=7)
    cg.ax_heatmap.set_xlabel("50 samples (top bar: orange = tumor, blue = adjacent normal)")
    cg.fig.subplots_adjust(top=0.93)
    cg.fig.suptitle("Top 40 DE genes by |log2FC| (descriptive; selected using all samples)",
                    x=0.02, y=0.985, ha="left", color=INK)
    cg.ax_cbar.set_position([0.04, 0.83, 0.014, 0.08])
    p = out / "heatmap_top_de.png"; cg.fig.savefig(p); plt.close(cg.fig); paths.append(p)
    return paths
