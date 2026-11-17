import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import data
from components.ui import download_csv, guarded, note, page_setup
from oncogene.visualization.plots import (MODEL_COLORS, MODEL_LABELS, _style, fig_confusion,
                                          fig_metric_distribution, fig_roc, fig_runtime)

page_setup("Machine learning", "Classifier performance on held-out patients",
           "Every number on this page comes from nested, patient-grouped cross-validation: models are "
           "always evaluated on patients whose samples they never saw during training or tuning.")

METRICS = ["roc_auc", "balanced_accuracy", "accuracy", "precision", "recall", "specificity", "f1"]

with guarded("model results"):
    summ = data.metric_json("model_summary.json")
    fm = data.metric_csv("fold_metrics.csv")
    oof = data.metric_csv("oof_predictions.csv")
    params = data.metric_csv("fold_best_params.csv")
    available = [m for m in MODEL_LABELS if m in summ["models"]]

    chosen = st.multiselect("Models", available, default=available, format_func=MODEL_LABELS.get)
    if not chosen:
        st.stop()

    st.subheader("Model comparison")
    shown = {"roc_auc": "ROC-AUC", "balanced_accuracy": "Balanced acc.", "recall": "Sensitivity",
             "specificity": "Specificity", "f1": "F1"}
    rows = []
    for m in chosen:
        s = summ["models"][m]
        ex = s["patient_level_exact_ci"]
        row = {"Model": MODEL_LABELS[m]}
        for k, label in shown.items():
            row[label] = f"{s['fold'][k]['mean']:.3f} ± {s['fold'][k]['sd']:.3f}"
        row["Patients correct"] = f"{ex['patients_fully_correct']}/{ex['n_patients']}"
        row["Exact 95% CI"] = f"{ex['ci95'][0]:.2f}–{ex['ci95'][1]:.2f}"
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table, use_container_width=True)
    download_csv(fm[fm["model"].isin(chosen)].set_index("model"), "fold_metrics.csv", "Download fold-level metrics")
    note(f"Values are mean ± SD across {summ['cv']['outer_folds']} outer folds × {summ['cv']['outer_repeats']} "
         "repeats (each fold holds out 5 patients = 10 samples). The last two columns treat each <b>patient</b> as "
         "one trial (correct only if both samples are), with an exact Clopper–Pearson interval. With only 25 "
         "independent patients, even a perfect score is compatible with a true patient-level error rate of "
         "roughly 14%.", warn=True)

    t1, t2, t3, t4, t5 = st.tabs(["Fold distributions", "ROC curves", "Confusion matrices",
                                  "Learning curve", "Hyperparameters & runtime"])
    with t1:
        metric = st.selectbox("Metric", METRICS, format_func=lambda s: s.replace("_", " "))
        st.plotly_chart(fig_metric_distribution(fm[fm["model"].isin(chosen)], metric), use_container_width=True)
    with t2:
        reps = sorted(oof["repeat"].unique())
        rsel = st.radio("CV repeat", ["All repeats"] + [f"Repeat {r}" for r in reps], horizontal=True)
        r = None if rsel == "All repeats" else int(rsel.split()[-1])
        st.plotly_chart(fig_roc(oof, chosen, r), use_container_width=True)
        note("ROC curves pool the out-of-fold scores of one repeat (every sample predicted exactly once, by a model "
             "that never saw that patient). Scores are P(tumor) for LR/RF/dummy and SVM decision values; "
             "ROC-AUC never uses hard class labels.")
    with t3:
        cols = st.columns(len(chosen))
        for c, m in zip(cols, chosen):
            with c:
                st.plotly_chart(fig_confusion(summ["models"][m]["confusion_matrix_repeat0"], MODEL_LABELS[m]),
                                use_container_width=True)
        st.caption("Out-of-fold predictions from repeat 0 (50 samples).")
    with t4:
        try:
            lc = data.metric_csv("learning_curve.csv")
            g = lc.groupby("n_train_patients")[["roc_auc", "balanced_accuracy"]].agg(["mean", "std", "min"])
            fig = go.Figure()
            for k, color, name in (("roc_auc", MODEL_COLORS["logistic_regression"], "ROC-AUC"),
                                   ("balanced_accuracy", MODEL_COLORS["random_forest"], "Balanced accuracy")):
                fig.add_trace(go.Scatter(x=g.index, y=g[(k, "mean")], mode="lines+markers", name=name,
                                         line=dict(color=color, width=2), marker=dict(size=8),
                                         error_y=dict(type="data", array=g[(k, "std")], thickness=1, width=4),
                                         hovertemplate="%{x} training patients<br>mean %{y:.3f}<extra>" + name + "</extra>"))
            fig.update_layout(xaxis_title="Patients in training set (2 samples each)",
                              yaxis_title="Held-out score (mean ± SD)", yaxis_range=[0.3, 1.05])
            st.plotly_chart(_style(fig, "Learning curve — logistic regression, fixed hyperparameters", 420),
                            use_container_width=True)
            st.dataframe(g.round(3), use_container_width=True)
            note("Each point trains on <i>n</i> randomly chosen patients and tests on all remaining patients "
                 f"({lc['repeat'].nunique()} random draws per size). The signal is learned from very few patients — a "
                 "sign of how large and consistent the tumor/adjacent-normal difference is.")
        except data.MissingArtifact:
            st.info("Learning-curve artifact not found.")
    with t5:
        p = params[params["model"].isin(chosen)].copy()
        p["model"] = p["model"].map(MODEL_LABELS)
        st.dataframe(p.dropna(axis=1, how="all"), use_container_width=True, hide_index=True, height=320)
        tied = params.dropna(subset=["n_tied_at_best"])
        tied = tied[tied["model"].isin(chosen)]
        if len(tied) and (tied["n_tied_at_best"] > 1).any():
            note(f"In {int((tied['n_tied_at_best'] > 1).sum())} of {len(tied)} outer folds, several grid points tied "
                 "for the best inner-CV score (often all at ROC-AUC = 1.0). GridSearchCV then keeps the <b>first</b> "
                 "tied grid point, so the 'selected' hyperparameters are not meaningfully better than the alternatives.",
                 warn=True)
        st.plotly_chart(fig_runtime(fm[fm["model"].isin(chosen)]), use_container_width=True)
        with st.expander("Final model cards (refit on all samples for interpretation only)"):
            for m, card in data.model_cards().items():
                if m in chosen:
                    st.markdown(f"**{MODEL_LABELS.get(m, m)}**")
                    st.json(card, expanded=False)

with st.expander("How evaluation works", expanded=False):
    st.markdown("""
**Nested, patient-grouped cross-validation**

1. **Outer loop** — the 25 patients are split into 5 folds (5 patients each); this is repeated 3 times with
   different random patient assignments. Each outer test fold contains both samples of its patients and nothing else.
2. **Inner loop** — inside each outer training set (20 patients), a 4-fold *grouped* grid search chooses
   hyperparameters (regularization, tree depth, kernel, number of selected genes).
3. **Inside every fit**, the full pipeline is refit from scratch on training patients only:
   median imputation → top-5,000 most variable probes → standardization → ANOVA-F *k*-best selection → classifier.
4. The tuned pipeline predicts the untouched outer patients. Those predictions are the only ones used for metrics.

A runtime check asserts that no patient appears in both the training and test side of any split.
""")
