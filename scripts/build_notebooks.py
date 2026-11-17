"""Generate (and optionally execute) the analysis notebooks.

Notebooks are thin, narrative wrappers around the `oncogene` package; all reusable logic lives
in `src/`. Run after the pipeline:  python scripts/build_notebooks.py --execute
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
from oncogene.config import Paths, load_config
from oncogene.data.load import load_processed
paths, cfg = Paths(), load_config()
ds = load_processed(paths)
X, meta, ann = ds.X, ds.meta, ds.annotation
M = paths.metrics
print(X.shape, meta.shape)"""


def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS = {
    "01_dataset_exploration.ipynb": [
        md("""# 01 · Dataset exploration
GEO **GSE109169**: 25 matched pairs of breast tumor and *adjacent* normal tissue, Affymetrix Human Exon 1.0 ST
(GPL5175), RMA-normalized. This notebook inspects samples, patients and the expression scale.
Run `python -m oncogene.cli all` first."""),
        code(SETUP),
        md("## Sample metadata\nLabels come from the `tissue` characteristic and are cross-checked against sample titles."),
        code("meta.head(10)"),
        code("meta.groupby('tissue_type').size(), meta['patient_id'].nunique(), meta.drop_duplicates('patient_id')['cancer_stage'].value_counts()"),
        md("## Expression scale\nValues range ~1.5–14, consistent with log2 RMA output, so **no additional log transform** is applied."),
        code("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(X.values.ravel(), bins=80, color='#2a78d6'); ax[0].set_xlabel('log2 expression'); ax[0].set_title('All values')
X.T.boxplot(ax=ax[1], showfliers=False, grid=False, rot=90, fontsize=5); ax[1].set_title('Per-sample distributions')
plt.tight_layout()"""),
        md("## Exploratory PCA (all samples — descriptive only)"),
        code("""from oncogene.analysis.exploratory import exploratory_pca
pca_df, evr = exploratory_pca(X, meta)
colors = pca_df['label'].map({0: '#2a78d6', 1: '#eb6834'})
fig, ax = plt.subplots(figsize=(6, 5))
for _, g in pca_df.groupby('patient_id'): ax.plot(g.PC1, g.PC2, color='#c3c2b7', lw=.8)
ax.scatter(pca_df.PC1, pca_df.PC2, c=colors, s=45, edgecolor='white')
ax.set_xlabel(f'PC1 ({evr[0]:.1%})'); ax.set_ylabel(f'PC2 ({evr[1]:.1%})'); ax.set_title('blue = adjacent normal, orange = tumor')"""),
        code("""from scipy.stats import mannwhitneyu
print('PC1 tumor vs normal Mann-Whitney p =', mannwhitneyu(pca_df.loc[pca_df.label==1,'PC1'], pca_df.loc[pca_df.label==0,'PC1']).pvalue)"""),
        md("## Outlier screen and composition scores"),
        code("pd.read_csv(M/'outlier_screen.csv', index_col=0).sort_values('robust_z').head(10)"),
        code("pd.read_csv(M/'marker_scores.csv', index_col=0).groupby('tissue_type').mean(numeric_only=True)"),
    ],
    "02_preprocessing.ipynb": [
        md("""# 02 · Preprocessing and validation
All global operations are **label-free and deterministic**. Anything learned (variance ranking, scaling,
feature selection, imputation) happens inside model pipelines, per training fold."""),
        code(SETUP),
        code("rep = json.loads((paths.reports/'preprocessing_report.json').read_text())\npd.DataFrame(rep['validation'])[['name','status','detail']]"),
        code("pd.json_normalize(rep['filters'])"),
        code("rep['annotation'], rep['near_constant_retained'], rep['log2_transform_applied']"),
        md("## The model pipeline\nIdentical structure for every model; fitted only on training patients."),
        code("""from oncogene.models.train import build_pipeline
build_pipeline('logistic_regression', cfg)"""),
        md("## Demonstration: why feature selection must live inside CV\nOn pure noise, selecting features on all samples *before* CV produces an inflated AUC; selecting inside each fold does not."),
        code("""from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from oncogene.models.tune import RepeatedPatientKFold
rng = np.random.default_rng(0)
Xn = rng.normal(size=(50, 5000)); y = meta['label'].values; g = meta['patient_id'].values
cv = list(RepeatedPatientKFold(5, 1, 0).split(Xn, y, g))
leaky_X = SelectKBest(f_classif, k=20).fit_transform(Xn, y)       # WRONG: uses all labels
leaky = cross_val_score(LogisticRegression(), leaky_X, y, cv=cv, scoring='roc_auc').mean()
safe = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression()), Xn, y, cv=cv, scoring='roc_auc').mean()
print(f'noise data  leaky AUC={leaky:.2f}   leakage-safe AUC={safe:.2f}')"""),
    ],
    "03_differential_expression.ipynb": [
        md("""# 03 · Paired differential expression
Paired t-test on within-patient log2 differences, Benjamini–Hochberg FDR; Wilcoxon signed-rank as a sensitivity check."""),
        code(SETUP),
        code("de = pd.read_csv(M/'de_results.csv', index_col='probe_id', dtype={'probe_id': str})\nde.head(15)[['gene_symbol','log2_fold_change','cohens_dz','p_adj','wilcoxon_p_adj','significant']]"),
        code("""sig = de[de.significant]
print(f"FDR<0.05: {(de.p_adj<0.05).sum()}, +|log2FC|>=1: {len(sig)} (up {(sig.log2_fold_change>0).sum()}, down {(sig.log2_fold_change<0).sum()})")
print('Agreement t-test vs Wilcoxon at FDR<0.05:', ((de.p_adj<0.05)==(de.wilcoxon_p_adj<0.05)).mean().round(3))"""),
        code("""fig, ax = plt.subplots(figsize=(6,5))
c = np.where(~de.significant, '#c3c2b7', np.where(de.log2_fold_change>0, '#eb6834', '#2a78d6'))
ax.scatter(de.log2_fold_change, de.neg_log10_p_adj, c=c, s=4); ax.set_xlabel('log2FC (tumor - normal)'); ax.set_ylabel('-log10 FDR')"""),
        code("sig.nlargest(15,'log2_fold_change')[['gene_symbol','gene_description','log2_fold_change','p_adj']]"),
        code("sig.nsmallest(15,'log2_fold_change')[['gene_symbol','gene_description','log2_fold_change','p_adj']]"),
        md("## P-value distribution\nA strong spike near 0 with a flat remainder indicates abundant true signal and well-behaved tests."),
        code("plt.hist(de.p_value, bins=50, color='#2a78d6'); plt.xlabel('raw p-value');"),
    ],
    "04_model_experiments.ipynb": [
        md("""# 04 · Model experiments
Results of nested, patient-grouped CV (5 outer folds × 3 repeats; 4 inner grouped folds)."""),
        code(SETUP),
        code("""s = json.loads((M/'model_summary.json').read_text())
pd.DataFrame({m: {k: f"{v['mean']:.3f} ± {v['sd']:.3f}" for k, v in d['fold'].items()} for m, d in s['models'].items()}).T"""),
        code("pd.DataFrame({m: d['patient_level_exact_ci'] for m, d in s['models'].items()}).T"),
        code("p = pd.read_csv(M/'fold_best_params.csv'); p.groupby('model')[['n_grid_points','n_tied_at_best']].mean()"),
        md("Ties at the best inner score mean hyperparameter 'selection' is mostly arbitrary for this easy task."),
        code("""lc = pd.read_csv(M/'learning_curve.csv')
g = lc.groupby('n_train_patients')[['roc_auc','balanced_accuracy']].agg(['mean','std','min']); g"""),
        code("""fig, ax = plt.subplots(figsize=(6,3.5))
ax.errorbar(g.index, g[('roc_auc','mean')], g[('roc_auc','std')], marker='o', color='#2a78d6', label='ROC-AUC')
ax.errorbar(g.index, g[('balanced_accuracy','mean')], g[('balanced_accuracy','std')], marker='o', color='#eb6834', label='balanced acc.')
ax.set_xlabel('training patients'); ax.legend(); ax.set_ylim(0.3, 1.05)"""),
        md("## Re-running a single model (optional, ~1–2 min)"),
        code("""# from oncogene.models.evaluate import run_nested_cv
# res = run_nested_cv('logistic_regression', X, ds.y, ds.groups, cfg, compute_permutation=False)
# res.summary['fold']['roc_auc']"""),
    ],
    "05_biological_interpretation.ipynb": [
        md("""# 05 · Biological interpretation
Model-based rankings, differential expression and pathway enrichment, read together. These are associations
with tissue type in one cohort, not causal claims."""),
        code(SETUP),
        code("""r = pd.read_csv(M/'gene_ranking.csv', dtype={'probe_id': str})
for m, g in r.groupby('method'):
    print(f"{m}:\\n  ", ', '.join(g.nsmallest(12, 'rank_within_method')['gene_symbol'].fillna('?')))"""),
        code("""lr = r[r.method.str.startswith('LR standardized')].head(30)
lr[['gene_symbol','signed_value','log2_fold_change','p_adj']]"""),
        code("pw = pd.read_csv(M/'pathway_results.csv')\npw.groupby(['library','direction']).head(5)[['library','direction','term_name','overlap','p_adj']]"),
        md("## Composition check\nAdipocyte vs. proliferation marker scores (fixed literature panels) separate the tissues on their own."),
        code("""ms = pd.read_csv(M/'marker_scores.csv', index_col=0)
c = ms.tissue_type.map({'Adjacent normal':'#2a78d6','Tumor':'#eb6834'})
plt.scatter(ms.adipocyte_score, ms.proliferation_score, c=c, s=45, edgecolor='white')
plt.xlabel('adipocyte score'); plt.ylabel('proliferation score');"""),
        code("""from sklearn.metrics import roc_auc_score
for col in ['adipocyte_score','proliferation_score']:
    print(col, 'AUC (descriptive, no CV needed: fixed panel)', round(roc_auc_score(ms.tissue_type=='Tumor', ms[col]), 3))"""),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
        path = NB_DIR / name
        if args.execute:
            from nbconvert.preprocessors import ExecutePreprocessor
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
