# Methodology

This document describes exactly what the pipeline does and why. Every parameter mentioned here
is set in [`configs/experiment.yaml`](../configs/experiment.yaml). A copy of the configuration
used for a run is saved to `artifacts/reports/experiment_config_used.yaml`.

---

## 1. Dataset characteristics

| Property | Value |
|---|---|
| Accession | GEO **GSE109169** (Chang *et al.*, PMID 29706651) |
| Platform | **GPL5175**, Affymetrix Human Exon 1.0 ST, *core* transcript clusters (meta-probeset file `HuEx-1_0-st-v2.r2.dt1.hg18.core.mps`) |
| Processing by submitters | RMA, Affymetrix Expression Console (values are already **log2**) |
| Samples | 50 = 25 tumor + 25 adjacent-normal |
| Patients | 25 Taiwanese women, ages 33–56 (median 46; 5 aged ≤ 40), stages I–IIIC |
| Features delivered | 19,076 transcript clusters |

### Acquisition and provenance

`oncogene download` fetches:

1. `GSE109169_series_matrix.txt.gz`: expression values plus all sample metadata.
2. The GPL5175 platform table, taken from GEO's *platform-only* endpoint
   (`acc.cgi?acc=GPL5175&targ=self&form=text&view=data`, about 87 MB, stored gzipped). The
   `GPL5175_family.soft.gz` file is about 4.5 GB because it bundles every sample on the platform,
   so it is deliberately avoided.
3. Gene-set libraries (Reactome 2022, GO Biological Process 2023) as GMT files from Enrichr.

Each file's URL, SHA-256, size and retrieval time are stored in `data/raw/provenance.json`.
Parsing uses a small built-in parser, not GEOparse. This keeps the dependency surface small and
makes every parsing rule visible in `src/oncogene/data/load.py`.

## 2. Patient-pair structure

Labels and patient IDs are derived from **two independent GEO fields** and must agree:

| Field | Example | Used for |
|---|---|---|
| `characteristics: tissue` | `breast cancer tissue` / `breast tumor adjacent normal tissue` | label (1 = tumor, 0 = adjacent normal) |
| `characteristics: patient identifier` | `2102` | patient ID |
| `title` | `2102-Tumor` | cross-check of both |

Any disagreement or unrecognized tissue string raises an error; nothing is guessed. Validation
then requires that every patient has exactly one tumor and one adjacent-normal sample. **All 25
pairs were confirmed, with no discrepancies.**

## 3. Expression preprocessing

### Validation (reported, never silently fixed)

The pipeline checks sample and patient counts, duplicate sample and probe IDs, missing patient
IDs, pair completeness, label values and balance, platform ID, non-numeric values, missing values,
zero-variance and near-constant probes, the scale (log vs. linear), per-sample median/IQR
consistency, and matrix–metadata alignment. Results are saved in
`artifacts/reports/validation_report.json`.

For GSE109169 the result is 0 missing values and 0 non-numeric values. Values range from 1.51 to
13.99, which is consistent with log2 RMA. Per-sample medians span 0.28 log2 units, and 17 probes
have SD < 0.1.

### Global, label-free filters

Only operations that use **no labels and no learned thresholds** are applied to the whole matrix.
They cannot leak information between folds.

| Step | Removed | Rationale |
|---|---|---|
| Probes absent from platform table | 0 | cannot be categorized or mapped |
| Design category ≠ `main` | 1,208 (1,195 `normgene->exon`, 13 unannotated) | control / background probesets don't measure genes |
| Exactly constant (SD = 0) | 0 | no information |

The result is **17,868 probes** (17,634 with a gene symbol; 17,324 unique symbols). The 17
near-constant probes are **kept**: variance filtering is learned inside each training fold.

### Scale and normalization

The data are RMA output (background correction, quantile normalization, median polish, log2).
`log2_if_needed` applies log2 only when the maximum value exceeds 30, which does not happen here.
**No RNA-seq count methods** (TPM, DESeq2 size factors, voom) are applied, because they are
inappropriate for microarray intensities.

### Gene annotation

The `gene_assignment` column has the form `accession // SYMBOL // description // cytoband // Entrez`,
with multiple assignments separated by `///`. The **first** assignment is used as the display
symbol, and symbols are upper-cased. The probe count per gene and the multi-gene flag are kept;
2,795 transcript clusters map to more than one symbol. Modeling and DE use probe-level features.
Collapsing to genes happens only for pathway analysis (§ 6). The annotation dates from the hg18
era, so some symbols are outdated aliases (e.g. *FIGF* → *VEGFD*, *SDPR* → *CAVIN2*).

## 4. Data-leakage prevention

**Rule:** anything that is *fitted* lives in a scikit-learn `Pipeline` and is refit on each
training fold.

```
SimpleImputer(median) → TopVarianceSelector(5000) → StandardScaler → SelectKBest(f_classif, k) → classifier
```

- **Imputation**: medians come from training samples (a no-op for GSE109169, but present for robustness).
- **Variance filter**: the top-N most variable probes *of the training fold*.
- **Scaling**: training-fold mean and SD.
- **Supervised selection**: ANOVA F computed on training labels only.
- **Hyperparameters**: chosen by an inner grouped CV that never sees the outer test patients.

The test `test_in_fold_selection_prevents_optimistic_bias_on_noise` checks that on pure-noise
data the held-out AUC stays near chance. Notebook 02 shows that selecting features *before* CV
on the same noise yields a strongly inflated AUC.

Exploratory PCA, the sample-correlation heatmap and the DE-based heatmap use all samples. They are
labeled **descriptive** everywhere and are never used as model input.

## 5. Statistical analysis (differential expression)

For each probe *g* and patient *i*: *d<sub>ig</sub>* = tumor − adjacent normal (log2).

- **Effect size:** mean(*d*) is a valid **log2 fold change** because the data are already log2.
  We also report fold change, Cohen's *d<sub>z</sub>* (mean / SD of differences) and the fraction
  of pairs that increase.
- **Primary test:** the paired *t*-test (one-sample *t* on *d*, 24 df). It assumes the paired
  differences are approximately normal. With 25 pairs and log-scale data this is reasonable, and
  the test is robust to moderate departures.
- **Sensitivity test:** the Wilcoxon signed-rank test, which assumes only symmetric differences.
  With n = 25 its smallest attainable *p*-values are bounded, so many top genes tie after correction.
- **Why not limma?** limma's moderated *t* shrinks gene-wise variances toward a common prior and is
  the field standard for microarrays. It is not available natively in Python. With 25 pairs, the
  ordinary paired *t* has enough degrees of freedom that moderation changes rankings little. This
  is listed as a future improvement.

### Multiple-testing correction

**Benjamini–Hochberg** FDR across all 17,868 probes, applied separately to the *t* and Wilcoxon
*p*-values. The implementation is verified against `statsmodels` in the tests. A probe is
"significant" if **FDR < 0.05 and |log2FC| ≥ 1** (configurable).

## 6. Pathway analysis

- **Method:** over-representation analysis (one-sided hypergeometric test).
- **Input:** genes passing the DE thresholds, split into **up** (361) and **down** (461) lists.
  Probes are collapsed to symbols; a gene with probes significant in *both* directions is excluded
  from both lists (the count is reported).
- **Universe:** the **17,324 symbols measured on the array**, not the whole genome. Each gene set is
  intersected with the universe before testing, so unmeasurable genes cannot create spurious depletion.
- **Gene sets:** Reactome 2022 and GO Biological Process 2023 (Enrichr GMT files), with 10–500
  measured genes per set. KEGG is excluded by default because of its licensing terms.
- **Correction:** BH FDR within each library × direction.
- **Caveats:** ORA ignores inter-gene correlation and depends on arbitrary DE thresholds, and GO
  terms overlap heavily. Results are exploratory.

## 7. Cross-validation methodology

| Level | Splitter | Purpose |
|---|---|---|
| Outer | `RepeatedPatientKFold(5 folds × 3 repeats, seed 42)` | performance estimate |
| Inner | `CheckedGroupKFold(4)` inside `GridSearchCV` | hyperparameter selection |

- Patients, not samples, are assigned to folds, so both samples of a patient always fall in the
  same partition. Each outer test fold holds 5 patients (10 samples) and is automatically
  class-balanced.
- Every split passes through `assert_no_patient_leakage`, which raises if a patient appears on
  both sides.
- Seeds: global seed 42. Repeat *r* reshuffles patients and uses classifier seed 42 + *r*.

## 8. Models and hyperparameter tuning

| Model | Tuned parameters (inner grouped grid search, scored by ROC-AUC) |
|---|---|
| Dummy (prior) | none; predicts the training prior, AUC 0.5 by construction |
| Logistic regression (L2) | k ∈ {50, 500}; C ∈ {0.01, 0.1, 1} |
| Random forest (300 trees) | k ∈ {200, 2000}; max_depth ∈ {None, 5}; min_samples_leaf ∈ {1, 3}; max_features ∈ {sqrt, 0.1} |
| SVM | linear: k ∈ {50, 500}, C ∈ {0.01, 0.1, 1}; RBF: k ∈ {50, 500}, C ∈ {1, 10}, γ = scale |

**Ties.** When several grid points share the best inner score (very common here: many reach
AUC = 1.0), `GridSearchCV` keeps the first. The number of tied points per fold is recorded in
`fold_best_params.csv` and shown in the dashboard.

**Final models.** Each model is also tuned (grouped CV) and refit on all 50 samples. These final
models are used **only** for interpretation (coefficients, impurity importance). They are saved
with a JSON model card that records parameters, seed, config hash, library versions and feature
counts. They are never used to report performance.

## 9. Model evaluation

- **Metrics:** accuracy, precision, recall (sensitivity), specificity, F1, balanced accuracy, and
  ROC-AUC. ROC-AUC uses P(tumor) for LR, RF and the dummy, and the decision function for SVM;
  hard labels are never used for AUC.
- **Fold level:** mean ± SD, min and max over 15 outer folds.
- **Pooled:** out-of-fold predictions for each repeat, where every sample is predicted exactly once.
- **Uncertainty:**
  - patient-bootstrap percentile CI (resampling patients) on pooled repeat-0 predictions;
  - an **exact Clopper–Pearson CI at the patient level**. A patient counts as a success only if both
    samples are correct. This interval stays informative under perfect accuracy: 25/25 gives a
    95% CI of [0.863, 1.000].
- **Learning curve:** logistic regression with fixed, untuned hyperparameters (k = 50, C = 0.1),
  trained on *n* ∈ {1, 2, 3, 4, 6, 8, 12, 16, 20} random patients and tested on all remaining
  patients, 30 draws per size.

## 10. Interpretation methods

| Method | Data | Caveat |
|---|---|---|
| LR standardized coefficients | final model | L2 spreads weight over correlated genes; the sign is conditional on the other genes |
| RF impurity importance | final model, training data | biased to high-variance features; split among correlated features |
| Permutation importance | **each outer validation fold**, features used by that fold's model | see below |
| Selection frequency | outer training folds | measures univariate stability, not mechanism |

**Permutation details.** Each pre-classifier step operates column-wise, so permuting an input
probe is equivalent to permuting its transformed column. Permutation therefore happens in the
reduced space. Features never used by a tree have exactly zero importance and are skipped. Two
scores are recorded:

- **ROC-AUC drop.** This was **0 for every gene**: the signal is so redundant that no single gene is
  necessary. That is a finding in itself.
- **Relative drop in the mean signed held-out margin**, (2y − 1)·score, averaged over 20 permutations.
  It is sensitive enough to rank genes and is the reported permutation score.

Fold results are averaged over all outer folds, and folds where a gene was not selected count as 0.

## 11. Biological interpretation assumptions

- DE and model importance show **association with tissue type in this cohort**, not causation.
- Gene descriptions come from the platform annotation, and links point to NCBI Gene, HGNC,
  UniProt, the Human Protein Atlas, Ensembl and PubMed.
- Tissue composition is an explicit alternative explanation. The fixed adipocyte, proliferation and
  myoepithelial marker panels (literature lists, not chosen from these labels) quantify it.

## 12. Limitations and sources of uncertainty

See [limitations.md](limitations.md). In short: there are 25 patients from one center on one
platform. The comparison is with *adjacent* normal tissue, there is strong composition confounding,
no external validation, outdated annotation, and the hyperparameter choices are often ties.
