# Limitations

OncoGene Explorer is an exploratory research and education project. It is **not** a clinically
validated diagnostic, screening, prognostic or treatment-selection tool.

## What the classifier task is, and is not

The models distinguish **breast tumor tissue from adjacent normal breast tissue of the same
patient**, in one Taiwanese cohort of 25 women profiled on one microarray platform. Performance
estimates must **not** be generalized to:

- cancer **screening** or early detection (the inputs are surgical tissue, not blood or imaging);
- **healthy** individuals or healthy-donor breast tissue;
- other cancers, other platforms (RNA-seq, other arrays) or other populations;
- clinical decisions of any kind.

## Small sample size

- There are only **25 independent units** (patients); the 50 samples are not independent.
- Each outer fold evaluates 5 patients, so a single misclassified sample changes fold accuracy by 10 points.
- Even with no observed errors, the exact 95% CI for the patient-level success rate is 0.863–1.000.
  A true per-patient error rate of up to ~14% is consistent with the data.
- Fold metrics from repeated CV are correlated, so their SD understates the true uncertainty.

## Overfitting risk

- There are ~17,900 features and 50 samples. Nested CV, in-fold feature selection and
  regularization keep the *estimates* honest, but they cannot add information.
- Hyperparameter search is mostly degenerate: many grid points tie at inner AUC = 1.0, and the
  "selected" values are simply the first tied ones.
- Importance rankings are unstable among the many highly correlated genes that carry the same
  signal. Permuting any single gene never changed held-out AUC.

## Limitations of adjacent-normal comparisons

- Adjacent tissue can harbor **field-cancerization** changes, which may *reduce* apparent differences.
- Distance from the tumor, sampling depth and histological review of the normal piece are not reported.

## Confounding

- **Tissue composition** is the dominant confounder. Normal breast is adipose-rich; tumors are
  epithelial, stromal and immune-rich. Fixed literature marker panels, with no training, separate
  the classes on their own: an 8-gene adipocyte score has AUC 0.98 (lower in tumor) and a 9-gene
  proliferation score has AUC 0.98. Many "top genes" are composition markers, not malignancy-specific
  changes. Deconvolution or composition-adjusted models would be needed to separate the two.
- **Unrecorded covariates**: molecular subtype, ER/PR/HER2 status, tumor purity, menopausal
  status, BMI, RNA quality, and array processing date or batch. If tumors and normals were
  processed in different batches, batch and tissue would be confounded; this cannot be checked
  from the public data.
- **Cohort**: a single center, all of one ancestry. The original study focused on early-onset
  disease (5 of 25 patients are aged ≤ 40).

## Statistical and annotation limitations

- The paired *t*-test is not variance-moderated (limma-style), and the Wilcoxon test has limited
  resolution at n = 25.
- Pathway ORA ignores gene–gene correlation, depends on DE cut-offs, and GO terms overlap.
- Probe-to-gene mapping uses the first assignment of an hg18-era annotation. 2,795 transcript
  clusters map to several genes, and some symbols are outdated (e.g. *FIGF*/*VEGFD*, *SDPR*/*CAVIN2*,
  *C2ORF40*/*ECRG4*).
- Exploratory PCA, correlation clustering and "top DE gene" heatmaps use all samples and labels.
  They are descriptive and would be optimistic as evidence of predictive performance.

## No external validation

All performance numbers are **internal** cross-validation estimates. External validation on an
independent cohort, with platform-aware gene alignment and batch handling fitted without test
labels, is the most important missing step (see Future work in the README).
