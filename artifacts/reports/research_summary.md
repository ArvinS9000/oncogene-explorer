# OncoGene Explorer - research summary

_Generated 2026-11-17 21:31 UTC from pipeline artifacts. Research and education only; not a diagnostic tool._

## Dataset
- GEO **GSE109169** (GPL5175, Affymetrix Human Exon 1.0 ST, core transcript clusters, RMA log2).
- 50 samples from 25 patients: 25 tumor, 25 paired **adjacent-normal** (not healthy-donor) tissue.
- 17,868 probes after label-free filtering; 0 missing values; log2 transform applied: False.
- Median patient age 46; stages: I (6), IIA (8), IIB (3), IIIA (7), IIIC (1).

## Differential expression (paired t-test, BH-FDR)
- 8,220 probes at FDR < 0.05; 830 also |log2FC| ≥ 1.0 (365 up, 465 down in tumor).
- Largest increases in tumor: CST1 (+3.52), MMP11 (+3.17), OLR1 (+2.75), CXCL10 (+2.72), TOP2A (+2.66), CXCL9 (+2.63), CKS2 (+2.60), TPX2 (+2.53), KIAA0101 (+2.53), COL11A1 (+2.51).
- Largest decreases in tumor: FABP4 (-5.60), FIGF (-4.14), ADIPOQ (-4.11), CHRDL1 (-3.92), ADH1B (-3.70), LYVE1 (-3.59), CD36 (-3.53), FDXACB1 (-3.29), AKR1C2 (-3.23), SFRP1 (-3.17).

## Pathway over-representation (exploratory)
- Universe: 17,324 array genes; 361 up / 461 down genes tested.
- Top terms, genes up in tumor: Cell Cycle, Mitotic (Reactome, FDR 2.0e-23); Cell Cycle Checkpoints (Reactome, FDR 5.0e-20); Resolution Of Sister Chromatid Cohesion (Reactome, FDR 6.3e-16); Mitotic Sister Chromatid Segregation (GO, FDR 7.7e-15); Mitotic Prometaphase (Reactome, FDR 2.4e-14)
- Top terms, genes down in tumor: Regulation Of Angiogenesis (GO, FDR 8.1e-06); Positive Regulation Of Intracellular Signal Transduction (GO, FDR 8.1e-06); Positive Regulation Of Protein Kinase B Signaling (GO, FDR 8.1e-06); Regulation Of Protein Kinase B Signaling (GO, FDR 1.2e-04); Regulation Of Cold-Induced Thermogenesis (GO, FDR 1.2e-04)

## Classification (nested, patient-grouped CV)
Outer: 5-fold × 3 repeats over patients; inner: 4-fold grouped grid search on roc_auc.

| Model               | ROC-AUC (fold mean ± SD)   | Balanced acc.   |   Recall (sens.) |   Specificity |   F1 | Patients fully correct (exact 95% CI)   |   Fit s/fold |
|:--------------------|:---------------------------|:----------------|-----------------:|--------------:|-----:|:----------------------------------------|-------------:|
| Dummy (prior)       | 0.500 ± 0.000              | 0.500 ± 0.000   |                0 |             1 |    0 | 0/25 [0.000, 0.137]                     |          0.1 |
| Logistic regression | 1.000 ± 0.000              | 1.000 ± 0.000   |                1 |             1 |    1 | 25/25 [0.863, 1.000]                    |          0.8 |
| Random forest       | 1.000 ± 0.000              | 1.000 ± 0.000   |                1 |             1 |    1 | 25/25 [0.863, 1.000]                    |          3.9 |
| SVM                 | 1.000 ± 0.000              | 1.000 ± 0.000   |                1 |             1 |    1 | 25/25 [0.863, 1.000]                    |          0.7 |

### Learning curve (logistic regression, fixed hyperparameters)

|   training patients |   roc auc (mean) |   roc auc (std) |   balanced accuracy (mean) |   balanced accuracy (std) |
|--------------------:|-----------------:|----------------:|---------------------------:|--------------------------:|
|                   1 |            0.983 |           0.027 |                      0.839 |                     0.099 |
|                   2 |            0.998 |           0.003 |                      0.936 |                     0.07  |
|                   3 |            0.999 |           0.002 |                      0.968 |                     0.052 |
|                   4 |            0.999 |           0.002 |                      0.989 |                     0.02  |
|                   6 |            1     |           0.001 |                      0.995 |                     0.014 |
|                   8 |            1     |           0.001 |                      0.994 |                     0.016 |
|                  12 |            1     |           0     |                      1     |                     0     |
|                  16 |            1     |           0     |                      1     |                     0     |
|                  20 |            1     |           0     |                      1     |                     0     |

## Interpretation and limitations
- The task (tumor vs. **adjacent** normal from the same breast) is close to trivially separable; near-perfect metrics reflect large, consistent tissue-level differences - notably adipocyte/stromal content vs. proliferation - and do **not** imply diagnostic utility.
- 25 independent patients: even zero observed errors are compatible with a substantial true error rate (see exact CIs above).
- Single cohort, single platform, no external validation; adjacent-normal tissue can carry field effects and differs from healthy-donor breast tissue.
- Feature importance and DE identify statistical associations, not causal mechanisms.