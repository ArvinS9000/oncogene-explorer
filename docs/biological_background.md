# Biological background

The minimum biology needed to understand this project, written for a computer-science student
entering computational biology.

## DNA and genes

Every human cell (with a few exceptions such as red blood cells) carries the same ~3.1 billion
base pairs of **DNA**, the genome. A **gene** is a region of DNA that encodes a functional product,
usually a protein. Humans have roughly 20,000 protein-coding genes plus many non-coding genes.
Genes are named with standardized **symbols** maintained by HGNC (e.g. *ESR1* = estrogen receptor 1)
and have stable numeric IDs (NCBI **Entrez Gene** IDs, Ensembl IDs).

Programming analogy: the genome is a large, mostly read-only code base. Each cell type "executes"
a different subset of it.

## RNA and gene expression

To use a gene, the cell **transcribes** its DNA into messenger RNA (**mRNA**), which is then
**translated** into protein. The amount of mRNA a cell makes from a gene is that gene's
**expression level**. Cells of different types (fat cells, epithelial cells, immune cells) and
cells in different states (dividing, resting, stressed, malignant) express different genes at
different levels. Expression is the main way the same genome produces different cell behaviors.

## Measuring gene expression

- **Microarrays** (used here) carry millions of short DNA *probes* on a glass chip. Labeled RNA
  from the sample hybridizes (binds) to complementary probes, and fluorescence intensity reflects
  abundance. The Affymetrix **Human Exon 1.0 ST** array has probes along each exon. Probes are
  grouped into **transcript clusters**, roughly one per gene, and those clusters are this project's
  features.
- **RMA** (Robust Multi-array Average) is the standard preprocessing step. It applies background
  correction, quantile normalization so all arrays share the same intensity distribution, a robust
  summary of the probes in each cluster, and a **log2** transform. A difference of 1 on this scale
  means a two-fold difference in abundance.
- **RNA-seq** counts sequencing reads instead. Its raw counts need different statistics (negative
  binomial models, library-size normalization), which is why this project does *not* apply RNA-seq
  methods to microarray data.

Expression measurements are noisy, relative (not absolute molecule counts), and reflect a
**mixture of all cells in the tissue piece**.

## Tumor versus adjacent normal tissue

A **tumor** sample is tissue taken from the cancer. An **adjacent normal** sample is
histologically normal-looking tissue taken from the same breast, near the tumor. Using pairs from
the same patient controls for genetic background, age and many lifestyle factors, which makes
paired comparisons powerful.

Adjacent normal tissue is **not** the same as breast tissue from a healthy person:

- It may carry **field effects**: molecular changes in tissue surrounding a tumor, caused by shared
  exposures or signals from the tumor.
- Its **cell-type composition** differs strongly from the tumor's. Normal breast is largely
  **adipose (fat) tissue** with scattered ducts and lobules. Tumors are dense masses of epithelial
  cancer cells, fibroblast-rich (desmoplastic) stroma and immune cells. Many expression differences
  therefore reflect *which cells are present*, not only changes inside cells.

## Differential expression

**Differential expression (DE)** analysis asks, for each gene, whether its expression differs
between conditions more than chance would explain. Here that means tumor vs. paired normal.

- The **log2 fold change** is the average within-patient difference on the log2 scale. +1 means
  2× higher in tumor; −2 means 4× lower.
- A **p-value** comes from a statistical test (here a paired *t*-test).
- Testing ~18,000 genes at p < 0.05 would yield ~900 false positives by chance alone, so we control
  the **false discovery rate** (FDR, Benjamini–Hochberg): among genes called significant at
  FDR < 5%, about 5% are expected to be false discoveries.

## Biological pathways

Genes work in groups. A **pathway** or **gene set** is a curated list of genes that act together:
a signaling cascade, a metabolic process, or the machinery of cell division.

- **Gene Ontology (GO)** describes biological processes, molecular functions and cellular components.
- **Reactome** is a peer-reviewed database of human pathways and reactions.
- **KEGG** is a widely used pathway database with restrictive licensing for some uses.

**Over-representation analysis** asks whether a pathway contains more DE genes than expected if DE
genes were drawn at random from all *measured* genes (the background universe). It helps turn a
list of hundreds of genes into interpretable themes, such as "cell division is up in tumors."

## Biomarkers

A **biomarker** is a measurable indicator of a biological state. Examples are *ESR1/PGR/ERBB2*
status, which guides breast cancer therapy, or proliferation markers such as *MKI67*. A good
clinical biomarker must be **validated** in independent cohorts, show clinical utility, and be
measurable reliably. Genes that separate tumor from adjacent normal in one 25-patient study are
*candidates* at best: such a contrast is easy, and the real clinical need is to separate
patients from each other (subtype, prognosis, treatment response).

## Disease-associated genes

A gene can be associated with a disease in several distinct ways. It's important not to conflate them:

| Kind of evidence | Example | What it shows |
|---|---|---|
| Expression association (this project) | *TOP2A* higher in tumors | the gene's mRNA level co-varies with tissue state |
| Predictive importance (this project) | a model weights *FABP4* heavily | the gene helps a model classify *this dataset* |
| Genetic association | inherited *BRCA1* variants raise risk | variation in the gene is linked to risk |
| Causal / functional | knocking out a gene stops tumor growth in experiments | the gene plays a mechanistic role |

Many genes highlighted here are **consequences or correlates** of cancer: proliferating cells
express cell-cycle genes, and less fat means less adipocyte gene expression. They are not drivers.
Establishing causation requires genetic, experimental or interventional evidence that expression
data alone cannot provide.

## Key genes you will see in the results

| Gene | Role | Why it shows up |
|---|---|---|
| *TOP2A*, *TPX2*, *CKS2*, *CCNB1* | cell division (mitosis) | tumors contain many dividing cells |
| *MMP11*, *COL11A1*, *COL10A1* | extracellular-matrix remodeling | tumor-associated (desmoplastic) stroma |
| *CXCL9*, *CXCL10* | immune chemokines | immune infiltration of tumors |
| *FABP4*, *ADIPOQ*, *PLIN1*, *LEP*, *CD36* | adipocyte function | normal breast is fat-rich |
| *LYVE1*, *VEGFD* (*FIGF*) | lymphatic/vascular biology | stromal and vascular composition of normal tissue |
