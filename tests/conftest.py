from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from oncogene.config import Paths, load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def small_cfg(cfg):
    """Config scaled down for fast synthetic tests."""
    c = copy.deepcopy(cfg)
    c["modeling"].update(outer_folds=3, outer_repeats=1, inner_folds=3, variance_top_n=100, n_jobs=1)
    c["modeling"]["models"]["logistic_regression"]["param_grid"] = {"select__k": [10], "clf__C": [0.1, 1.0]}
    c["modeling"]["models"]["random_forest"]["n_estimators"] = 25
    c["modeling"]["models"]["random_forest"]["param_grid"] = {"select__k": [10], "clf__max_depth": [3]}
    c["modeling"]["models"]["svm"]["param_grid"] = {"select__k": [10], "clf__C": [0.1]}
    c["interpretation"].update(permutation_repeats=3, bootstrap_iterations=50)
    c["learning_curve"].update(train_patients=[2, 4], repeats=2, params={"select__k": 10, "clf__C": 0.1})
    return c


def make_paired_dataset(n_patients: int = 15, n_features: int = 200, n_signal: int = 10,
                        effect: float = 2.0, seed: int = 0):
    """Synthetic matched-pair data: patient effect + tumor shift on the first n_signal features."""
    rng = np.random.default_rng(seed)
    patient_effect = rng.normal(0, 1, (n_patients, n_features))
    rows, meta = [], []
    for p in range(n_patients):
        for label in (0, 1):
            x = 7 + patient_effect[p] + rng.normal(0, 0.5, n_features)
            if label:
                x[:n_signal] += effect
            rows.append(x)
            sid = f"S{p:02d}{'T' if label else 'N'}"
            meta.append({"sample_id": sid, "patient_id": f"P{p:02d}", "label": label,
                         "tissue_type": "Tumor" if label else "Adjacent normal",
                         "platform_id": "GPL5175", "cancer_stage": "I", "age_years": 50})
    meta = pd.DataFrame(meta).set_index("sample_id")
    X = pd.DataFrame(rows, index=meta.index, columns=[f"{1000 + j}" for j in range(n_features)])
    ann = pd.DataFrame({"gene_symbol": [f"G{j}" for j in range(n_features)],
                        "gene_description": "synthetic", "entrez_id": None, "chromosome": "chr1",
                        "cytoband": None}, index=X.columns)
    ann.index.name = "probe_id"
    return X, meta, ann


@pytest.fixture
def paired_data():
    return make_paired_dataset()


@pytest.fixture(scope="session")
def real_dataset():
    from oncogene.data.load import load_processed
    try:
        return load_processed(Paths())
    except FileNotFoundError:
        pytest.skip("Processed GSE109169 data not available; run `oncogene download preprocess`")
