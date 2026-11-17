"""Preprocessing transforms, missing-value handling and differential-expression statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from oncogene.analysis.differential_expression import (benjamini_hochberg, paired_de,
                                                       paired_differences)
from oncogene.analysis.pathway_analysis import gene_hit_lists, ora
from oncogene.models.train import build_pipeline
from oncogene.preprocessing.cleaning import normalize_symbol, to_numeric_matrix
from oncogene.preprocessing.transforms import TopVarianceSelector, is_log_scale, log2_if_needed


def test_top_variance_selector_uses_training_data_only():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (20, 5))
    train[:, 2] *= 10
    test = rng.normal(0, 1, (5, 5))
    test[:, 4] *= 100           # huge variance in TEST only must not affect selection
    sel = TopVarianceSelector(n_features=1).fit(train)
    assert sel.get_support(indices=True).tolist() == [2]
    assert sel.transform(test).shape == (5, 1)


def test_top_variance_selector_none_keeps_all():
    X = np.random.default_rng(1).normal(size=(6, 4))
    assert TopVarianceSelector(None).fit_transform(X).shape == (6, 4)


def test_top_variance_selector_rejects_wrong_width():
    sel = TopVarianceSelector(2).fit(np.random.default_rng(2).normal(size=(5, 4)))
    with pytest.raises(ValueError):
        sel.transform(np.zeros((2, 3)))


def test_log_scale_detection():
    assert is_log_scale(np.array([[1.5, 13.9]]))
    assert not is_log_scale(np.array([[10.0, 25000.0]]))
    X = pd.DataFrame([[1.0, 1023.0]])
    out, applied = log2_if_needed(X)
    assert applied and np.allclose(out.values, [[1.0, 10.0]])
    same, applied2 = log2_if_needed(pd.DataFrame([[7.2, 8.1]]))
    assert not applied2 and same.equals(pd.DataFrame([[7.2, 8.1]]))


def test_to_numeric_and_symbol_normalization():
    X = pd.DataFrame({"a": ["1.5", "x"], "b": [2, 3]})
    out = to_numeric_matrix(X)
    assert out["a"].isna().tolist() == [False, True]
    assert normalize_symbol(" esr1 ") == "ESR1"
    assert normalize_symbol("") is None and normalize_symbol(None) is None


def test_pipeline_imputes_missing_values_inside_fit(paired_data, small_cfg):
    X, meta, _ = paired_data
    Xn = X.values.copy()
    Xn[0, 0] = np.nan
    Xn[5, 3] = np.nan
    pipe = build_pipeline("logistic_regression", small_cfg).set_params(select__k=10)
    pipe.fit(Xn, meta["label"].values)
    assert pipe.predict(Xn).shape == (len(meta),)
    # Imputation statistics come from the fitted data, not from anything else.
    assert np.isclose(pipe.named_steps["impute"].statistics_[0], np.nanmedian(Xn[:, 0]))


def test_benjamini_hochberg_matches_statsmodels():
    p = np.random.default_rng(3).uniform(size=200) ** 3
    ours = benjamini_hochberg(p)
    ref = multipletests(p, method="fdr_bh")[1]
    assert np.allclose(ours, ref)
    with_nan = benjamini_hochberg(np.array([0.01, np.nan, 0.04]))
    assert np.isnan(with_nan[1]) and np.allclose(with_nan[[0, 2]], [0.02, 0.04])


def test_paired_differences_are_within_patient(paired_data):
    X, meta, _ = paired_data
    D = paired_differences(X, meta)
    assert D.shape == (meta.patient_id.nunique(), X.shape[1])
    expected = X.loc["S03T"] - X.loc["S03N"]
    assert np.allclose(D.loc["P03"].values, expected.values)


def test_paired_de_matches_scipy_and_finds_signal(paired_data):
    X, meta, ann = paired_data
    de = paired_de(X, meta, ann, fdr_alpha=0.05, min_abs_log2fc=1.0)
    t = meta[meta.label == 1].sort_values("patient_id").index
    n = meta[meta.label == 0].sort_values("patient_id").index
    ref = stats.ttest_rel(X.loc[t].values, X.loc[n].values, axis=0)
    ours = de.loc[X.columns, "p_value"].values
    assert np.allclose(ours, ref.pvalue)
    sig = set(de.index[de["significant"]])
    assert set(X.columns[:10]) <= sig                # all planted signals recovered
    assert len(sig - set(X.columns[:10])) <= 2       # few false positives
    assert (de.loc[X.columns[:10], "log2_fold_change"] > 1.5).all()


def test_paired_de_requires_complete_pairs(paired_data):
    X, meta, ann = paired_data
    with pytest.raises(ValueError):
        paired_de(X.drop(index="S00T"), meta.drop(index="S00T"), ann)


def test_ora_hypergeometric():
    universe = {f"G{i}" for i in range(100)}
    sets = {"enriched": {f"G{i}" for i in range(10)} | {"NOT_ON_ARRAY"},
            "random": {f"G{i}" for i in range(50, 65)}}
    res = ora({f"G{i}" for i in range(8)}, universe, sets, min_size=5, max_size=50).set_index("term")
    assert res.loc["enriched", "overlap"] == 8
    assert res.loc["enriched", "set_size_in_universe"] == 10  # gene outside universe excluded
    assert res.loc["enriched", "p_adj"] < 1e-6
    assert res.loc["random", "p_value"] == 1.0


def test_gene_hit_lists_excludes_conflicts():
    de = pd.DataFrame({"gene_symbol": ["A", "A", "B", "C", None],
                       "log2_fold_change": [2, -2, 3, -3, 5],
                       "significant": [True, True, True, True, True]})
    universe, up, down, conflict = gene_hit_lists(de)
    assert up == {"B"} and down == {"C"} and conflict == {"A"} and universe == {"A", "B", "C"}
