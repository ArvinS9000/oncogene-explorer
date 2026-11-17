"""Patient-grouped splitting: no patient may appear on both sides of any split."""

from __future__ import annotations

import numpy as np
import pytest

from oncogene.models.evaluate import run_nested_cv
from oncogene.models.tune import (CheckedGroupKFold, PatientLeakageError, RepeatedPatientKFold,
                                  assert_no_patient_leakage)
from tests.conftest import make_paired_dataset


def _groups(n_patients=25):
    return np.repeat([f"P{i}" for i in range(n_patients)], 2)


@pytest.mark.parametrize("n_splits,n_repeats", [(5, 1), (5, 3), (4, 2)])
def test_repeated_patient_kfold_no_shared_patients(n_splits, n_repeats):
    g = _groups()
    cv = RepeatedPatientKFold(n_splits, n_repeats, random_state=1)
    splits = list(cv.iter_splits(g))
    assert len(splits) == n_splits * n_repeats == cv.get_n_splits()
    for s in splits:
        assert not set(g[s.train_idx]) & set(g[s.test_idx])
        assert len(s.train_idx) + len(s.test_idx) == len(g)


def test_each_patient_tested_exactly_once_per_repeat():
    g = _groups()
    for r in range(3):
        tested = np.concatenate([s.test_idx for s in RepeatedPatientKFold(5, 3, 0).iter_splits(g) if s.repeat == r])
        assert sorted(tested.tolist()) == list(range(len(g)))


def test_pairs_stay_together_and_folds_balanced():
    g = _groups()
    y = np.tile([0, 1], 25)
    for s in RepeatedPatientKFold(5, 2, 0).iter_splits(g):
        for pid in set(g[s.test_idx]):
            assert np.isin(np.flatnonzero(g == pid), s.test_idx).all()
        assert y[s.test_idx].sum() * 2 == len(s.test_idx)


def test_splits_are_reproducible_and_repeats_differ():
    g = _groups()
    a = [s.test_idx.tolist() for s in RepeatedPatientKFold(5, 2, 7).iter_splits(g)]
    b = [s.test_idx.tolist() for s in RepeatedPatientKFold(5, 2, 7).iter_splits(g)]
    assert a == b
    assert a[:5] != a[5:]


def test_leakage_guard_raises():
    g = np.array(["A", "A", "B", "B"])
    with pytest.raises(PatientLeakageError):
        assert_no_patient_leakage(g, np.array([0, 2]), np.array([1, 3]))


def test_checked_group_kfold():
    g = _groups(10)
    X = np.zeros((20, 2))
    for tr, te in CheckedGroupKFold(4).split(X, groups=g):
        assert not set(g[tr]) & set(g[te])
    with pytest.raises(ValueError):
        list(CheckedGroupKFold(4).split(X))


def test_too_few_groups():
    with pytest.raises(ValueError):
        list(RepeatedPatientKFold(5).iter_splits(np.array(["a", "a", "b", "b"])))


def test_nested_cv_oof_covers_every_sample_once(paired_data, small_cfg):
    X, meta, _ = paired_data
    res = run_nested_cv("logistic_regression", X, meta["label"], meta["patient_id"], small_cfg,
                        compute_permutation=False)
    oof = res.oof
    assert sorted(oof["sample_id"]) == sorted(X.index)
    for (r, f), g in oof.groupby(["repeat", "fold"]):
        assert g["patient_id"].value_counts().eq(2).all()   # both samples of each test patient


def test_in_fold_selection_prevents_optimistic_bias_on_noise(small_cfg):
    """Pure-noise data: with selection inside CV, held-out AUC must hover around chance.

    (Selecting features on all samples before CV would give AUC close to 1 here.)
    """
    X, meta, _ = make_paired_dataset(n_patients=20, n_features=1500, n_signal=0, effect=0.0, seed=11)
    res = run_nested_cv("logistic_regression", X, meta["label"], meta["patient_id"], small_cfg,
                        compute_permutation=False)
    assert res.fold_metrics["roc_auc"].mean() < 0.75
