"""Model construction, I/O shapes, metric calculations and artifact serialization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from oncogene.models.evaluate import (compute_metrics, patient_level_exact_ci, run_learning_curve,
                                      run_nested_cv)
from oncogene.models.interpret import (aggregate_permutation, fold_permutation_importance,
                                       linear_coefficients, tree_importance)
from oncogene.models.train import (MODEL_NAMES, build_estimator, build_pipeline, decision_scores,
                                   load_model, save_model, selected_feature_indices, unwrap)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_fit_and_predict_shapes(name, paired_data, small_cfg):
    X, meta, _ = paired_data
    est = build_estimator(name, small_cfg, n_jobs=1)
    y, g = meta["label"].values, meta["patient_id"].values
    if hasattr(est, "param_grid"):
        est.fit(X.values, y, groups=g)
    else:
        est.fit(X.values, y)
    pred = est.predict(X.values[:7])
    s = decision_scores(est, X.values[:7])
    assert pred.shape == (7,) and set(pred) <= {0, 1}
    assert s.shape == (7,) and np.isfinite(s).all()
    pipe = unwrap(est)
    used = selected_feature_indices(pipe)
    assert used.max() < X.shape[1] and len(np.unique(used)) == len(used)


def test_signal_models_beat_dummy(paired_data, small_cfg):
    X, meta, _ = paired_data
    y, g = meta["label"], meta["patient_id"]
    lr = run_nested_cv("logistic_regression", X, y, g, small_cfg, compute_permutation=False)
    dm = run_nested_cv("dummy", X, y, g, small_cfg, compute_permutation=False)
    assert lr.fold_metrics["roc_auc"].mean() > 0.9
    assert np.allclose(dm.fold_metrics["roc_auc"], 0.5)


def test_compute_metrics_known_values():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    pred = [0, 0, 0, 1, 1, 1, 1, 0]   # TN=3 FP=1 TP=3 FN=1
    score = [0.1, 0.2, 0.3, 0.6, 0.9, 0.8, 0.7, 0.4]
    m = compute_metrics(y, pred, score)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    assert m["accuracy"] == 0.75 and m["specificity"] == 0.75 and m["recall"] == 0.75
    assert m["precision"] == 0.75 and m["f1"] == 0.75 and m["balanced_accuracy"] == 0.75
    assert m["roc_auc"] == pytest.approx(15 / 16)  # only the (0.4 pos, 0.6 neg) pair is misordered


def test_roc_auc_uses_scores_not_labels():
    y = [0, 0, 1, 1]
    pred = [0, 0, 0, 0]  # hard predictions useless...
    m = compute_metrics(y, pred, [0.1, 0.2, 0.3, 0.4])  # ...but scores rank perfectly
    assert m["roc_auc"] == 1.0 and m["recall"] == 0.0


def test_patient_level_exact_ci():
    oof = pd.DataFrame({"patient_id": np.repeat(range(25), 2), "y_true": np.tile([0, 1], 25)})
    oof["y_pred"] = oof["y_true"]
    ci = patient_level_exact_ci(oof)
    assert ci["patients_fully_correct"] == 25
    assert ci["ci95"][1] == 1.0 and 0.85 < ci["ci95"][0] < 0.87   # 0/25 errors -> lower bound ~0.863
    oof.loc[1, "y_pred"] = 0
    assert patient_level_exact_ci(oof)["patients_fully_correct"] == 24


def test_permutation_importance_on_heldout_fold(paired_data, small_cfg):
    X, meta, _ = paired_data
    y = meta["label"].values
    tr = np.arange(0, 20)
    te = np.arange(20, 30)
    pipe = build_pipeline("logistic_regression", small_cfg).set_params(select__k=10).fit(X.values[tr], y[tr])
    sel = selected_feature_indices(pipe)
    pi = fold_permutation_importance(pipe, X.values[te], y[te], sel, n_repeats=5, seed=0)
    assert len(pi) == len(sel) and set(pi["feature_index"]) == set(sel)
    assert np.isfinite(pi["importance_mean"]).all()
    # Signal features (first 10) should dominate importance.
    top = pi.nlargest(5, "importance_mean")["feature_index"]
    assert (top < 10).mean() >= 0.6
    agg = aggregate_permutation(pi.assign(model="lr", probe_id=pi["feature_index"].astype(str)), {"lr": 2})
    assert np.allclose(agg["permutation_importance"].sum(), pi["importance_mean"].sum() / 2)


def test_coefficients_and_tree_importance(paired_data, small_cfg):
    X, meta, _ = paired_data
    y = meta["label"].values
    lr = build_pipeline("logistic_regression", small_cfg).set_params(select__k=10).fit(X.values, y)
    coef = linear_coefficients(lr, X.columns)
    assert len(coef) == 10 and coef["abs_coefficient"].is_monotonic_decreasing
    rf = build_pipeline("random_forest", small_cfg).set_params(select__k=10).fit(X.values, y)
    imp = tree_importance(rf, X.columns)
    assert np.isclose(imp["impurity_importance"].sum(), 1.0)
    assert linear_coefficients(rf, X.columns) is None


def test_model_serialization_roundtrip(tmp_path, paired_data, small_cfg):
    X, meta, _ = paired_data
    est = build_estimator("logistic_regression", small_cfg, n_jobs=1)
    est.fit(X.values, meta["label"].values, groups=meta["patient_id"].values)
    path = save_model(est, "logistic_regression", tmp_path, small_cfg, list(X.columns))
    loaded = load_model(path)
    assert np.allclose(decision_scores(loaded, X.values), decision_scores(est, X.values))
    card = json.loads((tmp_path / "logistic_regression.model_card.json").read_text())
    assert card["seed"] == small_cfg["seed"] and card["n_input_features"] == X.shape[1]
    assert "clf__C" in card["best_params"]
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


def test_learning_curve_shapes(paired_data, small_cfg):
    X, meta, _ = paired_data
    lc = run_learning_curve(X, meta["label"], meta["patient_id"], small_cfg)
    assert set(lc["n_train_patients"]) == {2, 4}
    assert (lc["n_test_patients"] == meta.patient_id.nunique() - lc["n_train_patients"]).all()
