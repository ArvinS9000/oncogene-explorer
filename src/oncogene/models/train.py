"""Model pipelines, final-model fitting and artifact serialization.

Pipeline (identical structure for every model, so comparisons are fair)::

    SimpleImputer(median) -> TopVarianceSelector(N) -> StandardScaler
        -> SelectKBest(f_classif, k) -> classifier

All steps are fitted on the training fold only. The dummy baseline uses the same pipeline
shape so its runtime and inputs are comparable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from oncogene.models.tune import make_search, normalize_grid
from oncogene.preprocessing.transforms import TopVarianceSelector

log = logging.getLogger("oncogene.models.train")

MODEL_NAMES = ["dummy", "logistic_regression", "random_forest", "svm"]
DISPLAY_NAMES = {"dummy": "Dummy (prior)", "logistic_regression": "Logistic regression",
                 "random_forest": "Random forest", "svm": "Support vector machine"}


def _classifier(name: str, cfg: dict[str, Any], seed: int):
    mcfg = cfg["modeling"]["models"].get(name, {})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
                                  random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=int(mcfg.get("n_estimators", 300)),
                                      random_state=seed, n_jobs=1)
    if name == "svm":
        return SVC(kernel="linear", C=1.0, random_state=seed)
    raise KeyError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def build_pipeline(name: str, cfg: dict[str, Any], seed: int | None = None) -> Pipeline:
    seed = cfg["seed"] if seed is None else seed
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", TopVarianceSelector(n_features=cfg["modeling"].get("variance_top_n"))),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k="all" if name == "dummy" else 500)),
        ("clf", _classifier(name, cfg, seed)),
    ])


def build_estimator(name: str, cfg: dict[str, Any], seed: int | None = None, n_jobs: int | None = None):
    """Pipeline wrapped in a patient-grouped inner GridSearchCV when the model has a grid."""
    m = cfg["modeling"]
    grid = normalize_grid(m["models"].get(name, {}).get("param_grid"))
    return make_search(build_pipeline(name, cfg, seed), grid, m["inner_folds"],
                       scoring=m.get("scoring", "roc_auc"),
                       n_jobs=m.get("n_jobs", -1) if n_jobs is None else n_jobs)


def unwrap(estimator) -> Pipeline:
    """Return the fitted Pipeline from a fitted GridSearchCV or Pipeline."""
    return getattr(estimator, "best_estimator_", estimator)


def best_params(estimator) -> dict[str, Any]:
    return dict(getattr(estimator, "best_params_", {}))


def selected_feature_indices(pipe: Pipeline) -> np.ndarray:
    """Indices (into the input columns) of the features that reach the classifier."""
    var_idx = pipe.named_steps["variance"].get_support(indices=True)
    kbest = pipe.named_steps["select"].get_support(indices=True)
    return var_idx[kbest]


def decision_scores(model, X) -> np.ndarray:
    """Continuous scores for ROC-AUC: P(tumor) when available, else decision_function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def save_model(model, name: str, out_dir: Path, cfg: dict[str, Any], feature_names: list[str],
               extra: dict[str, Any] | None = None) -> Path:
    """Persist a fitted model plus a JSON model card describing exactly how it was produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.joblib"
    joblib.dump(model, path)
    pipe = unwrap(model)
    card = {
        "model": name,
        "display_name": DISPLAY_NAMES.get(name, name),
        "model_version": cfg.get("model_version"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "config_hash": config_hash(cfg),
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in best_params(model).items()},
        "n_input_features": len(feature_names),
        "n_features_used": int(len(selected_feature_indices(pipe))),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "purpose": "Final model refit on all 50 samples for interpretation and demonstration. "
                   "Performance estimates come ONLY from nested patient-grouped CV.",
        **(extra or {}),
    }
    (out_dir / f"{name}.model_card.json").write_text(json.dumps(card, indent=2))
    log.info("Saved %s -> %s", name, path)
    return path


def load_model(path: Path):
    """Load a model artifact. Only load artifacts produced by this pipeline (joblib = pickle)."""
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)
