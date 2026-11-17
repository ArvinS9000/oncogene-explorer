"""Patient-grouped splitting and hyperparameter search.

Both samples of a patient always land in the same partition. ``assert_no_patient_leakage``
is called on every split the pipeline uses, so a regression would fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline


class PatientLeakageError(AssertionError):
    pass


def assert_no_patient_leakage(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    shared = set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[test_idx])
    if shared:
        raise PatientLeakageError(f"Patients present in both train and test: {sorted(shared)}")


@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


class RepeatedPatientKFold:
    """Repeated K-fold over *patients* (groups), reshuffled each repeat with a fixed seed.

    Each patient appears in exactly one test fold per repeat. With matched pairs every test
    fold is automatically class-balanced.
    """

    def __init__(self, n_splits: int = 5, n_repeats: int = 1, random_state: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits, self.n_repeats, self.random_state = n_splits, n_repeats, random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits * self.n_repeats

    def iter_splits(self, groups: np.ndarray) -> Iterator[Split]:
        groups = np.asarray(groups)
        uniq = np.array(sorted(set(groups)))
        if len(uniq) < self.n_splits:
            raise ValueError(f"Only {len(uniq)} groups for {self.n_splits} folds")
        rng = np.random.default_rng(self.random_state)
        for r in range(self.n_repeats):
            perm = rng.permutation(uniq)
            for f, test_groups in enumerate(np.array_split(perm, self.n_splits)):
                test_mask = np.isin(groups, test_groups)
                tr, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
                assert_no_patient_leakage(groups, tr, te)
                yield Split(r, f, tr, te)

    def split(self, X=None, y=None, groups=None):
        if groups is None:
            raise ValueError("groups (patient IDs) are required")
        for s in self.iter_splits(groups):
            yield s.train_idx, s.test_idx


class CheckedGroupKFold(GroupKFold):
    """GroupKFold that additionally asserts the no-shared-patient invariant."""

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups (patient IDs) are required for inner CV")
        for tr, te in super().split(X, y, groups):
            assert_no_patient_leakage(groups, tr, te)
            yield tr, te


def make_search(pipeline: Pipeline, param_grid: dict | list | None, inner_folds: int,
                scoring: str = "roc_auc", n_jobs: int = -1) -> Pipeline | GridSearchCV:
    """Wrap a pipeline in a patient-grouped GridSearchCV (or return it if there is no grid)."""
    if not param_grid:
        return pipeline
    return GridSearchCV(pipeline, param_grid=param_grid, cv=CheckedGroupKFold(n_splits=inner_folds),
                        scoring=scoring, n_jobs=n_jobs, refit=True, error_score="raise")


def normalize_grid(grid: Any) -> dict | list | None:
    """YAML ``null`` -> Python None inside grids (e.g. max_depth: [null, 5])."""
    if not grid:
        return None
    return grid
