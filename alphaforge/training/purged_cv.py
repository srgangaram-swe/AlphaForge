"""Purged K-Fold and Combinatorial Purged Cross-Validation (CPCV).

Standard K-fold is invalid for overlapping-label time series: a label at date
t spans (t, t+h], so samples near a train/test boundary share information.
Following Lopez de Prado (Advances in Financial Machine Learning, ch. 7 & 12):

- **Purging** removes training dates within ``purge`` days of a test block on
  both sides, eliminating label-interval overlap (set purge >= label horizon).
- **Embargo** drops an extra buffer *after* each test block to kill serial-
  correlation leakage from features computed on trailing windows.
- **CPCV** evaluates every combination of test groups, producing many
  out-of-sample paths instead of one — the input the PBO estimator
  (alphaforge.evaluation.overfitting) needs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd


def _unique_dates(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)).drop_duplicates().sort_values())


def _purged_train_mask(
    n: int, test_blocks: list[tuple[int, int]], purge: int, embargo: int
) -> np.ndarray:
    """True where a date index is usable for training given test blocks."""
    mask = np.ones(n, dtype=bool)
    for start, end in test_blocks:
        lo = max(0, start - purge)
        hi = min(n - 1, end + purge + embargo)
        mask[lo : hi + 1] = False
    return mask


@dataclass(frozen=True)
class PurgedKFold:
    """Contiguous K-fold over dates with purging and embargo."""

    n_splits: int = 5
    purge: int = 0
    embargo: int = 0

    def split(self, dates):
        """Yield (train_dates, test_dates) pairs."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        u = _unique_dates(dates)
        folds = np.array_split(np.arange(len(u)), self.n_splits)
        for fold in folds:
            if len(fold) == 0:
                continue
            mask = _purged_train_mask(len(u), [(fold[0], fold[-1])], self.purge, self.embargo)
            yield u[np.flatnonzero(mask)], u[fold]


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    """CPCV: every C(n_groups, n_test_groups) combination of test groups.

    Each date group appears in many distinct test sets, so OOS predictions can
    be assembled into multiple backtest paths rather than a single trajectory.
    """

    n_groups: int = 8
    n_test_groups: int = 2
    purge: int = 0
    embargo: int = 0

    @property
    def n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    def split(self, dates):
        """Yield (train_dates, test_dates, test_group_ids) triples."""
        if not 0 < self.n_test_groups < self.n_groups:
            raise ValueError("need 0 < n_test_groups < n_groups")
        u = _unique_dates(dates)
        groups = np.array_split(np.arange(len(u)), self.n_groups)
        for combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            blocks = [(groups[c][0], groups[c][-1]) for c in combo if len(groups[c])]
            test_idx = np.concatenate([groups[c] for c in combo if len(groups[c])])
            mask = _purged_train_mask(len(u), blocks, self.purge, self.embargo)
            yield u[np.flatnonzero(mask)], u[test_idx], combo


def run_purged_cv(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_specs: list[dict],
    target: str,
    splitter: PurgedKFold | CombinatorialPurgedCV | None = None,
) -> pd.DataFrame:
    """Fit/predict every model over purged splits; return OOS predictions.

    Complements the chronological walk-forward driver: CPCV predictions feed
    the PBO estimator, while walk-forward remains the primary honest backtest.
    """
    from alphaforge.models.registry import create_model
    from alphaforge.training.walk_forward import ID_COLUMNS, _model_matrix, supervised_frame

    splitter = splitter or PurgedKFold(n_splits=5, purge=20, embargo=5)
    data, x_cols = supervised_frame(features, labels, target)
    blocks = []
    for split_id, split in enumerate(splitter.split(data["date"])):
        train_dates, test_dates = split[0], split[1]
        combo = split[2] if len(split) > 2 else (split_id,)
        train = data[data["date"].isin(train_dates)].dropna(subset=[target])
        test = data[data["date"].isin(test_dates)].dropna(subset=[target])
        if train.empty or test.empty:
            continue
        for spec in model_specs or [{"name": "zero_baseline"}]:
            name = spec["name"]
            model = create_model(name, **spec.get("params", {}))
            model.fit(_model_matrix(train, x_cols, model), train[target].astype(float))
            block = test[ID_COLUMNS + [target]].rename(columns={target: "target"})
            block = block.assign(
                prediction=model.predict(_model_matrix(test, x_cols, model)),
                model=name,
                split_id=split_id,
                test_groups=str(combo),
            )
            blocks.append(block)
    if not blocks:
        return pd.DataFrame(
            columns=ID_COLUMNS + ["target", "prediction", "model", "split_id", "test_groups"]
        )
    return pd.concat(blocks, ignore_index=True)
