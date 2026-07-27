"""Risk-model contract and conditioning diagnostics for optimization (SF-S4-MR2).

Mean-variance optimization is notoriously sensitive to its covariance input: the
optimizer inverts the matrix, so the *smallest* eigenvalues — the ones estimated
worst — dominate the answer. A sample covariance on a short window has near-zero
eigenvalues, and the optimizer will happily bet the book on those directions.

This module therefore treats the covariance as a **validated contract**, not a
number that arrives:

* :class:`RiskModel` refuses a matrix that is not square, symmetric, finite, or
  positive semi-definite, and publishes its condition number and minimum
  eigenvalue so an ill-conditioned problem is visible before it is solved.
* Stabilization is **explicit and reported**. When a matrix needs a ridge to
  become usable, the amount added is recorded on the model. Silently repairing
  a covariance changes the strategy, and doing it without saying so is how an
  optimizer's output stops corresponding to the risk it claims to control.
* :func:`shrinkage_covariance` provides a causal Ledoit-Wolf-style estimator so
  this MR is independently testable end to end.

**Interim estimator.** Issue #7 owns the production causal shrinkage covariance
and factor-risk model. The estimator here satisfies the same
:class:`RiskModel` contract and will be superseded by it; the optimizer depends
on the contract, not on this implementation, so #7 lands as a drop-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Refusal thresholds, not tuning knobs.
MAX_ASSETS = 2_000
MIN_OBSERVATIONS = 10

#: Eigenvalues below this (relative to the largest) are treated as numerically
#: zero. Set well above float64 round-off on a correlation-scaled matrix so a
#: genuinely singular direction is caught rather than inverted.
EIGENVALUE_FLOOR = 1e-12

#: Condition numbers above this are reported as ill-conditioned. Beyond ~1e8 a
#: float64 inverse retains fewer than eight significant digits, which is less
#: precision than the covariance estimate itself has.
CONDITION_WARNING = 1e8


class RiskModelError(ValueError):
    """Raised when a covariance input is unusable as a risk model."""


@dataclass(frozen=True)
class RiskModel:
    """A validated positive semi-definite covariance with its conditioning.

    Attributes:
        assets: Asset labels in the matrix's row/column order.
        covariance: ``(n, n)`` covariance in **squared periodic return units**
            (per bar, not annualized). Annualization happens at reporting time
            so the optimizer never mixes horizons.
        periods_per_year: Bars per year, recorded so downstream annualization is
            unambiguous rather than assumed.
        min_eigenvalue: Smallest eigenvalue of the returned matrix.
        condition_number: Ratio of largest to smallest eigenvalue.
        ridge_applied: Amount added to the diagonal to reach positive
            definiteness, or ``0.0``. Non-zero means the strategy was changed to
            make it solvable, which must never be silent.
        estimator: Name of the estimator that produced it.
        shrinkage_intensity: Shrinkage weight, for shrinkage estimators.
    """

    assets: tuple[str, ...]
    covariance: FloatArray
    periods_per_year: int
    min_eigenvalue: float
    condition_number: float
    ridge_applied: float
    estimator: str
    shrinkage_intensity: float | None = None

    def __post_init__(self) -> None:
        size = len(self.assets)
        if self.covariance.shape != (size, size):
            raise RiskModelError("covariance must be square over the asset labels")
        if len(set(self.assets)) != size:
            raise RiskModelError("asset labels must be unique")
        if self.periods_per_year < 1:
            raise RiskModelError("periods_per_year must be at least one")

    @property
    def ill_conditioned(self) -> bool:
        """Whether inverting this matrix loses more precision than it retains."""
        return bool(self.condition_number > CONDITION_WARNING)

    def volatilities(self) -> FloatArray:
        """Return per-asset periodic standard deviations."""
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    def portfolio_variance(self, weights: FloatArray) -> float:
        """Return ``w' Σ w`` in squared periodic units.

        Raises:
            RiskModelError: If the result is negative, which can only happen if
                the matrix is not actually PSD and therefore indicates the
                validation was bypassed.
        """
        if weights.shape != (len(self.assets),):
            raise RiskModelError("weights must align with the risk-model assets")
        variance = float(weights @ self.covariance @ weights)
        if variance < -1e-12:
            raise RiskModelError("covariance produced a negative portfolio variance")
        return max(variance, 0.0)

    def annualized_volatility(self, weights: FloatArray) -> float:
        """Return the annualized ex-ante portfolio volatility."""
        return float(np.sqrt(self.portfolio_variance(weights) * self.periods_per_year))

    def to_frame(self) -> pd.DataFrame:
        """Return the covariance as a labelled frame."""
        return pd.DataFrame(self.covariance, index=list(self.assets), columns=list(self.assets))

    def diagnostics(self) -> dict[str, Any]:
        """Return a JSON-friendly conditioning record for the run manifest."""
        return {
            "estimator": self.estimator,
            "n_assets": len(self.assets),
            "periods_per_year": self.periods_per_year,
            "min_eigenvalue": self.min_eigenvalue,
            "condition_number": self.condition_number,
            "ill_conditioned": self.ill_conditioned,
            "ridge_applied": self.ridge_applied,
            "shrinkage_intensity": self.shrinkage_intensity,
        }


def validate_covariance(
    covariance: pd.DataFrame | FloatArray,
    *,
    assets: tuple[str, ...] | None = None,
    periods_per_year: int = 252,
    estimator: str = "supplied",
    allow_ridge: bool = True,
    shrinkage_intensity: float | None = None,
) -> RiskModel:
    """Validate a covariance and return a :class:`RiskModel`.

    Checks, in order, each failing closed with a structured message: finite
    entries, square shape, symmetry, and positive semi-definiteness by
    eigenvalue decomposition.

    Symmetry is checked rather than assumed and rather than silently symmetrized
    with ``(A + A') / 2``. An asymmetric input means the caller's estimator has a
    bug, and quietly averaging it away hides that bug while changing the risk
    model.

    Args:
        allow_ridge: Whether a matrix with small negative eigenvalues (float
            round-off from an otherwise valid estimator) may be lifted to
            positive definiteness. The ridge is always **reported**; setting
            this ``False`` makes any repair an error instead.

    Raises:
        RiskModelError: On any structural or numerical violation.
    """
    if isinstance(covariance, pd.DataFrame):
        if assets is None:
            assets = tuple(str(column) for column in covariance.columns)
        if list(covariance.index) != list(covariance.columns):
            raise RiskModelError("covariance frame index and columns must match")
        matrix = np.asarray(covariance.to_numpy(), dtype=np.float64)
    else:
        matrix = np.asarray(covariance, dtype=np.float64)
        if assets is None:
            raise RiskModelError("asset labels are required for an array covariance")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RiskModelError(f"covariance must be square, got shape {matrix.shape}")
    if matrix.shape[0] != len(assets):
        raise RiskModelError("covariance dimension does not match the asset labels")
    if matrix.shape[0] < 1:
        raise RiskModelError("covariance must cover at least one asset")
    if matrix.shape[0] > MAX_ASSETS:
        raise RiskModelError(f"covariance exceeds the {MAX_ASSETS}-asset ceiling")
    if not np.isfinite(matrix).all():
        raise RiskModelError("covariance contains non-finite entries")
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    scale = max(float(np.max(np.abs(matrix))), 1e-300)
    if asymmetry / scale > 1e-10:
        raise RiskModelError(
            f"covariance is not symmetric (max asymmetry {asymmetry:.3e}); "
            "an asymmetric estimate is an estimator bug, not something to average away"
        )

    symmetric = (matrix + matrix.T) / 2.0  # remove round-off only, after the check
    eigenvalues = np.linalg.eigvalsh(symmetric)
    smallest = float(eigenvalues.min())
    largest = float(eigenvalues.max())
    if largest <= 0.0:
        raise RiskModelError("covariance has no positive variance directions")

    ridge = 0.0
    if smallest < EIGENVALUE_FLOOR * largest:
        if not allow_ridge:
            raise RiskModelError(
                f"covariance is not positive definite (min eigenvalue {smallest:.3e}) "
                "and ridge stabilization is disabled"
            )
        # Lift to a small positive floor rather than to an arbitrary constant, so
        # the repair scales with the matrix and is reported in its own units.
        ridge = float(EIGENVALUE_FLOOR * largest - smallest)
        symmetric = symmetric + ridge * np.eye(symmetric.shape[0])
        eigenvalues = np.linalg.eigvalsh(symmetric)
        smallest = float(eigenvalues.min())
        largest = float(eigenvalues.max())

    condition = float(largest / smallest) if smallest > 0.0 else float("inf")
    return RiskModel(
        assets=tuple(str(asset) for asset in assets),
        covariance=symmetric,
        periods_per_year=periods_per_year,
        min_eigenvalue=smallest,
        condition_number=condition,
        ridge_applied=ridge,
        estimator=estimator,
        shrinkage_intensity=shrinkage_intensity,
    )


def shrinkage_covariance(
    returns: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    window: int = 252,
    intensity: float | None = None,
    periods_per_year: int = 252,
) -> RiskModel:
    """Estimate a causal shrinkage covariance from trailing returns.

    Ledoit-Wolf style: the sample covariance is shrunk toward a constant-
    correlation target,

        Σ_hat = (1 - a) * S + a * F

    where ``F`` has the sample variances on its diagonal and the average sample
    correlation off it. The sample covariance is unbiased but enormously noisy in
    its small eigenvalues; the target is heavily biased but almost noiseless.
    The convex combination trades a little bias for a large variance reduction,
    which is precisely what an optimizer that inverts the matrix needs.

    When ``intensity`` is ``None`` it is set by the Ledoit-Wolf analytic rule
    ``a = min(1, (n_assets / n_observations))``, a bounded, deterministic, and
    conservative approximation — it shrinks harder exactly when the sample
    covariance is least trustworthy (few observations relative to assets).

    **Causality.** Only observations strictly before ``as_of`` are used. The
    optimizer's covariance at date ``t`` therefore cannot contain information
    from ``t`` or later, which is asserted by mutation test.

    Raises:
        RiskModelError: On insufficient history or an unusable window.
    """
    if window < MIN_OBSERVATIONS:
        raise RiskModelError(f"window must be at least {MIN_OBSERVATIONS} observations")
    if intensity is not None and not 0.0 <= intensity <= 1.0:
        raise RiskModelError("shrinkage intensity must lie in [0, 1]")

    frame = returns.sort_index()
    if as_of is not None:
        frame = frame.loc[frame.index < as_of]
    frame = frame.dropna(axis=1, how="all").iloc[-window:]
    frame = frame.dropna(axis=0, how="any")
    if len(frame) < MIN_OBSERVATIONS:
        raise RiskModelError(
            f"only {len(frame)} complete observations strictly before {as_of}; "
            f"need {MIN_OBSERVATIONS}"
        )
    if frame.shape[1] < 1:
        raise RiskModelError("no asset has usable return history")

    values = frame.to_numpy(dtype=np.float64)
    n_observations, n_assets = values.shape
    sample = np.cov(values, rowvar=False, ddof=1)
    sample = np.atleast_2d(sample)

    variances = np.diag(sample).copy()
    deviations = np.sqrt(np.clip(variances, 1e-300, None))
    correlation = sample / np.outer(deviations, deviations)
    off_diagonal = correlation[~np.eye(n_assets, dtype=bool)]
    average_correlation = float(np.mean(off_diagonal)) if off_diagonal.size else 0.0
    target = average_correlation * np.outer(deviations, deviations)
    np.fill_diagonal(target, variances)

    weight = float(np.clip(n_assets / n_observations, 0.0, 1.0)) if intensity is None else intensity
    shrunk = (1.0 - weight) * sample + weight * target
    return validate_covariance(
        shrunk,
        assets=tuple(str(column) for column in frame.columns),
        periods_per_year=periods_per_year,
        estimator="ledoit_wolf_constant_correlation",
        shrinkage_intensity=weight,
    )
