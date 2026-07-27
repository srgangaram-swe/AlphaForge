"""Constrained Markowitz mean-variance optimization (SF-S4-MR2).

Four formulations over one shared contract, solved by a from-scratch accelerated
projected-gradient method so the mathematics, conditioning, and failure modes are
inspectable rather than delegated to a package.

The objective, in periodic (per-bar) units throughout:

    minimize   (risk_aversion / 2) * w' Σ w  -  mu' w  +  cost(w, w_prev)
    subject to w in C

where ``C`` is the intersection of the declared limits (budget, gross, net,
per-position, long-only, turnover, liquidity, and sector/factor exposure).
``cost`` is a linear turnover charge plus an optional quadratic impact term.

Each formulation is a choice of ``mu`` and ``risk_aversion``:

===================== ============== ============================================
formulation           mu             risk_aversion
===================== ============== ============================================
``minimum_variance``  0              1 (the mean term vanishes)
``target_return``     alpha          solved by bisection on the return constraint
``maximum_utility``   alpha          declared by the caller
``alpha_risk_cost``   alpha          declared, with costs inside the objective
===================== ============== ============================================

**Why projected gradient rather than a QP package.** The issue requires the
mathematics to be exposed, and the constraint set is one this repository already
projects onto exactly (SF-S4-MR1). Accelerated projected gradient converges at
``O(1/k^2)`` on a convex quadratic, needs no extra dependency, is deterministic
with a fixed step and iteration cap, and its step size follows analytically from
the largest eigenvalue of ``Σ``. The cost is that it reaches a tolerance rather
than an exact vertex; that tolerance is declared, checked, and reported.

**Nothing is trusted.** After solving, :func:`audit_solution` independently
recomputes feasibility and every objective component from the returned weights,
using code that does not share the solver's internals. A solution that fails its
own audit is returned as a failure, never as a portfolio.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alphaforge.optimization.risk_model import RiskModel, RiskModelError
from alphaforge.portfolio.contracts import (
    CONSTRAINT_TOLERANCE,
    InfeasibleConstraintsError,
    PortfolioConstraints,
    PortfolioError,
)

FloatArray = NDArray[np.float64]

Formulation = Literal["minimum_variance", "target_return", "maximum_utility", "alpha_risk_cost"]

SolverStatus = Literal["optimal", "max_iterations", "infeasible", "failed"]

#: Refusal thresholds, not tuning knobs.
MAX_ITERATIONS = 20_000
MAX_BISECTION_STEPS = 60

#: Declared convergence tolerance on the projected-gradient fixed-point residual,
#: ``||w_k - P(w_k - t grad f(w_k))|| / max(1, ||w_k||)``. At 1e-9 the weights
#: are stable far below any economically meaningful position size.
DEFAULT_TOLERANCE = 1e-9

#: Tolerance for the independent post-solve audit. Deliberately looser than the
#: solver tolerance so the audit tests feasibility, not float reproducibility.
AUDIT_TOLERANCE = 1e-7


class OptimizerError(PortfolioError):
    """Raised when an optimization request is unusable or fails closed."""


@dataclass(frozen=True)
class ExposureConstraint:
    """A bounded linear exposure, e.g. a sector or factor loading.

    ``lower <= loadings' w <= upper``. Loadings are per asset in the problem's
    asset order; a sector membership is just a 0/1 loading vector.
    """

    name: str
    loadings: FloatArray
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise OptimizerError("exposure constraint requires a name")
        if self.loadings.ndim != 1:
            raise OptimizerError(f"exposure {self.name!r} loadings must be one-dimensional")
        if not np.isfinite(self.loadings).all():
            raise OptimizerError(f"exposure {self.name!r} loadings must be finite")
        if not (np.isfinite(self.lower) and np.isfinite(self.upper)):
            raise OptimizerError(f"exposure {self.name!r} bounds must be finite")
        if self.upper < self.lower:
            raise OptimizerError(f"exposure {self.name!r} upper bound is below its lower bound")

    def value(self, weights: FloatArray) -> float:
        """Return the exposure achieved by ``weights``."""
        return float(self.loadings @ weights)

    def violation(self, weights: FloatArray) -> float:
        """Return the amount by which ``weights`` breach this bound, or ``0``."""
        achieved = self.value(weights)
        return float(max(self.lower - achieved, achieved - self.upper, 0.0))


@dataclass(frozen=True)
class CostModel:
    """Turnover cost entering the objective, in periodic return units.

    Attributes:
        linear_bps: Proportional cost charged on ``|w - w_prev|``, in basis
            points. This is the spread/commission term.
        quadratic_bps: Impact-like term charged on ``(w - w_prev)^2``, in basis
            points per unit of squared turnover. A crude stand-in for market
            impact whose real model is SF-S4-MR4; documented as such rather than
            presented as calibrated.
    """

    linear_bps: float = 0.0
    quadratic_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("linear_bps", "quadratic_bps"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise OptimizerError(f"{name} must be finite and non-negative")

    def value(self, weights: FloatArray, previous: FloatArray) -> float:
        """Return the total cost charged for trading into ``weights``."""
        trade = weights - previous
        linear = self.linear_bps / 10_000.0 * float(np.sum(np.abs(trade)))
        quadratic = self.quadratic_bps / 10_000.0 * float(np.sum(trade**2))
        return linear + quadratic

    def smooth_gradient(self, weights: FloatArray, previous: FloatArray) -> FloatArray:
        """Return the gradient of the **differentiable** part of the cost.

        Only the quadratic impact term. The linear turnover charge is an L1
        penalty around ``previous`` and is non-differentiable exactly where the
        solution wants to sit — at zero trade. Treating it by subgradient
        descent makes the iterate oscillate across that kink and *increases*
        turnover, which is the opposite of what the penalty is for. It is
        handled instead by :meth:`proximal_step`.
        """
        return np.asarray(
            2.0 * self.quadratic_bps / 10_000.0 * (weights - previous), dtype=np.float64
        )

    def proximal_step(self, weights: FloatArray, previous: FloatArray, step: float) -> FloatArray:
        """Apply the proximal operator of the linear turnover charge.

        Soft-thresholding toward ``previous``: any trade smaller than
        ``step * rate`` is not worth its cost and is snapped to zero. This is the
        exact prox of an L1 penalty centred on the previous book, and it is what
        makes a turnover penalty actually suppress small trades rather than
        merely re-price them.
        """
        rate = self.linear_bps / 10_000.0
        if rate <= 0.0:
            return weights
        trade = weights - previous
        shrunk = np.sign(trade) * np.maximum(np.abs(trade) - step * rate, 0.0)
        return np.asarray(previous + shrunk, dtype=np.float64)


@dataclass(frozen=True)
class MeanVarianceProblem:
    """A fully specified, deterministic optimization request.

    Attributes:
        risk_model: Validated PSD covariance in periodic units.
        expected_returns: Per-asset alpha in the **same periodic units** as the
            covariance. Mixing an annualized alpha with a daily covariance is the
            most common unit error in mean-variance code, so both are stated and
            the risk model carries its own ``periods_per_year``.
        constraints: The declared limit set.
        previous_weights: Book before the rebalance, for turnover and cost.
        exposures: Sector or factor bounds.
        cost_model: Turnover cost inside the objective.
        risk_aversion: Coefficient on the variance term.
        target_return: Required periodic return, for ``target_return``.
        liquidity_caps: Per-asset absolute weight ceilings.
        budget: Required **equality** on the summed weights, ``1'w = budget``.
            This is the constraint that makes Markowitz well posed: the variance
            objective is minimized at ``w = 0``, so without a budget the
            "minimum-variance portfolio" is an empty book. ``1.0`` is fully
            invested; ``0.0`` is dollar-neutral. ``None`` leaves the problem
            inequality-constrained only, which is meaningful for utility
            maximization but not for minimum variance.
    """

    risk_model: RiskModel
    expected_returns: pd.Series | None = None
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    previous_weights: pd.Series | None = None
    exposures: tuple[ExposureConstraint, ...] = ()
    cost_model: CostModel = field(default_factory=CostModel)
    risk_aversion: float = 1.0
    target_return: float | None = None
    liquidity_caps: pd.Series | None = None
    budget: float | None = None

    def __post_init__(self) -> None:
        if self.budget is not None:
            if not np.isfinite(self.budget):
                raise OptimizerError("budget must be finite when supplied")
            if abs(self.budget) > self.constraints.max_net + CONSTRAINT_TOLERANCE:
                raise OptimizerError(
                    f"budget {self.budget} cannot satisfy max_net {self.constraints.max_net}"
                )
            if abs(self.budget) > self.constraints.max_gross + CONSTRAINT_TOLERANCE:
                raise OptimizerError(
                    f"budget {self.budget} cannot satisfy max_gross {self.constraints.max_gross}"
                )
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0.0:
            raise OptimizerError("risk_aversion must be finite and positive")
        size = len(self.risk_model.assets)
        if self.expected_returns is not None:
            if list(self.expected_returns.index) != list(self.risk_model.assets):
                raise OptimizerError(
                    "expected_returns must be indexed by the risk-model assets in order"
                )
            if not np.isfinite(self.expected_returns.to_numpy(dtype=float)).all():
                raise OptimizerError("expected_returns must be finite")
        for exposure in self.exposures:
            if exposure.loadings.shape != (size,):
                raise OptimizerError(
                    f"exposure {exposure.name!r} loadings must cover {size} assets"
                )
        if self.target_return is not None and not np.isfinite(self.target_return):
            raise OptimizerError("target_return must be finite when supplied")

    @property
    def assets(self) -> tuple[str, ...]:
        """Asset labels in problem order."""
        return self.risk_model.assets

    def alpha(self) -> FloatArray:
        """Return expected returns as an array, zero when absent."""
        if self.expected_returns is None:
            return np.zeros(len(self.assets))
        return np.asarray(self.expected_returns.to_numpy(), dtype=np.float64)

    def previous(self) -> FloatArray:
        """Return the previous book aligned to problem order, zero when absent."""
        if self.previous_weights is None:
            return np.zeros(len(self.assets))
        return np.asarray(
            self.previous_weights.reindex(list(self.assets)).fillna(0.0).to_numpy(),
            dtype=np.float64,
        )

    def caps(self) -> FloatArray | None:
        """Return liquidity caps aligned to problem order, or ``None``."""
        if self.liquidity_caps is None:
            return None
        return np.asarray(
            self.liquidity_caps.reindex(list(self.assets)).fillna(0.0).to_numpy(),
            dtype=np.float64,
        )

    @property
    def identity(self) -> str:
        """Deterministic SHA-256 over every input that can change the solution.

        Includes the covariance and alpha *values*, not just their shapes, so two
        runs that produced different numbers can never be confused for one — the
        property that makes a stored allocation traceable to its inputs.
        """
        payload = {
            "assets": list(self.assets),
            "covariance": np.round(self.risk_model.covariance, 15).tolist(),
            "periods_per_year": self.risk_model.periods_per_year,
            "expected_returns": np.round(self.alpha(), 15).tolist(),
            "previous_weights": np.round(self.previous(), 15).tolist(),
            "constraints": self.constraints.to_dict(),
            "exposures": [
                {
                    "name": exposure.name,
                    "loadings": np.round(exposure.loadings, 15).tolist(),
                    "lower": exposure.lower,
                    "upper": exposure.upper,
                }
                for exposure in self.exposures
            ],
            "cost_model": {
                "linear_bps": self.cost_model.linear_bps,
                "quadratic_bps": self.cost_model.quadratic_bps,
            },
            "risk_aversion": self.risk_aversion,
            "target_return": self.target_return,
            "budget": self.budget,
            "liquidity_caps": (
                None if (caps := self.caps()) is None else np.round(caps, 15).tolist()
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SolverResult:
    """Weights plus everything needed to audit how they were produced."""

    weights: pd.Series
    formulation: str
    status: SolverStatus
    iterations: int
    max_iterations: int
    residual: float
    tolerance: float
    objective: float
    variance_term: float
    return_term: float
    cost_term: float
    ex_ante_volatility: float
    expected_return: float
    turnover: float
    gross: float
    net: float
    active_constraints: tuple[str, ...]
    problem_identity: str
    risk_diagnostics: dict[str, Any]
    audit_passed: bool
    audit_violations: tuple[str, ...] = ()

    @property
    def converged(self) -> bool:
        """Whether a stopping criterion fired rather than the budget expiring."""
        return self.status == "optimal"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record for the research ledger."""
        return {
            "formulation": self.formulation,
            "status": self.status,
            "converged": self.converged,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "objective": self.objective,
            "variance_term": self.variance_term,
            "return_term": self.return_term,
            "cost_term": self.cost_term,
            "ex_ante_volatility": self.ex_ante_volatility,
            "expected_return": self.expected_return,
            "turnover": self.turnover,
            "gross": self.gross,
            "net": self.net,
            "active_constraints": list(self.active_constraints),
            "problem_identity": self.problem_identity,
            "risk": self.risk_diagnostics,
            "audit_passed": self.audit_passed,
            "audit_violations": list(self.audit_violations),
        }


# ---------------------------------------------------------------------------
# Projection including exposure constraints
# ---------------------------------------------------------------------------


def _project_l1_ball(weights: FloatArray, radius: float) -> FloatArray:
    """Euclidean projection onto ``{w : ||w||_1 <= radius}`` by soft-thresholding.

    Radially rescaling an over-sized vector (``w * radius / ||w||_1``) lands *on*
    the ball but is **not** the nearest point in it. Alternating a non-projection
    with genuine projections breaks the convergence guarantee of the method, and
    the symptom is exactly what it sounds like: the iteration stalls on a
    knife-edge set such as the simplex and returns an infeasible point.

    The true projection subtracts a single threshold from every magnitude:
    ``theta >= 0`` chosen so ``sum(max(|w_i| - theta, 0)) = radius``, found here
    by the standard exact sort-and-scan.
    """
    total = float(np.sum(np.abs(weights)))
    if total <= radius + CONSTRAINT_TOLERANCE or total <= 0.0:
        return weights
    magnitudes = np.sort(np.abs(weights))[::-1]
    cumulative = np.cumsum(magnitudes)
    indices = np.arange(1, magnitudes.size + 1)
    candidates = (cumulative - radius) / indices
    valid = magnitudes - candidates > 0
    threshold = float(candidates[valid][-1]) if valid.any() else 0.0
    return np.asarray(
        np.sign(weights) * np.maximum(np.abs(weights) - threshold, 0.0), dtype=np.float64
    )


def _is_feasible(weights: FloatArray, problem: MeanVarianceProblem) -> bool:
    """Return whether every declared limit currently holds."""
    return audit_solution(weights, problem)[0]


def _project(
    weights: FloatArray, problem: MeanVarianceProblem, *, iterations: int = 500
) -> FloatArray:
    """Project onto the optimizer's feasible set.

    Deliberately **not** SF-S4-MR1's ``project_to_feasible``. That routine
    waterfills gross exposure *up* to its target, which is right for a ranking
    policy whose job is to deploy the capital it was given. Here gross is an
    inequality the objective is entitled to leave slack: a minimum-variance book
    that wants to hold less should be allowed to, and forcing it to full gross
    would silently change the problem being solved. The constraint names are the
    same; the semantics are not.

    Alternating projection over: the box (position cap, long-only, liquidity),
    the L1 ball (gross, shrink-only), the net band, the turnover ball around the
    previous book, the budget hyperplane, and each exposure half-space. All are
    convex, so alternating converges into the intersection when one exists.
    """
    constraints = problem.constraints
    size = weights.size
    caps = np.full(size, constraints.max_position, dtype=np.float64)
    liquidity = problem.caps()
    if liquidity is not None:
        caps = np.minimum(caps, liquidity)
    lower = np.zeros(size) if constraints.long_only else -caps
    # Gross is bounded by the declared policy target and the leverage ceiling,
    # scaled by deployable capital. Note this is *not* MR1's
    # min(max_gross, deployable, max_leverage): taking `deployable` as a hard cap
    # would pin gross at 1.0 and make a levered mandate unreachable, silently
    # solving a different problem than the one declared.
    gross_limit = min(constraints.max_gross, constraints.max_leverage) * constraints.deployable
    previous = problem.previous()

    current = np.asarray(weights, dtype=np.float64).copy()
    for _ in range(iterations):
        before = current.copy()
        current = np.clip(current, lower, caps)

        current = _project_l1_ball(current, gross_limit)

        net = float(np.sum(current))
        if abs(net) > constraints.max_net + CONSTRAINT_TOLERANCE:
            # Trim the offending side proportionally, preserving its ranking.
            excess = abs(net) - constraints.max_net
            side = current > 0.0 if net > 0.0 else current < 0.0
            magnitude = float(np.sum(np.abs(current[side])))
            if magnitude > 0.0:
                current[side] *= max(1.0 - excess / magnitude, 0.0)

        if constraints.max_turnover is not None:
            trade = current - previous
            turnover = float(np.sum(np.abs(trade)))
            if turnover > constraints.max_turnover + CONSTRAINT_TOLERANCE and turnover > 0.0:
                current = previous + trade * (constraints.max_turnover / turnover)

        for exposure in problem.exposures:
            achieved = exposure.value(current)
            norm = float(exposure.loadings @ exposure.loadings)
            if norm <= 0.0:
                continue
            if achieved > exposure.upper + CONSTRAINT_TOLERANCE:
                current = current - ((achieved - exposure.upper) / norm) * exposure.loadings
            elif achieved < exposure.lower - CONSTRAINT_TOLERANCE:
                current = current - ((achieved - exposure.lower) / norm) * exposure.loadings

        if problem.budget is not None:
            # Exact Euclidean projection onto {w : 1'w = budget}. Applied last so
            # the budget equality — the one constraint that is exact rather than
            # an inequality — holds on exit.
            drift = float(np.sum(current)) - problem.budget
            if abs(drift) > CONSTRAINT_TOLERANCE:
                current = current - drift / size

        # Exit on *feasibility*, not on a small step. A slowly cycling
        # alternation can stall while still violating a limit, and returning
        # that point would hand back an infeasible book that merely stopped
        # moving. When the intersection is empty this loop runs its full budget
        # and the caller reports infeasible.
        if _is_feasible(current, problem):
            break
        if float(np.max(np.abs(current - before))) <= 1e-15:
            break
    return current


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _objective_parts(
    weights: FloatArray, problem: MeanVarianceProblem
) -> tuple[float, float, float]:
    """Return ``(variance_term, return_term, cost_term)`` of the objective."""
    variance = problem.risk_model.portfolio_variance(weights)
    variance_term = 0.5 * problem.risk_aversion * variance
    return_term = float(problem.alpha() @ weights)
    cost_term = problem.cost_model.value(weights, problem.previous())
    return variance_term, return_term, cost_term


def _gradient(weights: FloatArray, problem: MeanVarianceProblem) -> FloatArray:
    """Return the objective gradient ``lambda * Σ w - mu + grad cost``."""
    return np.asarray(
        problem.risk_aversion * (problem.risk_model.covariance @ weights)
        - problem.alpha()
        + problem.cost_model.smooth_gradient(weights, problem.previous()),
        dtype=np.float64,
    )


def _solve_projected_gradient(
    problem: MeanVarianceProblem, *, max_iterations: int, tolerance: float
) -> tuple[FloatArray, int, float, SolverStatus]:
    """Accelerated projected gradient (FISTA) on the convex quadratic objective.

    The step size is ``1 / L`` with ``L`` the Lipschitz constant of the gradient,
    which for this objective is ``risk_aversion * lambda_max(Σ)`` plus the
    quadratic cost curvature. Deriving it analytically rather than line-searching
    keeps the solve deterministic — a line search introduces data-dependent
    branching and two runs on the same inputs could then differ.
    """
    size = len(problem.assets)
    lipschitz = problem.risk_aversion * float(
        np.linalg.eigvalsh(problem.risk_model.covariance).max()
    )
    lipschitz += 2.0 * problem.cost_model.quadratic_bps / 10_000.0
    if not np.isfinite(lipschitz) or lipschitz <= 0.0:
        lipschitz = 1.0
    step = 1.0 / lipschitz

    # Start from a budget-feasible point: an equal-weight book when a budget is
    # required, zeros otherwise. Starting at zero under a budget constraint puts
    # the first iterate on the wrong side of an equality and wastes the warm start.
    start = np.full(size, problem.budget / size) if problem.budget is not None else np.zeros(size)
    current = _project(start, problem)
    if not _is_feasible(current, problem):
        return current, 0, float("inf"), "infeasible"
    momentum = current.copy()
    theta = 1.0
    residual = float("inf")

    for iteration in range(1, max_iterations + 1):
        stepped = momentum - step * _gradient(momentum, problem)
        stepped = problem.cost_model.proximal_step(stepped, problem.previous(), step)
        candidate = _project(stepped, problem)
        residual = float(np.linalg.norm(candidate - current) / max(1.0, np.linalg.norm(current)))
        theta_next = (1.0 + np.sqrt(1.0 + 4.0 * theta**2)) / 2.0
        momentum = candidate + ((theta - 1.0) / theta_next) * (candidate - current)
        current, theta = candidate, theta_next
        if residual < tolerance:
            return current, iteration, residual, "optimal"
    return current, max_iterations, residual, "max_iterations"


def solve_mean_variance(
    problem: MeanVarianceProblem,
    *,
    formulation: Formulation = "maximum_utility",
    max_iterations: int = 5_000,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SolverResult:
    """Solve one constrained mean-variance problem and audit the result.

    Args:
        formulation: See the module docstring's table.
        max_iterations: Iteration budget, at most :data:`MAX_ITERATIONS`.
        tolerance: Fixed-point residual below which the solve is optimal.

    Returns:
        A :class:`SolverResult` whose ``status`` is ``optimal`` only if a
        stopping criterion fired *and* the independent audit passed.

    Raises:
        OptimizerError: On an unusable request — non-finite inputs, a missing
            alpha where the formulation requires one, an out-of-range budget, or
            an infeasible constraint set. **No silent relaxation ever occurs**;
            an infeasible problem is reported as infeasible.
    """
    if not 1 <= max_iterations <= MAX_ITERATIONS:
        raise OptimizerError(f"max_iterations must be in [1, {MAX_ITERATIONS}]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise OptimizerError("tolerance must be finite and positive")
    if formulation in {"target_return", "maximum_utility", "alpha_risk_cost"} and (
        problem.expected_returns is None
    ):
        raise OptimizerError(f"formulation {formulation!r} requires expected_returns")
    if formulation == "target_return" and problem.target_return is None:
        raise OptimizerError("formulation 'target_return' requires a target_return")

    try:
        problem.constraints.check_feasible(len(problem.assets), problem.caps())
    except InfeasibleConstraintsError as exc:
        raise OptimizerError(f"constraint set is infeasible: {exc}") from exc

    working = problem
    if formulation == "minimum_variance":
        working = _replace(problem, expected_returns=None, cost_model=CostModel())
    elif formulation == "maximum_utility":
        working = _replace(problem, cost_model=CostModel())

    if formulation == "target_return":
        weights, iterations, residual, status = _solve_target_return(
            working, max_iterations=max_iterations, tolerance=tolerance
        )
    else:
        weights, iterations, residual, status = _solve_projected_gradient(
            working, max_iterations=max_iterations, tolerance=tolerance
        )

    return _package(
        weights,
        problem,
        formulation=formulation,
        status=status,
        iterations=iterations,
        max_iterations=max_iterations,
        residual=residual,
        tolerance=tolerance,
    )


def _replace(problem: MeanVarianceProblem, **changes: Any) -> MeanVarianceProblem:
    """Return a copy of ``problem`` with the named fields replaced."""
    from dataclasses import replace

    return replace(problem, **changes)


def _solve_target_return(
    problem: MeanVarianceProblem, *, max_iterations: int, tolerance: float
) -> tuple[FloatArray, int, float, SolverStatus]:
    """Minimize variance subject to ``mu'w >= target``.

    Expressed as what it is — a linear inequality — rather than by sweeping the
    risk-aversion coefficient. Sweeping is the textbook trick, but it requires
    driving ``lambda`` toward zero to reach high-return corners, and at
    ``lambda -> 0`` the objective's curvature vanishes, the analytic step
    ``1 / (lambda * lambda_max(Sigma))`` explodes, and the iteration stops being
    numerically meaningful. Adding the return requirement as a half-space keeps
    the objective well conditioned and makes the constraint exact.

    Infeasibility is then detected honestly: if the returned book still violates
    the return half-space, the target is unreachable inside the constraint set
    and the status is ``infeasible`` — never a silently relaxed answer.
    """
    target = problem.target_return
    assert target is not None  # guarded by the caller
    alpha = problem.alpha()
    # A finite upper bound that can never bind: the largest return any feasible
    # book could attain is bounded by the L1 norm of alpha times gross.
    ceiling = float(np.abs(alpha).sum()) * max(problem.constraints.max_gross, 1.0) + 1.0
    requirement = ExposureConstraint(
        name="__target_return__", loadings=alpha, lower=target, upper=ceiling
    )
    constrained = _replace(
        problem,
        exposures=(*problem.exposures, requirement),
        expected_returns=None,
        cost_model=CostModel(),
    )
    weights, iterations, residual, status = _solve_projected_gradient(
        constrained, max_iterations=max_iterations, tolerance=tolerance
    )
    if requirement.violation(weights) > AUDIT_TOLERANCE or not _is_feasible(weights, problem):
        return weights, iterations, residual, "infeasible"
    return weights, iterations, residual, status


def _package(
    weights: FloatArray,
    problem: MeanVarianceProblem,
    *,
    formulation: str,
    status: SolverStatus,
    iterations: int,
    max_iterations: int,
    residual: float,
    tolerance: float,
) -> SolverResult:
    """Assemble the result and run the independent audit."""
    variance_term, return_term, cost_term = _objective_parts(weights, problem)
    previous = problem.previous()
    passed, violations = audit_solution(weights, problem)
    final_status: SolverStatus = status
    if not passed and status == "optimal":
        # A solution that fails its own audit is a failure, not a portfolio.
        final_status = "failed"
    return SolverResult(
        weights=pd.Series(weights, index=list(problem.assets), name="target_weight"),
        formulation=formulation,
        status=final_status,
        iterations=iterations,
        max_iterations=max_iterations,
        residual=residual,
        tolerance=tolerance,
        objective=variance_term - return_term + cost_term,
        variance_term=variance_term,
        return_term=return_term,
        cost_term=cost_term,
        ex_ante_volatility=problem.risk_model.annualized_volatility(weights),
        expected_return=return_term,
        turnover=float(np.sum(np.abs(weights - previous))),
        gross=float(np.sum(np.abs(weights))),
        net=float(np.sum(weights)),
        active_constraints=_active_constraints(weights, problem),
        problem_identity=problem.identity,
        risk_diagnostics=problem.risk_model.diagnostics(),
        audit_passed=passed,
        audit_violations=violations,
    )


def _active_constraints(weights: FloatArray, problem: MeanVarianceProblem) -> tuple[str, ...]:
    """Return the limits sitting at their boundary, for diagnosis.

    A solution pinned against many constraints is being determined by the limit
    set rather than by the forecast, which is worth knowing before reading its
    expected return as skill.
    """
    constraints = problem.constraints
    active: list[str] = []
    if np.any(np.abs(np.abs(weights) - constraints.max_position) < AUDIT_TOLERANCE):
        active.append("max_position")
    if problem.budget is not None:
        active.append("budget")
    gross_target = min(constraints.max_gross, constraints.max_leverage) * constraints.deployable
    if abs(float(np.sum(np.abs(weights))) - gross_target) < AUDIT_TOLERANCE:
        active.append("max_gross")
    if abs(abs(float(np.sum(weights))) - constraints.max_net) < AUDIT_TOLERANCE:
        active.append("max_net")
    if constraints.max_turnover is not None:
        turnover = float(np.sum(np.abs(weights - problem.previous())))
        if abs(turnover - constraints.max_turnover) < AUDIT_TOLERANCE:
            active.append("max_turnover")
    for exposure in problem.exposures:
        achieved = exposure.value(weights)
        if (
            abs(achieved - exposure.upper) < AUDIT_TOLERANCE
            or abs(achieved - exposure.lower) < AUDIT_TOLERANCE
        ):
            active.append(f"exposure:{exposure.name}")
    return tuple(active)


def audit_solution(
    weights: FloatArray, problem: MeanVarianceProblem
) -> tuple[bool, tuple[str, ...]]:
    """Independently re-verify feasibility from the returned weights alone.

    Deliberately written without reference to the solver's internals: it takes
    only the weights and the declared limits and re-derives every check. A bug in
    the projection cannot hide here, because this code does not call it.

    Returns:
        ``(passed, violations)`` where each violation names the limit and the
        amount by which it was breached.
    """
    violations: list[str] = []
    constraints = problem.constraints
    if not np.isfinite(weights).all():
        return False, ("weights are not finite",)

    if problem.budget is not None:
        drift = abs(float(np.sum(weights)) - problem.budget)
        if drift > AUDIT_TOLERANCE:
            violations.append(f"budget breached by {drift:.3e}")

    excess = float(np.max(np.abs(weights)) - constraints.max_position)
    if excess > AUDIT_TOLERANCE:
        violations.append(f"max_position breached by {excess:.3e}")

    gross = float(np.sum(np.abs(weights)))
    gross_limit = min(constraints.max_gross, constraints.max_leverage) * constraints.deployable
    if gross - gross_limit > AUDIT_TOLERANCE:
        violations.append(f"max_gross breached by {gross - gross_limit:.3e}")

    net = abs(float(np.sum(weights)))
    if net - constraints.max_net > AUDIT_TOLERANCE:
        violations.append(f"max_net breached by {net - constraints.max_net:.3e}")

    if constraints.long_only and float(np.min(weights)) < -AUDIT_TOLERANCE:
        violations.append(f"long_only breached by {abs(float(np.min(weights))):.3e}")

    if constraints.max_turnover is not None:
        turnover = float(np.sum(np.abs(weights - problem.previous())))
        if turnover - constraints.max_turnover > AUDIT_TOLERANCE:
            violations.append(f"max_turnover breached by {turnover - constraints.max_turnover:.3e}")

    caps = problem.caps()
    if caps is not None:
        breach = float(np.max(np.abs(weights) - caps))
        if breach > AUDIT_TOLERANCE:
            violations.append(f"liquidity cap breached by {breach:.3e}")

    for exposure in problem.exposures:
        amount = exposure.violation(weights)
        if amount > AUDIT_TOLERANCE:
            violations.append(f"exposure {exposure.name!r} breached by {amount:.3e}")

    try:
        problem.risk_model.portfolio_variance(weights)
    except RiskModelError as exc:  # pragma: no cover - guarded at construction
        violations.append(f"risk model rejected the solution: {exc}")

    return (not violations), tuple(violations)


# ---------------------------------------------------------------------------
# Analytic references, used for validation
# ---------------------------------------------------------------------------


def analytic_minimum_variance(risk_model: RiskModel) -> FloatArray:
    """Return the closed-form unconstrained minimum-variance portfolio.

    ``w = Σ⁻¹ 1 / (1' Σ⁻¹ 1)``, the exact solution subject only to the budget
    constraint. Used as the independent reference the constrained solver is
    validated against on well-conditioned problems where no other limit binds.
    """
    size = len(risk_model.assets)
    ones = np.ones(size)
    try:
        solved = np.linalg.solve(risk_model.covariance, ones)
    except np.linalg.LinAlgError as exc:
        raise OptimizerError("covariance is singular; analytic reference unavailable") from exc
    total = float(ones @ solved)
    if abs(total) < 1e-300:
        raise OptimizerError("degenerate covariance; analytic reference unavailable")
    return np.asarray(solved / total, dtype=np.float64)


def analytic_maximum_utility(
    risk_model: RiskModel, alpha: FloatArray, risk_aversion: float
) -> FloatArray:
    """Return the closed-form unconstrained utility maximum.

    ``w = (1 / lambda) Σ⁻¹ mu``, the exact stationary point of
    ``mu'w - (lambda/2) w'Σw`` with no constraints at all. The constrained
    solver must reproduce it when every limit is slack.
    """
    if not np.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise OptimizerError("risk_aversion must be finite and positive")
    try:
        solved = np.linalg.solve(risk_model.covariance, alpha)
    except np.linalg.LinAlgError as exc:
        raise OptimizerError("covariance is singular; analytic reference unavailable") from exc
    return np.asarray(solved / risk_aversion, dtype=np.float64)
