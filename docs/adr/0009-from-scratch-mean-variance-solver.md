# ADR 0009 — A from-scratch projected-gradient solver for constrained mean-variance

- **Status:** Accepted
- **Date:** 2026-07-26
- **Work item:** SF-S4-MR2 (#8), Signal Foundry Sprint 4
- **Supersedes / superseded by:** none

## Context

SF-S4-MR2 requires a deterministic, point-in-time Markowitz optimizer that
"exposes the mathematics, units, assumptions, conditioning, constraints, and
failure modes rather than treating an optimization package as a black box."

The problem is a convex QP: minimize `(λ/2) w'Σw − μ'w + cost(w, w_prev)` over
the intersection of budget, gross, net, per-position, long/short, turnover,
liquidity, and sector/factor limits.

Three options were considered.

**A general QP package (cvxpy, OSQP, quadprog).** Fastest to write and robust on
badly conditioned problems. Rejected for three reasons: it adds a dependency with
its own licence and supported-runtime surface for a single call site; the issue
explicitly requires the mathematics to be inspectable rather than delegated; and
solver internals (presolve, adaptive rho, scaling heuristics) introduce
data-dependent branching that makes bit-level reproducibility across versions
hard to guarantee.

**`scipy.optimize.minimize` with SLSQP.** No new dependency, but it is a general
nonlinear method applied to a problem that is exactly quadratic, its convergence
on degenerate constraint sets is opaque, and it offers no natural way to expose
per-constraint activity.

**Accelerated projected gradient (FISTA) with an exact projection.** Chosen.

## Decision

Solve by FISTA over the exact Euclidean projection onto the constraint set, with
a proximal step for the non-smooth turnover charge.

**Step size is analytic, not line-searched.** `t = 1/L` with
`L = λ·λ_max(Σ) + 2·quadratic_cost_rate`. A line search introduces
data-dependent branching; deriving `L` from the spectrum keeps the solve
deterministic, and two runs on identical inputs produce bit-identical weights
(asserted by test).

**Every projection is a true projection.** This is the decision that mattered
most in practice. Two natural-looking shortcuts are *not* projections and both
broke convergence during development:

- Clipping to a position cap and then rescaling to a gross target re-inflates
  the names the cap just pulled down.
- Radially rescaling an over-sized vector (`w · r/‖w‖₁`) lands *on* the L1 ball
  but is not the nearest point in it. Alternating a non-projection with genuine
  projections voids the convergence guarantee; the symptom was the iteration
  stalling on the simplex and returning an infeasible book. Replaced with the
  exact sort-and-scan soft-threshold projection.

**The turnover charge is proximal, not subgradient.** The linear cost is an L1
penalty centred on the previous book and is non-differentiable exactly where the
solution wants to sit — at zero trade. Subgradient descent oscillates across that
kink and *increases* turnover, which was observed (0.90 with a penalty against
0.56 without) before the proximal operator replaced it. Soft-thresholding toward
`w_prev` is the exact prox and makes the penalty suppress small trades.

**Target-return is a half-space, not a λ sweep.** The textbook approach traces
the frontier by sweeping risk aversion, but reaching high-return corners requires
`λ → 0`, at which point the objective's curvature vanishes, the analytic step
`1/(λ·λ_max)` explodes, and the iteration stops being numerically meaningful.
Adding `μ'w ≥ target` as a linear constraint keeps the objective well conditioned
and makes the requirement exact.

**Termination is on feasibility, not on a small step.** A slowly cycling
alternation can stall while still violating a limit; exiting on a small step
would return that point. The projection exits only when an independent
feasibility predicate holds, and when the intersection is empty it runs its full
budget and the caller reports `infeasible`.

**Nothing is trusted.** `audit_solution` re-derives every feasibility check from
the returned weights alone, without calling the projection, so a projection bug
cannot hide behind itself. A solve that fails its own audit is reported as
`failed`, never as a portfolio.

## Consequences

**Positive.** No new dependency; the licence and supported-runtime surface is
unchanged. Deterministic and reproducible. Validated against closed-form
references to 1e-8 on the weights (achieved 3.8e-11 for minimum variance and
7.3e-13 for maximum utility). Every limit, active constraint, conditioning
figure, and objective component is reported. `O(n²)` per iteration, dominated by
the covariance product.

**Negative.** First-order convergence reaches a declared tolerance rather than an
exact vertex, so a solution sitting on many active constraints is accurate to the
tolerance and no further. On badly conditioned covariances the alternating
projection needs more iterations than an interior-point method; the iteration cap
is declared and exhaustion is reported as `max_iterations`, never as success.
There is no warm start across rebalances yet.

**Rollback.** Removing the optimizer from allocation selection leaves SF-S4-MR1's
rule-based portfolios untouched — they share the constraint *record* but not the
projection, deliberately, because their gross semantics differ (MR1 waterfills to
deploy capital; MR2 treats gross as an upper bound the objective may leave slack).

## Residual risk

Mean-variance is far more sensitive to expected-return error than to covariance
error, and the sensitivity sweep in `alphaforge/optimization/evidence.py`
measures it rather than assuming it. The interim shrinkage covariance is an
approximation to what issue #7 will supply; the optimizer depends on the
`RiskModel` contract, not on the estimator, so #7 lands as a drop-in. Simulation
only: no broker, no live endpoint, no capital at risk.
