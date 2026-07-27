# Constrained Markowitz optimization (SF-S4-MR2)

A deterministic, point-in-time mean-variance optimizer with a validated risk-model
contract, four formulations, an independent feasibility audit, and walk-forward
evidence against the SF-S4-MR1 baselines and a no-trade arm.

> **No profit claim.** An attractive efficient frontier is an in-sample artefact.
> All evidence here is synthetic with a planted signal; the Sharpe levels are a
> property of that construction, not a market result. Markowitz optimization
> guarantees nothing out of sample and the frontier is not stable.

---

## 1. The problem

In periodic (per-bar) units throughout:

```
minimize   (λ/2) · w'Σw  −  μ'w  +  cost(w, w_prev)
subject to 1'w = budget                       (equality)
           |w_i| ≤ max_position               (box)
           ‖w‖₁ ≤ max_gross · deployable      (L1 ball)
           |1'w| ≤ max_net                    (net band)
           ‖w − w_prev‖₁ ≤ max_turnover       (turnover ball)
           |w_i| ≤ liquidity_cap_i            (participation)
           lower ≤ a'w ≤ upper                (sector / factor)
           w ≥ 0 if long_only
```

**Units are stated, not assumed.** `μ` and `Σ` must be in the *same* periodic
units; the `RiskModel` carries its own `periods_per_year` so annualization
happens once, at reporting time. Mixing an annualized alpha with a daily
covariance is the most common unit error in mean-variance code.

**Budget is an equality and it matters.** The variance objective is minimized at
`w = 0`, so without `1'w = budget` the "minimum-variance portfolio" is an empty
book. This was caught during development: the first implementation had only upper
bounds and returned all zeros, correctly optimal for the problem as posed and
useless as a portfolio.

### Formulations

| formulation | μ | λ | notes |
|---|---|---|---|
| `minimum_variance` | 0 | 1 | mean term vanishes; needs a budget |
| `target_return` | — | 1 | min variance s.t. `μ'w ≥ target`, as a half-space |
| `maximum_utility` | alpha | declared | classic utility maximization |
| `alpha_risk_cost` | alpha | declared | costs inside the objective |

---

## 2. The risk model is a contract

The optimizer inverts `Σ`, so the *smallest* eigenvalues — the ones estimated
worst — dominate the answer. `RiskModel` therefore refuses a matrix that is not
square, symmetric, finite, or PSD, and publishes its condition number and minimum
eigenvalue **before** the solve.

Two refusals worth naming:

- **Asymmetry is an error, not something to average away.** Silently applying
  `(A + A')/2` hides an estimator bug while changing the risk model.
- **Any ridge is reported.** When a matrix needs stabilization the amount is
  recorded on the model. A silent repair changes the strategy, and doing it
  without saying so is how an optimizer's output stops corresponding to the risk
  it claims to control.

`shrinkage_covariance` is a causal Ledoit-Wolf-style estimator toward a
constant-correlation target. It shrinks harder exactly when the sample covariance
is least trustworthy. **It is interim** — issue #7 owns the production estimator,
and because the optimizer depends on the `RiskModel` contract rather than on this
implementation, #7 lands as a drop-in.

---

## 3. Solver

Accelerated projected gradient (FISTA) with a proximal step for the turnover
charge. Rationale, alternatives, and residual numerical risk are in
[ADR 0009](adr/0009-from-scratch-mean-variance-solver.md). The short version:

- **Analytic step size**, not a line search, so the solve is deterministic and
  two runs produce bit-identical weights.
- **Every projection is a true projection.** Clip-then-rescale and radial L1
  rescaling are both *not* projections and both broke convergence during
  development; the L1 step is now the exact sort-and-scan soft-threshold.
- **The turnover charge is proximal.** As a subgradient it oscillates across the
  kink at zero trade and *increases* turnover — measured at 0.90 with a penalty
  against 0.56 without, before the prox replaced it.
- **Termination is on feasibility**, not on a small step.

### Validated against closed form

Declared tolerance 1e-8 on the weights, on well-conditioned problems where no
inequality binds:

| formulation | reference | achieved |
|---|---|---|
| minimum variance | `Σ⁻¹1 / (1'Σ⁻¹1)` | **3.8e-11** |
| maximum utility | `(1/λ)Σ⁻¹μ` | **7.3e-13** |

---

## 4. Nothing is trusted

`audit_solution` re-derives every feasibility check from the returned weights
alone, using code that does **not** call the projection. A projection bug
therefore cannot hide behind itself. A solve that fails its own audit is reported
as `failed`, never as a portfolio.

The result also carries active constraints — a book pinned against many limits is
being determined by the constraint set rather than by the forecast, which is
worth knowing before reading its expected return as skill.

**No silent relaxation, anywhere.** An infeasible constraint set raises. An
unreachable target return is reported `infeasible`, not approximated.

---

## 5. Measured evidence

Synthetic panel, 12 names, 240 evaluation days, 10bps turnover cost, long-only,
`max_position` 0.25, budget 1.0, capital $100M:

| arm | turnover budget | net return | gross | cost drag | turnover | max drawdown |
|---|---|---|---|---|---|---|
| alpha_risk_cost | none | 2.448 | 2.623 | 0.175 | 0.694 | −0.015 |
| alpha_risk_cost | 0.2 | 0.844 | 0.894 | 0.049 | 0.196 | −0.022 |
| maximum_utility | none | 2.430 | 2.624 | 0.195 | 0.773 | −0.014 |
| maximum_utility | 0.2 | 0.768 | 0.816 | 0.048 | 0.192 | −0.022 |
| minimum_variance | none | 0.042 | 0.046 | 0.004 | 0.017 | −0.045 |
| **no_trade** | — | 0.026 | 0.027 | 0.001 | 0.004 | −0.047 |

Read honestly: `minimum_variance` uses no alpha, so it lands just above the
do-nothing arm — which is the correct outcome, not a disappointment. The turnover
budget cuts return by roughly two-thirds and cost drag by three-quarters.

**The no-trade arm is the one most often omitted and hardest to beat net of
costs.** Without it, "our optimizer beat equal weight" can be true while "our
optimizer beat leaving it alone" is false.

### Sensitivity to input error

The most important diagnostic for mean-variance, which is famously more sensitive
to expected-return error than to covariance error:

| alpha error | net return | net Sharpe |
|---|---|---|
| 0% | 2.430 | 21.97 |
| 10% | 2.429 | 21.98 |
| 25% | 2.426 | 21.91 |
| 50% | 2.260 | 20.33 |
| 100% | 1.660 | 15.20 |

Degradation is graceful here only because the planted signal is strong. On a
realistic signal-to-noise ratio the same sweep is the check that decides whether
a result reflects skill or input precision.

### Capacity

Liquidity caps bind as capital grows. At $200M on 12 names at 5% participation
the constraint set becomes infeasible and every date is recorded
`feasible=False` with the reason — never dropped, because dropping the hard dates
is how an optimizer acquires a survivorship-flattered record.

---

## 6. Evidence and gates

`tests/test_mean_variance.py` — 54 tests: analytic agreement, independent audit
(including that the audit *fails* a book breaching each limit), non-symmetric /
non-finite / non-PSD / singular / dimension-mismatched covariance, infeasible
constraints, non-convergence reporting, asset permutation, alpha/λ scale
invariance, duplicate assets, deterministic ties, near-zero variance, extreme
correlation, single-asset and empty universes, turnover-penalty monotonicity,
problem identity, solver determinism, three mutation tests (future returns,
future covariance, future universe membership), and the walk-forward harness.

Coverage: `mean_variance.py` 92%, `risk_model.py` 90%, `evidence.py` 88%
(branch). Repository total 84.79% against an unchanged 78% floor.

---

## 7. Residual limitations

- **Synthetic evidence only**, with a planted signal.
- **Interim covariance.** #7 supplies the production causal shrinkage and factor
  risk model; this is a contract-compatible placeholder.
- **First-order convergence** reaches a declared tolerance, not an exact vertex.
- **Cost model is linear + quadratic in turnover.** Real slippage, impact, and
  latency are SF-S4-MR4; the quadratic term is a crude stand-in, documented as
  such rather than presented as calibrated.
- **No warm start** across rebalances.
- **Capacity is participation caps only** — no borrow, short fees, or crowding.
- **Simulation only**: no broker, no live endpoint, no capital at risk, and the
  optimizer stays out of any paper/live path until later qualification gates.
