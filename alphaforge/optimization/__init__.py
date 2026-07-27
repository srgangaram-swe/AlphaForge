"""Portfolio optimization contracts and solvers (SF-S4-MR2)."""

from alphaforge.optimization.evidence import (
    compare_markowitz_variants,
    run_markowitz_backtest,
    run_no_trade_backtest,
    sensitivity_to_input_error,
)
from alphaforge.optimization.mean_variance import (
    AUDIT_TOLERANCE,
    CostModel,
    ExposureConstraint,
    Formulation,
    MeanVarianceProblem,
    OptimizerError,
    SolverResult,
    analytic_maximum_utility,
    analytic_minimum_variance,
    audit_solution,
    solve_mean_variance,
)
from alphaforge.optimization.risk_model import (
    RiskModel,
    RiskModelError,
    shrinkage_covariance,
    validate_covariance,
)

__all__ = [
    "AUDIT_TOLERANCE",
    "CostModel",
    "ExposureConstraint",
    "Formulation",
    "MeanVarianceProblem",
    "OptimizerError",
    "RiskModel",
    "RiskModelError",
    "SolverResult",
    "analytic_maximum_utility",
    "analytic_minimum_variance",
    "audit_solution",
    "compare_markowitz_variants",
    "run_markowitz_backtest",
    "run_no_trade_backtest",
    "sensitivity_to_input_error",
    "shrinkage_covariance",
    "solve_mean_variance",
    "validate_covariance",
]
