"""Deterministic aggregate study for the SF-S3-MR10 abstention policy.

The committed reference is synthetic and deliberately separates policy inputs
from realized evaluation labels. It exercises mechanics and tradeoffs only; it
does not access a final holdout or establish an investable signal.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import seaborn as sns

from alphaforge.config import load_decision_policy_config
from alphaforge.decision import (
    DecisionAction,
    DecisionPolicy,
    DecisionReason,
    DecisionSignal,
    DecisionThresholds,
    RegimeSupport,
)
from alphaforge.visualization.decision_policy_plots import plot_decision_policy_study

STUDY_SCHEMA_VERSION = "1.0.0"
POLICY_NAMES = ("abstention_policy", "always_trade", "never_trade")


@dataclass(frozen=True, slots=True)
class DecisionPolicyStudyConfig:
    """Resolved immutable study configuration."""

    thresholds: DecisionThresholds
    schema_version: str
    scope: str
    baselines: tuple[str, ...]
    seed: int
    observation_count: int
    period_count: int
    anchor_time: datetime
    expected_return_scale: float
    realized_noise_scale: float
    minimum_expected_cost: float
    maximum_expected_cost: float
    cost_uncertainty_scale: float
    prediction_uncertainty_scale: float
    disagreement_scale: float
    regime_uncertainty_scale: float
    unsupported_regime_probability: float
    stale_probability: float
    future_probability: float
    drift_probability: float
    unit_turnover: float
    unit_notional: float
    period_capacity: float
    interpretation: str
    protected_holdout_access: bool
    broker_access: bool

    @property
    def study_id(self) -> str:
        """Return a stable identity over policy, generator, and baselines."""
        return f"decision-study-{_sha256_json(_study_config_payload(self))}"


@dataclass(frozen=True, slots=True)
class DecisionStudyObservation:
    """Synthetic policy input plus sequestered realized evaluation labels."""

    signal: DecisionSignal
    period: int
    realized_return: float
    realized_cost: float


@dataclass(frozen=True, slots=True)
class DecisionPolicyMetrics:
    """Aggregate policy tradeoffs in explicit units."""

    policy: str
    observation_count: int
    trade_count: int
    abstention_count: int
    coverage: float
    selective_risk: float | None
    mean_net_value: float
    missed_opportunity: float
    mean_turnover_per_period: float
    mean_capacity_demand: float
    peak_capacity_demand: float

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecisionPolicyStudyResult:
    """Paths and aggregates from one atomically published offline study."""

    output_dir: Path
    study_id: str
    policy_id: str
    metrics: tuple[DecisionPolicyMetrics, ...]
    reason_counts: tuple[tuple[DecisionReason, int], ...]


def load_decision_policy_study_config(path: str | Path) -> DecisionPolicyStudyConfig:
    """Load the central strict schema and resolve immutable domain types."""
    payload = load_decision_policy_config(path)
    policy = cast(dict[str, Any], payload["policy"])
    study = cast(dict[str, Any], payload["study"])
    anchor = datetime.fromisoformat(str(study["anchor_time"]).replace("Z", "+00:00"))
    return DecisionPolicyStudyConfig(
        thresholds=DecisionThresholds(**policy),
        schema_version=str(study["schema_version"]),
        scope=str(study["scope"]),
        baselines=tuple(cast(list[str], study["baselines"])),
        seed=int(study["seed"]),
        observation_count=int(study["observation_count"]),
        period_count=int(study["period_count"]),
        anchor_time=anchor.astimezone(UTC),
        expected_return_scale=float(study["expected_return_scale"]),
        realized_noise_scale=float(study["realized_noise_scale"]),
        minimum_expected_cost=float(study["minimum_expected_cost"]),
        maximum_expected_cost=float(study["maximum_expected_cost"]),
        cost_uncertainty_scale=float(study["cost_uncertainty_scale"]),
        prediction_uncertainty_scale=float(study["prediction_uncertainty_scale"]),
        disagreement_scale=float(study["disagreement_scale"]),
        regime_uncertainty_scale=float(study["regime_uncertainty_scale"]),
        unsupported_regime_probability=float(study["unsupported_regime_probability"]),
        stale_probability=float(study["stale_probability"]),
        future_probability=float(study["future_probability"]),
        drift_probability=float(study["drift_probability"]),
        unit_turnover=float(study["unit_turnover"]),
        unit_notional=float(study["unit_notional"]),
        period_capacity=float(study["period_capacity"]),
        interpretation=str(study["interpretation"]),
        protected_holdout_access=bool(study["protected_holdout_access"]),
        broker_access=bool(study["broker_access"]),
    )


def build_synthetic_decision_reference(
    config: DecisionPolicyStudyConfig,
) -> tuple[DecisionStudyObservation, ...]:
    """Build bounded synthetic opportunities with labels outside policy inputs."""
    if not isinstance(config, DecisionPolicyStudyConfig):
        raise TypeError("config must be DecisionPolicyStudyConfig")
    rng = np.random.default_rng(config.seed)
    threshold = config.thresholds
    observations: list[DecisionStudyObservation] = []
    for index in range(config.observation_count):
        period = index % config.period_count
        decision_time = config.anchor_time + timedelta(days=period)
        timing_draw = float(rng.random())
        if timing_draw < config.future_probability:
            age_seconds = -float(rng.uniform(1.0, threshold.maximum_data_age_seconds / 4))
        elif timing_draw < config.future_probability + config.stale_probability:
            age_seconds = float(
                rng.uniform(
                    threshold.maximum_data_age_seconds + 1.0,
                    threshold.maximum_data_age_seconds * 2.0,
                )
            )
        else:
            age_seconds = float(rng.uniform(0.0, threshold.maximum_data_age_seconds))

        expected_return = float(
            np.clip(
                rng.normal(0.0, config.expected_return_scale),
                -0.95 * threshold.maximum_absolute_expected_return,
                0.95 * threshold.maximum_absolute_expected_return,
            )
        )
        expected_cost = float(
            rng.uniform(config.minimum_expected_cost, config.maximum_expected_cost)
        )
        cost_uncertainty = float(abs(rng.normal(0.0, config.cost_uncertainty_scale)))
        prediction_uncertainty = float(abs(rng.normal(0.0, config.prediction_uncertainty_scale)))
        disagreement = float(np.clip(abs(rng.normal(0.0, config.disagreement_scale)), 0.0, 1.0))
        regime_uncertainty = float(
            np.clip(abs(rng.normal(0.0, config.regime_uncertainty_scale)), 0.0, 1.0)
        )
        regime = (
            RegimeSupport.UNSUPPORTED
            if rng.random() < config.unsupported_regime_probability
            else RegimeSupport.SUPPORTED
        )
        if rng.random() < config.drift_probability:
            drift_score = float(rng.uniform(threshold.maximum_drift_score, 1.0))
        else:
            drift_score = float(rng.uniform(0.0, threshold.maximum_drift_score))

        signal = DecisionSignal(
            signal_id=f"synthetic-{index:06d}",
            model_id="frozen-ensemble-v1",
            decision_time=decision_time,
            data_available_at=decision_time - timedelta(seconds=age_seconds),
            expected_return=expected_return,
            expected_cost=expected_cost,
            cost_uncertainty=cost_uncertainty,
            prediction_uncertainty=prediction_uncertainty,
            model_disagreement=disagreement,
            regime=regime,
            regime_uncertainty=regime_uncertainty,
            drift_score=drift_score,
        )
        realized_return = float(expected_return + rng.normal(0.0, config.realized_noise_scale))
        realized_cost = float(expected_cost + abs(rng.normal(0.0, config.cost_uncertainty_scale)))
        observations.append(
            DecisionStudyObservation(
                signal=signal,
                period=period,
                realized_return=realized_return,
                realized_cost=realized_cost,
            )
        )
    return tuple(observations)


def evaluate_decision_policy_study(
    observations: tuple[DecisionStudyObservation, ...],
    config: DecisionPolicyStudyConfig,
) -> tuple[
    tuple[DecisionPolicyMetrics, ...],
    tuple[tuple[DecisionReason, int], ...],
]:
    """Evaluate policy and frozen baselines without exposing realized labels."""
    if len(observations) != config.observation_count:
        raise ValueError("observation count does not match frozen study configuration")
    if any(not isinstance(item, DecisionStudyObservation) for item in observations):
        raise TypeError("observations must contain only DecisionStudyObservation")
    policy = DecisionPolicy(config.thresholds)
    decisions = policy.evaluate_many(item.signal for item in observations)
    selected_ids = {
        decision.signal_id for decision in decisions if decision.action is DecisionAction.TRADE
    }
    policy_mask = np.fromiter(
        (item.signal.signal_id in selected_ids for item in observations),
        dtype=bool,
        count=len(observations),
    )
    always_mask = np.ones(len(observations), dtype=bool)
    never_mask = np.zeros(len(observations), dtype=bool)
    masks = {
        "abstention_policy": policy_mask,
        "always_trade": always_mask,
        "never_trade": never_mask,
    }
    metrics = tuple(
        _aggregate_policy(name, masks[name], observations, config) for name in POLICY_NAMES
    )
    reason_counts = tuple(
        (reason, sum(reason in decision.reasons for decision in decisions))
        for reason in DecisionReason
        if any(reason in decision.reasons for decision in decisions)
    )
    return metrics, reason_counts


def run_decision_policy_study(
    config: DecisionPolicyStudyConfig,
    output_dir: str | Path,
) -> DecisionPolicyStudyResult:
    """Generate and atomically publish aggregate-only synthetic evidence."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"study destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".publishing-{destination.name}"
    if staging.exists():
        raise FileExistsError(f"stale study staging exists: {staging}")
    staging.mkdir()
    try:
        observations = build_synthetic_decision_reference(config)
        metrics, reason_counts = evaluate_decision_policy_study(observations, config)
        metrics_frame = pd.DataFrame([item.to_record() for item in metrics])
        reason_frame = pd.DataFrame(
            ({"reason": reason.value, "count": count} for reason, count in reason_counts),
            columns=["reason", "count"],
        )
        metrics_path = staging / "aggregate_metrics.csv"
        reasons_path = staging / "abstention_reasons.csv"
        plot_path = staging / "decision_policy_study.png"
        metrics_frame.to_csv(
            metrics_path,
            index=False,
            float_format="%.12g",
            lineterminator="\n",
        )
        reason_frame.to_csv(reasons_path, index=False, lineterminator="\n")
        plot_decision_policy_study(
            metrics_frame,
            reason_frame,
            plot_path,
            observation_count=config.observation_count,
            seed=config.seed,
        )
        summary = _summary_payload(
            config,
            metrics,
            reason_counts,
            artifact_hashes={
                metrics_path.name: _sha256_file(metrics_path),
                reasons_path.name: _sha256_file(reasons_path),
                plot_path.name: _sha256_file(plot_path),
            },
        )
        (staging / "summary.json").write_text(
            json.dumps(
                summary,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DecisionPolicyStudyResult(
        output_dir=destination,
        study_id=config.study_id,
        policy_id=config.thresholds.policy_id,
        metrics=metrics,
        reason_counts=reason_counts,
    )


def _aggregate_policy(
    name: str,
    selected: np.ndarray,
    observations: tuple[DecisionStudyObservation, ...],
    config: DecisionPolicyStudyConfig,
) -> DecisionPolicyMetrics:
    expected_sign = np.sign(
        np.fromiter(
            (item.signal.expected_return for item in observations),
            dtype=float,
            count=len(observations),
        )
    )
    realized_returns = np.fromiter(
        (item.realized_return for item in observations),
        dtype=float,
        count=len(observations),
    )
    realized_costs = np.fromiter(
        (item.realized_cost for item in observations),
        dtype=float,
        count=len(observations),
    )
    realized_net = expected_sign * realized_returns - realized_costs
    trade_count = int(selected.sum())
    abstention_count = len(observations) - trade_count
    coverage = trade_count / len(observations)
    selective_risk = None if trade_count == 0 else float(np.mean(realized_net[selected] < 0.0))
    selected_net = np.where(selected, realized_net, 0.0)
    missed = np.where(~selected, np.maximum(realized_net, 0.0), 0.0)
    period_trades = np.bincount(
        np.fromiter(
            (item.period for item in observations),
            dtype=np.int64,
            count=len(observations),
        )[selected],
        minlength=config.period_count,
    )
    capacity = period_trades * config.unit_notional / config.period_capacity
    return DecisionPolicyMetrics(
        policy=name,
        observation_count=len(observations),
        trade_count=trade_count,
        abstention_count=abstention_count,
        coverage=float(coverage),
        selective_risk=selective_risk,
        mean_net_value=float(np.mean(selected_net)),
        missed_opportunity=float(np.mean(missed)),
        mean_turnover_per_period=float(trade_count * config.unit_turnover / config.period_count),
        mean_capacity_demand=float(np.mean(capacity)),
        peak_capacity_demand=float(np.max(capacity)),
    )


def _study_config_payload(config: DecisionPolicyStudyConfig) -> dict[str, Any]:
    threshold_payload = asdict(config.thresholds)
    return {
        "policy": threshold_payload,
        "study": {
            "anchor_time": config.anchor_time.isoformat().replace("+00:00", "Z"),
            "baselines": list(config.baselines),
            "broker_access": config.broker_access,
            "cost_uncertainty_scale": config.cost_uncertainty_scale,
            "disagreement_scale": config.disagreement_scale,
            "drift_probability": config.drift_probability,
            "expected_return_scale": config.expected_return_scale,
            "future_probability": config.future_probability,
            "interpretation": config.interpretation,
            "maximum_expected_cost": config.maximum_expected_cost,
            "minimum_expected_cost": config.minimum_expected_cost,
            "observation_count": config.observation_count,
            "period_capacity": config.period_capacity,
            "period_count": config.period_count,
            "prediction_uncertainty_scale": config.prediction_uncertainty_scale,
            "protected_holdout_access": config.protected_holdout_access,
            "realized_noise_scale": config.realized_noise_scale,
            "regime_uncertainty_scale": config.regime_uncertainty_scale,
            "schema_version": config.schema_version,
            "scope": config.scope,
            "seed": config.seed,
            "stale_probability": config.stale_probability,
            "unit_notional": config.unit_notional,
            "unit_turnover": config.unit_turnover,
            "unsupported_regime_probability": config.unsupported_regime_probability,
        },
    }


def _summary_payload(
    config: DecisionPolicyStudyConfig,
    metrics: tuple[DecisionPolicyMetrics, ...],
    reason_counts: tuple[tuple[DecisionReason, int], ...],
    *,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": config.study_id,
        "policy_id": config.thresholds.policy_id,
        "scope": config.scope,
        "interpretation": config.interpretation,
        "configuration": _study_config_payload(config),
        "metrics": [item.to_record() for item in metrics],
        "abstention_reason_counts": {reason.value: count for reason, count in reason_counts},
        "evidence": {
            "artifacts": artifact_hashes,
            "aggregate_rows_published": len(metrics) + len(reason_counts),
            "row_level_signals_published": 0,
            "row_level_outcomes_published": 0,
            "protected_holdout_access": config.protected_holdout_access,
            "broker_access": config.broker_access,
            "orders_emitted": 0,
        },
        "compute": {
            "device": "cpu",
            "algorithmic_complexity": "O(n) time and O(n) bounded study memory",
            "observations_evaluated": config.observation_count,
            "maximum_batch_size": config.thresholds.maximum_batch_size,
            "seed": config.seed,
            "numpy_bit_generator": "PCG64",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "seaborn": sns.__version__,
        },
        "mathematics": {
            "eligibility": (
                "abs(expected_return) - cost_multiplier * "
                "(expected_cost + cost_uncertainty_multiplier * cost_uncertainty) - "
                "uncertainty_penalty * prediction_uncertainty > required_margin"
            ),
            "selective_risk": "loss count among eligible decisions / eligible decision count",
            "never_trade_selective_risk": None,
        },
        "limitations": [
            "Synthetic engineering evidence is not market, backtest, paper, or live evidence.",
            "The generator is not calibrated to historical execution or forecast distributions.",
            "Capacity and turnover are demand proxies before portfolio construction, not fills.",
            "Abstention reduces declared exposure; it cannot guarantee against loss.",
            "Thresholds were not selected on a protected final holdout.",
        ],
    }


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
