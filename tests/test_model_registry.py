"""Model registry seed-policy tests."""

from __future__ import annotations

import pytest

from alphaforge.models.registry import seed_model_specs


def test_model_seed_injection_is_order_independent_and_non_mutating() -> None:
    original = [
        {"name": "gradient_boosting", "params": {"max_iter": 10}},
        {
            "name": "ensemble",
            "params": {
                "members": [
                    {"name": "random_forest", "params": {}},
                    {"name": "ridge", "params": {"alpha": 1.0}},
                ]
            },
        },
    ]

    first = seed_model_specs(original, 17)
    second = seed_model_specs(list(reversed(original)), 17)

    assert original[0]["params"] == {"max_iter": 10}
    assert first[0]["params"]["random_state"] == 17
    assert first[1]["params"]["members"][0]["params"]["random_state"] == 17
    assert {spec["name"]: spec["params"] for spec in first} == {
        spec["name"]: spec["params"] for spec in second
    }


def test_explicit_model_seed_is_preserved() -> None:
    seeded = seed_model_specs(
        [{"name": "random_forest", "params": {"random_state": 99}}],
        17,
    )
    assert seeded[0]["params"]["random_state"] == 99


def test_model_seed_policy_rejects_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_model_specs([], -1)
    with pytest.raises(TypeError, match="params"):
        seed_model_specs([{"name": "ridge", "params": []}], 1)
    with pytest.raises(TypeError, match="ensemble members"):
        seed_model_specs([{"name": "ensemble", "params": {}}], 1)
