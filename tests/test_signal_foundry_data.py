"""Contract and anti-corruption tests for Signal Foundry bundles."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data import (
    SignalFoundryDataError,
    load_prices,
    load_signal_foundry_dataset,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures/signal_foundry_v1"


def _fixture_bundle() -> Path:
    pointer = json.loads((FIXTURE_ROOT / "current.json").read_text(encoding="utf-8"))
    return FIXTURE_ROOT / pointer["bundle_id"]


def _copied_bundle(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / _fixture_bundle().name
    shutil.copytree(_fixture_bundle(), destination)
    return destination


def test_committed_producer_fixture_validates_and_maps_adjusted_bars() -> None:
    dataset = load_signal_foundry_dataset(_fixture_bundle())

    assert dataset.bundle_id == _fixture_bundle().name
    assert dataset.manifest["schema_version"] == "1.0.0"
    assert dataset.manifest["license"]["observations_redistributable"] is True
    assert set(dataset.panel["symbol"]) == {"AAA", "SPY"}
    source_first = dataset.source_panel.iloc[0]
    panel_first = dataset.panel.loc[
        (dataset.panel["date"] == source_first["date"])
        & (dataset.panel["symbol"] == source_first["ticker"])
    ].iloc[0]
    adjustment_factor = source_first["adj_close"] / source_first["close"]
    assert panel_first["close"] == pytest.approx(source_first["adj_close"])
    assert panel_first["open"] == pytest.approx(source_first["open"] * adjustment_factor)
    assert dataset.decision_panel is not None
    assert dataset.decision_panel["date"].min() > dataset.panel["date"].min()
    assert len(dataset.decision_panel) < len(dataset.panel)


def test_as_of_rule_excludes_future_available_observations() -> None:
    early = load_signal_foundry_dataset(
        _fixture_bundle(),
        as_of="2024-01-01T06:00:00Z",
    )
    later = load_signal_foundry_dataset(
        _fixture_bundle(),
        as_of="2024-01-04T06:00:00Z",
    )

    assert early.source_panel["available_at"].max() <= pd.Timestamp("2024-01-01T06:00:00Z")
    assert later.source_panel["available_at"].max() <= pd.Timestamp("2024-01-04T06:00:00Z")
    assert len(early.panel) < len(later.panel)


def test_future_rows_cannot_change_earlier_as_of_view(tmp_path: Path) -> None:
    original = load_signal_foundry_dataset(
        _fixture_bundle(),
        as_of="2024-01-03T04:59:59Z",
    ).panel
    copied = _copied_bundle(tmp_path)
    future_partition = copied / "prices/year=2024/part-00000.parquet"
    future_partition.write_bytes(b"adversarial future mutation")

    with pytest.raises(SignalFoundryDataError, match="hash mismatch"):
        load_signal_foundry_dataset(copied, as_of="2024-01-03T04:59:59Z")
    assert not original.empty


def test_corrupt_partition_and_path_traversal_fail_closed(tmp_path: Path) -> None:
    corrupt = _copied_bundle(tmp_path / "corrupt")
    next(corrupt.glob("prices/year=*/part-00000.parquet")).write_bytes(b"corrupt")
    with pytest.raises(SignalFoundryDataError, match="hash mismatch"):
        load_signal_foundry_dataset(corrupt)

    traversal = _copied_bundle(tmp_path / "traversal")
    manifest_path = traversal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../licensed-data.parquet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SignalFoundryDataError, match="semantic identity mismatch|unsafe"):
        load_signal_foundry_dataset(traversal)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "2.0.0", "unsupported schema"),
        ("columns", ["date"], "unsupported column"),
        ("point_in_time_limits", {}, "point-in-time"),
        (
            "license",
            {
                "observations_redistributable": True,
                "bundle_must_remain_local": True,
                "public_evidence_must_be_aggregate_or_synthetic": False,
            },
            "internally inconsistent",
        ),
    ],
)
def test_unsupported_or_ambiguous_manifest_policy_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    bundle = _copied_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SignalFoundryDataError, match=match):
        load_signal_foundry_dataset(bundle)


def test_generic_loader_cannot_collapse_market_and_decision_panels() -> None:
    with pytest.raises(ValueError, match="governed dual-panel"):
        load_prices(
            {
                "source": "signal_foundry",
                "bundle_dir": str(_fixture_bundle()),
                "benchmark": "SPY",
            }
        )


def test_signal_foundry_generic_loader_always_fails_closed() -> None:
    with pytest.raises(ValueError, match="governed dual-panel"):
        load_prices({"source": "signal_foundry", "benchmark": "SPY"})
    with pytest.raises(ValueError, match="governed dual-panel"):
        load_prices(
            {
                "source": "signal_foundry",
                "bundle_dir": str(_fixture_bundle()),
                "benchmark": "MISSING",
            }
        )


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("exchange_calendar", "ambiguous", "unsupported exchange"),
        ("currency", "XYZ", "unsupported currencies"),
        ("effective_at", pd.Timestamp("2024-01-01"), "timezone-aware"),
    ],
)
def test_ambiguous_market_semantics_fail_closed(
    tmp_path: Path,
    column: str,
    value: object,
    match: str,
) -> None:
    bundle = _copied_bundle(tmp_path)
    partition_path = next(bundle.glob("prices/year=*/part-00000.parquet"))
    partition = pd.read_parquet(partition_path)
    partition[column] = value
    partition.to_parquet(partition_path, index=False)

    with pytest.raises(SignalFoundryDataError, match="hash mismatch"):
        load_signal_foundry_dataset(bundle)

    # Even a producer that recomputed only the file hash cannot bypass the
    # independent semantic checks; full bundle-identity rebuilding is tested
    # by the producer repository.
    import hashlib

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["files"]
        if item["path"] == str(partition_path.relative_to(bundle))
    )
    entry["sha256"] = hashlib.sha256(partition_path.read_bytes()).hexdigest()
    semantic_fields = (
        "contract",
        "schema_version",
        "source_snapshot_hash",
        "source_manifest_sha256",
        "producer_git_sha",
        "files",
        "columns",
        "rows",
        "date_min",
        "date_max",
        "tickers",
        "temporal_contract",
        "point_in_time_limits",
        "license",
        "source_provenance",
    )
    semantic = {key: manifest[key] for key in semantic_fields}
    manifest["bundle_id"] = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    renamed = bundle.with_name(manifest["bundle_id"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle.rename(renamed)

    with pytest.raises(SignalFoundryDataError, match=match):
        load_signal_foundry_dataset(renamed)
