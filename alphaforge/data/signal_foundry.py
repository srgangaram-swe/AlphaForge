"""Fail-closed consumer for the Signal Foundry market-data contract.

This module intentionally does not import Signalattice. The repositories
communicate through a versioned manifest and content-addressed Parquet files,
and AlphaForge independently verifies every trust-boundary invariant before
mapping observations into its canonical adjusted-OHLCV representation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.data.schemas import validate_panel

CONTRACT_NAME = "signal-foundry-market-data"
SCHEMA_VERSION = "1.0.0"
MANIFEST_NAME = "manifest.json"
CONTRACT_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "effective_at",
    "available_at",
    "observed_at",
    "provider_updated_at",
    "instrument_id",
    "currency",
    "exchange_calendar",
    "adjustment_state",
    "source",
    "source_table",
)
SEMANTIC_MANIFEST_FIELDS: tuple[str, ...] = (
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
ADJUSTED_CLOSE_STATES = frozenset(
    {
        "provider_adjusted_close_unadjusted_ohlc",
        "synthetic_fixture",
        "synthetic_benchmark",
    }
)
UNADJUSTED_STATES = frozenset({"provider_unadjusted"})
SUPPORTED_CALENDARS = frozenset({"XNYS"})
SUPPORTED_CURRENCIES = frozenset({"USD"})


class SignalFoundryDataError(ValueError):
    """Raised when a producer bundle cannot be trusted or mapped safely."""


@dataclass(frozen=True)
class SignalFoundryDataset:
    """Verified source observations and their canonical AlphaForge view."""

    bundle_dir: Path
    manifest: dict[str, Any]
    source_panel: pd.DataFrame
    panel: pd.DataFrame
    decision_panel: pd.DataFrame | None = None

    @property
    def bundle_id(self) -> str:
        """Return the immutable producer identity."""
        return str(self.manifest["bundle_id"])


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_file(bundle_dir: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SignalFoundryDataError(f"unsafe bundle path: {relative!r}")
    root = bundle_dir.resolve()
    resolved = (bundle_dir / relative_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise SignalFoundryDataError(f"bundle path escapes root: {relative!r}")
    return resolved


def _read_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignalFoundryDataError(f"cannot read canonical manifest at {path}") from exc
    if not isinstance(value, dict):
        raise SignalFoundryDataError("bundle manifest must be a JSON object")
    if value.get("contract") != CONTRACT_NAME:
        raise SignalFoundryDataError("unsupported Signal Foundry contract")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise SignalFoundryDataError(
            f"unsupported schema version {value.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    bundle_id = value.get("bundle_id")
    if not isinstance(bundle_id, str) or len(bundle_id) != 64 or bundle_dir.name != bundle_id:
        raise SignalFoundryDataError("bundle identity does not match its directory")
    if value.get("columns") != list(CONTRACT_COLUMNS):
        raise SignalFoundryDataError("bundle manifest declares an unsupported column schema")
    return value


def _validate_manifest_policy(manifest: dict[str, Any]) -> None:
    license_policy = manifest.get("license")
    if not isinstance(license_policy, dict):
        raise SignalFoundryDataError("bundle manifest lacks license policy")
    redistributable = license_policy.get("observations_redistributable")
    must_remain_local = license_policy.get("bundle_must_remain_local")
    aggregate_only = license_policy.get("public_evidence_must_be_aggregate_or_synthetic")
    policy_values = (redistributable, must_remain_local, aggregate_only)
    if not all(isinstance(value, bool) for value in policy_values):
        raise SignalFoundryDataError("bundle license policy must contain explicit booleans")
    if must_remain_local is redistributable or aggregate_only is redistributable:
        raise SignalFoundryDataError("bundle license policy is internally inconsistent")

    limits = manifest.get("point_in_time_limits")
    if not isinstance(limits, dict):
        raise SignalFoundryDataError("bundle manifest lacks point-in-time limitations")
    required_limits = {
        "historical_revisions_complete",
        "universe_membership_point_in_time",
        "corporate_actions_complete",
    }
    if set(limits) != required_limits or not all(isinstance(limits[key], bool) for key in limits):
        raise SignalFoundryDataError(
            "bundle point-in-time limitations must explicitly cover revisions, universe, "
            "and corporate actions"
        )

    temporal = manifest.get("temporal_contract")
    if not isinstance(temporal, dict) or temporal.get("as_of_rule") != (
        "available_at <= decision timestamp"
    ):
        raise SignalFoundryDataError("bundle temporal contract is missing the supported as-of rule")


def _coerce_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(CONTRACT_COLUMNS):
        raise SignalFoundryDataError("bundle partition column order does not match schema 1.0.0")
    out = frame.copy()
    dates = pd.to_datetime(out["date"], errors="raise")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        raise SignalFoundryDataError("bundle market dates must be timezone-naive session labels")
    if not dates.eq(dates.dt.normalize()).all():
        raise SignalFoundryDataError("bundle market dates must not contain intraday timestamps")
    out["date"] = dates
    for column in (
        "effective_at",
        "available_at",
        "observed_at",
        "provider_updated_at",
    ):
        raw = out[column]
        if raw.notna().any():
            try:
                aware = raw.dropna().map(lambda value: pd.Timestamp(value).tzinfo is not None)
            except (TypeError, ValueError) as exc:
                raise SignalFoundryDataError(
                    f"bundle contains invalid temporal values in {column!r}"
                ) from exc
            if not aware.all():
                raise SignalFoundryDataError(
                    f"bundle temporal column {column!r} must be timezone-aware"
                )
        out[column] = pd.to_datetime(raw, errors="coerce", utc=True)
    if out[["effective_at", "available_at", "observed_at"]].isna().any().any():
        raise SignalFoundryDataError("bundle contains invalid required temporal values")
    if (out["available_at"] < out["effective_at"]).any():
        raise SignalFoundryDataError("bundle contains availability before effective time")
    if (out["observed_at"] < out["effective_at"]).any():
        raise SignalFoundryDataError("bundle contains observation before effective time")
    if out.duplicated(["date", "ticker"]).any():
        raise SignalFoundryDataError("bundle contains duplicate (date, ticker) rows")

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if not np.isfinite(out[numeric_columns].to_numpy(dtype=float)).all():
        raise SignalFoundryDataError("bundle contains missing or non-finite market values")
    if (out[["open", "high", "low", "close", "adj_close"]] <= 0).any().any():
        raise SignalFoundryDataError("bundle contains non-positive prices")
    if (out["volume"] < 0).any():
        raise SignalFoundryDataError("bundle contains negative volume")
    if (
        (out["high"] < out["low"])
        | (out["open"] > out["high"])
        | (out["open"] < out["low"])
        | (out["close"] > out["high"])
        | (out["close"] < out["low"])
    ).any():
        raise SignalFoundryDataError("bundle violates OHLC bounds")

    text_columns = [
        "ticker",
        "instrument_id",
        "currency",
        "exchange_calendar",
        "adjustment_state",
        "source",
        "source_table",
    ]
    for column in text_columns:
        if out[column].isna().any() or out[column].astype(str).str.strip().eq("").any():
            raise SignalFoundryDataError(f"bundle column {column!r} contains empty values")
        out[column] = out[column].astype(str)
    unknown_calendars = sorted(set(out["exchange_calendar"]) - SUPPORTED_CALENDARS)
    if unknown_calendars:
        raise SignalFoundryDataError(
            f"bundle contains unsupported exchange calendars: {unknown_calendars}"
        )
    unknown_currencies = sorted(set(out["currency"]) - SUPPORTED_CURRENCIES)
    if unknown_currencies:
        raise SignalFoundryDataError(
            f"bundle contains unsupported currencies: {unknown_currencies}"
        )
    if not out["ticker"].eq(out["instrument_id"]).all():
        raise SignalFoundryDataError("bundle ticker and instrument identity disagree")
    return out.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def _to_alphaforge_panel(source: pd.DataFrame) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for adjustment_state, group in source.groupby("adjustment_state", sort=True):
        block = group.copy()
        if adjustment_state in ADJUSTED_CLOSE_STATES:
            factor = block["adj_close"] / block["close"]
            if not np.isfinite(factor).all() or (factor <= 0).any():
                raise SignalFoundryDataError("bundle contains an invalid adjustment factor")
            for column in ("open", "high", "low", "close"):
                block[column] = block[column] * factor
        elif adjustment_state not in UNADJUSTED_STATES:
            raise SignalFoundryDataError(
                f"unsupported adjustment state {adjustment_state!r}; mapping must be explicit"
            )
        blocks.append(block)
    adjusted = pd.concat(blocks, ignore_index=True)
    panel = adjusted.rename(columns={"ticker": "symbol"})[
        ["date", "symbol", "open", "high", "low", "close", "volume"]
    ]
    return validate_panel(panel)


def _to_decision_panel(source: pd.DataFrame) -> pd.DataFrame:
    """Date bars at the first session close where they were actually available."""
    session_closes = (
        source.groupby("date", as_index=False)["effective_at"]
        .max()
        .sort_values("effective_at", kind="stable")
    )
    close_ns = session_closes["effective_at"].astype("int64").to_numpy()
    available_ns = source["available_at"].astype("int64").to_numpy()
    positions = np.searchsorted(close_ns, available_ns, side="left")
    eligible = positions < len(session_closes)
    if not eligible.any():
        raise SignalFoundryDataError(
            "bundle contains no observations available by a represented decision session"
        )
    decision_source = source.loc[eligible].copy()
    decision_source["date"] = session_closes["date"].to_numpy()[positions[eligible]]
    if decision_source.duplicated(["date", "ticker"]).any():
        raise SignalFoundryDataError(
            "availability mapping produces duplicate decision-session observations"
        )
    return _to_alphaforge_panel(decision_source)


def load_signal_foundry_dataset(
    bundle_dir: str | Path,
    *,
    as_of: str | datetime | pd.Timestamp | None = None,
) -> SignalFoundryDataset:
    """Verify and load one immutable Signal Foundry bundle.

    ``as_of`` applies the producer rule ``available_at <= decision timestamp``.
    Omit it only when a downstream temporal split will enforce decision-time
    eligibility row by row.
    """
    root = Path(bundle_dir)
    manifest = _read_manifest(root)
    _validate_manifest_policy(manifest)
    try:
        semantic = {key: manifest[key] for key in SEMANTIC_MANIFEST_FIELDS}
    except KeyError as exc:
        raise SignalFoundryDataError(
            f"bundle manifest is missing identity field: {exc.args[0]}"
        ) from exc
    if _sha256_bytes(_canonical_json(semantic)) != manifest["bundle_id"]:
        raise SignalFoundryDataError("bundle semantic identity mismatch")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise SignalFoundryDataError("bundle manifest has no data partitions")
    frames: list[pd.DataFrame] = []
    seen_paths: set[str] = set()
    total_rows = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise SignalFoundryDataError("bundle file entry must be an object")
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_rows = entry.get("rows")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise SignalFoundryDataError("bundle file entry is incomplete")
        if relative in seen_paths:
            raise SignalFoundryDataError(f"duplicate bundle path: {relative}")
        seen_paths.add(relative)
        path = _safe_file(root, relative)
        if not path.is_file():
            raise SignalFoundryDataError(f"bundle data partition is missing: {relative}")
        if _sha256_file(path) != expected_hash:
            raise SignalFoundryDataError(f"bundle partition hash mismatch: {relative}")
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise SignalFoundryDataError(f"cannot read bundle partition: {relative}") from exc
        if not isinstance(expected_rows, int) or len(frame) != expected_rows:
            raise SignalFoundryDataError(f"bundle partition row-count mismatch: {relative}")
        frames.append(frame)
        total_rows += len(frame)
    if total_rows != manifest.get("rows"):
        raise SignalFoundryDataError("bundle aggregate row-count mismatch")

    source = _coerce_source_frame(pd.concat(frames, ignore_index=True))
    if str(source["date"].min().date()) != manifest.get("date_min"):
        raise SignalFoundryDataError("bundle minimum date mismatch")
    if str(source["date"].max().date()) != manifest.get("date_max"):
        raise SignalFoundryDataError("bundle maximum date mismatch")
    if sorted(source["ticker"].unique().tolist()) != manifest.get("tickers"):
        raise SignalFoundryDataError("bundle ticker universe mismatch")

    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize(UTC) if cutoff.tzinfo is None else cutoff.tz_convert(UTC)
        source = source.loc[source["available_at"].le(cutoff)].reset_index(drop=True)
        if source.empty:
            raise SignalFoundryDataError("as-of cutoff excludes every bundle observation")
    panel = _to_alphaforge_panel(source)
    decision_panel = _to_decision_panel(source)
    return SignalFoundryDataset(
        bundle_dir=root,
        manifest=manifest,
        source_panel=source,
        panel=panel,
        decision_panel=decision_panel,
    )
