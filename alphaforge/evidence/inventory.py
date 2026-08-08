"""Delivery scope derived from Git objects, not from counting by hand.

SF-S5-MR11. The original close-out figure plotted per-slice "tests added" from a
hand-typed table. Those numbers were produced by reading pytest output and typing
what it said, which is exactly the failure mode that makes a figure disagree with
its repository the moment either changes.

This module derives the same scope from **frozen commit identities and tracked
paths**, so the ledger and the plot cannot diverge from what was actually merged.

**It counts test functions, and says so.** A test function, a collected parameter
case, and a passing test are three different numbers, and conflating them
overstates coverage. `test_any_single_unmet_item_blocks` is one function and
seventeen collected cases; this module reports it as one function and labels the
column accordingly. The distinction is stated on every record rather than left
for a reader to guess.

Counting is textual — a module-level ``def test_`` — not by importing and
collecting. Importing test modules to count them would execute their import-time
code inside an evidence generator, which is both slower and a much larger blast
radius than reading files.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: Refusal thresholds, not tuning knobs.
MAX_SLICES: Final = 64
MAX_PATHS_PER_SLICE: Final = 256
MAX_FILE_BYTES: Final = 4 * 1024 * 1024

#: What the delivery count actually measures. Stated on every record so it is
#: never read as "tests that passed" or "cases collected".
COUNT_SEMANTICS: Final = (
    "module-level test functions defined in the named files, counted by parsing "
    "the source. Not collected parameter cases and not passing tests: one "
    "parametrized function is counted once."
)


class InventoryError(ValueError):
    """Raised when a delivery ledger cannot be derived or is malformed."""


def _run_git(arguments: Sequence[str], *, repository: Path) -> str:
    """Run a bounded read-only git command and return its stdout.

    Raises:
        InventoryError: On a non-zero exit or a timeout.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *arguments],  # noqa: S607 - git resolved from PATH by design
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise InventoryError(f"git {' '.join(arguments)} failed: {error}") from error
    if completed.returncode != 0:
        raise InventoryError(
            f"git {' '.join(arguments)} exited {completed.returncode}: "
            f"{completed.stderr.strip()[:200]}"
        )
    return completed.stdout


def resolve_commit(reference: str, *, repository: Path) -> str:
    """Return the full object id for a reference.

    Raises:
        InventoryError: If the reference does not resolve.
    """
    if not isinstance(reference, str) or not reference.strip():
        raise InventoryError("reference must be a non-empty string")
    resolved = _run_git(["rev-parse", "--verify", f"{reference}^{{commit}}"], repository=repository)
    identity = resolved.strip()
    if len(identity) != 40:
        raise InventoryError(f"{reference!r} did not resolve to a commit id")
    return identity


def count_test_functions(source: str) -> int:
    """Return the number of module-level ``test_`` functions in a source file.

    Parsed rather than imported: an evidence generator should not execute the
    import-time code of every test module it counts.

    Raises:
        InventoryError: If the source does not parse.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise InventoryError(f"test source does not parse: {error}") from error
    return sum(
        1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


@dataclass(frozen=True)
class SliceRecord:
    """One merged slice and the tracked paths it delivered."""

    label: str
    commit: str
    source_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    doc_paths: tuple[str, ...]
    test_function_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise InventoryError("label must be a non-empty string")
        if not isinstance(self.commit, str) or len(self.commit) != 40:
            raise InventoryError(f"{self.label}: commit must be a full 40-character object id")
        for field_name in ("source_paths", "test_paths", "doc_paths"):
            paths = tuple(getattr(self, field_name))
            if len(paths) > MAX_PATHS_PER_SLICE:
                raise InventoryError(f"{self.label}: {field_name} exceeds the path ceiling")
            if len(set(paths)) != len(paths):
                raise InventoryError(f"{self.label}: duplicate path in {field_name}")
            object.__setattr__(self, field_name, tuple(sorted(paths)))
        if isinstance(self.test_function_count, bool) or not isinstance(
            self.test_function_count, int
        ):
            raise InventoryError("test_function_count must be an int")
        if self.test_function_count < 0:
            raise InventoryError("test_function_count must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly ledger row."""
        return {
            "label": self.label,
            "commit": self.commit,
            "source_paths": list(self.source_paths),
            "test_paths": list(self.test_paths),
            "doc_paths": list(self.doc_paths),
            "test_function_count": self.test_function_count,
            "counts": COUNT_SEMANTICS,
        }


@dataclass(frozen=True)
class DeliveryLedger:
    """The complete machine-readable delivery scope for a sprint."""

    sprint: int
    slices: tuple[SliceRecord, ...]

    def __post_init__(self) -> None:
        if isinstance(self.sprint, bool) or not isinstance(self.sprint, int) or self.sprint < 1:
            raise InventoryError("sprint must be a positive int")
        slices = tuple(self.slices)
        if not slices:
            raise InventoryError("a ledger must contain at least one slice")
        if len(slices) > MAX_SLICES:
            raise InventoryError(f"exceeds the {MAX_SLICES}-slice ceiling")
        labels = [item.label for item in slices]
        if len(set(labels)) != len(labels):
            raise InventoryError("slice labels must be unique")
        commits = [item.commit for item in slices]
        if len(set(commits)) != len(commits):
            raise InventoryError(
                "two slices name the same commit; the ledger would double-count its work"
            )
        object.__setattr__(self, "slices", slices)

    @property
    def total_test_functions(self) -> int:
        """Total across slices. The number a plot may display."""
        return sum(item.test_function_count for item in self.slices)

    def identity(self) -> str:
        """Content identity over the whole ledger."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def reconciles_to(self, plotted: Mapping[str, int]) -> bool:
        """Whether a plotted mapping exactly matches this ledger."""
        return {item.label: item.test_function_count for item in self.slices} == dict(plotted)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly ledger."""
        return {
            "sprint": self.sprint,
            "slices": [item.to_dict() for item in self.slices],
            "total_test_functions": self.total_test_functions,
            "count_semantics": COUNT_SEMANTICS,
        }


def build_ledger(
    sprint: int,
    slice_specifications: Sequence[Mapping[str, Any]],
    *,
    repository: Path,
) -> DeliveryLedger:
    """Derive a ledger from frozen references and tracked paths.

    Each specification supplies a ``label``, a ``reference`` resolved to a commit
    id, and ``source_paths`` / ``test_paths`` / ``doc_paths``. Test functions are
    counted from the working tree, which the caller is responsible for having
    checked out at the reference.

    Raises:
        InventoryError: On an unresolvable reference, a missing path, or an
            oversized file.
    """
    records: list[SliceRecord] = []
    for specification in slice_specifications:
        missing_keys = {"label", "reference", "source_paths", "test_paths", "doc_paths"} - set(
            specification
        )
        if missing_keys:
            raise InventoryError(f"slice specification is missing {sorted(missing_keys)}")
        commit = resolve_commit(str(specification["reference"]), repository=repository)
        total = 0
        for relative in specification["test_paths"]:
            path = repository / str(relative)
            if not path.is_file():
                raise InventoryError(
                    f"{specification['label']}: test path {relative} is not a file; the ledger "
                    "must describe paths that exist"
                )
            if path.stat().st_size > MAX_FILE_BYTES:
                raise InventoryError(f"{relative} exceeds the {MAX_FILE_BYTES}-byte ceiling")
            total += count_test_functions(path.read_text(encoding="utf-8"))
        for relative in (*specification["source_paths"], *specification["doc_paths"]):
            if not (repository / str(relative)).exists():
                raise InventoryError(f"{specification['label']}: path {relative} does not exist")
        records.append(
            SliceRecord(
                label=str(specification["label"]),
                commit=commit,
                source_paths=tuple(str(item) for item in specification["source_paths"]),
                test_paths=tuple(str(item) for item in specification["test_paths"]),
                doc_paths=tuple(str(item) for item in specification["doc_paths"]),
                test_function_count=total,
            )
        )
    return DeliveryLedger(sprint=sprint, slices=tuple(records))


def ledger_from_payload(payload: Mapping[str, Any]) -> DeliveryLedger:
    """Rebuild a ledger from its stored payload.

    Raises:
        InventoryError: On missing fields.
    """
    if not isinstance(payload, Mapping) or "slices" not in payload or "sprint" not in payload:
        raise InventoryError("ledger payload must contain 'sprint' and 'slices'")
    return DeliveryLedger(
        sprint=int(payload["sprint"]),
        slices=tuple(
            SliceRecord(
                label=str(item["label"]),
                commit=str(item["commit"]),
                source_paths=tuple(str(path) for path in item["source_paths"]),
                test_paths=tuple(str(path) for path in item["test_paths"]),
                doc_paths=tuple(str(path) for path in item["doc_paths"]),
                test_function_count=int(item["test_function_count"]),
            )
            for item in payload["slices"]
        ),
    )


__all__ = [
    "COUNT_SEMANTICS",
    "MAX_SLICES",
    "DeliveryLedger",
    "InventoryError",
    "SliceRecord",
    "build_ledger",
    "count_test_functions",
    "ledger_from_payload",
    "resolve_commit",
]
