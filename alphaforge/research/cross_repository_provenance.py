"""Bounded verification of content-addressed evidence from a sibling repository.

Sprint 3 spans Signalattice and AlphaForge, but AlphaForge CI intentionally
does not clone or fetch Signalattice.  This module validates a committed,
non-executable receipt and can independently verify it against an already
available local Git object database.  Verification reads blobs from the pinned
commit; it never checks out a branch, consults the network, or trusts the
sibling worktree contents.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

CROSS_REPOSITORY_SCHEMA_VERSION = "1.0.0"
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_EXTERNAL_SOURCES = 32
MAX_EXTERNAL_SOURCE_BYTES = 4 * 1024 * 1024
MAX_EXTERNAL_TOTAL_BYTES = 16 * 1024 * 1024
GIT_TIMEOUT_SECONDS = 15

_RECEIPT_FIELDS = {
    "schema_version",
    "repository",
    "origin_url",
    "commit",
    "verification",
    "sources",
}
_VERIFICATION_FIELDS = {
    "mode",
    "verified_at_utc",
    "network_requests",
}
_SOURCE_FIELDS = {
    "family",
    "path",
    "git_blob_sha1",
    "bytes",
    "sha256",
    "claim_scope",
}


class CrossRepositoryProvenanceError(ValueError):
    """Raised when a receipt or pinned Git object fails validation."""


def _lower_hex(value: Any, *, length: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CrossRepositoryProvenanceError(
            f"{field} must be {length} lowercase hexadecimal characters"
        )
    return value


def _safe_text(value: Any, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
        or not value.isascii()
    ):
        raise CrossRepositoryProvenanceError(
            f"{field} must be non-empty, trimmed ASCII of at most {maximum} characters"
        )
    return value


def _safe_repository_path(value: Any) -> str:
    path_text = _safe_text(value, field="source.path", maximum=512)
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in path_text
        or ":" in path_text
        or any(
            not all(character.isalnum() or character in "._-" for character in part)
            for part in path.parts
        )
    ):
        raise CrossRepositoryProvenanceError(
            "source.path must be a safe repository-relative POSIX path"
        )
    return path_text


@dataclass(frozen=True)
class ExternalSource:
    """One exact Git blob supporting a bounded cross-repository claim."""

    family: str
    path: str
    git_blob_sha1: str
    bytes: int
    sha256: str
    claim_scope: str

    def __post_init__(self) -> None:
        _safe_text(self.family, field="source.family", maximum=128)
        _safe_repository_path(self.path)
        _lower_hex(self.git_blob_sha1, length=40, field="source.git_blob_sha1")
        _lower_hex(self.sha256, length=64, field="source.sha256")
        _safe_text(self.claim_scope, field="source.claim_scope", maximum=1000)
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or not 0 < self.bytes <= MAX_EXTERNAL_SOURCE_BYTES
        ):
            raise CrossRepositoryProvenanceError(
                f"source.bytes must be in [1, {MAX_EXTERNAL_SOURCE_BYTES}]"
            )


@dataclass(frozen=True)
class CrossRepositoryReceipt:
    """Immutable receipt for a single repository and commit."""

    repository: str
    origin_url: str
    commit: str
    verified_at_utc: str
    sources: tuple[ExternalSource, ...]

    def __post_init__(self) -> None:
        _safe_text(self.repository, field="repository", maximum=256)
        origin = _safe_text(self.origin_url, field="origin_url", maximum=512)
        if (
            not origin.startswith("https://github.com/")
            or not origin.endswith(".git")
            or "@" in origin
        ):
            raise CrossRepositoryProvenanceError(
                "origin_url must be a credential-free HTTPS GitHub repository URL"
            )
        _lower_hex(self.commit, length=40, field="commit")
        timestamp = _safe_text(
            self.verified_at_utc,
            field="verified_at_utc",
            maximum=64,
        )
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
        except ValueError as exc:
            raise CrossRepositoryProvenanceError(
                "verified_at_utc must be a valid ISO-8601 UTC timestamp"
            ) from exc
        if not timestamp.endswith("Z") or parsed_timestamp.tzinfo != UTC:
            raise CrossRepositoryProvenanceError(
                "verified_at_utc must be an explicit UTC timestamp ending in Z"
            )
        if not 0 < len(self.sources) <= MAX_EXTERNAL_SOURCES:
            raise CrossRepositoryProvenanceError(
                f"source count must be in [1, {MAX_EXTERNAL_SOURCES}]"
            )
        paths = [source.path for source in self.sources]
        if len(paths) != len(set(paths)):
            raise CrossRepositoryProvenanceError("source paths must be unique")
        total = sum(source.bytes for source in self.sources)
        if total > MAX_EXTERNAL_TOTAL_BYTES:
            raise CrossRepositoryProvenanceError(
                f"declared source bytes exceed {MAX_EXTERNAL_TOTAL_BYTES}"
            )


@dataclass(frozen=True)
class CrossRepositoryVerification:
    """Observed verification result without worktree-dependent state."""

    repository: str
    commit: str
    source_count: int
    total_bytes: int
    receipt_sha256: str


def _exact_fields(document: dict[str, Any], expected: set[str], context: str) -> None:
    observed = set(document)
    if observed != expected:
        raise CrossRepositoryProvenanceError(
            f"{context} fields mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def load_cross_repository_receipt(path: str | Path) -> CrossRepositoryReceipt:
    """Load one strict, bounded, non-symlinked JSON receipt."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise CrossRepositoryProvenanceError("receipt must be a regular non-symlink file")
    size = source.stat().st_size
    if not 0 < size <= MAX_RECEIPT_BYTES:
        raise CrossRepositoryProvenanceError(f"receipt bytes must be in [1, {MAX_RECEIPT_BYTES}]")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CrossRepositoryProvenanceError("receipt must be valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise CrossRepositoryProvenanceError("receipt root must be an object")
    _exact_fields(document, _RECEIPT_FIELDS, "receipt")
    if document["schema_version"] != CROSS_REPOSITORY_SCHEMA_VERSION:
        raise CrossRepositoryProvenanceError("unsupported receipt schema_version")
    verification = document["verification"]
    if not isinstance(verification, dict):
        raise CrossRepositoryProvenanceError("verification must be an object")
    _exact_fields(verification, _VERIFICATION_FIELDS, "verification")
    if verification["mode"] != "local_git_object_database":
        raise CrossRepositoryProvenanceError("unsupported verification mode")
    if verification["network_requests"] != 0:
        raise CrossRepositoryProvenanceError("receipt verification must be network-free")
    source_documents = document["sources"]
    if not isinstance(source_documents, list):
        raise CrossRepositoryProvenanceError("sources must be an array")
    sources: list[ExternalSource] = []
    for index, item in enumerate(source_documents):
        if not isinstance(item, dict):
            raise CrossRepositoryProvenanceError(f"sources[{index}] must be an object")
        _exact_fields(item, _SOURCE_FIELDS, f"sources[{index}]")
        sources.append(ExternalSource(**item))
    return CrossRepositoryReceipt(
        repository=document["repository"],
        origin_url=document["origin_url"],
        commit=document["commit"],
        verified_at_utc=verification["verified_at_utc"],
        sources=tuple(sources),
    )


def _run_git(
    checkout: Path,
    arguments: list[str],
    *,
    maximum_stdout: int,
) -> bytes:
    env = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
    }
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(checkout), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CrossRepositoryProvenanceError("bounded local Git command failed") from exc
    if len(result.stdout) > maximum_stdout or len(result.stderr) > 16_384:
        raise CrossRepositoryProvenanceError("local Git command exceeded output bounds")
    if result.returncode != 0:
        diagnostic = result.stderr.decode("utf-8", errors="replace").strip()[:1000]
        raise CrossRepositoryProvenanceError(
            f"local Git command rejected pinned evidence: {diagnostic}"
        )
    return result.stdout


def verify_cross_repository_receipt(
    receipt_path: str | Path,
    *,
    checkout: str | Path,
) -> CrossRepositoryVerification:
    """Verify every receipt source against a pinned local Git commit.

    The current branch, index, and worktree are irrelevant. Only objects
    reachable by the exact receipt commit are read.
    """

    checkout_path = Path(checkout)
    if checkout_path.is_symlink() or not checkout_path.is_dir():
        raise CrossRepositoryProvenanceError("checkout must be a real non-symlink directory")
    receipt_file = Path(receipt_path)
    try:
        initial_receipt_sha256 = hashlib.sha256(receipt_file.read_bytes()).hexdigest()
    except OSError as exc:
        raise CrossRepositoryProvenanceError("unable to read receipt") from exc
    receipt = load_cross_repository_receipt(receipt_path)
    inside = _run_git(
        checkout_path,
        ["rev-parse", "--is-inside-work-tree"],
        maximum_stdout=64,
    )
    if inside.strip() != b"true":
        raise CrossRepositoryProvenanceError("checkout is not a Git worktree")
    origin = (
        _run_git(
            checkout_path,
            ["config", "--get", "remote.origin.url"],
            maximum_stdout=1024,
        )
        .decode("utf-8", errors="strict")
        .strip()
    )
    if origin != receipt.origin_url:
        raise CrossRepositoryProvenanceError("origin mismatch")
    _run_git(
        checkout_path,
        ["cat-file", "-e", f"{receipt.commit}^{{commit}}"],
        maximum_stdout=0,
    )
    total_bytes = 0
    for source in receipt.sources:
        blob = (
            _run_git(
                checkout_path,
                ["rev-parse", f"{receipt.commit}:{source.path}"],
                maximum_stdout=128,
            )
            .decode("ascii")
            .strip()
        )
        if blob != source.git_blob_sha1:
            raise CrossRepositoryProvenanceError(
                f"Git blob mismatch for {source.path}: expected "
                f"{source.git_blob_sha1}, observed {blob}"
            )
        observed_size_text = (
            _run_git(
                checkout_path,
                ["cat-file", "-s", blob],
                maximum_stdout=64,
            )
            .decode("ascii")
            .strip()
        )
        try:
            observed_size = int(observed_size_text)
        except ValueError as exc:
            raise CrossRepositoryProvenanceError(
                f"Git returned an invalid byte count for {source.path}"
            ) from exc
        if observed_size != source.bytes:
            raise CrossRepositoryProvenanceError(
                f"byte-size mismatch for {source.path}: expected "
                f"{source.bytes}, observed {observed_size}"
            )
        content = _run_git(
            checkout_path,
            ["cat-file", "blob", blob],
            maximum_stdout=source.bytes,
        )
        if len(content) != source.bytes:
            raise CrossRepositoryProvenanceError(f"short Git blob read for {source.path}")
        observed_sha256 = hashlib.sha256(content).hexdigest()
        if observed_sha256 != source.sha256:
            raise CrossRepositoryProvenanceError(
                f"SHA-256 mismatch for {source.path}: expected "
                f"{source.sha256}, observed {observed_sha256}"
            )
        total_bytes += len(content)
    try:
        final_receipt_sha256 = hashlib.sha256(receipt_file.read_bytes()).hexdigest()
    except OSError as exc:
        raise CrossRepositoryProvenanceError("unable to re-read receipt") from exc
    if final_receipt_sha256 != initial_receipt_sha256:
        raise CrossRepositoryProvenanceError("receipt changed during verification")
    return CrossRepositoryVerification(
        repository=receipt.repository,
        commit=receipt.commit,
        source_count=len(receipt.sources),
        total_bytes=total_bytes,
        receipt_sha256=initial_receipt_sha256,
    )
