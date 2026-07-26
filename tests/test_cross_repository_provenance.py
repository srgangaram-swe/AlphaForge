"""Tests for bounded sibling-repository provenance verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alphaforge.research.cross_repository_provenance import (
    CrossRepositoryProvenanceError,
    load_cross_repository_receipt,
    verify_cross_repository_receipt,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    repository = tmp_path / "source"
    repository.mkdir(parents=True)
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "Test Owner")
    _git(repository, "config", "user.email", "owner@example.invalid")
    _git(
        repository,
        "remote",
        "add",
        "origin",
        "https://github.com/example/source.git",
    )
    evidence = repository / "docs" / "evidence.json"
    evidence.parent.mkdir()
    content = b'{"scope":"aggregate-only"}\n'
    evidence.write_bytes(content)
    _git(repository, "add", "docs/evidence.json")
    _git(repository, "commit", "-m", "Add evidence")
    commit = _git(repository, "rev-parse", "HEAD")
    blob = _git(repository, "rev-parse", f"{commit}:docs/evidence.json")
    document = {
        "schema_version": "1.0.0",
        "repository": "example/source",
        "origin_url": "https://github.com/example/source.git",
        "commit": commit,
        "verification": {
            "mode": "local_git_object_database",
            "verified_at_utc": "2026-07-26T22:00:00Z",
            "network_requests": 0,
        },
        "sources": [
            {
                "family": "aggregate_fixture",
                "path": "docs/evidence.json",
                "git_blob_sha1": blob,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "claim_scope": "aggregate fixture contract only",
            }
        ],
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return repository, receipt, document


def test_verification_reads_pinned_commit_not_dirty_worktree(tmp_path: Path) -> None:
    repository, receipt, _ = _fixture(tmp_path)

    first = verify_cross_repository_receipt(receipt, checkout=repository)
    (repository / "docs" / "evidence.json").write_text(
        '{"scope":"dirty-worktree"}\n',
        encoding="utf-8",
    )
    second = verify_cross_repository_receipt(receipt, checkout=repository)

    assert first == second
    assert first.source_count == 1
    assert first.total_bytes == len(b'{"scope":"aggregate-only"}\n')


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda document: document["sources"][0].update({"sha256": "0" * 64}),
            "SHA-256 mismatch",
        ),
        (
            lambda document: document["sources"][0].update({"bytes": 1}),
            "byte-size mismatch",
        ),
        (
            lambda document: document.update(
                {"origin_url": "https://github.com/example/other.git"}
            ),
            "origin mismatch",
        ),
        (
            lambda document: document["sources"][0].update({"path": "../escape"}),
            "safe repository-relative",
        ),
    ],
)
def test_tampered_receipts_fail_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    repository, receipt, document = _fixture(tmp_path)
    mutation(document)
    receipt.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CrossRepositoryProvenanceError, match=match):
        verify_cross_repository_receipt(receipt, checkout=repository)


def test_loader_rejects_unknown_fields_duplicates_and_symlinks(tmp_path: Path) -> None:
    _, receipt, document = _fixture(tmp_path)
    document["unknown"] = True
    receipt.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CrossRepositoryProvenanceError, match="fields mismatch"):
        load_cross_repository_receipt(receipt)

    _, receipt, document = _fixture(tmp_path / "duplicate")
    document["sources"].append(dict(document["sources"][0]))
    receipt.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CrossRepositoryProvenanceError, match="unique"):
        load_cross_repository_receipt(receipt)

    _, receipt, _ = _fixture(tmp_path / "linked")
    link = tmp_path / "receipt-link.json"
    link.symlink_to(receipt)
    with pytest.raises(CrossRepositoryProvenanceError, match="symlink"):
        load_cross_repository_receipt(link)


def test_verifier_rejects_symlinked_checkout(tmp_path: Path) -> None:
    repository, receipt, _ = _fixture(tmp_path)
    link = tmp_path / "checkout-link"
    link.symlink_to(repository, target_is_directory=True)

    with pytest.raises(CrossRepositoryProvenanceError, match="symlink"):
        verify_cross_repository_receipt(receipt, checkout=link)
