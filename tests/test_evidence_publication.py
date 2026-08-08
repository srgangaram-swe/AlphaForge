"""Fault-injection tests for atomic evidence publication (SF-S5-MR11).

The original close-out wrote directly into its destination with a manifest
listing only filenames. Two failures followed: a crash partway through left a
directory that looked like a bundle, and nothing could detect that an artifact
had changed after publication.

The guarantees these tests protect:

* **All-or-nothing.** An injected failure at any stage leaves no destination and
  no staging residue.
* **A published bundle is never overwritten.**
* **One changed byte fails verification.**
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alphaforge.evidence.publication import (
    MANIFEST_NAME,
    SELF_HASH_NOTE,
    ArtifactRecord,
    PublicationError,
    publish_bundle,
    verify_bundle,
)


def _writer(staging: Path) -> None:
    (staging / "alpha.json").write_text('{"a": 1}\n', encoding="utf-8")
    (staging / "beta.csv").write_text("x,y\n1,2\n", encoding="utf-8")


def _publish(destination: Path, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "writer": _writer,
        "schema_version": 1,
        "generator": "test",
        "source_ref": "abc1234",
        "environment": {"machine": "arm64"},
        "input_identities": {"raw": "a" * 64},
    }
    base.update(kw)
    return publish_bundle(destination, **base)


def _staging_residue(parent: Path) -> list[Path]:
    return [item for item in parent.iterdir() if item.name.startswith(".")]


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def test_a_bundle_publishes_with_a_manifest(tmp_path: Path) -> None:
    manifest = _publish(tmp_path / "bundle")
    assert (tmp_path / "bundle" / MANIFEST_NAME).is_file()
    assert manifest["artifact_count"] == 2
    assert {item["name"] for item in manifest["artifacts"]} == {"alpha.json", "beta.csv"}


def test_a_writer_failure_leaves_nothing_behind(tmp_path: Path) -> None:
    """Not a partial bundle, and not a staging directory."""

    def failing(staging: Path) -> None:
        (staging / "partial.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("simulated render failure")

    with pytest.raises(RuntimeError, match="simulated render failure"):
        _publish(tmp_path / "bundle", writer=failing)
    assert not (tmp_path / "bundle").exists()
    assert _staging_residue(tmp_path) == []


def test_a_rename_failure_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_replace(source: Any, target: Any) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("alphaforge.evidence.publication.os.replace", failing_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        _publish(tmp_path / "bundle")
    monkeypatch.undo()
    assert not (tmp_path / "bundle").exists()
    assert _staging_residue(tmp_path) == []


def test_an_empty_writer_result_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PublicationError, match="no artifacts"):
        _publish(tmp_path / "bundle", writer=lambda staging: None)
    assert not (tmp_path / "bundle").exists()


def test_an_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    """Something may already cite the published bundle."""
    _publish(tmp_path / "bundle")
    with pytest.raises(PublicationError, match="never overwritten"):
        _publish(tmp_path / "bundle")


def test_a_symlinked_destination_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(PublicationError, match="already exists"):
        _publish(link)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_a_freshly_published_bundle_verifies(tmp_path: Path) -> None:
    _publish(tmp_path / "bundle")
    assert verify_bundle(tmp_path / "bundle")["artifact_count"] == 2


def test_one_changed_byte_fails_verification(tmp_path: Path) -> None:
    _publish(tmp_path / "bundle")
    target = tmp_path / "bundle" / "alpha.json"
    content = target.read_bytes()
    target.write_bytes(content[:-2] + b"2\n")
    with pytest.raises(PublicationError, match="changed since publication"):
        verify_bundle(tmp_path / "bundle")


def test_a_removed_artifact_fails_verification(tmp_path: Path) -> None:
    _publish(tmp_path / "bundle")
    (tmp_path / "bundle" / "beta.csv").unlink()
    with pytest.raises(PublicationError, match="missing"):
        verify_bundle(tmp_path / "bundle")


def test_an_added_artifact_fails_verification(tmp_path: Path) -> None:
    """A file nobody recorded is as much a divergence as a changed one."""
    _publish(tmp_path / "bundle")
    (tmp_path / "bundle" / "smuggled.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(PublicationError, match="unrecorded"):
        verify_bundle(tmp_path / "bundle")


def test_a_bundle_without_a_manifest_is_refused(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(PublicationError, match="no manifest"):
        verify_bundle(bare)


def test_an_unreadable_manifest_is_refused(tmp_path: Path) -> None:
    _publish(tmp_path / "bundle")
    (tmp_path / "bundle" / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(PublicationError, match="not readable JSON"):
        verify_bundle(tmp_path / "bundle")


# ---------------------------------------------------------------------------
# Manifest contents
# ---------------------------------------------------------------------------


def test_the_manifest_records_a_hash_and_size_for_every_artifact(tmp_path: Path) -> None:
    manifest = _publish(tmp_path / "bundle")
    for record in manifest["artifacts"]:
        assert len(record["sha256"]) == 64
        assert record["bytes"] > 0


def test_the_manifest_excludes_itself_and_explains_why(tmp_path: Path) -> None:
    """A file cannot contain its own digest; the note says so rather than leaving
    a reader to wonder whether the omission is a bug."""
    manifest = _publish(tmp_path / "bundle")
    assert MANIFEST_NAME not in {item["name"] for item in manifest["artifacts"]}
    assert manifest["self_hash_note"] == SELF_HASH_NOTE
    assert "no fixed point exists" in SELF_HASH_NOTE


def test_the_manifest_carries_provenance(tmp_path: Path) -> None:
    manifest = _publish(tmp_path / "bundle")
    assert manifest["generator"] == "test"
    assert manifest["source_ref"] == "abc1234"
    assert manifest["environment"] == {"machine": "arm64"}
    assert manifest["input_identities"] == {"raw": "a" * 64}


def test_extra_fields_cannot_shadow_manifest_keys(tmp_path: Path) -> None:
    with pytest.raises(PublicationError, match="collide"):
        _publish(tmp_path / "bundle", extra={"generator": "impostor"})


def test_extra_fields_are_merged(tmp_path: Path) -> None:
    manifest = _publish(tmp_path / "bundle", extra={"sprint": 5})
    assert manifest["sprint"] == 5


def test_a_malformed_artifact_record_is_refused() -> None:
    with pytest.raises(PublicationError, match="bare filename"):
        ArtifactRecord(name="nested/path.json", sha256="a" * 64, bytes=1)
    with pytest.raises(PublicationError, match="full digest"):
        ArtifactRecord(name="a.json", sha256="short", bytes=1)


def test_the_manifest_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    first = _publish(tmp_path / "one")
    second = _publish(tmp_path / "two")
    assert first["artifacts"] == second["artifacts"]
    assert first["total_bytes"] == second["total_bytes"]
