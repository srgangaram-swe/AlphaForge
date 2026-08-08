"""All-or-nothing evidence publication with a verifiable manifest.

SF-S5-MR11. The original close-out wrote its files directly into the destination
and recorded a manifest listing only filenames. Two consequences: a failure
partway through left a directory that looked like a bundle but was not one, and
the manifest could not detect that any artifact had changed since publication.

**Publication is atomic.** Artifacts are written into a staging sibling,
validated, fsynced, and only then renamed into a destination that must not
already exist. A failure at any point removes the staging directory and leaves
nothing behind — there is no state in which a partial bundle is visible under the
final name.

**The manifest carries hashes and sizes**, so `verify_bundle` detects a
single changed byte. It cannot contain its own hash: writing the digest into the
file changes the file, and no fixed point exists. It therefore records every
*other* artifact, and the note in the manifest says exactly that rather than
leaving a reader to wonder whether the omission is an oversight.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: The manifest filename, excluded from its own artifact records.
MANIFEST_NAME: Final = "manifest.json"

#: Refusal thresholds, not tuning knobs.
MAX_ARTIFACTS: Final = 256
MAX_ARTIFACT_BYTES: Final = 64 * 1024 * 1024
MAX_BUNDLE_BYTES: Final = 512 * 1024 * 1024

#: Why the manifest cannot record its own digest, stated in the manifest itself.
SELF_HASH_NOTE: Final = (
    "manifest.json is absent from `artifacts` because a file cannot contain its own "
    "SHA-256: writing the digest changes the bytes the digest was computed over, and no "
    "fixed point exists. Verify the manifest against an external reference — the merge "
    "commit that introduced it — and every other artifact against this manifest."
)


class PublicationError(ValueError):
    """Raised when a bundle cannot be published or verified."""


def _digest_file(path: Path) -> str:
    """Return the SHA-256 of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_safe_destination(destination: Path) -> None:
    """Refuse an unsafe or already-populated destination.

    Raises:
        PublicationError: On a symlink, an existing entry, or an unsafe parent.
    """
    if destination.exists() or destination.is_symlink():
        raise PublicationError(
            f"{destination} already exists; a published bundle is never overwritten because "
            "something may already cite it. Choose a new path or remove it deliberately."
        )
    parent = destination.parent
    if parent.exists():
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise PublicationError("destination parent must not be a symlink")
        if not stat.S_ISDIR(info.st_mode):
            raise PublicationError("destination parent must be a directory")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise PublicationError("destination parent must be owned by the current user")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One published file's identity and size."""

    name: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "/" in self.name:
            raise PublicationError("artifact name must be a bare filename")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise PublicationError(f"{self.name}: sha256 must be a full digest")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise PublicationError(f"{self.name}: bytes must be a non-negative int")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {"name": self.name, "sha256": self.sha256, "bytes": self.bytes}


def publish_bundle(
    destination: Path | str,
    *,
    writer: Callable[[Path], None],
    schema_version: int,
    generator: str,
    source_ref: str,
    environment: Mapping[str, Any],
    input_identities: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a bundle atomically and return its manifest.

    ``writer`` receives a staging directory and populates it. It is called
    exactly once; if it raises, the staging directory is removed and the
    destination is never created.

    Raises:
        PublicationError: On an unsafe or existing destination, an oversized
            bundle, or an empty result.
    """
    target = Path(destination)
    _assert_safe_destination(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}-staging-"))
    try:
        os.chmod(staging, 0o700)
        writer(staging)

        entries = sorted(item for item in staging.iterdir() if item.is_file())
        if not entries:
            raise PublicationError("writer produced no artifacts")
        if len(entries) > MAX_ARTIFACTS:
            raise PublicationError(f"bundle exceeds the {MAX_ARTIFACTS}-artifact ceiling")
        total = 0
        records: list[ArtifactRecord] = []
        for entry in entries:
            size = entry.stat().st_size
            if size > MAX_ARTIFACT_BYTES:
                raise PublicationError(f"{entry.name} exceeds the artifact size ceiling")
            total += size
            if total > MAX_BUNDLE_BYTES:
                raise PublicationError("bundle exceeds the total size ceiling")
            records.append(ArtifactRecord(name=entry.name, sha256=_digest_file(entry), bytes=size))

        manifest: dict[str, Any] = {
            "schema_version": schema_version,
            "generator": generator,
            "source_ref": source_ref,
            "environment": dict(environment),
            "input_identities": dict(input_identities),
            "artifacts": [item.to_dict() for item in records],
            "artifact_count": len(records),
            "total_bytes": total,
            "self_hash_note": SELF_HASH_NOTE,
        }
        if extra:
            overlap = set(extra) & set(manifest)
            if overlap:
                raise PublicationError(
                    f"extra fields collide with manifest keys: {sorted(overlap)}"
                )
            manifest.update(extra)

        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # fsync every file and the staging directory before the rename, so the
        # bundle is durable at the instant it becomes visible.
        for entry in sorted(staging.iterdir()):
            if entry.is_file():
                with entry.open("rb") as stream:
                    os.fsync(stream.fileno())
        directory_handle = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_handle)
        finally:
            os.close(directory_handle)

        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def verify_bundle(bundle: Path | str) -> dict[str, Any]:
    """Recompute every artifact digest and refuse any mismatch.

    Raises:
        PublicationError: Naming the artifacts that changed, are missing, or
            appeared without being recorded.
    """
    root = Path(bundle)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PublicationError(f"{root} has no {MANIFEST_NAME}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PublicationError(f"{MANIFEST_NAME} is not readable JSON: {error}") from error
    if not isinstance(manifest, Mapping) or "artifacts" not in manifest:
        raise PublicationError(f"{MANIFEST_NAME} does not record artifacts")

    recorded = {item["name"]: item for item in manifest["artifacts"]}
    present = {item.name for item in root.iterdir() if item.is_file()} - {MANIFEST_NAME}

    missing = sorted(set(recorded) - present)
    unrecorded = sorted(present - set(recorded))
    if missing or unrecorded:
        raise PublicationError(
            f"bundle contents do not match the manifest; missing={missing}, "
            f"unrecorded={unrecorded}"
        )

    changed: list[str] = []
    for name, record in sorted(recorded.items()):
        path = root / name
        if path.stat().st_size != record["bytes"] or _digest_file(path) != record["sha256"]:
            changed.append(name)
    if changed:
        raise PublicationError(
            f"{len(changed)} artifact(s) changed since publication: {changed}. A single "
            "differing byte is enough to fail this check, which is the point."
        )
    return dict(manifest)


__all__ = [
    "MANIFEST_NAME",
    "MAX_ARTIFACTS",
    "SELF_HASH_NOTE",
    "ArtifactRecord",
    "PublicationError",
    "publish_bundle",
    "verify_bundle",
]
