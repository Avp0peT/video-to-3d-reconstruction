"""Hashing, identity and cache fingerprints (``photo-io-v1`` §2 and §6).

Three distinct digests live here and must not be confused:

* ``image_sha256`` / ``mask_sha256`` — over the **raw, undecoded file bytes**.
* ``frame_id`` — identity of a *dataset entry*, derived from
  ``[dataset_id, canonical image path, image bytes hash]`` only.  Deliberately
  excludes the absolute root, the list row number, mtime and the mask, so a
  dataset can be relocated without changing identities.
* ``cache_fingerprint`` — invalidation key for a *processing result*.  Includes
  the mask bytes and the implementation digest, so a mask edit or a code edit
  invalidates it even though ``frame_id`` stays the same.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
from pathlib import Path
from typing import Any, Iterable

from .paths import canonical_json

__all__ = [
    "CONTRACT_VERSION",
    "compute_cache_fingerprint",
    "compute_frame_id",
    "runtime_info",
    "sha256_bytes",
    "sha256_file",
    "source_tree_sha256",
]

#: The one and only schema version this package implements.
CONTRACT_VERSION = "photo-io-v1"

#: Packages whose source files participate in ``implementation_sha256``.
_IMPLEMENTATION_PACKAGES = ("io", "segmentation")


def sha256_bytes(payload: bytes) -> str:
    """Lowercase hex SHA-256 of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Lowercase hex SHA-256 of the file's bytes, read in one pass."""
    return sha256_bytes(path.read_bytes())


def compute_frame_id(dataset_id: str, canonical_image_path: str, image_sha256: str) -> str:
    """``"f_" + sha256(UTF8(JSON([dataset_id, path, hash])))``.

    The JSON is written with ``separators=(",", ":")`` and ``ensure_ascii=False``
    exactly as the contract prescribes, so the digest is reproducible across
    machines and locales.
    """
    identity_bytes = json.dumps(
        [dataset_id, canonical_image_path, image_sha256],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "f_" + sha256_bytes(identity_bytes)


def runtime_info() -> dict[str, str]:
    """Actual interpreter and library versions, recorded into ``run.json``."""

    def _version(distribution: str) -> str:
        try:
            return importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover - env dependent
            return "unknown"

    return {
        "python": platform.python_version(),
        "pillow": _version("pillow"),
        "numpy": _version("numpy"),
    }


def _iter_python_sources(project_root: Path) -> Iterable[Path]:
    """Every ``*.py`` under the implementation packages, sorted by relative path."""
    found: list[tuple[str, Path]] = []
    for package in _IMPLEMENTATION_PACKAGES:
        package_root = project_root / "src" / "kozlov_lab" / package
        if not package_root.is_dir():
            continue
        for path in package_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(project_root).as_posix()
            found.append((rel, path))
    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def source_tree_sha256(project_root: Path) -> str:
    """``implementation_sha256`` — digest of ``[relative_path, sha256]`` pairs.

    Only ``io/`` and ``segmentation/`` participate: the digest exists to make a
    cached result invalid when the IO implementation changes, not to fingerprint
    the whole repository.
    """
    pairs: list[list[str]] = []
    for path in _iter_python_sources(project_root):
        rel = path.relative_to(project_root).as_posix()
        pairs.append([rel, sha256_bytes(path.read_bytes())])
    return sha256_bytes(canonical_json(pairs).encode("utf-8"))


def compute_cache_fingerprint(
    *,
    dataset_id: str,
    data_kind: str,
    image_path: str | None,
    mask_path: str | None,
    image_sha256: str | None,
    mask_sha256: str | None,
    implementation_sha256: str,
    runtime: dict[str, Any],
    contract_version: str = CONTRACT_VERSION,
) -> str:
    """Invalidation key for one processed frame (contract clause 6).

    Paths, mtimes, ``frame_id`` and the image hash alone are **not** sufficient
    inputs; the mask bytes and the implementation digest are included so that a
    mask edit or a code edit invalidates the entry.
    """
    payload = {
        "contract_version": contract_version,
        "dataset_id": dataset_id,
        "data_kind": data_kind,
        "image_path": image_path,
        "mask_path": mask_path,
        "image_sha256": image_sha256,
        "mask_sha256": mask_sha256,
        "implementation_sha256": implementation_sha256,
        "runtime": runtime,
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))
