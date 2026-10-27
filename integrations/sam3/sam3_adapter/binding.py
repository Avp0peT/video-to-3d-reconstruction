"""Model code / weights / configuration binding.

Two facts drive this module:

1. ``build_sam3_image_model`` does **not** read ``config.json`` -- it builds the
   architecture from hard-coded literals. Recording a downloaded ``config.json``
   hash as "the verified configuration" would therefore dress up a file that is
   never consumed. What actually determines the forward pass here is the
   *effective construction options* (code commit, BPE asset, device, resolution,
   confidence threshold, load flags), so that is what gets hashed into
   ``config_sha256``. A user-supplied ``--config`` file is recorded separately
   and explicitly as *declared, not consumed*.
2. A local checkpoint is only meaningful together with its expected digest, so
   the hash requirement is enforced *before* any heavy import or load.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_json(payload: dict) -> str:
    """Deterministic JSON: sorted keys, no extra whitespace, UTF-8 preserved."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_options_digest(options: dict) -> str:
    """Digest of the options that actually reach the model constructor."""
    return sha256_of_text(canonical_json(options))


def file_binding(path: str | Path | None, expected_sha256: str | None = None) -> dict | None:
    if path is None:
        return None
    target = Path(path)
    actual = sha256_file(target) if target.is_file() else None
    return {
        "path": str(target),
        "bytes": target.stat().st_size if target.is_file() else None,
        "sha256": actual,
        "sha256_expected": expected_sha256,
        "sha256_matches": (actual == expected_sha256) if (actual and expected_sha256) else None,
    }
