"""Path normalisation, containment and deterministic sort keys (``photo-io-v1`` §2).

This module is the *only* place that decides what a legal dataset-relative path is.
Every other module consumes :func:`normalize_rel_path` / :func:`resolve_within_root`
instead of touching :mod:`pathlib` directly, so the rules cannot drift apart.

Contract rules implemented here
------------------------------
* Normalisation: ``\\`` -> ``/``, collapse repeated separators and ``.``,
  Unicode -> NFC, **case preserved**.  Reject empty, any ``..`` component,
  absolute paths, drive letters, UNC prefixes and ``NUL``.
* Root containment is checked on the **resolved** path, so a symlink or a
  Windows reparse point that escapes the declared root is rejected.
* Sort keys are a single string per contract clause 2: canonical path when the
  path is legal; the raw string when the field was a string but illegal;
  the empty string when the field is missing/``null``; otherwise the canonical
  JSON serialisation of the JSON value.  Strings compare by Unicode code point,
  which is plain Python ``str`` ordering.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

__all__ = [
    "DATASET_ID_RE",
    "canonical_json",
    "find_project_root",
    "is_utf8_encodable",
    "is_valid_dataset_id",
    "normalize_rel_path",
    "project_relative",
    "resolve_within_root",
    "sort_key_for_path_field",
]

#: ``dataset_id`` is case sensitive and must start alphanumeric.
#:
#: Deliberately **unanchored**: this pattern is used with :func:`re.fullmatch`.
#: Anchoring with ``$`` would be a defect rather than a style choice, because in
#: Python ``$`` also matches immediately *before a trailing newline*, so
#: ``"demo01\n"`` would be accepted as a legal id.
DATASET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def is_valid_dataset_id(value: Any) -> bool:
    """Whether ``value`` is a legal ``dataset_id`` (case sensitive, whole string).

    The match must cover the entire string: a trailing newline or any other
    stray character makes the id illegal.
    """
    return isinstance(value, str) and DATASET_ID_RE.fullmatch(value) is not None


def is_utf8_encodable(value: str) -> bool:
    """Whether ``value`` can be represented as UTF-8.

    A JSON string may contain an **unpaired surrogate** such as ``"\\ud800"``;
    Python holds it happily, but it is not a Unicode scalar value, cannot be
    encoded as UTF-8, and cannot name a file on any platform.  Such a value must
    be refused as a path *before* anything tries to write it, otherwise the
    failure surfaces later as a crash while serialising an unrelated record.
    """
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def normalize_rel_path(raw: Any) -> str | None:
    """Return the canonical relative path, or ``None`` if the value is illegal.

    ``None`` is returned for: non-strings, empty strings, embedded ``NUL``,
    strings that are not UTF-8 encodable, absolute paths, drive letters, UNC
    prefixes, and any path containing a ``..`` component.  The returned value
    uses ``/`` separators and NFC.
    """
    if not isinstance(raw, str):
        return None
    if raw == "" or "\x00" in raw:
        return None
    # Checked before normalisation: a lone surrogate must not reach unicodedata,
    # and must never be treated as a usable path.
    if not is_utf8_encodable(raw):
        return None

    unified = raw.replace("\\", "/")
    # UNC (``\\\\server\\share``) collapses to a leading ``//`` after unifying.
    if unified.startswith("/"):
        return None
    if _DRIVE_RE.match(unified):
        return None

    parts: list[str] = []
    for segment in unified.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            return None
        parts.append(segment)
    if not parts:
        return None
    return unicodedata.normalize("NFC", "/".join(parts))


def find_project_root(start: Path) -> Path:
    """Walk upwards from ``start`` for the directory holding ``pyproject.toml``.

    The contract runs the CLI from the project root; locating the marker makes
    that requirement checkable instead of assumed.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def project_relative(path: Path, project_root: Path) -> str:
    """Return ``path`` relative to ``project_root`` using ``/`` separators."""
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def resolve_within_root(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root`` and refuse anything that escapes it.

    Uses the *resolved* path so symlinks / reparse points that leave the
    declared root are rejected rather than silently followed.
    """
    root_resolved = root.resolve()
    candidate = (root_resolved / rel).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"path escapes the declared root: {rel!r}")
    return candidate


def canonical_json(value: Any) -> str:
    """Canonical JSON: sorted keys, no extra whitespace, ``ensure_ascii=False``."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sort_key_for_path_field(raw_value: Any, canonical: str | None) -> str:
    """Deterministic sort key for an ``image`` / ``mask`` manifest field.

    Exactly the four cases of contract clause 2, returning a single string so
    that ordering is a plain Unicode code-point comparison.
    """
    if canonical is not None:
        return canonical
    if raw_value is None:
        return ""
    if isinstance(raw_value, str):
        return raw_value
    return canonical_json(raw_value)
