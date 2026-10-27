"""Canonical manifest parsing and validation (``photo-io-v1`` §2).

The contract separates two severities and this module keeps them apart:

* a **run-level** problem (unreadable file, bad JSON, missing/invalid top-level
  field, empty ``frames``) means nothing at all can be processed.  That becomes
  ``INPUT_ERROR`` with zero counts;
* a **row-level** problem (a malformed row object, a row missing ``image`` or
  ``mask``, an illegal path) must still yield exactly one output record for that
  input line, because the contract forbids dropping rows.

Only *explicit* indexing is supported.  Nothing here enumerates directories,
guesses a mask filename, or auto-pairs by name: a file that is not listed simply
does not exist as far as this module is concerned.

Row ordering
------------
``parse_manifest`` returns rows already sorted by
``(image_sort_key, mask_sort_key, input_index)`` as clause 2 requires, but keeps
each row's original ``input_index``.  Sorting therefore never renumbers inputs,
and output filenames stay tied to the original line number.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ERROR_CODES, Issue
from .hashing import CONTRACT_VERSION, sha256_bytes
from .paths import is_valid_dataset_id, normalize_rel_path, sort_key_for_path_field

__all__ = [
    "DATA_KINDS",
    "Manifest",
    "ManifestError",
    "ManifestLoad",
    "ManifestRow",
    "REAL_DATASET_ROOT_PREFIX",
    "REQUIRED_TOP_LEVEL_FIELDS",
    "SYNTHETIC_DATASET_ROOT_PREFIX",
    "expected_dataset_root",
    "load_manifest",
    "parse_manifest",
]

#: ``data_kind`` values the contract permits.
DATA_KINDS = ("synthetic", "real")

#: Contract clause 2 pins the real input root literally.
REAL_DATASET_ROOT_PREFIX = "data/real/"

#: ...and pins where synthetic fixtures live, which is what stops a fixture from
#: being passed off as real data by pointing ``dataset_root`` at ``data/real/``.
SYNTHETIC_DATASET_ROOT_PREFIX = "outputs/delegation/T01A/fixtures/"

#: Top-level fields that must be present (clause 2).
REQUIRED_TOP_LEVEL_FIELDS = (
    "contract_version",
    "dataset_id",
    "data_kind",
    "dataset_root",
    "frames",
)


class ManifestError(Exception):
    """A run-level failure: no row can be processed.

    ``manifest_sha256`` is carried along so the run record can still report the
    digest of the bytes that *were* readable — the contract reserves ``null``
    for "the manifest could not be read at all".
    """

    def __init__(self, code: str, message: str, manifest_sha256: str | None = None) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.manifest_sha256 = manifest_sha256

    @property
    def issue(self) -> Issue:
        return Issue(self.code, self.message)


@dataclass(frozen=True)
class ManifestRow:
    """One input line, preserved verbatim plus its derived ordering keys.

    ``row_error`` is a field-level pre-error detected during parsing (a
    non-object row, a missing required field, an illegal path).  The row is
    still emitted so that the one-record-per-input-line rule holds; the runner
    turns it into an error record without touching the filesystem.
    """

    input_index: int
    raw_image: Any
    raw_mask: Any
    image_key: str
    mask_key: str
    image_canonical: str | None
    mask_canonical: str | None
    row_error: Issue | None

    @property
    def sort_key(self) -> tuple[str, str, int]:
        return (self.image_key, self.mask_key, self.input_index)


@dataclass(frozen=True)
class Manifest:
    """A validated manifest whose rows are already in contract order."""

    contract_version: str
    dataset_id: str
    data_kind: str
    dataset_root: str
    rows: tuple[ManifestRow, ...]


@dataclass(frozen=True)
class ManifestLoad:
    """Result of reading a manifest without raising.

    Exactly one of ``manifest`` / ``error`` is set.  ``manifest_sha256`` is
    ``None`` only when the file could not be read.
    """

    manifest_sha256: str | None
    manifest: Manifest | None
    error: Issue | None

    @property
    def ok(self) -> bool:
        return self.manifest is not None


def expected_dataset_root(data_kind: str, dataset_id: str) -> str:
    """The contract-mandated ``dataset_root`` for a dataset.

    Clause 2 pins both layouts.  Enforcing the equality is what turns "a
    synthetic fixture must not impersonate real data" into a checkable property
    instead of a convention, and it keeps every run's inputs at a predictable
    location.
    """
    if data_kind == "real":
        return f"{REAL_DATASET_ROOT_PREFIX}{dataset_id}"
    return f"{SYNTHETIC_DATASET_ROOT_PREFIX}{dataset_id}"


class _NonStandardConstant(ValueError):
    """Raised while parsing ``NaN`` / ``Infinity`` / ``-Infinity``."""


def _reject_constant(name: str) -> Any:
    """``json.loads`` hook that refuses the non-standard numeric constants.

    Python's ``json`` accepts these by default even though RFC 8259 does not
    define them.  A manifest containing one is not a manifest this contract
    describes, so it is rejected as a whole rather than special-cased per field.
    """
    raise _NonStandardConstant(name)


def _reject_non_finite(value: Any, digest: str, path: str = "$") -> None:
    """Refuse any non-finite float anywhere in the parsed document.

    ``parse_constant`` only sees the literal tokens ``NaN`` and ``Infinity``.  A
    *finite* literal that overflows a double — ``1e400`` — parses to ``inf``
    without that hook ever being called, so the document is walked as well.  Both
    routes are needed: neither alone is sufficient.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError("INVALID_MANIFEST", f"non-finite number at {path}", digest)
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, digest, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, digest, f"{path}[{index}]")


def parse_manifest(payload: bytes, expected_data_kind: str) -> Manifest:
    """Validate ``payload`` and return a :class:`Manifest`.

    Raises :class:`ManifestError` for every run-level problem.
    """
    digest = sha256_bytes(payload)

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError("INVALID_MANIFEST", f"manifest is not valid UTF-8: {exc}", digest) from exc

    try:
        document = json.loads(text, parse_constant=_reject_constant)
    except _NonStandardConstant as exc:
        raise ManifestError(
            "INVALID_MANIFEST",
            f"non-standard JSON constant {exc.args[0]!r} is not allowed; "
            "use a finite number or a string",
            digest,
        ) from exc
    except json.JSONDecodeError as exc:
        raise ManifestError("INVALID_MANIFEST", f"manifest is not valid JSON: {exc.msg}", digest) from exc

    _reject_non_finite(document, digest)

    if not isinstance(document, dict):
        raise ManifestError(
            "INVALID_MANIFEST",
            f"manifest top level must be a JSON object, got {type(document).__name__}",
            digest,
        )

    missing = [name for name in REQUIRED_TOP_LEVEL_FIELDS if name not in document]
    if missing:
        raise ManifestError(
            "INVALID_MANIFEST",
            f"manifest is missing required top-level field(s): {', '.join(missing)}",
            digest,
        )

    contract_version = document["contract_version"]
    if contract_version != CONTRACT_VERSION:
        raise ManifestError(
            "INVALID_MANIFEST",
            f"contract_version must be {CONTRACT_VERSION!r}, got {contract_version!r}",
            digest,
        )

    dataset_id = document["dataset_id"]
    if not is_valid_dataset_id(dataset_id):
        raise ManifestError(
            "INVALID_MANIFEST",
            "dataset_id must match [A-Za-z0-9][A-Za-z0-9_-]* (case sensitive)",
            digest,
        )

    data_kind = document["data_kind"]
    if data_kind not in DATA_KINDS:
        raise ManifestError(
            "INVALID_MANIFEST",
            f"data_kind must be one of {DATA_KINDS}, got {data_kind!r}",
            digest,
        )
    if data_kind != expected_data_kind:
        raise ManifestError(
            "INVALID_MANIFEST",
            f"manifest data_kind {data_kind!r} does not match the CLI argument {expected_data_kind!r}",
            digest,
        )

    dataset_root = normalize_rel_path(document["dataset_root"])
    if dataset_root is None:
        raise ManifestError(
            "INVALID_MANIFEST",
            "dataset_root must be a legal project-relative path",
            digest,
        )

    mandated_root = expected_dataset_root(data_kind, dataset_id)
    if dataset_root != mandated_root:
        raise ManifestError(
            "INVALID_MANIFEST",
            f"dataset_root must be {mandated_root!r} for data_kind {data_kind!r}, got {dataset_root!r}",
            digest,
        )

    frames = document["frames"]
    if not isinstance(frames, list) or not frames:
        raise ManifestError("INVALID_MANIFEST", "frames must be a non-empty JSON array", digest)

    rows = tuple(_parse_row(index, raw) for index, raw in enumerate(frames))
    rows = tuple(sorted(rows, key=lambda row: row.sort_key))

    return Manifest(
        contract_version=contract_version,
        dataset_id=dataset_id,
        data_kind=data_kind,
        dataset_root=dataset_root,
        rows=rows,
    )


def load_manifest(manifest_path: Path, expected_data_kind: str) -> ManifestLoad:
    """Read and validate ``manifest_path``, converting failures to an issue.

    Never raises for a bad manifest: the caller has to write a run record either
    way, so the failure is returned as data.
    """
    try:
        payload = manifest_path.read_bytes()
    except FileNotFoundError:
        return ManifestLoad(None, None, Issue("MISSING_FILE", "input manifest does not exist"))
    except OSError as exc:
        return ManifestLoad(None, None, Issue("MISSING_FILE", f"input manifest is unreadable: {exc.strerror}"))

    digest = sha256_bytes(payload)
    try:
        manifest = parse_manifest(payload, expected_data_kind)
    except ManifestError as exc:
        return ManifestLoad(digest, None, exc.issue)
    return ManifestLoad(digest, manifest, None)


def _parse_row(index: int, raw: Any) -> ManifestRow:
    """Build a :class:`ManifestRow`, recording a field error instead of raising.

    Whatever *is* reliably obtainable is kept even when the row is rejected: a
    row missing ``mask`` still has a usable ``image`` path, and the contract asks
    error records to preserve known metadata rather than blank it out.
    """
    if not isinstance(raw, dict):
        # A non-object row has no ``image`` and no ``mask`` *fields*, so both
        # ordering keys are the empty string.  Serialising the whole row into the
        # image key (e.g. the literal text ``null``) would be inventing an image
        # value the line never had, and would move the row in the output order.
        return ManifestRow(
            input_index=index,
            raw_image=None,
            raw_mask=None,
            image_key="",
            mask_key="",
            image_canonical=None,
            mask_canonical=None,
            row_error=Issue("INVALID_MANIFEST", f"frame row {index} must be a JSON object"),
        )

    has_image = "image" in raw
    has_mask = "mask" in raw
    raw_image = raw.get("image")
    raw_mask = raw.get("mask")

    image_canonical = normalize_rel_path(raw_image) if has_image else None
    mask_canonical = None if raw_mask is None else normalize_rel_path(raw_mask)

    row_error: Issue | None = None
    if not has_image or not has_mask:
        absent = [name for name, present in (("image", has_image), ("mask", has_mask)) if not present]
        row_error = Issue(
            "INVALID_MANIFEST",
            f"frame row {index} is missing required field(s): {', '.join(absent)}",
        )
    elif image_canonical is None:
        row_error = Issue("INVALID_PATH", f"frame row {index}: image is not a legal dataset-relative path")
    elif raw_mask is not None and mask_canonical is None:
        row_error = Issue("INVALID_PATH", f"frame row {index}: mask is not a legal dataset-relative path")

    return ManifestRow(
        input_index=index,
        raw_image=raw_image,
        raw_mask=raw_mask,
        image_key=sort_key_for_path_field(raw_image, image_canonical),
        mask_key=sort_key_for_path_field(raw_mask, mask_canonical),
        image_canonical=image_canonical,
        mask_canonical=mask_canonical,
        row_error=row_error,
    )
