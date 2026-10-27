"""Per-row orchestration and run artifacts (``photo-io-v1`` §2, §5, §6).

Three contract properties drive the shape of this module.

**One record per input line.**  Every manifest row produces exactly one
``frames.jsonl`` entry — including rows that are structurally broken.  Dropping
a row would make ``input_count`` and the record count disagree, and the
disagreement would be invisible to anyone reading only the summary, so rows are
never skipped and never merged.

**Failures are isolated and never inherit a previous success.**  Output paths are
only written into a record *after* the corresponding file was really written in
this run, and an error row explicitly clears both output fields.  A fresh run
directory is used every time, so there is no path by which an error row can point
at an earlier run's artifact.

**Identity, invalidation and duplication are three different things.**  A row can
have the same ``frame_id`` as another row (same dataset, path and image bytes) and
still be a separate input that must be processed independently — including its own
mask.  Duplication is *reported* through ``duplicate_of_input_index`` and never
used to skip work.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import FrameIssue, Issue, as_json
from .hashing import (
    CONTRACT_VERSION,
    compute_cache_fingerprint,
    compute_frame_id,
    runtime_info,
    sha256_bytes,
    sha256_file,
    source_tree_sha256,
)
from .images import decode_image, encode_rgb_png
from .manifest import (
    REAL_DATASET_ROOT_PREFIX,
    SYNTHETIC_DATASET_ROOT_PREFIX,
    ManifestLoad,
    ManifestRow,
    expected_dataset_root,
)
from .orientation import apply_orientation, canonical_size, orientation_matrix
from .paths import resolve_within_root

__all__ = [
    "REAL_DATASET_ROOT_PREFIX",
    "SYNTHETIC_DATASET_ROOT_PREFIX",
    "RunResult",
    "dumps_record",
    "execute_run",
    "expected_dataset_root",
]

#: Field order of a ``frames.jsonl`` record; fixed so the file is diffable.
_FRAME_FIELDS = (
    "input_index",
    "input_image",
    "input_mask",
    "frame_id",
    "image_path",
    "mask_path",
    "image_sha256",
    "mask_sha256",
    "duplicate_of_input_index",
    "status",
    "raw_width",
    "raw_height",
    "canonical_width",
    "canonical_height",
    "exif_orientation_original",
    "orientation_applied",
    "raw_to_canonical",
    "image_dtype",
    "image_mode",
    "mask_encoding",
    "mask_foreground_pixels",
    "image_output",
    "mask_output",
    "cache_fingerprint",
    "cache_hit",
    "camera",
    "distortion",
    "pose",
    "geometry_scale",
    "warnings",
    "errors",
)


@dataclass(frozen=True)
class RunResult:
    """Outcome of one CLI run."""

    run_id: str
    run_dir: Path
    status: str
    input_count: int
    ok_count: int
    error_count: int
    warning_count: int
    exit_code: int


@dataclass(frozen=True)
class _RunContext:
    dataset_id: str
    data_kind: str
    dataset_root: Path
    run_dir: Path
    implementation_sha256: str
    runtime: dict[str, str]


def expected_dataset_root(data_kind: str, dataset_id: str) -> str:
    """The contract-mandated ``dataset_root`` for a dataset.

    Clause 2 pins both layouts, which is what makes "a synthetic fixture cannot
    be passed off as real data" checkable rather than a matter of trust.
    """
    if data_kind == "real":
        return f"{REAL_DATASET_ROOT_PREFIX}{dataset_id}"
    return f"{SYNTHETIC_DATASET_ROOT_PREFIX}{dataset_id}"


def execute_run(
    manifest_load: ManifestLoad,
    *,
    output_base: Path,
    data_kind: str,
    project_root: Path,
) -> RunResult:
    """Create a fresh run directory, process every row, and write the artifacts.

    Raises ``OSError`` if the run directory cannot be created — the caller
    reports that on stderr and exits non-zero rather than pretending evidence
    was saved.
    """
    run_id = _make_run_id()
    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    implementation_sha256 = source_tree_sha256(project_root)
    runtime = runtime_info()

    if not manifest_load.ok:
        assert manifest_load.error is not None  # guaranteed by ManifestLoad
        return _write_run_level_failure(
            run_dir_path=run_dir,
            run_id=run_id,
            data_kind=data_kind,
            manifest_sha256=manifest_load.manifest_sha256,
            error=manifest_load.error,
            implementation_sha256=implementation_sha256,
            runtime=runtime,
        )

    manifest = manifest_load.manifest
    assert manifest is not None  # guaranteed by ManifestLoad.ok
    context = _RunContext(
        dataset_id=manifest.dataset_id,
        data_kind=manifest.data_kind,
        dataset_root=(project_root / manifest.dataset_root).resolve(),
        run_dir=run_dir,
        implementation_sha256=implementation_sha256,
        runtime=runtime,
    )

    duplicates = _DuplicateIndex()
    records = [process_frame(row, context, duplicates) for row in manifest.rows]

    ok_count = sum(1 for record in records if record["status"] == "OK")
    error_count = len(records) - ok_count
    warning_count = sum(len(record["warnings"]) for record in records)
    status = "SUCCESS" if error_count == 0 else "PARTIAL_FAILURE"

    _write_frames(run_dir / "frames.jsonl", records)
    _write_run_record(
        run_dir / "run.json",
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "dataset_id": manifest.dataset_id,
            "data_kind": manifest.data_kind,
            "manifest_sha256": manifest_load.manifest_sha256,
            "status": status,
            "input_count": len(records),
            "ok_count": ok_count,
            "error_count": error_count,
            "warning_count": warning_count,
            "runtime": runtime,
            "implementation_sha256": implementation_sha256,
            "errors": [],
        },
    )

    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        input_count=len(records),
        ok_count=ok_count,
        error_count=error_count,
        warning_count=warning_count,
        exit_code=0 if error_count == 0 else 1,
    )


def process_frame(row: ManifestRow, context: _RunContext, duplicates: "_DuplicateIndex") -> dict[str, Any]:
    """Produce the ``frames.jsonl`` record for one manifest row.

    Never raises: a row-level failure becomes an ``ERROR`` record, because the
    one-record-per-input-line rule outranks any individual failure.
    """
    record = _blank_record(row)
    try:
        if row.row_error is not None:
            raise FrameIssue(row.row_error.code, row.row_error.message)
        _process_row(record, row, context, duplicates)
    except FrameIssue as exc:
        record["errors"] = [as_json(exc.issue)]
        _clear_outputs(record)
        record["status"] = "ERROR"
    except Exception as exc:  # noqa: BLE001 - contract: never rewrite into success
        record["errors"] = [
            as_json(Issue("INTERNAL_ERROR", f"unexpected {type(exc).__name__} while processing row"))
        ]
        _clear_outputs(record)
        record["status"] = "ERROR"
    return record


def _process_row(
    record: dict[str, Any],
    row: ManifestRow,
    context: _RunContext,
    duplicates: "_DuplicateIndex",
) -> None:
    """Fill ``record`` for a row that passed parsing. Raises on any failure."""
    # ---------------------------------------------------------------- image --
    image_file = _resolve(context.dataset_root, row.image_canonical, "image")
    if not image_file.is_file():
        raise FrameIssue("MISSING_FILE", "image file does not exist")

    image_payload = _read_bytes(image_file, "image")
    image_digest = sha256_bytes(image_payload)
    record["image_sha256"] = image_digest
    record["frame_id"] = compute_frame_id(context.dataset_id, row.image_canonical, image_digest)
    # Duplication is a report, not a shortcut: the row is still processed below.
    record["duplicate_of_input_index"] = duplicates.register(image_digest, row.input_index)

    decoded = decode_image(image_payload)
    orientation = decoded.exif_orientation_original
    applied = 1 if orientation is None else orientation

    record["raw_width"] = decoded.raw_width
    record["raw_height"] = decoded.raw_height
    record["exif_orientation_original"] = orientation
    record["orientation_applied"] = applied
    record["raw_to_canonical"] = orientation_matrix(applied, decoded.raw_width, decoded.raw_height)
    canonical_width, canonical_height = canonical_size(applied, decoded.raw_width, decoded.raw_height)
    record["canonical_width"] = canonical_width
    record["canonical_height"] = canonical_height
    record["image_dtype"] = "uint8"
    record["image_mode"] = "RGB"

    canonical_rgb = apply_orientation(decoded.rgb, applied)

    # ----------------------------------------------------------------- mask --
    mask_bytes: bytes | None = None
    mask_digest: str | None = None
    if row.raw_mask is not None:
        mask_file = _resolve(context.dataset_root, row.mask_canonical, "mask")
        if not mask_file.is_file():
            raise FrameIssue("MISSING_FILE", "mask file does not exist")

        mask_payload = _read_bytes(mask_file, "mask")
        mask_digest = sha256_bytes(mask_payload)
        record["mask_sha256"] = mask_digest

        decoded_mask = _decode_mask(mask_payload)
        if (decoded_mask.raw_width, decoded_mask.raw_height) != (decoded.raw_width, decoded.raw_height):
            # Compared against the *raw* photo size: a mask is drawn in raw pixel
            # coordinates and is never pre-rotated.
            raise FrameIssue(
                "MASK_SIZE_MISMATCH",
                "mask raw size "
                f"{decoded_mask.raw_width}x{decoded_mask.raw_height} does not match "
                f"photo raw size {decoded.raw_width}x{decoded.raw_height}",
            )

        record["mask_encoding"] = decoded_mask.encoding
        record["mask_foreground_pixels"] = decoded_mask.foreground_pixels
        if decoded_mask.is_empty:
            record["warnings"].append(as_json(Issue("EMPTY_MASK", "mask contains no foreground pixel")))

        mask_bytes = _encode_mask(decoded_mask, applied)

    # ------------------------------------------------------ input stability --
    # Re-read and re-hash so a file edited mid-run is detected rather than being
    # silently reported with the hash of its earlier contents.
    _verify_unchanged(image_file, image_digest)
    if row.raw_mask is not None and mask_digest is not None:
        _verify_unchanged(_resolve(context.dataset_root, row.mask_canonical, "mask"), mask_digest)

    # --------------------------------------------------------------- output --
    image_output, mask_output = _write_pair(
        context.run_dir, row.input_index, encode_rgb_png(canonical_rgb), mask_bytes
    )
    record["image_output"] = image_output
    record["mask_output"] = mask_output

    record["cache_fingerprint"] = compute_cache_fingerprint(
        dataset_id=context.dataset_id,
        data_kind=context.data_kind,
        image_path=row.image_canonical,
        mask_path=row.mask_canonical,
        image_sha256=image_digest,
        mask_sha256=mask_digest,
        implementation_sha256=context.implementation_sha256,
        runtime=context.runtime,
    )
    # v1 does not reuse cached results; the field exists so a later version can.
    record["cache_hit"] = False
    record["status"] = "OK"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class _DuplicateIndex:
    """First-seen ``input_index`` per image digest, in contract sort order."""

    def __init__(self) -> None:
        self._first: dict[str, int] = {}

    def register(self, digest: str, input_index: int) -> int | None:
        if digest not in self._first:
            self._first[digest] = input_index
            return None
        return self._first[digest]


def _blank_record(row: ManifestRow) -> dict[str, Any]:
    """An all-``null`` record carrying only what parsing already established."""
    record: dict[str, Any] = dict.fromkeys(_FRAME_FIELDS)
    record.update(
        {
            "input_index": row.input_index,
            "input_image": row.raw_image,
            "input_mask": row.raw_mask,
            "image_path": row.image_canonical,
            "mask_path": row.mask_canonical,
            "status": "ERROR",
            "cache_hit": False,
            "camera": None,
            "distortion": None,
            "pose": None,
            "geometry_scale": None,
            "warnings": [],
            "errors": [],
        }
    )
    return record


def _clear_outputs(record: dict[str, Any]) -> None:
    """Ensure an error row cannot point at a result file."""
    record["image_output"] = None
    record["mask_output"] = None
    record["cache_fingerprint"] = None


def _resolve(dataset_root: Path, rel: str | None, kind: str) -> Path:
    """Resolve a dataset-relative path, refusing anything that escapes the root."""
    assert rel is not None  # callers only pass canonical paths
    try:
        return resolve_within_root(dataset_root, rel)
    except ValueError as exc:
        raise FrameIssue("INVALID_PATH", f"{kind} path escapes the dataset root") from exc


def _read_bytes(path: Path, kind: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FrameIssue("MISSING_FILE", f"{kind} file could not be read: {exc.strerror}") from exc


def _verify_unchanged(path: Path, expected_sha256: str) -> None:
    """Confirm the file still holds the bytes that were hashed and decoded."""
    try:
        current = sha256_file(path)
    except OSError as exc:
        raise FrameIssue("INPUT_CHANGED", f"input became unreadable during the run: {exc.strerror}") from exc
    if current != expected_sha256:
        raise FrameIssue("INPUT_CHANGED", "input bytes changed while the run was in progress")


def _decode_mask(payload: bytes):
    from ..segmentation.manual import decode_manual_mask

    return decode_manual_mask(payload)


def _encode_mask(decoded_mask: Any, orientation: int) -> bytes:
    from ..segmentation.manual import canonical_mask_array, encode_mask_png

    return encode_mask_png(canonical_mask_array(decoded_mask, orientation))


def _write_pair(
    run_dir: Path,
    input_index: int,
    image_bytes: bytes,
    mask_bytes: bytes | None,
) -> tuple[str, str | None]:
    """Write ``images/`` and optional ``masks/`` artifacts for one row.

    Written paths are returned only after both writes succeeded; a partial pair
    is cleaned up so the run directory never contains a stray half-result that no
    record refers to.
    """
    stem = f"{input_index:06d}"
    image_rel = f"images/{stem}.png"
    mask_rel = f"masks/{stem}.png" if mask_bytes is not None else None

    written: list[Path] = []
    try:
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_file = image_dir / f"{stem}.png"
        image_file.write_bytes(image_bytes)
        written.append(image_file)

        if mask_bytes is not None:
            mask_dir = run_dir / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_file = mask_dir / f"{stem}.png"
            mask_file.write_bytes(mask_bytes)
            written.append(mask_file)
    except OSError as exc:
        for path in written:
            try:
                path.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
        raise FrameIssue("OUTPUT_ERROR", f"cannot write run artifact: {exc.strerror}") from exc

    return image_rel, mask_rel


def _make_run_id() -> str:
    """UTC timestamp plus a random suffix, so no historical directory is reused."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:12]}"


def _write_run_level_failure(
    *,
    run_dir_path: Path,
    run_id: str,
    data_kind: str,
    manifest_sha256: str | None,
    error: Issue,
    implementation_sha256: str,
    runtime: dict[str, str],
) -> RunResult:
    """Write an ``INPUT_ERROR`` run record with zero counts."""
    _write_frames(run_dir_path / "frames.jsonl", [])
    _write_run_record(
        run_dir_path / "run.json",
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "dataset_id": None,
            "data_kind": data_kind,
            "manifest_sha256": manifest_sha256,
            "status": "INPUT_ERROR",
            "input_count": 0,
            "ok_count": 0,
            "error_count": 0,
            "warning_count": 0,
            "runtime": runtime,
            "implementation_sha256": implementation_sha256,
            "errors": [as_json(error)],
        },
    )
    return RunResult(
        run_id=run_id,
        run_dir=run_dir_path,
        status="INPUT_ERROR",
        input_count=0,
        ok_count=0,
        error_count=0,
        warning_count=0,
        exit_code=2,
    )


def dumps_record(payload: Any, *, indent: int | None = None) -> str:
    """Serialise a record so that it can always be written as UTF-8.

    A manifest may legitimately contain a value that Python holds but UTF-8
    cannot encode — most sharply an unpaired surrogate in a path, which is
    preserved in ``input_image`` for audit precisely because the path was
    rejected.  ``ensure_ascii=False`` cannot represent that character, so such a
    record falls back to ``ensure_ascii=True``, where the surrogate survives as
    the JSON escape ``\\udXXX``: the original value is kept exactly, as text that
    any JSON reader decodes back to the same string.

    The alternative behaviours are both worse: dropping the row would break the
    one-record-per-input-line rule, and rewriting the path would falsify the
    record being kept for audit.

    ``allow_nan=False`` is set on both routes so a non-standard constant can
    never reach an artifact.  With the manifest rejecting them at parse time this
    should be unreachable; it is here so that "should be" cannot quietly become
    "was not".
    """
    kwargs = {"separators": (",", ":"), "allow_nan": False}
    if indent is not None:
        kwargs = {"indent": indent, "allow_nan": False}

    text = json.dumps(payload, ensure_ascii=False, **kwargs)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(payload, ensure_ascii=True, **kwargs)
    return text


def _write_frames(path: Path, records: list[dict[str, Any]]) -> None:
    body = "".join(dumps_record(record) + "\n" for record in records)
    path.write_bytes(body.encode("utf-8"))


def _write_run_record(path: Path, record: dict[str, Any]) -> None:
    path.write_bytes((dumps_record(record, indent=2) + "\n").encode("utf-8"))
