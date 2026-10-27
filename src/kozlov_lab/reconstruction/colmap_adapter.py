"""T01 canonical input -> native sparse reconstruction (``sfm-native-v1``).

This module owns the input side of T02A and the boundary to the native backend:

* **reading a T01 run** — only a ``photo-io-v1`` run with ``status=SUCCESS`` and
  consistent counts is accepted.  A single ``ERROR`` row rejects the whole run
  rather than being skipped, because silently reconstructing from a subset would
  produce a model whose scope nobody declared;
* **staging** — canonical PNG bytes are re-hashed here.  The upstream
  ``image_sha256`` is the hash of the *original* input file, so reusing it as a
  canonical-image hash would compare the wrong thing; the digest is recomputed
  from the canonical bytes and both are kept;
* **deduplication** — byte-identical canonical images collapse to one staged
  file (representative = smallest ``input_index``), but every upstream line keeps
  its own mapping record.  Identical bytes are the *same observation*, not two
  viewpoints, so feeding both to SfM would fake extra parallax;
* **the native call** — every stage runs in its own subprocess.  A native
  exception, abort or timeout then costs one stage's evidence instead of the whole
  run's, and the process boundary is what makes a real exit code possible: an
  in-process API call has no exit code and this module must never invent one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image

from . import profiles
from ..io.errors import FrameIssue
from ..io.source_encoding import assert_eight_bit_source

CONTRACT_VERSION = "sfm-native-v1"
UPSTREAM_CONTRACT = "photo-io-v1"
COORDINATE_CONVENTION = "colmap_native_pixel_center_0.5_world_to_camera"

#: Fixed first-version engineering profile (contract §3).  These are the values
#: the contract freezes; everything else stays at the backend default and is
#: recorded in ``effective_options``.
MAX_DISTINCT_IMAGES = 12
MAX_LONG_SIDE = 1024
NUM_THREADS = 2
MAX_SIFT_FEATURES = 4096
RANDOM_SEED = 0
CAMERA_MODEL = "SIMPLE_RADIAL"
FOCAL_LENGTH_FACTOR = 1.2

#: The contract defines a *usable* model as >= 2 registered images and >= 1
#: point.  The mapper's own default ``min_model_size`` is larger, which would
#: discard a model the contract calls usable and report NO_MODEL for a model that
#: existed -- a false negative produced by a measurement setting.  It is
#: therefore lowered to the contract's own floor and recorded as an effective
#: option; ``multiple_models`` is on so that every component is preserved rather
#: than only the first.
MIN_MODEL_SIZE = 2
MULTIPLE_MODELS = True

#: Structural floor for a *usable* component (contract §6): a well-formed model
#: with at least two registered images and at least one 3D point.  This says
#: nothing about accuracy or object completeness.
MODEL_NOT_USABLE_MIN_IMAGES = 2
MODEL_NOT_USABLE_MIN_POINTS = 1

BACKEND_NAME = "pycolmap"
BACKEND_PIN = "4.2.1"

STAGES = ("extract_features", "match_exhaustive", "incremental_mapping", "export_models")

NATIVE_TIMEOUT_SECONDS = 600

#: How long the *backend probe's own* terminated child is given to be reaped so its
#: real exit status can be recorded.  Bounded on purpose: an unbounded second wait
#: would let the probe escape the shared deadline that bounds the native chain.
#:
#: Scope: this constant applies to the ``backend_info`` probe, which holds the
#: ``Popen`` handle itself and therefore can bound its own reap.  It is **not**
#: applied by the stage executor, which calls ``subprocess.run`` and has no handle of
#: its own to wait on.  A whole process tree and the final wall-clock bound of the
#: chain are the outer job supervisor's responsibility, not this module's.
_TERMINATION_GRACE_SECONDS = 30.0

#: The single, explicit monotonic bound shared by *all* steps of the backend probe's
#: cleanup (kill, wait, grace reap).  One deadline for the whole cleanup rather than
#: a fresh 30 s per step, so a stuck probe cannot stretch cleanup without limit.
#: Kept well below the 30 s grace so the two never disagree about which is tighter.
_PROBE_CLEANUP_DEADLINE_SECONDS = 20.0


class InputError(Exception):
    """Upstream input, parameters, paths or counts do not conform (exit 2)."""

    code = "INPUT_ERROR"

    def __init__(self, message: str, code: str = "INPUT_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class BackendUnavailable(Exception):
    """The pinned backend cannot be loaded or lacks a required capability (exit 3)."""


class BackendError(Exception):
    """The backend failed, crashed, timed out, or a model failed to round-trip (exit 4)."""


# --------------------------------------------------------------------------- #
# canonical JSON and digests — the contract's own definitions
# --------------------------------------------------------------------------- #
def canonical_json(value: Any) -> str:
    """Sorted keys, no insignificant whitespace, ``ensure_ascii=False``."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def dumps_strict(payload: Any, *, indent: int | None = None) -> str:
    """Strict JSON: no ``NaN``/``Infinity`` may ever reach an artifact."""
    if indent is None:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False)


def implementation_sha256(project_root: pathlib.Path) -> str:
    """Digest over ``reconstruction/`` Python files, sorted by relative path."""
    package = project_root / "src" / "kozlov_lab" / "reconstruction"
    pairs = [
        [path.relative_to(project_root).as_posix(), sha256_file(path)]
        for path in sorted(package.glob("*.py"))
    ]
    return canonical_digest(pairs)


# --------------------------------------------------------------------------- #
# upstream reading
# --------------------------------------------------------------------------- #
@dataclass
class UpstreamRow:
    input_index: int
    frame_id: str
    image_output: str
    mask_output: str | None
    upstream_image_sha256: str | None
    canonical_width: int
    canonical_height: int
    #: filled during staging
    canonical_image_sha256: str | None = None
    canonical_mask_sha256: str | None = None
    representative_input_index: int | None = None
    staged_image_name: str | None = None


@dataclass
class UpstreamRun:
    run_dir: pathlib.Path
    run_id: str
    data_kind: str
    contract_version: str
    status: str
    rows: list[UpstreamRow] = field(default_factory=list)
    run_json_sha256: str = ""
    frames_jsonl_sha256: str = ""
    run_json: dict = field(default_factory=dict)


def load_upstream_run(run_dir: pathlib.Path) -> UpstreamRun:
    """Read and validate a T01 run directory.

    Rejects the run as a whole on any contract violation: a partially-consumed
    input would make ``data_kind`` and the identity map describe something other
    than what was actually reconstructed.
    """
    run_dir = pathlib.Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise InputError(f"--io-run is not a directory: {run_dir}")

    run_json_path = run_dir / "run.json"
    frames_path = run_dir / "frames.jsonl"
    for path in (run_json_path, frames_path):
        if not path.is_file():
            raise InputError(f"upstream run is missing {path.name}")

    raw_run = run_json_path.read_bytes()
    raw_frames = frames_path.read_bytes()
    try:
        run_json = json.loads(raw_run.decode("utf-8"), parse_constant=_reject_constant)
    except ValueError as exc:
        raise InputError(f"upstream run.json is not usable strict JSON: {exc}") from exc

    if not isinstance(run_json, dict):
        raise InputError("upstream run.json must be a JSON object")

    contract = run_json.get("contract_version")
    if contract != UPSTREAM_CONTRACT:
        raise InputError(
            f"upstream contract_version must be {UPSTREAM_CONTRACT!r}, got {contract!r}"
        )
    status = run_json.get("status")
    if status != "SUCCESS":
        raise InputError(f"upstream status must be SUCCESS, got {status!r}")
    data_kind = run_json.get("data_kind")
    if data_kind not in ("synthetic", "real"):
        raise InputError(f"upstream data_kind is not usable: {data_kind!r}")

    rows: list[UpstreamRow] = []
    seen_indices: set[int] = set()
    for line_number, line in enumerate(raw_frames.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except ValueError as exc:
            raise InputError(f"frames.jsonl line {line_number} is not usable JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise InputError(f"frames.jsonl line {line_number} is not a JSON object")

        index = record.get("input_index")
        if not isinstance(index, int):
            raise InputError(f"frames.jsonl line {line_number} has no integer input_index")
        if index in seen_indices:
            raise InputError(f"duplicate input_index {index} in frames.jsonl")
        seen_indices.add(index)

        if record.get("status") == "ERROR":
            raise InputError(
                f"upstream row {index} is an ERROR row; the whole run is rejected rather than "
                "reconstructed from a subset"
            )
        if record.get("status") != "OK":
            raise InputError(f"upstream row {index} has status {record.get('status')!r}, expected OK")

        frame_id = record.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise InputError(f"upstream row {index} has an empty frame_id")

        image_output = record.get("image_output")
        if not isinstance(image_output, str) or not image_output:
            raise InputError(f"upstream row {index} has no image_output")

        width, height = record.get("canonical_width"), record.get("canonical_height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise InputError(f"upstream row {index} has no usable canonical dimensions")

        rows.append(
            UpstreamRow(
                input_index=index,
                frame_id=frame_id,
                image_output=image_output,
                mask_output=record.get("mask_output"),
                upstream_image_sha256=record.get("image_sha256"),
                canonical_width=width,
                canonical_height=height,
            )
        )

    if not rows:
        raise InputError("upstream frames.jsonl contains no rows")

    # Counters must be integers and must agree with the rows.  Checking only
    # input_count let a file declaring ok_count=0 / error_count=3 alongside three
    # OK rows be accepted, which is not a SUCCESS run by any reading.
    def require_int(name: str) -> int:
        value = run_json.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise InputError(
                f"upstream run.json field {name!r} must be an integer, got {value!r}"
            )
        return value

    declared_inputs = require_int("input_count")
    declared_ok = require_int("ok_count")
    declared_errors = require_int("error_count")

    if declared_inputs != len(rows):
        raise InputError(
            f"run.json input_count={declared_inputs} but frames.jsonl has {len(rows)} rows"
        )
    ok_rows = sum(1 for _ in rows)
    if declared_ok != ok_rows:
        raise InputError(
            f"run.json ok_count={declared_ok} but {ok_rows} rows have status OK"
        )
    if declared_errors != 0:
        raise InputError(
            f"run.json error_count={declared_errors}; a SUCCESS run must declare no error rows"
        )
    if declared_ok + declared_errors != declared_inputs:
        raise InputError(
            f"run.json counters are inconsistent: ok_count({declared_ok}) + "
            f"error_count({declared_errors}) != input_count({declared_inputs})"
        )

    rows.sort(key=lambda row: row.input_index)
    return UpstreamRun(
        run_dir=run_dir,
        run_id=str(run_json.get("run_id")),
        data_kind=data_kind,
        contract_version=contract,
        status=status,
        rows=rows,
        run_json_sha256=sha256_bytes(raw_run),
        frames_jsonl_sha256=sha256_bytes(raw_frames),
        run_json=run_json,
    )


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name!r}")


def _resolve_within(root: pathlib.Path, relative: str, what: str) -> pathlib.Path:
    """Resolve inside ``root`` and refuse anything that escapes it."""
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise InputError(f"{what} escapes the upstream run directory: {relative}")
    return candidate


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #
@dataclass
class StagedInputs:
    rows: list[UpstreamRow]
    distinct: list[UpstreamRow]          # representatives, one per distinct image
    images_dir: pathlib.Path
    masks_dir: pathlib.Path | None = None
    mask_bindings: list[dict] = field(default_factory=list)
    selected_indices: set[int] | None = None


def stage_inputs(upstream: UpstreamRun, staged_root: pathlib.Path,
                 profile: str = profiles.DEFAULT_PROFILE,
                 masks_by_index: dict[int, pathlib.Path] | None = None) -> StagedInputs:
    """Verify every canonical artifact, deduplicate by bytes, and stage copies."""
    profiles.require_valid_profile(profile)
    max_images = (profiles.MAX_OBJECT_VIDEO_IMAGES if profile == profiles.PROFILE_CAD_OBJECT_VIDEO_V1 else profiles.MAX_OBJECT_IMAGES if profile in (profiles.PROFILE_CAD_OBJECT_V1, profiles.PROFILE_CAD_OBJECT_PINHOLE_V1)
                  else MAX_DISTINCT_IMAGES)
    images_dir = staged_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for row in upstream.rows:
        image_path = _resolve_within(upstream.run_dir, row.image_output, "image_output")
        if not image_path.is_file():
            raise InputError(f"upstream image_output does not exist: {row.image_output}")
        payload = image_path.read_bytes()
        row.canonical_image_sha256 = sha256_bytes(payload)
        _check_canonical_png(payload, row, "image")

        if row.mask_output is None:
            row.canonical_mask_sha256 = None
        else:
            mask_path = _resolve_within(upstream.run_dir, row.mask_output, "mask_output")
            if not mask_path.is_file():
                raise InputError(f"upstream mask_output does not exist: {row.mask_output}")
            mask_payload = mask_path.read_bytes()
            row.canonical_mask_sha256 = sha256_bytes(mask_payload)
            _check_canonical_png(mask_payload, row, "mask", expect_mode="L")

    object_masks = {}
    if masks_by_index is not None:
        valid_indices = {r.input_index for r in upstream.rows}
        if not set(masks_by_index).issubset(valid_indices):
            raise InputError('foreground mask index not present in original upstream')
        for row in upstream.rows:
            if row.input_index not in masks_by_index:
                row.representative_input_index = None
                row.staged_image_name = None
                continue
            path = pathlib.Path(masks_by_index[row.input_index]).resolve(strict=True)
            payload = path.read_bytes()
            _check_canonical_png(payload,row,'foreground mask',expect_mode='L')
            from kozlov_lab.segmentation.manual import decode_manual_mask
            try:
                decoded = decode_manual_mask(payload)
            except FrameIssue as exc:
                raise InputError(str(exc.issue.message)) from exc
            if decoded.is_empty:
                raise InputError('selected foreground mask is empty')
            object_masks[row.input_index] = (path,payload,sha256_bytes(payload))

    by_digest: dict[Any, list[UpstreamRow]] = {}
    for row in upstream.rows:
        if masks_by_index is not None and row.input_index not in masks_by_index:
            continue
        key = row.canonical_image_sha256
        by_digest.setdefault(key, []).append(row)

    distinct: list[UpstreamRow] = []
    for digest, group in by_digest.items():
        if masks_by_index is not None and len({object_masks[r.input_index][2] for r in group}) != 1:
            raise InputError('byte-identical selected images have conflicting foreground masks')
        representative = min(group, key=lambda item: item.input_index)
        distinct.append(representative)
        for member in group:
            member.representative_input_index = representative.input_index
            member.staged_image_name = f"images/{representative.input_index:06d}.png"
    distinct.sort(key=lambda item: item.input_index)

    if len(distinct) < 2 and masks_by_index is None:
        raise InputError(
            f"only {len(distinct)} distinct canonical image(s); at least 2 are required to attempt SfM"
        )
    if len(distinct) > max_images:
        raise InputError(
            f"{len(distinct)} distinct images exceeds the fixed limit of {max_images}"
        )

    masks_dir = staged_root/'foreground_masks' if masks_by_index is not None else None
    mask_bindings = []
    for representative in distinct:
        source = _resolve_within(upstream.run_dir, representative.image_output, "image_output")
        target = images_dir / f"{representative.input_index:06d}.png"
        target.write_bytes(source.read_bytes())
        if sha256_file(target) != representative.canonical_image_sha256:
            raise InputError(f"staged copy of {representative.image_output} does not match its source")
        if masks_dir is not None:
            path,payload,digest = object_masks[representative.input_index]
            # COLMAP 4.2.1 image_reader.cc first resolves mask_root/(image_name+'.png').
            mask_target = masks_dir/(representative.staged_image_name+'.png')
            mask_target.parent.mkdir(parents=True,exist_ok=True)
            mask_target.write_bytes(payload)
            if sha256_file(mask_target)!=digest:
                raise InputError('staged foreground mask differs from source')
            mask_bindings.append(dict(input_index=representative.input_index,image_name=representative.staged_image_name,
                source_path=str(path),staged_path=str(mask_target.resolve()),sha256=digest,bytes=len(payload)))

    return StagedInputs(rows=upstream.rows, distinct=distinct, images_dir=images_dir,
        masks_dir=masks_dir,mask_bindings=mask_bindings,
        selected_indices=set(masks_by_index) if masks_by_index is not None else None)


def _check_canonical_png(payload: bytes, row: UpstreamRow, what: str,
                         expect_mode: str = "RGB") -> None:
    """The staged artifact must be an 8-bit canonical PNG of the recorded size."""
    import io as _io

    try:
        image = Image.open(_io.BytesIO(payload))
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise InputError(f"row {row.input_index}: {what} is not a decodable image: {exc}") from exc
    if (image.format or "").upper() != "PNG":
        raise InputError(f"row {row.input_index}: {what} is not a PNG ({image.format})")
    if image.mode != expect_mode:
        raise InputError(
            f"row {row.input_index}: {what} has mode {image.mode}, expected {expect_mode}"
        )
    # Decoded mode is not evidence about the source: a valid 16-bit RGB PNG opens
    # as mode ``RGB`` and its samples are already reduced.  The accepted T01
    # source-encoding reader (read-only reuse, not a reimplementation) inspects the
    # container header instead.
    try:
        depth = assert_eight_bit_source(payload, "PNG", code="UNSUPPORTED_IMAGE",
                                        subject=f"row {row.input_index} {what}")
    except FrameIssue as exc:
        raise InputError(str(exc.issue.message)) from exc
    if depth != 8:
        raise InputError(f"row {row.input_index}: {what} source depth is {depth}, expected 8")
    if image.size != (row.canonical_width, row.canonical_height):
        raise InputError(
            f"row {row.input_index}: {what} is {image.size} but the record says "
            f"({row.canonical_width}, {row.canonical_height})"
        )
    if max(image.size) > MAX_LONG_SIDE:
        raise InputError(
            f"row {row.input_index}: {what} longest side {max(image.size)} exceeds "
            f"{MAX_LONG_SIDE}; it is rejected rather than silently resized"
        )


# --------------------------------------------------------------------------- #
# effective options and fingerprints
# --------------------------------------------------------------------------- #
def effective_options(profile: str = profiles.DEFAULT_PROFILE) -> dict[str, Any]:
    """Every option this version actually applies, with its reason.

    Recorded rather than assumed: the native defaults are read back at run time
    and merged in, so a changed backend default shows up in the artifact instead
    of silently altering behaviour.

    ``profile`` only *adds* fields for the shared profile.  The default profile's
    dictionary is deliberately left exactly as it was, because the profile contract
    states that introducing the selector must not rewrite the default's recorded
    options.
    """
    options: dict[str, Any] = {
        "device": "cpu",
        "num_threads": NUM_THREADS,
        "max_distinct_images": MAX_DISTINCT_IMAGES,
        "max_long_side": MAX_LONG_SIDE,
        "sift_max_num_features": MAX_SIFT_FEATURES,
        "random_seed": RANDOM_SEED,
        "random_seed_note": (
            "set explicitly because the version supports it; bit-level SfM determinism is "
            "nevertheless not guaranteed"
        ),
        "camera_model": CAMERA_MODEL,
        "camera_sharing": "per_image_independent",
        "focal_length_init": f"{FOCAL_LENGTH_FACTOR} * max(width, height)",
        "principal_point_init": "(width/2, height/2) in COLMAP native pixel coordinates",
        "radial_param_init": 0.0,
        "calibration_status": "estimated",
        "intrinsics_initialization": "heuristic",
        "matching": "exhaustive",
        "min_model_size": MIN_MODEL_SIZE,
        "min_model_size_note": (
            "lowered to the contract's usable-model floor (>= 2 registered images) so a "
            "contract-usable model is not discarded by a larger backend default"
        ),
        "multiple_models": MULTIPLE_MODELS,
        "ba_refine_focal_length": True,
        "ba_refine_principal_point": False,
        "ba_refine_extra_params": True,
        "principal_point_optimization_note": "principal point is not freely optimised",
    }
    options.update(profiles.effective_profile_fields(profile))
    if profile == profiles.PROFILE_CAD_OBJECT_PINHOLE_V1:
        options.update(camera_model="SIMPLE_PINHOLE", ba_refine_extra_params=False)
        options.pop("radial_param_init")
    return options


def input_fingerprint(
    upstream: UpstreamRun,
    rows: list[UpstreamRow],
    options: dict[str, Any],
    backend: dict[str, Any],
    implementation: str,
    contract_version: str = CONTRACT_VERSION,
) -> str:
    """Canonical digest of everything the run depends on.

    ``contract_version`` is passed in rather than read from the module constant, so a
    profile run fingerprints itself with *its own* version.  The default profile
    therefore gains no new fingerprint field — only a new value, because the
    implementation digest changed.
    """
    payload = {
        "contract_version": contract_version,
        "upstream_run_json_sha256": upstream.run_json_sha256,
        "upstream_frames_jsonl_sha256": upstream.frames_jsonl_sha256,
        "rows": [
            [
                row.input_index,
                row.frame_id,
                row.canonical_image_sha256,
                row.canonical_mask_sha256,
            ]
            for row in sorted(rows, key=lambda item: item.input_index)
        ],
        "effective_options": options,
        "backend": backend,
        "implementation_sha256": implementation,
    }
    return canonical_digest(payload)


# --------------------------------------------------------------------------- #
# the native backend
# --------------------------------------------------------------------------- #
#: The process-level environment variables this task is *permitted* to set on a
#: child, and therefore the only ones worth recording.  The full ``os.environ`` is
#: never persisted: it contains credentials and machine-specific paths, it is not
#: reproducible elsewhere, and its presence in a log creates a disclosure problem
#: that no amount of "we only read it for diagnosis" retracts.
ENVIRONMENT_OVERRIDE_KEYS = ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED")


#: Where the CLI tells ``backend_info`` to write its evidence.  A module-level dict
#: rather than a function attribute because the pre-existing regressions replace
#: ``cli.backend_info`` wholesale with a plain function, which would not carry an
#: attribute; the context then has to live somewhere the replacement cannot erase.
backend_probe_context: dict[str, Any] = {}


def backend_probe_command(python_executable: str) -> list[str]:
    """The exact argv used for the version probe, so it can be compared, not guessed.

    ``-u`` is passed **explicitly on the command line**, not merely set as
    ``PYTHONUNBUFFERED`` in the environment.  ``-I`` (isolated mode) is present, and
    isolated mode makes the interpreter *ignore* ``PYTHON*`` environment variables —
    so ``PYTHONUNBUFFERED=1`` on a probe launched with ``-I`` is not a buffering
    instruction at all.  Recording it as though it took effect would be recording a
    setting instead of an effect.  ``-u`` is a command-line option and cannot be
    ignored by ``-I``, which is what actually makes the probe's stdout unbuffered.
    """
    return [python_executable, "-I", "-u", "-B", "-c", _BACKEND_PROBE_CODE]


_BACKEND_PROBE_CODE = (
    "import json,sys;"
    "import pycolmap,numpy;"
    "print(json.dumps({'version':pycolmap.__version__,'file':pycolmap.__file__,"
    "'numpy':numpy.__version__,'python':sys.version.split()[0]}))"
)


def environment_overrides(project_root: pathlib.Path) -> dict[str, str]:
    """Only the overrides this module sets, never a copy of the ambient environment."""
    return {
        "PYTHONPATH": str(project_root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def backend_info(
    python_executable: str,
    *,
    evidence_path: pathlib.Path | None = None,
    project_root: pathlib.Path | None = None,
    package_root: pathlib.Path | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    """Read the installed backend version rather than assuming the pinned one.

    When ``evidence_path`` is given, a start event is appended and flushed *before*
    the probe process is launched, and a finish event is appended afterwards with
    the real exit code (or ``null`` plus a reason when none was obtained).  The
    probe is a subprocess like any other stage; deriving its outcome from a later
    stage's success would be an inference, and an outer kill between the two would
    leave no trace that the probe had ever run.

    The probe is launched through the *same* mechanism as the four native stages:
    one ``Popen``, the child's stdout/stderr going straight into file objects, the
    real ``args``/``cwd``/``env`` being the ones the start record names.  The probe
    used to be a ``subprocess.run(..., capture_output=True)`` while its log claimed
    a cwd and an override whitelist that were never passed — a log that describes a
    launch other than the one that happened.  Nothing about the launch depends on
    the test environment: there is no branch here on whether ``subprocess.run`` has
    been replaced.

    Output reaching a file is not the same as output reaching it *incrementally*.
    The probe's two streams are the child's own ``stdout``/``stderr`` file objects,
    handed to ``Popen`` directly, so the OS writes each stream as the child produces
    it: a short line the probe has already flushed can be read from the log file
    before the probe exits.  There is no pipe and no reader thread in this path — an
    earlier revision's docstring claimed one, and that claim was corrected by the
    CAD-02A-pre-P-R3 close-out because it described an implementation this code never
    had.  The property that matters is the one the file redirection provides, not a
    pump this function does not use.

    The evidence location may also arrive through the module-level
    ``backend_probe_context`` (a dict) instead of as arguments — see the CLI, which
    needs to stay positionally compatible with the pre-existing test stubs.  An
    explicit argument always wins over the context.
    """
    context = backend_probe_context or {}
    if evidence_path is None and context.get("evidence_path"):
        evidence_path = pathlib.Path(context["evidence_path"])
    if project_root is None and context.get("project_root"):
        project_root = pathlib.Path(context["project_root"])
    if package_root is None and context.get("package_root"):
        package_root = pathlib.Path(context["package_root"])

    argv = backend_probe_command(python_executable)
    started_utc, started_clock = _utc_now(), time.monotonic()
    override = environment_overrides(project_root) if project_root is not None else {}

    logs_dir = evidence_path.parent if evidence_path is not None else None
    stdout_path = logs_dir / "backend_process.stdout.txt" if logs_dir is not None else None
    stderr_path = logs_dir / "backend_process.stderr.txt" if logs_dir is not None else None

    # The environment the probe will really be launched with, built once so the
    # record cannot describe a different one.  ``PYTHONUNBUFFERED`` is kept for
    # non-``-I`` callers, but it is **not** what makes this probe unbuffered: the
    # probe argv carries ``-I``, under which the interpreter ignores ``PYTHON*``
    # variables.  ``-u`` on the command line is the effective mechanism, and the
    # record separates "what was passed" from "what the interpreter adopts".
    env = dict(os.environ)
    env.update(override)
    env["PYTHONUNBUFFERED"] = "1"

    interpreter_flags = [token for token in argv if token.startswith("-") and token != "-c"]

    if evidence_path is not None:
        # Refuse to launch when the pre-launch record cannot be written.  A probe
        # that ran with no start line would leave a record that disagrees with what
        # happened, and the only cheap repair point is before the child exists.
        try:
            _evidence_append(evidence_path, {
                "event": "start",
                "kind": "backend_probe",
                "started_utc": started_utc,
                "started_monotonic": started_clock,
                "argv": argv,
                "cwd": str(project_root) if project_root is not None else None,
                "interpreter": python_executable,
                "interpreter_isolation_flags": interpreter_flags,
                "environment_overrides": override,
                # ``environment_unbuffered`` records what was *passed*; under ``-I``
                # it is inert.  The two keys below keep that distinction explicit so
                # the log cannot be read as "the interpreter is unbuffered because
                # this variable was set".
                "environment_unbuffered": env.get("PYTHONUNBUFFERED"),
                "environment_unbuffered_effective": "-u" in argv,
                "unbuffered_mechanism": (
                    "argv -u (command line, not ignored by -I)"
                    if "-u" in argv else
                    "PYTHONUNBUFFERED only (inert under -I)"
                ),
                "code_sha256": _code_sha256(package_root),
                "stdout_path": str(stdout_path) if stdout_path is not None else None,
                "stderr_path": str(stderr_path) if stderr_path is not None else None,
                "timeout_seconds": timeout,
            })
        except EvidenceWriteError as exc:
            raise BackendUnavailable(
                "refused to launch the backend probe: its launch record could not be "
                f"written ({exc})"
            ) from exc

    finished = False

    def finish(payload: dict[str, Any]) -> bool:
        """Write the finish record.  Never raises; returns whether it was written.

        A raise here would escape *after* a child has been created, skipping its
        cleanup — the failure a probe must not have.  The boolean is what callers
        use to decide whether the evidence is complete, and the cleanup below runs
        regardless of this return value.
        """
        nonlocal finished
        finished = True
        if evidence_path is None:
            # No evidence location was given: there is nothing to record, and the
            # record-writing helpers require a real path.  Calling them with ``None``
            # was the ``None.parent`` failure; the guard belongs here, once.
            return True
        return _evidence_append_tolerant(evidence_path, {
            "event": "finish",
            "kind": "backend_probe",
            "finished_utc": _utc_now(),
            "duration_monotonic_seconds": round(time.monotonic() - started_clock, 6),
            **payload,
        })

    def note_timeout() -> bool:
        """Record the timeout event.  Returns whether the record was written.

        Separate from ``_handle_timeout`` so a write failure here cannot decide
        whether the child gets cleaned up: it is reported, and the kill proceeds.
        """
        if evidence_path is None:
            return True
        return _evidence_append_tolerant(evidence_path, {
            "event": "timeout",
            "stage": "backend_probe",
            "recorded_utc": _utc_now(),
            "timeout_seconds": round(timeout, 3),
            "note": "the child is being terminated; the exit code of a killed process is "
                    "not this probe's result and is not reported as one",
        })

    #: The cleanup budget is created **once** for the whole probe and shared by every
    #: attempt.  A one-element list rather than a plain variable, so the nested
    #: closure below can fill it in.  ``cleanup`` never creates a deadline of its
    #: own; a retry therefore spends only the *remaining* time instead of opening a
    #: fresh window, which is what "one shared 20 s deadline" has to mean at run time
    #: (a single source expression does not by itself prove a single evaluation).
    cleanup_deadline_holder: list[float] = []

    def cleanup_deadline() -> float:
        if not cleanup_deadline_holder:
            cleanup_deadline_holder.append(
                time.monotonic() + _PROBE_CLEANUP_DEADLINE_SECONDS)
        return cleanup_deadline_holder[0]

    def cleanup(process: Any, *, reason: str) -> dict[str, Any]:
        """Terminate and reap the probe's own child; report what actually happened.

        Uses the probe's single shared deadline (see ``cleanup_deadline``), so every
        step — the kill, the wait, and any later retry — is bounded by the *same*
        instant rather than by a fresh 20 s period per call.

        Returns a reportable outcome instead of a bare boolean, because a boolean
        cannot carry the facts a caller has to record: whether the exit status was
        really obtained, what it was, and which error stopped the kill or the wait.
        ``finished`` means the reap completed; ``exit_confirmed`` means a real exit
        status was read, which is deliberately a *separate* fact — a reaped child
        whose code cannot be read is not a confirmed exit.  It never raises: a
        cleanup failure must be *reported*, not thrown from a place where the
        exception itself would skip the remaining release.

        Three facts are recorded separately because they are three different events
        and a single boolean cannot carry all of them:

        * ``termination_requested`` — the termination call returned **without raising**.
          A real ``Popen.kill()`` returns ``None`` on success, so the return value is
          not read at all: ``bool(None)`` is ``False``, and deriving "the request
          failed" from it recorded a successful termination as a failure.  A request
          that could not be issued is reported in ``kill_error`` instead.
        * ``finished`` — the bounded reap actually completed.
        * ``exit_confirmed`` + ``returncode`` — a real exit status was read.
        """
        deadline = cleanup_deadline()
        outcome: dict[str, Any] = {
            "attempted": True,
            "finished": False,
            "exit_confirmed": False,
            "returncode": None,
            "termination_requested": False,
            "kill_error": None,
            "wait_error": None,
            "budget_exhausted": False,
            "reason": reason,
        }
        try:
            process.kill()
            # No inspection of the return value: ``None`` is what a real, successful
            # ``Popen.kill()`` returns.  Reaching this line is the fact being recorded.
            outcome["termination_requested"] = True
        except Exception as exc:  # noqa: BLE001 - a kill failure must not skip the wait
            outcome["kill_error"] = f"{type(exc).__name__}: {exc}"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            outcome["budget_exhausted"] = True
            outcome["wait_error"] = (
                "the shared cleanup deadline was already spent before the reap, so the "
                "child's exit status was not observed")
            return outcome
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            outcome["wait_error"] = (
                "the reap did not complete within the shared cleanup deadline")
            return outcome
        except Exception as exc:  # noqa: BLE001 - an unknown wait failure is a failed reap
            outcome["wait_error"] = f"{type(exc).__name__}: {exc}"
            return outcome
        outcome["finished"] = True
        try:
            code = process.returncode
        except Exception as exc:  # noqa: BLE001 - the reap succeeded, the code did not
            outcome["wait_error"] = (
                f"the exit status could not be read: {type(exc).__name__}: {exc}")
            return outcome
        if code is None:
            outcome["wait_error"] = (
                "the reap completed but the process reported no exit status, so no "
                "exit code is claimed")
            return outcome
        outcome["returncode"] = int(code)
        outcome["exit_confirmed"] = True
        return outcome

    #: Whether a termination request was issued and returned *without raising*.  It is
    #: not "the child is dead" and it is not read out of the kill's return value: a real
    #: ``Popen.kill()`` returns ``None`` on success.  The reap is recorded separately in
    #: ``cleanup_finished`` and the status in ``exit_confirmed``/``returncode``.
    #:
    #: It is also **not** the reason the probe fails.  Whether this call must fail is
    #: decided by ``probe_timed_out`` alone.  A refused kill (``termination_requested``
    #: false) must not turn a timeout into a success, and a successful kill must not by
    #: itself be the evidence that a timeout happened.
    termination_requested = False
    #: Whether this call's first ``wait`` raised ``TimeoutExpired``: the probe exceeded
    #: the caller's bound.  This is the fact that makes the call a failure.  It is set in
    #: the ``except`` arm that catches the timeout and nowhere else, deliberately separate
    #: from ``termination_requested`` -- the earlier shape set the termination flag on
    #: entering the timeout arm, so a kill that raised still left the record claiming a
    #: termination had been requested, and the timeout's failure condition was expressed
    #: through a flag that also drove the kill.
    probe_timed_out = False
    cleanup_finished = True
    cleanup_attempted = False
    timeout_recorded = False
    process: Any = None
    child_created = False
    exit_confirmed = False
    #: Bound up-front so a ``wait`` that raises before returning can never leave the
    #: name unbound.  ``exit_confirmed`` — not ``None``-ness — is what says whether
    #: the value is a real measurement, so ``null`` here still means "not obtained".
    returncode: int | None = None
    #: Whether ``returncode`` came from the bounded reap rather than from the initial
    #: wait.  The two are different measurements of different events and the record must
    #: not describe a code obtained during cleanup as one obtained before it.
    code_from_cleanup = False
    cleanup_errors: list[str] = []
    stdout_text = ""
    stderr_text = ""
    scratch_paths: tuple[pathlib.Path, ...] = ()
    scratch_released: list[dict] = []
    scratch_residual: list[dict] = []
    release_done = False
    finish_recorded = True
    #: Set once a *definitive* outcome record has been written, so a later generic
    #: handler cannot append a second record that would overwrite an exit code or a
    #: termination status already known and already recorded.
    outcome_recorded = False
    #: The payload the single record is built from.  It is filled while the failure
    #: is known but is written only in the ``finally``, after the scratch release has
    #: actually run, so the residual in that record is real rather than a
    #: not-yet-computed empty list.
    failure_payload: dict[str, Any] | None = None
    #: The exception that is leaving this call, bound explicitly in the ``except`` below
    #: rather than looked up with ``sys.exc_info()`` in the ``finally``.  ``sys.exc_info()``
    #: is *thread* state, not frame state: inside a ``finally`` that is not handling
    #: anything itself it can still name an exception being handled further out, so a
    #: probe called from a caller's ``except`` block would have this probe's release facts
    #: attached to somebody else's error.  Bound here so the ``finally`` cannot confuse the
    #: two, and left ``None`` on every path that completes normally.
    propagated: BaseException | None = None

    def release() -> tuple[list[dict], list[dict]]:
        """Release the scratch pair exactly once; return (released, residual)."""
        nonlocal scratch_released, scratch_residual, release_done
        if not release_done:
            release_done = True
            scratch_released, scratch_residual = _discard(scratch_paths)
        return scratch_released, scratch_residual

    # Everything from the first resource acquisition to the last release lives inside
    # one ``try``.  The earlier shape created the scratch pair *before* the ``try``
    # and opened the log files inside it, so a failure anywhere between the two — a
    # second ``mkstemp`` failing, an unreadable output file, a JSON root that is not
    # an object — could leave a held resource unclosed with no record of it.
    #
    # Resource acquisition order, and what each step owns:
    #   * scratch pair   — held from ``mkstemp``; released by ``_discard``.
    #   * ``Popen``      — held from the successful constructor call; reaped by
    #                      ``cleanup()`` only while the child's exit is *not* confirmed.
    #   * log handles    — owned by their own ``with`` and closed by it, before the wait.
    try:
        if stdout_path is None or stderr_path is None:
            # No evidence location was given, so the probe runs without captured
            # streams.  This is the bare ``backend_info(interpreter)`` form the legacy
            # regression uses: a probe is still a probe, and its version is still read
            # from its stdout, so a temporary file pair stands in for the evidence
            # directory.
            #
            # The pair is a plain ``tempfile.mkstemp`` file.  It is **not** created
            # with ``O_TEMPORARY`` and no reader thread pumps its descriptors: the
            # child's stdout/stderr are handed to the file objects directly and the OS
            # writes them.  An earlier revision's comments claimed both, and those
            # comments were corrected by the CAD-02A-pre-P-R3 close-out because they
            # described an implementation this code never had.
            stdout_path = _make_probe_stream("backend_probe_", ".out")
            try:
                stderr_path = _make_probe_stream("backend_probe_", ".err")
            except OSError:
                # The first scratch file exists and the second does not.  The held
                # path is handed to the ``finally`` rather than left on disk with no
                # record, so the release attempt happens and its outcome is reported.
                scratch_paths = (stdout_path,)
                raise
            scratch_paths = (stdout_path, stderr_path)

        # ``stdout``/``stderr`` are file objects so the OS writes them as produced;
        # the ``-u`` in ``argv`` is what makes a *short* line appear promptly rather
        # than at exit.  The child is launched through ``subprocess.Popen`` directly,
        # so the pre-existing regressions that replace ``subprocess.run`` keep
        # working: they never reach this call, and this call does not consult whether
        # ``run`` has been replaced.
        with open(stdout_path, "wb") as out_handle, open(stderr_path, "wb") as err_handle:
            process = subprocess.Popen(
                argv,
                stdout=out_handle,
                stderr=err_handle,
                cwd=str(project_root) if project_root is not None else None,
                env=env,
            )
            # The child exists from here on.  Recorded *before* the wait, so the
            # ``except`` cannot mistake an already-created child for one that never
            # started — the exact confusion that produced the false "could not be
            # created, cleanup_finished=true" record.
            child_created = True

        try:
            returncode = process.wait(timeout=timeout)
            exit_confirmed = True
        except subprocess.TimeoutExpired:
            # The child exists and the launch record may be unwritable.  Record
            # the timeout best-effort, then clean up unconditionally: neither a
            # missing evidence path nor a failed timeout record may skip the
            # kill/wait.  ``timeout_recorded`` is carried into the finish payload
            # so a lost timeout line is visible rather than inferred from silence.
            timeout_recorded = note_timeout()
            # The timeout is the failure condition, and it is recorded on its own flag
            # before any termination is attempted.  ``termination_requested`` is *not* set
            # here: it means "a kill call returned without raising" and is written only by
            # ``cleanup`` once that call has actually returned.  The earlier shape set it
            # on entering this arm, so a kill that raised still produced a record claiming
            # the request had succeeded.
            probe_timed_out = True
            cleanup_attempted = True
            outcome = cleanup(process, reason="probe exceeded its bound")
            # Taken from the outcome: the fact is the kill call returning without raising,
            # which the outcome recorded at the point it happened.
            termination_requested = bool(outcome["termination_requested"])
            cleanup_finished = bool(outcome["finished"])
            for key in ("kill_error", "wait_error"):
                if outcome.get(key):
                    cleanup_errors.append(f"{key}: {outcome[key]}")
            # ``cleanup_attempted`` records that the kill/reap *was performed*, not that
            # it succeeded.  ``exit_confirmed`` is set only from the outcome's own
            # ``exit_confirmed``, which requires a real status to have been read: a
            # failed reap must not be reported as a confirmed exit, and marking it so
            # would also suppress the bounded retry in the ``except`` below.
            if outcome["exit_confirmed"]:
                exit_confirmed = True
                returncode = outcome["returncode"]
                code_from_cleanup = True
            else:
                returncode = None

        # Read the streams after the child is gone, but their content was produced while
        # it ran: the files were the child's own stdout/stderr, not a buffer assembled at
        # the end.  A terminated probe still has whatever it managed to write, and that
        # partial output is reported rather than discarded.
        stdout_text = _read_text(stdout_path)
        stderr_text = _read_text(stderr_path)

        if probe_timed_out:
            # A probe that exceeded its bound has no reportable result, and a timeout must
            # not be passed off as "the probe ran and reported".  The condition is
            # ``probe_timed_out``, not ``termination_requested``: a kill that could not be
            # issued does not make the timeout go away, and a registered termination is not
            # by itself a timeout.
            #
            # The payload is *staged* here and written by the ``finally`` once the scratch
            # release has run, so the one record for this invocation carries the real
            # residual instead of an empty list captured before ``_discard`` was ever
            # called.  ``cleanup_finished`` is never set to True to tidy the record.
            #
            # ``terminated`` keeps its compatibility meaning -- "a termination request was
            # issued and returned without raising" -- and is therefore *not* hardwired true.
            # Whether the child is dead is a different fact and is reported through
            # ``exit_confirmed``/``process_exit_code``; the reason text below names the
            # timeout as the cause rather than announcing a termination that may not have
            # been issued.
            failure_payload = {
                "process_exit_code": returncode,
                "process_exit_code_reason": (
                    "the probe exceeded its bound; no exit code was obtained, because the "
                    "bounded reap did not complete or did not yield a status"
                    if returncode is None else
                    # The code above came from the bounded reap, and that is all this field
                    # claims.  It does not say the termination *caused* the exit: when the
                    # kill itself raised, the reap that still ran read the child's own,
                    # natural status, so naming it "the exit status of the termination" would
                    # assert a cause that was never established.  It is the status the
                    # bounded reclamation obtained, whichever way the child came to end.
                    "the probe exceeded its bound; the code above is the exit status the "
                    "bounded reclamation obtained, not an algorithm return"
                ),
                "timed_out": True,
                "terminated": termination_requested,
                "termination_requested": termination_requested,
                "exit_confirmed": exit_confirmed,
                "timeout_recorded": timeout_recorded,
                "cleanup_attempted": cleanup_attempted,
                "cleanup_finished": cleanup_finished,
                "cleanup_errors": cleanup_errors,
                "stdout_tail": stdout_text.strip().splitlines()[-3:],
                "stderr_tail": stderr_text.strip().splitlines()[-3:],
            }
            raise BackendUnavailable(
                f"the backend probe exceeded its {round(timeout, 3)}s bound and no version "
                f"was reported (termination_requested={termination_requested}, "
                f"exit_confirmed={exit_confirmed})"
            )

        if returncode != 0:
            failure_payload = {
                "process_exit_code": int(returncode),
                "stderr_tail": stderr_text.strip().splitlines()[-3:],
                "terminated": False,
                "termination_requested": termination_requested,
                "exit_confirmed": exit_confirmed,
                "cleanup_attempted": cleanup_attempted,
                "cleanup_finished": cleanup_finished,
                "cleanup_errors": cleanup_errors,
            }
            raise BackendUnavailable(
                "pycolmap is not importable in the backend interpreter: "
                f"{stderr_text.strip().splitlines()[-1:] or ['no stderr']}"
            )
        try:
            parsed = json.loads(stdout_text)
        except ValueError as exc:
            failure_payload = {
                "process_exit_code": int(returncode),
                "note": f"the probe exited 0 but its output was not JSON: {type(exc).__name__}",
                "terminated": False,
                "termination_requested": termination_requested,
                "exit_confirmed": exit_confirmed,
                "cleanup_attempted": cleanup_attempted,
                "cleanup_finished": cleanup_finished,
                "cleanup_errors": cleanup_errors,
            }
            raise BackendUnavailable(
                f"the backend probe exited 0 but did not report a version: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            # A JSON *array* parses cleanly and then fails on ``info.get`` far from
            # here; catching it at the parse site keeps the failure attributed and,
            # through the outer ``except``/``finally``, still releases every held
            # resource instead of skipping the cleanup.
            failure_payload = {
                "process_exit_code": int(returncode),
                "note": "the probe exited 0 but its output was JSON that is not an object",
                "terminated": False,
                "termination_requested": termination_requested,
                "exit_confirmed": exit_confirmed,
                "cleanup_attempted": cleanup_attempted,
                "cleanup_finished": cleanup_finished,
                "cleanup_errors": cleanup_errors,
            }
            raise BackendUnavailable(
                "the backend probe exited 0 but did not report a version: its JSON root "
                f"is {type(parsed).__name__}, not an object"
            )

        info = parsed
        info["runtime"] = "cpu"
        info["name"] = BACKEND_NAME
        # The scratch release runs *before* the record is written, so this invocation's
        # single record carries the real residual rather than a placeholder captured
        # before ``_discard`` ran, and no second record is appended afterwards that
        # could restate an exit code already recorded.  A failed finish write is still
        # not ordinary success: the version was read but the record that the probe was
        # measured is missing, and the error is raised after the release so cleanup
        # cannot be blocked by a logging failure.
        released, residual = release()
        finish_recorded = finish({
            "process_exit_code": int(returncode),
            "reported_version": info.get("version"),
            "terminated": False,
            "termination_requested": termination_requested,
            "exit_confirmed": exit_confirmed,
            "cleanup_attempted": cleanup_attempted,
            "cleanup_finished": cleanup_finished,
            "cleanup_errors": cleanup_errors,
            "scratch_released": released,
            "scratch_residual": residual,
        })
        outcome_recorded = True
    except BaseException as leaving:
        # One exit path for every failure after the first resource was acquired: a
        # ``wait`` that raises ``OSError`` (or anything else) on an already-created
        # child, a failed stream read, a non-object JSON root, or a fresh ``OSError``
        # from the second ``mkstemp``.  The original exception is preserved and
        # re-raised unchanged; every cleanup fact is *reported*, so neither a cleanup
        # error nor a logging error can replace the original error.
        if child_created and not exit_confirmed and not cleanup_attempted:
            # The child exists, its exit was never confirmed, and no cleanup has been
            # attempted yet — the case the old code mislabelled "could not be created".
            # The guard on ``cleanup_attempted`` is what stops a *failed* earlier
            # kill/reap from starting a second one, and the call shares the probe's
            # single deadline, so it can only spend the time already left rather than
            # opening a fresh 20 s window.
            cleanup_attempted = True
            outcome = cleanup(process, reason="the probe failed after launch")
            cleanup_finished = bool(outcome["finished"])
            for key in ("kill_error", "wait_error"):
                if outcome.get(key):
                    cleanup_errors.append(f"{key}: {outcome[key]}")
            # Taken from the outcome's own ``termination_requested``, which is set when
            # the kill call returned without raising.  It is *not* derived from what
            # ``kill()`` returned: a real ``Popen.kill()`` returns ``None`` on success, so
            # asking whether that value was truthy recorded a successful termination as
            # ``terminated=False`` whenever the first wait had failed and cleanup saved it.
            termination_requested = termination_requested or bool(
                outcome["termination_requested"])
            if outcome["exit_confirmed"]:
                # A real status obtained during cleanup is a measurement and is kept,
                # not discarded because the *first* wait happened to fail.  The record
                # distinguishes the two, so "cleanup obtained the code" is not reported
                # as "the code was known before the failure".
                exit_confirmed = True
                returncode = outcome["returncode"]
                code_from_cleanup = True
        if failure_payload is None:
            failure_payload = {
                "process_exit_code": returncode if exit_confirmed else None,
                "process_exit_code_reason": (
                    "the exit status above was obtained by the bounded reap that ran as "
                    "part of cleanup, after the initial wait failed"
                    if code_from_cleanup else
                    "the exit status above was obtained before this failure"
                    if exit_confirmed and returncode is not None else
                    "the probe process was created but its exit status was never obtained"
                    if child_created else
                    "the probe process was never created"
                ),
                "terminated": termination_requested,
                "termination_requested": termination_requested,
                "exit_confirmed": exit_confirmed,
                "cleanup_attempted": cleanup_attempted,
                "cleanup_finished": cleanup_finished,
                "cleanup_errors": cleanup_errors,
            }
        propagated = leaving
        raise
    finally:
        # Release runs on every path and never replaces a propagated error.  On the
        # failure path it runs *before* the record below is written, so the residual it
        # produces is the real one; the success path already released inside the
        # ``try`` and ``release`` is one-shot.
        released, residual = release()
        if failure_payload is not None and not outcome_recorded:
            try:
                finish_recorded = finish({
                    **failure_payload,
                    "scratch_released": released,
                    "scratch_residual": residual,
                })
                outcome_recorded = True
            except Exception:  # noqa: BLE001 - a record failure must not mask the cause
                # The release above has already happened, so a logging failure cannot
                # withhold it.  Recorded rather than swallowed silently: "the failure was
                # summarised" and "the summary could not be written" are different facts,
                # and the caller of a probe with no evidence path has no other source.
                finish_recorded = False
        # Attached after the release, so what is attached is what happened.  This is the
        # only channel a caller that passed no ``evidence_path`` has for the residual:
        # ``finish`` is a no-op without a location, and the return-mapping form below is
        # never reached once an exception is propagating.
        _attach_release_facts(propagated, {
            "scratch_released": released,
            "scratch_residual": residual,
            "cleanup_attempted": cleanup_attempted,
            "cleanup_finished": cleanup_finished,
            "termination_requested": termination_requested,
            "probe_timed_out": probe_timed_out,
            "exit_confirmed": exit_confirmed,
            "process_exit_code": returncode if exit_confirmed else None,
            "code_from_cleanup": code_from_cleanup,
            "cleanup_errors": list(cleanup_errors),
            "finish_record_write_attempted": failure_payload is not None,
            "finish_recorded": finish_recorded,
            "evidence_path": str(evidence_path) if evidence_path is not None else None,
            "source": "backend_info, after the scratch release and after the finish write",
        })

    if not finish_recorded:
        raise BackendUnavailable(
            "the backend probe read the backend version but its finish record could not "
            f"be written; the measurement is incomplete (version={info.get('version')!r}, "
            f"scratch_released={scratch_released}, scratch_residual={scratch_residual})"
        )
    if evidence_path is None and scratch_residual:
        # No evidence location: ``finish`` is a no-op, so a file that could not be
        # released has nowhere to be recorded.  It is surfaced on the returned mapping
        # instead — only when there is one, so the ordinary return is unchanged — which
        # is what keeps the no-evidence form from silently reporting success over a
        # residual.
        info["scratch_residual"] = scratch_residual
    return info


#: Attribute under which ``backend_info`` attaches the post-release facts to an exception
#: that is leaving it.  A named constant so a caller does not have to hard-code the
#: spelling, and so the attachment point is greppable from the test that checks it.
RELEASE_FACTS_ATTR = "backend_release_facts"


def _attach_release_facts(exc: BaseException | None, facts: dict[str, Any]) -> None:
    """Attach the post-release cleanup facts to ``exc`` without changing it at all.

    Called from ``backend_info``'s ``finally`` *after* the scratch release has run, so
    ``facts`` describes what happened rather than what was intended.  This is the only
    channel open to the no-evidence form: ``finish`` is a no-op without an
    ``evidence_path``, so a caller that supplied none would otherwise never learn that a
    scratch file survived — the earlier shape surfaced the residual only on the return
    mapping, and an exception path never reaches it.

    Deliberately narrow.  Nothing here touches ``args``, ``__cause__``, ``__context__``,
    the traceback or the class, and nothing raises: the original exception has to reach
    the caller with its own identity, type and reason, and a cleanup fact must never be
    able to replace it or to withhold the release that already happened.

    The structured attribute is the machine-readable half.  ``add_note`` adds the same
    facts for a human reading the traceback, and is used only for what the caller cannot
    get anywhere else — a residual, the absence of an evidence location, a lost finish
    record — so an ordinary failure's traceback is not padded.
    """
    if exc is None:
        return
    try:
        setattr(exc, RELEASE_FACTS_ATTR, facts)
    except Exception:  # noqa: BLE001 - annotation must never matter more than the error
        return
    bits: list[str] = []
    if facts.get("scratch_residual"):
        bits.append(
            "scratch file(s) could not be removed: "
            + "; ".join(
                f"{entry.get('path')}: {entry.get('error')}"
                for entry in facts["scratch_residual"]))
    if facts.get("evidence_path") is None:
        bits.append(
            "no evidence_path was given, so this exception is the only report of the "
            "release and of any residual")
    if facts.get("finish_record_write_attempted") and not facts.get("finish_recorded"):
        bits.append("the finish record could not be written")
    if bits:
        try:
            exc.add_note("backend probe release: " + "; ".join(bits))
        except Exception:  # noqa: BLE001 - a note is a convenience, not the record
            pass


def _make_probe_stream(prefix: str, suffix: str) -> pathlib.Path:
    """Create the scratch stream file for the no-evidence probe form.

    Returns a real, writable path usable as a child's stdout/stderr.  ``mkstemp``
    hands back an **open** descriptor as well as a name; the descriptor is closed
    here so the caller never inherits a half-managed fd (the old form let it leak).
    The file itself is removed by ``_discard`` once the child has been reaped.
    """
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return pathlib.Path(name)


def _discard(paths: tuple[pathlib.Path, ...]) -> tuple[list[dict], list[dict]]:
    """Remove the temporary stream files the no-evidence probe form created.

    Returns ``(released, residual)``.  Each entry is a small record
    ``{"path": ..., "error": ...}``: ``released`` for files that are gone, and
    ``residual`` for files that are still on disk with the exception that stopped
    the removal.  The caller carries both into the evidence, so "all resources were
    cleaned" is never claimed on a path where a delete failed — the residual names
    the exact path and the failure.

    Best effort on purpose: failing to delete a scratch file must not turn a
    successful version read into an error, and the file's existence is not part of
    any claim about the backend.

    A single attempt is deliberate.  On POSIX the child has been reaped by the time
    this runs, so the unlink succeeds.  If a file survives anyway, the reason is
    **UNKNOWN**: this round observed residuals on this host but did not establish
    the holder, the handle or the process tree, so no statement is made here that
    the condition is OS-inevitable or unbreakable.  A retry loop would only add its
    budget to every probe while still failing, so none is used.
    """
    released: list[dict] = []
    residual: list[dict] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            residual.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        else:
            released.append({"path": str(path), "error": None})
    return released, residual


def _read_text(path: pathlib.Path | None) -> str:
    """Read a captured stream; a missing or unreadable file reads as empty.

    The absence of the file is itself part of the evidence: the probe's exit status
    and the launch record still say what happened, so a read failure must not abort
    the reporting path that exists to record that failure.
    """
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class EvidenceWriteError(Exception):
    """An evidence line could not be written.

    Evidence that cannot be written must not be silently skipped: a run whose
    incremental record is missing is a run whose *claims* about that record are
    unverifiable.  Raising lets the caller either refuse to start the work or mark
    the run's evidence as invalid, instead of reporting success next to a hole.
    """


def _evidence_append(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Append one JSONL evidence line and flush it to the OS immediately.

    Buffered writes are the failure mode this exists to prevent: a line still in a
    Python buffer when the outer process is killed is a line nobody can read, which
    makes the "incremental" part of incremental evidence a claim rather than a fact.

    A write failure is raised, not swallowed.  The previous version discarded
    ``OSError`` with the note that "absence is itself visible" -- but absence is only
    visible if someone looks for it, and the run went on to report success beside a
    missing record.  Callers decide what a failure means: before a child is created
    it means "do not start"; during a run it means "this evidence is invalid".
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(dumps_strict(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise EvidenceWriteError(
            f"cannot write evidence to {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _evidence_append_tolerant(path: pathlib.Path, payload: dict[str, Any]) -> bool:
    """Best-effort evidence write *after* work has already happened.

    Returns ``False`` when the line could not be written.  Used only where refusing
    is no longer possible: the child has run, so the honest options are "record that
    the record is incomplete" or "lie by omission".  The caller propagates the
    boolean into the stage result and the run's status.
    """
    try:
        _evidence_append(path, payload)
    except EvidenceWriteError:
        return False
    return True


def _code_sha256(package_root: pathlib.Path | None) -> str | None:
    """Digest of the code the child will import, so a log names its own revision."""
    if package_root is None:
        return None
    package_root = pathlib.Path(package_root)
    pairs = [
        [path.name, sha256_file(path)]
        for path in sorted(package_root.glob("*.py"))
    ]
    return canonical_digest(pairs)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _handle_timeout(launch_record: pathlib.Path, stage: str, timeout: float) -> None:
    """Record the kill *before* performing it, so the reason survives the kill."""
    _evidence_append(launch_record, {
        "event": "timeout",
        "stage": stage,
        "recorded_utc": _utc_now(),
        "timeout_seconds": round(timeout, 3),
        "note": "the child is being terminated; the exit code of a killed process is not "
                "this stage's result and is not reported as one",
    })


def require_pinned_version(info: dict[str, Any]) -> None:
    version = str(info.get("version"))
    if version != BACKEND_PIN:
        raise BackendUnavailable(
            f"backend version {version!r} does not match the approved pin {BACKEND_PIN!r}"
        )


@dataclass
class NativeStageResult:
    stage: str
    started_utc: str
    finished_utc: str
    returncode: int | None
    timed_out: bool
    stdout_path: str
    stderr_path: str
    result: dict[str, Any] | None
    error: str | None
    is_process: bool = True
    #: False when the shared budget was exhausted and the stage never ran.  A
    #: stage that did not start must never be reported as a stage that ran.
    started: bool = True
    timeout_seconds: float | None = None
    #: Path of the start/launch record written *before* the child was created.
    launch_record_path: str | None = None
    #: Why ``returncode`` is null, when it is.  ``null`` alone is ambiguous between
    #: "timed out", "never started" and "the process object gave no code".
    returncode_reason: str | None = None
    #: True when the child was killed for exceeding its bound.  Distinguishes a
    #: termination from a normal return even though both reach this result.
    terminated: bool = False
    #: False when a terminated child did not exit within the bounded grace period,
    #: i.e. reaping did not complete.  Recorded rather than assumed.
    cleanup_finished: bool = True
    #: False when a post-run launch-record write failed.  The stage may have done
    #: real work, but its outcome is not verifiable and must not be counted as one.
    evidence_incomplete: bool = False


#: Fields each stage's result document must carry to count as a completed stage.
STAGE_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "extract_features": ("num_images", "cameras", "image_names_to_ids", "actual_options"),
    "match_exhaustive": ("num_matched_image_pairs", "actual_options"),
    "incremental_mapping": ("model_count", "models", "actual_options"),
    "export_models": ("model_count", "models"),
}


def snapshot_upstream_files(upstream: UpstreamRun) -> dict[str, str]:
    """Byte digest of every upstream file the run depends on.

    *Every* original row contributes, not only the deduplicated representatives:
    a mask, or the image belonging to a duplicate alias, can be edited during the
    run while the representative is untouched, and a representative-only check
    reports "no change".  Files are keyed by their upstream-relative path so the
    same underlying file is read once when several rows share it.

    The snapshot is taken from disk and compared with another snapshot taken from
    disk — never against hashes recomputed from in-memory state, which would
    trivially agree with themselves.
    """
    paths = {"run.json", "frames.jsonl"}
    for row in upstream.rows:
        paths.add(row.image_output)
        if row.mask_output is not None:
            paths.add(row.mask_output)

    snapshot: dict[str, str] = {}
    for relative in sorted(paths):
        path = _resolve_within(upstream.run_dir, relative, "upstream file")
        if not path.is_file():
            raise InputError(f"an upstream file disappeared during the run: {relative}")
        snapshot[relative] = sha256_file(path)
    return snapshot


def diff_snapshots(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Human-readable list of changed, added or removed upstream files."""
    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key not in after:
            changes.append(f"{key}: removed during the run")
        elif key not in before:
            changes.append(f"{key}: appeared during the run")
        elif before[key] != after[key]:
            changes.append(f"{key}: bytes changed during the run")
    return changes


class BudgetExhausted(Exception):
    """The shared deadline passed before this stage could start."""


@dataclass
class StageExchange:
    """What an executor observed while running one stage.

    The fields are deliberately the *observations*, never a verdict: ``returncode``
    is the status the process reported, and ``timed_out``/``terminated`` say how it
    was obtained.  A caller cannot express "it timed out" by leaving the code out,
    so a termination cannot decay into "the stage returned 1" on the way out.
    """
    returncode: int | None
    timed_out: bool = False
    terminated: bool = False
    cleanup_finished: bool = True
    returncode_reason: str | None = None


def _default_stage_executor(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> StageExchange:
    """Run the stage for real.  This is what ``executor=None`` means.

    ``subprocess.run`` is called with ``stdout``/``stderr`` handed straight to open
    file objects.  The OS writes the child's output into those files as it is
    produced, so a short flushed line is readable before the child exits — there is
    no pipe, no reader thread, and therefore no ``read(8192)`` that could wait for a
    buffer to fill or for EOF.  ``capture_output`` is deliberately not used: it
    assembles both streams in memory and returns them only after the child is gone,
    which is the opposite of an incremental record.

    ``cwd``/``env``/``timeout`` are used verbatim — they are the same values the
    launch record names, so a record cannot describe a launch other than this one.
    Passing them **to** ``subprocess.run`` is what makes replacing ``subprocess.run``
    a replacement of the executor rather than a change of run semantics: the call
    expression is the same one; only the callee differs.

    **Scope of the termination guarantee — do not overstate it.** This executor
    calls ``subprocess.run``, which on timeout kills the *direct child* and then
    reaps it internally (on the Windows path, after ``kill()`` it waits again
    without a separate bound of our own). It does **not** implement a hard reap
    deadline of its own, and it does **not** reap grandchildren: a process tree, the
    final wall-clock bound of the whole chain, and the case where a reaped child's
    own descendants outlive it are the responsibility of the separately accepted
    outer job supervisor. What this function does guarantee is narrower and must be
    stated as such: a *finite-lifetime* worker is bounded by ``timeout``, and the
    outcome observed here is labelled (``terminated``) rather than passed off as an
    algorithm return. ``_TERMINATION_GRACE_SECONDS`` is **not** applied on this path
    — the ``subprocess.run`` timeout path has no separate grace wait of ours — so it
    must not be cited as evidence that a stage's reaping is bounded by it.
    """
    with open(stdout_path, "wb") as out_handle, open(stderr_path, "wb") as err_handle:
        try:
            # ``check=False`` + no ``capture_output``: the return code is the result,
            # and the streams go to the files, not into memory.
            done = subprocess.run(
                argv, stdout=out_handle, stderr=err_handle,
                cwd=cwd, env=env, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            # ``subprocess.run`` kills the child before raising, but the handle it
            # killed is gone from this frame, so the real status of the termination is
            # recorded as ``None`` with the reason spelled out rather than invented.
            # No separate grace wait of ours is applied here: a hard reap deadline for
            # a whole process tree belongs to the outer supervisor, and claiming one
            # from this call site would be a guarantee this code does not implement.
            return StageExchange(
                returncode=None, timed_out=True, terminated=True,
                cleanup_finished=True,
                returncode_reason=(
                    "the stage exceeded its bound and was terminated; the exit status of "
                    "the termination is not observable through this executor and is not "
                    "invented"
                ),
            )

    # A replacement executor (or a callee that captured its own streams) may return
    # the child's output as text instead of letting it reach the file objects.  When
    # it does, that text *is* the child's output and belongs in the same log files
    # the default path writes, so the record is complete whichever call shape ran.
    # Appending here rather than overwriting keeps output already streamed to the
    # files by the OS intact.
    _append_returned_stream(stdout_path, getattr(done, "stdout", None))
    _append_returned_stream(stderr_path, getattr(done, "stderr", None))
    return StageExchange(returncode=int(done.returncode))


def _append_returned_stream(path: pathlib.Path, value: Any) -> None:
    """Write text a call returned in place of streaming it, without clobbering a file.

    ``capture_output``-style replacements hand back ``stdout``/``stderr`` as text;
    the incremental path leaves those attributes ``None`` because the OS already
    wrote the bytes.  Both cases must end with the log file holding the child's
    output, so this fills in only what is missing and never truncates.
    """
    if value is None:
        return
    if isinstance(value, bytes):
        text = value.decode("utf-8", "replace")
    elif isinstance(value, str):
        text = value
    else:  # pragma: no cover - a non-text attribute is not a stream
        return
    if not text:
        return
    with open(path, "ab") as handle:
        handle.write(text.encode("utf-8"))


#: The executor type accepted by ``run_native_stage``.
StageExecutor = Callable[..., StageExchange]


def run_native_stage(
    stage: str,
    spec: dict[str, Any],
    *,
    run_dir: pathlib.Path,
    logs_dir: pathlib.Path,
    project_root: pathlib.Path,
    python_executable: str | None = None,
    timeout: float = NATIVE_TIMEOUT_SECONDS,
    deadline: float | None = None,
    executor: "StageExecutor | None" = None,
) -> NativeStageResult:
    """Run one native stage in its own process and record the real outcome.

    ``deadline`` is a single ``time.monotonic()`` value shared by the whole native
    chain.  Each stage receives only what remains of it, and a stage is *not
    started* once the budget is gone — giving every stage its own full timeout
    would silently multiply the budget by the number of stages.

    ``executor`` is where this stage *may* be run.  It is a programmable
    dependency, not an environment probe: the default is the real executor, and the
    value is a parameter the caller passes, never something read out of the process
    (there is no check of ``subprocess.run``'s identity, no pytest variable, no
    "is this a test?" branch).  Replacing it changes *which* executor runs the
    stage; it never changes what running a stage means.  The default executor is
    itself exercised by the counterexample test, so "the default path was used" is
    an assertion rather than an assumption.
    """
    runner = executor if executor is not None else _default_stage_executor

    started = _utc_now()
    launch_record = logs_dir / f"{stage}.launch.jsonl"

    # Creating the log directory is itself a pre-launch step.  If it fails, no
    # record can ever be written, so the stage must not start: an error raised out
    # of here would be an unrecorded crash instead of a refusal, and a caller could
    # not tell "the stage crashed" from "the stage was never run".
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return NativeStageResult(
            stage=stage, started_utc=started, finished_utc=started,
            returncode=None, timed_out=False,
            stdout_path="", stderr_path="", result=None,
            error=(
                f"refused to start the native stage: the log directory {logs_dir} could "
                f"not be created ({type(exc).__name__}: {exc}); a stage whose record has "
                "nowhere to go must not be run"
            ),
            started=False,
            timeout_seconds=round(timeout, 3),
            launch_record_path=None,
            returncode_reason="the stage never started, so it has no exit code",
        )

    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _evidence_append(launch_record, {
                "event": "not_started",
                "stage": stage,
                "recorded_utc": started,
                "reason": "the shared runtime budget was exhausted before this stage could start",
                "deadline_monotonic": deadline,
                "now_monotonic": time.monotonic(),
            })
            return NativeStageResult(
                stage=stage,
                started_utc=started,
                finished_utc=started,
                returncode=None,
                timed_out=False,
                stdout_path="",
                stderr_path="",
                result=None,
                error=(
                    "the shared runtime budget was exhausted before this stage could start; "
                    "it was not executed"
                ),
                started=False,
                launch_record_path=str(launch_record.relative_to(run_dir)),
                returncode_reason="the stage never started, so it has no exit code",
            )
        timeout = min(timeout, max(1.0, remaining))

    spec_path = logs_dir / f"{stage}.spec.json"
    result_path = logs_dir / f"{stage}.result.json"
    spec_path.write_text(dumps_strict(spec, indent=2) + "\n", encoding="utf-8", newline="\n")
    result_path.unlink(missing_ok=True)

    executable = python_executable or sys.executable
    # Invoked with ``-c`` rather than ``-m``: the package ``__init__`` already
    # imports this module, so ``-m`` makes runpy warn about the module being in
    # sys.modules before execution, which pollutes the native stderr that is kept
    # as evidence.  Importing the entry point explicitly avoids the duplicate
    # import while still loading it as part of the package.
    bootstrap = (
        "import sys;"
        "from kozlov_lab.reconstruction.colmap_adapter import _worker_main;"
        "sys.exit(_worker_main(sys.argv[1:]))"
    )
    argv = [
        executable, "-B", "-c", bootstrap,
        "--stage", stage,
        "--spec", str(spec_path),
        "--result", str(result_path),
    ]
    # Only the overrides this task is authorised to set, never a copy of
    # ``os.environ``: the ambient environment carries credentials and is not part of
    # what makes the run reproducible.
    overrides = environment_overrides(project_root)

    stdout_path = logs_dir / f"{stage}.stdout.txt"
    stderr_path = logs_dir / f"{stage}.stderr.txt"

    # The environment the child is really launched with, assembled *before* the
    # start record so the record's ``environment_unbuffered`` is the value that
    # takes effect rather than a promise.  The inherited environment is not logged:
    # only the whitelist of overrides this task sets is.
    launch_env = dict(os.environ)
    launch_env.update(overrides)
    launch_env["PYTHONUNBUFFERED"] = "1"

    # Start event first, flushed, *before* the child exists.  If the outer process
    # is killed one millisecond later, this line is already on disk and says which
    # stage was launching, under which interpreter, with which budget.
    #
    # If this write fails the child is *not* created: evidence that cannot be
    # recorded before the work begins cannot be repaired afterwards, and a stage
    # that ran without its start line would leave a record that silently disagrees
    # with what happened.  Refusing here is the only failure point that is still
    # cheap.  The child has not been created, so no cleanup is owed.
    try:
        _evidence_append(launch_record, {
            "event": "start",
            "stage": stage,
            "started_utc": started,
            "started_monotonic": time.monotonic(),
            "argv": argv,
            "cwd": str(project_root),
            # The launch arguments the executor is actually handed.  A record that
            # names a cwd or an environment the child never received is a record of
            # a different launch, so these are the same objects passed below.
            "executor": getattr(runner, "__name__", type(runner).__name__),
            "interpreter": executable,
            "environment_overrides": overrides,
            "environment_unbuffered": launch_env.get("PYTHONUNBUFFERED"),
            "code_sha256": _code_sha256(pathlib.Path(__file__).resolve().parent),
            "spec_path": str(spec_path),
            "result_path": str(result_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "timeout_seconds": round(timeout, 3),
            "deadline_monotonic": deadline,
        })
    except EvidenceWriteError as exc:
        return NativeStageResult(
            stage=stage, started_utc=started, finished_utc=_utc_now(),
            returncode=None, timed_out=False,
            stdout_path="", stderr_path="", result=None,
            error=(
                "refused to start the native stage: the launch record could not be "
                f"written ({exc}); evidence that cannot be recorded before the work "
                "begins must not be assumed to exist afterwards"
            ),
            started=False,
            timeout_seconds=round(timeout, 3),
            launch_record_path=str(launch_record.relative_to(run_dir)),
            returncode_reason="the stage never started, so it has no exit code",
        )

    timed_out = False
    terminated = False
    returncode: int | None = None
    returncode_reason: str | None = None
    cleanup_finished = True
    try:
        exchange = runner(
            argv,
            cwd=str(project_root),
            env=launch_env,
            timeout=timeout,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        returncode = exchange.returncode
        timed_out = exchange.timed_out
        terminated = exchange.terminated
        cleanup_finished = exchange.cleanup_finished
        returncode_reason = exchange.returncode_reason
        if timed_out:
            # Recorded *before* the interpretation is applied: the kill already
            # happened inside the executor, and this line is what separates "the
            # stage returned this code" from "this code is the exit status of a
            # termination".
            _handle_timeout(launch_record, stage, timeout)
    except OSError as exc:
        # The child could not even be created.  That is a real, reportable outcome
        # and it is *not* an exit code: nothing ran, so there is no code to report.
        finished = _utc_now()
        _evidence_append_tolerant(launch_record, {
            "event": "finish",
            "stage": stage,
            "finished_utc": finished,
            "process_exit_code": None,
            "process_exit_code_reason": f"the child process could not be created: {type(exc).__name__}",
            "terminated": False,
            "cleanup_finished": True,
            "error": str(exc),
        })
        stderr_path.write_bytes(f"{type(exc).__name__}: {exc}\n".encode("utf-8"))
        return NativeStageResult(
            stage=stage, started_utc=started, finished_utc=finished,
            returncode=None, timed_out=False,
            stdout_path=str(stdout_path.relative_to(run_dir)),
            stderr_path=str(stderr_path.relative_to(run_dir)),
            result=None,
            error=f"the native stage process could not be created: {type(exc).__name__}: {exc}",
            timeout_seconds=round(timeout, 3),
            launch_record_path=str(launch_record.relative_to(run_dir)),
            returncode_reason="the child process was never created, so it has no exit code",
        )
    finished = _utc_now()

    # A real exit status obtained from a terminated child is still a real exit
    # status; ``terminated`` and ``timed_out`` carry the interpretation, so the code
    # is never passed off as a normal algorithm return.
    #
    # A failure here happens *after* the child ran, so refusing is not an option;
    # the run must instead know that its own record is incomplete.
    #
    # ``_evidence_append_tolerant`` returns *whether the write succeeded*, so the
    # completeness flag is its negation.  Getting this backwards is silently
    # invisible: every successful run would be marked unverifiable, and — worse in
    # this file's history — an inverted mapping would let a failed write be read as
    # a finished record.
    evidence_recorded = _evidence_append_tolerant(launch_record, {
        "event": "finish",
        "stage": stage,
        "finished_utc": finished,
        "process_exit_code": returncode,
        "process_exit_code_reason": returncode_reason,
        "timed_out": timed_out,
        "terminated": terminated,
        "cleanup_finished": cleanup_finished,
        "interpretation": (
            "terminated" if terminated else
            ("not_started" if returncode is None and not terminated else "returned")
        ),
    })
    evidence_incomplete = not evidence_recorded

    payload: dict[str, Any] | None = None
    error: str | None = None
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            error = f"stage result file is not readable JSON: {exc}"

    # The stage may have succeeded, but its own record did not land.  A run that
    # cannot show what happened must not be reported as one that can, so an
    # incomplete record is an error in its own right regardless of the exit status.
    if evidence_incomplete:
        error = (
            "the stage's evidence record could not be completed (a launch-record write "
            "failed after the child ran); this stage's outcome is not verifiable and "
            "must not be treated as a completed stage"
        )

    # Exit status alone is not completion.  A stage that returns 0 without writing
    # a usable result used to be marked OK, and the run then reported a normal
    # NO_MODEL -- a fabricated "the backend simply found nothing".
    #
    # A terminated stage reports ``returncode=None`` even when the reaping produced a
    # real status: the status of a kill is not a result.  The real code and the
    # interpretation both survive in the launch record.
    if timed_out:
        error = f"stage exceeded its {round(timeout, 3)}s bound"
        recorded_termination_status = returncode
        returncode = None
    elif error is None and returncode not in (0, None):
        tail = _tail_lines(stderr_path, 3)
        error = f"native stage exited {returncode}: {tail}"
    elif error is None and returncode is None:
        error = (
            "the native stage produced no exit code"
            + (f" ({returncode_reason})" if returncode_reason else "")
            + "; completion cannot be assumed"
        )
    elif error is None:
        if payload is None:
            error = (
                f"stage exited {returncode} but produced no result document; "
                "completion cannot be assumed from the exit status alone"
            )
        elif not isinstance(payload, dict):
            error = f"stage result is {type(payload).__name__}, expected a JSON object"
        elif not payload.get("ok", False):
            error = f"stage reported failure: {payload.get('error')}"
        else:
            missing = [name for name in STAGE_REQUIRED_FIELDS.get(stage, ()) if name not in payload]
            if missing:
                error = f"stage result is missing required field(s): {missing}"

    return NativeStageResult(
        stage=stage,
        started_utc=started,
        finished_utc=finished,
        returncode=returncode,
        timed_out=timed_out,
        stdout_path=str(stdout_path.relative_to(run_dir)),
        stderr_path=str(stderr_path.relative_to(run_dir)),
        result=payload,
        error=error,
        timeout_seconds=round(timeout, 3),
        launch_record_path=str(launch_record.relative_to(run_dir)),
        returncode_reason=returncode_reason,
        terminated=terminated,
        cleanup_finished=cleanup_finished,
        evidence_incomplete=evidence_incomplete,
    )


def _tail_lines(path: pathlib.Path, count: int) -> list[str]:
    """Last ``count`` non-empty lines of a file that may not exist or be partial."""
    try:
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if line.strip()]
    except OSError:
        return []
    return lines[-count:]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- #
# native worker (this module is also the child's entry point)
# --------------------------------------------------------------------------- #
def _worker_main(argv: list[str]) -> int:
    """Execute one stage inside the backend process and write a JSON result.

    Every failure is reported as data (``{"ok": false, "error": ...}``) so the
    parent can distinguish "the stage ran and the backend declined" from "the
    process died", which is what the contract's status table depends on.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="kozlov_lab.reconstruction.colmap_adapter")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)

    spec = json.loads(pathlib.Path(args.spec).read_text(encoding="utf-8"))
    outcome: dict[str, Any] = {"stage": args.stage, "ok": False}
    try:
        import pycolmap  # imported here so an import failure is a reported error
        if args.stage == "extract_features":
            outcome.update(_stage_extract(pycolmap, spec))
        elif args.stage == "match_exhaustive":
            outcome.update(_stage_match(pycolmap, spec))
        elif args.stage == "incremental_mapping":
            outcome.update(_stage_mapping(pycolmap, spec))
        else:
            outcome.update(_stage_export(pycolmap, spec))
        outcome["ok"] = True
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        outcome["traceback_tail"] = __import__("traceback").format_exc().strip().splitlines()[-6:]

    pathlib.Path(args.result).write_text(
        # ``default=str`` keeps native objects that the option serializers leave
        # unserializable (paths, enum-like wrappers) from aborting the result write
        # after the work is done.
        json.dumps(outcome, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    return 0 if outcome["ok"] else 1


def _stage_extract(pycolmap: Any, spec: dict) -> dict:
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.use_gpu = False
    extraction.num_threads = int(spec["num_threads"])
    extraction.max_image_size = int(spec["max_long_side"])
    extraction.sift.max_num_features = int(spec["max_sift_features"])

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = spec["camera_model"]
    reader.default_focal_length_factor = float(spec["focal_length_factor"])
    reader.camera_params = ""
    reader.existing_camera_id = -1
    if spec.get('foreground_mask_path'):
        if {b['image_name'] for b in spec['foreground_mask_bindings']} != set(spec['image_names']):
            raise BackendError('foreground image/mask coverage differs before feature extraction')
        verify_foreground_masks(spec['foreground_mask_bindings'])
        reader.mask_path = pathlib.Path(spec['foreground_mask_path'])

    pycolmap.extract_features(
        spec["database_path"], spec["image_path"],
        image_names=list(spec["image_names"]),
        camera_mode=getattr(pycolmap.CameraMode, spec["camera_mode"]),
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )

    selection_audit = None
    if spec.get('point_selection', 'native') != 'native':
        if spec['point_selection'] != 'rcover-v1' or not spec.get('foreground_mask_path'):
            raise BackendError('point selection requires rcover-v1 and bound foreground masks')
        audit_foreground_keypoints(spec['database_path'],spec['foreground_mask_bindings'])
        from .point_selection import apply_point_selection
        selection_audit=apply_point_selection(spec['database_path'],spec['foreground_mask_bindings'],
            pathlib.Path(spec['database_path']).parent/'point_selection.json', **({'profile':spec['reconstruction_profile']} if 'reconstruction_profile' in spec else {}))

    database = pycolmap.Database.open(spec["database_path"])
    cameras = {int(camera.camera_id): camera for camera in database.read_all_cameras()}
    images = {int(image.image_id): image for image in database.read_all_images()}
    keypoints = {
        int(image_id): int(database.num_keypoints_for_image(image_id)) for image_id in images
    }
    database.close()
    mask_audit = None
    if spec.get('foreground_mask_path'):
        expected_root = pathlib.Path(spec['foreground_mask_path']).resolve()
        if pathlib.Path(reader.mask_path).resolve() != expected_root:
            raise BackendError('actual reader mask_path differs from frozen foreground root')
        mask_audit = audit_foreground_keypoints(spec['database_path'],spec['foreground_mask_bindings'])
        verify_foreground_masks(spec['foreground_mask_bindings'])

    return {
        'foreground_mask_audit': mask_audit,
        **({'point_selection_audit':selection_audit} if selection_audit is not None else {}),
        "num_cameras": len(cameras),
        "num_images": len(images),
        "num_keypoints_total": int(sum(keypoints.values())),
        "cameras": {
            str(camera_id): {
                "model_name": camera.model_name,
                "width": int(camera.width),
                "height": int(camera.height),
                "params": [float(v) for v in camera.params],
                "has_prior_focal_length": bool(camera.has_prior_focal_length),
            }
            for camera_id, camera in sorted(cameras.items())
        },
        "database_image_ids": sorted(images),
        # The backend's own image id, keyed by the staging name it was given.
        # Read from the database: it is neither the input_index nor guaranteed to
        # be contiguous, so it must never be inferred from ordering.
        "image_names_to_ids": {
            image.name: int(image_id) for image_id, image in sorted(images.items())
        },
        # The database's own image -> camera reference.  Required to verify a shared
        # run really produced one camera referenced by every image, rather than
        # trusting the option that was requested.
        "image_camera_ids": {
            image.name: int(image.camera_id) for image_id, image in sorted(images.items())
        },
        "keypoints_per_image": {
            str(image_id): count for image_id, count in sorted(keypoints.items())
        },
        # The configuration actually handed to 4.2.1, serialized by the version's
        # own ``todict`` so native defaults are recorded rather than summarised by
        # hand.  What was read back from the database is kept separately, because a
        # request and its effect are different facts.
        "actual_options": {
            **({'point_selection':selection_audit['policy_id']} if selection_audit is not None else {}),
            "camera_mode": spec["camera_mode"],
            "extraction": _todict(extraction),
            "reader": _todict(reader),
        },
    }


def verify_foreground_masks(bindings: list[dict]) -> None:
    for item in bindings:
        for field in ('source_path','staged_path'):
            path = pathlib.Path(item[field])
            if not path.is_file() or path.stat().st_size!=item['bytes'] or sha256_file(path)!=item['sha256']:
                raise BackendError('foreground mask binding drift: '+str(path))


def audit_foreground_keypoints(database_path, bindings: list[dict]) -> dict:
    """Read every DB keypoint at COLMAP's truncation-to-int original pixel rule.

    Fixed source: 4.2.1 feature_extraction.cc MaskFeatures (lines 55-78), after
    ScaleKeypoints. image_reader.cc resolves mask_root/(image_name+'.png').
    This proves saved keypoint support plus the caller's actual reader binding;
    it is not a correspondence or geometry truth certificate.
    """
    import sqlite3
    import struct
    connection = sqlite3.connect(pathlib.Path(database_path).resolve().as_uri()+'?mode=ro',uri=True)
    connection.execute('PRAGMA query_only=ON')
    try:
        images = {name:int(ident) for ident,name in connection.execute('SELECT image_id,name FROM images')}
        if set(images)!={b['image_name'] for b in bindings}:
            raise BackendError('foreground extraction database image coverage differs')
        per_image=[]; total=0
        for binding in bindings:
            ident = images[binding['image_name']]
            rows = connection.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(ident,)).fetchall()
            if len(rows)!=1:
                raise BackendError('foreground extraction keypoint record missing')
            count,cols,data=rows[0]
            if count<0 or cols not in (2,4,6) or len(data or b'')!=count*cols*4:
                raise BackendError('foreground extraction malformed keypoint matrix')
            with Image.open(binding['staged_path']) as mask:
                mask.load(); width,height=mask.size; pixels=mask.load()
                violations=0
                for offset in range(0,len(data or b''),cols*4):
                    x,y=struct.unpack_from('<ff',data,offset)
                    if not math.isfinite(x) or not math.isfinite(y):
                        violations+=1;continue
                    ix,iy=int(x),int(y)  # C++ static_cast<int>: truncate toward zero.
                    if not (0<=ix<width and 0<=iy<height) or pixels[ix,iy]==0:
                        violations+=1
            per_image.append(dict(image_name=binding['image_name'],database_image_id=ident,
                keypoints=count,violations=violations,mask_sha256=binding['sha256']))
            total+=count
        if any(row['violations'] for row in per_image):
            raise BackendError('foreground mask consumption verification failed: out-of-mask DB keypoints')
        return dict(status='VERIFIED',rule='COLMAP 4.2.1 static_cast<int> truncation toward zero in canonical camera pixels; nonzero mask retains',
            checked_images=len(per_image),checked_keypoints=total,violations=0,per_image=per_image,
            mask_source_urls=['https://github.com/colmap/colmap/blob/4.2.1/src/colmap/controllers/feature_extraction.cc#L55-L78',
                'https://github.com/colmap/colmap/blob/4.2.1/src/colmap/controllers/image_reader.cc#L134-L160'])
    finally:
        connection.close()


def _todict(obj: Any) -> dict[str, Any]:
    """Serialize a native options object with the version's own interface."""
    if hasattr(obj, "todict"):
        return obj.todict()
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {"<unserializable>": repr(obj)}


def _stage_match(pycolmap: Any, spec: dict) -> dict:
    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = False
    matching.num_threads = int(spec["num_threads"])
    pairing = pycolmap.ExhaustivePairingOptions()

    # Two-view geometry verification is not re-configured: this build's own default
    # object is constructed here *and handed to the call*, so the values recorded
    # below belong to the same object that did the work.
    #
    # The previous version constructed this object and then never passed it, while
    # still labelling its ``todict`` as "actual options" — a record of an object the
    # backend never saw.  pycolmap 4.2.1's ``match_exhaustive`` takes a
    # ``verification_options`` parameter of exactly this type, so passing it is both
    # possible and the only way the record can be a fact rather than an inference.
    # No default value is changed: the object is this version's own default.
    geometry = _geometry_options(pycolmap)
    call_kwargs: dict[str, Any] = {
        "matching_options": matching,
        "pairing_options": pairing,
        "device": pycolmap.Device.cpu,
    }
    if geometry is None:
        # This build exposes no such object, so nothing could be passed.  The record
        # says so rather than describing a default that was never in the call.
        geometry_note = {
            "<not_recorded>": (
                "this pycolmap build exposes no two-view geometry option object, so none "
                "could be passed and the backend's internal defaults are not observable "
                "from here; the values in force are unverified"
            )
        }
    else:
        call_kwargs["verification_options"] = geometry
        geometry_note = _todict(geometry)

    pycolmap.match_exhaustive(spec["database_path"], **call_kwargs)

    database = pycolmap.Database.open(spec["database_path"])
    pairs = int(database.num_matched_image_pairs())
    verified = int(database.num_verified_image_pairs())
    inliers = int(database.num_inlier_matches())
    database.close()
    actual: dict[str, Any] = {"matching": _todict(matching), "pairing": _todict(pairing)}
    actual["two_view_geometry"] = geometry_note
    # Which parameters were passed, by name, so the record can be checked against
    # the call shape rather than trusted.
    actual["two_view_geometry_passed_as_argument"] = geometry is not None
    return {
        "num_matched_image_pairs": pairs,
        "num_verified_image_pairs": verified,
        "num_inlier_matches": inliers,
        "actual_options": actual,
    }


def _geometry_options(pycolmap: Any) -> Any | None:
    """The two-view geometry option object this build actually has, or ``None``.

    The prior comparison's records did not carry these values separately, so they
    are reported here as *this* version's defaults and explicitly marked as not
    having been recorded for the historical run.  Nothing is inferred backwards:
    "the old run did not record it" and "the old run used a different value" are
    different statements and only the first one is supported.
    """
    for name in ("TwoViewGeometryOptions", "TwoViewGeometryVerificationOptions"):
        if hasattr(pycolmap, name):
            try:
                return getattr(pycolmap, name)()
            except Exception:  # noqa: BLE001 - fall through to the next candidate
                continue
    return None


def _stage_mapping(pycolmap: Any, spec: dict) -> dict:
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = int(spec["num_threads"])
    options.random_seed = int(spec["random_seed"])
    options.multiple_models = bool(spec["multiple_models"])
    options.min_model_size = int(spec["min_model_size"])
    # The backend asserts ``max_num_models > 0`` and aborts the process otherwise,
    # so "unlimited" cannot be expressed as 0.  Distinct input count is an upper
    # bound on the number of components and is positive by construction.
    max_models = int(spec.get("max_num_models", 0))
    if max_models < 1:
        raise ValueError(
            "max_num_models must be >= 1: the backend aborts on 0, and 0 does not mean "
            "unlimited here"
        )
    options.max_num_models = max_models
    options.ba_refine_focal_length = True
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = spec.get("camera_model") != "SIMPLE_PINHOLE"
    options.image_names = list(spec["image_names"])

    models = pycolmap.incremental_mapping(
        spec["database_path"], spec["image_path"], spec["output_path"], options=options
    )
    summary = []
    for model_id, model in sorted(dict(models).items()):
        summary.append(
            {
                "model_id": int(model_id),
                "num_reg_images": int(model.num_reg_images()),
                "num_points3D": int(model.num_points3D()),
                "registered_image_ids": sorted(int(i) for i in model.reg_image_ids()),
            }
        )
    return {
        "model_count": len(summary),
        "models": summary,
        # Includes the dynamically chosen max_num_models and the explicit seed,
        # both of which earlier went unrecorded in the option summary.
        "actual_options": {"pipeline": _todict(options)},
    }


def _stage_export(pycolmap: Any, spec: dict) -> dict:
    """Save every component in binary *and* text, then read both back.

    Both forms are written and separately re-read because a reader may prefer the
    binary and mask an error that only the text form has; a single readback would
    not notice.
    """
    models_root = pathlib.Path(spec["models_root"])
    models_root.mkdir(parents=True, exist_ok=True)
    tolerance = float(spec.get("tolerance", 1e-9))
    from . import export as export_module  # imported here: export imports this module

    exported = []
    for source in sorted(pathlib.Path(spec["output_path"]).iterdir()):
        if not source.is_dir():
            continue
        model_id = source.name
        model_dir = models_root / model_id
        binary_dir = model_dir / "binary"
        text_dir = model_dir / "text"
        binary_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

        reconstruction = pycolmap.Reconstruction(str(source))
        source_description = export_module.describe_model(reconstruction)
        reconstruction.write(str(binary_dir))
        reconstruction.write_text(str(text_dir))

        from_binary = pycolmap.Reconstruction(str(binary_dir))
        from_text = pycolmap.Reconstruction()
        from_text.read_text(str(text_dir))
        binary_description = export_module.describe_model(from_binary)
        text_description = export_module.describe_model(from_text)

        # Source against each form separately.  Comparing binary with text alone
        # would accept a writer that dropped something from BOTH.
        source_vs_binary = export_module.compare_descriptions(
            source_description, binary_description, tolerance=tolerance
        )
        source_vs_text = export_module.compare_descriptions(
            source_description, text_description, tolerance=tolerance
        )

        descriptions_path = model_dir / "descriptions.json"
        descriptions_path.write_text(
            json.dumps(
                {
                    "test_data": "descriptions of the source model and of both native exports",
                    "source": source_description,
                    "binary": binary_description,
                    "text": text_description,
                },
                ensure_ascii=False, indent=2, default=str,
            ) + "\n",
            encoding="utf-8", newline="\n",
        )

        exported.append({
            "model_id": model_id,
            "binary_dir": str(binary_dir),
            "text_dir": str(text_dir),
            "descriptions_path": str(descriptions_path),
            "binary_files": sorted(p.name for p in binary_dir.iterdir()),
            "text_files": sorted(p.name for p in text_dir.iterdir()),
            "roundtrip_verified": not source_vs_binary and not source_vs_text,
            "source_vs_binary_differences": source_vs_binary[:20],
            "source_vs_text_differences": source_vs_text[:20],
            "counts": {
                "num_cameras": source_description["num_cameras"],
                "num_images": source_description["num_images"],
                "num_points3D": source_description["num_points3D"],
                "num_rigs": source_description["num_rigs"],
                "num_frames": source_description["num_frames"],
                "num_reg_images": source_description["num_reg_images"],
            },
        })
    return {
        "model_count": len(exported),
        "models": exported,
        "actual_options": {
            "<none>": (
                "this stage only reads and writes native models; it applies no extraction, "
                "matching or mapping options"
            )
        },
    }


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1:]))
