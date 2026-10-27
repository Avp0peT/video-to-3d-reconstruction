"""``python -m kozlov_lab.reconstruction`` — T01 canonical input to native sparse model.

Status and exit code are the contract's own table, and the two are computed from
the same facts:

===========================  ==========================================================  ====
status                       condition                                                   exit
===========================  ==========================================================  ====
``SUCCESS``                  exactly one usable component, every distinct input          0
                             registered, export and readback complete
``PARTIAL``                  at least one usable component, but unregistered images      1
                             or more than one component
``NO_MODEL``                 the backend finished normally and produced no usable        1
                             component
``INPUT_ERROR``              upstream, parameters, paths or counts do not conform        2
``BACKEND_UNAVAILABLE``      the pinned backend cannot load or lacks a capability         3
``BACKEND_ERROR``            backend exception, real failure exit, timeout, or a         4
                             model that does not round-trip
===========================  ==========================================================  ====

Three properties this module deliberately maintains:

* **one shared budget.**  A single monotonic deadline covers the whole native
  chain; each stage receives only the remainder and no stage starts once it is
  gone.  Per-stage timeouts would multiply the budget by the number of stages.
* **identity survives failure.**  Every upstream row's mapping record is created
  as soon as the input is verified, so a later backend failure still reports who
  the inputs were instead of an empty map.
* **the recorded configuration is read from the version, not summarised.**  The
  option objects actually passed to 4.2.1 are serialized with its own ``todict``
  and bound into the fingerprint; a stage that never ran records that it did not.
"""

from __future__ import annotations

import argparse
import math
import os
import json
import pathlib
import sys
import time
import uuid

from . import export as export_module
from . import profiles
from .colmap_adapter import (
    BACKEND_NAME,
    BACKEND_PIN,
    CONTRACT_VERSION,
    COORDINATE_CONVENTION,
    MAX_DISTINCT_IMAGES,
    MAX_LONG_SIDE,
    MAX_SIFT_FEATURES,
    MULTIPLE_MODELS,
    NUM_THREADS,
    STAGES,
    BackendError,
    BackendUnavailable,
    InputError,
    UpstreamRun,
    backend_info,
    backend_probe_command,
    backend_probe_context,
    canonical_digest,
    diff_snapshots,
    dumps_strict,
    effective_options,
    implementation_sha256,
    input_fingerprint,
    load_upstream_run,
    require_pinned_version,
    run_native_stage,
    snapshot_upstream_files,
    stage_inputs,
    sha256_file,
    verify_foreground_masks,
)

__all__ = ["EXIT_CODES", "build_parser", "main", "run_reconstruction"]

EXIT_CODES = {
    "SUCCESS": 0,
    "PARTIAL": 1,
    "NO_MODEL": 1,
    "INPUT_ERROR": 2,
    "BACKEND_UNAVAILABLE": 3,
    "BACKEND_ERROR": 4,
}

#: Where the contract requires run output to live.
OUTPUT_PARENT = "outputs/delegation/T02A"

#: Total wall-clock budget for the native chain, shared by all four stages.
TOTAL_BUDGET_SECONDS = 600

#: Numeric tolerance for model comparison, taken from the project's own scale.
MODEL_TOLERANCE = 1e-9


class CliError(Exception):
    """A parameter problem that must exit 2 without writing a run."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kozlov_lab.reconstruction",
        description=f"Native sparse reconstruction from a T01 run ({CONTRACT_VERSION}).",
    )
    parser.add_argument("--io-run", required=True,
                        help="a complete photo-io-v1 run directory with status SUCCESS")
    parser.add_argument("--output", required=True,
                        help=f"artifact base directory; must be inside the project {OUTPUT_PARENT}/")
    parser.add_argument("--profile", default=profiles.DEFAULT_PROFILE,
                        help=("configuration profile; " + " | ".join(profiles.VALID_PROFILES)
                              + f" (default {profiles.DEFAULT_PROFILE})"))
    parser.add_argument('--object-mask-manifest',help='identity-bound SAM3 object-mask-v1 manifest; only verified selected foreground views enter SfM')
    parser.add_argument('--point-selection',default='native',help='native or rcover-v1; rcover requires cad-object-v1 and selected object masks')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        project_root = _project_root()
        output_base = _resolve_output_base(args.output, project_root)
        run_dir = output_base / _make_run_id()
        run_dir.mkdir(parents=True, exist_ok=False)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot create the run directory: {exc}", file=sys.stderr)
        return 2

    return run_reconstruction(
        io_run=args.io_run, run_dir=run_dir, project_root=project_root, profile=args.profile,
        object_mask_manifest=args.object_mask_manifest, point_selection=args.point_selection
    )


def run_reconstruction(*, io_run: str, run_dir: pathlib.Path, project_root: pathlib.Path,
                       profile: str = profiles.DEFAULT_PROFILE, object_mask_manifest=None,
                       point_selection: str = 'native') -> int:
    """Perform one run inside an already-created ``run_dir``.

    ``profile`` selects the versioned configuration.  An unknown value is not
    rejected by argparse, so that it still reaches the normal error path and
    produces a written INPUT_ERROR run instead of only a usage message.

    The selector is therefore validated *here*, at the top of the function and
    inside its own try, rather than as an argument to ``effective_options``.  A
    ``ProfileError`` raised while building the record's initial ``effective_options``
    would happen before the ``try`` below, so it would escape the handler entirely:
    no ``run.json``, no ``frames_map.jsonl``, and a traceback instead of the
    contract's INPUT_ERROR/exit 2.  Deriving the version from the validated name
    keeps "the profile is legal" a fact established before anything else reads it.
    """
    print(f"run {run_dir.name} (profile {profile})")

    try:
        profiles.require_valid_profile(profile)
    except profiles.ProfileError as exc:
        records = _base_record(run_dir, project_root, profile, contract_version=CONTRACT_VERSION)
        records["status"] = "INPUT_ERROR"
        records["errors"].append({"code": "INPUT_ERROR", "message": str(exc)})
        records["selector_note"] = (
            "the profile selector was rejected before any upstream file was read and before "
            "any backend process was started"
        )
        _write_frames_map(run_dir / "frames_map.jsonl", [])
        (run_dir / "run.json").write_text(
            dumps_strict(records, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        exit_code = EXIT_CODES["INPUT_ERROR"]
        print(f"status={records['status']} models=0 exit={exit_code}")
        print(f"artifacts: {run_dir}")
        return exit_code

    shared = profiles.is_shared(profile)
    records = _base_record(
        run_dir, project_root, profile,
        contract_version=profiles.contract_version(profile),
    )
    frames_map: list[dict] = []

    try:
        _execute(records, frames_map, io_run=io_run, run_dir=run_dir, project_root=project_root,
                 profile=profile, shared=shared, object_mask_manifest=object_mask_manifest,
                 point_selection=point_selection)
    except InputError as exc:
        records["status"] = "INPUT_ERROR"
        records["errors"].append({"code": exc.code, "message": exc.message})
    except BackendUnavailable as exc:
        records["status"] = "BACKEND_UNAVAILABLE"
        records["errors"].append({"code": "BACKEND_UNAVAILABLE", "message": str(exc)})
    except BackendError as exc:
        records["status"] = "BACKEND_ERROR"
        records["errors"].append({"code": "BACKEND_ERROR", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - recorded, never presented as success
        records["status"] = "BACKEND_ERROR"
        records["errors"].append(
            {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"}
        )

    # Both artifacts are written on every path, including failures, so a failed
    # run still reports which inputs it had.
    _write_frames_map(run_dir / "frames_map.jsonl", frames_map)
    (run_dir / "run.json").write_text(
        dumps_strict(records, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    exit_code = EXIT_CODES[records["status"]]
    print(f"status={records['status']} models={records['counts']['model_count']} "
          f"exit={exit_code}")
    print(f"artifacts: {run_dir}")
    return exit_code


def _base_record(run_dir: pathlib.Path, project_root: pathlib.Path, profile: str, *,
                 contract_version: str) -> dict:
    """The record's fixed fields, built without ever calling the selector.

    Deliberately tolerant of an illegal ``profile``: it is called on the rejected-
    selector path too, where ``effective_options(profile)`` would raise again and
    take the evidence with it.  ``contract_version`` is passed in by the caller,
    which has already validated the selector when a profile-specific value exists.
    """
    profile_is_valid = profile in profiles.VALID_PROFILES
    try:
        options = effective_options(profile)
    except profiles.ProfileError:
        # Unreachable for a validated profile; kept so a bad selector still yields
        # a record rather than a second exception.
        options = {"<rejected>": "the profile selector was not accepted, so no options were built"}

    records: dict = {
        "contract_version": contract_version,
        "run_id": run_dir.name,
        "upstream_io_run_id": None,
        "data_kind": None,
        "status": "INPUT_ERROR",
        "backend": {"name": BACKEND_NAME, "version": None, "runtime": "cpu"},
        "effective_options": options,
        "input_fingerprint": None,
        "input_fingerprint_before_native": None,
        "implementation_sha256": implementation_sha256(project_root),
        "runtime_budget_seconds": TOTAL_BUDGET_SECONDS,
        "counts": {
            "input_rows": 0,
            "unique_images": 0,
            "registered_unique_images": 0,
            "unregistered_unique_images": 0,
            "model_count": 0,
        },
        "models": [],
        "stages": [],
        "errors": [],
        "cache_hit": False,
        "coordinate_convention": COORDINATE_CONVENTION,
        "scale_status": "unknown",
        "length_unit": None,
        "reconstruction_scope": "scene_sparse",
        "mask_usage": "not_applied",
        "quality_status": "NOT_EVALUATED",
    }
    if profile_is_valid and profiles.is_shared(profile):
        records["profile_id"] = profile
    return records


# --------------------------------------------------------------------------- #
# the actual pipeline
# --------------------------------------------------------------------------- #
def _execute(records, frames_map, *, io_run: str, run_dir: pathlib.Path,
             project_root: pathlib.Path, profile: str = profiles.DEFAULT_PROFILE,
             shared: bool | None = None, object_mask_manifest=None, point_selection='native') -> None:
    """Run the pipeline.  The selector has already been validated by the caller.

    ``require_valid_profile`` is *not* re-run here: doing so inside this function,
    after ``records`` was built, was exactly the shape of the earlier defect — the
    check had to happen before anything derived from the selector was used.  The
    caller now validates once, up front, and passes the resulting ``shared`` flag.
    """
    if shared is None:
        shared = profiles.is_shared(profile)
    records["profile"] = profiles.summary(profile)
    records["pipeline_changes"] = "none" if not shared else "camera_sharing: per_image_independent -> single"
    selection_sources=[]
    if point_selection != 'native':
        from .point_selection import selection_policy
        try: policy=selection_policy(point_selection, profile=profile)
        except (ValueError,TypeError) as exc: raise InputError(str(exc)) from exc
        if profile not in (profiles.PROFILE_CAD_OBJECT_V1, profiles.PROFILE_CAD_OBJECT_VIDEO_V1, profiles.PROFILE_CAD_OBJECT_PINHOLE_V1) or object_mask_manifest is None:
            raise InputError('rcover point selection requires cad-object-v1 and object-mask-manifest')
        source_root=pathlib.Path(__file__).resolve().parents[3]
        selection_sources=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256_file(p)) for p in
            (pathlib.Path(__file__).with_name('point_selection.py'),source_root/'src/kozlov_lab/stereo/point_coverage.py')]
        records['effective_options']['point_selection']=dict(policy=policy,implementation_bindings=selection_sources)
        records['point_selection']=dict(requested=point_selection,status='NOT_RUN',geometry_acceptance=False)
    if profile in (profiles.PROFILE_CAD_OBJECT_V1, profiles.PROFILE_CAD_OBJECT_VIDEO_V1, profiles.PROFILE_CAD_OBJECT_PINHOLE_V1):
        runtime = {key: os.environ.get(key) for key in profiles.RUNTIME_THREAD_ENV}
        records["runtime_environment"] = runtime
        records["pipeline_changes"] = "shared camera; explicit runtime threads=2; max_distinct_images=" + str(profiles.effective_profile_fields(profile)["max_distinct_images"])
        if any(value != "2" for value in runtime.values()):
            raise InputError("cad-object-v1 requires numeric runtime thread environment=2 before "
                             "Python startup; use scripts/reconstruct_object.py")

    deadline = time.monotonic() + TOTAL_BUDGET_SECONDS
    records["native_deadline_monotonic_start"] = deadline

    upstream: UpstreamRun = load_upstream_run(pathlib.Path(io_run))
    records["upstream_io_run_id"] = upstream.run_id
    records["data_kind"] = upstream.data_kind
    # Preserve original input identity even when the object manifest is rejected.
    records['counts']['input_rows']=len(upstream.rows)
    _init_frames_map(upstream,upstream,frames_map)
    records['foreground']=dict(requested=object_mask_manifest is not None,manifest_sha256=None,
        ledger_path=None,coverage=None,mask_consumption_verified=False,extraction_audit=None)
    foreground=None; source_mask_snapshot={}
    if object_mask_manifest is not None:
        records['foreground']['contract_version']='sfm-foreground-v1'
        records['reconstruction_scope']='foreground_requested_not_verified'
        records['mask_usage']='foreground_requested_not_verified'
        records['effective_options']['foreground']=dict(requested=True)
        from .foreground import select_foreground
        try:
            foreground=select_foreground(upstream,pathlib.Path(object_mask_manifest),run_dir/'foreground', **({'profile':profile} if profile in (profiles.PROFILE_CAD_OBJECT_VIDEO_V1, profiles.PROFILE_CAD_OBJECT_PINHOLE_V1) else {}))
        except (ValueError,OSError,KeyError) as exc:
            raise InputError('object mask manifest input rejected: '+str(exc)) from exc
        source_root=pathlib.Path(__file__).resolve().parents[3]
        implementation_paths=[pathlib.Path(__file__).with_name('foreground.py'),source_root/'integrations/mask_review/review.py']
        foreground_implementation=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha256_file(p)) for p in implementation_paths]
        selected=set(foreground['selected_indices'])
        if selected!=set(foreground['masks_by_index']):raise InputError('foreground selected mask coverage differs')
        source_mask_snapshot={str(pathlib.Path(p).resolve()):sha256_file(pathlib.Path(p)) for p in foreground['masks_by_index'].values()}
        records['foreground'].update(manifest_sha256=foreground['manifest_sha256'],ledger_path=str(foreground['ledger_path']),coverage=foreground['coverage'])
        records['effective_options']['foreground'].update(manifest_sha256=foreground['manifest_sha256'],selected_indices=sorted(selected),
            source_mask_bindings=source_mask_snapshot,implementation_bindings=foreground_implementation)
        staged=stage_inputs(upstream,run_dir,profile=profile,masks_by_index=foreground['masks_by_index'])
    else:
        staged = (stage_inputs(upstream, run_dir, profile=profile)
                  if profile in (profiles.PROFILE_CAD_OBJECT_V1, profiles.PROFILE_CAD_OBJECT_VIDEO_V1, profiles.PROFILE_CAD_OBJECT_PINHOLE_V1) else stage_inputs(upstream, run_dir))
    _publish_input_facts(records, frames_map, upstream, staged)
    if foreground is not None:
        records['counts']['foreground_selected_rows']=len(selected)
        records['counts']['foreground_excluded_rows']=len(upstream.rows)-len(selected)
        records['counts']['original_unique_images']=len({r.canonical_image_sha256 for r in staged.rows})
        records['effective_options']['foreground'].update(mask_bindings=staged.mask_bindings,
            mask_root=str(staged.masks_dir.resolve()),mask_naming_rule="image_name + '.png'")
        if len(staged.distinct)<2:raise InputError('foreground requires at least two distinct selected masked images')

    # Shared mode has one camera for every image, so images of different sizes
    # cannot share it.  This is rejected before any native stage runs, and it is
    # deliberately not "fixed" by resampling: silently resizing to make the profile
    # applicable would change the input the contract froze.
    #
    # The identity map and the counters are published *first*.  Rejecting before
    # they were set left a written run claiming ``unique_images=3`` next to
    # ``registered+unregistered=0`` with an empty map -- a self-contradictory pair
    # of numbers, and no way to tell which inputs were involved.  The rejection
    # itself is unchanged; only its position relative to the evidence moved.
    if shared:
        sizes = sorted({(row.canonical_width, row.canonical_height) for row in staged.distinct})
        if len(sizes) > 1:
            raise InputError(
                f"profile {profile} requires all canonical images to have the same "
                f"size; found {sizes} across {len(staged.distinct)} distinct image(s). "
                "Resampling to fit is not permitted."
            )

    snapshot_before = snapshot_upstream_files(upstream)
    records["upstream_file_snapshot_sha256"] = canonical_digest(snapshot_before)

    # The version probe is a process too, and it is the *first* one -- so its start
    # event is written before it is launched.  Nothing about it is inferred from
    # the fact that a later stage succeeded.
    #
    # The evidence location travels in ``backend_probe_context`` rather than as
    # arguments: this call stays positionally compatible with every existing caller
    # and with the stubs the pre-existing regressions install (``lambda _e: ...``),
    # and a module-level dict survives those stubs replacing the function itself.
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    backend_records = logs_dir / "backend_process.jsonl"
    backend_probe_context.clear()
    backend_probe_context.update({
        "evidence_path": str(backend_records),
        "project_root": str(project_root),
        "package_root": str(pathlib.Path(__file__).resolve().parent),
    })
    try:
        backend = backend_info(sys.executable)
    finally:
        backend_probe_context.clear()
    require_pinned_version(backend)
    records["backend"] = backend
    records["backend_probe"] = {
        "argv": backend_probe_command(sys.executable),
        "cwd": str(project_root),
        "interpreter": sys.executable,
        "evidence": str(backend_records.relative_to(run_dir)),
        "evidence_written": backend_records.is_file(),
    }
    _persist_intermediate(records, run_dir)

    options = records["effective_options"]
    implementation = records["implementation_sha256"]
    records["input_fingerprint_before_native"] = input_fingerprint(
        upstream, staged.rows, options, backend, implementation,
        contract_version=records["contract_version"],
    )
    _persist_intermediate(records, run_dir)

    options_for_worker = {
        "num_threads": NUM_THREADS,
        "max_long_side": MAX_LONG_SIDE,
        "max_sift_features": MAX_SIFT_FEATURES,
        "random_seed": 0,
        "camera_model": options["camera_model"],
        "camera_mode": profiles.camera_mode(profile),
        "focal_length_factor": 1.2,
        "multiple_models": MULTIPLE_MODELS,
        "min_model_size": 2,
        # Upper bound on components; a positive value is required because the
        # backend asserts max_num_models > 0 and aborts on 0.
        "max_num_models": len(staged.distinct),
    }
    if profile in (profiles.PROFILE_CAD_OBJECT_VIDEO_V1, profiles.PROFILE_CAD_OBJECT_PINHOLE_V1):
        options_for_worker["reconstruction_profile"] = profile
    image_names = [row.staged_image_name for row in staged.distinct]

    database_path = run_dir / "database.db"
    models_output = run_dir / "native_models"
    models_root = run_dir / "models"
    models_output.mkdir(parents=True, exist_ok=True)

    specs = {
        "extract_features": {
            "database_path": str(database_path), "image_path": str(run_dir),
            "image_names": image_names, **options_for_worker,
        },
        "match_exhaustive": {
            "database_path": str(database_path), **options_for_worker,
        },
        "incremental_mapping": {
            "database_path": str(database_path), "image_path": str(run_dir),
            "output_path": str(models_output), "image_names": image_names,
            **options_for_worker,
        },
        "export_models": {
            "output_path": str(models_output), "models_root": str(models_root),
            "tolerance": MODEL_TOLERANCE,
        },
    }
    if foreground is not None:
        specs['extract_features'].update(foreground_mask_path=str(staged.masks_dir.resolve()),
            foreground_mask_bindings=staged.mask_bindings,foreground_manifest_sha256=foreground['manifest_sha256'])
    if point_selection != 'native':
        specs['extract_features']['point_selection']=point_selection

    stage_results: dict[str, dict] = {}
    for stage in STAGES:
        outcome = run_native_stage(
            stage, specs[stage], run_dir=run_dir, logs_dir=logs_dir,
            project_root=project_root, deadline=deadline,
        )
        records["stages"].append({
            "name": stage,
            "started_utc": outcome.started_utc,
            "finished_utc": outcome.finished_utc,
            "status": "OK" if outcome.error is None else "FAILED",
            "started": outcome.started,
            "timeout_seconds": outcome.timeout_seconds,
            "log_stdout": outcome.stdout_path,
            "log_stderr": outcome.stderr_path,
            "launch_record_path": outcome.launch_record_path,
            "process_exit_code": outcome.returncode,
            "process_exit_code_reason": outcome.returncode_reason,
            "timed_out": outcome.timed_out,
            "exception": outcome.error,
            "note": "process exit code belongs to the native subprocess, not to an API call",
        })
        stage_results[stage] = outcome.result or {}
        _record_actual_options(records, stage, outcome.result, started=outcome.started)
        # Persist before acting on the outcome, so a kill during the *next* stage
        # still finds this one recorded.
        _persist_intermediate(records, run_dir)

        if outcome.error is not None:
            raise BackendError(f"stage {stage} failed: {outcome.error}")

        if stage == "extract_features":
            _check_extraction(stage_results[stage], staged, profile)
            if point_selection != 'native':
                selected_audit=stage_results[stage].get('point_selection_audit') or {}
                if (selected_audit.get('status')!='VERIFIED' or selected_audit.get('policy_id')!=point_selection
                        or selected_audit.get('after_total')!=stage_results[stage].get('num_keypoints_total')
                        or selected_audit.get('coordinates_modified') is not False
                        or selected_audit.get('descriptors_preserved') is not True):
                    raise BackendError('point selection missing verified database row/descriptor evidence')
                selection_file=run_dir/'point_selection.json'
                if not selection_file.is_file():raise BackendError('point selection evidence file missing')
                persisted=json.loads(selection_file.read_text(encoding='utf-8'))
                if persisted!=selected_audit:raise BackendError('point selection evidence differs from worker result')
                records['point_selection']=dict(selected_audit,evidence_sha256=sha256_file(selection_file))
            if foreground is not None:
                audit=stage_results[stage].get('foreground_mask_audit')
                reader=(stage_results[stage].get('actual_options') or {}).get('reader') or {}
                if pathlib.Path(str(reader.get('mask_path',''))).resolve()!=staged.masks_dir.resolve() or not audit or audit.get('status')!='VERIFIED' or audit.get('violations')!=0 or audit.get('checked_images')!=len(staged.distinct) or audit.get('checked_keypoints')!=stage_results[stage].get('num_keypoints_total'):
                    raise BackendError('foreground extraction lacks actual reader binding and verified DB consumption')
                records['foreground'].update(mask_consumption_verified=True,extraction_audit=audit)
                records['reconstruction_scope']='foreground_feature_centres'
                records['mask_usage']='foreground_verified'
            # Preserve real database identities even if a later stage fails.
            _fill_registrations(upstream, staged, stage_results, frames_map)
        elif stage == "export_models":
            _verified_models = _verify_export(records, stage_results[stage], run_dir,
                                               profile)

    # Input stability is established by comparing two *disk* snapshots, not by
    # recomputing the fingerprint: a fingerprint recomputed from staged state
    # would agree with itself no matter what happened on disk.  It is checked at
    # the end so that every original row's file -- masks and duplicate aliases
    # included -- participates.
    snapshot_after = snapshot_upstream_files(upstream)
    for ref in selection_sources:
        path=pathlib.Path(ref['path'])
        if path.stat().st_size!=ref['bytes'] or sha256_file(path)!=ref['sha256']:
            raise InputError('point selection implementation changed during native run')
    changes = diff_snapshots(snapshot_before, snapshot_after)
    if changes:
        raise InputError("upstream inputs changed during the run: " + "; ".join(changes))
    if foreground is not None:
        verify_foreground_masks(staged.mask_bindings)
        if sha256_file(pathlib.Path(object_mask_manifest))!=foreground['manifest_sha256']:
            raise InputError('object mask manifest changed during native run')
        if any(sha256_file(pathlib.Path(p))!=digest for p,digest in source_mask_snapshot.items()):
            raise InputError('selected object mask source changed during native run')
        if any(p.stat().st_size!=b['bytes'] or sha256_file(p)!=b['sha256'] for p,b in zip(implementation_paths,foreground_implementation)):
            raise InputError('foreground implementation changed during native run')

    _fill_registrations(upstream, staged, stage_results, frames_map)
    models = records["models"]
    usable_models = [model for model in models if model["usable"]]
    registered_unique = sum(1 for row in frames_map
                            if row["representative_input_index"] == row["input_index"]
                            and row["registrations"])
    records["counts"]["registered_unique_images"] = registered_unique
    records["counts"]["unregistered_unique_images"] = len(staged.distinct) - registered_unique
    records["counts"]["model_count"] = len(models)

    # The final fingerprint uses the configuration that was actually completed,
    # so the recorded options and the fingerprint describe the same facts.
    records["input_fingerprint"] = input_fingerprint(
        upstream, staged.rows, options, backend, implementation,
        contract_version=records["contract_version"],
    )

    if not usable_models:
        records["status"] = "NO_MODEL"
        records["errors"].append({
            "code": "NO_MODEL",
            "message": "the backend finished without producing a usable component; this is "
                       "not a successful reconstruction",
        })
    elif len(usable_models) == 1 and registered_unique == len(staged.distinct):
        records["status"] = "SUCCESS"
    else:
        records["status"] = "PARTIAL"
        reasons = []
        if len(usable_models) != 1:
            reasons.append(f"{len(usable_models)} usable components")
        if registered_unique != len(staged.distinct):
            reasons.append(f"{len(staged.distinct) - registered_unique} unregistered image(s)")
        records["errors"].append({"code": "PARTIAL", "message": "; ".join(reasons)})


def _record_actual_options(records, stage: str, result: dict | None, *, started: bool) -> None:
    """Bind the stage's real configuration, or say plainly that it was not obtained."""
    native = records["effective_options"].setdefault("native", {})
    actual = (result or {}).get("actual_options")
    if isinstance(actual, dict) and actual:
        native[stage] = actual
    elif started:
        native[stage] = {"<not_reported>": "the stage ran but did not report its configuration"}
    else:
        native[stage] = {"<not_executed>": "the stage did not start, so it has no configuration"}


def _check_extraction(extract_result: dict, staged, profile: str) -> None:
    """Confirm the backend saw exactly the staged pixels.

    If the extractor had silently downscaled an image, the camera dimensions in
    the database would differ from the staged PNG; that would quietly change every
    subsequent pixel measurement, so it is checked rather than trusted.
    """
    cameras = extract_result.get("cameras") or {}
    expected = {row.staged_image_name: (row.canonical_width, row.canonical_height)
                for row in staged.distinct}
    sizes = sorted((camera["width"], camera["height"]) for camera in cameras.values())
    for width, height in sorted(set(expected.values())):
        if (width, height) not in sizes:
            raise BackendError(
                f"the backend recorded no camera of the staged size {width}x{height}; "
                f"recorded sizes were {sizes}, which suggests a silent rescale"
            )
    if int(extract_result.get("num_images", 0)) != len(staged.distinct):
        raise BackendError(
            f"the backend registered {extract_result.get('num_images')} images for "
            f"{len(staged.distinct)} staged inputs"
        )
    names_to_ids = extract_result.get("image_names_to_ids") or {}
    missing = [name for name in expected if name not in names_to_ids]
    if missing:
        raise BackendError(f"the database has no image record for {missing}")

    if profiles.is_shared(profile):
        _check_shared_camera(extract_result, staged, expected, profile=profile)


def _check_shared_camera(extract_result: dict, staged, expected: dict, *,
                         profile: str = profiles.PROFILE_CAD02A_SHARED_V1) -> None:
    """Verify a shared run *actually* produced one camera, not merely requested it.

    The requested option is not evidence: a backend that silently ignored the mode
    would still leave a run that looked configured for sharing.  What is checked is
    the database's own content — one camera record, every image referencing it, and
    the initial intrinsics required by the profile and staged image dimensions.
    """
    cameras = extract_result.get("cameras") or {}
    if int(extract_result.get("num_cameras", len(cameras))) != 1 or len(cameras) != 1:
        raise BackendError(
            f"profile {profile} requires exactly one camera in the database, "
            f"found {len(cameras)}: {sorted(cameras)}"
        )
    camera_id = next(iter(cameras))
    camera = cameras[camera_id]

    expected_model = effective_options(profile)["camera_model"]
    if camera.get("model_name") != expected_model:
        raise BackendError(
            f"shared camera is {camera.get('model_name')!r}, expected {expected_model}"
        )
    expected_size = (1024, 683)
    expected_params = SHARED_EXPECTED_PARAMS
    if profile in (profiles.PROFILE_CAD_OBJECT_V1, profiles.PROFILE_CAD_OBJECT_VIDEO_V1, profiles.PROFILE_CAD_OBJECT_PINHOLE_V1):
        staged_sizes = set(expected.values())
        if len(staged_sizes) != 1:
            raise BackendError("profile cad-object-v1 requires exactly one staged image size")
        expected_size = next(iter(staged_sizes))
        width, height = expected_size
        expected_params = [1.2 * max(width, height), width / 2.0, height / 2.0]
        if expected_model == "SIMPLE_RADIAL":
            expected_params.append(0.0)
    if (camera.get("width"), camera.get("height")) != expected_size:
        raise BackendError(
            f"shared camera is {camera.get('width')}x{camera.get('height')}, "
            f"expected {expected_size[0]}x{expected_size[1]}"
        )

    # ``has_prior_focal_length`` distinguishes "we supplied a focal length" from
    # "the backend guessed one".  The profile's intrinsics are a heuristic starting
    # point, so a camera that claims a prior focal length is reporting a different
    # fact about itself than the one this run intends; that is refused rather than
    # smoothed over by looking only at the numbers.
    if camera.get("has_prior_focal_length") is not False:
        raise BackendError(
            "shared camera reports has_prior_focal_length="
            f"{camera.get('has_prior_focal_length')!r}; profile {profile} supplies a "
            "heuristic initial guess, not a calibration, so no prior focal length is expected"
        )

    _check_shared_intrinsics(camera, expected_params)
    _check_shared_image_identity(extract_result, expected, camera_id)


#: The initial intrinsics a shared SIMPLE_RADIAL camera must have been created with
#: for the 1024x683 inputs of the frozen cad02a-shared-v1 experiment.
SHARED_EXPECTED_PARAMS = [1.2 * 1024, 512.0, 341.5, 0.0]
SHARED_PARAM_TOLERANCE = 1e-6


def _check_shared_intrinsics(camera: dict, expected_params: list[float] | None = None) -> None:
    """Compare the *recorded* parameters against the required initial values.

    ``NaN`` is the reason this cannot be a plain ``abs(actual - wanted) > tol``
    test.  Every comparison involving NaN is false, so ``abs(nan - wanted) > tol``
    is false too and a camera carrying NaN would have passed the old check -- and
    NaN is precisely the value that means "the intrinsics are not usable".  The
    finiteness gate below is therefore load-bearing, not decorative, and ``±Inf``
    (which the tolerance does catch, but only by accident of magnitude) is refused
    through the same explicit path.
    """
    if expected_params is None:
        expected_params = SHARED_EXPECTED_PARAMS
    raw = camera.get("params")
    if not isinstance(raw, (list, tuple)):
        raise BackendError(f"shared camera reports no parameter vector: {raw!r}")

    params: list[float] = []
    for index, value in enumerate(raw):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise BackendError(
                f"shared camera parameter {index} is not a number: {value!r}"
            ) from exc
        if not math.isfinite(number):
            raise BackendError(
                f"shared camera parameter {index} is {value!r}, which is not finite; the "
                "initial intrinsics must be finite numbers"
            )
        params.append(number)

    if len(params) != len(expected_params):
        raise BackendError(
            f"shared camera has {len(params)} parameter(s) ({params}), expected "
            f"{len(expected_params)} ({expected_params})"
        )
    for index, (actual, wanted) in enumerate(zip(params, expected_params)):
        if abs(actual - wanted) > SHARED_PARAM_TOLERANCE:
            raise BackendError(
                f"shared camera parameter {index} is {actual}, expected {wanted}; the "
                f"initial intrinsics are {expected_params}, which is a heuristic "
                "starting point and not a calibration"
            )


def _check_shared_image_identity(extract_result: dict, expected: dict, camera_id) -> None:
    """Walk names -> real database image ids -> camera id, as three separate facts.

    The previous version compared only ``len(image_camera_ids)`` and the camera
    value, so a database whose *names* were wrong — a renamed set, an extra image,
    a missing one — passed as long as the count and the camera value happened to
    match.  Count equality is not identity equality, so each link is checked
    against the set the run actually staged:

    * the name set must equal the staged name set exactly (no extra, no missing);
    * the names must be unique, and the database image ids must be unique and agree
      with ``image_names_to_ids`` — ids are read back from the database and never
      assumed contiguous or inferred from order;
    * every image must reference the one camera, and that camera must exist.
    """
    names_to_ids_raw = extract_result.get("image_names_to_ids")
    image_camera_ids = extract_result.get("image_camera_ids")
    if not isinstance(names_to_ids_raw, dict) or not isinstance(image_camera_ids, dict):
        raise BackendError(
            "the extraction result does not carry a usable names->ids map and image->camera "
            f"map (names->ids {type(names_to_ids_raw).__name__}, "
            f"image->camera {type(image_camera_ids).__name__})"
        )

    staged_names = set(expected)
    reported_names = set(image_camera_ids)

    missing = sorted(staged_names - reported_names)
    unexpected = sorted(reported_names - staged_names)
    if missing or unexpected:
        raise BackendError(
            "the shared camera's image set does not match the staged inputs; "
            f"missing from the database: {missing}; not staged but present: {unexpected}"
        )

    if len(names_to_ids_raw) != len(staged_names):
        raise BackendError(
            f"the database reports {len(names_to_ids_raw)} image records for "
            f"{len(staged_names)} staged inputs; the names must correspond one to one"
        )

    image_ids = [int(value) for value in names_to_ids_raw.values()]
    if len(set(image_ids)) != len(image_ids):
        raise BackendError(
            f"the database reports duplicate image ids: {sorted(image_ids)}"
        )

    # Names are the keys of a JSON object, so a duplicate name cannot even be
    # expressed there; what can be checked is that the *staged* names are unique,
    # which they are by construction, and that the two maps key the same set.
    if set(names_to_ids_raw) != staged_names:
        raise BackendError(
            "the names->ids and image->camera maps key different image sets: "
            f"{sorted(set(names_to_ids_raw) ^ staged_names)}"
        )

    wrong: dict[str, int] = {}
    for name, raw_camera in sorted(image_camera_ids.items()):
        try:
            referenced = int(raw_camera)
        except (TypeError, ValueError) as exc:
            raise BackendError(
                f"image {name!r} references a non-numeric camera {raw_camera!r}"
            ) from exc
        if referenced != int(camera_id):
            wrong[name] = referenced
    if wrong:
        raise BackendError(
            f"every image must reference shared camera {camera_id}; these do not: {wrong}"
        )

    # A reference is not existence: the shared id must be the id of a real camera
    # record, which is what makes the chain names -> image id -> camera id -> camera
    # closed rather than three numbers that happen to look consistent.
    if int(camera_id) not in {int(value) for value in extract_result.get("cameras") or {}}:
        raise BackendError(
            f"the shared camera id {camera_id} has no camera record in the database"
        )


def _verify_export(records, export_result: dict, run_dir: pathlib.Path,
                   profile: str = profiles.DEFAULT_PROFILE) -> list[dict]:
    """Parent-side re-verification of what the worker wrote.

    The worker compares the source against both exports; this function checks the
    worker's verdict *and* re-reads the written files here, because a parent that
    ignored ``roundtrip_verified`` would accept a model the worker had already
    rejected.
    """
    import json as _json

    models: list[dict] = []
    for entry in export_result.get("models", []):
        model_id = str(entry["model_id"])
        binary_dir = pathlib.Path(entry["binary_dir"])
        text_dir = pathlib.Path(entry["text_dir"])

        if not entry.get("roundtrip_verified", False):
            raise BackendError(
                f"model {model_id} failed the worker's source-preservation check: "
                f"source-vs-binary {entry.get('source_vs_binary_differences')}, "
                f"source-vs-text {entry.get('source_vs_text_differences')}"
            )

        descriptions_path = pathlib.Path(entry["descriptions_path"])
        if not descriptions_path.is_file():
            raise BackendError(f"model {model_id} wrote no descriptions document")
        worker_descriptions = _json.loads(descriptions_path.read_text(encoding="utf-8"))

        from_binary = export_module.read_binary_model(binary_dir)
        from_text = export_module.read_text_model(text_dir)
        parent_binary = export_module.describe_model(from_binary)
        parent_text = export_module.describe_model(from_text)

        if profiles.is_shared(profile):
            _check_model_shared_camera(from_binary, model_id)

        source_description = worker_descriptions.get("source")
        if not isinstance(source_description, dict):
            raise BackendError(f"model {model_id}: descriptions document has no source model")

        # Source against each export, verified independently in this process.
        for label, parent_description in (("binary", parent_binary), ("text", parent_text)):
            differences = export_module.compare_descriptions(
                source_description, parent_description, tolerance=MODEL_TOLERANCE
            )
            if differences:
                raise BackendError(
                    f"model {model_id} does not match its {label} export in the parent's "
                    f"own comparison: {differences[:5]}"
                )

        usable, reasons = export_module.is_usable(from_binary)
        models.append({
            "model_id": model_id,
            "binary_path": str(binary_dir.relative_to(run_dir).as_posix()),
            "text_path": str(text_dir.relative_to(run_dir).as_posix()),
            "descriptions_path": str(descriptions_path.relative_to(run_dir).as_posix()),
            "registered_image_ids": sorted(int(i) for i in from_binary.reg_image_ids()),
            "num_registered_images": int(from_binary.num_reg_images()),
            "num_points3D": int(from_binary.num_points3D()),
            "num_rigs": int(from_binary.num_rigs()),
            "num_frames": int(from_binary.num_frames()),
            "roundtrip_verified": True,
            "source_vs_binary_differences": [],
            "source_vs_text_differences": [],
            "usable": usable,
            "not_usable_reasons": reasons,
        })
    records["models"] = models
    return models


def _check_model_shared_camera(reconstruction, model_id: str) -> None:
    """Every registered image of a shared-profile model must use one valid camera.

    Camera *parameters* may differ between models — each model is optimised
    separately — so only identity is checked here.  A shared run must never be able
    to report models whose images came from several cameras.
    """
    camera_ids = sorted({int(reconstruction.image(int(image_id)).camera_id)
                         for image_id in reconstruction.reg_image_ids()})
    if len(camera_ids) != 1:
        raise BackendError(
            f"model {model_id} registers images from {len(camera_ids)} cameras "
            f"({camera_ids}) under profile cad02a-shared-v1"
        )
    camera_id = camera_ids[0]
    if not reconstruction.exists_camera(camera_id):
        raise BackendError(
            f"model {model_id} references camera {camera_id}, which the model does not contain"
        )


def _publish_input_facts(records, frames_map: list[dict], upstream: UpstreamRun, staged) -> None:
    """Write the identity map and the self-consistent counters, together.

    These two are the run's account of *what the inputs were*, and they are
    published as one step so that no later rejection can leave one of them set and
    the other at its initial value.  ``unique_images`` and
    ``registered+unregistered`` are then a pair that can actually hold: every
    distinct image is, at this point, unregistered by construction.
    """
    records["counts"]["input_rows"] = len(staged.rows)
    records["counts"]["unique_images"] = len(staged.distinct)
    records["counts"]["registered_unique_images"] = 0
    records["counts"]["unregistered_unique_images"] = len(staged.distinct)
    _init_frames_map(upstream, staged, frames_map)


def _persist_intermediate(records, run_dir: pathlib.Path) -> None:
    """Flush the run state to disk at a stage boundary.

    ``run.json`` is rewritten at the end of the run, so an outer kill between two
    native stages would otherwise leave *no* record of which stages had started.
    Rewriting the same file at each boundary costs one small write and makes the
    progress readable while the run is still going -- and after it is gone.  It is
    best-effort: a write failure here must not abort a run whose real work is the
    native chain, and the final write is the authoritative one.
    """
    try:
        (run_dir / "run.json").write_text(
            dumps_strict(records, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    except OSError:
        pass


def _init_frames_map(upstream: UpstreamRun, staged, frames_map: list[dict]) -> None:
    """One record per upstream row, created before any native work happens."""
    frames_map.clear()
    for row in staged.rows:
        frames_map.append({
            "upstream_io_run_id": upstream.run_id,
            "input_index": row.input_index,
            "frame_id": row.frame_id,
            "upstream_image_output": row.image_output,
            "canonical_image_sha256": row.canonical_image_sha256,
            "upstream_mask_output": row.mask_output,
            "canonical_mask_sha256": row.canonical_mask_sha256,
            "representative_input_index": row.representative_input_index,
            "staged_image_name": row.staged_image_name,
            "database_image_id": None,
            "registrations": [],
            "status": "UNREGISTERED",
            "reason": "the native backend has not reported yet",
        })
        if getattr(staged,'selected_indices',None) is not None:
            item=frames_map[-1]
            item['foreground_selected']=row.input_index in staged.selected_indices
            if not item['foreground_selected']:
                item.update(status='EXCLUDED_FOREGROUND',reason='not selected by identity-bound object mask manifest; no whole-image fallback',
                    representative_input_index=None,staged_image_name=None)


def _fill_registrations(upstream: UpstreamRun, staged, stage_results, frames_map: list[dict]) -> None:
    """Attach real database ids and registrations once they are known."""
    extract = stage_results.get("extract_features") or {}
    names_to_ids = {
        str(name): int(image_id) for name, image_id in (extract.get("image_names_to_ids") or {}).items()
    }

    names_to_registration: dict[str, list[dict]] = {}
    models_root = None
    for entry in (stage_results.get("export_models") or {}).get("models", []):
        models_root = pathlib.Path(entry["binary_dir"])
        try:
            reconstruction = export_module.read_binary_model(models_root)
            description = export_module.describe_model(reconstruction)
        except Exception:  # noqa: BLE001 - registration detail is best-effort here
            continue
        for image in description["images"].values():
            names_to_registration.setdefault(image["name"], []).append({
                "model_id": str(entry["model_id"]),
                "image_id": int(image["image_id"]),
                "camera_id": int(image["camera_id"]),
            })

    by_index = {item["input_index"]: item for item in frames_map}
    for row in staged.rows:
        item = by_index[row.input_index]
        if getattr(staged,'selected_indices',None) is not None and row.input_index not in staged.selected_indices:
            item.update(database_image_id=None,registrations=[],status='EXCLUDED_FOREGROUND',
                reason='not selected by identity-bound object mask manifest; no whole-image fallback')
            continue
        item["database_image_id"] = names_to_ids.get(row.staged_image_name)
        registrations = names_to_registration.get(row.staged_image_name, [])
        item["registrations"] = registrations
        if registrations:
            item["status"] = "REGISTERED"
            item["reason"] = None
        else:
            item["status"] = "UNREGISTERED"
            item["reason"] = (
                "not registered in any component"
                if item["database_image_id"] is not None
                else "no database image record for this input"
            )


def _write_frames_map(path: pathlib.Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: row["input_index"])
    body = "".join(dumps_strict(row) + "\n" for row in rows)
    path.write_bytes(body.encode("utf-8"))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_run_id() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _project_root() -> pathlib.Path:
    current = pathlib.Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise CliError("run this command from the project root (no pyproject.toml found above the cwd)")


def _resolve_output_base(raw: str, project_root: pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(raw)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()
    required = (project_root / OUTPUT_PARENT).resolve()
    if required not in resolved.parents:
        raise CliError(f"--output must lie inside {required}")
    return resolved
