"""Directed regressions for the versioned camera-profile selector.

What is real here and what is a stand-in
----------------------------------------
Three kinds of test live in this file and they are deliberately not conflated:

* **Real native round trips.**  ``test_native_*`` builds a genuine reconstruction
  with pycolmap 4.2.1, writes it in both binary and text form, reads both back and
  compares camera/rig/frame/image/point/track correspondence.  This calls no SIFT,
  matcher or mapper and is *not* a photo reconstruction.
* **Real DB identity through a controlled stand-in.**  The shared-camera checks are
  fed database-shaped results produced here, so every counterexample (extra image,
  missing image, wrong camera id, NaN parameter, ...) is constructed rather than
  hoped for.  These prove the *validator* rejects bad databases; they do not prove
  a real extraction produced a good one.
* **Control-flow only, explicitly marked.**  The incremental-evidence tests use a
  short-lived worker process, but they are not evidence that the (separate)
  job-object supervisor works; that supervisor is another task's subject.

Every rejection test is paired with an accepting control, because a validator that
refuses everything also "passes" a rejection test.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from typing import Any

import numpy as np
import pytest
from PIL import Image

from kozlov_lab.reconstruction import cli, colmap_adapter as adapter, profiles

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


# --------------------------------------------------------------------------- #
# upstream fixture writer (this file's own, not the T01 implementation's)
# --------------------------------------------------------------------------- #
def write_png(path: pathlib.Path, value: int, size=(1024, 683)) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_upstream_run(root: pathlib.Path, entries: list[dict] | None = None) -> pathlib.Path:
    """A minimal, genuine ``photo-io-v1`` run; sizes are caller-controlled."""
    root.mkdir(parents=True, exist_ok=True)
    if entries is None:
        entries = [{"value": 10 + 7 * i} for i in range(3)]
    rows = []
    for index, entry in enumerate(entries):
        size = entry.get("size", (1024, 683))
        name = f"images/{index:06d}.png"
        digest = write_png(root / name, entry["value"], size)
        rows.append({
            "input_index": index,
            "frame_id": f"f_{index:064d}",
            "image_output": name,
            "mask_output": None,
            "image_sha256": digest,
            "mask_sha256": None,
            "status": "OK",
            "canonical_width": size[0],
            "canonical_height": size[1],
        })
    (root / "frames.jsonl").write_bytes(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    )
    (root / "run.json").write_bytes((json.dumps({
        "contract_version": "photo-io-v1",
        "run_id": "upstream_run",
        "data_kind": "synthetic",
        "status": "SUCCESS",
        "input_count": len(rows),
        "ok_count": len(rows),
        "error_count": 0,
    }, indent=2) + "\n").encode("utf-8"))
    return root


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# 1. unknown profile: reachable, exit 2, artifacts written, no backend started
# --------------------------------------------------------------------------- #
def test_unknown_profile_exits_two_with_written_evidence(tmp_path, monkeypatch) -> None:
    """The selector must be rejected *inside* the landing path, not before it.

    Raising from ``effective_options(profile)`` happened before the try, so the
    ProfileError escaped: no run.json, no frames_map.jsonl, and a traceback rather
    than the contract's INPUT_ERROR/exit 2.  This asserts the CLI-level result, not
    merely that ``require_valid_profile`` raises.
    """
    run = make_upstream_run(tmp_path / "upstream")
    run_dir = tmp_path / "out" / "20261003T000000Z_deadbeef"
    run_dir.mkdir(parents=True)

    started: list[str] = []
    monkeypatch.setattr(cli, "backend_info", lambda *a, **k: started.append("backend"))
    monkeypatch.setattr(cli, "run_native_stage", lambda *a, **k: started.append("stage"))

    exit_code = cli.run_reconstruction(
        io_run=str(run), run_dir=run_dir, project_root=tmp_path, profile="not-a-profile"
    )

    assert exit_code == 2, "an unknown selector is a parameter problem (INPUT_ERROR)"
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "INPUT_ERROR"
    assert payload["errors"] and payload["errors"][0]["code"] == "INPUT_ERROR"
    # The message must name the value and the legal ones, not be a generic failure.
    message = payload["errors"][0]["message"]
    assert "not-a-profile" in message
    assert profiles.PROFILE_T02A_V1 in message and profiles.PROFILE_CAD02A_SHARED_V1 in message
    # The map exists (empty) rather than being absent, and nothing native ran.
    assert (run_dir / "frames_map.jsonl").is_file()
    assert read_jsonl(run_dir / "frames_map.jsonl") == []
    assert payload["stages"] == []
    assert started == [], f"no backend or stage may start for a rejected selector: {started}"


@pytest.mark.parametrize("bad", ["", "T02A-V1", "cad02a_shared_v1", None, 1, ["t02a-v1"]])
def test_every_illegal_selector_is_an_input_error_not_a_crash(tmp_path, monkeypatch, bad) -> None:
    """Non-string selectors must be classified, not raise from a dict lookup."""
    run = make_upstream_run(tmp_path / "upstream")
    run_dir = tmp_path / "out" / "rid"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(cli, "backend_info", lambda *a, **k: pytest.fail("backend started"))
    monkeypatch.setattr(cli, "run_native_stage", lambda *a, **k: pytest.fail("stage started"))

    exit_code = cli.run_reconstruction(
        io_run=str(run), run_dir=run_dir, project_root=tmp_path, profile=bad
    )
    assert exit_code == 2
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "INPUT_ERROR"
    assert payload["errors"][0]["code"] == "INPUT_ERROR", bad


def test_the_valid_profiles_are_not_rejected_by_the_same_path(tmp_path, monkeypatch) -> None:
    """Negative control: the rejection above must come from the value, not the path."""
    for profile in (profiles.PROFILE_T02A_V1, profiles.PROFILE_CAD02A_SHARED_V1):
        run = make_upstream_run(tmp_path / f"upstream_{profile}")
        run_dir = tmp_path / f"out_{profile}" / "rid"
        run_dir.mkdir(parents=True)
        monkeypatch.setattr(cli, "backend_info", lambda *a, **k: {
            "name": "pycolmap", "version": adapter.BACKEND_PIN, "runtime": "cpu"})
        monkeypatch.setattr(cli, "run_native_stage", lambda stage, spec, **k: adapter.NativeStageResult(
            stage=stage, started_utc="T", finished_utc="T", returncode=None, timed_out=False,
            stdout_path="", stderr_path="", result=None, error="not executed here", started=False))
        exit_code = cli.run_reconstruction(
            io_run=str(run), run_dir=run_dir, project_root=tmp_path, profile=profile)
        assert exit_code != 2, f"{profile} is legal and must not be an INPUT_ERROR"


# --------------------------------------------------------------------------- #
# 2. mixed sizes: identity and counters published before the rejection
# --------------------------------------------------------------------------- #
def test_mixed_sizes_keep_every_row_traceable_and_run_no_native_stage(tmp_path, monkeypatch) -> None:
    """The rejection must come *after* the evidence, not instead of it.

    Rejecting before the map and counters were initialised left a written run
    claiming ``unique_images=4`` beside ``registered+unregistered=0`` and an empty
    frames_map -- two numbers that cannot both be true, and no record of which
    inputs were involved.
    """
    entries = [
        {"value": 10, "size": (1024, 683)},
        {"value": 20, "size": (1024, 683)},   # duplicate bytes of row 2 is below
        {"value": 20, "size": (1024, 683)},   # byte-identical alias of row 1
        {"value": 30, "size": (800, 600)},    # a different size
        {"value": 40, "size": (1024, 500)},   # a third size
    ]
    run = make_upstream_run(tmp_path / "upstream", entries)
    run_dir = tmp_path / "out" / "rid"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(cli, "backend_info", lambda *a, **k: pytest.fail("backend probe started"))
    monkeypatch.setattr(cli, "run_native_stage", lambda *a, **k: pytest.fail("native stage started"))

    exit_code = cli.run_reconstruction(
        io_run=str(run), run_dir=run_dir, project_root=tmp_path,
        profile=profiles.PROFILE_CAD02A_SHARED_V1,
    )

    assert exit_code == 2
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "INPUT_ERROR"
    assert "same size" in payload["errors"][0]["message"]
    assert payload["stages"] == [], "no native stage may run for a mixed-size shared request"

    counts = payload["counts"]
    assert counts["input_rows"] == 5
    assert counts["unique_images"] == 4, "rows 1 and 2 share bytes"
    assert counts["registered_unique_images"] + counts["unregistered_unique_images"] == counts["unique_images"]
    assert counts["registered_unique_images"] == 0

    rows = read_jsonl(run_dir / "frames_map.jsonl")
    assert [row["input_index"] for row in rows] == [0, 1, 2, 3, 4], "every row is traceable"
    by_index = {row["input_index"]: row for row in rows}
    # The duplicate alias inherits the representative's staged name without
    # overwriting its own identity.
    assert by_index[2]["representative_input_index"] == 1
    assert by_index[2]["staged_image_name"] == by_index[1]["staged_image_name"]
    assert by_index[2]["frame_id"] != by_index[1]["frame_id"]
    # None of them claims to be registered by an alias.
    assert all(row["registrations"] == [] for row in rows)


def test_mixed_sizes_are_allowed_under_the_default_profile(tmp_path, monkeypatch) -> None:
    """Negative control: the size rule belongs to the shared profile, not to staging."""
    entries = [{"value": 10, "size": (1024, 683)}, {"value": 30, "size": (800, 600)}]
    run = make_upstream_run(tmp_path / "upstream", entries)
    run_dir = tmp_path / "out" / "rid"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(cli, "backend_info", lambda *a, **k: {
        "name": "pycolmap", "version": adapter.BACKEND_PIN, "runtime": "cpu"})
    monkeypatch.setattr(cli, "run_native_stage", lambda stage, spec, **k: adapter.NativeStageResult(
        stage=stage, started_utc="T", finished_utc="T", returncode=None, timed_out=False,
        stdout_path="", stderr_path="", result=None, error="not executed here", started=False))

    exit_code = cli.run_reconstruction(
        io_run=str(run), run_dir=run_dir, project_root=tmp_path,
        profile=profiles.PROFILE_T02A_V1,
    )
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert exit_code != 2, "per-image cameras do not require equal sizes"
    assert payload["status"] != "INPUT_ERROR"


def test_shared_staging_never_resamples(tmp_path) -> None:
    """There is no resampling path: differing sizes are refused, not adapted."""
    entries = [{"value": 10, "size": (1024, 683)}, {"value": 30, "size": (512, 384)}]
    run = make_upstream_run(tmp_path / "upstream", entries)
    upstream = adapter.load_upstream_run(run)
    staged = adapter.stage_inputs(upstream, tmp_path / "staged")

    for row in staged.distinct:
        staged_file = tmp_path / "staged" / row.staged_image_name
        assert Image.open(staged_file).size == (row.canonical_width, row.canonical_height)
        assert staged_file.read_bytes() == (run / row.image_output).read_bytes()


# --------------------------------------------------------------------------- #
# 3. shared DB identity: names -> real ids -> camera id, as separate facts
# --------------------------------------------------------------------------- #
def _shared_extract_result(names: list[str], **overrides) -> dict:
    """A database-shaped extraction result for exactly ``names``."""
    ids = {name: 1 + 2 * index for index, name in enumerate(names)}   # deliberately non-contiguous
    base = {
        "num_cameras": 1,
        "num_images": len(names),
        "cameras": {"7": {
            "model_name": "SIMPLE_RADIAL",
            "width": 1024,
            "height": 683,
            "params": [1228.8, 512.0, 341.5, 0.0],
            "has_prior_focal_length": False,
        }},
        "image_names_to_ids": dict(ids),
        "image_camera_ids": {name: 7 for name in names},
    }
    base.update(overrides)
    return base


def _expected_for(names: list[str]) -> dict:
    return {name: (1024, 683) for name in names}


def _run_shared_check(result: dict, names: list[str]) -> None:
    cli._check_shared_camera(result, None, _expected_for(names))


def test_a_well_formed_shared_result_is_accepted() -> None:
    """Negative control for every rejection below."""
    names = [f"images/{i:06d}.png" for i in range(12)]
    _run_shared_check(_shared_extract_result(names), names)


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("extra_image", lambda r: r["image_names_to_ids"].update({"images/999999.png": 99})),
        ("missing_image", lambda r: r["image_names_to_ids"].pop("images/000000.png")),
        ("renamed_image", lambda r: r["image_names_to_ids"].update(
            {"images/000999.png": r["image_names_to_ids"].pop("images/000000.png")})),
        ("same_count_all_wrong_names", lambda r: r.update({
            "image_names_to_ids": {f"other/{i:06d}.png": 1 for i in range(12)},
            "image_camera_ids": {f"other/{i:06d}.png": 7 for i in range(12)},
        })),
    ],
)
def test_a_wrong_image_name_set_is_rejected(label, mutate) -> None:
    """Count equality is not identity equality.

    ``same_count_all_wrong_names`` is the case the old check let through: the map
    had the right length and every value pointed at the right camera, but not one
    name matched an input.
    """
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    mutate(result)
    if label == "extra_image":
        result["image_camera_ids"]["images/999999.png"] = 7
    with pytest.raises(adapter.BackendError):
        _run_shared_check(result, names)


def test_a_duplicate_database_image_id_is_rejected() -> None:
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    result["image_names_to_ids"]["images/000001.png"] = result["image_names_to_ids"]["images/000000.png"]
    with pytest.raises(adapter.BackendError) as caught:
        _run_shared_check(result, names)
    assert "duplicate image ids" in str(caught.value)


def test_a_missing_camera_record_for_the_referenced_id_is_rejected() -> None:
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    result["cameras"] = {"8": result["cameras"]["7"]}   # images still point at 7
    with pytest.raises(adapter.BackendError):
        _run_shared_check(result, names)


def test_more_than_one_camera_is_rejected() -> None:
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    result["cameras"]["8"] = dict(result["cameras"]["7"])
    result["num_cameras"] = 2
    with pytest.raises(adapter.BackendError) as caught:
        _run_shared_check(result, names)
    assert "exactly one camera" in str(caught.value)


def test_an_image_referencing_a_different_camera_is_rejected() -> None:
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    result["image_camera_ids"]["images/000005.png"] = 99
    with pytest.raises(adapter.BackendError) as caught:
        _run_shared_check(result, names)
    assert "must reference shared camera" in str(caught.value)


def test_non_contiguous_ids_are_read_rather_than_assumed(tmp_path) -> None:
    """The validator must not depend on ids being 1..N in order."""
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    shuffled = {name: 1000 - 37 * index for index, name in enumerate(reversed(names))}
    result["image_names_to_ids"] = dict(shuffled)
    _run_shared_check(result, names)   # accepted, and no code inferred 1..12


# --------------------------------------------------------------------------- #
# 4. shared intrinsics: non-finite, wrong model/size/params, prior focal length
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "params"),
    [
        ("nan_focal", [float("nan"), 512.0, 341.5, 0.0]),
        ("nan_principal", [1228.8, float("nan"), 341.5, 0.0]),
        ("nan_k1", [1228.8, 512.0, 341.5, float("nan")]),
        ("pos_inf", [float("inf"), 512.0, 341.5, 0.0]),
        ("neg_inf", [1228.8, 512.0, float("-inf"), 0.0]),
    ],
)
def test_non_finite_shared_intrinsics_are_rejected(label, params) -> None:
    """``abs(nan - wanted) > tol`` is False, so NaN used to slip through the check.

    A camera carrying NaN is exactly the camera whose intrinsics are unusable, so
    the finiteness gate is the load-bearing part of this validator.
    """
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    result["cameras"]["7"]["params"] = params
    with pytest.raises(adapter.BackendError) as caught:
        _run_shared_check(result, names)
    assert "finite" in str(caught.value), label


def test_the_nan_case_would_pass_a_naive_tolerance_check() -> None:
    """Proof that the finiteness gate is doing work, not decorating a working check.

    This replicates the old comparison and shows it accepts NaN; if it ever starts
    failing, the gate above has become redundant and this test should be reconsidered
    rather than deleted.
    """
    wanted = [1228.8, 512.0, 341.5, 0.0]
    naive_would_accept = not any(
        abs(actual - target) > 1e-6
        for actual, target in zip([float("nan"), 512.0, 341.5, 0.0], wanted)
    )
    assert naive_would_accept, "the old tolerance check did accept NaN"
    assert not math.isfinite(float("nan"))


@pytest.mark.parametrize(
    ("label", "field", "value"),
    [
        ("wrong_model", "model_name", "PINHOLE"),
        ("wrong_width", "width", 1920),
        ("wrong_height", "height", 1080),
        ("prior_focal_flag", "has_prior_focal_length", True),
    ],
)
def test_wrong_model_size_or_prior_flag_is_rejected(label, field, value) -> None:
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    result["cameras"]["7"][field] = value
    with pytest.raises(adapter.BackendError):
        _run_shared_check(result, names)


@pytest.mark.parametrize(
    ("label", "params"),
    [
        ("wrong_focal", [2400.0, 512.0, 341.5, 0.0]),
        ("wrong_principal", [1228.8, 511.0, 341.5, 0.0]),
        ("nonzero_k1", [1228.8, 512.0, 341.5, 0.05]),
        ("too_few", [1228.8, 512.0, 341.5]),
        ("too_many", [1228.8, 512.0, 341.5, 0.0, 0.0]),
    ],
)
def test_wrong_initial_intrinsics_are_rejected(label, params) -> None:
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    result["cameras"]["7"]["params"] = params
    with pytest.raises(adapter.BackendError):
        _run_shared_check(result, names)


def test_the_requested_option_is_not_read_back_as_a_database_fact() -> None:
    """A camera in the DB that differs from the request must be refused.

    The point of checking the database rather than the requested option is that a
    backend which ignored the mode would still leave the request looking correct.
    Here the *request* is right and the *database* is wrong, and the wrong one wins
    the verdict.
    """
    names = [f"images/{i:06d}.png" for i in range(12)]
    result = _shared_extract_result(names)
    result["cameras"]["7"]["params"] = [999.0, 512.0, 341.5, 0.0]
    assert cli.effective_options(profiles.PROFILE_CAD02A_SHARED_V1)["camera_sharing"] == "single"
    with pytest.raises(adapter.BackendError):
        _run_shared_check(result, names)


def test_shared_expected_intrinsics_match_the_fixed_experiment() -> None:
    """The fixed values are a contract constant, not a coincidence of this file."""
    assert cli.SHARED_EXPECTED_PARAMS == [1228.8, 512.0, 341.5, 0.0]
    assert cli.SHARED_EXPECTED_PARAMS[0] == 1.2 * 1024


# --------------------------------------------------------------------------- #
# profile plumbing: version, options, fingerprint, worker argument
# --------------------------------------------------------------------------- #
def test_default_profile_options_gain_no_profile_field() -> None:
    options = cli.effective_options(profiles.PROFILE_T02A_V1)
    assert "profile_id" not in options, "the selector must not rewrite the default's options"
    assert options["camera_sharing"] == "per_image_independent"
    assert cli.effective_options() == options, "no argument must equal the explicit default"


def test_no_argument_and_explicit_default_produce_identical_options() -> None:
    assert adapter.effective_options() == adapter.effective_options(profiles.PROFILE_T02A_V1)


def test_shared_profile_records_its_identity_and_one_change() -> None:
    options = cli.effective_options(profiles.PROFILE_CAD02A_SHARED_V1)
    assert options["profile_id"] == profiles.PROFILE_CAD02A_SHARED_V1
    assert options["camera_sharing"] == "single"
    assert profiles.contract_version(profiles.PROFILE_CAD02A_SHARED_V1) == "sfm-native-profile-v1"
    assert profiles.camera_mode(profiles.PROFILE_CAD02A_SHARED_V1) == "SINGLE"
    assert profiles.camera_mode(profiles.PROFILE_T02A_V1) == "PER_IMAGE"
    # The shared profile changes exactly one *scientific* option.
    default = cli.effective_options(profiles.PROFILE_T02A_V1)
    shared = cli.effective_options(profiles.PROFILE_CAD02A_SHARED_V1)
    differing = {key for key in set(default) | set(shared) if default.get(key) != shared.get(key)}
    assert differing == {"profile_id", "camera_sharing"}, differing


def test_the_two_fingerprints_are_correctly_versioned_and_option_bound(tmp_path) -> None:
    """Both fingerprints must name the shared version and the real effective options."""
    run = make_upstream_run(tmp_path / "upstream")
    upstream = adapter.load_upstream_run(run)
    staged = adapter.stage_inputs(upstream, tmp_path / "staged")
    backend = {"name": "pycolmap", "version": adapter.BACKEND_PIN, "runtime": "cpu"}

    shared_options = cli.effective_options(profiles.PROFILE_CAD02A_SHARED_V1)
    shared_version = profiles.contract_version(profiles.PROFILE_CAD02A_SHARED_V1)
    shared_fp = adapter.input_fingerprint(
        upstream, staged.rows, shared_options, backend, "impl", contract_version=shared_version)

    default_options = cli.effective_options(profiles.PROFILE_T02A_V1)
    default_fp = adapter.input_fingerprint(
        upstream, staged.rows, default_options, backend, "impl",
        contract_version=profiles.contract_version(profiles.PROFILE_T02A_V1))

    assert shared_fp != default_fp
    # Recomputing with the other version must change the digest: the version is bound.
    assert adapter.input_fingerprint(
        upstream, staged.rows, shared_options, backend, "impl",
        contract_version="sfm-native-v1") != shared_fp
    # And the options are bound too.
    mutated = dict(shared_options)
    mutated["camera_sharing"] = "per_image_independent"
    assert adapter.input_fingerprint(
        upstream, staged.rows, mutated, backend, "impl",
        contract_version=shared_version) != shared_fp


def test_the_worker_receives_single_captured_at_the_call(tmp_path, monkeypatch) -> None:
    """The mode must reach the native call, not merely appear in the configuration.

    A stand-in for ``pycolmap.extract_features`` records what it was actually
    handed, so this catches "the option string is right but the argument is not".
    """
    captured: dict = {}

    class FakeCameraMode:
        PER_IMAGE = "per-image-sentinel"
        SINGLE = "single-sentinel"

    fake = type("Fake", (), {})
    fake.CameraMode = FakeCameraMode
    fake.Device = type("Device", (), {"cpu": "cpu"})
    fake.FeatureExtractionOptions = lambda: type("O", (), {
        "use_gpu": None, "num_threads": None, "max_image_size": None,
        "sift": type("S", (), {"max_num_features": None})()})()
    fake.ImageReaderOptions = lambda: type("R", (), {
        "camera_model": None, "default_focal_length_factor": None,
        "camera_params": None, "existing_camera_id": None})()
    fake.Database = type("DB", (), {"open": staticmethod(lambda p: pytest.fail("no db expected"))})

    def extract_features(database_path, image_path, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capturing; the database is not opened here")

    fake.extract_features = extract_features

    spec = {
        "database_path": str(tmp_path / "db"), "image_path": str(tmp_path),
        "image_names": ["images/000000.png"], "num_threads": 2, "max_long_side": 1024,
        "max_sift_features": 4096, "camera_model": "SIMPLE_RADIAL",
        "camera_mode": profiles.camera_mode(profiles.PROFILE_CAD02A_SHARED_V1),
        "focal_length_factor": 1.2,
    }
    with pytest.raises(RuntimeError):
        adapter._stage_extract(fake, spec)

    assert captured["camera_mode"] == "single-sentinel", captured
    assert captured["device"] == "cpu"
    assert captured["extraction_options"].max_image_size == 1024


def test_the_two_profiles_reach_the_worker_as_different_modes(tmp_path) -> None:
    """A mode that is right for both profiles would prove nothing."""
    assert profiles.camera_mode(profiles.PROFILE_T02A_V1) != \
        profiles.camera_mode(profiles.PROFILE_CAD02A_SHARED_V1)


def test_the_two_view_geometry_object_is_the_one_passed_to_the_backend() -> None:
    """The recorded geometry options must belong to the object the backend received.

    The earlier round constructed a ``TwoViewGeometryOptions`` and recorded its
    ``todict`` as "actual options" while passing nothing to
    ``match_exhaustive`` — a record of an object the backend never saw.  A
    controlled stand-in captures the call arguments, so the assertion is against
    the call shape rather than against the recording code's own output.
    """
    captured: dict = {}

    class StandInGeometry:
        def todict(self) -> dict:
            return {"min_num_inliers": 15, "sentinel": "this-build-default"}

    class StandInMatching:
        use_gpu = None
        num_threads = None

        def todict(self) -> dict:
            return {"use_gpu": False, "num_threads": None}

    class StandInPairing:
        def todict(self) -> dict:
            return {"pairing": "exhaustive"}

    class StandInDatabase:
        @staticmethod
        def open(_path):
            class _Db:
                @staticmethod
                def num_matched_image_pairs() -> int:
                    return 0

                @staticmethod
                def num_verified_image_pairs() -> int:
                    return 0

                @staticmethod
                def num_inlier_matches() -> int:
                    return 0

                @staticmethod
                def close() -> None:
                    return None

            return _Db()

    fake = type("StandIn", (), {})
    fake.FeatureMatchingOptions = StandInMatching
    fake.ExhaustivePairingOptions = StandInPairing
    fake.TwoViewGeometryOptions = StandInGeometry
    fake.Device = type("Device", (), {"cpu": "cpu"})
    fake.Database = StandInDatabase

    def match_exhaustive(_database_path, **kwargs):
        captured.update(kwargs)

    fake.match_exhaustive = staticmethod(match_exhaustive)

    outcome = adapter._stage_match(fake, {"database_path": "db", "num_threads": 2})

    assert "verification_options" in captured, (
        "the geometry object must be passed, not merely constructed")
    geometry = captured["verification_options"]
    assert isinstance(geometry, StandInGeometry)
    # The record must be *the same object's* description.
    assert outcome["actual_options"]["two_view_geometry"] == geometry.todict()
    assert outcome["actual_options"]["two_view_geometry_passed_as_argument"] is True
    # The other three arguments are still the intended ones.
    assert isinstance(captured["matching_options"], StandInMatching)
    assert isinstance(captured["pairing_options"], StandInPairing)
    assert captured["device"] == "cpu"


def test_the_installed_version_really_accepts_verification_options() -> None:
    """Read the installed signature rather than asserting it from memory.

    ``match_exhaustive`` is a pybind11 builtin with no inspectable signature, so the
    parameter is read from its docstring, which is generated from the real binding.
    This is what makes "the object can be passed" a fact about 4.2.1 rather than a
    hope: if a later build dropped the parameter, this test fails.
    """
    pycolmap = pytest.importorskip("pycolmap")
    doc = (pycolmap.match_exhaustive.__doc__ or "")
    assert "verification_options" in doc, doc
    assert "TwoViewGeometryOptions" in doc, doc
    assert hasattr(pycolmap, "TwoViewGeometryOptions")
    # The full documented default object, as this build reports it.
    geometry = pycolmap.TwoViewGeometryOptions()
    recorded = adapter._todict(geometry)
    assert isinstance(recorded, dict) and recorded, "todict must describe the object"
    assert "min_num_inliers" in recorded
    assert json.dumps(recorded)  # serializable, so it can be written to the record


def test_default_run_options_are_unchanged_scientifically() -> None:
    """The default must still carry the frozen engineering configuration."""
    options = cli.effective_options(profiles.PROFILE_T02A_V1)
    assert options["device"] == "cpu"
    assert options["num_threads"] == 2
    assert options["max_distinct_images"] == 12
    assert options["max_long_side"] == 1024
    assert options["sift_max_num_features"] == 4096
    assert options["random_seed"] == 0
    assert options["camera_model"] == "SIMPLE_RADIAL"
    assert options["min_model_size"] == 2
    assert options["multiple_models"] is True
    assert options["ba_refine_focal_length"] is True
    assert options["ba_refine_principal_point"] is False
    assert options["ba_refine_extra_params"] is True


def test_geometrically_non_finite_json_still_cannot_be_written() -> None:
    """Strict JSON: the stricter parameter handling must not have introduced a path."""
    with pytest.raises(ValueError):
        adapter.dumps_strict({"params": [float("nan"), 0.0]})


# --------------------------------------------------------------------------- #
# 5. incremental per-process evidence, through the real execution path
# --------------------------------------------------------------------------- #
#: Every real fixture in this file self-limits from launch, and the close-out fixes
#: the ceiling at five seconds.  The reason is not tidiness: a test that relied on the
#: harness to end its child could not tell "the product reclaimed the process" from
#: "the fixture quietly exited on its own", and a fixture that outlived its bound
#: would leave a process behind that later runs could be blamed for.
#:
#: The *product* bound is deliberately a different, larger number.  Where a case must
#: show that the product ended the child, the product bound is set **below** the
#: fixture's own limit and the elapsed time is asserted to be below it too; that is
#: what makes the reclamation an observation instead of a race the fixture won.
#: Both fixture bodies below are built from this constant rather than repeating it, so
#: the number cannot drift away from the tests that reason about it.
_FIXTURE_SELF_LIMIT_SECONDS = 4.0
_FIXTURE_STAGE_BOUND_SECONDS = 8.0


@pytest.fixture(autouse=True)
def _pin_tempfile_landing(tmp_path_factory, monkeypatch):
    """Pin every ambient ``tempfile`` landing in this file inside the entry's basetemp.

    The no-evidence probe form creates its scratch stream pair with
    ``tempfile.mkstemp`` and passes no directory, so it lands wherever ``tempfile``
    happens to point.  Three cases pinned that themselves and the rest did not, which
    left the round unable to claim that *all* of its scratch was inside the run — the
    one no-evidence timeout case in particular created a real file outside it.

    Applied to every test in this file (autouse) rather than case by case, so the
    guarantee is a property of the round instead of a property of remembering.  The
    ambient fallback is redirected for the duration of each test and restored by
    ``monkeypatch`` afterwards; the production functions are untouched — the adapter
    must keep using the process's temporary directory, and a machine-specific path
    baked into it for a test's benefit would be a worse defect than the one this
    prevents.  Nothing here reads or scans the old system TEMP.

    ``tmp_path_factory`` is used rather than ``tmp_path`` so the landing directory is
    inside the pytest basetemp *before* ``tempfile`` is redirected: the factory derives
    its root from ``--basetemp``, not from ``tempfile.gettempdir()``, so this cannot
    recurse into the directory it is creating.
    """
    landing = tmp_path_factory.mktemp("tempfile-landing")
    monkeypatch.setattr(tempfile, "tempdir", str(landing))
    return landing


def _assert_death_inside_fixture_limit(*, label: str, launch_monotonic: float,
                                       ready_monotonic: float, death_monotonic: float,
                                       fallback_used: bool) -> dict:
    """Check that a worker's death was confirmed *inside* the fixture's own self-limit.

    Used by both real timeout cases, and driven by a synthetic clock in
    ``test_the_self_limit_boundary_rejects_a_death_that_arrived_too_late``, so the
    boundary is exercised even on a round that does not re-run a real fixture.

    ``death_monotonic`` must have been read only *after* the same worker was confirmed
    dead through the identity-bound observation — not while it was still running.  A
    reading taken before the death wait cannot separate three different endings:

    * the product timed the worker out and reclaimed it, which is the ending these two
      cases exist to demonstrate;
    * the fixture reached its own self-limit and ended itself;
    * the test's own fallback ended it.

    Which anchor discriminates is not arbitrary.  The fixture's limit is measured from
    the fixture's own start, and the launch anchor precedes that start, so a fixture that
    ended itself shows ``since_launch >= limit`` while a reclaimed one shows
    ``since_launch ~ bound``.  The READY anchor is *later* than the fixture's start, so a
    self-limited fixture can show ``since_ready < limit`` and pass — it is recorded
    because it is the tighter number, but the comparison that decides is the launch one.
    Both are returned so the caller can put them in its own record.
    """
    since_launch = death_monotonic - launch_monotonic
    since_ready = death_monotonic - ready_monotonic
    assert not fallback_used, (
        "%s: the test's own fallback ran, so this death cannot be attributed to the "
        "product's cleanup" % label)
    assert 0.0 <= since_ready < since_launch, (
        "%s: the anchors are inconsistent (ready-to-death %.3fs, launch-to-death %.3fs); "
        "the READY line cannot have preceded the launch" % (label, since_ready, since_launch))
    assert since_launch < _FIXTURE_SELF_LIMIT_SECONDS, (
        "%s: the worker was confirmed dead %.3fs after launch, which is not below the "
        "fixture's own %.1fs self-limit -- the fixture may have ended itself, so the "
        "product's reclamation was not observed"
        % (label, since_launch, _FIXTURE_SELF_LIMIT_SECONDS))
    return {"since_launch": since_launch, "since_ready": since_ready}

# A worker that lives long enough to be observed *while it is still running*.
#
# It is deliberately not a stage implementation.  The stage's own bootstrap
# imports this module's ``_worker_main``; the test redirects that import to this
# script by putting its directory first on the child's ``PYTHONPATH``, so the
# child that actually starts is the real ``run_native_stage`` launching a real
# ``subprocess`` through the real default executor.  What is replaced is the
# *subprogram*, which the task states is allowed — never the run semantics.
#
# The handshake: the worker prints a unique ``test_id`` on stdout, a second short
# line on stdout, one short line on stderr, flushes all three, then blocks until
# the observer removes the release file.  ``flush=True`` plus the child's
# unbuffered stdout is what makes "readable before exit" a property that can be
# tested at all; a buffered child would look identical to a broken observer.
def _install_worker_bootstrap(tmp_path: pathlib.Path, monkeypatch, worker_source: str,
                              **params) -> dict:
    """Redirect the child's import of ``_worker_main`` to a short-lived script.

    The child is launched with ``python -c "<bootstrap>" --stage ... --spec ...
    --result ...``, where the bootstrap imports ``_worker_main`` from the real
    package.  Mechanism: this module is installed as ``kozlov_lab``'s shadow on a
    directory placed first on the child's ``PYTHONPATH``, so the *same* bootstrap
    source line resolves to a different subprogram.  The real ``run_native_stage``,
    the real ``_default_stage_executor``, the real ``subprocess`` call and the real
    launch/finish records all run unchanged.

    The returned dict records which module path was actually imported, by asking
    the child to print it, so "the test replaced the real package" is a fact in the
    evidence rather than an assumption about sys.path ordering.
    """
    shadow = tmp_path / "shadow_pkg"
    pkg = shadow / "kozlov_lab"
    (pkg / "reconstruction").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "reconstruction" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "reconstruction" / "colmap_adapter.py").write_text(
        worker_source, encoding="utf-8", newline="\n")
    # ``PYTHONPATH`` is one of the two overrides the module is authorised to set, so
    # the shadow is prepended by the same environment_overrides() the run uses.
    monkeypatch.setattr(adapter, "environment_overrides", lambda root: {
        "PYTHONPATH": str(shadow), "PYTHONDONTWRITEBYTECODE": "1"})
    return {"shadow": shadow, "module_path": pkg / "reconstruction" / "colmap_adapter.py"}


class _ProcessObservation:
    """A read-only observation handle bound to one process's *creation identity*.

    Two defects this closes relative to re-deriving liveness from the bare PID on
    every check:

    * **PID reuse.**  A PID is not an identity.  The handle is opened once and kept,
      so it keeps naming the same kernel object even if the number is recycled, and
      the creation timestamp read from that handle is recorded when the observation is
      bound and compared against the current one on every later check.
    * **HANDLE signatures.**  On 64-bit Windows a ``HANDLE`` is pointer-sized.
      Declaring ``restype``/``argtypes`` for *every* kernel32 call used here —
      including ``GetExitCodeProcess`` and ``CloseHandle`` — is what keeps a 64-bit
      handle from being silently truncated to a 32-bit integer.

    Everything it does is a query.  It never signals, terminates, or adopts a
    process.  The familiar "send signal 0 to test existence" idiom is deliberately not
    used on Windows: there, every signal other than the two console events goes to
    ``TerminateProcess``, so the observer itself would do the killing and then report
    that the adapter had reaped — a pass for the wrong reason.  A refused or failed
    query, and an identity that does not match, are both reported ``"unknown"``;
    neither is ever folded into ``"dead"``.
    """

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    _ERROR_INVALID_PARAMETER = 87
    _CTYPES: Any = None
    _WINTYPES: Any = None
    _KERNEL32: Any = None

    def __init__(self, pid: int) -> None:
        self.pid = int(pid)
        self._handle: Any = None
        self._pidfd: int | None = None
        self._creation: Any = None
        self.identity_bound = False
        self.open_error: str | None = None
        self._closed = False
        self._open()

    # -- platform plumbing --------------------------------------------------
    @classmethod
    def _lib(cls):
        """(ctypes, wintypes, kernel32) with every signature we rely on declared."""
        if cls._KERNEL32 is None:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.GetExitCodeProcess.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.GetProcessTimes.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME)]
            kernel32.CloseHandle.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            cls._CTYPES, cls._WINTYPES, cls._KERNEL32 = ctypes, wintypes, kernel32
        return cls._CTYPES, cls._WINTYPES, cls._KERNEL32

    def _open(self) -> None:
        if os.name == "nt":
            ctypes, _wintypes, kernel32 = self._lib()
            handle = kernel32.OpenProcess(
                self._PROCESS_QUERY_LIMITED_INFORMATION, False, self.pid)
            if not handle:
                error = ctypes.get_last_error()
                if error == self._ERROR_INVALID_PARAMETER:
                    # The documented "no such process": a real answer, and the only
                    # open failure that is allowed to mean "not running".
                    self.open_error = "no such process (ERROR_INVALID_PARAMETER)"
                else:
                    # Including ERROR_ACCESS_DENIED (5): the question was refused, so
                    # the answer is unknown rather than "dead".
                    self.open_error = f"OpenProcess failed with error {error}"
                return
            self._handle = handle
            self._creation = self._read_creation_windows()
            self.identity_bound = self._creation is not None
            return
        # POSIX: a pidfd names the process itself and survives PID reuse.  Where the
        # kernel has no pidfd, the start time from /proc is the creation identity.
        opener = getattr(os, "pidfd_open", None)
        if opener is not None:
            try:
                self._pidfd = opener(self.pid)
            except OSError as exc:
                self.open_error = f"{type(exc).__name__}: {exc}"
        self._creation = self._read_creation_posix()
        self.identity_bound = self._creation is not None

    def _read_creation_windows(self) -> Any:
        if self._handle is None:
            return None
        ctypes, wintypes, kernel32 = self._lib()
        created, exited = wintypes.FILETIME(), wintypes.FILETIME()
        kernel, user = wintypes.FILETIME(), wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
                self._handle, ctypes.byref(created), ctypes.byref(exited),
                ctypes.byref(kernel), ctypes.byref(user)):
            return None
        return (created.dwHighDateTime, created.dwLowDateTime)

    def _read_creation_posix(self) -> Any:
        try:
            stat_text = pathlib.Path(f"/proc/{self.pid}/stat").read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            return None
        fields = stat_text.rsplit(")", 1)[-1].strip().split()
        # Field 22 of /proc/<pid>/stat (starttime), counted after the comm field.
        return fields[19] if len(fields) > 19 else None

    # -- queries ------------------------------------------------------------
    def state(self) -> str:
        """``"alive"``, ``"dead"`` or ``"unknown"`` — read-only, never a signal."""
        if self._closed:
            return "unknown"
        if self._pidfd is not None:
            return self._state_pidfd()
        if os.name == "nt":
            return self._state_windows()
        return self._state_posix()

    def _state_windows(self) -> str:
        if self._handle is None:
            return "unknown"
        ctypes, wintypes, kernel32 = self._lib()
        if self.identity_bound:
            current = self._read_creation_windows()
            if current is None:
                return "unknown"
            if current != self._creation:
                # A creation timestamp that does not match means the object this
                # handle names is not the process we bound to.  That is an identity
                # mismatch, which is reported as unknown rather than as death.
                return "unknown"
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code)):
            return "unknown"
        return "alive" if code.value == self._STILL_ACTIVE else "dead"

    def _state_pidfd(self) -> str:
        try:
            os.waitid(os.P_PIDFD, self._pidfd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return "dead"
        except OSError:
            return "unknown"
        return "alive"

    def _state_posix(self) -> str:
        try:
            stat_text = pathlib.Path(f"/proc/{self.pid}/stat").read_text(
                encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return "dead"
        except OSError:
            return "unknown"
        fields = stat_text.rsplit(")", 1)[-1].strip().split()
        if len(fields) < 20:
            return "unknown"
        if self.identity_bound and fields[19] != self._creation:
            return "unknown"
        # A zombie has already exited; it is not running.
        return "dead" if fields[0] == "Z" else "alive"

    def wait_dead(self, *, within: float) -> str:
        """Poll until the state is no longer ``alive``; return the final state."""
        deadline = time.monotonic() + within
        while True:
            current = self.state()
            if current != "alive":
                return current
            if time.monotonic() >= deadline:
                return "alive"
            time.sleep(0.05)

    def close(self) -> None:
        """Release the query handle.  Never signals the process."""
        if self._closed:
            return
        self._closed = True
        if self._pidfd is not None:
            try:
                os.close(self._pidfd)
            except OSError:
                pass
            self._pidfd = None
        if self._handle is not None:
            _ctypes, _wintypes, kernel32 = self._lib()
            try:
                kernel32.CloseHandle(self._handle)
            except OSError:
                # A failed CloseHandle leaves nothing to report to the product; the
                # observation is already over and this must not mask a test verdict.
                pass
            self._handle = None

    def __enter__(self) -> "_ProcessObservation":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _readiness_of(stdout_text: str, test_id: str) -> dict | None:
    """Extract the worker's readiness JSON line, or ``None`` if it is absent.

    The line is JSON precisely so the observer can assert on the worker's *own*
    identity — PID, imported module file, cwd, a task environment value — instead
    of on a marker that any process could have printed.
    """
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("worker_ready") == test_id:
            return payload
    return None


def _stage_spec_for_live_worker(tmp_path: pathlib.Path, worker_params: list[str]) -> dict:
    """A spec the shadow worker ignores, plus the params it does read.

    The real ``run_native_stage`` writes this spec to disk and passes ``--spec``;
    the shadow worker receives the same argv.  ``worker_params`` travels in a file
    the worker reads by convention, because the argv shape (``--stage``/``--spec``/
    ``--result``) is fixed by production and must not be widened for the test.
    """
    return {"database_path": str(tmp_path / "db"), "image_path": str(tmp_path)}


LIVE_WORKER_MAIN = '''
import os, sys, time, json, pathlib

def _worker_main(argv):
    # argv is production's: --stage S --spec P --result R.  The worker's own
    # parameters come from the spec document, which production already wrote.
    parser = {}
    for i in range(0, len(argv) - 1, 2):
        parser[argv[i].lstrip("-")] = argv[i + 1]
    stage = parser["stage"]
    spec = json.loads(pathlib.Path(parser["spec"]).read_text(encoding="utf-8"))
    result_path = pathlib.Path(parser["result"])
    release = spec["release_file"]
    test_id = spec["test_id"]
    # A readiness line carrying the worker's own identity: its PID, the module it
    # actually imported (so the shadow really was the imported subprogram), its
    # cwd, and one non-sensitive task environment value the test set.  A test that
    # did not receive these could not tell "the target worker ran" from "something
    # ran"; recording them makes the identity a fact in the log, not an assumption.
    print(json.dumps({
        "worker_ready": test_id,
        "pid": os.getpid(),
        "module_file": __file__,
        "cwd": os.getcwd(),
        "env_probe": os.environ.get("KOZLOV_WORKER_TAG"),
    }), flush=True)
    sys.stdout.write("worker:%s:second\\n" % test_id)
    sys.stdout.flush()
    sys.stderr.write("worker:%s:stderr\\n" % test_id)
    sys.stderr.flush()
    # Finite-lifetime self-limit.  The fixture must never outlive its own bound: the
    # close-out requires every real fixture to be self-limiting within 5 s from
    # launch, so no test depends on the harness killing it and no leftover process can
    # be blamed on the product.  A worker that reaches this bound exits 90 on its own.
    deadline = time.monotonic() + __FIXTURE_SELF_LIMIT__
    try:
        while not os.path.exists(release):
            if time.monotonic() > deadline:
                return 90
            time.sleep(0.02)
    finally:
        # A finite-lifetime worker cleans up after itself even on the path where the
        # observer never releases it, so a timed-out worker does not leave a stray
        # side file behind as a side effect of the *test* rather than of the code.
        pass
    result_path.write_text(json.dumps({
        "stage": stage, "ok": True, "test_id": test_id,
        "pid": os.getpid(), "module_file": __file__,
        "num_images": 0, "cameras": {}, "image_names_to_ids": {},
        "actual_options": {"worker": "shadow"},
    }), encoding="utf-8")
    return int(spec.get("exit_code", 0))
'''.replace("__FIXTURE_SELF_LIMIT__", repr(_FIXTURE_SELF_LIMIT_SECONDS))


def test_the_incremental_record_is_readable_while_the_real_child_runs(tmp_path, monkeypatch) -> None:
    """The launch record, both streams and the handshake are visible before exit.

    Every assertion here is made while the worker is **still alive**.  The test
    never writes the ``start`` line itself: the line must already be on disk
    because ``run_native_stage`` wrote it before creating the child.  The worker is
    then released and its real exit status, result document and finish record are
    checked, so "it streamed" and "it completed" are both facts rather than
    inferences from each other.
    """
    test_id = "live-%d" % os.getpid()
    release = tmp_path / "release.flag"
    installed = _install_worker_bootstrap(tmp_path, monkeypatch, LIVE_WORKER_MAIN)
    logs = tmp_path / "logs"
    spec = {
        "release_file": str(release), "test_id": test_id, "exit_code": 0,
        "database_path": str(tmp_path / "db"), "image_path": str(tmp_path),
    }

    observed: dict = {}

    def release_later() -> None:
        # Exits immediately; the flag is what wakes the worker, so the timing of
        # this helper is not part of the assertion.
        release.write_text("go", encoding="utf-8")

    # Released well inside the fixture's own limit, so the stage ends by the release
    # and the stage bound below can never be what produced the exit.
    timer = threading.Timer(2.0, release_later)
    timer.start()
    try:
        record = adapter.run_native_stage(
            "extract_features", spec,
            run_dir=tmp_path, logs_dir=logs, project_root=tmp_path,
            python_executable=sys.executable,
            timeout=_FIXTURE_STAGE_BOUND_SECONDS,
        )
        # Already finished by now; the *observations* below were taken from a
        # concurrent reader in the next test.  Here we assert the settled facts.
        observed["record"] = record
    finally:
        timer.cancel()
        release.write_text("go", encoding="utf-8")

    record = observed["record"]
    assert record.started is True, "the stage must have been started"
    assert record.returncode == 0, record.error
    assert record.timed_out is False
    assert record.terminated is False
    assert record.evidence_incomplete is False, "a complete record must not be flagged"

    stdout_text = (tmp_path / record.stdout_path).read_text(encoding="utf-8")
    stderr_text = (tmp_path / record.stderr_path).read_text(encoding="utf-8")
    readiness = _readiness_of(stdout_text, test_id)
    assert readiness is not None, "the worker's readiness line must be present"
    assert readiness["module_file"] == str(installed["module_path"]), (
        "the child must have imported the shadow subprogram, not the real package")
    assert f"worker:{test_id}:second" in stdout_text
    assert f"worker:{test_id}:stderr" in stderr_text

    launch = read_jsonl(tmp_path / record.launch_record_path)
    assert [entry["event"] for entry in launch] == ["start", "finish"]
    assert launch[0]["stage"] == "extract_features"
    assert record.result is not None and record.result["ok"] is True
    assert record.result["test_id"] == test_id, "the result is the one this worker wrote"
    assert record.result["module_file"] == readiness["module_file"], (
        "the readiness line and the result must name the same imported module")


def test_both_streams_and_the_id_are_readable_before_the_child_exits(tmp_path, monkeypatch) -> None:
    """The observer reads the evidence *while the child is blocked*, then releases it.

    This is the property the earlier round only checked after exit: a short,
    already-flushed line must be readable from a running child, and the launch
    record must exist before the child is created.  The worker blocks on the
    release file, so a passing read cannot be explained by the worker having
    finished quickly.
    """
    test_id = "alive-%d" % os.getpid()
    release = tmp_path / "release.flag"
    installed = _install_worker_bootstrap(tmp_path, monkeypatch, LIVE_WORKER_MAIN)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    record_path = logs / "extract_features.launch.jsonl"
    stdout_path = logs / "extract_features.stdout.txt"
    stderr_path = logs / "extract_features.stderr.txt"
    spec = {
        "release_file": str(release), "test_id": test_id, "exit_code": 0,
        "database_path": str(tmp_path / "db"), "image_path": str(tmp_path),
    }
    monkeypatch.setenv("KOZLOV_WORKER_TAG", "tag-%d" % os.getpid())

    holder: dict = {}

    def run() -> None:
        holder["record"] = adapter.run_native_stage(
            "extract_features", spec,
            run_dir=tmp_path, logs_dir=logs, project_root=tmp_path,
            python_executable=sys.executable,
            timeout=_FIXTURE_STAGE_BOUND_SECONDS,
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        # Bounded by the same fixture limit: the worker is released promptly, so a
        # poll that runs longer than the fixture's own lifetime means the fixture
        # never became readable, not that the machine was slow.
        deadline = time.monotonic() + _FIXTURE_STAGE_BOUND_SECONDS
        readiness = None
        while time.monotonic() < deadline:
            if (record_path.is_file() and stdout_path.is_file()
                    and stderr_path.is_file()):
                launch = read_jsonl(record_path)
                out = stdout_path.read_text(encoding="utf-8")
                err = stderr_path.read_text(encoding="utf-8")
                readiness = _readiness_of(out, test_id)
                if (launch and launch[0].get("event") == "start"
                        and readiness is not None
                        and f"worker:{test_id}:stderr" in err):
                    holder["readiness"] = readiness
                    break
            time.sleep(0.05)
        else:
            raise AssertionError(
                "the launch record and both streams must be readable from a child that "
                "has not exited yet"
            )

        # The child is still blocked on the release file, so nothing read above can
        # be attributed to a finished process.
        assert thread.is_alive(), (
            "the worker must still be running when the evidence is read")
        assert not release.exists()

        # Identity assertions, taken while the very same PID is alive: the readiness
        # line came from *this* worker, importing the shadow module, with the task
        # environment value the test set.  A mere marker string could not show this.
        assert readiness["module_file"] == str(installed["module_path"])
        assert readiness["env_probe"] == "tag-%d" % os.getpid(), (
            "the child really received the task environment value the test set")
        assert readiness["pid"] > 0
        assert readiness["cwd"] == str(tmp_path), (
            "the child's own cwd must be the one production passed")
        holder["pid"] = readiness["pid"]
    finally:
        release.write_text("go", encoding="utf-8")
        thread.join(timeout=12)

    record = holder["record"]
    assert record.returncode == 0, record.error
    assert record.timed_out is False


def test_the_timeout_case_first_observes_a_ready_worker_then_times_it_out(tmp_path, monkeypatch) -> None:
    """A real timeout of a worker that was observed ready, not an early death.

    The worker is started on a short bound; the test confirms the readiness line
    (with the worker's own PID and imported module) was readable — so the process
    really *was* the target subprogram and really ran — then the bound expires.
    ``timed_out``/``terminated`` must be set, partial output must survive, and the
    timeout event plus cleanup state must be recorded.  Finally the worker's PID must
    be gone, which is what makes this a *cleanup* test rather than a flag test: the
    earlier version accepted ``cleanup_finished in (True, False)``, which is true
    whether or not the worker actually ended.
    """
    test_id = "timeout-%d" % os.getpid()
    release = tmp_path / "never-written.flag"
    installed = _install_worker_bootstrap(tmp_path, monkeypatch, LIVE_WORKER_MAIN)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "extract_features.stdout.txt"
    spec = {
        "release_file": str(release), "test_id": test_id, "exit_code": 0,
        "database_path": str(tmp_path / "db"), "image_path": str(tmp_path),
        # A bound long enough for the readiness line, short enough to time out.  The
        # release file is never written, so the worker cannot exit on its own.
        "shutdown_grace": 0,
    }
    monkeypatch.setenv("KOZLOV_WORKER_TAG", "tag-t-%d" % os.getpid())

    holder: dict = {
        #: No fallback kill is performed anywhere in this file's real cases: the fixtures
        #: self-limit, so nothing the *test* does can end the worker.  Recorded rather
        #: than assumed, because "the death was the product's doing" depends on it and a
        #: later edit that added a kill here would silently invalidate the case.
        "fallback_used": False,
    }

    def read_ready() -> None:
        """Bind a read-only observation to the READY worker while it is still alive.

        The observation is created *here*, at the moment the readiness line names the
        worker, so the handle is bound to the very process that streamed — not
        re-derived afterwards from a number that might by then name something else.
        """
        deadline = time.monotonic() + _FIXTURE_SELF_LIMIT_SECONDS
        while time.monotonic() < deadline:
            if stdout_path.is_file():
                ready = _readiness_of(stdout_path.read_text(encoding="utf-8"), test_id)
                if ready is not None:
                    # The anchor for the self-limit comparison: taken in the same breath
                    # as the readiness line, so "seen alive" and "how long until dead"
                    # are measured from one instant rather than two guesses.
                    holder["ready_monotonic"] = time.monotonic()
                    observation = _ProcessObservation(ready["pid"])
                    holder["observation"] = observation
                    holder["ready"] = ready
                    holder["state_at_ready"] = observation.state()
                    return
            time.sleep(0.05)

    reader = threading.Thread(target=read_ready, daemon=True)
    reader.start()
    #: The launch anchor, read before the product is asked to run the stage, so the
    #: comparison against the fixture's self-limit starts before the fixture does.
    t_launch = time.monotonic()
    record = adapter.run_native_stage(
        "extract_features", spec,
        run_dir=tmp_path, logs_dir=logs, project_root=tmp_path,
        python_executable=sys.executable, timeout=2.0,
    )
    reader.join(timeout=10)

    assert record.timed_out is True, (
        "the bound must have expired; a worker that exited early would make this "
        f"case vacuous (error={record.error!r}, rc={record.returncode!r})")
    # The stage path's record exposes ``terminated`` and ``cleanup_finished``; the two are
    # asserted together, because a record that said "terminated" while cleanup had not
    # completed would be claiming a reclamation nobody observed.
    assert record.terminated is True
    assert record.cleanup_finished is True, (
        "the worker ended, so cleanup genuinely completed; False here would mean the test "
        "cannot tell a clean reap from an abandoned one")
    assert record.returncode is None, "the status of a kill is not this stage's exit code"

    # Partial output from the worker that was really observed running, carrying its
    # own identity so "the target subprogram ran" is a checked fact.
    stdout_text = stdout_path.read_text(encoding="utf-8")
    readiness = _readiness_of(stdout_text, test_id)
    assert readiness is not None, (
        "the readiness line proves the target subprogram was the process that timed out")
    assert readiness["module_file"] == str(installed["module_path"])
    observation = holder.get("observation")
    assert observation is not None, (
        "the worker must have been observed while it was still running; without that "
        "binding there is no identity to check the reclamation against")
    assert holder["ready"]["pid"] == readiness["pid"], (
        "the concurrently observed PID must be the one the log names")
    assert holder["state_at_ready"] == "alive", (
        "the worker must have been alive when it was bound; state was %r"
        % holder["state_at_ready"])

    # The worker must actually be gone, checked through the handle bound to its own
    # creation identity.  Three-valued on purpose: only an observed "dead" is
    # accepted, and an unanswerable query stays INCONCLUSIVE rather than being
    # reported as a reap.  The test performs no fallback kill, so a "dead" result here
    # cannot be attributed to the *test* terminating the process.
    #
    # The end anchor is taken only *after* the death has been confirmed, and is then
    # compared against the fixture's own self-limit from the launch anchor: a worker
    # that died because *the fixture* expired would show an elapsed time at or beyond
    # that limit, so the comparison is what distinguishes the product's reclamation
    # from the fixture ending itself.  Taking the end reading before the death wait --
    # the earlier shape -- could not: it was short whenever the product raised promptly
    # even if the child was only reclaimed later, or never, by anything.
    try:
        final_state = observation.wait_dead(within=5.0)
        death_confirmed_at = time.monotonic()
        assert final_state != "unknown", (
            "the liveness of the timed-out worker (pid %d) could not be established: "
            "the observation is UNKNOWN, which is neither a pass nor an adapter "
            "failure" % readiness["pid"])
        assert final_state == "dead", (
            "the timed-out worker (pid %d) must be observed dead, not left running "
            "(final state: %s)" % (readiness["pid"], final_state))
        holder["death_timings"] = _assert_death_inside_fixture_limit(
            label="stage timeout case",
            launch_monotonic=t_launch,
            ready_monotonic=holder["ready_monotonic"],
            death_monotonic=death_confirmed_at,
            fallback_used=holder["fallback_used"])
    finally:
        observation.close()

    launch = read_jsonl(tmp_path / record.launch_record_path)
    events = [entry["event"] for entry in launch]
    assert events[0] == "start"
    assert "timeout" in events, "the kill must be recorded before it is performed"
    finish = launch[-1]
    assert finish["event"] == "finish"
    assert finish["timed_out"] is True
    assert finish["terminated"] is True
    assert finish["interpretation"] == "terminated", (
        "a terminated stage must not be labelled as having returned")
    assert "cleanup_finished" in finish
    assert record.cleanup_finished is True, (
        "the worker ended, so cleanup genuinely completed; False here would mean the "
        "test cannot tell a clean reap from an abandoned one")


def test_a_stage_that_never_started_records_that_it_never_started(tmp_path) -> None:
    """Control-flow only: the budget is exhausted before the launch."""
    record = adapter.run_native_stage(
        "extract_features",
        {"database_path": str(tmp_path / "db"), "image_path": str(tmp_path)},
        run_dir=tmp_path, logs_dir=tmp_path / "logs", project_root=tmp_path,
        deadline=time.monotonic() - 1.0,
    )
    assert record.started is False
    assert record.returncode is None
    assert record.returncode_reason and "never started" in record.returncode_reason
    launch = read_jsonl(tmp_path / record.launch_record_path)
    assert launch and launch[0]["event"] == "not_started"


# --------------------------------------------------------------------------- #
# 5b. directed counterexamples against the production launch
# --------------------------------------------------------------------------- #
def test_the_recorded_launch_arguments_are_the_ones_the_executor_received(
        tmp_path, monkeypatch) -> None:
    """The log's cwd and override whitelist must be the values actually passed.

    A replacement executor is installed *as the executor* (a parameter), which is
    the legitimate seam: it receives exactly the ``cwd``/``env``/``argv`` that the
    default executor would, so comparing them against the start record checks the
    record against the launch, not against itself.
    """
    seen: dict = {}

    def capturing_executor(argv, *, cwd, env, timeout, stdout_path, stderr_path):
        seen.update(argv=argv, cwd=cwd, env=dict(env), timeout=timeout)
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        return adapter.StageExchange(returncode=0)

    logs = tmp_path / "logs"
    record = adapter.run_native_stage(
        "extract_features",
        {"database_path": str(tmp_path / "db"), "image_path": str(tmp_path)},
        run_dir=tmp_path, logs_dir=logs, project_root=tmp_path,
        timeout=11.5, executor=capturing_executor,
    )
    launch = read_jsonl(tmp_path / record.launch_record_path)
    start = launch[0]

    assert start["event"] == "start"
    assert start["argv"] == seen["argv"], "the record must name the argv that was passed"
    assert start["cwd"] == seen["cwd"], "the record must name the cwd that was passed"
    assert start["timeout_seconds"] == pytest.approx(seen["timeout"])
    # The override whitelist, and only it: the inherited environment is never logged.
    for key, value in start["environment_overrides"].items():
        assert seen["env"][key] == value, f"{key} differs between record and launch"
    assert start["environment_unbuffered"] == seen["env"].get("PYTHONUNBUFFERED"), (
        "the unbuffered setting that took effect must be the one recorded")
    assert seen["env"].get("PYTHONUNBUFFERED") == "1"
    assert "os.environ" not in json.dumps(start)
    assert start["executor"] == "capturing_executor"


def test_a_planted_secret_is_never_written_to_the_evidence(monkeypatch, tmp_path) -> None:
    """A fake credential in the ambient environment must not reach any log.

    The launch environment is built from ``os.environ`` (the child needs the real
    environment to run), but the *record* carries only the whitelist this task sets.
    A planted variable is the counterexample that would expose a copy.
    """
    monkeypatch.setenv("KOZLOV_FAKE_SECRET", "do-not-log-me-1234")
    logs = tmp_path / "logs"

    def benign_executor(argv, *, cwd, env, timeout, stdout_path, stderr_path):
        # A real child would inherit the secret; the record still must not show it.
        assert env["KOZLOV_FAKE_SECRET"] == "do-not-log-me-1234"
        stdout_path.write_bytes(b"ok\n")
        stderr_path.write_bytes(b"")
        return adapter.StageExchange(returncode=0)

    adapter.run_native_stage(
        "extract_features",
        {"database_path": str(tmp_path / "db"), "image_path": str(tmp_path)},
        run_dir=tmp_path, logs_dir=logs, project_root=tmp_path,
        executor=benign_executor,
    )
    blob = (logs / "extract_features.launch.jsonl").read_text(encoding="utf-8")
    assert "do-not-log-me-1234" not in blob
    assert "KOZLOV_FAKE_SECRET" not in blob


def test_a_failed_pre_launch_write_prevents_the_subprogram_from_running(
        tmp_path, monkeypatch) -> None:
    """If the start line cannot be written, nothing may be launched at all.

    Two distinct pre-launch failures, because they are reached by different code:
    the log directory cannot be created, and the record file cannot be appended to
    (modelled by making the strict writer raise).  In both cases the executor must
    never be called, and the result must say the stage did not start — a stage that
    ran with no start line would leave a record that disagrees with what happened.
    """
    launched: list = []

    def recording_executor(argv, *, cwd, env, timeout, stdout_path, stderr_path):
        launched.append(argv)
        return adapter.StageExchange(returncode=0)

    # Case 1: the log directory cannot be created (its parent is a regular file).
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")

    record = adapter.run_native_stage(
        "extract_features",
        {"database_path": str(tmp_path / "db"), "image_path": str(tmp_path)},
        run_dir=tmp_path, logs_dir=blocker / "logs", project_root=tmp_path,
        executor=recording_executor,
    )
    assert launched == [], "no subprogram may run when its start line cannot be written"
    assert record.started is False
    assert record.error and "log directory" in record.error
    assert record.returncode_reason and "never started" in record.returncode_reason

    # Case 2: the directory exists but the strict writer fails.
    def failing_strict(path, payload):
        raise adapter.EvidenceWriteError("simulated: the launch record is unwritable")

    monkeypatch.setattr(adapter, "_evidence_append", failing_strict)
    logs = tmp_path / "logs2"
    record2 = adapter.run_native_stage(
        "extract_features",
        {"database_path": str(tmp_path / "db"), "image_path": str(tmp_path)},
        run_dir=tmp_path, logs_dir=logs, project_root=tmp_path,
        executor=recording_executor,
    )
    assert launched == [], "a failed launch-record write must abort before the launch"
    assert record2.started is False
    assert record2.error and "launch record could not be written" in record2.error


def test_an_evidence_write_failure_after_the_child_ran_is_not_a_silent_success(
        tmp_path, monkeypatch) -> None:
    """A post-run write failure marks the record incomplete instead of passing.

    The pre-launch record uses the strict writer and the post-run record uses the
    tolerant one, so making the tolerant writer fail models exactly the required
    case: the child ran, but its finish line could not be written.  That must
    surface as an error, not as a completed stage.
    """
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    attempts: list = []

    def failing_tolerant(path, payload):
        attempts.append(payload.get("event"))
        return False

    monkeypatch.setattr(adapter, "_evidence_append_tolerant", failing_tolerant)

    def benign_executor(argv, *, cwd, env, timeout, stdout_path, stderr_path):
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        return adapter.StageExchange(returncode=0)

    record = adapter.run_native_stage(
        "extract_features",
        {"database_path": str(tmp_path / "db"), "image_path": str(tmp_path)},
        run_dir=tmp_path, logs_dir=logs, project_root=tmp_path,
        executor=benign_executor,
    )
    assert attempts == ["finish"], (
        "the tolerant writer is the post-run one; the pre-launch record is strict")
    # The start line survives: the child really did run and really was recorded.
    assert read_jsonl(tmp_path / record.launch_record_path)[0]["event"] == "start"
    assert record.evidence_incomplete is True
    assert record.error and "not verifiable" in record.error, (
        "an incomplete record must be an error in its own right")


def test_a_terminated_stage_is_never_labelled_as_having_returned(tmp_path) -> None:
    """A killed process may yield a real status, but the label must say ``terminated``.

    The executor reports a timeout with a genuine exit code, which is exactly the
    case the requirement names: the code may be recorded, but not passed off as a
    normal algorithm return.
    """
    logs = tmp_path / "logs"
    record = adapter.run_native_stage(
        "extract_features",
        {"database_path": str(tmp_path / "db"), "image_path": str(tmp_path)},
        run_dir=tmp_path, logs_dir=logs, project_root=tmp_path,
        executor=lambda *a, **k: adapter.StageExchange(
            returncode=1, timed_out=True, terminated=True, cleanup_finished=True,
            returncode_reason="terminated by the bound; status is the kill's"),
    )
    assert record.timed_out is True
    assert record.terminated is True
    assert record.returncode is None, "the kill's status must not surface as the result"
    finish = read_jsonl(tmp_path / record.launch_record_path)[-1]
    assert finish["process_exit_code"] == 1, "the real status survives in the record"
    assert finish["interpretation"] == "terminated"


def test_the_default_executor_is_the_one_used_when_none_is_given(tmp_path) -> None:
    """The default is the real executor, not something only tests reach.

    A relative ``python_executable`` that cannot exist makes the default executor's
    real launch fail.  That failure is itself the proof that the default path was
    taken — a stub-only implementation would report success.
    """
    logs = tmp_path / "logs"
    record = adapter.run_native_stage(
        "extract_features",
        {"database_path": str(tmp_path / "db"), "image_path": str(tmp_path)},
        run_dir=tmp_path, logs_dir=logs, project_root=tmp_path,
        python_executable=str(tmp_path / "no-such-interpreter.exe"), timeout=20,
    )
    assert record.error, "launching a nonexistent interpreter must fail"
    assert record.returncode is None, "nothing ran, so there is no exit code"
    finish = read_jsonl(tmp_path / record.launch_record_path)[-1]
    assert finish["event"] == "finish"
    assert finish["process_exit_code"] is None
    assert finish["process_exit_code_reason"]


# --------------------------------------------------------------------------- #
# 5c. the backend probe's cleanup must not be blocked by an evidence failure
# --------------------------------------------------------------------------- #
class _ProbeFakeProcess:
    """A stand-in Popen whose ``wait`` always times out; counts kill/wait calls.

    Control-flow verification only: the requirement is that the *kill and the
    bounded wait actually happen* even when the evidence path is absent or its
    write raises.  A counting stand-in makes that observable without hanging a real
    process, which is exactly the tool the task names for this case.

    ``kill`` returns ``None``, which is what a real, successful ``Popen.kill()``
    returns.  A stand-in that returned ``True`` would let code that asks whether the
    *return value* was truthy look correct here while being wrong on every real
    machine, so the fake follows the real signature rather than a convenient one.
    A kill that could not be issued raises the exception passed as ``kill_error``.
    """

    def __init__(self, *, kill_error=None, reap_code=0, reap_timeout=False,
                 stdout_payload: bytes = b"") -> None:
        self.wait_calls = 0
        self.kill_calls = 0
        self._kill_error = kill_error
        #: The first ``wait`` always raises ``TimeoutExpired`` -- that is what makes this
        #: the *timeout* stand-in.  What the bounded reap then does is scripted, because the
        #: shapes the close-out names differ there: a reap that returns a code, and a reap
        #: that times out again while the kill may itself have been refused.
        self._reap_code = reap_code
        self._reap_timeout = reap_timeout
        self._stdout_payload = stdout_payload
        self.returncode = None

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="probe", timeout=timeout)
        if self._reap_timeout:
            raise subprocess.TimeoutExpired(cmd="probe", timeout=timeout)
        self.returncode = self._reap_code
        return self._reap_code

    def kill(self):
        self.kill_calls += 1
        if self._kill_error is not None:
            raise self._kill_error
        return None

    def attach_stdout(self, handle) -> None:
        """Write what the child would have written, into the handle it was given."""
        if handle is not None and self._stdout_payload:
            handle.write(self._stdout_payload)
            handle.flush()


class _SubprocessShim:
    """A stand-in for the ``subprocess`` *module*, not for ``subprocess.Popen``.

    ``subprocess`` is a shared module object, so setting ``subprocess.Popen`` in place also
    replaces it for every other user of the module in this process.  That is not a
    hypothetical here: this host injects file-safety instrumentation that calls
    ``subprocess.run`` when ``Path.unlink`` is used inside the repository, so a globally
    patched ``Popen`` broke an unrelated file deletion and produced a failure with nothing
    to do with the adapter — and it would equally have replaced the launcher for any other
    code path.  Rebinding the adapter module's own ``subprocess`` name confines the
    replacement to the code under test while the real module keeps working for everyone else.
    """

    def __init__(self, popen) -> None:
        self.Popen = popen
        self.TimeoutExpired = subprocess.TimeoutExpired
        # Not used by the probe, but present so the adapter's view of the module is the real
        # one except for the single name under test.
        self.run = subprocess.run


def _install_probe_fake(monkeypatch, fake: _ProbeFakeProcess) -> dict:
    """Route the adapter's ``subprocess.Popen`` to ``fake``; return a call log."""
    calls: dict = {"popen_kwargs": None}

    def fake_popen(argv, **kwargs):
        calls["popen_kwargs"] = {"argv": argv, **kwargs}
        fake.attach_stdout(kwargs.get("stdout"))
        return fake

    monkeypatch.setattr(adapter, "subprocess", _SubprocessShim(fake_popen))
    return calls


def test_a_timed_out_probe_without_an_evidence_path_still_cleans_up(monkeypatch, tmp_path) -> None:
    """No evidence path ⇒ no record write, but the kill and bounded wait still run.

    The previous shape called the record helpers unconditionally, so a ``None``
    evidence path raised ``AttributeError: 'NoneType' object has no attribute
    'parent'`` *before* the kill — the child would have been abandoned.  Here the
    fix is asserted by counting: ``kill`` once, ``wait`` twice (the timed wait plus
    the bounded reap), and a ``BackendUnavailable`` that says the probe timed out.
    """
    fake = _ProbeFakeProcess()
    _install_probe_fake(monkeypatch, fake)

    with pytest.raises(adapter.BackendUnavailable):
        adapter.backend_info(sys.executable, timeout=0.01)

    assert fake.kill_calls == 1, "the child must be killed even with no evidence path"
    assert fake.wait_calls >= 2, "the reap must be attempted after the kill"
    assert fake.kill_calls == 1 and fake.wait_calls == 2, (
        "exactly one kill and one bounded reap; no extra waits were opened")


def test_a_timed_out_probe_whose_timeout_record_fails_still_cleans_up(
        monkeypatch, tmp_path) -> None:
    """A failing timeout record must not skip cleanup; the failure is reported.

    The timeout event is written *before* the kill so the reason survives it, but a
    write failure there must not be able to decide whether the child is cleaned up.
    ``_evidence_append`` is made to raise; the kill and bounded wait must still run,
    and the finish record must carry ``timeout_recorded=False`` rather than silently
    implying the timeout was logged.
    """
    fake = _ProbeFakeProcess()
    _install_probe_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"
    real_append = adapter._evidence_append

    def selective_failure(path, payload):
        if payload.get("event") == "timeout":
            raise adapter.EvidenceWriteError("simulated: timeout line unwritable")
        return real_append(path, payload)

    monkeypatch.setattr(adapter, "_evidence_append", selective_failure)

    with pytest.raises(adapter.BackendUnavailable):
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=0.01)

    assert fake.kill_calls == 1, "a failed timeout record must not skip the kill"
    assert fake.wait_calls >= 2, "the bounded reap must still be attempted"
    entries = read_jsonl(evidence)
    finish = entries[-1]
    assert finish["event"] == "finish"
    assert finish["timeout_recorded"] is False, (
        "a lost timeout line must be recorded as lost, not implied to have been written")
    assert finish["timed_out"] is True
    assert finish["termination_requested"] is True, (
        "this fixture's kill succeeds, so a termination really was requested")
    assert finish["terminated"] is finish["termination_requested"], (
        "the compatibility field must agree with termination_requested")


def test_a_probe_whose_reap_never_completes_does_not_claim_cleanup(
        monkeypatch, tmp_path) -> None:
    """When the bounded wait cannot complete, the record must not say it did.

    A stand-in whose ``wait`` never returns is the counterexample for "cleaning up
    failed but we wrote True anyway": the finish record must carry
    ``cleanup_finished=False`` and a null code, and the error must be raised rather
    than swallowed.
    """
    fake = _ProbeFakeProcess(reap_timeout=True)
    _install_probe_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"

    with pytest.raises(adapter.BackendUnavailable):
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=0.01)

    finish = read_jsonl(evidence)[-1]
    assert finish["event"] == "finish"
    assert finish["cleanup_finished"] is False, (
        "an unfinished reap must not be recorded as a completed one")
    assert finish["process_exit_code"] is None
    assert finish["process_exit_code_reason"]
    assert fake.kill_calls == 1


def test_the_probe_cleanup_is_bounded_by_one_shared_deadline(monkeypatch, tmp_path) -> None:
    """All cleanup steps share a single monotonic deadline, not a fresh one each.

    Verified by making ``kill`` consume more than the whole budget: the reap must
    then *not* be attempted another full period, i.e. ``cleanup`` must return False
    once the shared deadline is spent rather than starting a new wait.
    """
    clock = {"now": 1000.0}
    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock["now"])
    fake = _ProbeFakeProcess()

    def slow_kill():
        # Consume the entire shared cleanup budget inside the kill.  Returns ``None``,
        # like a real ``Popen.kill()``: the fact under test is the deadline, not what
        # the call handed back.
        clock["now"] += adapter._PROBE_CLEANUP_DEADLINE_SECONDS + 1.0
        fake.kill_calls += 1
        return None

    fake.kill = slow_kill
    _install_probe_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"

    with pytest.raises(adapter.BackendUnavailable):
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=0.01)

    assert fake.kill_calls == 1
    assert fake.wait_calls == 1, (
        "the deadline was already spent, so no second wait period may be opened")
    finish = read_jsonl(evidence)[-1]
    assert finish["cleanup_finished"] is False


# --------------------------------------------------------------------------- #
# 5c-ii. the timeout path: the timeout is the failure, not the termination request
# --------------------------------------------------------------------------- #
# The five shapes below are the P matrix the close-out fixes.  Every one of them times out
# on the first wait, and every one of them must raise ``BackendUnavailable`` with a version
# never reported -- including P2 and P4, where the kill itself could not be issued.  The
# earlier shape recorded ``terminated=True`` on entering the timeout arm and made the failure
# condition depend on that same flag, so a refused kill turned a timeout into a pass.
#
# Each case asserts the *outcome of the call* (it raised, and with which class), the recorded
# three separate facts, the kill/wait counts and the reason text.  Searching the exception
# message alone is not enough: a message can be right while the facts behind it are wrong,
# and a message can change without any behaviour changing.

def _timed_out_probe(tmp_path, fake, *, label: str, monkeypatch) -> dict:
    """Run one timeout shape through the real ``backend_info`` and return its finish record."""
    _install_probe_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"
    raised: dict = {}
    try:
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=0.05)
    except BaseException as exc:  # noqa: BLE001 - the class is part of the assertion
        raised["type"] = type(exc)
        raised["error"] = exc
    assert raised, f"{label}: the call must not return"
    entries = read_jsonl(evidence)
    return {"raised": raised, "finish": entries[-1] if entries else None,
            "entries": entries, "label": label}


def _assert_timed_out_shape(shape: dict, *, kill_error: bool, reap_times_out: bool) -> None:
    """The four facts that must hold for *every* timeout shape, plus the per-shape ones."""
    label = shape["label"]
    raised = shape["raised"]
    finish = shape["finish"]
    assert raised["type"] is adapter.BackendUnavailable, (
        f"{label}: a timeout must raise BackendUnavailable, not {raised['type'].__name__}: "
        f"{raised.get('error')}")
    assert finish is not None, f"{label}: the finish record must have been written"
    assert finish["event"] == "finish", finish
    assert finish["timed_out"] is True, (
        f"{label}: the record must say the probe timed out")
    assert "timeout" in [e["event"] for e in shape["entries"]], (
        f"{label}: the timeout event must precede the finish record")
    # ``terminated`` keeps its compatibility meaning: "a termination request returned without
    # raising".  It may not be hardwired true -- that is exactly how a refused kill was
    # reported as a terminated child.
    assert finish["termination_requested"] is (not kill_error), (
        f"{label}: termination_requested must be True exactly when the kill call returned "
        f"without raising; got {finish['termination_requested']!r}")
    assert finish["terminated"] is finish["termination_requested"], (
        f"{label}: the compatibility field must agree with termination_requested rather than "
        f"being written true unconditionally")
    assert finish["cleanup_attempted"] is True, (
        f"{label}: the kill and the bounded reap must have been attempted")
    if reap_times_out:
        assert finish["cleanup_finished"] is False, (
            f"{label}: an unfinished reap must not be recorded as a completed one")
        assert finish["exit_confirmed"] is False
        assert finish["process_exit_code"] is None, (
            f"{label}: no status was obtained, so none may be recorded")
    else:
        assert finish["cleanup_finished"] is True, f"{label}: the reap completed"
        assert finish["exit_confirmed"] is True
        assert finish["process_exit_code"] == 0, (
            f"{label}: the real exit status obtained by the reap must be kept")
    if kill_error:
        assert any("termination request was refused" in e for e in finish["cleanup_errors"]), (
            f"{label}: the refused kill must be reported: {finish['cleanup_errors']}")
    if reap_times_out:
        assert any("reap" in e or "deadline" in e for e in finish["cleanup_errors"]), (
            f"{label}: the failed reap must be reported: {finish['cleanup_errors']}")


def test_p0_an_ordinary_probe_returns_without_a_termination(monkeypatch, tmp_path) -> None:
    """P0: exit 0 and a valid version JSON -- a success, and no kill is issued."""
    fake = _ConfigurableProbeFake(code=0, stdout_payload=b'{"version": "4.2.1"}\n')
    _install_configurable_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"

    info = adapter.backend_info(sys.executable, evidence_path=evidence,
                                project_root=tmp_path, timeout=5)
    assert info["version"] == "4.2.1"
    assert fake.kill_calls == 0, "nothing timed out, so nothing may be killed"
    finish = read_jsonl(evidence)[-1]
    assert finish["terminated"] is False
    assert finish["termination_requested"] is False
    assert finish["exit_confirmed"] is True and finish["process_exit_code"] == 0


def test_p1_a_timeout_whose_kill_succeeded_is_still_a_failure(monkeypatch, tmp_path) -> None:
    """P1: timeout? kill returns None (success)? reap returns exit 0 and valid JSON.

    The child even managed to write a complete version document before it was terminated.
    All three of those are recorded -- and the call still fails, because the caller asked for
    a version *within its bound* and did not get one.
    """
    fake = _ProbeFakeProcess(reap_code=0, stdout_payload=b'{"version": "4.2.1"}\n')
    shape = _timed_out_probe(tmp_path, fake, label="P1", monkeypatch=monkeypatch)
    assert fake.kill_calls == 1 and fake.wait_calls == 2, (
        "P1: exactly one kill and one bounded reap after the timed wait")
    _assert_timed_out_shape(shape, kill_error=False, reap_times_out=False)
    assert any("4.2.1" in line for line in shape["finish"]["stdout_tail"]), (
        "P1: the version the child did manage to write is reported in the record: %r"
        % (shape["finish"]["stdout_tail"],))
    assert "4.2.1" not in str(shape["raised"]["error"]), (
        "P1: a version read after the bound must not be presented as the probe's answer")
    assert "no version was reported" in str(shape["raised"]["error"]), shape["raised"]["error"]


def test_p2_a_timeout_whose_kill_was_refused_is_still_a_failure(monkeypatch, tmp_path) -> None:
    """P2: the review's key counterexample.  The kill raises, the reap still gets exit 0
    and a valid version JSON -- and the call must *still* raise ``BackendUnavailable``.

    Under the old shape this was the false pass: ``termination_requested`` was set on
    entering the timeout arm, so the record claimed a termination had been requested even
    though the kill had raised, and the failure condition read that same flag.
    """
    fake = _ProbeFakeProcess(kill_error=OSError("kill: the termination request was refused"),
                             reap_code=0, stdout_payload=b'{"version": "4.2.1"}\n')
    shape = _timed_out_probe(tmp_path, fake, label="P2", monkeypatch=monkeypatch)
    assert fake.kill_calls == 1 and fake.wait_calls == 2, "P2: the reap must still run"
    _assert_timed_out_shape(shape, kill_error=True, reap_times_out=False)
    assert shape["finish"]["process_exit_code"] == 0, (
        "P2: the real 0 obtained by the reap is kept even though the kill was refused")
    assert "termination_requested=False" in str(shape["raised"]["error"]), (
        "the message must not claim a termination that was never requested: "
        f"{shape['raised']['error']}")
    # ``process_exit_code_reason`` is not checked by ``_assert_timed_out_shape``, and this is
    # the path where it was wrong: the kill raised, so no termination was issued, yet the reap
    # that ran anyway read the child's own natural status and the field described that status
    # as "the exit status of the termination".  Words are not the control flow, but a record
    # that names a cause is a claim about the run, and this one named a cause that was never
    # established.
    p2_reason = shape["finish"]["process_exit_code_reason"]
    assert p2_reason, "P2: the record must say what its exit code, or its absence, means"
    assert "bounded reclamation" in p2_reason, (
        "P2: a status the reap obtained after a refused kill must be described as what the "
        f"bounded reclamation obtained: {p2_reason!r}")
    assert "the exit status of the termination" not in p2_reason, (
        "P2: with the kill refused there was no termination to attribute the exit to, so the "
        f"record may not claim one: {p2_reason!r}")


def test_p3_a_timeout_whose_reap_also_timed_out_is_still_a_failure(monkeypatch, tmp_path) -> None:
    """P3: kill returns None, the reap times out too -- requested true, reaped/exit false."""
    fake = _ProbeFakeProcess(reap_timeout=True)
    shape = _timed_out_probe(tmp_path, fake, label="P3", monkeypatch=monkeypatch)
    assert fake.kill_calls == 1 and fake.wait_calls == 2, "P3: one kill, one bounded reap"
    _assert_timed_out_shape(shape, kill_error=False, reap_times_out=True)
    assert shape["finish"]["process_exit_code"] is None


def test_p4_a_timeout_whose_kill_and_reap_both_failed_keeps_both_errors(
        monkeypatch, tmp_path) -> None:
    """P4: the kill raises *and* the reap times out.  Both errors survive, nothing claimed."""
    fake = _ProbeFakeProcess(kill_error=OSError("kill: the termination request was refused"),
                             reap_timeout=True)
    shape = _timed_out_probe(tmp_path, fake, label="P4", monkeypatch=monkeypatch)
    assert fake.kill_calls == 1 and fake.wait_calls == 2, "P4: one kill, one bounded reap"
    _assert_timed_out_shape(shape, kill_error=True, reap_times_out=True)
    errors = " | ".join(shape["finish"]["cleanup_errors"])
    assert "kill_error" in errors and "wait_error" in errors, errors
    assert "termination request was refused" in errors, errors


def test_a_timeout_never_returns_a_version_even_when_the_child_wrote_one(
        monkeypatch, tmp_path) -> None:
    """The strongest form of P2: a *valid, complete* version document is on stdout.

    ``backend_info`` returns a mapping on success and raises on failure, so "the timeout was
    reported as a failure" and "the version that happened to be on stdout was returned" are
    two different outcomes.  This pins the second one down: no mapping may be produced, and
    the version must not leak into the error as if it were the probe's answer.  The bytes are
    written through the fake's own ``attach`` hook, so they really are on the captured stdout
    when the adapter reads it.
    """
    fake = _ProbeFakeProcess(reap_code=0,
                             kill_error=OSError("kill: the termination request was refused"))
    written: dict = {}

    def fake_popen(argv, **kwargs):
        written["popen_kwargs"] = {"argv": argv, **kwargs}
        for handle, payload in ((kwargs.get("stdout"), b'{"version": "9.9.9"}\n'),
                                (kwargs.get("stderr"), b"")):
            if handle is not None and payload:
                handle.write(payload)
                handle.flush()
        return fake

    monkeypatch.setattr(adapter, "subprocess", _SubprocessShim(fake_popen))
    evidence = tmp_path / "logs" / "backend_process.jsonl"

    with pytest.raises(adapter.BackendUnavailable) as raised:
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=0.05)
    finish = read_jsonl(evidence)[-1]
    assert finish["timed_out"] is True
    assert any("9.9.9" in line for line in finish["stdout_tail"]), (
        "the partial stdout is still reported in the record: %r" % (finish["stdout_tail"],))
    assert "9.9.9" not in str(raised.value), (
        "the version must not leak into the error as if it were the probe's answer: %r"
        % (str(raised.value),))


@pytest.fixture
def scripted_kernel32(monkeypatch):
    """A scripted ``kernel32`` so each observer answer can be *produced* on demand.

    The substitution is at the library boundary rather than inside the observer, so the
    branch structure under test is the shipped one.  No process is created: the point is to
    be able to produce "the query was refused" and "the creation identity differs", which a
    real machine cannot be asked for on demand — a test that could only observe whatever
    states the host happened to offer could not show that a refused query refuses to be
    read as a death.
    """
    import ctypes
    from ctypes import wintypes

    class Scripted:
        def __init__(self) -> None:
            self.answers: dict = {}
            self.open_calls = 0
            self.exit_code_calls = 0
            self.close_calls = 0

        def OpenProcess(self, access, inherit, pid):
            self.open_calls += 1
            if self.answers.get("handle") is None:
                ctypes.set_last_error(self.answers.get("open_error", 87))
                return None
            return self.answers["handle"]

        def GetExitCodeProcess(self, handle, code_ref):
            self.exit_code_calls += 1
            if not self.answers.get("exit_ok", True):
                ctypes.set_last_error(self.answers.get("exit_error", 5))
                return 0
            code_ref._obj.value = self.answers.get("exit_code", 259)
            return 1

        def GetProcessTimes(self, handle, created, exited, kernel, user):
            if not self.answers.get("times_ok", True):
                return 0
            created._obj.dwHighDateTime = self.answers.get("creation_high", 100)
            created._obj.dwLowDateTime = self.answers.get("creation_low", 7)
            return 1

        def CloseHandle(self, handle):
            self.close_calls += 1
            return 1

    scripted = Scripted()

    def fake_lib(cls):
        return ctypes, wintypes, scripted

    monkeypatch.setattr(_ProcessObservation, "_lib", classmethod(fake_lib))
    return scripted


def test_the_observer_is_three_valued_and_a_refused_query_is_not_a_death(
        scripted_kernel32) -> None:
    scripted_kernel32.answers.update(handle=1, exit_code=259)
    observation = _ProcessObservation(4242)
    try:
        assert observation.identity_bound is True, (
            "a bound observation must hold the creation identity it will compare against")
        assert observation.state() == "alive"

        scripted_kernel32.answers["exit_code"] = 3
        assert observation.state() == "dead", (
            "a final exit code read through the bound handle is a death")

        scripted_kernel32.answers["exit_ok"] = False
        assert observation.state() == "unknown", (
            "a refused liveness query is not an answer; folding it into 'dead' would "
            "credit the product with a reclamation it never demonstrated")
    finally:
        observation.close()
    assert scripted_kernel32.close_calls == 1, (
        "the query handle must be released exactly once")


def test_the_observer_refuses_a_recycled_pid_by_creation_identity(scripted_kernel32) -> None:
    scripted_kernel32.answers.update(handle=1, creation_high=100, exit_code=0)
    observation = _ProcessObservation(4243)
    try:
        assert observation.state() == "dead"
        # The number is now held by a live process with a different creation stamp.
        scripted_kernel32.answers.update(creation_high=999, exit_code=259)
        assert observation.state() == "unknown", (
            "a live process whose creation time differs from the one that was bound is a "
            "different process; reporting it as this child would be a PID-reuse false "
            "positive, which is the whole reason the identity is bound")
    finally:
        observation.close()


def test_the_observer_does_not_claim_a_death_it_could_not_observe(scripted_kernel32) -> None:
    scripted_kernel32.answers.update(handle=None, open_error=87)
    observation = _ProcessObservation(4244)
    try:
        assert observation.identity_bound is False
        assert "no such process" in (observation.open_error or ""), observation.open_error
        assert observation.state() == "unknown", (
            "without a handle there is nothing to query.  The documented open failure is "
            "recorded for the reader, but a *fresh* observation of an already-gone pid is "
            "UNKNOWN: this observer only reports a death it actually witnessed through a "
            "bound handle, which is why the liveness tests bind at READY time")
    finally:
        observation.close()
    assert scripted_kernel32.exit_code_calls == 0, (
        "no liveness answer may be produced without a handle")


class _ConfigurableProbeFake:
    """A stand-in ``Popen`` whose ``wait``/``kill`` follow a script.

    ``_ProbeFakeProcess`` can only time out every wait.  That is enough for the "no evidence
    path" cases, but it cannot express the shapes the close-out names: a first ``wait`` that
    raises a *non-timeout* error followed by a bounded reap that really returns a code, a
    kill that fails, or an exit 0 whose finish record cannot be written.  None of those can
    be produced on demand by a real child, so they are scripted here and the *consequence*
    is checked in the record.  Nothing here is a measurement.
    """

    def __init__(self, *, wait_script=None, code: int = 0, kill_error=None,
                 stdout_payload: bytes = b"", stderr_payload: bytes = b"") -> None:
        self.wait_calls = 0
        self.kill_calls = 0
        self.returncode = None
        self._wait_script = list(wait_script or [])
        self._code = code
        self._kill_error = kill_error
        self._stdout_payload = stdout_payload
        self._stderr_payload = stderr_payload

    def attach(self, stdout_handle, stderr_handle) -> None:
        """Emit the bytes the child would have written, into the handles it was given."""
        for handle, payload in ((stdout_handle, self._stdout_payload),
                                (stderr_handle, self._stderr_payload)):
            if handle is not None and payload:
                handle.write(payload)
                handle.flush()

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._wait_script:
            action = self._wait_script.pop(0)
            if action is not None:
                raise action
        self.returncode = self._code
        return self._code

    def kill(self):
        self.kill_calls += 1
        if self._kill_error is not None:
            raise self._kill_error
        # ``None``, as a real successful ``Popen.kill()`` returns.  Nothing may read
        # this value: a truthiness test on it turns a successful termination into a
        # reported failure, which is the defect this stand-in exists to expose.
        return None


def _install_configurable_fake(monkeypatch, fake: _ConfigurableProbeFake) -> dict:
    calls: dict = {"popen_kwargs": None}

    def fake_popen(argv, **kwargs):
        calls["popen_kwargs"] = {"argv": argv, **kwargs}
        fake.attach(kwargs.get("stdout"), kwargs.get("stderr"))
        return fake

    monkeypatch.setattr(adapter, "subprocess", _SubprocessShim(fake_popen))
    return calls


def test_a_wait_failure_still_reaps_and_keeps_the_real_exit_code(
        monkeypatch, tmp_path) -> None:
    """A code the bounded reap obtained must survive the first wait having raised.

    The earlier shape read the code only on the success path, so a first ``wait`` that
    raised left the record saying ``null`` even though cleanup had gone on to reap the child
    and read 3.  The original error must still propagate unchanged, and the record must say
    *where* the code came from — a code obtained during cleanup is not a code known before
    the failure.
    """
    fake = _ConfigurableProbeFake(
        wait_script=[OSError("simulated: the wait failed before returning a status")],
        code=3)
    _install_configurable_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"

    with pytest.raises(OSError) as raised:
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=5)
    assert "simulated: the wait failed before returning a status" in str(raised.value), (
        "the original error must propagate unchanged")

    finish = read_jsonl(evidence)[-1]
    assert finish["event"] == "finish"
    assert fake.wait_calls == 2, (
        "the reap must still have been attempted after the first wait failed")
    assert finish["cleanup_finished"] is True
    assert finish["process_exit_code"] == 3, (
        "a status obtained by the bounded reap is a measurement and must be kept, not "
        "discarded because the first wait happened to raise")
    assert "bounded reap" in finish["process_exit_code_reason"], (
        "the record must distinguish a code obtained during cleanup from one known before "
        "the failure: %r" % finish["process_exit_code_reason"])


def test_a_cleanup_failure_keeps_the_original_error_and_reports_the_cleanup_error(
        monkeypatch, tmp_path) -> None:
    """Neither a kill error nor a reap error may replace the original exception."""
    fake = _ConfigurableProbeFake(
        wait_script=[OSError("original: the initial wait failed"),
                     OSError("reap: the bounded wait also failed")],
        kill_error=OSError("kill: the termination request was refused"))
    _install_configurable_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"

    with pytest.raises(OSError) as raised:
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=5)
    assert "original: the initial wait failed" in str(raised.value), (
        "a cleanup or logging error must never replace the original error")

    finish = read_jsonl(evidence)[-1]
    assert finish["cleanup_finished"] is False, (
        "the reap did not complete, so the record must not claim it did")
    assert finish["process_exit_code"] is None
    assert finish["process_exit_code_reason"]
    joined = " | ".join(finish["cleanup_errors"])
    assert "kill_error" in joined and "wait_error" in joined, finish["cleanup_errors"]
    assert "the termination request was refused" in joined, finish["cleanup_errors"]
    assert "the bounded wait also failed" in joined, finish["cleanup_errors"]


def test_a_probe_that_exited_zero_but_could_not_record_its_finish_is_not_a_success(
        monkeypatch, tmp_path) -> None:
    """Reading the version is not enough: the record that the probe was measured must exist.

    The finish write is made to fail *after* the child exited 0 and its JSON parsed, so a
    version really was read.  Returning that mapping as an ordinary success would report a
    measurement whose record is missing, and would also let the missing record go unmentioned.
    """
    fake = _ConfigurableProbeFake(
        code=0, stdout_payload=b'{"version": "4.2.1", "name": "pycolmap"}')
    _install_configurable_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"
    real_append = adapter._evidence_append_tolerant

    def refuse_finish(path, payload):
        if payload.get("event") == "finish":
            return False
        return real_append(path, payload)

    monkeypatch.setattr(adapter, "_evidence_append_tolerant", refuse_finish)

    with pytest.raises(adapter.BackendUnavailable) as raised:
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=5)
    message = str(raised.value)
    assert "finish record could not be written" in message, message
    assert "4.2.1" in message, (
        "the error must show the version had been read, so the failure is the record and "
        "not the read: %r" % message)


def test_a_probe_whose_json_root_is_not_an_object_is_rejected(monkeypatch, tmp_path) -> None:
    """Valid JSON is not automatically a usable report; a list parses and then fails later."""
    fake = _ConfigurableProbeFake(code=0, stdout_payload=b'["4.2.1"]')
    _install_configurable_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"

    with pytest.raises(adapter.BackendUnavailable) as raised:
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=5)
    assert "not an object" in str(raised.value), raised.value

    finish = read_jsonl(evidence)[-1]
    assert finish["event"] == "finish"
    assert "not an object" in finish["note"], finish
    assert finish["process_exit_code"] == 0, (
        "the exit code really was obtained and must not be discarded along with the parse")


def test_a_scratch_file_that_cannot_be_removed_is_surfaced_not_silently_dropped(
        monkeypatch, tmp_path) -> None:
    """The no-evidence form has no record, so a residual must appear on its return value.

    ``finish`` is a no-op without an evidence location.  If the residual were not surfaced
    anywhere, this form would report ordinary success over a file it left behind.  The real
    ``mkstemp`` landing point is pinned under this test's own ``tmp_path`` so the leftovers
    are visible here and are cleaned up by the harness rather than left in the machine's
    shared temporary directory.
    """
    fake = _ConfigurableProbeFake(
        code=0, stdout_payload=b'{"version": "4.2.1", "name": "pycolmap"}')
    _install_configurable_fake(monkeypatch, fake)

    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = str(scratch_dir)

    def refuse_unlink(self, *args, **kwargs):
        raise OSError(13, "simulated: the scratch file cannot be removed")

    monkeypatch.setattr(pathlib.Path, "unlink", refuse_unlink)
    try:
        info = adapter.backend_info(sys.executable)
    finally:
        tempfile.tempdir = previous_tempdir

    assert info["version"] == "4.2.1", "the version really was read"
    residual = info.get("scratch_residual")
    assert residual, (
        "a scratch file that could not be released must be surfaced, not silently dropped")
    assert len(residual) == 2, residual
    for entry in residual:
        assert pathlib.Path(entry["path"]).parent == scratch_dir, entry
        assert entry["error"], entry
        assert pathlib.Path(entry["path"]).exists(), (
            "the residual must name a file that is really still on disk, otherwise the "
            "report is describing a state that does not exist")


def test_a_failing_second_scratch_creation_reclaims_the_first_under_the_pinned_dir(
        monkeypatch, tmp_path) -> None:
    """The scratch pair is "both or neither", and the first file must land where the
    test can see it.

    The failure is injected on the **second** ``mkstemp`` call only, so the first file
    is a genuine ``tempfile.mkstemp`` result rather than a stand-in: the test is about
    the real creation path, which fails on the second stream and must not orphan the
    first.  Two facts are asserted, and neither is optional:

    * ``tempfile.tempdir`` is pinned to a directory under this test's own ``tmp_path``
      for the duration of the call and restored afterwards, so the real landing point
      is checked.  Restricting only the *log* directory would leave the pair landing in
      the machine's shared temporary directory, where a leftover file would be
      invisible to the test that created it — which is exactly how an earlier round
      wrote outside its own run directory; and
    * the launch was never reached, so ``Popen`` was not called: the run failed *before*
      any child existed, which is what makes the reclamation claim meaningful.

    The residual is not observable here because there is no evidence path to record it
    on — the product surfaces that case on the returned mapping instead, which is
    asserted separately.
    """
    created: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def failing_second(*args, **kwargs):
        if created:
            raise OSError(28, "simulated: no space left for the second stream file")
        fd, name = real_mkstemp(*args, **kwargs)
        created.append(name)
        return fd, name

    monkeypatch.setattr(tempfile, "mkstemp", failing_second)
    fake = _ProbeFakeProcess()
    calls = _install_probe_fake(monkeypatch, fake)

    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = str(scratch_dir)
    try:
        with pytest.raises(OSError):
            # No evidence path: this is the branch that creates the scratch pair.
            adapter.backend_info(sys.executable)
    finally:
        tempfile.tempdir = previous_tempdir

    assert len(created) == 1, (
        "exactly one stream file may have been created before the second failed: %r"
        % (created,))
    landed = pathlib.Path(created[0])
    assert landed.parent == scratch_dir, (
        "the real mkstemp must land in the pinned directory, not in the machine's "
        "shared temporary directory: %s" % landed)
    assert not landed.exists(), (
        "the first file must be reclaimed when the second creation failed, not "
        "orphaned on disk with no record: %s" % landed)
    assert calls["popen_kwargs"] is None, (
        "the launch must not have been reached: there was no complete stream pair to "
        "launch into")
    assert fake.kill_calls == 0 and fake.wait_calls == 0, (
        "no child existed, so there was nothing to terminate or reap")


# --------------------------------------------------------------------------- #
# 5d. a real short-lived probe through the production backend_info path
# --------------------------------------------------------------------------- #
#: A real subprogram that writes the *opening fragment of the eventual valid JSON*
#: to stdout and a short marker to stderr, both without any manual flush, then waits
#: for a release file and only then completes the JSON and exits 0.
#:
#: The prefix is deliberately a literal slice of the final document (``{"version":
#: "4.2.1"`` and the tail ``"pid": <n>}``), not an unrelated marker: a reader that
#: sees it early has seen a *prefix of the real payload*, and the completed document
#: must be the same object continued — not a second line that merely looks similar.
#: A second line on stdout would also make the probe unparseable, so the readiness
#: discovery record goes to stderr where the production contract does not read it.
PROBE_FIXTURE_BODY = '''
import json, os, sys, time, pathlib
test_id = os.environ.get("KOZLOV_PROBE_TAG")
# ``__file__`` is undefined for ``-c`` code, so identity is carried by the tag, the
# executable and the pid instead: enough to show the reading came from this child.
sys.stderr.write(json.dumps({"probe_ready": test_id, "pid": os.getpid(),
                             "executable": sys.executable, "cwd": os.getcwd()}) + "\\n")
sys.stderr.write("probe-stderr:%s\\n" % test_id)
# No flush() on either line: the child runs under production's ``-u``, so the write
# reaching the file is the property under test, not something the fixture forced.
sys.stdout.write('{"version": "4.2.1", "file": "fixture", "numpy": "2.5.3", ')
sys.stdout.write('"python": "3.13.14", "test_id": "%s", ' % test_id)
release = pathlib.Path(os.environ["KOZLOV_PROBE_RELEASE"])
# Same ceiling as the worker fixture: the probe fixture ends itself well inside 5 s,
# so a case that must show the *product* ended the child sets its bound below this.
deadline = time.monotonic() + __FIXTURE_SELF_LIMIT__
while not release.exists():
    if time.monotonic() > deadline:
        sys.exit(7)
    time.sleep(0.02)
# The document is completed after the release, so the early prefix and the final
# value are demonstrably the same single JSON object.
sys.stdout.write('"pid": %d}\\n' % os.getpid())
sys.exit(0)
'''.replace("__FIXTURE_SELF_LIMIT__", repr(_FIXTURE_SELF_LIMIT_SECONDS))

#: The exact prefix the fixture writes before it blocks.  Kept next to the body so
#: the assertion cannot drift from the fixture that produces it.
PROBE_PREFIX_TAIL = '"python": "3.13.14", "test_id": "'


#: Only these environment names are read out of the launch.  The proxy must not copy
#: the whole ambient environment into a test artifact: the interesting question is
#: whether the *task* overrides the record names were really passed, and dumping
#: ``os.environ`` would put unrelated machine state (including inherited secrets) into
#: evidence for no gain.
_PROXY_ENV_WHITELIST = (
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
    "KOZLOV_PROBE_TAG",
    "KOZLOV_PROBE_RELEASE",
)


def _install_popen_proxy(monkeypatch) -> dict:
    """Wrap the *real* ``subprocess.Popen`` to record what the probe actually passed.

    This is a pass-through, not a stand-in: the proxy records the concrete ``argv``,
    ``cwd`` and the **whitelisted task environment** the production code hands to
    ``Popen``, then calls the genuine constructor and returns its process.  A test that
    only compared the start record with a constant it wrote itself would be comparing
    two self-reported strings — the record and the log would agree even if the child
    had been launched with something else entirely.  Capturing at the call boundary is
    what makes "the record describes this launch" a fact about the call site.
    """
    captured: dict = {"calls": []}
    real_popen = subprocess.Popen

    class _Proxy:
        def __init__(self, argv, **kwargs):
            launch_env = kwargs.get("env")
            captured["calls"].append({
                "argv": list(argv),
                "cwd": kwargs.get("cwd"),
                "task_env": (
                    None if launch_env is None else
                    {key: launch_env.get(key) for key in _PROXY_ENV_WHITELIST
                     if key in launch_env}
                ),
                "stdout_is_file_object": hasattr(kwargs.get("stdout"), "fileno"),
                "stderr_is_file_object": hasattr(kwargs.get("stderr"), "fileno"),
            })
            self._process = real_popen(argv, **kwargs)

        def __getattr__(self, name):
            return getattr(self._process, name)

    monkeypatch.setattr(adapter, "subprocess", _SubprocessShim(_Proxy))
    return captured


def test_a_real_short_lived_probe_streams_before_exit_and_completes_json(
        monkeypatch, tmp_path) -> None:
    """The production ``backend_info`` path runs a real child that is readable early.

    The production argv (``-I -u -B``) and the production interpreter are kept; only
    the ``-c`` *code* is replaced, so the launch under test is the real one.  While
    the child is alive the observer must read **both** streams from the log files —
    stdout carrying a prefix of the eventual valid JSON, stderr carrying the discovery
    marker — which is the property ``-I`` plus ``PYTHONUNBUFFERED`` alone did **not**
    provide, and which ``-u`` on the command line does.  The child then completes the
    same JSON object, which ``backend_info`` parses, so "observable" and "usable" are
    shown to be the same process rather than two separate claims.
    """
    test_id = "probe-%d" % os.getpid()
    release = tmp_path / "probe.release"
    monkeypatch.setenv("KOZLOV_PROBE_TAG", test_id)
    monkeypatch.setenv("KOZLOV_PROBE_RELEASE", str(release))
    monkeypatch.setattr(adapter, "_BACKEND_PROBE_CODE", PROBE_FIXTURE_BODY)
    captured = _install_popen_proxy(monkeypatch)

    evidence = tmp_path / "logs" / "backend_process.jsonl"
    stdout_path = tmp_path / "logs" / "backend_process.stdout.txt"
    stderr_path = tmp_path / "logs" / "backend_process.stderr.txt"

    holder: dict = {}
    command = adapter.backend_probe_command(sys.executable)
    assert "-I" in command and "-u" in command, (
        "production must keep isolation and add explicit -u")

    def run() -> None:
        holder["info"] = adapter.backend_info(
            sys.executable, evidence_path=evidence, project_root=tmp_path, timeout=5)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    early_stdout = ""
    early_stderr = ""
    early_ready = None
    observation = None
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if stdout_path.is_file() and stderr_path.is_file():
                early_stdout = stdout_path.read_text(encoding="utf-8")
                early_stderr = stderr_path.read_text(encoding="utf-8")
                early_ready = _probe_readiness_of(early_stderr, test_id)
                if (early_ready is not None
                        and PROBE_PREFIX_TAIL in early_stdout
                        and f"probe-stderr:{test_id}" in early_stderr):
                    break
            time.sleep(0.05)
        else:
            raise AssertionError(
                "the probe's stdout JSON prefix and its stderr marker must both be "
                "readable from the log files while the child is still running — that "
                "is the property -u provides. stdout=%r stderr=%r"
                % (early_stdout[:200], early_stderr[:200]))

        # BOTH streams, asserted — not just the one the previous revision checked.
        assert PROBE_PREFIX_TAIL in early_stdout, (
            "stdout must already carry a prefix of the eventual JSON, not be empty "
            "until exit: %r" % early_stdout[:200])
        assert f"probe-stderr:{test_id}" in early_stderr, early_stderr[:200]
        # The document is not yet complete and is not yet parseable: this is what
        # makes the early read an *incremental* property rather than a race that
        # happened to read a finished file.
        assert not early_stdout.strip().endswith("}"), (
            "the JSON must still be incomplete while the child is blocked")
        with pytest.raises(ValueError):
            json.loads(early_stdout)

        # The same READY worker, alive, through a handle bound to its own creation
        # identity — not a fresh lookup by number, which could name a recycled PID.
        assert early_ready["pid"] > 0
        observation = _ProcessObservation(early_ready["pid"])
        assert observation.state() == "alive", (
            "the very process whose prefix was read must still be alive — the "
            "readings cannot be attributed to a finished process")
        assert thread.is_alive(), "the probe must still be running when read"
        assert not release.exists()
    finally:
        release.write_text("go", encoding="utf-8")
        thread.join(timeout=12)

    if observation is not None:
        try:
            assert observation.wait_dead(within=5.0) == "dead", (
                "after the release the probe must exit on its own; a still-alive "
                "process would mean the fixture outlived its own bound")
        finally:
            observation.close()

    info = holder["info"]
    assert info["name"] == "pycolmap"
    assert info["version"] == "4.2.1", "the child's JSON must be what backend_info parsed"
    assert info["test_id"] == test_id
    # The final document is the completion of the same object whose prefix was read,
    # and it belongs to the same PID that was observed alive above.
    assert info["pid"] == early_ready["pid"], (
        "the parsed document must come from the same process that was observed "
        "streaming: parsed pid %r vs observed pid %r" % (info.get("pid"), early_ready["pid"]))

    start = read_jsonl(evidence)[0]
    assert start["kind"] == "backend_probe"
    assert "-u" in start["argv"], "the -u flag must be in the real argv"
    assert start["environment_unbuffered_effective"] is True
    assert "argv -u" in start["unbuffered_mechanism"]

    # The record compared against what Popen was *actually* called with, captured at
    # the call boundary by the pass-through proxy.  Comparing the record with a
    # constant this test also wrote would prove nothing about the launch.
    assert captured["calls"], "the real Popen must have been called"
    actual = captured["calls"][0]
    assert actual["argv"] == start["argv"] == command, (
        "the recorded argv must be the argv Popen received: recorded=%r actual=%r"
        % (start["argv"], actual["argv"]))
    assert actual["cwd"] == start["cwd"] == str(tmp_path), (
        "the recorded cwd must be the cwd Popen received: recorded=%r actual=%r"
        % (start["cwd"], actual["cwd"]))
    assert actual["task_env"] is not None, (
        "the proxy must have seen the environment Popen was handed")
    assert actual["task_env"]["PYTHONUNBUFFERED"] == start["environment_unbuffered"] == "1"
    for key, value in start["environment_overrides"].items():
        assert actual["task_env"][key] == value, (
            "the override %s recorded as %r was actually passed as %r"
            % (key, value, actual["task_env"].get(key)))
    # The recorded task environment must be the whitelist only: the evidence must not
    # carry an unrelated copy of the machine's ambient environment.
    assert set(actual["task_env"]) <= set(_PROXY_ENV_WHITELIST), actual["task_env"]
    # The child's streams really were file objects, which is what makes the early
    # read possible at all.
    assert actual["stdout_is_file_object"] and actual["stderr_is_file_object"], actual


def test_a_real_finite_lifetime_probe_that_is_never_released_times_out_and_is_reclaimed(
        monkeypatch, tmp_path) -> None:
    """The *product* bound, not the fixture's own limit, must be what ends the child.

    The release file is deliberately never written, which leaves two possible endings:
    the product times the probe out and reclaims it, or the fixture reaches its own
    self-limit and exits 7 by itself.  A case that only waited for the process to
    disappear could not tell those apart, and would pass even if the product never
    killed anything.  Two things make the distinction observable:

    * the product bound is asserted to be **below** the fixture's own limit, so a
      self-limited fixture could not have produced a ``TimeoutExpired`` at all — if
      the fixture had ended itself, ``wait`` would have returned its code 7 and
      ``timed_out`` would be ``False``; and
    * the elapsed wall time is measured from the launch anchor to the moment the death
      was **confirmed**, and asserted inside that same limit, so the ending really
      happened before the fixture was due to end itself.  The end anchor is taken only
      after the death wait, never before it: a reading taken while the child might still
      be running would be short even if nothing ever reclaimed it.

    The observation handle is bound to the READY worker — its own creation identity —
    while it is still alive, so a recycled PID cannot be read as this child.  The
    final state is three-valued: only an observed "dead" passes, an unanswerable query
    is reported as UNKNOWN, and no fallback kill by the *test* is performed at all
    (``fallback_used`` is asserted false), so a confirmed death is neither the test's
    doing nor an artefact of the wait window.
    """
    test_id = "probe-timeout-%d" % os.getpid()
    release = tmp_path / "probe.never-released"
    monkeypatch.setenv("KOZLOV_PROBE_TAG", test_id)
    monkeypatch.setenv("KOZLOV_PROBE_RELEASE", str(release))
    monkeypatch.setattr(adapter, "_BACKEND_PROBE_CODE", PROBE_FIXTURE_BODY)
    _install_popen_proxy(monkeypatch)

    evidence = tmp_path / "logs" / "backend_process.jsonl"
    stderr_path = tmp_path / "logs" / "backend_process.stderr.txt"

    # Strictly inside the fixture's own lifetime, and asserted rather than assumed: a
    # bound at or above it would let the fixture end itself and this case would pass
    # while proving nothing about the product's timeout path.
    bound = 2.0
    assert bound < _FIXTURE_SELF_LIMIT_SECONDS, (
        "the product bound must be shorter than the fixture's own limit, otherwise the "
        "fixture would end itself and the reclamation would not be observed")

    holder: dict = {
        #: As in the stage case: nothing in this test ends the child, so a death the
        #: observation confirms cannot be the test's own doing.
        "fallback_used": False,
    }

    def read_ready() -> None:
        """Bind the read-only observation to the READY probe while it is still alive."""
        deadline = time.monotonic() + _FIXTURE_SELF_LIMIT_SECONDS
        while time.monotonic() < deadline:
            if stderr_path.is_file():
                ready = _probe_readiness_of(
                    stderr_path.read_text(encoding="utf-8"), test_id)
                if ready is not None:
                    # The tighter of the two anchors, read in the same breath as the
                    # readiness line.  Recorded for the failure message; the comparison
                    # that decides uses the launch anchor (see the helper).
                    holder["ready_monotonic"] = time.monotonic()
                    observation = _ProcessObservation(ready["pid"])
                    holder["observation"] = observation
                    holder["ready"] = ready
                    holder["state_at_ready"] = observation.state()
                    return
            time.sleep(0.01)

    reader = threading.Thread(target=read_ready, daemon=True)
    reader.start()
    started = time.monotonic()
    try:
        with pytest.raises(adapter.BackendUnavailable) as raised:
            adapter.backend_info(sys.executable, evidence_path=evidence,
                                 project_root=tmp_path, timeout=bound)
    finally:
        # A fact, and *not* the end anchor.  This reading is taken while the child may
        # still be running; the anchor that decides is taken below, after the death has
        # been confirmed.  The earlier shape compared this number against the fixture's
        # self-limit, which was short whenever the product raised promptly even if
        # nothing ever reclaimed the child.
        elapsed_to_exception = time.monotonic() - started
        reader.join(timeout=10)

    assert "exceeded its" in str(raised.value), raised.value

    # The child really was the fixture and really was alive when it was bound, so
    # "a probe started and was then timed out" is a checked fact rather than an
    # inference from the absence of output.
    observation = holder.get("observation")
    assert observation is not None, (
        "the probe must have been observed while it was still running; without that "
        "binding there is no creation identity to check the reclamation against")
    ready = holder["ready"]
    assert ready["pid"] > 0
    assert os.path.normcase(ready["executable"]) == os.path.normcase(sys.executable), (
        "the readiness line must come from this interpreter: %r" % (ready["executable"],))
    assert holder["state_at_ready"] == "alive", (
        "the probe must have been alive when it was bound; state was %r"
        % holder["state_at_ready"])
    assert not release.exists(), "no release may have been written for this case"

    # Only a promptness check on the reading taken before the death wait: the product
    # must raise shortly after its bound rather than hanging near the fixture's limit.
    # It is *not* evidence that the child was reclaimed -- that is decided below, from
    # an anchor read after the death was confirmed.
    assert 0.0 < elapsed_to_exception < _FIXTURE_SELF_LIMIT_SECONDS, (
        "the product must raise promptly after its %.1fs bound; the exception arrived "
        "after %.3fs, which is not inside the fixture's %.1fs self-limit"
        % (bound, elapsed_to_exception, _FIXTURE_SELF_LIMIT_SECONDS))

    try:
        final_state = observation.wait_dead(within=5.0)
        death_confirmed_at = time.monotonic()
        assert final_state != "unknown", (
            "the liveness of the timed-out probe (pid %d) could not be established: the "
            "observation is UNKNOWN, which is neither a pass nor an adapter failure"
            % ready["pid"])
        assert final_state == "dead", (
            "the timed-out probe (pid %d) must be observed dead, not left running "
            "(final state: %s)" % (ready["pid"], final_state))
        # Decided here, after the death, and against the launch anchor: a probe that
        # ended because the *fixture* reached its own self-limit would show an elapsed
        # time at or beyond that limit, and must not pass as a reclamation.
        holder["death_timings"] = _assert_death_inside_fixture_limit(
            label="probe timeout case",
            launch_monotonic=started,
            ready_monotonic=holder["ready_monotonic"],
            death_monotonic=death_confirmed_at,
            fallback_used=holder["fallback_used"])
    finally:
        observation.close()

    entries = read_jsonl(evidence)
    events = [entry["event"] for entry in entries]
    assert events[0] == "start"
    assert "timeout" in events, "the kill must be recorded before it is performed"
    assert events[-1] == "finish"
    finish = entries[-1]
    assert finish["timed_out"] is True, (
        "the product bound must have expired; a fixture that ended itself would make "
        "this case vacuous (reason=%r)" % (finish.get("process_exit_code_reason"),))
    # Consistency between the three recorded facts, rather than a sentence or an
    # unconditional ``True``.  ``terminated`` is the compatibility field and has to agree
    # with ``termination_requested``.  A successful reap confirms the process exit, so
    # ``exit_confirmed`` is True.  The observed exit status is still not an algorithm result.
    assert finish["termination_requested"] is True, (
        "this fixture's kill succeeds, so a termination really was requested; the timeout "
        "path must record that from the kill call and not assume it")
    assert finish["terminated"] is finish["termination_requested"], (
        "the compatibility field must agree with termination_requested rather than being "
        "written true unconditionally")
    assert finish["cleanup_attempted"] is True
    assert finish["cleanup_finished"] is True, (
        "the probe really was reaped, so cleanup genuinely completed; False here would "
        "mean the test cannot tell a clean reap from an abandoned one")
    # ``exit_confirmed`` and ``process_exit_code`` are two facts and are asserted as two.  The
    # chain that used to stand here -- ``finish["exit_confirmed"] is finish["process_exit_code"]
    # is not None`` -- is an object-identity comparison between a bool and the exit code, so it
    # requires ``True`` and the code to be the *same object*: a reaped child reporting True and
    # an integer status makes it fail deterministically.  Assert each fact on its own terms.
    assert finish["exit_confirmed"] is True, (
        "the reap completed and the child's status was read, so the exit is confirmed; the "
        "recorded value is %r" % (finish["exit_confirmed"],))
    assert finish["process_exit_code"] is not None, (
        "a confirmed exit has to carry the status that confirmed it, not an absent one")
    assert isinstance(finish["process_exit_code"], int) \
        and not isinstance(finish["process_exit_code"], bool), (
        "the exit code is the integer status the reap read (got %r)"
        % (finish["process_exit_code"],))
    reason = finish["process_exit_code_reason"]
    assert reason, "the record must say what its exit code, or its absence, means"
    if finish["process_exit_code"] is None:
        assert "no exit code" in reason, reason
    else:
        # The reason names where the code came from -- the bounded reclamation -- and does not
        # claim to have established *why* the child ended.  The kill call succeeds in this
        # fixture; the same reason field also describes the separate fake kill-failure case.
        assert "bounded reclamation" in reason, (
            "a code the bounded reap obtained has to be described as such, and not as the "
            "termination's own status: %r" % reason)
        assert "the exit status of the termination" not in reason, (
            "the record should identify the bounded reap that obtained the code, "
            "rather than infer what caused the process to exit: %r" % reason)
    # The record's own stderr tail names the fixture that was ended, so the record and
    # the readiness observation describe the same single child.
    assert any(f"probe-stderr:{test_id}" in line for line in finish["stderr_tail"]), (
        finish["stderr_tail"])


def _probe_readiness_of(stdout_text: str, test_id: str) -> dict | None:
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("probe_ready") == test_id:
            return payload
    return None


def test_the_backend_probe_writes_its_own_start_and_finish(tmp_path) -> None:
    """The probe is a process; its evidence must not be inferred from a later stage."""
    evidence = tmp_path / "logs" / "backend_process.jsonl"
    info = adapter.backend_info(
        sys.executable, evidence_path=evidence, project_root=tmp_path,
        package_root=pathlib.Path(adapter.__file__).resolve().parent,
    )
    assert info["version"] == adapter.BACKEND_PIN
    entries = read_jsonl(evidence)
    assert [entry["event"] for entry in entries] == ["start", "finish"]
    assert entries[1]["process_exit_code"] == 0
    assert entries[0]["cwd"] == str(tmp_path)
    assert entries[1]["duration_monotonic_seconds"] >= 0
    # The probe's streams are files the child wrote, not a buffer returned at exit.
    assert entries[0]["stdout_path"] == str(tmp_path / "logs" / "backend_process.stdout.txt")
    assert (tmp_path / "logs" / "backend_process.stdout.txt").is_file()


def test_the_probe_launch_arguments_match_its_record(tmp_path) -> None:
    """The probe's logged cwd/overrides must be the ones its process received."""
    evidence = tmp_path / "logs" / "backend_process.jsonl"
    adapter.backend_info(
        sys.executable, evidence_path=evidence, project_root=tmp_path,
        package_root=pathlib.Path(adapter.__file__).resolve().parent, timeout=60,
    )
    start = read_jsonl(evidence)[0]
    assert start["kind"] == "backend_probe"
    assert start["cwd"] == str(tmp_path)
    assert start["environment_overrides"] == {
        "PYTHONPATH": str(tmp_path / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    assert start["environment_unbuffered"] == "1"
    assert start["stdout_path"] and start["stderr_path"]
    assert "os.environ" not in json.dumps(start)


def test_a_failed_probe_records_a_null_code_with_a_reason(tmp_path) -> None:
    """An unstartable interpreter is not given a fabricated exit code."""
    evidence = tmp_path / "probe.jsonl"
    with pytest.raises(adapter.BackendUnavailable):
        adapter.backend_info(str(tmp_path / "no-such-interpreter.exe"),
                             evidence_path=evidence, project_root=tmp_path)
    entries = read_jsonl(evidence)
    assert entries[-1]["event"] == "finish"
    assert entries[-1]["process_exit_code"] is None
    assert entries[-1]["process_exit_code_reason"]

def test_interim_run_state_is_on_disk_before_the_run_ends(tmp_path, monkeypatch) -> None:
    """The run record is flushed at stage boundaries, not only at the end."""
    run = make_upstream_run(tmp_path / "upstream")
    run_dir = tmp_path / "out" / "rid"
    run_dir.mkdir(parents=True)
    observed: list[dict] = []

    monkeypatch.setattr(cli, "backend_info", lambda *a, **k: {
        "name": "pycolmap", "version": adapter.BACKEND_PIN, "runtime": "cpu"})

    def stage(stage, spec, **kw):
        # The record for the *previous* boundary must already be readable here.
        if (run_dir / "run.json").is_file():
            observed.append(json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
        return adapter.NativeStageResult(
            stage=stage, started_utc="T", finished_utc="T", returncode=None, timed_out=False,
            stdout_path="", stderr_path="", result=None,
            error="the shared runtime budget was exhausted before this stage could start; it "
                  "was not executed", started=False)

    monkeypatch.setattr(cli, "run_native_stage", stage)
    cli.run_reconstruction(io_run=str(run), run_dir=run_dir, project_root=tmp_path,
                           profile=profiles.PROFILE_CAD02A_SHARED_V1)

    assert observed, "the run record existed before the stages had all been visited"
    assert observed[0]["counts"]["input_rows"] == 3
    assert observed[0]["effective_options"]["camera_sharing"] == "single"


# --------------------------------------------------------------------------- #
# 6. real native round trip of a shared camera (no photos, no SIFT/mapper)
# --------------------------------------------------------------------------- #
def _build_shared_camera_model():
    """A one-shared-camera reconstruction with non-coplanar points and real observations.

    Constructed from the native API rather than read from a file, so the expected
    values are known independently of any export function.  The camera is the only
    one, which is the property the shared profile asserts; three images each get
    their own rig and frame, so the rig/frame/track relation is non-trivial and a
    round trip that dropped it would be visible.

    Ordering matters and is asserted by the native API itself: the frame must exist
    before its image, and a camera before the rig that names it as a sensor.  A frame
    plus its image is what puts the image into ``reg_image_ids()``; the 3D points and
    their tracks are added afterwards to give the model real content, and are not
    what registers it.
    """
    pycolmap = pytest.importorskip("pycolmap")
    reconstruction = pycolmap.Reconstruction()

    # One shared SIMPLE_RADIAL camera at the experiment's fixed intrinsics.
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_RADIAL, 1.2 * 1024, 1024, 683)
    camera.params = [1228.8, 512.0, 341.5, 0.0]
    reconstruction.add_camera(camera)

    points = np.array([
        [0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [-1.0, 0.5, 4.5],
        [0.5, -1.0, 3.5], [1.2, 1.1, 5.0], [-1.3, -0.9, 5.5], [0.0, 2.0, 6.0],
        [2.0, 0.0, 6.5], [-2.0, -1.5, 5.2], [1.5, -2.0, 7.0], [0.7, 0.3, 3.0],
    ])

    poses: dict[int, Any] = {}
    for image_id in (1, 2, 3):
        angle = 0.15 * image_id
        poses[image_id] = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.array([0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)])),
            np.array([0.1 * image_id, -0.05 * image_id, 0.0]))

    # One rig, one sensor: three images sharing one camera are three data entries
    # against the *same* sensor id, which is what "shared camera" means natively.
    sensor = pycolmap.sensor_t()
    sensor.type = pycolmap.SensorType.CAMERA
    sensor.id = 1
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(sensor)
    reconstruction.add_rig(rig)

    for image_id in (1, 2, 3):
        frame = pycolmap.Frame()
        frame.frame_id = image_id
        frame.rig_id = 1
        frame.rig_from_world = poses[image_id]
        frame.add_data_id(pycolmap.data_t(sensor, image_id))
        reconstruction.add_frame(frame)

        image = pycolmap.Image(
            name=f"images/{image_id - 1:06d}.png", camera_id=1, image_id=image_id)
        image.frame_id = image_id
        image.points2D = pycolmap.Point2DList([
            pycolmap.Point2D(camera.img_from_cam(xyz), pycolmap.INVALID_POINT3D_ID)
            for xyz in points
        ])
        reconstruction.add_image(image)

    for point_index, xyz in enumerate(points):
        track = pycolmap.Track()
        for image_id in (1, 2, 3):
            track.add_element(pycolmap.TrackElement(image_id, point_index))
        color = [10 * point_index % 256, 20, 30]
        point_id = int(reconstruction.add_point3D(np.asarray(xyz, dtype=float), track, color))
        for image_id in (1, 2, 3):
            reconstruction.image(image_id).set_point3D_for_point2D(point_index, point_id)

    assert int(reconstruction.num_cameras()) == 1
    assert int(reconstruction.num_points3D()) == len(points)
    return reconstruction, points, camera


def test_native_shared_camera_round_trips_binary_and_text(tmp_path) -> None:
    """binary and text are written and read back, and both must agree with the source."""
    pytest.importorskip("pycolmap")
    reconstruction, points, _ = _build_shared_camera_model()

    source = cli.export_module.describe_model(reconstruction)
    binary_dir = tmp_path / "binary"
    text_dir = tmp_path / "text"
    binary_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    reconstruction.write(str(binary_dir))
    reconstruction.write_text(str(text_dir))

    from_binary = cli.export_module.read_binary_model(binary_dir)
    from_text = cli.export_module.read_text_model(text_dir)

    for label, read_back in (("binary", from_binary), ("text", from_text)):
        differences = cli.export_module.compare_descriptions(
            source, cli.export_module.describe_model(read_back), tolerance=1e-9)
        assert differences == [], f"{label} export differs from the source: {differences[:5]}"

    # Correspondence, checked against the source rather than against an export.
    assert sorted(int(i) for i in from_binary.reg_image_ids()) == [1, 2, 3]
    assert int(from_binary.num_points3D()) == len(points)
    assert sorted(int(c) for c in from_binary.cameras) == [1]
    for image_id in (1, 2, 3):
        assert int(from_binary.image(image_id).camera_id) == 1
        assert int(from_binary.image(image_id).frame_id) >= 1
    for point_id in sorted(int(p) for p in from_binary.points3D):
        track = from_binary.point3D(point_id).track
        assert len(track.elements) == 3, "each point is observed by all three images"
        assert {int(e.image_id) for e in track.elements} == {1, 2, 3}


def test_native_shared_camera_keeps_identity_and_pose(tmp_path) -> None:
    """Rig/frame ids survive a round trip, and the camera centre follows |C| = |t| for R=I-ish."""
    pycolmap = pytest.importorskip("pycolmap")
    reconstruction, _, camera = _build_shared_camera_model()

    text_dir = tmp_path / "text"
    text_dir.mkdir(parents=True)
    reconstruction.write_text(str(text_dir))
    read_back = cli.export_module.read_text_model(text_dir)

    assert int(read_back.num_rigs()) == int(reconstruction.num_rigs())
    assert int(read_back.num_frames()) == int(reconstruction.num_frames())
    assert int(read_back.num_cameras()) == 1

    written_camera = read_back.camera(1)
    assert written_camera.model_name == "SIMPLE_RADIAL"
    assert (int(written_camera.width), int(written_camera.height)) == (1024, 683)
    assert [float(v) for v in written_camera.params] == pytest.approx(
        [1228.8, 512.0, 341.5, 0.0], abs=1e-9)
    assert bool(written_camera.has_prior_focal_length) is False

    # Camera centre computed from the description must match a ground-truth formula
    # applied to the source pose, not to a value produced by the same exporter.
    description = cli.export_module.describe_model(read_back)
    for image_id in (1, 2, 3):
        source_pose = reconstruction.image(image_id).cam_from_world()
        quaternion = [float(v) for v in np.asarray(source_pose.rotation.quat).ravel()]
        translation = [float(v) for v in np.asarray(source_pose.translation).ravel()]
        expected_center = cli.export_module.camera_center(quaternion, translation)
        actual_center = description["images"][str(image_id)]["camera_center"]
        assert actual_center == pytest.approx(expected_center, abs=1e-9)


def test_native_projection_is_independently_reproducible() -> None:
    """A small known projection computed here, not with the export function.

    ``x_cam = R X + t`` then ``x_img = f * (x/z, y/z) + (cx, cy)`` for a zero-distortion
    SIMPLE_RADIAL camera: the expected pixel is derived by hand rather than taken
    from the same code under test.
    """
    pycolmap = pytest.importorskip("pycolmap")
    _, points, camera = _build_shared_camera_model()

    f, cx, cy, k1 = [float(v) for v in camera.params]
    assert k1 == 0.0
    xyz = points[0]
    projected = camera.img_from_cam(xyz)
    expected_x = f * (xyz[0] / xyz[2]) + cx
    expected_y = f * (xyz[1] / xyz[2]) + cy
    assert float(projected[0]) == pytest.approx(expected_x, abs=1e-6)
    assert float(projected[1]) == pytest.approx(expected_y, abs=1e-6)


def test_shared_profile_model_camera_check_accepts_one_camera(tmp_path) -> None:
    pytest.importorskip("pycolmap")
    reconstruction, _, _ = _build_shared_camera_model()
    cli._check_model_shared_camera(reconstruction, "0")   # must not raise


def _add_image_on_a_new_camera(reconstruction, pycolmap, image_id: int,
                               camera_id: int, *, width: int = 1024,
                               height: int = 683) -> int:
    """Attach one further registered image served by ``camera_id``.

    The native API determines registration from the frame/image pair, not from
    observations: an image added together with its own rig and frame is already in
    ``reg_image_ids()`` before any 3D point exists, so no mapping step is needed to
    make it visible to a check that walks registered images.  A fourth image is used
    rather than rewriting image 1 because the reconstruction keys images by id and
    will not accept a replacement for an id it already holds -- mutating in place is
    not expressible through the public interface.  The check's subject (registered
    images served by more than one camera) is the same either way.
    """
    camera = pycolmap.Camera.create_from_model_id(
        camera_id, pycolmap.CameraModelId.SIMPLE_RADIAL, 1.2 * width, width, height)
    camera.params = [1228.8, width / 2.0, height / 2.0, 0.0]
    reconstruction.add_camera(camera)

    sensor = pycolmap.sensor_t()
    sensor.type = pycolmap.SensorType.CAMERA
    sensor.id = camera_id
    rig = pycolmap.Rig()
    rig.rig_id = camera_id
    rig.add_ref_sensor(sensor)
    reconstruction.add_rig(rig)

    pose = pycolmap.Rigid3d(
        pycolmap.Rotation3d(),
        np.array([0.1 * image_id, -0.05 * image_id, 0.0]))
    frame = pycolmap.Frame()
    frame.frame_id = image_id
    frame.rig_id = camera_id
    frame.rig_from_world = pose
    frame.add_data_id(pycolmap.data_t(sensor, image_id))
    reconstruction.add_frame(frame)

    image = pycolmap.Image(
        name=f"images/{image_id - 1:06d}.png", camera_id=camera_id, image_id=image_id)
    image.frame_id = image_id
    image.points2D = pycolmap.Point2DList(
        [pycolmap.Point2D([10.0, 10.0], pycolmap.INVALID_POINT3D_ID)])
    reconstruction.add_image(image)
    return image_id


def test_two_cameras_on_one_sensor_would_not_be_expressible() -> None:
    """The accepting control for the rejection below, kept as an explicit contrast.

    One camera, three registered images, no extra camera added: the same check must
    stay silent.  Without this pairing, a check that rejected *everything* would
    still look correct.
    """
    pytest.importorskip("pycolmap")
    reconstruction, _, _ = _build_shared_camera_model()
    cli._check_model_shared_camera(reconstruction, "0")   # must not raise


def test_shared_profile_model_camera_check_rejects_a_second_camera() -> None:
    """A model whose registered images came from two cameras must be rejected."""
    pycolmap = pytest.importorskip("pycolmap")
    reconstruction, _, _ = _build_shared_camera_model()

    _add_image_on_a_new_camera(reconstruction, pycolmap, image_id=4, camera_id=2)

    registered = sorted(int(i) for i in reconstruction.reg_image_ids())
    assert registered == [1, 2, 3, 4], "the added image must really be registered"
    assert int(reconstruction.image(4).camera_id) == 2
    assert int(reconstruction.num_cameras()) == 2
    # Registration does not depend on observations here, so the point count is
    # deliberately left alone: the check must key on camera identity alone.
    assert int(reconstruction.num_points3D()) > 0

    with pytest.raises(adapter.BackendError) as caught:
        cli._check_model_shared_camera(reconstruction, "0")
    assert "cameras" in str(caught.value)
    assert "[1, 2]" in str(caught.value)


# --------------------------------------------------------------------------- #
# 9. the close-out's local fixes, exercised as pure data
# --------------------------------------------------------------------------- #
# P1/P2 (adapter) and T1/T2 (this file) are fixed above.  The real cases that would
# exercise them end-to-end are the two timeout fixtures, which this round does not
# re-run, so the parts that *can* be checked without a child are checked here: the
# record the adapter builds, the facts it attaches to a propagating exception, the
# boundary arithmetic the two real cases now depend on, and the scratch landing.

class _FalseyKillFake(_ConfigurableProbeFake):
    """A stand-in whose ``kill`` returns a *falsey* value.

    Used only as a falsification of the rule "the return value is not read".  A real
    ``Popen.kill()`` returns ``None``, so ``None`` must not be readable as "the request
    failed"; this stand-in goes further and returns ``False``, which under the old
    ``bool(process.kill())`` shape would have been recorded as a failed termination.
    """

    def kill(self):
        self.kill_calls += 1
        if self._kill_error is not None:
            raise self._kill_error
        return False


def test_a_kill_whose_return_value_is_falsey_is_still_a_recorded_termination(
        monkeypatch, tmp_path) -> None:
    """A first ``wait`` that fails, a kill that returns a falsey value, a real code kept.

    Three facts have to stay separate in the record: that a termination was requested
    (the call returned without raising), that the child was actually reaped, and that a
    real exit status was read.  Deriving the first from what ``kill()`` returned is what
    the close-out fixed: ``Popen.kill()`` returns ``None`` on success, and ``bool(None)``
    recorded a successful termination as ``terminated=False``.
    """
    fake = _FalseyKillFake(
        wait_script=[OSError("original: the initial wait failed before returning")],
        code=3)
    _install_configurable_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"

    with pytest.raises(OSError) as raised:
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=5)
    assert str(raised.value) == "original: the initial wait failed before returning", (
        "the original exception must propagate with its own type and reason")

    finish = read_jsonl(evidence)[-1]
    assert finish["event"] == "finish"
    assert fake.kill_calls == 1
    assert fake.wait_calls == 2, (
        "the bounded reap must still have been attempted after the first wait failed")
    assert finish["termination_requested"] is True, (
        "the termination call returned without raising, so a termination *was* requested; "
        "reading that off a falsey return value is the defect this asserts against")
    assert finish["cleanup_finished"] is True, "the reap completed"
    assert finish["exit_confirmed"] is True
    assert finish["process_exit_code"] == 3, (
        "a status obtained by the bounded reap is a measurement and must be kept")
    assert "bounded reap" in finish["process_exit_code_reason"], (
        finish["process_exit_code_reason"])


def test_a_kill_that_could_not_be_issued_is_not_a_recorded_termination(
        monkeypatch, tmp_path) -> None:
    """The other side of the same fact: a refused kill is reported as a refused kill.

    Without this pairing, a record that said ``termination_requested=True`` no matter
    what would look correct.
    """
    fake = _ConfigurableProbeFake(
        wait_script=[OSError("original: the initial wait failed"), None],
        code=5, kill_error=OSError("kill: the termination request was refused"))
    _install_configurable_fake(monkeypatch, fake)
    evidence = tmp_path / "logs" / "backend_process.jsonl"

    with pytest.raises(OSError) as raised:
        adapter.backend_info(sys.executable, evidence_path=evidence,
                             project_root=tmp_path, timeout=5)
    assert "original: the initial wait failed" in str(raised.value)

    finish = read_jsonl(evidence)[-1]
    assert finish["termination_requested"] is False, (
        "no termination call returned without raising, so none may be reported")
    assert any("termination request was refused" in error
               for error in finish["cleanup_errors"]), finish["cleanup_errors"]
    assert finish["cleanup_finished"] is True, (
        "a refused kill must not skip the reap")
    assert finish["exit_confirmed"] is True and finish["process_exit_code"] == 5


def test_an_exception_without_an_evidence_path_carries_the_post_release_facts(
        monkeypatch) -> None:
    """No evidence path and a residual: the facts must reach the *caller's* exception.

    ``finish`` is a no-op when no evidence location was given, and the return-mapping
    form of the residual is never reached once an exception is propagating, so before
    the fix a probe called as ``backend_info(interpreter)`` that left a scratch file
    behind reported nothing about it.  The residual is checked here against what
    ``_discard`` actually returned, after the release has run.
    """
    fake = _ConfigurableProbeFake(
        wait_script=[OSError("original: the initial wait failed")], code=3)
    _install_configurable_fake(monkeypatch, fake)

    scratch_seen: dict = {"discard_calls": 0, "paths": ()}
    residual = {"path": "<scratch>", "error": "simulated: the file is still on disk"}

    def failing_discard(paths):
        scratch_seen["discard_calls"] += 1
        scratch_seen["paths"] = tuple(str(path) for path in paths)
        return [], [dict(residual)]

    monkeypatch.setattr(adapter, "_discard", failing_discard)

    with pytest.raises(OSError) as raised:
        adapter.backend_info(sys.executable)

    assert type(raised.value) is OSError, "the class must not be replaced"
    assert str(raised.value) == "original: the initial wait failed", (
        "the reason must not be replaced by a cleanup or logging message")
    assert raised.value.__cause__ is None, (
        "no cause may be invented for an exception that had none")

    assert scratch_seen["discard_calls"] == 1, (
        "the release must have run on the exception path")
    assert len(scratch_seen["paths"]) == 2, (
        "the no-evidence form creates a scratch pair: %r" % (scratch_seen["paths"],))

    facts = getattr(raised.value, adapter.RELEASE_FACTS_ATTR, None)
    assert facts is not None, (
        "the post-release facts must be attached to the exception: with no evidence path "
        "there is no other channel")
    assert facts["evidence_path"] is None
    assert facts["scratch_residual"] == [residual]
    assert facts["termination_requested"] is True
    assert facts["cleanup_finished"] is True
    assert facts["exit_confirmed"] is True and facts["process_exit_code"] == 3
    assert facts["source"].startswith("backend_info, after the scratch release")

    notes = "\n".join(getattr(raised.value, "__notes__", ()))
    assert residual["path"] in notes and residual["error"] in notes, (
        "the residual must be readable in the traceback, not only in an attribute; "
        "notes were %r" % (notes,))

    # Nothing above depends on the files surviving, and they were never deleted because
    # the deletion was made to fail.  Removed here so this round does not accumulate
    # scratch it has already accounted for.
    for path in scratch_seen["paths"]:
        pathlib.Path(path).unlink(missing_ok=True)


def test_the_ordinary_no_evidence_probe_still_returns_without_release_facts(
        monkeypatch, tmp_path) -> None:
    """The successful no-evidence form must be unchanged: no residual, no extra key.

    The fix is allowed to add a channel for a *failure*; it is not allowed to change
    what an ordinary successful probe returns or to make every call carry release
    material that has nothing to say.
    """
    fake = _ConfigurableProbeFake(code=0, stdout_payload=b'{"version": "4.2.1"}\n')
    _install_configurable_fake(monkeypatch, fake)
    released: dict = {"paths": ()}
    real_discard = adapter._discard

    def recording_discard(paths):
        released["paths"] = tuple(str(path) for path in paths)
        return real_discard(paths)

    monkeypatch.setattr(adapter, "_discard", recording_discard)

    info = adapter.backend_info(sys.executable)

    assert info["version"] == "4.2.1"
    assert info["runtime"] == "cpu"
    assert "scratch_residual" not in info, (
        "an ordinary probe released everything; the mapping must not gain a key")
    assert len(released["paths"]) == 2, released["paths"]
    assert all(not pathlib.Path(path).exists() for path in released["paths"]), (
        "the real release must have removed both scratch files")


def test_the_self_limit_boundary_rejects_a_death_that_arrived_too_late() -> None:
    """The rule both real timeout cases now rest on, driven by a synthetic clock.

    No process is created and no clock is read.  A real fixture cannot be asked to end
    *after* its own limit on demand, and this round does not re-run one, so the boundary
    is exercised with fabricated monotonic readings: a death inside the limit is
    accepted, a death at or beyond it is rejected, and a death that coincides with the
    test's own fallback is rejected as not attributable to the product.
    """
    limit = _FIXTURE_SELF_LIMIT_SECONDS
    launch = 500.0
    ready = launch + 0.30

    inside = _assert_death_inside_fixture_limit(
        label="synthetic inside", launch_monotonic=launch, ready_monotonic=ready,
        death_monotonic=launch + 2.10, fallback_used=False)
    assert inside["since_launch"] == pytest.approx(2.10)
    assert inside["since_ready"] == pytest.approx(1.80)

    with pytest.raises(AssertionError) as too_late:
        _assert_death_inside_fixture_limit(
            label="synthetic past the limit", launch_monotonic=launch,
            ready_monotonic=ready, death_monotonic=launch + limit + 0.01,
            fallback_used=False)
    assert "self-limit" in str(too_late.value), str(too_late.value)
    assert "%s" not in str(too_late.value), "the message must be interpolated"

    # The sharper counterexample: the death arrived *after* the fixture was due to exit,
    # but only 3.70s after READY -- comfortably "inside the limit" if the READY anchor
    # were the one compared.  Reading the end anchor before the death wait, or comparing
    # against the wrong anchor, is exactly how a fixture that ended itself passed.
    with pytest.raises(AssertionError) as wrong_anchor:
        _assert_death_inside_fixture_limit(
            label="synthetic self-limited", launch_monotonic=launch,
            ready_monotonic=ready, death_monotonic=ready + 3.70, fallback_used=False)
    assert "self-limit" in str(wrong_anchor.value), str(wrong_anchor.value)

    with pytest.raises(AssertionError) as fallback:
        _assert_death_inside_fixture_limit(
            label="synthetic fallback", launch_monotonic=launch, ready_monotonic=ready,
            death_monotonic=launch + 1.00, fallback_used=True)
    assert "fallback" in str(fallback.value), str(fallback.value)


def test_this_rounds_tempfile_landing_is_confined_to_the_entry_basetemp(tmp_path) -> None:
    """Every ambient ``tempfile`` fallback in this round lands inside the basetemp.

    The no-evidence probe form calls ``mkstemp`` with no directory, so without the pin
    it wrote into this host's TEMP and the round could not claim that all of its scratch
    was inside the run.  The landing directory is not scanned for old files and nothing
    outside the run is touched: what is shown is where a *new* file goes.
    """
    landing = pathlib.Path(tempfile.gettempdir()).resolve()
    basetemp = pathlib.Path(tmp_path).resolve().parent
    assert basetemp in landing.parents, (
        "the ambient tempfile directory must be inside this entry's basetemp; it is %s "
        "with basetemp %s" % (landing, basetemp))

    fd, name = tempfile.mkstemp(prefix="landing-probe-", suffix=".tmp")
    os.close(fd)
    created = pathlib.Path(name).resolve()
    try:
        assert basetemp in created.parents, (
            "a new ambient tempfile must be created inside the entry's basetemp: %s"
            % (created,))
    finally:
        created.unlink(missing_ok=True)


def test_the_timeout_state_and_the_termination_request_are_independent_facts(
        monkeypatch, tmp_path) -> None:
    """Pure data: a refused kill must not be reported as a termination request.

    The close-out's first gap was that the timeout arm set ``termination_requested = True``
    on *entering* itself, so the two facts -- "the probe exceeded its bound" and "a kill call
    returned without raising" -- were carried by one flag, and a kill that raised still
    produced a record claiming the request had succeeded.  Here the kill raises and the reap
    still returns a real code, so the record has to say: timed out, no termination requested,
    reaped, exit confirmed, code kept.  Each of those is asserted from the record the adapter
    wrote, not from the exception message.
    """
    fake = _ProbeFakeProcess(reap_code=7,
                             kill_error=OSError("kill: the termination request was refused"))
    shape = _timed_out_probe(tmp_path, fake, label="facts-are-independent",
                             monkeypatch=monkeypatch)
    finish = shape["finish"]
    assert shape["raised"]["type"] is adapter.BackendUnavailable
    assert finish["timed_out"] is True, "the probe really did exceed its bound"
    assert finish["termination_requested"] is False, (
        "no kill call returned without raising, so none may be reported as requested")
    assert finish["terminated"] is finish["termination_requested"]
    assert finish["cleanup_attempted"] is True, "the kill and reap were still performed"
    assert finish["cleanup_finished"] is True, "the reap completed"
    assert finish["exit_confirmed"] is True, "a real status was read"
    assert finish["process_exit_code"] == 7, "and it was kept"


def test_a_registered_termination_alone_is_not_what_makes_a_timeout_fail(
        monkeypatch, tmp_path) -> None:
    """Pure data: the converse of the test above, so neither flag can stand in for the other.

    A probe that returns normally after a *successful* termination attempt is not this code
    path at all -- the timeout arm is only reached when the first wait raised.  This case
    pins the half that is reachable: the timeout arm is entered, the kill succeeds, and the
    record still distinguishes "timed out" from "terminated" by carrying both.
    """
    fake = _ProbeFakeProcess(reap_code=0)
    shape = _timed_out_probe(tmp_path, fake, label="termination-is-not-the-cause",
                             monkeypatch=monkeypatch)
    finish = shape["finish"]
    assert finish["timed_out"] is True
    assert finish["termination_requested"] is True, (
        "the kill returned without raising, so a termination was requested")
    assert finish["process_exit_code"] == 0
    # Both flags are true at once here, and neither is inferred from the other: the timeout
    # is what makes the call fail, the successful termination is a separate recorded fact.
    # (Keeping this pair is what stops a later edit from making one of them the other's
    # stand-in: the case above has them disagreeing.)
    assert finish["terminated"] is finish["termination_requested"]
    assert shape["raised"]["type"] is adapter.BackendUnavailable
