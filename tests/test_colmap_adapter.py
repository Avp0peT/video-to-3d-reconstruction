"""``sfm-native-v1`` input-side and control-flow regression tests.

What is real here and what is mocked
------------------------------------
The upstream fixtures are written by *this* file, not produced by the T01
implementation, so a shared misunderstanding between producer and consumer cannot
make both agree.  The native backend is genuinely invoked for the version check;
the failure and timeout paths use monkeypatched ``subprocess.run`` because a real
abort or a real 10-minute timeout is not something a test should wait for — those
tests exercise control flow only and are marked as such, and they never count as
evidence that the backend ran.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from kozlov_lab.reconstruction import colmap_adapter as adapter

# --------------------------------------------------------------------------- #
# upstream fixture, built here rather than by the T01 implementation
# --------------------------------------------------------------------------- #
def write_png(path: pathlib.Path, value: int, size=(16, 12)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path, format="PNG")
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_upstream_run(
    root: pathlib.Path,
    *,
    entries: list[dict] | None = None,
    status: str = "SUCCESS",
    contract: str = "photo-io-v1",
    declared_count: int | None = None,
    row_statuses: list[str] | None = None,
) -> pathlib.Path:
    """Create a minimal but genuine ``photo-io-v1`` run directory."""
    root.mkdir(parents=True, exist_ok=True)
    if entries is None:
        entries = [
            {"value": 10, "mask": None},
            {"value": 60, "mask": 1},
            {"value": 200, "mask": None},
        ]
    rows = []
    for index, entry in enumerate(entries):
        image_name = f"images/{index:06d}.png"
        image_hash = write_png(root / image_name, entry["value"])
        mask_name = None
        if entry["mask"] is not None:
            mask_name = f"masks/{index:06d}.png"
            (root / "masks").mkdir(exist_ok=True)
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[0, 0] = entry["mask"]
            Image.fromarray(mask).save(root / mask_name, format="PNG")
        rows.append({
            "input_index": index,
            "input_image": image_name,
            "input_mask": mask_name,
            "frame_id": f"f_{index:064d}",
            "image_path": image_name,
            "mask_path": mask_name,
            "image_sha256": image_hash,
            "mask_sha256": None,
            "duplicate_of_input_index": None,
            "status": (row_statuses or ["OK"] * len(entries))[index],
            "raw_width": 16,
            "raw_height": 12,
            "canonical_width": 16,
            "canonical_height": 12,
            "exif_orientation_original": None,
            "orientation_applied": 1,
            "raw_to_canonical": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "image_dtype": "uint8",
            "image_mode": "RGB",
            "mask_encoding": "zero_255" if mask_name else None,
            "mask_foreground_pixels": 1 if mask_name else None,
            "image_output": image_name,
            "mask_output": mask_name,
            "cache_fingerprint": "0" * 64,
            "cache_hit": False,
            "camera": None,
            "distortion": None,
            "pose": None,
            "geometry_scale": None,
            "warnings": [],
            "errors": [],
        })

    (root / "frames.jsonl").write_bytes(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        .encode("utf-8")
    )
    run_json = {
        "contract_version": contract,
        "run_id": "upstream_run",
        "dataset_id": "test",
        "data_kind": "synthetic",
        "manifest_sha256": "0" * 64,
        "status": status,
        "input_count": len(rows) if declared_count is None else declared_count,
        "ok_count": len(rows),
        "error_count": 0,
        "warning_count": 0,
        "runtime": {"python": "3.13.14", "pillow": "12.3.0", "numpy": "2.5.3"},
        "implementation_sha256": "0" * 64,
        "errors": [],
    }
    (root / "run.json").write_bytes(
        (json.dumps(run_json, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return root


# --------------------------------------------------------------------------- #
# upstream validation
# --------------------------------------------------------------------------- #
def test_valid_upstream_is_read(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run")
    upstream = adapter.load_upstream_run(run)

    assert upstream.status == "SUCCESS"
    assert upstream.data_kind == "synthetic"
    assert [row.input_index for row in upstream.rows] == [0, 1, 2]
    assert upstream.run_json_sha256 and upstream.frames_jsonl_sha256
    # The upstream image hash is the *original* file's; the canonical hash is new.
    assert upstream.rows[0].canonical_image_sha256 is None


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("status_partial", {"status": "PARTIAL_FAILURE"}),
        ("status_input_error", {"status": "INPUT_ERROR"}),
        ("wrong_contract", {"contract": "photo-io-v2"}),
        ("count_mismatch", {"declared_count": 7}),
        ("error_row", {"row_statuses": ["OK", "ERROR", "OK"]}),
    ],
)
def test_non_conforming_upstream_is_rejected_whole(tmp_path, label, kwargs) -> None:
    run = make_upstream_run(tmp_path / "run", **kwargs)
    with pytest.raises(adapter.InputError) as caught:
        adapter.load_upstream_run(run)
    assert caught.value.code == "INPUT_ERROR", label


def test_missing_files_are_rejected(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run")
    (run / "frames.jsonl").unlink()
    with pytest.raises(adapter.InputError):
        adapter.load_upstream_run(run)


def test_non_finite_json_is_rejected(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run")
    (run / "run.json").write_text('{"contract_version":"photo-io-v1","status":NaN}', encoding="utf-8")
    with pytest.raises(adapter.InputError):
        adapter.load_upstream_run(run)


def test_path_escaping_the_run_directory_is_rejected(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run")
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["image_output"] = "../../outside.png"
    (run / "frames.jsonl").write_bytes(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    )
    upstream = adapter.load_upstream_run(run)
    with pytest.raises(adapter.InputError) as caught:
        adapter.stage_inputs(upstream, tmp_path / "staged")
    assert "escapes" in caught.value.message


def test_missing_canonical_image_is_rejected(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run")
    (run / "images/000001.png").unlink()
    upstream = adapter.load_upstream_run(run)
    with pytest.raises(adapter.InputError):
        adapter.stage_inputs(upstream, tmp_path / "staged")


def test_dimension_mismatch_is_rejected(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run")
    write_png(run / "images/000000.png", 10, size=(32, 8))  # record still says 16x12
    upstream = adapter.load_upstream_run(run)
    with pytest.raises(adapter.InputError) as caught:
        adapter.stage_inputs(upstream, tmp_path / "staged")
    assert "the record says" in caught.value.message


def test_oversized_image_is_rejected_not_resized(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run", entries=[{"value": 1, "mask": None},
                                                       {"value": 2, "mask": None}])
    wide = adapter.MAX_LONG_SIDE + 8
    write_png(run / "images/000000.png", 1, size=(wide, 8))
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["canonical_width"], rows[0]["canonical_height"] = wide, 8
    (run / "frames.jsonl").write_bytes(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    )
    upstream = adapter.load_upstream_run(run)
    with pytest.raises(adapter.InputError) as caught:
        adapter.stage_inputs(upstream, tmp_path / "staged")
    assert "rejected rather than silently resized" in caught.value.message


# --------------------------------------------------------------------------- #
# staging: deduplication and identity
# --------------------------------------------------------------------------- #
def test_identical_bytes_collapse_to_one_representative(tmp_path: pathlib.Path) -> None:
    """Two lines with the same bytes are one observation, not two viewpoints."""
    run = make_upstream_run(tmp_path / "run", entries=[
        {"value": 30, "mask": None},
        {"value": 30, "mask": 1},      # same image bytes, *different* mask
        {"value": 90, "mask": None},
    ])
    upstream = adapter.load_upstream_run(run)
    staged = adapter.stage_inputs(upstream, tmp_path / "staged")

    assert len(staged.rows) == 3, "every upstream line keeps a mapping record"
    assert len(staged.distinct) == 2, "byte-identical images are one input to SfM"

    by_index = {row.input_index: row for row in staged.rows}
    assert by_index[0].representative_input_index == 0
    assert by_index[1].representative_input_index == 0, "the alias follows the smaller index"
    assert by_index[1].staged_image_name == by_index[0].staged_image_name
    # the alias keeps its own mask identity even though it shares the image
    assert by_index[1].canonical_mask_sha256 is not None
    assert by_index[0].canonical_mask_sha256 is None

    staged_files = sorted(p.name for p in staged.images_dir.iterdir())
    assert staged_files == ["000000.png", "000002.png"]


def test_distinct_bytes_are_never_merged_by_name_or_frame_id(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run", entries=[
        {"value": 30, "mask": None},
        {"value": 31, "mask": None},
    ])
    upstream = adapter.load_upstream_run(run)
    staged = adapter.stage_inputs(upstream, tmp_path / "staged")
    assert len(staged.distinct) == 2
    assert staged.distinct[0].canonical_image_sha256 != staged.distinct[1].canonical_image_sha256


def test_fewer_than_two_distinct_images_is_an_input_error(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run", entries=[
        {"value": 30, "mask": None},
        {"value": 30, "mask": None},
    ])
    upstream = adapter.load_upstream_run(run)
    with pytest.raises(adapter.InputError) as caught:
        adapter.stage_inputs(upstream, tmp_path / "staged")
    assert "at least 2" in caught.value.message


def test_more_than_twelve_distinct_images_is_an_input_error(tmp_path: pathlib.Path) -> None:
    entries = [{"value": value, "mask": None} for value in range(1, adapter.MAX_DISTINCT_IMAGES + 2)]
    run = make_upstream_run(tmp_path / "run", entries=entries)
    upstream = adapter.load_upstream_run(run)
    with pytest.raises(adapter.InputError) as caught:
        adapter.stage_inputs(upstream, tmp_path / "staged")
    assert "exceeds the fixed limit" in caught.value.message


def test_staged_copies_match_the_upstream_bytes(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run")
    upstream = adapter.load_upstream_run(run)
    staged = adapter.stage_inputs(upstream, tmp_path / "staged")
    for row in staged.distinct:
        staged_bytes = (staged.images_dir / f"{row.input_index:06d}.png").read_bytes()
        assert adapter.sha256_bytes(staged_bytes) == row.canonical_image_sha256
        assert staged_bytes == (run / row.image_output).read_bytes()


# --------------------------------------------------------------------------- #
# digests, options and fingerprints
# --------------------------------------------------------------------------- #
def test_canonical_json_definition() -> None:
    assert adapter.canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'
    assert adapter.canonical_json({"a": "é"}) == '{"a":"é"}', "ensure_ascii must stay false"


def test_input_fingerprint_tracks_every_documented_input(tmp_path: pathlib.Path) -> None:
    run = make_upstream_run(tmp_path / "run")
    upstream = adapter.load_upstream_run(run)
    staged = adapter.stage_inputs(upstream, tmp_path / "staged")
    options = adapter.effective_options()
    backend = {"name": "pycolmap", "version": "4.2.1", "runtime": "cpu"}
    reference = adapter.input_fingerprint(upstream, staged.rows, options, backend, "impl")

    changed = adapter.input_fingerprint(
        upstream, staged.rows, {**options, "num_threads": 99}, backend, "impl"
    )
    assert changed != reference, "options must participate"

    other_backend = adapter.input_fingerprint(
        upstream, staged.rows, options, {**backend, "version": "4.2.2"}, "impl"
    )
    assert other_backend != reference, "backend version must participate"

    mutated = list(staged.rows)
    mutated[0] = adapter.UpstreamRow(
        **{**vars(staged.rows[0]), "canonical_image_sha256": "f" * 64}
    )
    assert adapter.input_fingerprint(upstream, mutated, options, backend, "impl") != reference


def test_effective_options_record_that_intrinsics_are_estimates() -> None:
    options = adapter.effective_options()
    assert options["calibration_status"] == "estimated"
    assert options["intrinsics_initialization"] == "heuristic"
    assert options["camera_model"] == "SIMPLE_RADIAL"
    assert options["camera_sharing"] == "per_image_independent"
    assert options["device"] == "cpu"
    assert options["num_threads"] == 2
    assert options["sift_max_num_features"] == 4096
    assert options["random_seed"] == 0
    # principal point is initialised but not freely optimised
    assert options["ba_refine_principal_point"] is False


def test_implementation_sha256_covers_only_the_reconstruction_package() -> None:
    project_root = pathlib.Path(adapter.__file__).resolve().parents[3]
    digest = adapter.implementation_sha256(project_root)
    assert len(digest) == 64
    # it must be the digest of the reconstruction package's own files
    pairs = [
        [path.relative_to(project_root).as_posix(), adapter.sha256_file(path)]
        for path in sorted((project_root / "src/kozlov_lab/reconstruction").glob("*.py"))
    ]
    assert digest == adapter.canonical_digest(pairs)
    assert all("reconstruction/" in rel for rel, _ in pairs)


def test_strict_dumps_refuse_non_finite_values() -> None:
    with pytest.raises(ValueError):
        adapter.dumps_strict({"a": float("nan")})
    with pytest.raises(ValueError):
        adapter.dumps_strict({"a": float("inf")})


# --------------------------------------------------------------------------- #
# backend identity — this part really executes pycolmap
# --------------------------------------------------------------------------- #
def test_backend_info_reads_the_installed_version() -> None:
    info = adapter.backend_info(sys.executable)
    assert info["name"] == "pycolmap"
    assert info["runtime"] == "cpu"
    assert info["version"], "the installed version must be read, not assumed"
    assert info["numpy"]


def test_require_pinned_version_is_a_real_gate() -> None:
    adapter.require_pinned_version({"version": adapter.BACKEND_PIN})
    with pytest.raises(adapter.BackendUnavailable) as caught:
        adapter.require_pinned_version({"version": "0.0.0-not-approved"})
    assert adapter.BACKEND_PIN in str(caught.value)


# --------------------------------------------------------------------------- #
# control flow around the native subprocess (mocked, not backend evidence)
# --------------------------------------------------------------------------- #
def _stage_spec(tmp_path: pathlib.Path) -> dict:
    return {"database_path": str(tmp_path / "database.db"), "image_path": str(tmp_path)}


def test_stage_timeout_is_recorded_as_a_timeout(tmp_path, monkeypatch) -> None:
    """Control-flow only: no native call happens here."""

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    record = adapter.run_native_stage(
        "extract_features", _stage_spec(tmp_path),
        run_dir=tmp_path, logs_dir=tmp_path / "logs",
        project_root=tmp_path, timeout=7,
    )
    assert record.timed_out is True
    assert record.returncode is None, "a timeout has no exit code and must not invent one"
    assert "7s" in (record.error or "")
    assert record.is_process is True


def test_stage_failure_exit_code_is_preserved(tmp_path, monkeypatch) -> None:
    """Control-flow only: the native backend is not invoked."""

    class Completed:
        returncode = 41
        stdout = "partial stdout"
        stderr = "native abort line"

    monkeypatch.setattr(adapter.subprocess, "run", lambda *args, **kwargs: Completed())
    record = adapter.run_native_stage(
        "incremental_mapping", _stage_spec(tmp_path),
        run_dir=tmp_path, logs_dir=tmp_path / "logs", project_root=tmp_path,
    )
    assert record.returncode == 41
    assert record.timed_out is False
    assert "41" in (record.error or "")
    assert (tmp_path / "logs/incremental_mapping.stderr.txt").read_text(encoding="utf-8") == "native abort line"


def test_stage_exit_zero_without_a_result_is_a_failure(tmp_path, monkeypatch) -> None:
    """Exit 0 alone is not completion.

    A native stage that returns 0 but writes no result document used to be marked
    OK; the run then ended as a normal ``NO_MODEL``, which is a fabricated "the
    backend simply found nothing".  The missing result must be reported instead.
    """

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(adapter.subprocess, "run", lambda *args, **kwargs: Completed())
    record = adapter.run_native_stage(
        "export_models", _stage_spec(tmp_path),
        run_dir=tmp_path, logs_dir=tmp_path / "logs", project_root=tmp_path,
    )
    assert record.result is None
    assert record.returncode == 0
    assert record.error is not None, "an exit-0 stage with no result must not count as complete"
    assert "no result document" in record.error


def test_stage_result_missing_a_required_field_is_a_failure(tmp_path, monkeypatch) -> None:
    """A result object that exists but omits required fields is not completion."""

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        result_path = pathlib.Path(argv[argv.index("--result") + 1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text('{"stage": "match_exhaustive", "ok": true}', encoding="utf-8")
        return Completed()

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    record = adapter.run_native_stage(
        "match_exhaustive", _stage_spec(tmp_path),
        run_dir=tmp_path, logs_dir=tmp_path / "logs", project_root=tmp_path,
    )
    assert record.error is not None
    assert "required field" in record.error


def test_stage_result_reporting_failure_is_a_failure(tmp_path, monkeypatch) -> None:
    """``ok: false`` in the result must win over a zero exit code."""

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        result_path = pathlib.Path(argv[argv.index("--result") + 1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            '{"stage": "extract_features", "ok": false, "error": "no feature extractor"}',
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    record = adapter.run_native_stage(
        "extract_features", _stage_spec(tmp_path),
        run_dir=tmp_path, logs_dir=tmp_path / "logs", project_root=tmp_path,
    )
    assert record.error is not None
    assert "no feature extractor" in record.error


def test_exhausted_budget_prevents_a_stage_from_starting(tmp_path, monkeypatch) -> None:
    """The shared budget must stop the chain instead of granting a fresh timeout."""
    calls = {"count": 0}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        return Completed()

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    record = adapter.run_native_stage(
        "incremental_mapping", _stage_spec(tmp_path),
        run_dir=tmp_path, logs_dir=tmp_path / "logs", project_root=tmp_path,
        deadline=adapter.time.monotonic() - 1.0,
    )
    assert record.started is False
    assert calls["count"] == 0, "no subprocess may be launched once the budget is gone"
    assert record.error is not None and "budget" in record.error


def test_remaining_budget_shrinks_the_stage_timeout(tmp_path, monkeypatch) -> None:
    """Each stage receives the remainder, not a fresh full timeout."""
    seen = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return Completed()

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    adapter.run_native_stage(
        "extract_features", _stage_spec(tmp_path),
        run_dir=tmp_path, logs_dir=tmp_path / "logs", project_root=tmp_path,
        timeout=adapter.NATIVE_TIMEOUT_SECONDS,
        deadline=adapter.time.monotonic() + 12.5,
    )
    assert seen["timeout"] is not None and 0 < seen["timeout"] <= 12.5


def test_stage_argv_records_the_real_worker_invocation(tmp_path, monkeypatch) -> None:
    """The recorded command must be the one actually executed."""
    captured = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    adapter.run_native_stage(
        "extract_features", _stage_spec(tmp_path),
        run_dir=tmp_path, logs_dir=tmp_path / "logs", project_root=pathlib.Path("/proj"),
    )
    argv = captured["argv"]
    assert "-c" in argv, "the worker is launched with an explicit bootstrap"
    assert "extract_features" in argv
    assert captured["kwargs"]["cwd"] == str(pathlib.Path("/proj"))
    assert captured["kwargs"]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["kwargs"]["env"]["PYTHONPATH"].endswith("src")


# =========================================================================== #
# R1 regressions: upstream counters, real source depth, whole-input stability
# =========================================================================== #
def _raw_png(width: int, height: int, bit_depth: int, colour_type: int, row: bytes) -> bytes:
    """A PNG assembled from raw bytes, so the encoded depth is under our control.

    ``row`` is one scanline *without* its filter byte; the filter byte and the
    repetition for every row are added here, because a PNG whose IDAT does not
    contain ``height`` scanlines is not decodable and would exercise the decode
    failure path instead of the bit-depth check.
    """
    import struct
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, bit_depth, colour_type, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row for _ in range(height))
    payload = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
               + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))
    assert payload[24] == bit_depth and payload[25] == colour_type, "fixture is not what was asked"
    return payload


@pytest.mark.parametrize(
    ("label", "counts"),
    [
        ("ok_zero_errors_three", {"input_count": 3, "ok_count": 0, "error_count": 3}),
        ("ok_plus_error_mismatch", {"input_count": 3, "ok_count": 2, "error_count": 1}),
        ("input_count_only", {"input_count": 3, "ok_count": None, "error_count": None}),
        ("non_integer_count", {"input_count": 3, "ok_count": "3", "error_count": 0}),
    ],
)
def test_inconsistent_upstream_counters_are_input_errors(tmp_path, label, counts) -> None:
    """A SUCCESS run whose counters contradict its rows is not a valid upstream.

    Only ``input_count`` used to be checked, so a manifest declaring
    ``ok_count=0, error_count=3`` next to three OK rows was accepted.
    """
    run = make_upstream_run(tmp_path / "run")
    payload = json.loads((run / "run.json").read_text(encoding="utf-8"))
    for key, value in counts.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    (run / "run.json").write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )

    with pytest.raises(adapter.InputError) as caught:
        adapter.load_upstream_run(run)
    assert caught.value.code == "INPUT_ERROR", label


def test_consistent_upstream_counters_are_accepted(tmp_path) -> None:
    """Control: the same fixture with honest counters still loads."""
    run = make_upstream_run(tmp_path / "run")
    upstream = adapter.load_upstream_run(run)
    assert len(upstream.rows) == 3


@pytest.mark.parametrize("root_value", [[], None, 5, "invalid", True])
def test_non_object_run_json_is_a_cli_input_error(tmp_path, root_value) -> None:
    """Valid JSON with a wrong root type must not look like a backend failure."""
    from kozlov_lab.reconstruction import cli

    upstream = make_upstream_run(tmp_path / "upstream")
    (upstream / "run.json").write_text(json.dumps(root_value), encoding="utf-8")
    run_dir = tmp_path / "out"
    run_dir.mkdir()
    exit_code = cli.run_reconstruction(
        io_run=str(upstream), run_dir=run_dir, project_root=tmp_path
    )
    report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == "INPUT_ERROR"
    assert report["errors"][0]["code"] == "INPUT_ERROR"
    assert report["stages"] == []


def test_sixteen_bit_rgb_png_is_not_accepted_as_canonical(tmp_path) -> None:
    """A real 16-bit RGB PNG opens as mode ``RGB``; only the header shows the depth."""
    import struct

    run = make_upstream_run(tmp_path / "run", entries=[{"value": 1, "mask": None},
                                                       {"value": 2, "mask": None}])
    sixteen_bit = _raw_png(16, 12, 16, 2, struct.pack(">HHH", 65535, 256, 1) * 16)
    (run / "images/000000.png").write_bytes(sixteen_bit)

    from PIL import Image as _Image
    import io as _io
    assert _Image.open(_io.BytesIO(sixteen_bit)).mode == "RGB", "fixture must exercise the trap"

    upstream = adapter.load_upstream_run(run)
    with pytest.raises(adapter.InputError) as caught:
        adapter.stage_inputs(upstream, tmp_path / "staged")
    assert "bit" in caught.value.message.lower()


def test_a_mask_changed_during_the_run_is_detected(tmp_path) -> None:
    """A mask edit must be noticed even though the image representative is untouched."""
    run = make_upstream_run(tmp_path / "run", entries=[
        {"value": 30, "mask": None},
        {"value": 60, "mask": 1},
    ])
    upstream = adapter.load_upstream_run(run)
    staged = adapter.stage_inputs(upstream, tmp_path / "staged")

    before = adapter.snapshot_upstream_files(upstream)
    mask_path = run / staged.rows[1].mask_output
    mask_path.write_bytes(mask_path.read_bytes() + b"\x00")  # still a valid file, new bytes
    after = adapter.snapshot_upstream_files(upstream)

    changes = adapter.diff_snapshots(before, after)
    assert any(staged.rows[1].mask_output in change for change in changes), changes


def test_a_duplicate_alias_image_changed_during_the_run_is_detected(tmp_path) -> None:
    """The alias points at its own file; only checking representatives would miss it."""
    run = make_upstream_run(tmp_path / "run", entries=[
        {"value": 30, "mask": None},
        {"value": 30, "mask": None},   # same bytes -> deduplicated alias
        {"value": 90, "mask": None},
    ])
    upstream = adapter.load_upstream_run(run)
    staged = adapter.stage_inputs(upstream, tmp_path / "staged")
    alias = staged.rows[1]
    assert alias.representative_input_index == 0
    assert alias.image_output != staged.rows[0].image_output, "the alias has its own file"

    before = adapter.snapshot_upstream_files(upstream)
    alias_path = run / alias.image_output
    alias_path.write_bytes(alias_path.read_bytes() + b"\x00")
    after = adapter.snapshot_upstream_files(upstream)

    changes = adapter.diff_snapshots(before, after)
    assert any(alias.image_output in change for change in changes), changes


def test_unchanged_inputs_produce_an_empty_difference(tmp_path) -> None:
    """Negative control: the comparison must not report changes that are not there."""
    run = make_upstream_run(tmp_path / "run")
    upstream = adapter.load_upstream_run(run)
    adapter.stage_inputs(upstream, tmp_path / "staged")
    assert adapter.diff_snapshots(
        adapter.snapshot_upstream_files(upstream), adapter.snapshot_upstream_files(upstream)
    ) == []


# =========================================================================== #
# R2 regressions: controlled failure and identity mapping
# =========================================================================== #
def test_missing_backend_is_reported_as_unavailable(tmp_path, monkeypatch) -> None:
    """A verified upstream with no backend must exit 3 with evidence, not crash.

    ``backend_info`` is patched to raise, which is exactly what the real probe does
    when the module cannot be imported; the point under test is that the failure is
    *classified and recorded*, not that pycolmap is absent.
    """
    from kozlov_lab.reconstruction import cli

    run = make_upstream_run(tmp_path / "upstream")
    run_dir = tmp_path / "out" / "20261002T000000Z_deadbeef"
    run_dir.mkdir(parents=True)

    def unavailable(_executable):
        raise adapter.BackendUnavailable("pycolmap is not importable")

    monkeypatch.setattr(cli, "backend_info", unavailable)
    exit_code = cli.run_reconstruction(
        io_run=str(run), run_dir=run_dir, project_root=tmp_path
    )

    assert exit_code == 3
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "BACKEND_UNAVAILABLE"
    assert payload["errors"][0]["code"] == "BACKEND_UNAVAILABLE"
    # the identity map must still describe every original row
    rows = [json.loads(line) for line in
            (run_dir / "frames_map.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3
    assert [row["input_index"] for row in rows] == [0, 1, 2]


def test_mapping_survives_a_match_stage_failure(tmp_path, monkeypatch) -> None:
    """A stage failure must not empty the identity map or skew the counters."""
    from kozlov_lab.reconstruction import cli

    run = make_upstream_run(tmp_path / "upstream", entries=[
        {"value": 30, "mask": None}, {"value": 30, "mask": None}, {"value": 90, "mask": None},
    ])
    run_dir = tmp_path / "out" / "20261002T000000Z_deadbeef"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(cli, "backend_info",
                        lambda _e: {"name": "pycolmap", "version": adapter.BACKEND_PIN,
                                    "runtime": "cpu"})

    def extract_ok(stage, spec, **kwargs):
        return adapter.NativeStageResult(
            stage=stage, started_utc="T", finished_utc="T", returncode=0, timed_out=False,
            stdout_path="", stderr_path="", result={
                "num_images": 2, "cameras": {"1": {"width": 16, "height": 12}},
                "image_names_to_ids": {"images/000000.png": 1, "images/000002.png": 3},
                "actual_options": {"extraction": {}},
            }, error=None,
        )

    def match_fails(stage, spec, **kwargs):
        return adapter.NativeStageResult(
            stage=stage, started_utc="T", finished_utc="T", returncode=41, timed_out=False,
            stdout_path="", stderr_path="", result=None, error="native stage exited 41",
        )

    monkeypatch.setattr(cli, "run_native_stage",
                        lambda stage, spec, **kw: extract_ok(stage, spec) if stage == "extract_features"
                        else match_fails(stage, spec))

    exit_code = cli.run_reconstruction(io_run=str(run), run_dir=run_dir, project_root=tmp_path)

    assert exit_code == 4
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "BACKEND_ERROR"
    rows = [json.loads(line) for line in
            (run_dir / "frames_map.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3, "all original rows keep a mapping record on failure"
    assert [row["database_image_id"] for row in rows] == [1, 1, 3]
    assert all(row["registrations"] == [] for row in rows)
    counts = payload["counts"]
    assert counts["input_rows"] == 3
    assert counts["unique_images"] == 2
    assert counts["registered_unique_images"] + counts["unregistered_unique_images"] == 2, counts
    assert counts["unregistered_unique_images"] == 2
    assert counts["registered_unique_images"] == 0
    # the stages that never ran are absent, not fabricated
    assert [stage["name"] for stage in payload["stages"]] == ["extract_features", "match_exhaustive"]


# =========================================================================== #
# R3 regressions: recorded configuration
# =========================================================================== #
def test_actual_options_are_recorded_per_stage_from_the_version(tmp_path, monkeypatch) -> None:
    """The real serialized options must reach ``effective_options.native``."""
    from kozlov_lab.reconstruction import cli

    run = make_upstream_run(tmp_path / "upstream", entries=[
        {"value": 30, "mask": None}, {"value": 30, "mask": None}, {"value": 90, "mask": None},
    ])
    run_dir = tmp_path / "out" / "20261002T000000Z_deadbeef"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(cli, "backend_info",
                        lambda _e: {"name": "pycolmap", "version": adapter.BACKEND_PIN,
                                    "runtime": "cpu"})

    def stage(stage, spec, **kwargs):
        return adapter.NativeStageResult(
            stage=stage, started_utc="T", finished_utc="T", returncode=0, timed_out=False,
            stdout_path="", stderr_path="", result={
                "num_images": 2, "cameras": {"1": {"width": 16, "height": 12}},
                "image_names_to_ids": {"images/000000.png": 1, "images/000002.png": 3},
                "actual_options": {"probe": stage},
            }, error=None,
        )

    monkeypatch.setattr(cli, "run_native_stage", stage)
    cli.run_reconstruction(io_run=str(run), run_dir=run_dir, project_root=tmp_path)

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    native = payload["effective_options"]["native"]
    assert native["extract_features"] == {"probe": "extract_features"}
    assert native["incremental_mapping"] == {"probe": "incremental_mapping"}


def test_a_stage_that_never_ran_records_that_plainly(tmp_path, monkeypatch) -> None:
    """An unexecuted stage must not look like an executed one."""
    from kozlov_lab.reconstruction import cli

    run = make_upstream_run(tmp_path / "upstream")
    run_dir = tmp_path / "out" / "20261002T000000Z_deadbeef"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(cli, "backend_info",
                        lambda _e: {"name": "pycolmap", "version": adapter.BACKEND_PIN,
                                    "runtime": "cpu"})
    monkeypatch.setattr(cli, "run_native_stage", lambda stage, spec, **kw:
                        adapter.NativeStageResult(
                            stage=stage, started_utc="T", finished_utc="T", returncode=None,
                            timed_out=False, stdout_path="", stderr_path="", result=None,
                            error="the shared runtime budget was exhausted before this stage "
                                  "could start; it was not executed",
                            started=False,
                        ))
    cli.run_reconstruction(io_run=str(run), run_dir=run_dir, project_root=tmp_path)

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    native = payload["effective_options"]["native"]
    assert "<not_executed>" in native["extract_features"]
    assert payload["stages"][0]["started"] is False
