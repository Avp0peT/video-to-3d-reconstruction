"""Object-input session contract, using TEST_DATA only and no native processes.

Photo preparation exercises the actual T01 Python API in an isolated project.
Synthetic video records exercise receipt validation, never prove extraction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from PIL import Image

from integrations.object_input import session


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "TEST_DATA_project"
    (root / "outputs").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "TEST_DATA_object_input"\n', encoding="utf-8"
    )
    for package in ("io", "segmentation"):
        source = REPO_ROOT / "src" / "kozlov_lab" / package
        target = root / "src" / "kozlov_lab" / package
        target.mkdir(parents=True)
        for path in source.glob("*.py"):
            shutil.copyfile(path, target / path.name)
    implementation = root / "TEST_DATA_implementation.py"
    implementation.write_text("# TEST_DATA immutable implementation binding\n", encoding="utf-8")
    monkeypatch.setattr(session, "implementation_bindings", lambda project_root: [session.binding(implementation)])
    return root


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_photos(parent: Path, sizes: tuple[tuple[int, int], ...] = ((8, 6), (8, 6))) -> Path:
    source = parent / "TEST_DATA_photos"
    source.mkdir()
    for index, size in enumerate(sizes):
        image = Image.new("RGB", size, (10 + index, 30, 50))
        image.putpixel((index % size[0], index % size[1]), (220, 180, 60))
        image.save(source / f"photo_{index:02d}.png")
    return source


def prepare(project: Path, source: Path, *, dataset_id: str = "TEST_DATA_photos", output: Path | None = None) -> tuple[Path, dict]:
    output = output or project / "outputs" / "TEST_DATA_session"
    result = session.prepare_photos(source, dataset_id, "synthetic", output, project_root=project)
    return output, result


def test_missing_source_is_rejected_without_creating_output(project: Path, tmp_path: Path) -> None:
    output = project / "outputs" / "unused"
    with pytest.raises((ValueError, OSError)):
        prepare(project, tmp_path / "missing", output=output)
    assert not output.exists()


def test_file_source_is_rejected_without_creating_output(project: Path, tmp_path: Path) -> None:
    source = tmp_path / "not_a_directory.png"
    Image.new("RGB", (8, 6)).save(source)
    output = project / "outputs" / "unused"
    with pytest.raises((ValueError, OSError)):
        prepare(project, source, output=output)
    assert not output.exists()


@pytest.mark.parametrize("count", [0, 1, 97], ids=["empty", "one", "too_many"])
def test_photo_count_is_bounded_before_output(project: Path, tmp_path: Path, count: int) -> None:
    source = make_photos(tmp_path, ((8, 6),) * count)
    output = project / "outputs" / "unused"
    with pytest.raises(ValueError):
        prepare(project, source, output=output)
    assert not output.exists()


def test_existing_output_is_never_reused(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path)
    output = project / "outputs" / "existing"
    output.mkdir()
    marker = output / "previous_failure.txt"
    marker.write_bytes(b"TEST_DATA preserved old failure")
    with pytest.raises(ValueError):
        prepare(project, source, output=output)
    assert marker.read_bytes() == b"TEST_DATA preserved old failure"
    assert sorted(p.name for p in output.iterdir()) == [marker.name]


def test_output_outside_project_outputs_is_rejected(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path)
    output = tmp_path / "outside"
    with pytest.raises(ValueError):
        prepare(project, source, output=output)
    assert not output.exists()


def test_existing_dataset_is_never_reused(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path)
    dataset = project / "outputs/delegation/T01A/fixtures/TEST_DATA_photos"
    dataset.mkdir(parents=True)
    marker = dataset / "TEST_DATA_previous.txt"
    marker.write_bytes(b"previous bytes")
    with pytest.raises(ValueError):
        prepare(project, source)
    assert marker.read_bytes() == b"previous bytes"
    assert not (project / "outputs/TEST_DATA_session").exists()


def test_source_output_overlap_is_rejected(project: Path) -> None:
    source = make_photos(project / "outputs")
    output = source / "session"
    with pytest.raises(ValueError):
        prepare(project, source, output=output)
    assert not output.exists()


def test_dataset_output_overlap_is_rejected(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path)
    output = project / "outputs/delegation/T01A/fixtures/TEST_DATA_photos/session"
    with pytest.raises(ValueError):
        prepare(project, source, output=output)
    assert not output.exists()


@pytest.mark.parametrize(
    "dataset_id,data_kind", [("../escape", "synthetic"), ("valid", "unclassified")],
    ids=["unsafe_id", "invalid_kind"],
)
def test_invalid_identity_is_rejected_before_output(project: Path, tmp_path: Path, dataset_id: str, data_kind: str) -> None:
    source = make_photos(tmp_path)
    output = project / "outputs" / "unused"
    with pytest.raises(ValueError):
        session.prepare_photos(source, dataset_id, data_kind, output, project_root=project)
    assert not output.exists()


def test_prepared_photos_round_trip_without_claiming_reconstruction(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output, result = prepare(project, source)
    assert result["status"] == "INPUTS_PREPARED"
    assert result["exit_code"] == 0
    assert result["data_kind"] == "synthetic"
    assert result["photo_io"]["count"] == 2
    assert result["photo_io"]["unique_decoded_images"] == 2
    assert result["photo_io"]["sfm_compatibility"] == "READY_FOR_SEPARATE_SFM"
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    manifest = read_json(output / "photo_manifest.json")
    assert manifest["dataset_root"] == "outputs/delegation/T01A/fixtures/TEST_DATA_photos"
    assert manifest["frames"] == [
        {"image": "frames/000000.png", "mask": None},
        {"image": "frames/000001.png", "mask": None},
    ]
    run = read_json(Path(result["photo_io"]["run"]["path"]))
    assert run["status"] == "SUCCESS" and run["input_count"] == run["ok_count"] == 2
    assert run["error_count"] == 0
    for row, identity in zip(result["frames"], result["photo_io"]["identities"]):
        assert row["status"] == "PREPARED"
        assert identity["input_index"] == row["source_index"]
        assert identity["decoded_rgb_sha256"] == row["metadata"]["decoded_rgb_sha256"]
        assert row["metadata"]["lossy_resampling"] is False
        assert identity["frame_id"].startswith("f_")
    for key in ("segmentation_status", "sfm_status", "surface_status", "CAD_status"):
        assert result[key] == "NOT_RUN"
    assert result["coverage_status"] == "NOT_EVALUATED"
    assert result["scale_status"] == "unknown" and result["length_unit"] is None
    verified = session.verify_input(output, project_root=project)
    assert verified["status"] == "INPUT_BINDINGS_VALID"
    assert verified["count"] == 2 and verified["native_processes_started"] == 0


def test_directory_enumeration_is_sorted_and_does_not_recurse(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path)
    (source / "photo_00.png").rename(source / "z.png")
    (source / "photo_01.png").rename(source / "a.png")
    (source / "notes.txt").write_text("TEST_DATA non-photo", encoding="utf-8")
    nested = source / "nested"
    nested.mkdir()
    Image.new("RGB", (8, 6)).save(nested / "not_selected.png")
    _, result = prepare(project, source)
    assert result["status"] == "INPUTS_PREPARED"
    assert [Path(row["path"]).name for row in result["source_bindings"]] == ["a.png", "z.png"]
    assert [row["name"] for row in result["ignored_entries"]] == ["nested", "notes.txt"]
    assert result["photo_io"]["count"] == 2


def test_unsupported_photo_preserves_failure_and_unvisited_rows(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path, ((8, 6),) * 3)
    Image.new("RGBA", (8, 6), (1, 2, 3, 255)).save(source / "photo_01.png")
    output, result = prepare(project, source)
    assert result["status"] == "INPUT_ERROR" and result["exit_code"] == 2
    assert [row["status"] for row in result["frames"]] == ["PREPARED", "ERROR", "NOT_RUN"]
    assert "prepared" not in result["frames"][1]
    assert "prepared" not in result["frames"][2]
    assert result["photo_io"] is None
    assert not (output / "photo_manifest.json").exists()
    assert not (output / "photo_io").exists()
    assert read_json(output / "RESULT.json") == result
    with pytest.raises(ValueError, match="not successful"):
        session.verify_input(output, project_root=project)


def test_different_dimensions_are_preserved_and_sfm_is_blocked(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path, ((8, 6), (6, 8)))
    output, result = prepare(project, source)
    assert result["status"] == "INPUTS_PREPARED"
    assert result["photo_io"]["sfm_compatibility"] == "REQUIRES_EQUAL_DIMENSIONS"
    assert [row["metadata"]["prepared_dimensions"] for row in result["frames"]] == [[8, 6], [6, 8]]
    assert all(row["metadata"]["resample"] == "NONE" for row in result["frames"])
    assert result["sfm_status"] == "NOT_RUN"
    assert session.verify_input(output, project_root=project)["sfm_compatibility"] == "REQUIRES_EQUAL_DIMENSIONS"


def test_distinct_encodings_of_one_image_do_not_satisfy_sfm(project: Path, tmp_path: Path) -> None:
    source = tmp_path / "TEST_DATA_photos"
    source.mkdir()
    image = Image.new("RGB", (8, 6), (20, 30, 40))
    image.putpixel((2, 1), (200, 0, 0))
    image.save(source / "a.png", compress_level=0)
    image.save(source / "b.png", compress_level=9)
    assert sha256(source / "a.png") != sha256(source / "b.png")
    output, result = prepare(project, source)
    assert result["status"] == "INPUTS_PREPARED"
    assert result["photo_io"]["unique_decoded_images"] == 1
    assert result["photo_io"]["sfm_compatibility"] == "INSUFFICIENT_UNIQUE_IMAGES"
    assert session.verify_input(output, project_root=project)["sfm_compatibility"] == "INSUFFICIENT_UNIQUE_IMAGES"


@pytest.mark.parametrize("artifact", ["source", "prepared", "canonical", "run", "frames", "manifest"])
def test_verification_rejects_changed_bound_artifacts(project: Path, tmp_path: Path, artifact: str) -> None:
    source = make_photos(tmp_path)
    output, result = prepare(project, source)
    assert result["status"] == "INPUTS_PREPARED"
    if artifact == "source":
        path = source / "photo_00.png"
        path.write_bytes(path.read_bytes() + b"TEST_DATA changed original")
    elif artifact in ("prepared", "canonical"):
        key = "prepared_inputs" if artifact == "prepared" else "canonical_images"
        path = Path(result["photo_io"][key][0]["path"])
        with Image.open(path) as original:
            modified = original.copy()
        modified.putpixel((0, 0), (255, 0, 255))
        modified.save(path)
    elif artifact == "run":
        path = Path(result["photo_io"]["run"]["path"])
        data = read_json(path)
        data["input_count"] = 3
        write_json(path, data)
    elif artifact == "frames":
        path = Path(result["photo_io"]["frames"]["path"])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["image_path"] = "frames/000001.png"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        path = output / "photo_manifest.json"
        data = read_json(path)
        data["dataset_id"] = "TEST_DATA_other"
        write_json(path, data)
    with pytest.raises(ValueError):
        session.verify_input(output, project_root=project)


@pytest.mark.parametrize("scope", ["object_input", "photo_io"])
def test_verification_rejects_implementation_drift(project: Path, tmp_path: Path, scope: str) -> None:
    source = make_photos(tmp_path)
    output, result = prepare(project, source)
    assert result["status"] == "INPUTS_PREPARED"
    path = project / ("TEST_DATA_implementation.py" if scope == "object_input" else "src/kozlov_lab/io/runner.py")
    path.write_bytes(path.read_bytes() + b"\n# TEST_DATA changed implementation\n")
    with pytest.raises(ValueError, match="implementation"):
        session.verify_input(output, project_root=project)


def test_tampered_photo_metadata_is_rejected(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path)
    output, result = prepare(project, source)
    result["frames"][0]["metadata"]["decoded_rgb_sha256"] = "f" * 64
    write_json(output / "RESULT.json", result)
    with pytest.raises(ValueError, match="metadata"):
        session.verify_input(output, project_root=project)


def test_tampered_source_transform_is_rejected(project: Path, tmp_path: Path) -> None:
    source = make_photos(tmp_path)
    output, result = prepare(project, source)
    result["frames"][0]["metadata"]["raw_to_prepared"][0][2] += 1.0
    write_json(output / "RESULT.json", result)
    with pytest.raises(ValueError, match="transformation"):
        session.verify_input(output, project_root=project)


def make_test_video_receipt(project: Path, tmp_path: Path) -> tuple[Path, dict]:
    """Build a TEST_DATA receipt around real T01 processing, without a decoder.

    Source and tool files are text markers, not executable tools or real video.
    This only tests receipt validation; it is never extraction evidence.
    """
    photos = make_photos(tmp_path)
    directory, prepared = prepare(project, photos, output=project / "outputs/TEST_DATA_video_run")
    assert prepared["status"] == "INPUTS_PREPARED"
    source = tmp_path / "TEST_DATA_not_a_video.bin"
    source.write_bytes(b"TEST_DATA synthetic source marker; no video decoding performed")
    implementation = {}
    for relative in ("scripts/prepare_video.py", "integrations/video_io/timeline.py"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# TEST_DATA implementation identity marker\n", encoding="utf-8")
        implementation[relative] = sha256(path)
    tools = {}
    for name in ("TEST_DATA_ffmpeg_marker.bin", "TEST_DATA_ffprobe_marker.bin"):
        path = project / name
        path.write_bytes(b"TEST_DATA nonexecutable tool identity marker")
        tools[str(path)] = sha256(path)
    timeline = {
        "stream_index": 0, "time_base": [1, 1000], "width": 8, "height": 6,
        "pix_fmt": "rgb24", "sample_aspect_ratio": "1:1", "color_metadata": {},
        "rotation_degrees": 0, "display_matrix": None, "pts_field": "best_effort_timestamp",
        "frames": [
            {"decode_index": 0, "pts": 0, "time_base": [1, 1000], "time_seconds": 0.0},
            {"decode_index": 1, "pts": 500, "time_base": [1, 1000], "time_seconds": 0.5},
        ],
    }
    write_json(directory / "timeline.json", timeline)
    photo_io = prepared["photo_io"]
    manifest = read_json(directory / "photo_manifest.json")
    frames = []
    for index, identity in enumerate(photo_io["identities"]):
        frames.append({
            **timeline["frames"][index], "output_index": index, "status": "PHOTO_IO_OK",
            "photo_input_index": index, "photo_frame_id": identity["frame_id"],
            "image": manifest["frames"][index]["image"],
            "width": identity["prepared_dimensions"][0], "height": identity["prepared_dimensions"][1],
            "sha256": photo_io["prepared_inputs"][index]["sha256"],
            "decoded_rgb_sha256": identity["decoded_rgb_sha256"],
            "sharpness_laplacian_variance": 0.0, "export_pts_verified": True,
        })
    video = {
        "contract_version": "video-photo-v1", "status": "SUCCESS", "exit_code": 0,
        "test_fixture": "TEST_DATA fabricated receipt, not native extraction evidence",
        "source_video": str(source), "source_sha256": sha256(source), "source_sha256_after": sha256(source),
        "data_kind": "synthetic", "dataset_root": manifest["dataset_root"],
        "implementation": implementation, "tools": tools, "frames": frames,
        "timeline": {key: value for key, value in timeline.items() if key != "frames"},
        "decoded_count": 2, "counts": {"selected": 2, "accepted": 2, "rejected": 0},
        "selection": {"method": "uniform_presentation_time_nearest", "requested": 2,
                      "min_sharpness": 0.0, "exact_duplicate_policy": "reject"},
        "photo_io": {"run": photo_io["run"]["path"], "sha256": photo_io["run"]["sha256"],
                     "frames_sha256": photo_io["frames"]["sha256"]},
    }
    write_json(directory / "RESULT.json", video)
    return directory, video


def test_synthetic_video_receipt_can_be_adopted_without_native_execution(project: Path, tmp_path: Path) -> None:
    directory, video = make_test_video_receipt(project, tmp_path)
    original = (directory / "RESULT.json").read_bytes()
    output = project / "outputs/TEST_DATA_video_adoption"
    result = session.adopt_video(directory, output, project_root=project)
    assert result["status"] == "INPUTS_PREPARED" and result["exit_code"] == 0
    assert result["native_processes_started"] == 0
    assert result["frames"] == video["frames"]
    assert result["video_timeline_verification"] == "BOUND_EXISTING_REPORT_NOT_DECODE_REPLAY"
    assert (directory / "RESULT.json").read_bytes() == original
    verified = session.verify_input(output, project_root=project)
    assert verified["status"] == "INPUT_BINDINGS_VALID"
    assert verified["input_kind"] == "video" and verified["data_kind"] == "synthetic"
    assert verified["native_processes_started"] == 0


def test_swapped_video_pixels_cannot_keep_original_t01_identity(project: Path, tmp_path: Path) -> None:
    directory, video = make_test_video_receipt(project, tmp_path)
    first, second = video["frames"]
    identities_before = [(row["photo_input_index"], row["photo_frame_id"]) for row in video["frames"]]
    for key in ("image", "sha256", "decoded_rgb_sha256", "width", "height"):
        first[key], second[key] = second[key], first[key]
    assert [(row["photo_input_index"], row["photo_frame_id"]) for row in video["frames"]] == identities_before
    write_json(directory / "RESULT.json", video)
    output = project / "outputs/TEST_DATA_bad_video_adoption"
    with pytest.raises(ValueError, match="same pixels"):
        session.adopt_video(directory, output, project_root=project)
    assert not output.exists()


def test_failed_video_run_cannot_be_adopted(project: Path, tmp_path: Path) -> None:
    video_run = tmp_path / "TEST_DATA_failed_video"
    video_run.mkdir()
    write_json(video_run / "RESULT.json", {
        "contract_version": "video-photo-v1", "status": "INPUT_ERROR", "exit_code": 2,
        "data_kind": "synthetic", "frames": [], "test_fixture": "TEST_DATA",
    })
    before = (video_run / "RESULT.json").read_bytes()
    output = project / "outputs" / "adopted"
    with pytest.raises(ValueError, match="not a successful"):
        session.adopt_video(video_run, output, project_root=project)
    assert not output.exists()
    assert (video_run / "RESULT.json").read_bytes() == before


def test_video_wrapper_forwards_request_once_and_preserves_failure(project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A TEST_DATA loader stub proves delegation only, with zero native runs."""
    video = tmp_path / "TEST_DATA_video.bin"
    video.write_bytes(b"TEST_DATA deliberately not a real video")
    calls = []

    def execute(args):
        calls.append(args)
        return 5

    loader = SimpleNamespace(exec_module=lambda module: None)
    monkeypatch.setattr(session, "ROOT", project)
    monkeypatch.setattr(session.importlib.util, "spec_from_file_location", lambda name, path: SimpleNamespace(loader=loader))
    monkeypatch.setattr(session.importlib.util, "module_from_spec", lambda spec: SimpleNamespace(execute=execute))
    output = project / "outputs" / "video_wrapper"
    result = session.prepare_video(video, "TEST_DATA_video", "synthetic", output,
                                   count=7, ffmpeg="TEST_DATA_ffmpeg", ffprobe="TEST_DATA_ffprobe",
                                   project_root=project)
    assert len(calls) == 1
    assert vars(calls[0]) == {
        "video": str(video.resolve()), "dataset_id": "TEST_DATA_video", "data_kind": "synthetic",
        "output": str(output / "video"), "count": 7, "ffmpeg": "TEST_DATA_ffmpeg",
        "ffprobe": "TEST_DATA_ffprobe", "stream_index": None, "min_sharpness": 0.0,
    }
    assert result["status"] == "PREPARATION_FAILED" and result["exit_code"] == 5
    assert result["photo_io"] is None and result["frames"] == []
    assert read_json(output / "RESULT.json") == result
    with pytest.raises(ValueError, match="not successful"):
        session.verify_input(output, project_root=project)
