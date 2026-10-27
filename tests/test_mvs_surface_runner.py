"""TEST_DATA: isolated recipe and failure-control tests; no native process runs.

The scene reader and supervisor are mocked. Images really are tiny PNGs, and
geometry really is parsed from explicit ASCII PLY fixtures. Dummy .mvs outputs
only test nonempty file binding: they are not valid native scenes and do not
prove OpenMVS loading, mask consumption, reconstruction quality, or CAD delivery.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from integrations.mvs_surface import runner, supervision


POINT_CLOUD = b"""ply
format ascii 1.0
comment TEST_DATA explicit triangle vertices, no faces
element vertex 3
property float x
property float y
property float z
end_header
0 0 0
2 0 0
0 3 0
"""
TRIANGLE = b"""ply
format ascii 1.0
comment TEST_DATA explicit nondegenerate triangle
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
2 0 0
0 3 0
3 0 1 2
"""
EMPTY_CLOUD = b"""ply
format ascii 1.0
element vertex 0
property float x
property float y
property float z
end_header
"""
IDENTITY4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png(path: Path, pixels: list[int], *, mode: str = "L", size=(3, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, size)
    image.putdata(pixels)
    image.save(path, format="PNG")


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """TEST_DATA: exactly two metadata records, never a real scene decode."""
    source = tmp_path / "inputs"
    source.mkdir()
    paths = {key: source / key for key in ("images", "masks", "validity", "tools")}
    for path in paths.values():
        path.mkdir()
    scene_path = source / "TEST_DATA_scene.mvs"
    scene_bytes = b"TEST_DATA mocked MVSI input; not an actual native scene\n"
    scene_path.write_bytes(scene_bytes)
    tool_hashes = {}
    for name in ("DensifyPointCloud.exe", "ReconstructMesh.exe"):
        data = b"TEST_DATA non-executable tool placeholder: " + name.encode("ascii")
        (paths["tools"] / name).write_bytes(data)
        tool_hashes[name] = _sha(data)
    (paths["tools"] / "TEST_DATA_runtime.dll").write_bytes(b"TEST_DATA DLL placeholder")
    monkeypatch.setattr(runner, "TOOL_HASHES", tool_hashes)
    images = []
    for index, stem in enumerate(("first", "second")):
        name = f"images/{stem}.png"
        _png(paths["images"] / name, [20, 40, 60, 80, 100, 120])
        _png(paths["masks"] / f"{stem}.mask.png", [255, 255, 0, 0, 0, 0])
        _png(paths["validity"] / f"{stem}.validity.png", [255, 255, 0, 255, 255, 255])
        images.append({
            "index": index, "name": name, "name_raw": name, "mask_name": "",
            "platform_id": 0, "camera_id": 0, "pose_id": index,
            "global_id": (17, 91)[index], "width": 3, "height": 2,
            "K": [[2, 0, 1], [0, 2, 0.5], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "C": [index, 0, 0],
            "min_depth": 0, "avg_depth": 0, "max_depth": 0,
            "view_scores": [{"image_index": 1 - index, "points": 3,
                             "scale": 1, "angle": 0.25, "area": 0.5, "score": 7}],
        })
    metadata = {
        "path": str(scene_path.resolve()), "bytes": len(scene_bytes), "sha256": _sha(scene_bytes),
        "version": 7, "byte_order": "little", "images": images, "point_count": 3,
        "platforms": [{"name": "TEST_DATA rig", "cameras": [{
            "name": "TEST_DATA camera", "band_name": "RGB", "width": 3, "height": 2,
            "K": [[2, 0, 1], [0, 2, 0.5], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "C": [0, 0, 0],
        }], "poses": [
            {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "C": [0, 0, 0]},
            {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "C": [1, 0, 0]},
        ]}],
        "min_point_views": 2, "point_bbox": {"min": [0, 0, 0], "max": [2, 3, 0]},
        "line_count": 0, "line_bbox": None, "observation_count": 6,
        "point_normal_count": 0, "point_color_count": 0,
        "line_normal_count": 0, "line_color_count": 0, "transform": copy.deepcopy(IDENTITY4),
        "obb": {"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "pt_min": [0, 0, 0], "pt_max": [2, 3, 0]},
    }
    reads = []

    def fake_read_scene(path):
        assert Path(path).resolve() == scene_path.resolve()
        reads.append(str(path))
        return copy.deepcopy(metadata)

    def forbidden_native(*args, **kwargs):
        pytest.fail("TEST_DATA fixture must never launch native code")

    monkeypatch.setattr(runner, "read_scene", fake_read_scene)
    monkeypatch.setattr(supervision, "run_supervised", forbidden_native)
    arguments = {
        "scene_path": scene_path, "image_root": paths["images"], "mask_root": paths["masks"],
        "validity_root": paths["validity"], "tool_root": paths["tools"],
        "total_seconds": 30, "stage_seconds": 10,
        "memory_bytes": 128 * 1024**2, "output_bytes": 16 * 1024**2,
        "input_origin": "synthetic",
    }
    return SimpleNamespace(root=source, paths=paths, scene=scene_path, metadata=metadata,
                           arguments=arguments, reads=reads)


def _plan(inputs, **changes):
    return runner.make_plan(**(inputs.arguments | changes))


def _install_supervisor(monkeypatch, *, response=None, mutate=None):
    """Return fabricated supervisor data; TEST_DATA pid is never an OS process."""
    calls = []

    def fake(argv, *, cwd, output, wall_seconds, memory_bytes, output_bytes, label):
        calls.append({"argv": argv, "cwd": cwd, "output": output, "label": label,
                      "wall_seconds": wall_seconds, "memory_bytes": memory_bytes,
                      "output_bytes": output_bytes})
        stem = "__mvs_dense" if label == "dense" else "__mvs_surface"
        (cwd / (stem + ".mvs")).write_bytes(b"TEST_DATA opaque nonempty .mvs; size binding only")
        (cwd / (stem + ".ply")).write_bytes(POINT_CLOUD if label == "dense" else TRIANGLE)
        if label == "mesh":
            (cwd / (stem + "_raw.ply")).write_bytes(TRIANGLE)
        # TEST_DATA mirrors the real adapter's output directory and captured
        # streams. The dense native log is required by the mask warning guard;
        # these bytes prove only fixture control flow, never native consumption.
        (output / "stdout.txt").write_bytes(b"")
        (output / "stderr.txt").write_bytes(b"")
        if label == "dense":
            (cwd / "DensifyPointCloud-TEST_DATA.log").write_bytes(
                b"TEST_DATA OpenMVS log: normal completion\n")
        if mutate:
            mutate(label, cwd)
        # This sentinel only exercises the returned schema. No process is
        # created, inspected, signalled, or claimed to exist by these tests.
        result = {"TEST_DATA": "mocked supervisor; pid is synthetic, no process created", "pid": 730001,
                  "status": "completed", "root_exit_code": 0, "reclaim_confirmed": True,
                  "argv": list(argv), "cwd": str(cwd), "errors": [], "log_errors": [],
                  "mvs_surface_adapter": {"schema": "mvs-surface-supervision-v1",
                                          "output": str(output)}}
        if response:
            result.update(response)
        return result

    monkeypatch.setattr(supervision, "run_supervised", fake)
    return calls


def test_planning_is_read_only_deterministic_and_explicitly_unverified(inputs):
    before = _snapshot(inputs.root)
    first = _plan(inputs)
    assert first == _plan(inputs)
    assert _snapshot(inputs.root) == before
    assert len(inputs.reads) == 2
    assert first["native_execution"] == "NOT_RUN"
    assert first["input_origin"] == "caller_declared_synthetic"
    assert first["mask_source_chain"] == first["mask_consumption"] == "NOT_VERIFIED"
    assert first["mask_quality"] == first["cross_view"] == "NOT_EVALUATED"
    assert first["scale_status"] == "unknown" and first["length_unit"] is None
    assert [(f["foreground_pixels"], f["unknown_pixels"]) for f in first["frames"]] == [(2, 1), (2, 1)]
    assert first["budgets"]["native_launch_cap"] == 2
    assert first["budgets"]["retries"] == 0


@pytest.mark.parametrize("kind,suffix", [("masks", ".mask.png"), ("validity", ".validity.png")])
def test_missing_required_labels_are_rejected(inputs, kind, suffix):
    (inputs.paths[kind] / ("first" + suffix)).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        _plan(inputs)


@pytest.mark.parametrize("mask,validity", [
    ([0, 0, 0, 0, 0, 0], [255, 255, 255, 255, 255, 255]),
    ([255, 0, 0, 0, 0, 0], [0, 255, 255, 255, 255, 255]),
])
def test_empty_foreground_and_foreground_in_unknown_domain_are_rejected(inputs, mask, validity):
    _png(inputs.paths["masks"] / "first.mask.png", mask)
    _png(inputs.paths["validity"] / "first.validity.png", validity)
    with pytest.raises(ValueError, match="empty foreground|unknown mask domain"):
        _plan(inputs)


@pytest.mark.parametrize("kind,suffix", [("masks", ".mask.png"), ("validity", ".validity.png")])
def test_nonbinary_labels_are_rejected(inputs, kind, suffix):
    _png(inputs.paths[kind] / ("first" + suffix), [255, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="non-binary"):
        _plan(inputs)


def test_label_dimensions_must_match_the_scene_camera(inputs):
    _png(inputs.paths["masks"] / "first.mask.png", [255] * 6, size=(2, 3))
    with pytest.raises(ValueError, match="dimensions"):
        _plan(inputs)


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", 1)])
def test_scene_reader_snapshot_must_match_frozen_source(inputs, field, value):
    inputs.metadata[field] = value
    with pytest.raises(ValueError):
        _plan(inputs)


def test_tool_hash_drift_fails_preflight(inputs):
    (inputs.paths["tools"] / "DensifyPointCloud.exe").write_bytes(b"TEST_DATA changed tool")
    with pytest.raises(ValueError, match="tool identity mismatch"):
        _plan(inputs)


def test_private_dll_set_is_required(inputs):
    (inputs.paths["tools"] / "TEST_DATA_runtime.dll").unlink()
    with pytest.raises(ValueError, match="DLL"):
        _plan(inputs)


@pytest.mark.parametrize("bad_views", [0, 1])
def test_sparse_points_need_at_least_two_observing_images(inputs, bad_views):
    inputs.metadata["min_point_views"] = bad_views
    with pytest.raises(ValueError):
        _plan(inputs)


@pytest.mark.parametrize("field,value", [("point_count", 0), ("images", [])])
def test_insufficient_scene_structure_is_rejected(inputs, field, value):
    inputs.metadata[field] = value
    with pytest.raises(ValueError, match="images|sparse"):
        _plan(inputs)


def test_scene_transform_requires_a_separate_profile(inputs):
    inputs.metadata["transform"][0][3] = 1
    with pytest.raises(ValueError, match="nonidentity"):
        _plan(inputs)


@pytest.mark.parametrize("name", ["other/first.png", "other/FIRST.png"])
def test_mask_basename_collisions_are_rejected_even_across_directories(inputs, name):
    inputs.metadata["images"][1]["name"] = name
    with pytest.raises(ValueError, match="mask basename collision"):
        _plan(inputs)


def test_unicode_normalized_mask_basename_collision_is_rejected(inputs):
    inputs.metadata["images"][0]["name"] = "images/café.png"
    inputs.metadata["images"][1]["name"] = "other/cafe\u0301.png"
    _png(inputs.paths["images"] / "images/café.png", [10] * 6)
    _png(inputs.paths["masks"] / "café.mask.png", [255] * 6)
    _png(inputs.paths["validity"] / "café.validity.png", [255] * 6)
    with pytest.raises(ValueError, match="mask basename collision"):
        _plan(inputs)


@pytest.mark.parametrize("change", [{"mask_name": "provided.png"}, {"name": "__mvs_private/first.png"}])
def test_scene_mask_names_and_reserved_staging_names_are_rejected(inputs, change):
    inputs.metadata["images"][0].update(change)
    with pytest.raises(ValueError, match="maskName|reserved"):
        _plan(inputs)


def test_image_path_cannot_escape_declared_root(inputs):
    _png(inputs.root / "outside.png", [10] * 6)
    inputs.metadata["images"][0]["name"] = "../outside.png"
    with pytest.raises(ValueError):
        _plan(inputs)


@pytest.mark.parametrize("key,bad", [
    ("total_seconds", 9), ("total_seconds", 1801), ("total_seconds", True),
    ("stage_seconds", 4), ("stage_seconds", 31), ("stage_seconds", 5.0),
    ("memory_bytes", 128 * 1024**2 - 1), ("memory_bytes", 12 * 1024**3 + 1),
    ("output_bytes", 16 * 1024**2 - 1), ("output_bytes", 8 * 1024**3 + 1),
])
def test_budget_bounds_reject_before_scene_reads(inputs, key, bad):
    with pytest.raises(ValueError, match="budget"):
        _plan(inputs, **{key: bad})
    assert inputs.reads == []


@pytest.mark.parametrize("change", [{"input_origin": "measured"}, {"ignore_label": True}, {"ignore_label": 255}])
def test_unrecognized_origin_or_mask_semantics_are_rejected(inputs, change):
    with pytest.raises(ValueError):
        _plan(inputs, **change)


def _options(argv):
    assert len(argv[1:]) % 2 == 0
    pairs = list(zip(argv[1::2], argv[2::2]))
    assert len(dict(pairs)) == len(pairs), "duplicate arguments could override frozen policy"
    return dict(pairs)


def test_exact_two_stage_recipe_preserves_raw_geometry_without_implicit_repair(inputs, tmp_path):
    plan = _plan(inputs)
    work = tmp_path / "not-created-work"
    commands = runner.commands(plan, work)
    assert [name for name, _ in commands] == ["dense", "mesh"]
    dense, mesh = [_options(argv) for _, argv in commands]
    assert Path(commands[0][1][0]).name == "DensifyPointCloud.exe"
    assert Path(commands[1][1][0]).name == "ReconstructMesh.exe"
    for options in (dense, mesh):
        # OpenMVS MAKE_PATH uses the working-folder prefix for relative outputs.
        # A trailing separator keeps both native logs and depth maps in work.
        assert options['-w'] == str(work) + os.sep
        assert Path(options['-w'] + 'DensifyPointCloud-TEST_DATA.log').parent == work
        assert Path(options['-w'] + 'depth0004.dmap').parent == work
        assert options["--max-threads"] == "2"
        assert options["--archive-type"] == "1"
        assert options["--verbosity"] == "3"  # OpenMVS raw mesh export path.
        assert options["--config-file"] == str(work / "__mvs_empty.cfg")
    assert dense["-i"] == str(work / "__mvs_input.mvs")
    assert dense["-o"] == mesh["-i"] == str(work / "__mvs_dense.mvs")
    assert dense["--ignore-mask-label"] == "0"
    assert dense["--mask-path"] == str(work / "__mvs_masks")
    assert dense["--postprocess-dmaps"] == "0"
    assert dense["--remove-dmaps"] == "0"
    assert dense["--filter-point-cloud"] == "0"
    assert mesh["-o"] == str(work / "__mvs_surface.mvs")
    assert mesh["--export-type"] == "ply"
    assert mesh["--decimate"] == "1"
    for option in ("target-face-num", "remove-spurious", "remove-spikes", "close-holes",
                   "smooth", "edge-length", "crop-to-roi", "split-max-area"):
        assert mesh["--" + option] == "0"
    assert not work.exists()


def test_unmasked_control_is_explicit_and_remains_separate(inputs, tmp_path):
    plan = _plan(inputs, ignore_label=-1)
    dense = _options(runner.commands(plan, tmp_path)[0][1])
    assert dense["--ignore-mask-label"] == "-1"
    assert len(plan["frames"]) == 2  # Control does not drop frozen mask inputs.


def test_success_reads_explicit_ply_and_does_not_claim_surface_acceptance(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)
    before = _snapshot(inputs.root)
    calls = _install_supervisor(monkeypatch)
    target = tmp_path / "result"
    result = runner.execute_plan(plan, target)
    assert [call["label"] for call in calls] == ["dense", "mesh"]
    assert result["status"] == "SURFACE_GENERATED_NOT_ACCEPTED" and result["exit_code"] == 0
    assert result["dense_status"] == result["raw_mesh_status"] == "NONEMPTY_FINITE_GEOMETRY"
    assert result["mask_consumption"] == result["mask_source_chain"] == "NOT_VERIFIED"
    assert result["quality_status"] == "NOT_EVALUATED" and result["CAD_status"] == "NOT_RUN"
    assert result["input_origin"] == "caller_declared_synthetic"
    assert result["scale_status"] == "unknown" and result["length_unit"] is None
    assert result["input_drift"] == []
    assert result["launch_attempts"] == result["native_launches"] == result["known_native_launches"] == 2
    assert result["launch_count_complete"] is True
    assert _snapshot(inputs.root) == before
    for call in calls:
        assert call["cwd"] == target / "work"
        assert 0 < call["wall_seconds"] <= 10
        assert call["memory_bytes"] == 128 * 1024**2
        assert call["output_bytes"] == 16 * 1024**2
    dense, mesh = result["stages"]
    assert [(row["vertices"], row["triangles"], row["bbox"]) for row in dense["outputs"]["geometry"]] == [
        (3, 0, [[0, 0, 0], [2, 3, 0]])]
    assert [(row["role"], row["vertices"], row["triangles"]) for row in mesh["outputs"]["geometry"]] == [
        ("pre_cleanup_surface", 3, 1), ("mesh", 3, 1)]
    assert (target / "work" / "__mvs_surface_raw.ply").read_bytes() == TRIANGLE
    assert json.loads((target / "result.json").read_text(encoding="utf-8")) == result


@pytest.mark.parametrize("response", [
    {"status": "failed"}, {"status": "stopped"}, {"root_exit_code": 1},
    {"pid": None}, {"pid": 0}, {"pid": -1}, {"pid": True}, {"pid": "730001"},
    {"argv": ["TEST_DATA wrong tool"]}, {"cwd": "TEST_DATA wrong working directory"},
    {"root_exit_code": None}, {"root_exit_code": False}, {"root_exit_code": "0"},
    {"reclaim_confirmed": False}, {"reclaim_confirmed": None}, {"reclaim_confirmed": 1},
    {"errors": ["TEST_DATA sampler failed"]}, {"log_errors": ["TEST_DATA log write failed"]},
])
def test_supervision_failure_stops_before_mesh_without_retry(inputs, tmp_path, monkeypatch, response):
    plan = _plan(inputs)
    calls = _install_supervisor(monkeypatch, response=response)
    result = runner.execute_plan(plan, tmp_path / "failed")
    assert [call["label"] for call in calls] == ["dense"]
    assert result["status"] == "FAILED" and result["exit_code"] == 4
    pid = response.get("pid", 730001)
    expected_dense = ("PROCESS_CREATED_OUTPUT_UNVERIFIED"
                      if type(pid) is int and pid > 0 else "NOT_RUN")
    assert result["dense_status"] == expected_dense
    assert result["raw_mesh_status"] == "NOT_RUN"
    assert len(result["stages"]) == 1
    if "pid" in response and response["pid"] is not None:
        assert result["native_launches"] is None
        assert result["known_native_launches"] == 0
        assert result["launch_count_complete"] is False


def test_supervisor_exception_is_recorded_without_mesh_or_retry(inputs, tmp_path, monkeypatch):
    calls = []

    def unknown_observation(*args, **kwargs):
        calls.append(kwargs["label"])
        raise supervision.ObservationError("TEST_DATA failed resource sample")

    monkeypatch.setattr(supervision, "run_supervised", unknown_observation)
    result = runner.execute_plan(_plan(inputs), tmp_path / "observation-failed")
    assert calls == ["dense"]
    assert result["status"] == "FAILED"
    assert result["raw_mesh_status"] == "NOT_RUN"
    assert "failed resource sample" in result["error"]
    assert result["launch_attempts"] == 1
    assert result["native_launches"] is None
    assert result["known_native_launches"] == 0
    assert result["launch_count_complete"] is False


def test_missing_creation_outcome_keeps_native_launch_count_unknown(inputs, tmp_path, monkeypatch):
    calls = []

    def incomplete(*args, **kwargs):
        calls.append(kwargs["label"])
        return {"TEST_DATA": "incomplete mocked outcome", "status": "failed"}

    monkeypatch.setattr(supervision, "run_supervised", incomplete)
    result = runner.execute_plan(_plan(inputs), tmp_path / "unknown-creation")
    assert calls == ["dense"]
    assert result["status"] == "FAILED"
    assert result["native_launches"] is None
    assert result["known_native_launches"] == 0
    assert result["launch_count_complete"] is False


@pytest.mark.parametrize("payload", [EMPTY_CLOUD, b"TEST_DATA invalid PLY", POINT_CLOUD.replace(b"2 0 0", b"nan 0 0")])
def test_invalid_or_empty_dense_geometry_blocks_mesh(inputs, tmp_path, monkeypatch, payload):
    def alter(label, work):
        (work / "__mvs_dense.ply").write_bytes(payload)

    calls = _install_supervisor(monkeypatch, mutate=alter)
    result = runner.execute_plan(_plan(inputs), tmp_path / "bad-cloud")
    assert [call["label"] for call in calls] == ["dense"]
    assert result["status"] == "FAILED"
    assert result["raw_mesh_status"] == "NOT_RUN"


@pytest.mark.parametrize("filename,payload", [
    ("__mvs_surface_raw.ply", POINT_CLOUD),
    ("__mvs_surface_raw.ply", TRIANGLE.replace(b"0 3 0", b"4 0 0")),
    ("__mvs_surface.ply", TRIANGLE.replace(b"3 0 1 2", b"3 0 1 3")),
    ("__mvs_surface.mvs", b""),
])
def test_mesh_requires_nonempty_raw_and_final_nondegenerate_geometry(inputs, tmp_path, monkeypatch, filename, payload):
    def alter(label, work):
        if label == "mesh":
            (work / filename).write_bytes(payload)

    calls = _install_supervisor(monkeypatch, mutate=alter)
    result = runner.execute_plan(_plan(inputs), tmp_path / "bad-mesh")
    assert [call["label"] for call in calls] == ["dense", "mesh"]
    assert result["status"] == "FAILED"
    assert result["dense_status"] == "NONEMPTY_FINITE_GEOMETRY"
    assert result["raw_mesh_status"] == "NOT_RUN"


def test_missing_raw_mesh_export_cannot_be_silently_replaced_by_final_mesh(inputs, tmp_path, monkeypatch):
    def alter(label, work):
        if label == "mesh":
            (work / "__mvs_surface_raw.ply").unlink()

    _install_supervisor(monkeypatch, mutate=alter)
    result = runner.execute_plan(_plan(inputs), tmp_path / "missing-raw")
    assert result["status"] == "FAILED"
    assert result["raw_mesh_status"] == "NOT_RUN"


@pytest.mark.parametrize("response,expected_calls", [({}, 2), ({"root_exit_code": 1}, 1)])
def test_existing_output_is_never_resumed_or_overwritten(inputs, tmp_path, monkeypatch, response, expected_calls):
    calls = _install_supervisor(monkeypatch, response=response)
    plan = _plan(inputs)
    target = tmp_path / "one-run"
    runner.execute_plan(plan, target)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="must not exist"):
        runner.execute_plan(plan, target)
    assert len(calls) == expected_calls
    assert _snapshot(target) == before


@pytest.mark.parametrize("section,key,value", [
    ("budgets", "retries", 1), ("budgets", "threads", 8),
    ("options", "mesh", {"close-holes": 1}),
])
def test_modified_plan_is_rejected_before_output_creation_or_launch(inputs, tmp_path, section, key, value):
    plan = _plan(inputs)
    plan[section][key] = value
    target = tmp_path / "tampered"
    with pytest.raises(ValueError, match="plan does not match"):
        runner.execute_plan(plan, target)
    assert not target.exists()


def test_changed_source_after_planning_is_rejected_before_launch(inputs, tmp_path):
    plan = _plan(inputs)
    _png(inputs.paths["images"] / "images/first.png", [30] * 6)
    target = tmp_path / "changed-source"
    with pytest.raises(ValueError, match="plan does not match"):
        runner.execute_plan(plan, target)
    assert not target.exists()


@pytest.mark.parametrize("root_name", ["images", "masks", "validity", "tools", "scene-parent"])
def test_output_cannot_be_nested_in_any_input_root(inputs, root_name):
    base = inputs.root if root_name == "scene-parent" else inputs.paths[root_name]
    target = base / "new-output"
    with pytest.raises(ValueError, match="inside"):
        runner.execute_plan(_plan(inputs), target)
    assert not target.exists()


@pytest.mark.parametrize("staged", [False, True], ids=["source", "staged-copy"])
def test_drift_during_execution_is_reported_even_with_geometry(inputs, tmp_path, monkeypatch, staged):
    changed = None

    def alter(label, work):
        nonlocal changed
        if label == "dense":
            changed = work / "images/first.png" if staged else inputs.paths["images"] / "images/first.png"
            changed.write_bytes(b"TEST_DATA changed after staging")

    _install_supervisor(monkeypatch, mutate=alter)
    result = runner.execute_plan(_plan(inputs), tmp_path / "drift")
    assert result["status"] == "INPUT_DRIFT" and result["exit_code"] == 4
    assert str(changed.resolve()) in result["input_drift"]


def test_changed_empty_config_cannot_affect_mesh_command(inputs, tmp_path, monkeypatch):
    def change_config(label, work):
        if label == "dense":
            (work / "__mvs_empty.cfg").write_text("close-holes=100\n", encoding="utf-8")

    calls = _install_supervisor(monkeypatch, mutate=change_config)
    target = tmp_path / "changed-config"
    result = runner.execute_plan(_plan(inputs), target)
    assert [call["label"] for call in calls] == ["dense"]
    assert result["status"] == "INPUT_DRIFT" and result["exit_code"] == 4
    assert result["raw_mesh_status"] == "NOT_RUN"
    assert "empty command configuration changed" in result["error"]
    assert str((target / "work/__mvs_empty.cfg").resolve()) in result["input_drift"]


def test_copy_rechecks_identity_and_uses_exclusive_creation(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"TEST_DATA original")
    frozen = runner.binding(source)
    target = tmp_path / "target.bin"
    target.write_bytes(b"TEST_DATA retained target")
    with pytest.raises(FileExistsError):
        runner._copy(frozen, target)
    assert target.read_bytes() == b"TEST_DATA retained target"
    source.write_bytes(b"TEST_DATA modified")
    with pytest.raises(ValueError, match="source identity changed"):
        runner._copy(frozen, tmp_path / "new.bin")
    assert not (tmp_path / "new.bin").exists()


def test_staging_respects_deadline_before_images(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)
    work = tmp_path / "stage"
    work.mkdir()
    monkeypatch.setattr(runner.time, "monotonic", lambda: 20.0)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        runner.stage_inputs(plan, work, deadline=10.0)
    assert not (work / "images").exists()


def test_post_native_disk_threshold_stops_before_mesh(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)
    calls = _install_supervisor(monkeypatch)
    monkeypatch.setattr(supervision, "strict_directory_size", lambda path: 16 * 1024**2 + 1)
    result = runner.execute_plan(plan, tmp_path / "over-output-threshold")
    assert [call["label"] for call in calls] == ["dense"]
    assert result["status"] == "BUDGET_EXCEEDED" and result["exit_code"] == 5
    assert result["raw_mesh_status"] == "NOT_RUN"


def test_shared_deadline_after_mesh_prevents_success(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)
    clock = {"now": 0.0}

    def expire(label, work):
        if label == "mesh":
            clock["now"] = 31.0

    monkeypatch.setattr(runner.time, "monotonic", lambda: clock["now"])
    calls = _install_supervisor(monkeypatch, mutate=expire)
    result = runner.execute_plan(plan, tmp_path / "late-mesh")
    assert [call["label"] for call in calls] == ["dense", "mesh"]
    assert result["status"] == "BUDGET_EXCEEDED" and result["exit_code"] == 5
    assert result["raw_mesh_status"] == "NOT_RUN"


def test_final_directory_observation_failure_cannot_be_reported_as_success(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)
    calls = _install_supervisor(monkeypatch)
    real_sample = supervision.strict_directory_size
    samples = []

    def final_unknown(path):
        samples.append(path)
        if len(samples) == 3:
            raise supervision.ObservationError("TEST_DATA final observation unavailable")
        return real_sample(path)

    monkeypatch.setattr(supervision, "strict_directory_size", final_unknown)
    result = runner.execute_plan(plan, tmp_path / "unknown-final-size")
    assert [call["label"] for call in calls] == ["dense", "mesh"]
    assert len(samples) == 3
    assert result["status"] == "OBSERVATION_ERROR" and result["exit_code"] == 4
    assert "final observation unavailable" in result["final_observation_error"]


def test_strict_directory_size_counts_files_and_empty_subdirectories(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"abc")
    (tmp_path / "nested/b.bin").write_bytes(b"12345")
    assert supervision.strict_directory_size(tmp_path) == 8


@pytest.mark.parametrize("full_link_count", [1, 2], ids=["ordinary-file", "multiple-hardlinks"])
def test_windows_cached_entry_metadata_cannot_replace_full_path_identity(tmp_path, monkeypatch, full_link_count):
    """TEST_DATA stat records; no OS hardlinks or native processes are created."""
    path = tmp_path / "TEST_DATA_file.bin"
    payload = b"TEST_DATA"
    path.write_bytes(payload)
    original_lstat = Path.lstat
    actual = original_lstat(path)
    cached = SimpleNamespace(st_mode=actual.st_mode, st_size=len(payload), st_nlink=0,
                             st_ino=0, st_dev=0, st_file_attributes=0)
    full = SimpleNamespace(st_mode=actual.st_mode, st_size=len(payload), st_nlink=full_link_count,
                           st_ino=actual.st_ino, st_dev=actual.st_dev, st_file_attributes=0)
    full_reads = []

    def full_stat(candidate):
        if candidate == path:
            full_reads.append(candidate)
            return full
        return original_lstat(candidate)

    class CachedEntry:
        def __init__(self):
            self.path = str(path)

        def stat(self, *, follow_symlinks):
            assert follow_symlinks is False
            return cached

    class Entries:
        def __enter__(self):
            return iter([CachedEntry()])

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(Path, "lstat", full_stat)
    monkeypatch.setattr(supervision.os, "scandir", lambda directory: Entries())
    if full_link_count == 1:
        assert supervision.strict_directory_size(tmp_path) == len(payload)
    else:
        with pytest.raises(supervision.ObservationError, match="multiply linked"):
            supervision.strict_directory_size(tmp_path)
    assert full_reads == [path]


def test_monitored_file_cannot_stand_in_for_directory(tmp_path):
    path = tmp_path / "ordinary-file.bin"
    path.write_bytes(b"TEST_DATA")
    with pytest.raises(supervision.ObservationError, match="not a directory"):
        supervision.strict_directory_size(path)


@pytest.mark.parametrize("exception", [PermissionError("TEST_DATA denied"), FileNotFoundError("TEST_DATA disappeared")])
def test_directory_observation_errors_propagate_instead_of_undercounting(tmp_path, monkeypatch, exception):
    def fail_scandir(path):
        raise exception

    monkeypatch.setattr(supervision.os, "scandir", fail_scandir)
    with pytest.raises(supervision.ObservationError, match="directory observation failed"):
        supervision.strict_directory_size(tmp_path)


def test_disappearing_entry_during_size_sample_is_unknown(tmp_path, monkeypatch):
    missing = tmp_path / "TEST_DATA_disappeared.bin"
    original_lstat = Path.lstat

    def disappearing_stat(path):
        if path == missing:
            raise FileNotFoundError("TEST_DATA vanished after enumeration")
        return original_lstat(path)

    class DisappearingEntry:
        path = str(missing)

    class Entries:
        def __enter__(self):
            return iter([DisappearingEntry()])

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(supervision.os, "scandir", lambda path: Entries())
    monkeypatch.setattr(Path, "lstat", disappearing_stat)
    with pytest.raises(supervision.ObservationError, match="vanished"):
        supervision.strict_directory_size(tmp_path)


def test_private_adapter_promotes_accounting_errors_using_only_synthetic_source(tmp_path, monkeypatch):
    """TEST_DATA Python module, no accepted/native supervisor import or process."""
    source = b'''# TEST_DATA: no native imports or process APIs.
class JobError(RuntimeError):
    pass
class JobObject:
    def accounting(self):
        raise JobError("TEST_DATA accounting unavailable")
class LogError(RuntimeError):
    pass
class EventLog:
    def append(self, record):
        pass
def directory_size(path):
    return -999
'''
    path = tmp_path / "TEST_DATA_supervisor.py"
    path.write_bytes(source)
    monkeypatch.setattr(supervision, "SUPERVISOR_PATH", path)
    monkeypatch.setattr(supervision, "SUPERVISOR_SHA256", _sha(source))
    module, bound = supervision._load_supervisor()
    assert callable(module.directory_size)
    assert module._mvs_directory_samples["samples"] == 0
    with pytest.raises(supervision.ObservationError, match="accounting unavailable") as failure:
        module.JobObject().accounting()
    assert not isinstance(failure.value, module.JobError)
    assert bound == {"path": str(path), "bytes": len(source), "sha256": _sha(source)}
    assert path.read_bytes() == source


@pytest.mark.parametrize("kind", ["cleanup", "reclaim"])
def test_swallowed_late_log_failure_is_observed_and_rejects_completed_result(tmp_path, monkeypatch, kind):
    """TEST_DATA imitates only the legacy swallowed-log control flow, no native API."""
    source = b'''# TEST_DATA: a synthetic supervisor module with no imports.
class JobError(RuntimeError):
    pass
class JobObject:
    def accounting(self):
        return {}
class LogError(RuntimeError):
    pass
class EventLog:
    def append(self, record):
        raise LogError("TEST_DATA isolated late write failure")
def supervise(kind):
    result = {"status": "completed", "termination_reason": "normal_exit",
              "errors": [], "log_errors": [], "TEST_DATA_caught_LogError": False}
    try:
        EventLog().append({"kind": kind})
    except LogError:
        result["TEST_DATA_caught_LogError"] = True
    return result
'''
    path = tmp_path / "TEST_DATA_late_log_supervisor.py"
    path.write_bytes(source)
    monkeypatch.setattr(supervision, "SUPERVISOR_PATH", path)
    monkeypatch.setattr(supervision, "SUPERVISOR_SHA256", _sha(source))
    module, _ = supervision._load_supervisor()
    result = module.supervise(kind)
    assert result["status"] == "completed"  # Synthetic legacy result before adapter gate.
    assert result["TEST_DATA_caught_LogError"] is True  # Exception type is preserved.
    assert module._mvs_eventlog_failures == [{
        "kind": kind, "type": "LogError", "message": "TEST_DATA isolated late write failure",
    }]
    gated = supervision._apply_log_failure_gate(result, module._mvs_eventlog_failures)
    assert gated["status"] == "completed_with_errors"
    assert gated["termination_reason"] == "normal_exit"
    assert gated["log_errors"] == [{
        "kind": kind, "type": "LogError", "message": "TEST_DATA isolated late write failure",
        "source": "mvs_surface_event_log",
    }]
    assert path.read_bytes() == source


def test_private_supervisor_hash_is_checked_before_source_execution(tmp_path, monkeypatch):
    source = tmp_path / "TEST_DATA_not_executed.py"
    source.write_text("raise AssertionError('must not execute this source')\n", encoding="utf-8")
    monkeypatch.setattr(supervision, "SUPERVISOR_PATH", source)
    monkeypatch.setattr(supervision, "SUPERVISOR_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hash does not match"):
        supervision._load_supervisor()


def _load_cli_for_function_test(monkeypatch):
    """Import the CLI in this interpreter; preserve its environment/path effects."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    path = Path(__file__).resolve().parents[1] / "scripts/build_mvs_surface.py"
    spec = importlib.util.spec_from_file_location("TEST_DATA_mvs_surface_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_plan_hash_before_execute_plan(tmp_path, monkeypatch, capsys):
    cli = _load_cli_for_function_test(monkeypatch)
    plan_file = tmp_path / "TEST_DATA_plan.json"
    plan_file.write_text('{"TEST_DATA": true}\n', encoding="utf-8")
    target = tmp_path / "must-not-be-created"

    def forbidden_execute(*args, **kwargs):
        pytest.fail("hash rejection must happen before execution")

    monkeypatch.setattr(runner, "execute_plan", forbidden_execute)
    monkeypatch.setattr(sys, "argv", [str(cli.__file__), "run", "--plan", str(plan_file),
                                    "--plan-sha256", "0" * 64, "--output", str(target)])
    assert cli.main() == 2
    captured = capsys.readouterr()
    assert "plan size/hash mismatch" in captured.err
    assert not captured.out
    assert not target.exists()


def test_cli_plan_only_persists_mock_recipe_without_execution(tmp_path, monkeypatch, capsys):
    cli = _load_cli_for_function_test(monkeypatch)
    output = tmp_path / "TEST_DATA_plan.json"
    recipe = {"TEST_DATA": "mock plan; no scene read or native process", "native_execution": "NOT_RUN"}
    calls = []

    def fake_make_plan(**arguments):
        calls.append(arguments)
        return copy.deepcopy(recipe)

    def forbidden_execute(*args, **kwargs):
        pytest.fail("plan-only CLI must not call execute_plan")

    monkeypatch.setattr(runner, "make_plan", fake_make_plan)
    monkeypatch.setattr(runner, "execute_plan", forbidden_execute)
    argv = [str(cli.__file__), "plan"]
    expected = {}
    for option in ("scene-path", "image-root", "mask-root", "validity-root", "tool-root"):
        value = str(tmp_path / ("TEST_DATA_" + option))
        argv.extend(["--" + option, value])
        expected[option.replace("-", "_")] = value
    for option, value in (("total-seconds", 30), ("stage-seconds", 10),
                          ("memory-bytes", 128 * 1024**2), ("output-bytes", 16 * 1024**2)):
        argv.extend(["--" + option, str(value)])
        expected[option.replace("-", "_")] = value
    argv.extend(["--input-origin", "synthetic", "--plan-output", str(output)])
    expected.update(input_origin="synthetic", ignore_label=0, photo_source=None)
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 0
    assert calls == [expected]
    assert json.loads(output.read_text(encoding="utf-8")) == recipe
    captured = capsys.readouterr()
    assert not captured.err
    assert json.loads(captured.out) == {"status": "PLAN_ONLY", "native_launches": 0,
                                     "path": str(output.resolve()), "sha256": _sha(output.read_bytes())}


def test_real_plan_requires_explicit_photo_source(inputs):
    with pytest.raises(ValueError,match="photo-source"):
        _plan(inputs,input_origin="real")


def test_synthetic_plan_cannot_claim_photo_source(inputs):
    with pytest.raises(ValueError,match="photo-source"):
        _plan(inputs,photo_source="TEST_DATA")


def test_photo_validation_failure_stops_plan_before_native(inputs,monkeypatch):
    calls=[]
    def reject(*args):
        calls.append(args)
        raise ValueError("TEST_DATA source mismatch")
    monkeypatch.setattr(runner,"validate_photo_source",reject)
    with pytest.raises(ValueError,match="source mismatch"):
        _plan(inputs,input_origin="real",photo_source="TEST_DATA")
    assert len(calls)==1


def test_photo_bindings_participate_in_prelaunch_drift_checks(inputs,tmp_path,monkeypatch):
    spec=tmp_path/"TEST_DATA_source.json";spec.write_text('{"fixture":true}')
    def validate(*args):
        assert str(args[0])==str(spec)
        return {"status":"PHOTO_INPUT_IDENTITIES_BOUND","bindings":[runner.binding(spec)],
                "geometry_lineage":"NOT_VERIFIED"}
    monkeypatch.setattr(runner,"validate_photo_source",validate)
    plan=_plan(inputs,input_origin="real",photo_source=str(spec))
    assert plan['arguments']['photo_source']==str(spec)
    assert plan['photo_inputs']['status']=='PHOTO_INPUT_IDENTITIES_BOUND'
    assert runner.binding(spec) in runner.plan_files(plan)
    spec.write_text('{"fixture":"changed"}')
    target=tmp_path/"must-not-launch"
    with pytest.raises(ValueError):runner.execute_plan(plan,target)
    assert not target.exists()


def test_runtime_directory_sample_reports_real_entry_disappearance(tmp_path, monkeypatch):
    transient = tmp_path / "depth0015.dmap.tmp"
    transient.write_bytes(b"old")
    (tmp_path / "steady.bin").write_bytes(b"12345")
    original = Path.lstat
    vanished = []
    def stat_with_native_rename(path):
        if path == transient and not vanished:
            transient.rename(tmp_path / "depth0015.dmap")
            vanished.append(True)
        return original(path)
    monkeypatch.setattr(Path, "lstat", stat_with_native_rename)
    sample = supervision.runtime_directory_observation(tmp_path)
    assert 5 <= sample["bytes"] <= 8 and vanished == [True]
    assert sample["skipped_entry_count"] == 1
    assert sample["semantics"] == "NONATOMIC_OBSERVED_BYTES_WITH_DISAPPEARING_REGULAR_FILES"
