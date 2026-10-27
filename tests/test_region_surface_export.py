"""TEST_DATA only: tiny real native models; no solver, GPU, or source repair."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from integrations.video_masks import region_surface_export as export

pc = pytest.importorskip("pycolmap")


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _fixture(tmp_path, defect=None):
    source = tmp_path / "regional"
    geometry = source / "geometry"
    binary = geometry / "models/0/binary"
    binary.mkdir(parents=True)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    staged = source / "input/source_sfm/images"
    staged.mkdir(parents=True)
    model = pc.Reconstruction()
    model.add_camera_with_trivial_rig(pc.Camera(camera_id=7, model="PINHOLE", width=12, height=8,
                                               params=np.array([10., 11., 6., 4.])))
    model.add_camera_with_trivial_rig(pc.Camera(camera_id=13, model="SIMPLE_RADIAL", width=12, height=8,
                                               params=np.array([10., 6., 4., .01])))
    timeline, frames, refs = [], [], {}
    for position, (iid, index, decode) in enumerate(zip((11, 23, 42), (10, 20, 30), (5, 11, 20))):
        suffix = ".jpg" if position == 1 else ".png"
        image_path = canonical / f"source_{iid}{suffix}"
        pixels = np.arange(12 * 8 * 3, dtype=np.uint8).reshape(8, 12, 3) + position
        Image.fromarray(pixels).save(image_path)
        mask_path = canonical / f"mask_{iid}.png"
        Image.new("L", (12, 8), 255).save(mask_path)
        original_name = f"images/staged_{iid}{suffix}"
        stage_path = staged / Path(original_name).name
        stage_path.write_bytes(image_path.read_bytes())
        camera_id = 13 if position == 2 else 7
        angle = position * .1
        pose = pc.Rigid3d(pc.Rotation3d(np.array([0., 0., np.sin(angle / 2), np.cos(angle / 2)])),
                         np.array([position * .125, .25, 0.]))
        points = pc.Point2DList([pc.Point2D(xy=np.array(xy, dtype=float)) for xy in ((2, 2), (4, 3), (5, 5))])
        image = pc.Image(image_id=iid, camera_id=camera_id, name=original_name, points2D=points)
        model.add_image_with_trivial_frame(image, pose)
        timeline.append(dict(input_index=index, decode_index=decode, image_path=str(image_path),
                             image_sha256=export._sha(image_path), mask_path=str(mask_path),
                             mask_sha256=export._sha(mask_path), width=12, height=8))
        frames.append(dict(image_id=iid, input_index=index, decode_index=decode,
                           name=original_name, image_path=str(stage_path)))
        for path in (image_path, stage_path, mask_path):
            refs[str(path)] = export._sha(path)
    good = pc.Track()
    for iid in (11, 23, 42):
        good.add_element(iid, 0)
    good_id = int(model.add_point3D(np.array([0., 0., 5.]), good, np.array([15, 30, 45], dtype=np.uint8)))
    suspect = pc.Track()
    suspect.add_element(11, 1)
    if defect != "one_view":
        suspect.add_element(23, 1)
    if defect in ("duplicate", "duplicate_broken"):
        suspect.add_element(11, 2)
    bad_id = int(model.add_point3D(np.array([.2, .1, 4.]), suspect, np.array([60, 70, 80], dtype=np.uint8)))
    model.points3D[good_id].error = .25
    model.points3D[bad_id].error = .5
    if defect == "unregistered":
        model.points3D[bad_id].track.add_element(999, 0)
    if defect in ("broken", "duplicate_broken"):
        model.images[11].points2D[1].point3D_id = pc.Point2D().point3D_id
    model.write_binary(str(binary))
    input_path = source / "input/PLAN.json"
    _save(input_path, dict(schema="glove-region-plan-v1", timeline=timeline, source_bindings={}))
    refs[str(input_path)] = export._sha(input_path)
    _save(geometry / "PLAN.json", dict(frames=frames, source_bindings=refs))
    _save(geometry / "RESULT.json", dict(schema="glove-region-geometry-v1", source_unchanged=True,
         source_before=refs, source_after=refs, models=[dict(model_id=0, binary_path=str(binary),
         registered_views=3, registered_input_indices=[10, 20, 30], points=2)]))
    return source, binary, good_id, bad_id


def _all_hashes(path):
    return {str(item): export._sha(item) for item in path.rglob("*") if item.is_file()}


def test_export_mixed_jpeg_png_preserves_native_fields_and_pixel_identity(tmp_path):
    source, binary, _, _ = _fixture(tmp_path)
    before = _all_hashes(tmp_path)
    result = export.export(source, tmp_path / "export")
    assert result["status"] == "EXPORTED_NOT_GEOMETRY_ACCEPTED"
    assert result["geometry_unchanged"] and result["source_geometry_unchanged"] and result["source_unchanged"]
    assert result["excluded_point_ids"] == [] and result["explicit_point_exclusion"] is False
    assert result["registered_views"] == 3 and result["points"] == 2
    assert result["geometry_accepted"] is False and result["quality_status"] == "NOT_EVALUATED"
    assert result["native_solver_executed"] is False and result["surface_status"] == "NOT_RUN"
    for kind in ("binary", "text"):
        assert result["native_readback"][kind]["status"] == "PASS"
    original = pc.Reconstruction(str(binary))
    for kind in ("binary", "text"):
        actual = pc.Reconstruction(result[f"{kind}_path"])
        assert set(actual.images) == {11, 23, 42}
        assert set(actual.cameras) == {7, 13}
        assert set(actual.points3D) == set(original.points3D)
        for iid in original.images:
            assert actual.images[iid].camera_id == original.images[iid].camera_id
            assert actual.images[iid].frame_id == original.images[iid].frame_id
            np.testing.assert_array_equal(actual.images[iid].cam_from_world().translation,
                                          original.images[iid].cam_from_world().translation)
    frame_map = export._read(Path(result["frame_map_path"]))
    assert frame_map["schema"] == "region-surface-frame-map-v1" and frame_map["frames"] == result["frames"]
    assert [frame["input_index"] for frame in result["frames"]] == [10, 20, 30]
    assert [frame["decode_index"] for frame in result["frames"]] == [5, 11, 20]
    for frame in result["frames"]:
        assert frame["export_image_id"] == frame["image_id"]
        assert frame["source_mask_path"] == frame["canonical_mask_path"]
        assert frame["source_mask_sha256"] == frame["canonical_mask_sha256"]
        assert frame["exported_name"] == f"image_{frame['image_id']:08d}.png"
        assert frame["pixel_equal"] and frame["decoded_pixel_sha256"] == frame["exported_pixel_sha256"]
        with Image.open(frame["source_image_path"]) as image:
            expected_pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        assert frame["decoded_pixel_sha256"] == expected_pixel_hash
        if frame["source_image_format"] == "PNG":
            assert frame["source_image_sha256"] == frame["exported_png_sha256"]
            assert frame["conversion"] == "PNG_BYTE_COPY"
        else:
            assert frame["source_image_sha256"] != frame["exported_png_sha256"]
            assert frame["conversion"] == "JPEG_DECODE_TO_LOSSLESS_PNG"
    assert all(export._sha(Path(path)) == digest for path, digest in before.items())
    assert all(export._sha(Path(item["path"])) == item["sha256"] for item in result["source_bindings"] + result["outputs"])
    output_names = {Path(item["path"]).name for item in result["outputs"]}
    assert {"FRAME_MAP.json", "DERIVATION.json"} <= output_names


@pytest.mark.parametrize("defect", ["duplicate", "one_view"])
def test_default_refuses_invalid_tracks_without_creating_output(tmp_path, defect):
    source, _, _, _ = _fixture(tmp_path, defect)
    before = _all_hashes(tmp_path)
    with pytest.raises(ValueError, match="explicit whole-point exclusion"):
        export.export(source, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    assert _all_hashes(tmp_path) == before


@pytest.mark.parametrize("allowlist", [[], [999], [1, 2], [2, 2], [True], "2"])
def test_whole_point_allowlist_must_be_exact_and_typed(tmp_path, allowlist):
    source, _, _, _ = _fixture(tmp_path, "duplicate")
    with pytest.raises(ValueError):
        export.export(source, tmp_path / "refused", expected_excluded_point_ids=allowlist)
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("defect", ["duplicate", "one_view"])
def test_exact_allowlist_derives_copy_preserves_source_and_every_retained_field(tmp_path, defect):
    source, binary, good_id, bad_id = _fixture(tmp_path, defect)
    before = _all_hashes(tmp_path)
    result = export.export(source, tmp_path / "derived", expected_excluded_point_ids=[bad_id])
    assert result["source_geometry_unchanged"] is False and result["geometry_unchanged"] is True
    assert result["explicit_point_exclusion"] is True and result["excluded_point_ids"] == [bad_id]
    assert result["points"] == 1
    derivation = export._read(Path(result["derivation_path"]))
    assert derivation["status"] == "EXPLICIT_WHOLE_POINT_EXCLUSION"
    assert derivation["before"]["points"] == 2 and derivation["after"]["points"] == 1
    assert derivation["removed_points"][0]["point3D_id"] == bad_id
    assert derivation["removed_points"][0]["xyz"] == [.2, .1, 4.]
    assert derivation["removed_points"][0]["color_rgb"] == [60, 70, 80]
    assert derivation["removed_points"][0]["error"] == .5
    assert len(derivation["removed_points"][0]["track"]) == (3 if defect == "duplicate" else 1)
    source_model = pc.Reconstruction(str(binary))
    actual = pc.Reconstruction(result["binary_path"])
    assert set(source_model.points3D) == {good_id, bad_id} and set(actual.points3D) == {good_id}
    for iid in source_model.images:
        for old, new in zip(source_model.images[iid].points2D, actual.images[iid].points2D):
            np.testing.assert_array_equal(old.xy, new.xy)
            assert new.point3D_id == (pc.Point2D().point3D_id if old.point3D_id == bad_id else old.point3D_id)
    assert all(export._sha(Path(path)) == digest for path, digest in before.items())


@pytest.mark.parametrize("defect", ["unregistered", "broken", "duplicate_broken"])
def test_exclusion_never_repairs_unregistered_or_broken_reciprocity(tmp_path, defect):
    source, _, _, bad_id = _fixture(tmp_path, defect)
    with pytest.raises(ValueError, match="unregistered view|broken track reciprocity"):
        export.export(source, tmp_path / "refused", expected_excluded_point_ids=[bad_id])
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("fault", ["image", "mask", "input_plan", "frame_name", "decode", "missing_frame",
                                   "duplicate_image_id", "count", "registered_identity", "source_contract"])
def test_refuses_identity_or_source_binding_faults(tmp_path, fault):
    source, _, _, _ = _fixture(tmp_path)
    geometry_path = source / "geometry/PLAN.json"
    geometry = export._read(geometry_path)
    result_path = source / "geometry/RESULT.json"
    result = export._read(result_path)
    input_path = source / "input/PLAN.json"
    input_plan = export._read(input_path)
    if fault == "image":
        Path(geometry["frames"][0]["image_path"]).write_bytes(b"changed source image")
    elif fault == "mask":
        Image.new("L", (12, 8), 0).save(input_plan["timeline"][0]["mask_path"])
    elif fault == "input_plan":
        input_plan["unbound_change"] = True
        _save(input_path, input_plan)
    elif fault == "frame_name":
        geometry["frames"][0]["name"] = "images/wrong.png"
    elif fault == "decode":
        geometry["frames"][0]["decode_index"] = 6
    elif fault == "missing_frame":
        geometry["frames"].pop()
    elif fault == "duplicate_image_id":
        geometry["frames"][1]["image_id"] = geometry["frames"][0]["image_id"]
    elif fault == "count":
        result["models"][0]["points"] = 3
    elif fault == "registered_identity":
        result["models"][0]["registered_input_indices"] = [10, 20, 31]
    elif fault == "source_contract":
        result["source_after"] = {}
    _save(geometry_path, geometry)
    _save(result_path, result)
    with pytest.raises(ValueError):
        export.export(source, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("field", ["camera", "pose", "rig_id", "frame_id", "xyz", "color", "error", "track", "point2D", "point_id"])
def test_native_field_comparison_detects_mutations(field, tmp_path):
    _, binary, good_id, _ = _fixture(tmp_path)
    original = export._describe(pc.Reconstruction(str(binary)))
    changed = deepcopy(original)
    if field == "camera": changed["cameras"]["7"]["params"][0] += .1
    elif field == "pose": changed["images"]["11"]["translation"][0] += .1
    elif field == "rig_id": next(iter(changed["rigs"].values()))["rig_id"] += 1
    elif field == "frame_id": changed["images"]["11"]["frame_id"] += 1
    elif field == "xyz": changed["points3D"][str(good_id)]["xyz"][0] += .1
    elif field == "color": changed["points3D"][str(good_id)]["color"][0] += 1
    elif field == "error": changed["points3D"][str(good_id)]["error"] += .1
    elif field == "track": changed["points3D"][str(good_id)]["track"][0][0] += 1
    elif field == "point2D": changed["images"]["11"]["points2D"][0][0] += .1
    elif field == "point_id": changed["images"]["42"]["point3D_ids"][2] -= 1
    with pytest.raises(ValueError, match="native"):
        export._equal(original, changed, tolerance=1e-12)


@pytest.mark.parametrize("model_id", [True, -1, "0", 1])
def test_model_selection_and_fresh_disjoint_output(tmp_path, model_id):
    source, _, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError):
        export.export(source, tmp_path / "refused", model_id=model_id)
    with pytest.raises(ValueError, match="fresh disjoint"):
        export.export(source, source / "new_output")
    with pytest.raises(ValueError, match="fresh disjoint"):
        export.export(source, tmp_path)


@pytest.mark.parametrize("name", ["../escape.png", "C:/absolute.png", "/absolute.png", "images\\file.png", "image.png\n"])
def test_unsafe_native_image_names_are_rejected(name):
    with pytest.raises(ValueError, match="unsafe"):
        export._safe_name(name)
