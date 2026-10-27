"""Pure contracts for adapting native descriptions, without importing pycolmap.

The fixture follows ``export.describe_model`` (including native data-id records),
but its camera projections are specified independently by elementary arithmetic.
No native files, backend, subprocess, or reconstruction run is needed here.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
import importlib
import math

import numpy as np
import pytest

from kozlov_lab.stereo.baseline.camera import Camera
from kozlov_lab.stereo.model import PointSet3D


UNMATCHED = 2**64 - 1
LARGE_POINT_ID = 2**63 + 17
POINT_IDS = [5, 91, LARGE_POINT_ID]
XYZ = [[2.0, -1.0, 6.0], [-1.0, 2.0, 8.0], [2.0, -1.0, 6.0]]
PIXELS = {
    11: [
        [311.25 + 800.0 * 3.0 / 9.0, 247.75 + 600.0 / 9.0],
        [311.25, 247.75 + 600.0 * 4.0 / 11.0],
        [311.25 + 800.0 * 3.0 / 9.0, 247.75 + 600.0 / 9.0],
    ],
    42: [[242.0, 495.625], [49.5, 233.125], [242.0, 495.625]],
}


def _sensor(camera_id):
    return {"type": "SensorType.CAMERA", "id": camera_id}


def _data_id(camera_id, image_id):
    return {"sensor_id": _sensor(camera_id), "id": image_id}


def _fixture_description():
    """Two trivial rigs; distinct models, nonidentity pose and noncontiguous IDs."""
    half_root = math.sqrt(0.5)
    description = {
        "num_cameras": 2,
        "num_images": 2,
        "num_points3D": 3,
        "num_frames": 2,
        "num_rigs": 2,
        "num_reg_images": 2,
        "registered_image_ids": [11, 42],
        "cameras": {
            "7": {
                "camera_id": 7, "model_name": "PINHOLE", "width": 1280,
                "height": 960, "params": [800.0, 600.0, 311.25, 247.75],
                "has_prior_focal_length": False,
            },
            "19": {
                "camera_id": 19, "model_name": "SIMPLE_PINHOLE", "width": 1024,
                "height": 720, "params": [700.0, 329.5, 233.125],
                "has_prior_focal_length": True,
            },
        },
        "images": {
            "42": {
                "name": "offset rotated.png", "camera_id": 19, "image_id": 42,
                "frame_id": 205, "quaternion_xyzw": [0.0, 0.0, half_root, half_root],
                "translation": [-2.0, 1.0, 2.0], "camera_center": [-1.0, -2.0, -2.0],
                "points2D": [PIXELS[42][0], PIXELS[42][2], PIXELS[42][1], [89.25, 10.5]],
                "point3D_ids": [5, LARGE_POINT_ID, 91, UNMATCHED],
            },
            "11": {
                "name": "unequal focal.png", "camera_id": 7, "image_id": 11,
                "frame_id": 101, "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "translation": [1.0, 2.0, 3.0], "camera_center": [-1.0, -2.0, -3.0],
                "points2D": [PIXELS[11][1], [12.125, 77.875], PIXELS[11][0], PIXELS[11][2]],
                "point3D_ids": [91, UNMATCHED, 5, LARGE_POINT_ID],
            },
        },
        # Insertion order deliberately differs from numerical ID order.
        "points3D": {
            str(LARGE_POINT_ID): {
                "xyz": XYZ[2], "track": [[42, 1], [11, 3]], "color": [30, 40, 50],
            },
            "91": {"xyz": XYZ[1], "track": [[11, 0], [42, 2]], "color": [60, 70, 80]},
            "5": {"xyz": XYZ[0], "track": [[42, 0], [11, 2]], "color": [10, 20, 30]},
        },
        "rigs": {},
        "frames": {},
    }
    for camera_id, rig_id, frame_id, image_id in [(7, 3, 101, 11), (19, 8, 205, 42)]:
        sensor = _sensor(camera_id)
        description["rigs"][str(rig_id)] = {
            "rig_id": rig_id, "sensor_ids": [sensor], "ref_sensor_id": sensor,
            "num_sensors": 1, "sensor_from_rig": {str(sensor): None},
        }
        image = description["images"][str(image_id)]
        description["frames"][str(frame_id)] = {
            "frame_id": frame_id, "rig_id": rig_id,
            "data_ids": [_data_id(camera_id, image_id)],
            "image_ids": [_data_id(camera_id, image_id)],
            "rig_from_world": {
                "rotation": {"quat": image["quaternion_xyzw"]},
                "translation": image["translation"],
            },
        }
    # Tests may mutate arbitrary nested values without changing module constants.
    return deepcopy(description)


@pytest.fixture
def description():
    return _fixture_description()


@pytest.fixture
def adapter(monkeypatch):
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def without_native(name, *args, **kwargs):
        if name == "pycolmap" or name.startswith("pycolmap."):
            raise AssertionError("pure description adaptation must not import pycolmap")
        return original_import(name, *args, **kwargs)

    def without_dynamic_native(name, *args, **kwargs):
        if name == "pycolmap" or name.startswith("pycolmap."):
            raise AssertionError("pure description adaptation must not import pycolmap")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_native)
    monkeypatch.setattr(importlib, "import_module", without_dynamic_native)
    return importlib.import_module("kozlov_lab.reconstruction.to_kozlov")


def _replace(description, path, value):
    parent = description
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = deepcopy(value)


def test_module_import_and_adaptation_are_native_backend_free(adapter, description):
    reloaded = importlib.reload(adapter)
    assert issubclass(reloaded.AdapterError, ValueError)
    assert isinstance(reloaded.adapt_description(description), reloaded.ResearchScene)


def test_native_ids_coordinates_and_exact_duplicate_groups_are_preserved(adapter, description):
    original = deepcopy(description)
    scene = adapter.adapt_description(description)
    assert description == original
    assert scene.native_description == original
    assert scene.native_description is not description
    assert isinstance(scene.point_ids, np.ndarray)
    assert scene.point_ids.dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(scene.point_ids, np.array(POINT_IDS, dtype=np.uint64))
    np.testing.assert_array_equal(scene.native_xyz, XYZ)
    assert isinstance(scene.body, PointSet3D)
    np.testing.assert_array_equal(scene.body.as_array(), XYZ[:2])
    assert scene.point_id_to_body_row == {5: 0, 91: 1, LARGE_POINT_ID: 0}
    assert scene.duplicate_groups == [[5, LARGE_POINT_ID]]


def test_report_declares_identity_transform_and_unknown_scale(adapter, description):
    report = adapter.adapt_description(description).report
    assert report["status"] == "ADAPTED_PINHOLE_NATIVE"
    assert report["scale_status"] == "unknown"
    assert report["length_unit"] is None
    np.testing.assert_array_equal(report["native_to_research"], np.eye(4))
    np.testing.assert_array_equal(report["research_to_native"], np.eye(4))


@pytest.mark.parametrize("image_id", [11, 42])
def test_manual_native_pixel_projection_has_no_half_pixel_shift(adapter, description, image_id):
    scene = adapter.adapt_description(description)
    assert set(scene.cameras) == {11, 42}
    camera = scene.cameras[image_id]
    assert isinstance(camera, Camera)
    # Values above were calculated from Xc = R Xw + t and u = fx Xc/Zc + cx.
    np.testing.assert_allclose(camera.project(scene.native_xyz), PIXELS[image_id], atol=1e-10, rtol=0)
    image = scene.native_description["images"][str(image_id)]
    for feature_index, point_id in enumerate(image["point3D_ids"]):
        if point_id != UNMATCHED:
            expected = PIXELS[image_id][POINT_IDS.index(point_id)]
            np.testing.assert_array_equal(image["points2D"][feature_index], expected)
    assert image["point3D_ids"].count(UNMATCHED) == 1
    assert image["points2D"] == description["images"][str(image_id)]["points2D"]


def test_rotation_translation_intrinsics_and_centres_use_world_to_camera_convention(adapter, description):
    scene = adapter.adapt_description(description)
    camera = scene.cameras[42]
    expected_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected_k = np.array([[700.0, 0.0, 329.5], [0.0, 700.0, 233.125], [0.0, 0.0, 1.0]])
    expected_t = np.array([-2.0, 1.0, 2.0])
    np.testing.assert_allclose(camera.rotation, expected_rotation, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(camera.intrinsics, expected_k)
    np.testing.assert_array_equal(camera.translation, expected_t)
    np.testing.assert_allclose(camera.projection, expected_k @ np.column_stack((expected_rotation, expected_t)), atol=1e-10, rtol=0)
    np.testing.assert_allclose(camera.centre, [-1.0, -2.0, -2.0], atol=1e-10, rtol=0)
    for camera in scene.cameras.values():
        np.testing.assert_allclose(camera.rotation.T @ camera.rotation, np.eye(3), atol=1e-12, rtol=0)
        assert np.linalg.det(camera.rotation) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(scene.cameras[11].intrinsics, [[800.0, 0.0, 311.25], [0.0, 600.0, 247.75], [0.0, 0.0, 1.0]])


def test_mutating_input_after_adaptation_does_not_change_scene(adapter, description):
    scene = adapter.adapt_description(description)
    original = deepcopy(description)
    description["points3D"]["5"]["xyz"][0] = 999.0
    description["images"]["11"]["points2D"][1][0] = 555.0
    description["cameras"]["7"]["params"][0] = 12.0
    description["frames"]["101"]["data_ids"][0]["id"] = 999
    assert scene.native_description == original
    np.testing.assert_array_equal(scene.native_xyz, XYZ)
    np.testing.assert_array_equal(scene.body.as_array(), XYZ[:2])
    assert scene.cameras[11].intrinsics[0, 0] == 800.0


def test_scene_storage_does_not_alias_input_or_other_coordinate_representations(adapter, description):
    original = deepcopy(description)
    scene = adapter.adapt_description(description)
    assert not np.shares_memory(scene.body.points, scene.native_xyz)
    scene.native_description["points3D"]["5"]["xyz"][0] = 555.0
    scene.native_description["cameras"]["7"]["params"][0] = 123.0
    np.testing.assert_array_equal(scene.native_xyz, XYZ)
    assert scene.cameras[11].intrinsics[0, 0] == 800.0
    scene.native_xyz[0, 0] = 777.0
    np.testing.assert_array_equal(scene.body.as_array(), XYZ[:2])
    scene.body.points[0, 1] = 888.0
    assert scene.native_xyz[0, 1] == -1.0
    assert description == original


def test_nearby_coordinates_are_not_rounded_into_duplicates(adapter, description):
    description["points3D"][str(LARGE_POINT_ID)]["xyz"][0] += 1e-12
    scene = adapter.adapt_description(description)
    assert scene.body.n == 3
    assert scene.duplicate_groups == []
    assert scene.point_id_to_body_row == {5: 0, 91: 1, LARGE_POINT_ID: 2}


def test_smallest_usable_scene_keeps_one_point_seen_in_two_views(adapter, description):
    description["points3D"] = {"5": description["points3D"]["5"]}
    description["num_points3D"] = 1
    for image in description["images"].values():
        image["point3D_ids"] = [value if value == 5 else UNMATCHED for value in image["point3D_ids"]]
    scene = adapter.adapt_description(description)
    assert scene.body.n == 1
    np.testing.assert_array_equal(scene.point_ids, np.array([5], dtype=np.uint64))


def test_quaternion_sign_does_not_change_projection(adapter, description):
    description["images"]["42"]["quaternion_xyzw"] = [-v for v in description["images"]["42"]["quaternion_xyzw"]]
    scene = adapter.adapt_description(description)
    np.testing.assert_allclose(scene.cameras[42].project(scene.native_xyz), PIXELS[42], atol=1e-10, rtol=0)


def test_largest_non_sentinel_uint64_point_id_is_preserved(adapter, description):
    replacement = UNMATCHED - 1
    description["points3D"][str(replacement)] = description["points3D"].pop(str(LARGE_POINT_ID))
    for image in description["images"].values():
        image["point3D_ids"] = [replacement if v == LARGE_POINT_ID else v for v in image["point3D_ids"]]
    scene = adapter.adapt_description(description)
    assert int(scene.point_ids[-1]) == replacement
    assert scene.point_id_to_body_row[replacement] == 0


def test_adaptation_never_calls_quadratic_body_diameter(adapter, description, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("adaptation must not allocate pairwise point distances")

    monkeypatch.setattr(PointSet3D, "diameter", forbidden)
    assert adapter.adapt_description(description).body.n == 2


@pytest.mark.parametrize("path,value", [
    pytest.param(("points3D", "5", "xyz"), [float("nan"), -1.0, 6.0], id="nan-xyz"),
    pytest.param(("points3D", "5", "xyz"), [2.0, float("inf"), 6.0], id="infinite-xyz"),
    pytest.param(("points3D", "5", "xyz"), [True, -1.0, 6.0], id="bool-xyz"),
    pytest.param(("points3D", "5", "xyz"), ["2", -1.0, 6.0], id="string-xyz"),
    pytest.param(("points3D", "5", "xyz"), [[2.0], [-1.0, 6.0]], id="ragged-xyz"),
    pytest.param(("points3D", "5", "xyz"), [2.0, -1.0], id="short-xyz"),
    pytest.param(("cameras", "7", "params"), [float("nan"), 600.0, 311.25, 247.75], id="nan-intrinsics"),
    pytest.param(("cameras", "7", "params"), [800.0, 600.0, float("inf"), 247.75], id="infinite-principal-point"),
    pytest.param(("cameras", "7", "params"), [True, 600.0, 311.25, 247.75], id="bool-intrinsics"),
    pytest.param(("cameras", "7", "params"), [[800.0, 600.0], [311.25, 247.75]], id="nested-intrinsics"),
    pytest.param(("cameras", "19", "params"), [700.0, 329.5, 233.125, 0.0], id="extra-simple-pinhole-parameter"),
    pytest.param(("images", "11", "translation"), [1.0, 2.0, float("inf")], id="infinite-translation"),
    pytest.param(("images", "11", "translation"), [True, 2.0, 3.0], id="bool-translation"),
    pytest.param(("images", "11", "translation"), [[1.0], [2.0], [3.0]], id="nested-translation"),
    pytest.param(("images", "11", "quaternion_xyzw"), [0.0, 0.0, float("nan"), 1.0], id="nan-quaternion"),
    pytest.param(("images", "11", "quaternion_xyzw"), [False, 0.0, 0.0, 1.0], id="bool-quaternion"),
    pytest.param(("images", "11", "quaternion_xyzw"), [0.0, 0.0, 1.0], id="short-quaternion"),
    pytest.param(("images", "11", "quaternion_xyzw"), [0.0, 0.0, 0.0, 0.0], id="zero-quaternion"),
    pytest.param(("images", "11", "quaternion_xyzw"), [0.0, 0.0, 0.0, 2.0], id="quaternion-must-not-be-normalized"),
    pytest.param(("images", "11", "quaternion_xyzw"), [0.0, 0.0, 0.0, 1.0 + 1e-8], id="quaternion-outside-unit-tolerance"),
    pytest.param(("images", "11", "camera_center"), [float("nan"), -2.0, -3.0], id="nan-camera-centre"),
    pytest.param(("images", "11", "camera_center"), [-1.0, True, -3.0], id="bool-camera-centre"),
    pytest.param(("images", "11", "points2D", 1), [float("inf"), 77.875], id="unmatched-feature-must-be-finite"),
    pytest.param(("images", "11", "points2D", 0), [True, 247.75], id="bool-feature"),
    pytest.param(("images", "11", "points2D", 0), [311.25], id="ragged-feature"),
])
def test_invalid_numeric_arrays_fail_closed(adapter, description, path, value):
    _replace(description, path, value)
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("camera_id,parameter", [("7", 0), ("7", 1), ("19", 0)])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_focal_lengths_must_be_positive(adapter, description, camera_id, parameter, value):
    description["cameras"][camera_id]["params"][parameter] = value
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("model_name,params", [
    ("SIMPLE_RADIAL", [800.0, 311.25, 247.75, 0.0]),
    ("RADIAL", [800.0, 311.25, 247.75, 0.0, 0.0]),
    ("OPENCV", [800.0, 600.0, 311.25, 247.75, 0.0, 0.0, 0.0, 0.0]),
    ("UNKNOWN", [800.0, 600.0, 311.25, 247.75]),
])
def test_unsupported_models_are_rejected_even_with_zero_distortion(adapter, description, model_name, params):
    description["cameras"]["7"].update(model_name=model_name, params=params)
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_unsupported_unreferenced_camera_is_also_rejected(adapter, description):
    camera = deepcopy(description["cameras"]["7"])
    camera.update(camera_id=99, model_name="SIMPLE_RADIAL", params=[800.0, 311.25, 247.75, 0.0])
    description["cameras"]["99"] = camera
    description["num_cameras"] = 3
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("field", ["width", "height"])
@pytest.mark.parametrize("value", [True, 0, 640.0])
def test_image_dimensions_are_positive_integers(adapter, description, field, value):
    description["cameras"]["7"][field] = value
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("field", ["num_cameras", "num_images", "num_points3D", "num_frames", "num_rigs", "num_reg_images"])
@pytest.mark.parametrize("value", [True, 2.0, "2", -1])
def test_count_fields_require_nonboolean_integers(adapter, description, field, value):
    description[field] = value
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("field", ["num_cameras", "num_images", "num_points3D", "num_frames", "num_rigs", "num_reg_images"])
def test_declared_counts_must_match_collections(adapter, description, field):
    description[field] += 1
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("collection,key", [("cameras", "7"), ("images", "11"), ("points3D", "5"), ("frames", "101"), ("rigs", "3")])
@pytest.mark.parametrize("bad_key", [0, "0", "-1", "01"])
def test_identity_map_keys_must_be_positive_canonical_decimal_strings(adapter, description, collection, key, bad_key):
    description[collection][bad_key] = description[collection].pop(key)
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("value", [True, 5.0, -1, 0, UNMATCHED + 1])
def test_feature_point_ids_are_exact_native_ids_or_uint64_sentinel(adapter, description, value):
    description["images"]["11"]["point3D_ids"][1] = value
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("value", [UNMATCHED, UNMATCHED + 1])
def test_point_identity_cannot_be_sentinel_or_overflow_uint64(adapter, description, value):
    description["points3D"][str(value)] = description["points3D"].pop(str(LARGE_POINT_ID))
    for image in description["images"].values():
        image["point3D_ids"] = [value if v == LARGE_POINT_ID else v for v in image["point3D_ids"]]
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("path,value", [
    pytest.param(("images", "11", "image_id"), 12, id="image-record-id-mismatch"),
    pytest.param(("cameras", "7", "camera_id"), 8, id="camera-record-id-mismatch"),
    pytest.param(("frames", "101", "frame_id"), 102, id="frame-record-id-mismatch"),
    pytest.param(("rigs", "3", "rig_id"), 4, id="rig-record-id-mismatch"),
    pytest.param(("images", "11", "camera_id"), 999, id="unknown-camera"),
    pytest.param(("images", "11", "camera_id"), True, id="bool-camera-id"),
    pytest.param(("images", "11", "frame_id"), 999, id="unknown-frame"),
    pytest.param(("images", "11", "frame_id"), 101.0, id="float-frame-id"),
    pytest.param(("frames", "101", "rig_id"), 999, id="unknown-rig"),
    pytest.param(("frames", "101", "image_ids", 0, "id"), 42, id="frame-image-link-mismatch"),
    pytest.param(("frames", "101", "image_ids", 0, "sensor_id", "id"), 19, id="frame-camera-link-mismatch"),
    pytest.param(("frames", "101", "data_ids", 0, "id"), 42, id="frame-data-link-mismatch"),
    pytest.param(("rigs", "3", "sensor_ids", 0, "id"), 19, id="rig-sensor-link-mismatch"),
    pytest.param(("rigs", "3", "ref_sensor_id", "id"), 999, id="missing-reference-sensor"),
    pytest.param(("rigs", "3", "num_sensors"), 2, id="sensor-count-mismatch"),
    pytest.param(("rigs", "3", "num_sensors"), True, id="boolean-sensor-count"),
])
def test_identity_frame_and_rig_links_are_checked(adapter, description, path, value):
    _replace(description, path, value)
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("registered", [[11], [11, 11], [11, 42, 999], [True, 42]])
def test_registered_ids_must_be_exactly_all_described_images(adapter, description, registered):
    description["registered_image_ids"] = registered
    description["num_reg_images"] = len(registered)
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_scene_without_points_is_rejected(adapter, description):
    description["points3D"] = {}
    description["num_points3D"] = 0
    for image in description["images"].values():
        image["point3D_ids"] = [UNMATCHED] * len(image["points2D"])
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_forward_feature_link_requires_reciprocal_track(adapter, description):
    # Existing valid track is unchanged; an additional feature claims the point.
    description["images"]["11"]["point3D_ids"][1] = 5
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_reverse_track_link_requires_matching_feature_point_id(adapter, description):
    description["images"]["11"]["point3D_ids"][2] = UNMATCHED
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("element", [[999, 2], [11, 100], [11, -1], [11, True], [11, 2.0], [True, 2], [11, 0], [11]])
def test_track_observations_require_existing_exact_image_and_feature_indices(adapter, description, element):
    description["points3D"]["5"]["track"][1] = element
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_duplicate_track_observation_is_rejected(adapter, description):
    description["points3D"]["5"]["track"].append([11, 2])
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_two_observations_in_one_view_cannot_repeat_a_point(adapter, description):
    description["images"]["11"]["point3D_ids"][1] = 5
    description["points3D"]["5"]["track"].append([11, 1])
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_point_requires_at_least_two_distinct_views(adapter, description):
    description["points3D"]["5"]["track"] = [[11, 2]]
    description["images"]["42"]["point3D_ids"][0] = UNMATCHED
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_image_coordinate_and_point_id_lengths_must_agree(adapter, description):
    description["images"]["11"]["point3D_ids"].pop()
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


class _UnreadableSizedDict(dict):
    """Advertise oversized input without allocating or permitting traversal."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("oversized collections must be rejected before traversal or copy")

    __iter__ = keys = values = items = __deepcopy__ = _forbidden


class _UnreadableSizedList(list):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("oversized feature arrays must be rejected before traversal or copy")

    __iter__ = __getitem__ = __deepcopy__ = _forbidden


@pytest.mark.parametrize("collection,count_field,size", [("points3D", "num_points3D", 100001), ("images", "num_images", 257)])
def test_fixed_collection_limits_reject_before_traversal(adapter, description, collection, count_field, size):
    description[collection] = _UnreadableSizedDict(size)
    description[count_field] = size
    if collection == "images":
        description["registered_image_ids"] = list(range(1, size + 1))
        description["num_reg_images"] = size
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_aggregate_feature_limit_rejects_before_array_traversal(adapter, description):
    for image_id, size in [("11", 500001), ("42", 500000)]:
        description["images"][image_id]["points2D"] = _UnreadableSizedList(size)
        description["images"][image_id]["point3D_ids"] = _UnreadableSizedList(size)
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


def test_finite_camera_inputs_cannot_overflow_projection_for_untracked_image(adapter, description):
    """An empty registered image still needs a finite camera projection matrix."""
    camera_id, image_id, frame_id, rig_id = 50, 73, 309, 12
    description["cameras"][str(camera_id)] = {
        "camera_id": camera_id, "model_name": "SIMPLE_PINHOLE", "width": 640,
        "height": 480, "params": [700.0, 0.0, 0.0], "has_prior_focal_length": False,
    }
    description["images"][str(image_id)] = {
        "name": "registered without tracks.png", "camera_id": camera_id,
        "image_id": image_id, "frame_id": frame_id,
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "translation": [2.0, 0.0, 1.0], "camera_center": [-2.0, 0.0, -1.0],
        "points2D": [], "point3D_ids": [],
    }
    sensor = _sensor(camera_id)
    description["rigs"][str(rig_id)] = {
        "rig_id": rig_id, "sensor_ids": [sensor], "ref_sensor_id": sensor,
        "num_sensors": 1, "sensor_from_rig": {str(sensor): None},
    }
    description["frames"][str(frame_id)] = {
        "frame_id": frame_id, "rig_id": rig_id,
        "data_ids": [_data_id(camera_id, image_id)],
        "image_ids": [_data_id(camera_id, image_id)],
        "rig_from_world": {
            "rotation": {"quat": [0.0, 0.0, 0.0, 1.0]},
            "translation": [2.0, 0.0, 1.0],
        },
    }
    description["registered_image_ids"].append(image_id)
    for field in ["num_cameras", "num_images", "num_frames", "num_rigs", "num_reg_images"]:
        description[field] += 1
    # A positive control prevents unrelated empty-feature validation from hiding the bug.
    assert image_id in adapter.adapt_description(description).cameras
    # Every input remains finite; fx * tx overflows while forming K [R | t].
    description["cameras"][str(camera_id)]["params"][0] = 1e308
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("path,value", [
    pytest.param(("translation",), [2.0, 2.0, 3.0], id="frame-translation-conflicts-with-image"),
    pytest.param(("rotation", "quat"), [0.0, 0.0, 1.0, 0.0], id="frame-rotation-conflicts-with-image"),
])
def test_finite_unit_frame_pose_must_compose_to_the_image_pose(adapter, description, path, value):
    # Replace the list so the fixture's shared raw pose references cannot alter the image.
    _replace(description, ("frames", "101", "rig_from_world") + path, value)
    assert description["images"]["11"]["translation"] == [1.0, 2.0, 3.0]
    assert description["images"]["11"]["quaternion_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(adapter.AdapterError):
        adapter.adapt_description(description)


@pytest.mark.parametrize("image_id", [11, 42])
def test_scene_rays_use_inverse_intrinsics_and_rotation_without_legacy_ray(adapter, description, monkeypatch, image_id):
    def forbidden(*args, **kwargs):
        raise AssertionError("ResearchScene.rays must not call legacy Camera.ray")

    monkeypatch.setattr(Camera, "ray", forbidden)
    if image_id == 11:
        # Origin cameras exposed the zero-vector failure in the legacy SVD ray path.
        description["images"]["11"]["translation"] = [0.0, 0.0, 0.0]
        description["images"]["11"]["camera_center"] = [0.0, 0.0, 0.0]
        description["frames"]["101"]["rig_from_world"]["translation"] = [0.0, 0.0, 0.0]
        for index, point_id in enumerate(description["images"]["11"]["point3D_ids"]):
            if point_id != UNMATCHED:
                x, y, z = description["points3D"][str(point_id)]["xyz"]
                description["images"]["11"]["points2D"][index] = [800.0 * x / z + 311.25, 600.0 * y / z + 247.75]
        pixels = [[311.25, 247.75], [1111.25, -952.25], [511.25, 547.75]]
        expected_k = np.array([[800.0, 0.0, 311.25], [0.0, 600.0, 247.75], [0.0, 0.0, 1.0]])
        expected_rotation = np.eye(3)
        expected_center = [0.0, 0.0, 0.0]
    else:
        pixels = [[329.5, 233.125], [1029.5, -1166.875], [504.5, 583.125]]
        expected_k = np.array([[700.0, 0.0, 329.5], [0.0, 700.0, 233.125], [0.0, 0.0, 1.0]])
        expected_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        expected_center = [-1.0, -2.0, -2.0]
    scene = adapter.adapt_description(description)
    center, directions = scene.rays(image_id, pixels)
    camera_directions = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(expected_k).T
    expected_directions = camera_directions @ expected_rotation
    assert isinstance(center, np.ndarray) and center.shape == (3,)
    assert isinstance(directions, np.ndarray) and directions.shape == (len(pixels), 3)
    np.testing.assert_allclose(center, expected_center, atol=1e-12, rtol=0)
    np.testing.assert_allclose(directions, expected_directions, atol=1e-12, rtol=0)
    np.testing.assert_allclose((directions @ expected_rotation.T)[:, 2], 1.0, atol=1e-12, rtol=0)
    # The principal-point ray remains nonzero; other rays retain camera z=1 length.
    assert np.linalg.norm(directions[0]) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(directions[1]) > 1.0
    np.testing.assert_allclose(scene.cameras[image_id].project(center + directions), pixels, atol=1e-10, rtol=0)
