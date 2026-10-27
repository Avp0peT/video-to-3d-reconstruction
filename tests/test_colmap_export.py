"""``sfm-native-v1`` model roundtrip tests against a known, hand-checked fixture.

The fixture is built with the **real** backend and then verified against values
this file computes by hand from the camera equations.  Taking expectations from
``cam.img_from_cam`` or from the same export path would only prove the library
agrees with itself, so the projections below are re-derived:

    SIMPLE_RADIAL   x'=X/Z, y'=Y/Z, r2=x'^2+y'^2
                    x''=x'(1+k1 r2), y''=y'(1+k1 r2);  u=f x''+cx, v=f y''+cy
    PINHOLE         u=fx X/Z+cx, v=fy Y/Z+cy

The camera holds a genuine non-zero radial term, because a fixture with k1=0
would not exercise the distortion code path at all — and a distorted camera is
exactly the case the contract forbids reducing to a bare ``K``.

Numeric comparisons use the project's own tolerances from ``kozlov_lab.config``;
those are floating-point comparison bounds, never SfM quality thresholds.
"""

from __future__ import annotations

import math

import numpy as np
import pycolmap
import pytest

from kozlov_lab.config import FLOAT_TOL
from kozlov_lab.reconstruction import export as export_module

def unit_quaternion(axis: str, angle: float) -> list[float]:
    """A precisely unit ``wxyz`` quaternion for a rotation about one axis.

    Written with ``cos``/``sin`` rather than rounded decimals on purpose: a
    rounded quaternion is not unit, and constructing a rotation matrix from a
    non-unit quaternion scales it by ``|q|^2``.  That produced a fixture whose
    camera centres were off by ~1e-6 while everything else looked correct — the
    kind of error that is easy to accept as "close enough" instead of fixing.
    """
    half = angle / 2.0
    sin, cos = math.sin(half), math.cos(half)
    return {
        "x": [sin, 0.0, 0.0, cos],
        "y": [0.0, sin, 0.0, cos],
        "z": [0.0, 0.0, sin, cos],
    }[axis]


# --------------------------------------------------------------------------- #
# fixture definition: 3 cameras, 12 non-coplanar points, explicit observations
# --------------------------------------------------------------------------- #
CAMERAS = {
    1: ("SIMPLE_RADIAL", 640, 480, [768.0, 320.0, 240.0, 0.0]),
    4: ("SIMPLE_RADIAL", 800, 600, [960.0, 400.0, 300.0, -0.05]),  # non-zero radial
    9: ("PINHOLE", 640, 480, [800.0, 810.0, 320.0, 240.0]),
}

#: z varies independently of x and y, so the twelve points fit no plane — a
#: coplanar set would let a degenerate configuration pass as a real model.
POINTS = np.array([
    [-1.0, -1.0, 3.0], [1.0, -1.0, 3.5], [1.0, 1.0, 4.0], [-1.0, 1.0, 3.2],
    [0.0, 0.0, 5.0], [0.5, -0.5, 2.6], [-0.5, 0.5, 5.5], [0.7, 0.3, 4.4],
    [-0.3, -0.8, 6.1], [1.2, 0.4, 2.9], [-1.1, 0.6, 6.6], [0.2, 1.1, 3.8],
])

#: image_id -> (unit quaternion xyzw, translation); ids are deliberately
#: non-contiguous, as are the camera ids above.
POSES = {
    1: (unit_quaternion("x", 0.0), [0.0, 0.0, 0.0]),
    3: (unit_quaternion("y", 0.4), [0.5, 0.0, 0.2]),
    7: (unit_quaternion("z", 0.28), [-0.3, 0.4, 0.1]),
}
IMAGE_CAMERA = {1: 1, 3: 4, 7: 9}

OBSERVATIONS = {
    0: [1, 3], 1: [1, 3], 2: [1, 7], 3: [3, 7], 4: [1, 3, 7], 5: [3, 7],
    6: [1, 7], 7: [1, 3, 7], 8: [3, 7], 9: [1, 3], 10: [1, 3, 7], 11: [3, 7],
}


def quaternion_to_rotation(quaternion) -> np.ndarray:
    """Rotation matrix for a Hamilton ``wxyz`` quaternion (normalised first)."""
    values = np.asarray(quaternion, dtype=float).ravel()
    norm = float(np.linalg.norm(values))
    x, y, z, w = values / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project(model_name: str, params, camera_point) -> np.ndarray:
    """Independent projection, written directly from the camera equations."""
    x, y, z = camera_point
    if model_name == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        xp, yp = x / z, y / z
        r2 = xp * xp + yp * yp
        factor = 1.0 + k1 * r2
        return np.array([f * xp * factor + cx, f * yp * factor + cy])
    if model_name == "PINHOLE":
        fx, fy, cx, cy = params
        return np.array([fx * x / z + cx, fy * y / z + cy])
    raise AssertionError(f"unexpected model {model_name}")


def observed_points(image_id: int) -> list[int]:
    return [index for index in sorted(OBSERVATIONS) if image_id in OBSERVATIONS[index]]


def expected_pixels() -> dict[int, dict[int, np.ndarray]]:
    """Every observation's pixel position, computed here rather than read back."""
    result: dict[int, dict[int, np.ndarray]] = {}
    for image_id, (quaternion, translation) in POSES.items():
        model_name, _w, _h, params = CAMERAS[IMAGE_CAMERA[image_id]]
        rotation = quaternion_to_rotation(quaternion)
        result[image_id] = {}
        for point_index in observed_points(image_id):
            camera_point = rotation @ POINTS[point_index] + np.asarray(translation, dtype=float)
            result[image_id][point_index] = project(model_name, params, camera_point)
    return result


@pytest.fixture(scope="module")
def fixture_model() -> dict:
    """Build the TEST_DATA model with the real backend, once per module."""
    reconstruction, point_ids = _build_fixture_model(pycolmap)
    return {
        "reconstruction": reconstruction,
        "point_ids": point_ids,
        "pixels": expected_pixels(),
        "test_data_marker": "TEST_DATA: procedurally constructed, no photograph involved",
    }


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory, fixture_model) -> dict:
    """Write binary and text forms and read both back."""
    root = tmp_path_factory.mktemp("model_roundtrip")
    binary_dir = root / "binary"
    text_dir = root / "text"
    binary_dir.mkdir()
    text_dir.mkdir()

    fixture_model["reconstruction"].write(str(binary_dir))
    fixture_model["reconstruction"].write_text(str(text_dir))

    return {
        "root": root,
        "binary_dir": binary_dir,
        "text_dir": text_dir,
        "from_binary": export_module.read_binary_model(binary_dir),
        "from_text": export_module.read_text_model(text_dir),
    }


# --------------------------------------------------------------------------- #
# fixture sanity: the fixture itself must be what the contract asks for
# --------------------------------------------------------------------------- #
def test_fixture_shape_is_test_data(fixture_model) -> None:
    assert "TEST_DATA" in fixture_model["test_data_marker"]
    assert fixture_model["reconstruction"].num_cameras() == 3
    assert fixture_model["reconstruction"].num_images() == 3
    assert fixture_model["reconstruction"].num_points3D() == 12


def test_fixture_covers_both_required_models(fixture_model) -> None:
    names = sorted(camera.model_name for camera in fixture_model["reconstruction"].cameras.values())
    assert names == ["PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL"]
    radial = fixture_model["reconstruction"].camera(4).params[3]
    assert radial != 0.0, "the SIMPLE_RADIAL fixture must have a non-zero radial term"


def test_fixture_points_are_not_coplanar(fixture_model) -> None:
    """A coplanar set would be degenerate, so the fixture must not be one."""
    centred = POINTS - POINTS.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    assert singular_values[2] > 0.1 * singular_values[0], "points are close to coplanar"
    residual = singular_values[2] / singular_values[0]
    assert residual > 1e-3


# --------------------------------------------------------------------------- #
# both native forms must survive a roundtrip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("form", ["binary", "text"])
def test_counts_survive_both_forms(roundtrip, form: str) -> None:
    model = roundtrip[f"from_{form}"]
    assert model.num_cameras() == 3
    assert model.num_images() == 3
    assert model.num_points3D() == 12
    # rigs and frames produced by this version are preserved too
    assert model.num_rigs() == 3
    assert model.num_frames() == 3


@pytest.mark.parametrize("form", ["binary", "text"])
def test_camera_models_and_full_params_survive(roundtrip, form: str) -> None:
    model = roundtrip[f"from_{form}"]
    for camera_id, (model_name, width, height, params) in CAMERAS.items():
        camera = model.camera(camera_id)
        assert camera.model_name == model_name, f"camera {camera_id} changed model"
        assert (camera.width, camera.height) == (width, height)
        assert np.allclose(camera.params, params, atol=FLOAT_TOL, rtol=0), (
            f"camera {camera_id} parameters changed: {list(camera.params)} vs {params}"
        )


@pytest.mark.parametrize("form", ["binary", "text"])
def test_poses_survive_in_world_to_camera_form(roundtrip, form: str) -> None:
    model = roundtrip[f"from_{form}"]
    for image_id, (quaternion, translation) in POSES.items():
        pose = model.image(image_id).cam_from_world()
        # quaternions are sign-ambiguous representations of the same rotation
        expected_rotation = quaternion_to_rotation(quaternion)
        actual_rotation = np.asarray(pose.rotation.matrix())
        assert np.allclose(actual_rotation, expected_rotation, atol=1e-9, rtol=0), (
            f"image {image_id} rotation changed"
        )
        assert np.allclose(pose.translation, translation, atol=FLOAT_TOL, rtol=0)


@pytest.mark.parametrize("form", ["binary", "text"])
def test_camera_centre_uses_the_contract_formula(roundtrip, form: str) -> None:
    model = roundtrip[f"from_{form}"]
    for image_id, (quaternion, translation) in POSES.items():
        description = export_module.describe_model(model)["images"][str(image_id)]
        expected = -quaternion_to_rotation(quaternion).T @ np.asarray(translation, dtype=float)
        assert np.allclose(description["camera_center"], expected, atol=1e-9, rtol=0)


def test_frame_poses_survive_source_and_native_forms(fixture_model, roundtrip) -> None:
    """Trivial-rig frame poses must retain the independently specified poses.

    The nonidentity frames catch treating rig_from_world as a method and silently
    describing every pose as null, which source/export equality alone misses.
    """
    models = {
        "source": fixture_model["reconstruction"],
        "binary": roundtrip["from_binary"],
        "text": roundtrip["from_text"],
    }
    for form, model in models.items():
        description = export_module.describe_model(model)
        for image_id, (quaternion, translation) in POSES.items():
            frame_id = model.image(image_id).frame_id
            pose = description["frames"][str(frame_id)]["rig_from_world"]
            assert pose is not None, f"{form} frame {frame_id} lost its known pose"
            assert np.allclose(pose["translation"], translation, atol=FLOAT_TOL, rtol=0)
            actual_rotation = quaternion_to_rotation(pose["rotation"]["quat"])
            expected_rotation = quaternion_to_rotation(quaternion)
            assert np.allclose(actual_rotation, expected_rotation, atol=1e-9, rtol=0), (
                f"{form} frame {frame_id} rotation changed"
            )


@pytest.mark.parametrize("form", ["binary", "text"])
def test_points_and_tracks_survive(roundtrip, fixture_model, form: str) -> None:
    model = roundtrip[f"from_{form}"]
    point_ids = fixture_model["point_ids"]
    for point_index, point_id in sorted(point_ids.items()):
        point = model.point3D(point_id)
        assert np.allclose(point.xyz, POINTS[point_index], atol=FLOAT_TOL, rtol=0)
        elements = sorted((int(e.image_id), int(e.point2D_idx)) for e in point.track.elements)
        expected = sorted(
            (image_id, observed_points(image_id).index(point_index))
            for image_id in OBSERVATIONS[point_index]
        )
        assert elements == expected, f"point {point_id} track changed"


@pytest.mark.parametrize("form", ["binary", "text"])
def test_native_pixel_coordinates_match_independent_projection(roundtrip, form: str) -> None:
    """The stored observations must equal the projection computed by hand."""
    model = roundtrip[f"from_{form}"]
    pixels = expected_pixels()
    for image_id, expectations in pixels.items():
        image = model.image(image_id)
        assert image.num_points2D() == len(expectations)
        for point_index, expected in expectations.items():
            stored = np.asarray(image.points2D[observed_points(image_id).index(point_index)].xy)
            assert np.allclose(stored, expected, atol=1e-6, rtol=0), (
                f"image {image_id} point {point_index}: stored {stored} != projected {expected}"
            )


def test_projection_helper_disagrees_with_a_deliberate_wrong_model() -> None:
    """Guard: the projection check must be able to fail."""
    correct = project("SIMPLE_RADIAL", [960.0, 400.0, 300.0, -0.05], np.array([0.1, 0.2, 4.0]))
    wrong_radial = project("SIMPLE_RADIAL", [960.0, 400.0, 300.0, 0.0], np.array([0.1, 0.2, 4.0]))
    assert not np.allclose(correct, wrong_radial, atol=1e-6), "the radial term must matter here"


def test_binary_and_text_describe_the_same_model(roundtrip) -> None:
    differences = export_module.compare_models(
        roundtrip["from_binary"], roundtrip["from_text"], tolerance=1e-9
    )
    assert differences == [], f"binary and text forms disagree: {differences[:5]}"


def test_compare_models_detects_a_perturbation(roundtrip, tmp_path) -> None:
    """Negative control: an identical-looking but different model must be caught."""
    altered = pycolmap.Reconstruction(str(roundtrip["binary_dir"]))
    camera = altered.camera(4)
    perturbed = np.asarray(camera.params).copy()
    perturbed[3] += 1e-4  # radial term, the parameter a K-only export would drop
    camera.params = perturbed

    differences = export_module.compare_models(
        roundtrip["from_binary"], altered, tolerance=1e-9
    )
    assert any("cameras.4.params" in item for item in differences), differences


def test_compare_models_detects_a_moved_image(roundtrip) -> None:
    altered = pycolmap.Reconstruction(str(roundtrip["binary_dir"]))
    image = altered.image(3)
    pose = image.cam_from_world()
    moved = pycolmap.Rigid3d(pose.rotation, np.asarray(pose.translation) + np.array([1.0, 0.0, 0.0]))
    # the frame must be addressed with the image's own camera id: (CAMERA, 3) is
    # not a sensor in this rig, since image 3 uses camera 4
    altered.frame(image.frame_id).set_cam_from_world(image.camera_id, moved)

    differences = export_module.compare_models(
        roundtrip["from_binary"], altered, tolerance=1e-9
    )
    assert any("translation" in item for item in differences), differences


# --------------------------------------------------------------------------- #
# usability floor
# --------------------------------------------------------------------------- #
def test_fixture_model_is_structurally_usable(roundtrip) -> None:
    usable, reasons = export_module.is_usable(roundtrip["from_binary"])
    assert usable, reasons


def test_usability_floor_is_two_images_and_one_point(roundtrip) -> None:
    """The floor is structural; it must reject a fewer-than-two-image model."""
    trimmed = pycolmap.Reconstruction(str(roundtrip["binary_dir"]))
    for image_id in (3, 7):
        trimmed.deregister_frame(trimmed.image(image_id).frame_id)
    usable, reasons = export_module.is_usable(trimmed)
    assert not usable
    assert any("registered image" in reason for reason in reasons), reasons


def test_model_files_digest_changes_when_a_file_changes(roundtrip) -> None:
    before = export_module.model_files_digest(roundtrip["binary_dir"])
    target = roundtrip["binary_dir"] / "cameras.bin"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\x00")
        assert export_module.model_files_digest(roundtrip["binary_dir"]) != before
    finally:
        target.write_bytes(original)
    assert export_module.model_files_digest(roundtrip["binary_dir"]) == before


# =========================================================================== #
# R4 regressions: rig/frame identity and source preservation
# =========================================================================== #
def _renumber_text_model(text_dir, offset: int = 100):
    """Return a copy of a text model with every rig and frame id shifted.

    This mirrors the main line's counterexample: the renumbered model is still
    native-valid, so a count-only comparison cannot tell it apart from the
    original.
    """
    import pathlib
    import shutil

    target = text_dir.parent / f"{text_dir.name}_renumbered"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(text_dir, target)

    def rewrite(name: str, columns: tuple[int, ...]) -> None:
        path = target / name
        out_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                parts = line.split()
                for column in columns:
                    if column < len(parts) and parts[column].isdigit():
                        parts[column] = str(int(parts[column]) + offset)
                line = " ".join(parts)
            out_lines.append(line)
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8", newline="\n")

    rewrite("rigs.txt", (0,))     # RIG_ID ...
    rewrite("frames.txt", (0, 1))  # FRAME_ID RIG_ID ...
    return target


def test_rig_and_frame_identities_are_described(roundtrip) -> None:
    description = export_module.describe_model(roundtrip["from_binary"])
    assert set(description["rigs"]) == {"1", "4", "9"}
    assert set(description["frames"]) == {"1", "3", "7"}
    assert description["images"]["1"]["frame_id"] == 1
    assert description["rigs"]["1"]["rig_id"] == 1
    assert description["frames"]["1"]["rig_id"] == 1
    # the rig-relative sensor poses are part of the description
    assert description["rigs"]["1"]["sensor_from_rig"]


def test_renumbered_rig_and_frame_ids_are_detected(roundtrip) -> None:
    """The proven blind spot: a valid model with shifted ids must not compare equal."""
    renumbered_dir = _renumber_text_model(roundtrip["text_dir"])
    renumbered = export_module.read_text_model(renumbered_dir)

    # the changed model is still structurally valid
    assert renumbered.num_rigs() == roundtrip["from_text"].num_rigs()
    assert renumbered.num_frames() == roundtrip["from_text"].num_frames()

    differences = export_module.compare_models(
        roundtrip["from_text"], renumbered, tolerance=1e-9
    )
    assert differences, "renumbered rig/frame ids must be reported as a difference"
    joined = " ".join(differences)
    assert "rigs" in joined and "frames" in joined, differences[:10]


def test_a_renumbered_frame_is_seen_through_the_image_link(roundtrip) -> None:
    """Images carry frame_id, so a frame renumbering must surface there too."""
    renumbered_dir = _renumber_text_model(roundtrip["text_dir"])
    renumbered = export_module.read_text_model(renumbered_dir)

    left = export_module.describe_model(roundtrip["from_text"])
    right = export_module.describe_model(renumbered)
    frame_ids_left = [image["frame_id"] for image in left["images"].values()]
    frame_ids_right = [image["frame_id"] for image in right["images"].values()]
    assert frame_ids_left != frame_ids_right
    assert export_module.compare_descriptions(left, right, tolerance=1e-9)


def test_compare_descriptions_catches_a_single_changed_camera_parameter(roundtrip) -> None:
    """Negative control for the comparison itself."""
    left = export_module.describe_model(roundtrip["from_binary"])
    right = export_module.describe_model(roundtrip["from_binary"])
    right["cameras"]["4"]["params"][3] += 1e-4
    differences = export_module.compare_descriptions(left, right, tolerance=1e-9)
    assert any("cameras.4.params" in item for item in differences), differences


def test_compare_descriptions_is_not_confused_by_tiny_numeric_noise(roundtrip) -> None:
    """The tolerance must absorb representation noise without hiding identity."""
    left = export_module.describe_model(roundtrip["from_binary"])
    right = export_module.describe_model(roundtrip["from_binary"])
    right["images"]["1"]["translation"] = [v + 1e-13 for v in right["images"]["1"]["translation"]]
    assert export_module.compare_descriptions(left, right, tolerance=1e-9) == []


def test_worker_export_stage_verifies_source_against_both_forms(tmp_path) -> None:
    """The real worker export path must record a source-versus-both verdict."""
    from kozlov_lab.reconstruction import colmap_adapter as adapter_module

    pycolmap = export_module.require_pycolmap()
    source_dir = tmp_path / "native" / "0"
    source_dir.mkdir(parents=True)
    fixture, _point_ids = _build_fixture_model(pycolmap)
    fixture.write(str(source_dir))

    spec = {
        "output_path": str(tmp_path / "native"),
        "models_root": str(tmp_path / "models"),
        "tolerance": 1e-9,
    }
    result = adapter_module._stage_export(pycolmap, spec)

    assert result["model_count"] == 1
    entry = result["models"][0]
    assert entry["roundtrip_verified"] is True, entry["source_vs_binary_differences"][:3]
    assert entry["source_vs_binary_differences"] == []
    assert entry["source_vs_text_differences"] == []
    # both native forms were written into their own directories
    assert "cameras.bin" in entry["binary_files"]
    assert "cameras.txt" in entry["text_files"]
    # and the source description is persisted for the parent to compare against
    import json as _json
    saved = _json.loads(open(entry["descriptions_path"], encoding="utf-8").read())
    assert set(saved) >= {"source", "binary", "text"}
    assert saved["source"]["num_points3D"] == 12


def test_parent_rejects_a_worker_that_reports_a_failed_roundtrip(tmp_path) -> None:
    """A failed worker verdict must become BACKEND_ERROR, not be ignored.

    Control-flow test: the export result is fabricated, because a real writer that
    drops data on demand is not something to reproduce.  It checks propagation
    only, and is not evidence that the native writer loses anything.
    """
    from kozlov_lab.reconstruction import cli

    result = {
        "model_count": 1,
        "models": [{
            "model_id": "0",
            "binary_dir": str(tmp_path / "binary"),
            "text_dir": str(tmp_path / "text"),
            "descriptions_path": str(tmp_path / "descriptions.json"),
            "roundtrip_verified": False,
            "source_vs_binary_differences": ["$.rigs.1.rig_id: 1 != 101"],
            "source_vs_text_differences": [],
        }],
    }
    with pytest.raises(Exception) as caught:
        cli._verify_export({"models": []}, result, tmp_path)
    assert "source-preservation" in str(caught.value)


def test_parent_rejects_a_model_whose_export_has_no_descriptions(tmp_path) -> None:
    """A verified claim with no descriptions document is not verifiable."""
    from kozlov_lab.reconstruction import cli

    result = {
        "model_count": 1,
        "models": [{
            "model_id": "0",
            "binary_dir": str(tmp_path / "binary"),
            "text_dir": str(tmp_path / "text"),
            "descriptions_path": str(tmp_path / "missing.json"),
            "roundtrip_verified": True,
            "source_vs_binary_differences": [],
            "source_vs_text_differences": [],
        }],
    }
    with pytest.raises(Exception) as caught:
        cli._verify_export({"models": []}, result, tmp_path)
    assert "no descriptions document" in str(caught.value)


def _build_fixture_model(pycolmap_module):
    """The 3-camera / 12-point fixture; returns ``(reconstruction, point_ids)``.

    The real point ids are returned rather than assumed to be ``index + 1``: the
    backend assigns ids, and inferring them from insertion order would be the same
    "ids are the input order" mistake the contract warns about.
    """
    reconstruction = pycolmap_module.Reconstruction()
    for camera_id, (model_name, width, height, params) in CAMERAS.items():
        camera = pycolmap_module.Camera.create_from_model_id(
            camera_id, getattr(pycolmap_module.CameraModelId, model_name), params[0], width, height
        )
        camera.params = np.asarray(params, dtype=float)
        reconstruction.add_camera_with_trivial_rig(camera)

    pixels = expected_pixels()
    for image_id, (quaternion, translation) in POSES.items():
        points2D = pycolmap_module.Point2DList()
        for point_index in observed_points(image_id):
            points2D.append(pycolmap_module.Point2D(xy=pixels[image_id][point_index]))
        image = pycolmap_module.Image(
            name=f"img_{image_id}.png", points2D=points2D,
            camera_id=IMAGE_CAMERA[image_id], image_id=image_id,
        )
        reconstruction.add_image_with_trivial_frame(
            image,
            pycolmap_module.Rigid3d(
                pycolmap_module.Rotation3d(quaternion_to_rotation(quaternion)),
                np.asarray(translation, dtype=float),
            ),
        )

    point_ids = {}
    for point_index in sorted(OBSERVATIONS):
        track = pycolmap_module.Track()
        for image_id in OBSERVATIONS[point_index]:
            track.add_element(image_id, observed_points(image_id).index(point_index))
        point_ids[point_index] = reconstruction.add_point3D(
            POINTS[point_index], track, np.array([10, 20, 30], dtype=np.uint8)
        )
    for point_index, image_ids in OBSERVATIONS.items():
        for image_id in image_ids:
            reconstruction.image(image_id).set_point3D_for_point2D(
                observed_points(image_id).index(point_index), point_ids[point_index]
            )
    return reconstruction, point_ids
