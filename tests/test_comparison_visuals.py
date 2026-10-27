"""Synthetic display checks; no real object or reconstruction is loaded."""
from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from integrations.mvs_surface.comparison_visuals import (
    BACKGROUND, HEADER, PANEL_HEIGHT, PANEL_WIDTH, RASTER_HEIGHT,
    render_comparison, write_projection_comparison,
)


def mesh(vertices=None, faces=None, title="TEST_DATA"):
    return dict(title=title,
                vertices=np.asarray([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]]) if vertices is None else vertices,
                faces=np.array([[0, 1, 2]], np.int64) if faces is None else faces,
                color=(70, 170, 245))


def raster(path, column=0, row=0):
    with Image.open(path) as image:
        return np.asarray(image)[row * PANEL_HEIGHT + HEADER:row * PANEL_HEIGHT + HEADER + RASTER_HEIGHT,
                                 column * PANEL_WIDTH:(column + 1) * PANEL_WIDTH].copy()


def test_zbuffer_hides_far_triangle_for_both_input_orders(tmp_path):
    front = mesh()["vertices"]
    back = np.array([[-.8, -.8, 1.], [.8, -.8, 1.], [0., .8, 2.]])
    combined = np.concatenate([front, back])
    meshes = [mesh(combined, np.array([[3, 4, 5], [0, 1, 2]])),
              mesh(combined, np.array([[0, 1, 2], [3, 4, 5]])), mesh(front)]
    path = tmp_path / "occlusion.png"
    result = render_comparison(meshes, path, rotations=[np.eye(3)], reference_vertices=combined)
    assert result["panels"][0]["counts"]["rasterized_faces"] == 2
    assert [panel["counts"]["visible_faces"] for panel in result["panels"]] == [1, 1, 1]
    np.testing.assert_array_equal(raster(path, 0), raster(path, 1))
    np.testing.assert_array_equal(raster(path, 1), raster(path, 2))
    assert result["bytes"] == path.stat().st_size
    assert result["geometry_modified"] is False and result["independent_shape_truth"] is False


def test_shared_reference_preserves_size_and_offset_without_mutation(tmp_path):
    base = mesh()["vertices"]
    shifted = base * .4 + [.4, 0., .2]
    meshes = [mesh(base), mesh(shifted), mesh(base)]
    reference = np.vstack([base, [[0., 0., 1.]]])
    rotations = [np.eye(3), np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])]
    before = deepcopy((meshes, reference, rotations))
    path = tmp_path / "shared.png"
    result = render_comparison(meshes, path, rotations=rotations, reference_vertices=reference)
    assert len(result["panels"]) == 6
    assert len({p["projection"]["pixels_per_world_unit"] for p in result["panels"]}) == 1
    for row in range(2):
        assert result["panels"][3 * row]["projection"] == result["panels"][3 * row + 1]["projection"]
    large = np.any(raster(path, 0) != BACKGROUND, axis=2)
    small = np.any(raster(path, 1) != BACKGROUND, axis=2)
    large_x, small_x = np.where(large)[1], np.where(small)[1]
    assert .38 < np.ptp(small_x) / np.ptp(large_x) < .42
    assert small_x.mean() > large_x.mean() + 20
    with Image.open(path) as image:
        assert image.size == (1080, 1120)
    for actual, old in zip(meshes, before[0]):
        np.testing.assert_array_equal(actual["vertices"], old["vertices"])
        np.testing.assert_array_equal(actual["faces"], old["faces"])
    np.testing.assert_array_equal(reference, before[1])
    for actual, old in zip(rotations, before[2]):
        np.testing.assert_array_equal(actual, old)
    second = tmp_path / "repeat.png"
    repeated = render_comparison(meshes, second, rotations=rotations, reference_vertices=reference)
    assert repeated["sha256"] == result["sha256"]
    with pytest.raises(ValueError, match="fresh"):
        render_comparison(meshes, path, rotations=rotations, reference_vertices=reference)


@pytest.mark.parametrize("rotations", [[], [np.eye(3)] * 4, [np.eye(2)], [np.eye(3) * 2],
                                      [np.diag([1., 1., -1.])], [np.full((3, 3), np.nan)],
                                      [np.eye(3, dtype=bool)]])
def test_bad_rotations_never_publish(tmp_path, rotations):
    path = tmp_path / "bad.png"
    with pytest.raises(ValueError, match="rotation"):
        render_comparison([mesh()] * 3, path, rotations=rotations, reference_vertices=mesh()["vertices"])
    assert not path.exists()


@pytest.mark.parametrize("kind", ["nan", "index", "float_faces", "vertices_limit", "faces_limit", "title"])
def test_bad_meshes_never_publish(tmp_path, kind):
    bad = mesh()
    if kind == "nan":
        bad["vertices"][0, 0] = np.nan
    elif kind == "index":
        bad["faces"][0, 2] = 3
    elif kind == "float_faces":
        bad["faces"] = bad["faces"].astype(float)
    elif kind == "vertices_limit":
        bad["vertices"] = np.zeros((200001, 3))
    elif kind == "faces_limit":
        bad["faces"] = np.zeros((400001, 3), np.int64)
    else:
        bad["title"] = "\u4e2d\u6587"
    path = tmp_path / "bad.png"
    with pytest.raises(ValueError):
        render_comparison([mesh(), mesh(), bad], path, rotations=[np.eye(3)], reference_vertices=mesh()["vertices"])
    assert not path.exists()


def test_cancellation_is_checked_at_2048_faces_and_does_not_publish(tmp_path):
    faces = np.tile([[0, 1, 2]], (2050, 1))
    tiny = mesh(mesh()["vertices"] * .001, faces)
    calls = []

    def cancel():
        calls.append(None)
        # Entry, row start, face zero, then face 2048.
        if len(calls) == 4:
            raise RuntimeError("cancelled TEST_DATA")

    path = tmp_path / "cancelled.png"
    with pytest.raises(RuntimeError, match="cancelled"):
        render_comparison([tiny] * 3, path, rotations=[np.eye(3)],
                          reference_vertices=mesh()["vertices"], check=cancel)
    assert len(calls) == 4 and not path.exists()


def test_empty_mesh_and_projection_empty_union_are_explicit(tmp_path):
    empty = mesh(np.empty((0, 3)), np.empty((0, 3), np.int64))
    result = render_comparison([empty] * 3, tmp_path / "empty.png", rotations=[np.eye(3)],
                               reference_vertices=mesh()["vertices"])
    assert all(panel["counts"]["visible_pixels"] == 0 for panel in result["panels"])
    view = dict(image=np.zeros((8, 8, 3), np.uint8), mask=np.zeros((8, 8), bool), validity=np.ones((8, 8), bool))
    report = write_projection_comparison(tmp_path / "empty_projection.png", view, [view["mask"]] * 3)
    assert all(count["iou"] is None and count["union"] == 0 for count in report["counts"])


def test_projection_counts_exclude_invalid_and_inputs_are_unchanged(tmp_path):
    image = np.full((8, 8, 3), 80, np.uint8)
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    valid = np.ones((8, 8), bool)
    valid[0, :] = False
    partial = np.zeros((8, 8), bool)
    partial[2:6, 2:4] = True
    partial[2:4, 6:8] = True
    partial[0, :] = True
    view = dict(view_id=7, image=image, mask=mask, validity=valid)
    footprints = [mask.copy(), partial, np.zeros_like(mask)]
    before = deepcopy((view, footprints))
    path = tmp_path / "projection.png"
    result = write_projection_comparison(path, view, footprints)
    first, second, third = result["counts"]
    assert (first["fg"], first["bg"], first["missing"], first["iou"]) == (16, 0, 0, 1.)
    assert (second["fg"], second["bg"], second["missing"], second["invalid"]) == (8, 4, 8, 8)
    assert second["iou"] == .4
    assert (third["fg"], third["bg"], third["missing"], third["iou"]) == (0, 0, 16, 0.)
    assert result["independent_shape_truth"] is False and "not visibility" in result["semantics"]
    assert result["view_id"] == 7 and len(result["footprint_bindings"]) == 3
    for key in ("image", "mask", "validity"):
        np.testing.assert_array_equal(view[key], before[0][key])
    for actual, old in zip(footprints, before[1]):
        np.testing.assert_array_equal(actual, old)
    with Image.open(path) as output:
        assert output.size == (1440, 560)
    with pytest.raises(ValueError, match="fresh"):
        write_projection_comparison(path, view, footprints)


def test_projection_requires_boolean_native_masks_and_validity(tmp_path):
    mask = np.ones((8, 8), bool)
    view = dict(image=np.zeros((8, 8, 3), np.uint8), mask=mask, validity=mask.copy())
    with pytest.raises(ValueError, match="footprint"):
        write_projection_comparison(tmp_path / "wrong.png", view, [mask, mask, mask.astype(np.uint8)])
    del view["validity"]
    with pytest.raises(ValueError, match="validity"):
        write_projection_comparison(tmp_path / "missing.png", view, [mask] * 3)
