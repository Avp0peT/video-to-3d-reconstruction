"""Pure geometry/data tests; never import FreeCAD, VTK, or other native engines."""

import importlib.util
import math
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "integrations" / "meshprep" / "geometry.py"
_SPEC = importlib.util.spec_from_file_location("meshprep_geometry_under_test", _PATH)
geometry = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(geometry)

POINTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
FACES = [(0, 1, 2)]


def test_vertex_touching_closed_shells_are_separate():
    # Two tetrahedral shells share vertex 0, but no undirected edge.
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2),
             (0, 4, 5), (0, 6, 4), (0, 5, 6), (4, 6, 5)]
    assert geometry.edge_components(faces) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_shared_edges_connect_in_either_orientation():
    assert geometry.edge_components([(0, 1, 2), (2, 1, 3), (3, 1, 4)]) == [[0, 1, 2]]


def test_components_have_deterministic_face_order():
    faces = [(7, 8, 9), (1, 2, 3), (20, 21, 22), (3, 2, 4), (9, 8, 10)]
    assert geometry.edge_components(faces) == [[0, 4], [1, 3], [2]]


def test_nonmanifold_edge_connects_but_does_not_certify_mesh():
    assert geometry.edge_components([(0, 1, 2), (1, 0, 3), (0, 1, 4)]) == [[0, 1, 2]]


def test_empty_component_input():
    assert geometry.edge_components([]) == []
    assert geometry.compact_component(POINTS, FACES, []) == ([], [], [])


def test_compaction_preserves_original_ids_values_winding_and_requested_order():
    points = [(99, 99, 99), (8.0, -0.0, 3.0), (9.0, 2.0, 3.0),
              (8.0, -0.0, 3.0), (4.0, 5.0, 6.0), (1.0, 2.0, 3.0)]
    faces = [(5, 3, 1), (4, 2, 3)]
    output_points, output_faces, original = geometry.compact_component(points, faces, [1, 0])
    assert original == [1, 2, 3, 4, 5]
    assert output_points == [points[1], points[2], points[3], points[4], points[5]]
    assert output_faces == [(3, 1, 2), (4, 2, 0)]
    assert len(output_points) == 5  # Equal coordinates at IDs 1/3 are not welded.
    assert math.copysign(1.0, output_points[0][1]) == -1.0
    assert points[0] == (99, 99, 99)


def test_ascii_double_ply_roundtrips_far_origin_coordinates(tmp_path):
    points = [(1000000000.125, -1000000000.0000002, 0.12345678901234566),
              (1000000000.25, -1000000000.5, 7.0),
              (1000000000.375, -1000000000.75, -0.0)]
    target = tmp_path / "precise.ply"
    geometry.write_double_ply(target, points, [(2, 0, 1)])
    raw = target.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("ascii").splitlines()
    assert [line for line in lines if line.startswith("property double")] == [
        "property double x", "property double y", "property double z"]
    assert not any(line.startswith("property float") for line in lines)
    start = lines.index("end_header") + 1
    assert [tuple(map(float, line.split())) for line in lines[start:start + 3]] == points
    assert lines[start + 3:] == ["3 2 0 1"]
    assert math.copysign(1.0, float(lines[start + 2].split()[2])) == -1.0


def test_ply_refuses_overwrite(tmp_path):
    target = tmp_path / "existing.ply"
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        geometry.write_double_ply(target, POINTS, FACES)
    assert target.read_bytes() == b"original"


@pytest.mark.parametrize("bad_face", [(0, 1), (0, 1, 2, 3), (0, True, 2),
                                      (0, 1.0, 2), (-1, 1, 2), (0, 0, 1)])
def test_invalid_faces_rejected(bad_face, tmp_path):
    with pytest.raises(ValueError):
        geometry.edge_components([bad_face])
    with pytest.raises(ValueError):
        geometry.compact_component(POINTS, [bad_face], [0])
    target = tmp_path / "invalid.ply"
    with pytest.raises(ValueError):
        geometry.write_double_ply(target, POINTS, [bad_face])
    assert not target.exists()


def test_out_of_range_indices_rejected_when_points_available(tmp_path):
    with pytest.raises(ValueError):
        geometry.compact_component(POINTS, [(0, 1, 3)], [0])
    with pytest.raises(ValueError):
        geometry.write_double_ply(tmp_path / "invalid.ply", POINTS, [(0, 1, 3)])


@pytest.mark.parametrize("point", [(0, 0), (0, 0, 0, 0), (False, 0, 0),
                                   (float("nan"), 0, 0), (0, float("inf"), 0), ("0", 0, 0)])
def test_bad_coordinates_rejected_even_when_not_selected(point, tmp_path):
    points = POINTS + [point]
    with pytest.raises(ValueError):
        geometry.compact_component(points, FACES, [0])
    with pytest.raises(ValueError):
        geometry.write_double_ply(tmp_path / "invalid.ply", points, FACES)


@pytest.mark.parametrize("selection", [[0, 0], [1], [-1], [False], [0.0], None])
def test_bad_face_selections_rejected(selection):
    with pytest.raises(ValueError):
        geometry.compact_component(POINTS, FACES, selection)


def stats(**changes):
    result = dict(components=1, vertices=4, edges=6, faces=4, signed_volume=1.0)
    result.update(changes)
    return result


def test_metrics_preserve_euler_and_allow_threshold_boundary():
    report = geometry.validate_metrics(stats(), stats(vertices=6, edges=12, faces=8,
                                                       signed_volume=1.125), 0.125)
    assert report["passed"] is True
    assert report["before_euler"] == report["after_euler"] == 2
    assert report["relative_volume_change"] == 0.125
    assert "not topology" in report["scope"]


@pytest.mark.parametrize("which", ["before", "after"])
@pytest.mark.parametrize("changes,problem", [({"components": 2}, "components"),
                                           ({"signed_volume": 0}, "positive"),
                                           ({"signed_volume": -1.0}, "positive")])
def test_metric_components_and_positive_volume_gates(which, changes, problem):
    before, after = stats(), stats()
    (before if which == "before" else after).update(changes)
    report = geometry.validate_metrics(before, after, 0.01)
    assert report["passed"] is False
    assert any(problem in item for item in report["problems"])


def test_changed_euler_rejected_despite_identical_volume():
    report = geometry.validate_metrics(stats(), stats(edges=7), 0.0)
    assert report["passed"] is False
    assert "Euler characteristic changed" in report["problems"]


@pytest.mark.parametrize("volume", [0.874, 1.126])
def test_relative_volume_change_denominator_is_before(volume):
    report = geometry.validate_metrics(stats(), stats(signed_volume=volume), 0.125)
    assert report["passed"] is False
    assert "relative volume change exceeds limit" in report["problems"]


@pytest.mark.parametrize("limit", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_metric_threshold_rejected(limit):
    with pytest.raises(ValueError):
        geometry.validate_metrics(stats(), stats(), limit)


@pytest.mark.parametrize("changes", [{"vertices": True}, {"components": 1.0},
                                      {"faces": -1}, {"signed_volume": float("nan")},
                                      {"signed_volume": float("inf")}, {"signed_volume": False}])
def test_invalid_metric_measurements_rejected(changes):
    with pytest.raises(ValueError):
        geometry.validate_metrics(stats(), stats(**changes), 0.01)


def test_missing_metric_measurement_rejected():
    incomplete = stats()
    del incomplete["edges"]
    with pytest.raises(ValueError, match="missing fields"):
        geometry.validate_metrics(stats(), incomplete, 0.01)
