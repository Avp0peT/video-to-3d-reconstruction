"""TEST_DATA only: pure triangle metrics, no native or model execution."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.video_masks.density_model_comparison import mesh_metrics


def test_single_triangle_boundary_and_scale_invariance():
    first = mesh_metrics([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    scaled = mesh_metrics([[0, 0, 0], [9, 0, 0], [0, 9, 0]], [[0, 1, 2]])
    assert first == scaled
    assert first['boundary_edges'] == 3
    assert first['nonmanifold_edges'] == 0
    assert first['largest_one_percent_face_area_fraction'] == 1


def test_empty_and_degenerate_area_unknown():
    assert mesh_metrics([], [])['largest_one_percent_face_area_fraction'] is None
    doc = mesh_metrics([[0, 0, 0]], [[0, 0, 0]])
    assert doc['degenerate_faces'] == 1
    assert doc['largest_one_percent_face_area_fraction'] is None


def test_nonmanifold_incidence():
    doc = mesh_metrics([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
                       [[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    assert doc['nonmanifold_edges'] == 1
    assert doc['boundary_edges'] == 6


def test_outside_index_rejected():
    with pytest.raises(ValueError):
        mesh_metrics([[0, 0, 0]], [[0, 1, 2]])
