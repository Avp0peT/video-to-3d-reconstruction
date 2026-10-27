"""TEST_DATA: synthetic masks and image points, no geometry execution."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.video_masks.point_density import DensityPolicy, assess_density


def test_equal_totals_clustered_vs_distributed():
    mask = np.ones((8, 8), np.uint8)
    policy = DensityPolicy(grid_rows=2, grid_cols=2, min_points_per_cell=1)
    assert assess_density(mask, [[1, 1], [6, 1], [1, 6], [6, 6]], policy)['status'] == 'DENSITY_TARGET_MET'
    clustered = assess_density(mask, [[0, 0], [1, 0], [0, 1], [1, 1]], policy)
    assert clustered['status'] == 'BELOW_DENSITY_TARGET'
    assert clustered['met_area_fraction'] == .25
    assert clustered['largest_deficient_region_fraction'] == .75


def test_background_boundary_duplicates_and_no_mutation():
    mask = np.zeros((8, 8), np.uint8); mask[1:7, 1:7] = 255; mask[1:4, 1:4] = 0
    points = np.array([[0, 0], [1, 4], [1.1, 4.1], [7, 7], [-10, -10]], float)
    original = points.copy(); original_mask = mask.copy()
    doc = assess_density(mask, points, DensityPolicy(grid_rows=2, grid_cols=2))
    assert doc['unique_in_mask_points'] == 1
    assert sum(c['foreground_pixels'] for c in doc['cells']) == int((mask != 0).sum())
    assert np.array_equal(points, original) and np.array_equal(mask, original_mask)


def test_small_boundary_target_and_disconnected_regions():
    mask = np.zeros((8, 8), np.uint8); mask[0, 0] = 1; mask[7, 7] = 1
    doc = assess_density(mask, [])
    assert len(doc['cells']) == 2
    assert all(c['target_points'] == 1 for c in doc['cells'])
    assert doc['largest_deficient_region_fraction'] == .5


def test_empty_and_no_points():
    assert assess_density(np.zeros((3, 3)), [])['status'] == 'EMPTY_MASK'
    doc = assess_density(np.ones((3, 3)), [])
    assert doc['status'] == 'BELOW_DENSITY_TARGET'
    assert doc['largest_deficient_region_fraction'] == 1


@pytest.mark.parametrize('points', [[[float('nan'), 0]], [[True, False]], [[True, 1.5]], [1, 2], [[float('inf'), 0]]])
def test_invalid_coordinates(points):
    with pytest.raises(ValueError):
        assess_density(np.ones((3, 3)), points)


@pytest.mark.parametrize('kwargs', [dict(grid_rows=True), dict(min_points_per_cell=0), dict(grid_cols=129)])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        DensityPolicy(**kwargs)


def test_one_pixel_targets_cannot_exceed_distinct_raster_support():
    doc = assess_density(np.ones((1, 1)), [[0, 0]])
    assert doc['cells'][0]['target_points'] == 1
    assert doc['status'] == 'DENSITY_TARGET_MET'


def test_coordinate_half_open_rounding_boundary():
    doc = assess_density(np.ones((2, 2)), [[-.5, -.5], [1.49, 1.49], [1.5, 0], [-.5001, 0]])
    assert doc['unique_in_mask_points'] == 2
    assert sum(c['point_count'] for c in doc['cells']) == 2
