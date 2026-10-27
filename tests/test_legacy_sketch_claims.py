"""Legacy values cannot certify S-2002 quotient nearest neighbours."""
import numpy as np
import pytest
from kozlov_lab.compression.official.kozlov_sketch import (
    certified_nearest, kozlov_image_distance, kozlov_sketch_distance_bounds,
)


def test_translated_exact_match_cannot_be_excluded_by_placement_distance():
    a = [(0, 0), (1, 0)]
    b = [(10, 0), (11, 0)]
    c = [(0, 0), (2, 0)]
    result = certified_nearest(a, [b, c], [0, 0], r_query=0)
    # B is exactly congruent to A by translation. C has a different diameter;
    # its quotient Hausdorff distance is 1/2, attained by centring the segments.
    assert kozlov_image_distance(a, [(x-10, y) for x,y in b], align=False)["distance"] == 0
    assert kozlov_image_distance(a, [(-0.5, 0), (1.5, 0)], align=False)["distance"] == 0.5
    assert [row["sketch_distance"] for row in result["rows"]] == [10, 1]
    assert result["placement_candidate_index"] == 1
    assert result["certified"] is False
    assert result["nearest_index"] is None
    assert result["intervals_source_certified"] is False
    assert result["reason"] == "SOURCE_QUOTIENT_METRIC_NOT_VERIFIED"


@pytest.mark.parametrize("candidates,radii", [([], []), ( [[(0,0)]], [0])])
def test_even_empty_or_single_candidate_does_not_create_source_certificate(candidates,radii):
    result = certified_nearest([(0,0)], candidates, radii, r_query=0)
    assert result["certified"] is False
    assert result["nearest_index"] is None


def test_distance_modes_expose_different_units_without_changing_legacy_value():
    a = np.array([(0.,0.),(1.,0.),(0.,2.)])
    b = a + (10.,3.)
    fixed = kozlov_image_distance(a,b,align=False)
    grid = kozlov_image_distance(a,b,n_rotations=4,n_scales=1)
    assert fixed["distance"] == pytest.approx(np.sqrt(109))
    assert fixed["metric_kind"] == "fixed_placement_hausdorff"
    assert fixed["units"] == "input_coordinate_units"
    assert grid["distance"] == pytest.approx(0,abs=1e-14)
    assert grid["metric_kind"] == "normalized_grid_heuristic"
    assert grid["units"] == "normalized"
    assert grid["reflection_search"] is False
    assert grid["optimal_translation_search"] is False
    assert grid["author_exact"] is False
    assert grid["source_metric_verified"] is False


def test_scalar_bounds_remain_compatible_but_have_no_input_certification():
    assert kozlov_sketch_distance_bounds(3,1,2) == (1,5)
    assert "Requires a verified" in kozlov_sketch_distance_bounds.__doc__
