"""Real OpenCV CPU counterexamples for observation-only boundary evidence."""
import json

import cv2 as cv
import numpy as np
import pytest

from kozlov_lab.stereo.boundary_evidence import assess_candidate_boundary

cv.setNumThreads(2)


def rectangle(x0=25, y0=25, x1=75, y1=75):
    mask = np.zeros((100, 100), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def image_for(mask):
    return np.repeat(mask[:, :, None], 3, axis=2)


def assess(image, masks, positive=None, exterior=None):
    return assess_candidate_boundary(image, [{"candidate_id": i, "mask": m} for i, m in enumerate(masks)],
                                     positive or [], exterior or [], cv)


def test_true_edge_and_offset_are_observations_not_selection():
    target = rectangle()
    report = assess(image_for(target), [target, rectangle(35, 35, 65, 65)])
    first, second = report["candidates"]
    assert first["boundary"]["image_edge_alignment"] > .95
    assert second["boundary"]["image_edge_alignment"] < .05
    assert report["status"] == "EVIDENCE_ONLY"
    assert report["selected_candidate_id"] is None
    assert report["physical_identity"] == "UNVERIFIED"
    assert report["publishable_for_geometry"] is False
    json.dumps(report, allow_nan=False)


def test_uniform_no_edges_is_unknown():
    result = assess(np.full((100, 100, 3), 127, np.uint8), [rectangle()])["candidates"][0]
    assert result["boundary"]["status"] == "UNKNOWN"
    assert result["boundary"]["image_edge_alignment"] is None
    assert result["boundary"]["distance_p50_px"] is None
    assert "NO_IMAGE_EDGES" in result["reasons"]


def test_clipped_boundary_retains_full_denominator():
    mask = rectangle(0, 20, 70, 80)
    b = assess(image_for(mask), [mask])["candidates"][0]["boundary"]
    assert b["clipped_px"] > 0
    assert b["measurable_px"] + b["clipped_px"] == b["total_px"]
    assert b["image_edge_alignment_all_boundary"] < b["image_edge_alignment"]


def test_texture_edge_can_align_but_never_implies_semantics():
    image = image_for(rectangle(10, 10, 90, 90))
    image[35:65, 35:65] = 0  # Interior pattern, not outer object silhouette.
    result = assess(image, [rectangle(35, 35, 65, 65)])
    assert result["candidates"][0]["boundary"]["image_edge_alignment"] > .95
    assert result["physical_identity"] == "UNVERIFIED"


def test_positive_and_reference_exterior_conflicts_are_independent():
    positives = [{"track_id": 1, "query_point_xy": [50, 50]}, {"track_id": 2, "query_point_xy": [20, 20]}]
    exterior = [{"witness_id": "outside-ref", "query_point_xy": [65, 65]}]
    reports = assess(image_for(rectangle()), [rectangle(10, 10, 90, 90), rectangle(45, 45, 55, 55), rectangle()], positives, exterior)["candidates"]
    assert reports[0]["positive_evidence"]["inside_count"] == 2
    assert "REFERENCE_EXTERIOR_CONFLICT" in reports[0]["reasons"]
    assert "POSITIVE_POINT_CONFLICT" in reports[1]["reasons"]
    assert {"POSITIVE_POINT_CONFLICT", "REFERENCE_EXTERIOR_CONFLICT"}.issubset(reports[2]["reasons"])


def test_absent_evidence_is_unavailable_not_pass():
    result = assess(image_for(rectangle()), [rectangle()])["candidates"][0]
    for key in ("positive_evidence", "exterior_evidence"):
        assert result[key]["status"] == "UNAVAILABLE"
        assert result[key]["count"] == 0
        assert result[key]["inside_fraction"] is None


def test_entirely_clipped_mask_has_no_measurable_claim():
    mask = rectangle(0, 0, 100, 100)
    image = image_for(rectangle())
    boundary = assess(image, [mask])["candidates"][0]["boundary"]
    assert boundary["measurable_px"] == 0
    assert boundary["status"] == "UNKNOWN"
    assert boundary["image_edge_alignment"] is None


def test_unknown_band_is_exactly_three_pixel_rows_on_every_edge():
    for bounds, unknown_count in [((3, 20, 50, 80), 0), ((2, 20, 50, 80), 60),
                                  ((50, 20, 97, 80), 0), ((50, 20, 98, 80), 60),
                                  ((20, 3, 80, 50), 0), ((20, 2, 80, 50), 60),
                                  ((20, 50, 80, 97), 0), ((20, 50, 80, 98), 60)]:
        mask = rectangle(*bounds)
        boundary = assess(image_for(mask), [mask])["candidates"][0]["boundary"]
        assert boundary["clipped_px"] == unknown_count
        assert boundary["measurable_px"] == boundary["total_px"] - unknown_count


def test_closed_pixel_center_bounds_and_empty_candidates():
    mask = rectangle(0, 0, 100, 100)
    report = assess(image_for(mask), [mask], [{"track_id": 1, "query_point_xy": [99, 99]}])
    assert report["candidates"][0]["positive_evidence"]["points"][0]["sample_xy"] == [99, 99]
    for xy in ([99.01, 99], [99, 99.01]):
        with pytest.raises(ValueError):
            assess(image_for(mask), [mask], [{"track_id": 1, "query_point_xy": xy}])
    empty = assess(image_for(mask), [])
    assert empty["candidates"] == []
    assert empty["reasons"] == ["NO_CANDIDATES"]
    assert empty["selected_candidate_id"] is None


@pytest.mark.parametrize("image", [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8),
                                    np.zeros((100, 100, 3), float), np.zeros((0, 10, 3), np.uint8)])
def test_reject_invalid_images(image):
    with pytest.raises(ValueError):
        assess(image, [rectangle()])


def test_rounding_and_immutability_and_hole_boundary():
    mask = rectangle()
    mask[45:55, 45:55] = 0
    image = image_for(mask)
    before = mask.copy(), image.copy()
    points = [{"track_id": 1, "query_point_xy": [44.49, 50]}, {"track_id": 2, "query_point_xy": [44.5, 50]}]
    report = assess(image, [mask], points)["candidates"][0]
    assert [p["inside"] for p in report["positive_evidence"]["points"]] == [True, False]
    assert report["boundary"]["total_px"] > 196
    np.testing.assert_array_equal(mask, before[0])
    np.testing.assert_array_equal(image, before[1])
    assert points[0]["query_point_xy"] == [44.49, 50]


@pytest.mark.parametrize("xy", [[True, 20], [float("nan"), 20], [float("inf"), 20], [-.1, 20], [100, 20], [20], ["1", 20]])
def test_reject_bad_coordinates(xy):
    with pytest.raises(ValueError):
        assess(image_for(rectangle()), [rectangle()], [{"track_id": 1, "query_point_xy": xy}])


@pytest.mark.parametrize("mutation", ["bool_id", "duplicate_id", "float_mask", "nonbinary", "empty", "shape", "too_many", "bool_track", "duplicate_track", "too_many_points", "bad_witness"])
def test_input_schema(mutation):
    mask = rectangle()
    image = image_for(mask)
    candidates = [{"candidate_id": "c", "mask": mask}]
    positive, exterior = [], []
    if mutation == "bool_id": candidates[0]["candidate_id"] = True
    if mutation == "duplicate_id": candidates *= 2
    if mutation == "float_mask": candidates[0]["mask"] = mask.astype(float)
    if mutation == "nonbinary": candidates[0]["mask"][0, 0] = 1
    if mutation == "empty": candidates[0]["mask"] = np.zeros_like(mask)
    if mutation == "shape": candidates[0]["mask"] = mask[:50]
    if mutation == "too_many": candidates = [{"candidate_id": i, "mask": mask} for i in range(9)]
    if mutation == "bool_track": positive = [{"track_id": True, "query_point_xy": [20, 20]}]
    if mutation == "duplicate_track": positive = [{"track_id": 1, "query_point_xy": [20, 20]}] * 2
    if mutation == "too_many_points": positive = [{"track_id": i, "query_point_xy": [20, 20]} for i in range(769)]
    if mutation == "bad_witness": exterior = [{"witness_id": 1, "query_point_xy": [20, 20]}]
    with pytest.raises(ValueError):
        assess_candidate_boundary(image, candidates, positive, exterior, cv)
