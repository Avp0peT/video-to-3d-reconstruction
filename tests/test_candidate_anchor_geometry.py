from copy import deepcopy
import math

import pytest

from kozlov_lab.stereo.candidate_anchor_geometry import evaluate_anchor_group


def observations(n=3, depth=5., baseline=1.):
    rows = []
    for i in range(n):
        center = baseline * i
        rows.append(dict(input_index=i, image_id=i + 1, frame_id=f"f_{i}", track_id=7,
                         identity_segment=2, model_key="saved_model_0", raw_pixels=[100 * (.2-center)/depth + 50, 52.],
                         camera_model="SIMPLE_RADIAL", params=[100., 50., 50., 0.],
                         R=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], t=[-center, 0., 0.],
                         source_label="foreground", source_target_verified=True, original_mask_inside=True,
                         strong=True, fb_error_px=.5, lk_error=10.))
    return rows


@pytest.mark.parametrize("n,status", [(2, "TWO_VIEW_PROVISIONAL"), (3, "CONDITIONAL_ANCHOR_SUPPORTED")])
def test_numeric_truth(n, status):
    source = observations(n)
    before = deepcopy(source)
    result = evaluate_anchor_group(source)
    assert result["status"] == status
    assert result["xyz"] == pytest.approx([.2, .1, 5.])
    assert source == before
    assert all(result[k] is False for k in ("publishable_for_geometry", "geometry_accepted", "independent_quality_truth"))
    assert result["scale_status"] == "unknown"
    assert "not_native_feature_index" in result["feature_identity_domain"]


@pytest.mark.parametrize("n", [0, 1])
def test_minimum_views(n):
    result = evaluate_anchor_group(observations(n))
    assert result["status"] == "INSUFFICIENT_ORIGINAL_ANCHORS"
    assert result["anchor_count"] == n


@pytest.mark.parametrize("baseline", [0., .001])
def test_rotation_or_low_parallax_rejected(baseline):
    result = evaluate_anchor_group(observations(baseline=baseline))
    assert result["status"] == "STRICT_GATE_REJECTED"
    assert result["xyz"] is None


def test_bad_depth():
    result = evaluate_anchor_group(observations(depth=-5.))
    assert result["status"] == "STRICT_GATE_REJECTED"
    assert "NON_POSITIVE_DEPTH" in result["reasons"]


def test_pure_rotation_with_distinct_rotations():
    source = observations()
    for i, row in enumerate(source):
        c, s = math.cos(i * .1), math.sin(i * .1)
        row["R"] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
        row["t"] = [0., 0., 0.]
        x, y, z = c * .2 + s * 5., .1, -s * .2 + c * 5.
        row["raw_pixels"] = [100. * x / z + 50., 100. * y / z + 50.]
    before = deepcopy(source)
    result = evaluate_anchor_group(source)
    assert result["status"] == "STRICT_GATE_REJECTED"
    assert source == before


def test_outlier():
    source = observations()
    source[1]["raw_pixels"][1] += 30
    assert evaluate_anchor_group(source)["status"] == "STRICT_GATE_REJECTED"


def test_relaxed_only_never_promoted():
    # Exact geometry at 1.53 degrees passes the existing relaxed angle gate,
    # while failing the frozen strict two-degree requirement.
    result = evaluate_anchor_group(observations(baseline=.067))
    assert result["raw_helper_result"]["tier"] == "RELAXED_SUPPORTED"
    assert result["status"] == "STRICT_GATE_REJECTED"
    assert result["xyz"] is None


@pytest.mark.parametrize("field,value", [
    ("input_index", True), ("image_id", -1), ("track_id", False), ("identity_segment", True),
    ("frame_id", True), ("model_key", ""), ("raw_pixels", [True, 1.]),
    ("t", [0., float("nan"), 0.]), ("params", [100., 50., 50., .1]),
    ("params", [True, 50., 50., 0.]), ("camera_model", None),
    ("camera_model", "PINHOLE"), ("fb_error_px", True), ("fb_error_px", 1.01),
    ("lk_error", 25.01), ("strong", 1), ("source_target_verified", 1),
    ("original_mask_inside", 1), ("source_label", "unknown"),
])
def test_invalid_data_rejected(field, value):
    source = observations()
    source[0][field] = value
    result = evaluate_anchor_group(source)
    assert result["xyz"] is None
    assert result["raw_helper_result"] is None
    assert result["status"] in ("INVALID_ORIGINAL_ANCHORS", "CONDITIONAL_SCOPE_NOT_ORIGINAL_FOREGROUND")


@pytest.mark.parametrize("field,value", [("model_key", "other_model"), ("identity_segment", 3),
                                         ("track_id", 8), ("image_id", 1),
                                         ("frame_id", "f_0"), ("input_index", 0)])
def test_group_identity(field, value):
    source = observations()
    source[1][field] = value
    assert evaluate_anchor_group(source)["status"] == "INVALID_ORIGINAL_ANCHORS"


def test_absent_model_and_boolean_rotation():
    source = observations()
    del source[0]["camera_model"]
    assert evaluate_anchor_group(source)["raw_helper_result"] is None
    source = observations()
    source[0]["R"][0][0] = True
    assert evaluate_anchor_group(source)["raw_helper_result"] is None
