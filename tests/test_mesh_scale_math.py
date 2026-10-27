"""TEST_DATA: pure scale arithmetic regressions; no native engine or CAD output."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "integrations" / "mesh_scale" / "calibration.py"
_SPEC = importlib.util.spec_from_file_location("mesh_scale_math_under_test", _PATH)
scale = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scale)

POINTS = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0)]


def reference(length=2.0, unit="mm", ids=(0, 1), **extra):
    return {"vertex_indices": list(ids), "length": length, "unit": unit, **extra}


@pytest.mark.parametrize("length,unit,expected", [
    (25.4, "cm", 254.0), (25.4, "mm", 25.4), (0.254, "m", 254.0),
    (1.0, "in", 25.4),
], ids=["cm254", "mm254", "m0254", "inch"])
def test_unit_conversion_and_reference_are_explicit(length, unit, expected):
    transform = scale.fit_transform(POINTS, reference(length, unit))
    assert transform["factor_to_mm"] == pytest.approx(expected / 2.0)
    assert transform["origin"] == [0.0, 0.0, 0.0]
    assert transform["origin_vertex"] is None
    assert transform["reference"] == {
        "vertex_indices": [0, 1], "length": length, "unit": unit,
        "distance_model": 2.0, "length_mm": expected,
    }
    converted, report = scale.apply_transform(POINTS, transform)
    assert converted[1] == pytest.approx((expected, 0.0, 0.0))
    assert report["roundtrip_pass"] is True
    assert report["physical_accuracy_assessed"] is False


def test_euclidean_segment_and_matrices_define_positive_uniform_scale():
    points = [(1.0, 2.0, 3.0), (4.0, 6.0, 3.0), (1.0, 2.0, 9.0)]
    transform = scale.fit_transform(points, reference(10.0), origin_vertex=0)
    assert transform["factor_to_mm"] == 2.0
    assert transform["matrix4x4"] == [
        [2.0, 0.0, 0.0, -2.0], [0.0, 2.0, 0.0, -4.0],
        [0.0, 0.0, 2.0, -6.0], [0.0, 0.0, 0.0, 1.0],
    ]
    assert transform["inverse4x4"] == [
        [0.5, 0.0, 0.0, 1.0], [0.0, 0.5, 0.0, 2.0],
        [0.0, 0.0, 0.5, 3.0], [0.0, 0.0, 0.0, 1.0],
    ]
    converted, report = scale.apply_transform(points, transform)
    assert converted == [(0.0, 0.0, 0.0), (6.0, 8.0, 0.0), (0.0, 0.0, 12.0)]
    assert report["max_inverse_abs_error"] == 0.0


def test_vertex_order_duplicate_mesh_vertices_and_input_objects_are_preserved():
    points = POINTS + [POINTS[0], (1.0, 1.0, 1.0)]
    ref = reference(4.0)
    frozen = copy.deepcopy((points, ref))
    transform = scale.fit_transform(points, ref)
    transform_before = copy.deepcopy(transform)
    converted, _ = scale.apply_transform(points, transform)
    assert converted == [(2.0 * x, 2.0 * y, 2.0 * z) for x, y, z in points]
    assert len(converted) == len(points)
    assert (points, ref) == frozen
    assert transform == transform_before


def test_reversed_calibration_segment_retains_original_indices():
    forward = scale.fit_transform(POINTS, reference(10))
    reverse = scale.fit_transform(POINTS, reference(10, ids=(1, 0)))
    assert reverse["factor_to_mm"] == forward["factor_to_mm"]
    assert reverse["reference"]["vertex_indices"] == [1, 0]


def test_origin_changes_translation_without_changing_scale():
    default = scale.fit_transform(POINTS, reference(20))
    shifted = scale.fit_transform(POINTS, reference(20), origin_vertex=1)
    converted, _ = scale.apply_transform(POINTS, shifted)
    assert shifted["factor_to_mm"] == default["factor_to_mm"]
    assert shifted["origin"] == [2.0, 0.0, 0.0]
    assert shifted["origin_vertex"] == 1
    assert converted == [(-20.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-20.0, 30.0, 0.0)]


def test_all_report_dictionaries_survive_json_roundtrip_exactly():
    transform = scale.fit_transform(POINTS, reference(4))
    converted, numeric = scale.apply_transform(POINTS, transform)
    checks = scale.measure_checks(converted, [reference(6, ids=(0, 2))], [0, 1])
    report = {"transform": transform, "numeric": numeric, "checks": checks}
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("length,allowed,expected,signed", [
    (3.0, 0.0, True, 0.0), (4.0, 1.0, True, -1.0),
    (2.0, 1.0, True, 1.0), (4.0, 0.5, False, -1.0),
], ids=["exact", "equal-minus", "equal-plus", "outside"])
def test_independent_threshold_is_absolute_inclusive_and_signed(length, allowed, expected, signed):
    check = reference(length, ids=(0, 2), allowed_absolute_error_mm=allowed)
    result, = scale.measure_checks(POINTS, [check], [0, 1])
    assert result["model_distance_mm"] == 3.0
    assert result["reference_length_mm"] == length
    assert result["signed_error_mm"] == signed
    assert result["absolute_error_mm"] == abs(signed)
    assert result["relative_error"] == abs(signed) / length
    assert result["allowed_absolute_error_mm"] == allowed
    assert result["within_threshold"] is expected


def test_missing_threshold_does_not_claim_pass_or_fail():
    result, = scale.measure_checks(POINTS, [reference(100, ids=(2, 0))], [0, 1])
    assert result["vertex_indices"] == [2, 0]
    assert result["allowed_absolute_error_mm"] is None
    assert result["within_threshold"] is None
    assert scale.measure_checks(POINTS, [], [0, 1]) == []


def test_check_unit_conversion_and_inputs_are_preserved():
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 25.4, 0.0)]
    checks = [reference(1, "in", ids=(0, 2), allowed_absolute_error_mm=0)]
    frozen = copy.deepcopy((points, checks))
    result, = scale.measure_checks(points, checks, [0, 1])
    assert result["reference_length_mm"] == 25.4
    assert result["within_threshold"] is True
    assert (points, checks) == frozen


@pytest.mark.parametrize("ids", [(0, 1), (1, 0)], ids=["forward", "reverse"])
def test_calibration_pair_is_never_an_independent_check(ids):
    with pytest.raises(ValueError, match="calibration"):
        scale.measure_checks(POINTS, [reference(ids=ids)], [1, 0])


@pytest.mark.parametrize("ids", [(0, 2), (2, 0)], ids=["same", "reverse"])
def test_duplicate_independent_pairs_rejected(ids):
    with pytest.raises(ValueError, match="duplicate"):
        scale.measure_checks(POINTS, [reference(ids=(0, 2)), reference(ids=ids)], [0, 1])


@pytest.mark.parametrize("ids", [(0, 0), (True, 1), (0, 1.0), (-1, 1), (0, 3), (0,), None],
                         ids=["same", "bool", "float", "negative", "range", "short", "null"])
def test_invalid_endpoint_indices_rejected(ids):
    ref = reference()
    ref["vertex_indices"] = ids
    with pytest.raises(ValueError):
        scale.fit_transform(POINTS, ref)
    with pytest.raises(ValueError):
        scale.measure_checks(POINTS, [ref], [0, 1])
    with pytest.raises(ValueError):
        scale.measure_checks(POINTS, [], ids)


@pytest.mark.parametrize("origin", [True, 1.0, -1, 3, "0"],
                         ids=["bool", "float", "negative", "range", "text"])
def test_invalid_origin_index_rejected(origin):
    with pytest.raises(ValueError):
        scale.fit_transform(POINTS, reference(), origin_vertex=origin)


@pytest.mark.parametrize("length", [0, -1, True, float("nan"), float("inf"), "2", 10 ** 400],
                         ids=["zero", "negative", "bool", "nan", "inf", "text", "hugeint"])
def test_invalid_reference_and_check_lengths_rejected(length):
    with pytest.raises(ValueError):
        scale.fit_transform(POINTS, reference(length))
    with pytest.raises(ValueError):
        scale.measure_checks(POINTS, [reference(length, ids=(0, 2))], [0, 1])


@pytest.mark.parametrize("unit", ["MM", "inch", "ft", "", None, True, []],
                         ids=["uppercase", "word", "feet", "empty", "null", "bool", "list"])
def test_unsupported_units_rejected(unit):
    with pytest.raises(ValueError):
        scale.fit_transform(POINTS, reference(unit=unit))


@pytest.mark.parametrize("allowed", [-1, True, float("inf"), float("nan"), None, "1"],
                         ids=["negative", "bool", "inf", "nan", "null", "text"])
def test_invalid_check_thresholds_rejected(allowed):
    with pytest.raises(ValueError):
        scale.measure_checks(POINTS, [reference(ids=(0, 2), allowed_absolute_error_mm=allowed)], [0, 1])


@pytest.mark.parametrize("bad", [(0, 0), (0, 0, 0, 0), (True, 0, 0), ("0", 0, 0),
                                  (float("nan"), 0, 0), (0, float("inf"), 0)],
                         ids=["short", "long", "bool", "text", "nan", "inf"])
def test_every_point_is_validated_even_if_not_a_measurement_endpoint(bad):
    transform = scale.fit_transform(POINTS, reference())
    points = POINTS + [bad]
    with pytest.raises(ValueError):
        scale.fit_transform(points, reference())
    with pytest.raises(ValueError):
        scale.apply_transform(points, transform)
    with pytest.raises(ValueError):
        scale.measure_checks(points, [], [0, 1])


@pytest.mark.parametrize("points", [[], [(0, 0, 0)], None, "points"],
                         ids=["empty", "one", "null", "text"])
def test_insufficient_or_malformed_point_sets_rejected(points):
    transform = scale.fit_transform(POINTS, reference())
    with pytest.raises(ValueError):
        scale.fit_transform(points, reference())
    with pytest.raises(ValueError):
        scale.apply_transform(points, transform)
    with pytest.raises(ValueError):
        scale.measure_checks(points, [], [0, 1])


def test_distinct_indices_with_same_coordinates_do_not_define_a_length():
    points = POINTS + [POINTS[0]]
    with pytest.raises(ValueError, match="different coordinates"):
        scale.fit_transform(points, reference(ids=(0, 3)))
    with pytest.raises(ValueError, match="different coordinates"):
        scale.measure_checks(points, [reference(ids=(0, 3))], [0, 1])


@pytest.mark.parametrize("distance,length,unit", [
    (1e-308, 1e308, "mm"), (1e308, 1e-308, "mm"),
    (1.0, math.ulp(0.0), "mm"), (1.0, 1e308, "m"),
], ids=["scale-over", "scale-under", "inverse-over", "unit-over"])
def test_nonrepresentable_factors_and_units_rejected(distance, length, unit):
    with pytest.raises(ValueError):
        scale.fit_transform([(0, 0, 0), (distance, 0, 0)], reference(length, unit))


def test_endpoint_difference_overflow_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        scale.fit_transform([(-1e308, 0, 0), (1e308, 0, 0)], reference())


@pytest.mark.parametrize("origin,length", [(1e308, 2.0), (math.ulp(0.0), 0.5)],
                         ids=["over", "under"])
def test_matrix_translation_must_be_representable(origin, length):
    with pytest.raises(ValueError):
        scale.fit_transform([(0, 0, 0), (1, 0, 0), (origin, 0, 0)], reference(length), origin_vertex=2)


def test_nonreference_vertex_output_overflow_rejected():
    points = [(0, 0, 0), (1, 0, 0), (1e308, 0, 0)]
    transform = scale.fit_transform(points, reference(2))
    with pytest.raises(ValueError, match="transformed coordinate"):
        scale.apply_transform(points, transform)


def test_centering_overflow_rejected():
    points = [(1e308, 0, 0), (1e308, 1, 0), (-1e308, 0, 0)]
    transform = scale.fit_transform(points, reference(1), origin_vertex=0)
    with pytest.raises(ValueError, match="origin-relative coordinate"):
        scale.apply_transform(points, transform)


def test_underflow_cannot_silently_collapse_nonzero_geometry():
    points = [(0, 0, 0), (1, 0, 0), (math.ulp(0.0), 0, 0)]
    transform = scale.fit_transform(points, reference(0.5))
    with pytest.raises(ValueError, match="underflow"):
        scale.apply_transform(points, transform)


def test_roundtrip_budget_tracks_local_shape_not_large_world_origin():
    points = [(1e9, 1e9, 1e9), (1e9 + 1, 1e9, 1e9), (1e9 + 2, 1e9, 1e9)]
    transform = scale.fit_transform(points, reference(3), origin_vertex=0)
    converted, report = scale.apply_transform(points, transform)
    assert converted == [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (6.0, 0.0, 0.0)]
    assert report["max_inverse_abs_error"] == 0.0
    assert report["numeric_tolerance"]["max_inverse_absolute_budget"] < 1e-12
    assert report["numeric_tolerance"]["global_maximum_used_as_gate"] is False


def test_distant_origin_cannot_hide_loss_of_small_source_coordinate():
    points = [(1e16, 0, 0), (1e16, 2, 0), (1, 0, 0)]
    transform = scale.fit_transform(points, reference(2), origin_vertex=0)
    with pytest.raises(ValueError, match="roundtrip"):
        scale.apply_transform(points, transform)


def test_distinct_xyz_vertices_cannot_collapse_even_within_error_budget():
    points = [(0, 0, 0), (1e16, 0, 0), (2 ** 53 - 2, 0, 0), (2 ** 53 - 1, 0, 0)]
    transform = scale.fit_transform(points, reference(1e16 * math.nextafter(1.0, math.inf)))
    with pytest.raises(ValueError, match="collapsed"):
        scale.apply_transform(points, transform)


def test_near_zero_source_coordinate_can_round_normally_after_translation():
    points = [(0, 0, 1), (1, 0, 1), (0.5, 0, 6.123233995736766e-17)]
    transform = scale.fit_transform(points, reference(50), origin_vertex=0)
    converted, report = scale.apply_transform(points, transform)
    assert len(set(converted)) == len(points)
    assert 0.0 < report["max_inverse_abs_error"] < 1e-15
    assert report["max_origin_relative_inverse_abs_error"] == 0.0
    assert report["roundtrip_policy"] == "per_component_origin_relative_ulp_v2"
    assert report["numeric_tolerance"]["reference_distance_model"] == 1.0
    assert report["numeric_tolerance"]["reference_relative_cap"] == 1e-12
    assert report["numeric_tolerance"]["reference_absolute_cap"] == 1e-12


def test_one_component_collision_does_not_reject_distinct_xyz_vertices():
    points = [(0, 0, 0), (1, 0, 0),
              (-0.1294095225512603, -0.4829629131445341, 0.8660254037844387),
              (0.12940952255126012, -0.48296291314453416, 0.8660254037844387)]
    transform = scale.fit_transform(points, reference(50))
    converted, report = scale.apply_transform(points, transform)
    assert points[2][1] != points[3][1]
    assert converted[2][1] == converted[3][1]
    assert converted[2] != converted[3]
    assert report["max_inverse_abs_error"] <= report["numeric_tolerance"]["reference_absolute_cap"]
    assert report["distinct_vertex_collapse_rejected"] is True
    assert report["collapse_check_scope"] == "complete_xyz"


@pytest.mark.parametrize("key,value", [("distance_model", 1e30), ("distance_model", True),
                                        ("distance_model", float("nan")), ("length_mm", 1e30)],
                         ids=["distance", "bool", "nan", "length"])
def test_reference_budget_metadata_cannot_be_forged(key, value):
    transform = scale.fit_transform(POINTS, reference())
    transform["reference"][key] = value
    with pytest.raises(ValueError):
        scale.apply_transform(POINTS, transform)


def test_reference_error_cap_is_recomputed_from_actual_input_vertices():
    transform = scale.fit_transform(POINTS, reference())
    changed_points = [POINTS[0], (3.0, 0.0, 0.0), POINTS[2]]
    with pytest.raises(ValueError, match="reference distance"):
        scale.apply_transform(changed_points, transform)


@pytest.mark.parametrize("key", ["matrix4x4", "inverse4x4"], ids=["forward", "inverse"])
def test_transform_matrices_cannot_disagree_with_parameters(key):
    transform = scale.fit_transform(POINTS, reference())
    transform[key][0][3] += 1.0
    with pytest.raises(ValueError, match="disagrees"):
        scale.apply_transform(POINTS, transform)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, 0.0, -1.0],
                         ids=["nan", "inf", "bool", "zero", "negative"])
def test_untrusted_transform_scale_is_validated(bad):
    transform = scale.fit_transform(POINTS, reference())
    transform["factor_to_mm"] = bad
    with pytest.raises(ValueError):
        scale.apply_transform(POINTS, transform)


def test_check_relative_error_overflow_is_rejected():
    with pytest.raises(ValueError, match="relative error"):
        scale.measure_checks(POINTS, [reference(math.ulp(0.0), ids=(0, 2))], [0, 1])
