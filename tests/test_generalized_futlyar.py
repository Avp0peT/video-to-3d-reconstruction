from dataclasses import replace
from fractions import Fraction as F

import pytest

from kozlov_lab.kozlov2025.generalized_futlyar import (
    verify_generalized_futlyar_2d as solve,
    verify_generalized_futlyar_certificate as check,
)

P = {7: (0, 0), 9: (F(1, 2), 0), 11: (3, 0)}
C = {20: (0, 0), 30: (3, 0)}


def test_hand_computed_complete_certificate():
    result = solve(P, C, 1)
    assert result.status == "TRUE"
    cert = result.certificate
    assert cert.blocks == ((20, (7, 9)), (30, (11,)))
    assert cert.center_distances_squared == ((20, 30, F(9)),)
    assert cert.distances_squared[1] == (9, ((20, F(1, 4)), (30, F(25, 4))))
    assert check(P, C, 1, cert) == "TRUE"


def test_radius_upper_boundary_is_closed():
    assert solve(P, C, 3).status == "TRUE"
    assert solve(P, C, F(3001, 1000)).reason == "radius exceeds center separation"


def test_point_radius_boundary_is_open_and_exact():
    points = {0: (1, 0), 1: (3, 0)}
    assert solve(points, C, 1).status == "FALSE"
    assert solve(points, C, F(1001, 1000)).status == "TRUE"
    assert solve(points, C, F(999, 1000)).status == "FALSE"


def test_equal_nearest_is_false_inside_radius():
    result = solve({0: (F(3, 2), 0), 1: (3, 0)}, C, 2)
    assert result.status == "FALSE"
    assert result.reason == "nearest center is not unique"


def test_empty_block_despite_all_points_in_radius():
    result = solve({0: (0, 0), 1: (F(1, 2), 0)}, C, 1)
    assert result.reason == "empty block"
    assert result.witness == (30,)


def test_budget_unknown_even_on_failing_geometry():
    result = solve(P, C, 4, max_work=6)
    assert result.status == "UNKNOWN" and result.certificate is None
    assert result.work_required == 7
    assert solve(P, C, 1, max_work=7).status == "TRUE"
    assert check(P, C, 1, solve(P, C, 1).certificate, max_work=6) == "UNKNOWN"


@pytest.mark.parametrize("centers", [{0: (0, 0)}, {0: (0, 0), 1: (0, 0)}])
def test_unresolved_center_domains(centers):
    assert solve(P, centers, 1).status == "UNSUPPORTED"
    assert check(P, centers, 1, None) == "UNSUPPORTED"


@pytest.mark.parametrize("changes", [
    {"blocks": ((20, (7, 9, 11)), (30, ()))},
    {"blocks": ((20, (7, 9)), (30, (11, 11)))},
    {"distances_squared": ()},
    {"center_distances_squared": ((20, 30, F(10)),)},
    {"radius": F(2)}, {"radius": 1.0}, {"version": "other"},
    {"points": ((7, (F(0), F(0))),)},
])
def test_tampered_certificate_rejected(changes):
    cert = replace(solve(P, C, 1).certificate, **changes)
    assert check(P, C, 1, cert) == "FALSE"


def test_certificate_bound_to_complete_external_input():
    cert = solve(P, C, 1).certificate
    assert check({**P, 12: (F(1, 4), 0)}, C, 1, cert) == "FALSE"
    assert check({7: (0, 0), 11: (3, 0)}, C, 1, cert) == "FALSE"
    assert check(P, {20: (0, 0), 30: (4, 0)}, 1, cert) == "FALSE"
    assert check(P, C, 1, {}) == "FALSE"


def test_verifier_does_not_call_producer(monkeypatch):
    import kozlov_lab.kozlov2025.generalized_futlyar as module
    cert = solve(P, C, 1).certificate
    monkeypatch.setattr(module, "verify_generalized_futlyar_2d", lambda *a, **k: None)
    assert check(P, C, 1, cert) == "TRUE"


@pytest.mark.parametrize("points,centers,radius,budget", [
    ({}, C, 1, 100), ({True: (0, 0), 2: (3, 0)}, C, 1, 100),
    ({0: (0., 0), 1: (3, 0)}, C, 1, 100),
    ({0: (0, 0), 1: (0, 0)}, C, 1, 100),
    (P, C, 0, 100), (P, C, -1, 100), (P, C, True, 100),
    (P, C, 1, True), (P, C, 1, 262145),
    ({0: (1 << 256, 0), 1: (3, 0)}, C, 1, 100),
    ({0: (0, 0)}, C, 1, 100),
    ({0: (0, 0, 0), 1: (3, 0)}, C, 1, 100),
])
def test_invalid_input_rejected(points, centers, radius, budget):
    with pytest.raises(ValueError):
        solve(points, centers, radius, max_work=budget)


def test_fraction_near_tie_is_not_float_tie():
    points = {0: (F(3, 2) - F(1, 10**60), 0), 1: (3, 0)}
    assert solve(points, C, 2).status == "TRUE"


def test_certificate_frozen():
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        solve(P, C, 1).certificate.radius = F(2)


def test_pythagorean_radius_boundary_and_nonaxis_points():
    centers = {0: (0, 0), 1: (3, 4)}
    points = {0: (0, 1), 1: (4, 4)}
    result = solve(points, centers, 5)
    assert result.status == "TRUE"
    assert result.certificate.center_distances_squared == ((0, 1, F(25)),)
    assert solve(points, centers, F(5001, 1000)).status == "FALSE"


def test_input_order_irrelevant_and_original_preserved():
    pts = dict(reversed(list(P.items())))
    ctrs = dict(reversed(list(C.items())))
    before = repr((pts, ctrs))
    assert solve(pts, ctrs, 1).certificate == solve(P, C, 1).certificate
    assert repr((pts, ctrs)) == before


def test_size_limits_are_input_rejections():
    with pytest.raises(ValueError):
        solve({i: (i, 0) for i in range(4097)}, C, 1)
    with pytest.raises(ValueError):
        solve({i: (i, 0) for i in range(33)}, {i: (i, 0) for i in range(33)}, 1)
