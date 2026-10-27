"""Counterexamples for the COLMAP<->OpenMVS point-correspondence acceptance rule.

The acceptance rule is: point IDs may be renumbered and geometry may repeat, but a
*perfect bijection* over the whole point set must exist, and every matched pair
must satisfy the XYZ tolerance, an identical RGB triple and an identical
view-member set. A unique correspondence is not required; where identity is not
determined, the ambiguity must be reported instead of being silently resolved.

These tests exercise the pure data layer of ``scripts/prepare_mvs.py``. They never
run InterfaceCOLMAP, pycolmap or any other native process. Every expected value is
written out by hand here, independently of the code under test.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
from pathlib import Path

import pytest

# ``MVS_BRIDGE_MODULE`` lets a negative-control run point the same assertions at a
# deliberately mutated copy of the module, proving these tests can fail.
MODULE_PATH = Path(
    os.environ.get("MVS_BRIDGE_MODULE", str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_mvs.py"))
).resolve()


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_mvs_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = _load_module()

TOL = 1.0
VIEWS_AB = ["images/000000.png", "images/000001.png"]
VIEWS_ABC = ["images/000000.png", "images/000001.png", "images/000002.png"]


def point(pid, xyz, rgb=(10, 20, 30), views=None):
    """Plain record; identical to the shape produced by ``point_records``."""
    return {
        "id": pid,
        "xyz": list(xyz),
        "rgb": list(rgb),
        "views": list(VIEWS_AB if views is None else views),
    }


def naive_first_fit(adjacency):
    """Deliberately simple greedy matching, used only to show it is insufficient."""
    taken, assignment = set(), {}
    for i in sorted(adjacency):
        for j in adjacency[i]:
            if j not in taken:
                taken.add(j)
                assignment[i] = j
                break
    return assignment


def assert_bijection_is_sound(src, dst, result, tolerance):
    """Independent re-check of the witness, written without the module's helpers."""
    witness = result["witness"]
    assert len(witness) == len(src) == len(dst)
    used = set()
    for i, record in enumerate(src):
        partner = witness[str(record["id"])]
        assert partner not in used, f"right point {partner} used twice"
        used.add(partner)
        other = next(r for r in dst if r["id"] == partner)
        assert record["rgb"] == other["rgb"], "witness pairs different colors"
        assert record["views"] == other["views"], "witness pairs different memberships"
        gap = max(abs(a - b) for a, b in zip(record["xyz"], other["xyz"]))
        assert math.isfinite(gap) and gap <= tolerance, f"witness gap {gap} over tolerance"
    assert len(used) == len(dst)


# --------------------------------------------------------------------------- #
# the pure data layer must not pull native dependencies at import time
# --------------------------------------------------------------------------- #
def test_module_imports_no_native_dependency():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module.split(".")[0])
    assert "pycolmap" not in top_level, "pycolmap must be imported on demand, not at module scope"
    assert "numpy" not in top_level, "numpy must be imported on demand, not at module scope"
    # ... and the acceptance functions really are importable data transformations.
    for name in ("point_records", "compatibility_edges", "correspondence"):
        assert callable(getattr(bridge, name))


# --------------------------------------------------------------------------- #
# 1. the ordinary case: every point has exactly one partner
# --------------------------------------------------------------------------- #
def test_unique_one_to_one_matching_is_reported_as_unique():
    src = [point(1, (0, 0, 0)), point(2, (5, 0, 0)), point(3, (0, 5, 0))]
    dst = [point(9, (0, 0, 0)), point(8, (0, 5, 0)), point(7, (5, 0, 0))]
    result = bridge.correspondence(src, dst, TOL)
    assert result["witness"] == {"1": 9, "2": 7, "3": 8}
    assert result["unique_correspondence"] is True
    assert result["ambiguity_groups"] == []
    assert result["ambiguous_source_count"] == 0
    assert_bijection_is_sound(src, dst, result, TOL)


# --------------------------------------------------------------------------- #
# 2. legitimate duplicated geometry: 2 indistinguishable points against 2
# --------------------------------------------------------------------------- #
def test_duplicate_geometry_2_to_2_is_accepted_and_reported_as_ambiguous():
    xyz = (1.0, 2.0, 3.0)
    src = [point(14, xyz), point(15, xyz), point(16, (4.0, 4.0, 4.0))]
    dst = [point(13, (1.0 + 1e-9, 2.0, 3.0)), point(12, (1.0, 2.0, 3.0)), point(11, (4.0, 4.0, 4.0))]
    result = bridge.correspondence(src, dst, TOL)
    # both duplications are legal: the pair set as a whole must be bijective
    assert result["witness"]["16"] == 11
    assert {result["witness"]["14"], result["witness"]["15"]} == {12, 13}
    assert result["unique_correspondence"] is False
    assert result["ambiguous_source_count"] == 2
    assert len(result["ambiguity_groups"]) == 1
    group = result["ambiguity_groups"][0]
    assert group["size"] == 2
    assert group["source_ids"] == [14, 15]
    assert group["roundtrip_ids"] == [12, 13]
    assert_bijection_is_sound(src, dst, result, TOL)


def test_ambiguity_boundary_one_point_of_the_pair_is_forced():
    """A duplicate that shares XYZ, color and views is ambiguous; its neighbour is not."""
    xyz = (0.0, 0.0, 0.0)
    src = [point(1, xyz), point(2, xyz), point(3, (9.0, 0.0, 0.0))]
    dst = [point(101, xyz), point(102, xyz), point(103, (9.0, 0.0, 0.0))]
    result = bridge.correspondence(src, dst, TOL)
    assert result["uniquely_determined_source_count"] == 1  # point 3
    assert result["witness"]["3"] == 103
    assert result["ambiguity_groups"] == [
        {"source_ids": [1, 2], "roundtrip_ids": [101, 102], "size": 2}
    ]
    assert_bijection_is_sound(src, dst, result, TOL)


# --------------------------------------------------------------------------- #
# 3. multiplicity that does not agree
# --------------------------------------------------------------------------- #
def test_total_point_count_mismatch_is_rejected():
    src = [point(1, (0, 0, 0)), point(2, (1, 0, 0))]
    dst = [point(1, (0, 0, 0))]
    with pytest.raises(ValueError, match="point multiplicity changed"):
        bridge.correspondence(src, dst, TOL)


def test_duplicate_multiplicity_mismatch_is_rejected():
    """Three identical source points cannot be matched onto two identical targets."""
    xyz = (0.0, 0.0, 0.0)
    src = [point(1, xyz), point(2, xyz), point(3, xyz)]
    dst = [point(10, xyz), point(11, xyz), point(12, (5.0, 0.0, 0.0))]
    adjacency = bridge.compatibility_edges(src, dst, TOL)
    assert sorted(len(v) for v in adjacency.values()) == [2, 2, 2], "each source sees both duplicates"
    with pytest.raises(ValueError, match="no perfect bijection"):
        bridge.correspondence(src, dst, TOL)


def test_missing_partner_is_rejected():
    src = [point(1, (0, 0, 0)), point(2, (50, 0, 0))]
    dst = [point(10, (0, 0, 0)), point(11, (1, 0, 0))]
    with pytest.raises(ValueError, match="no perfect bijection"):
        bridge.correspondence(src, dst, TOL)


# --------------------------------------------------------------------------- #
# 4. identical XYZ is not enough: color and membership are part of identity
# --------------------------------------------------------------------------- #
def test_same_xyz_different_color_is_not_interchangeable():
    xyz = (0.0, 0.0, 0.0)
    src = [point(1, xyz, rgb=(1, 2, 3)), point(2, xyz, rgb=(4, 5, 6))]
    dst = [point(10, xyz, rgb=(4, 5, 6)), point(11, xyz, rgb=(1, 2, 3))]
    adjacency = bridge.compatibility_edges(src, dst, TOL)
    assert adjacency[0] == [1], "color must restrict the candidate to the same color"
    assert adjacency[1] == [0]
    result = bridge.correspondence(src, dst, TOL)
    assert result["witness"] == {"1": 11, "2": 10}
    assert result["unique_correspondence"] is True

    broken = [point(1, xyz, rgb=(1, 2, 3)), point(2, xyz, rgb=(1, 2, 3))]
    with pytest.raises(ValueError, match="no perfect bijection"):
        bridge.correspondence(broken, dst, TOL)


def test_same_xyz_different_view_membership_is_not_interchangeable():
    xyz = (0.0, 0.0, 0.0)
    src = [point(1, xyz, views=VIEWS_AB), point(2, xyz, views=VIEWS_ABC)]
    dst = [point(10, xyz, views=VIEWS_ABC), point(11, xyz, views=VIEWS_AB)]
    adjacency = bridge.compatibility_edges(src, dst, TOL)
    assert adjacency[0] == [1]
    assert adjacency[1] == [0]
    result = bridge.correspondence(src, dst, TOL)
    assert result["witness"] == {"1": 11, "2": 10}

    broken = [point(1, xyz, views=VIEWS_AB), point(2, xyz, views=VIEWS_AB)]
    with pytest.raises(ValueError, match="no perfect bijection"):
        bridge.correspondence(broken, dst, TOL)


def test_extra_view_on_the_roundtrip_side_is_rejected():
    xyz = (0.0, 0.0, 0.0)
    src = [point(1, xyz, views=VIEWS_AB)]
    dst = [point(10, xyz, views=VIEWS_ABC)]
    adjacency = bridge.compatibility_edges(src, dst, TOL)
    assert adjacency[0] == []
    with pytest.raises(ValueError, match="no perfect bijection"):
        bridge.correspondence(src, dst, TOL)


# --------------------------------------------------------------------------- #
# 5. non-finite coordinates are evidence errors, not matching problems
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_source_coordinate_is_rejected(bad):
    src = [point(1, (bad, 0.0, 0.0))]
    dst = [point(10, (0.0, 0.0, 0.0))]
    with pytest.raises(ValueError, match="non-finite XYZ"):
        bridge.compatibility_edges(src, dst, TOL)


def test_non_finite_roundtrip_coordinate_is_rejected():
    src = [point(1, (0.0, 0.0, 0.0))]
    dst = [point(10, (0.0, float("nan"), 0.0))]
    with pytest.raises(ValueError, match="non-finite XYZ"):
        bridge.correspondence(src, dst, TOL)


def test_non_positive_tolerance_is_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        bridge.compatibility_edges([point(1, (0, 0, 0))], [point(10, (0, 0, 0))], 0.0)


# --------------------------------------------------------------------------- #
# 6. the tolerance itself is honoured at its boundary
# --------------------------------------------------------------------------- #
def test_difference_just_over_tolerance_is_rejected():
    src = [point(1, (0.0, 0.0, 0.0))]
    dst = [point(10, (1.0 + 1e-9, 0.0, 0.0))]
    with pytest.raises(ValueError, match="no perfect bijection"):
        bridge.correspondence(src, dst, TOL)


def test_difference_exactly_at_tolerance_is_accepted():
    src = [point(1, (0.0, 0.0, 0.0))]
    dst = [point(10, (1.0, 0.0, 0.0))]
    result = bridge.correspondence(src, dst, TOL)
    assert result["witness"] == {"1": 10}
    assert_bijection_is_sound(src, dst, result, TOL)


# --------------------------------------------------------------------------- #
# 7. overlapping candidate sets that force a re-assignment
# --------------------------------------------------------------------------- #
def test_overlapping_candidates_require_augmenting_reassignment():
    """Source 10 is flexible, source 20 is restricted to the slot 10 grabs first."""
    src = [point(10, (0.4, 0.0, 0.0)), point(20, (0.0, 0.0, 0.0))]
    dst = [point(100, (0.0, 0.0, 0.0)), point(101, (1.2, 0.0, 0.0))]
    adjacency = bridge.compatibility_edges(src, dst, TOL)
    assert adjacency[0] == [0, 1], "flexible source reaches both targets"
    assert adjacency[1] == [0], "restricted source reaches only the first target"
    # A greedy pass keeps its first choice and strands the restricted source:
    assert naive_first_fit(adjacency) == {0: 0}
    # The real search must re-assign instead of failing.
    result = bridge.correspondence(src, dst, TOL)
    assert result["witness"] == {"10": 101, "20": 100}
    # Source 20 can only take slot 100, so slot 100 is forced and source 10 must be
    # 101: the assignment is unique even though a greedy pass cannot find it.
    assert result["unique_correspondence"] is True
    assert result["ambiguity_groups"] == []
    assert_bijection_is_sound(src, dst, result, TOL)


def test_overlapping_chain_without_a_unique_assignment_still_verifies():
    """Three points with overlapping windows: a bijection exists, identity does not."""
    src = [point(1, (0.0, 0.0, 0.0)), point(2, (0.9, 0.0, 0.0)), point(3, (1.8, 0.0, 0.0))]
    dst = [point(11, (0.0, 0.0, 0.0)), point(12, (0.9, 0.0, 0.0)), point(13, (1.8, 0.0, 0.0))]
    result = bridge.correspondence(src, dst, TOL)
    assert result["unique_correspondence"] is False
    assert result["ambiguity_groups"] == [
        {"source_ids": [1, 2, 3], "roundtrip_ids": [11, 12, 13], "size": 3}
    ]
    assert_bijection_is_sound(src, dst, result, TOL)


# --------------------------------------------------------------------------- #
# 8. every point has a neighbour, yet no perfect matching exists (Hall violation)
# --------------------------------------------------------------------------- #
def test_all_points_have_neighbours_but_no_perfect_matching_exists():
    src = [point(1, (0.0, 0.0, 0.0)), point(2, (0.0, 0.0, 0.0)), point(3, (9.0, 0.0, 0.0))]
    dst = [
        point(11, (0.0, 0.0, 0.0)),
        point(12, (10.0, 0.0, 0.0)),
        point(13, (20.0, 0.0, 0.0)),
    ]
    adjacency = bridge.compatibility_edges(src, dst, TOL)
    assert all(len(v) >= 1 for v in adjacency.values()), "no source is isolated"
    assert adjacency[0] == [0] and adjacency[1] == [0], "two sources share a single slot"
    assert adjacency[2] == [1]
    with pytest.raises(ValueError, match="no perfect bijection"):
        bridge.correspondence(src, dst, TOL)


def test_hall_violation_with_overlapping_multi_candidate_sets():
    """Neighbour sets overlap so that a naive count of points is not enough."""
    src = [point(1, (0.0, 0.0, 0.0)), point(2, (0.5, 0.0, 0.0)), point(3, (0.5, 0.0, 0.0))]
    dst = [point(11, (0.0, 0.0, 0.0)), point(12, (0.5, 0.0, 0.0)), point(13, (30.0, 0.0, 0.0))]
    adjacency = bridge.compatibility_edges(src, dst, TOL)
    assert all(len(v) >= 1 for v in adjacency.values())
    with pytest.raises(ValueError, match="no perfect bijection"):
        bridge.correspondence(src, dst, TOL)


# --------------------------------------------------------------------------- #
# 9. determinism and the "witness is not a recovered identity" contract
# --------------------------------------------------------------------------- #
def test_witness_is_deterministic():
    xyz = (0.0, 0.0, 0.0)
    src = [point(1, xyz), point(2, xyz), point(3, (7.0, 0.0, 0.0))]
    dst = [point(9, xyz), point(8, xyz), point(7, (7.0, 0.0, 0.0))]
    first = bridge.correspondence(src, dst, TOL)
    second = bridge.correspondence(src, dst, TOL)
    assert first["witness"] == second["witness"]
    assert first["ambiguity_groups"] == second["ambiguity_groups"]


def test_reordering_either_side_does_not_change_the_verdict():
    xyz = (0.0, 0.0, 0.0)
    src = [point(1, xyz), point(2, (3.0, 0.0, 0.0)), point(3, (9.0, 0.0, 0.0))]
    dst_a = [point(11, xyz), point(12, (3.0, 0.0, 0.0)), point(13, (9.0, 0.0, 0.0))]
    dst_b = [point(13, (9.0, 0.0, 0.0)), point(11, xyz), point(12, (3.0, 0.0, 0.0))]
    first = bridge.correspondence(src, dst_a, TOL)
    second = bridge.correspondence(src, dst_b, TOL)
    assert first["unique_correspondence"] is True
    assert second["unique_correspondence"] is True
    # identity, not order, decides the pairing
    assert first["witness"] == second["witness"] == {"1": 11, "2": 12, "3": 13}


def test_ambiguity_group_does_not_claim_recovered_identity():
    """The witness is an existential certificate; the group stays explicitly open."""
    xyz = (0.0, 0.0, 0.0)
    src = [point(14, xyz), point(15, xyz)]
    dst = [point(13, xyz), point(12, xyz)]
    result = bridge.correspondence(src, dst, TOL)
    assert result["unique_correspondence"] is False
    assert result["ambiguous_source_count"] == 2
    assert result["allowed_edge_count"] == 4, "every pair is allowed, so every identity is open"
    assert result["ambiguity_groups"] == [
        {"source_ids": [14, 15], "roundtrip_ids": [12, 13], "size": 2}
    ]
    assert_bijection_is_sound(src, dst, result, TOL)
