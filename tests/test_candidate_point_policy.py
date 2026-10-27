"""Pure JSON policy checks using the saved boundary schema (no model imports)."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from kozlov_lab.stereo.candidate_point_policy import decide_candidate_points


def row():
    positive = {"track_id": 66, "query_point_xy": [3.2, 8.7], "before_cell_id": "r2c2", "after_cell_id": "r1c2"}
    exterior = {"witness_id": "80_82:0", "query_point_xy": [1., 5.],
                "reference_exterior_witness": True, "verified_background": False,
                "selection_condition": "RETROSPECTIVE_SURVIVOR_CONDITIONED"}
    def membership(point, inside):
        p = deepcopy(point)
        p = {k: p[k] for k in ("track_id" if "track_id" in p else "witness_id", "query_point_xy")}
        p.update(sample_xy=[3, 9] if "track_id" in p else [1, 5], inside=inside)
        return dict(count=1, inside_count=int(inside), inside_fraction=float(inside), status="OBSERVED", points=[p])
    return dict(input_index=81, frame_id="f_fixture", original_status="NEEDS_SELECTION", before_index=80, after_index=82,
                positive_evidence=[positive], exterior_evidence=[exterior], exterior_reason="RETROSPECTIVE_SURVIVOR_CONDITIONED",
                candidates=[dict(candidate_id="c00"), dict(candidate_id="c01")], status="EVIDENCE_ONLY",
                selected_candidate_id=None, publishable_for_geometry=False,
                assessment=dict(status="EVIDENCE_ONLY", selected_candidate_id=None, publishable=False,
                                publishable_for_geometry=False, candidates=[
                                    dict(candidate_id="c00", positive_evidence=membership(positive, False),
                                         exterior_evidence=membership(exterior, False), reasons=["POSITIVE_POINT_CONFLICT"]),
                                    dict(candidate_id="c01", positive_evidence=membership(positive, True),
                                         exterior_evidence=membership(exterior, False), reasons=["CLIPPED_BOUNDARY_UNKNOWN"])]))


def test_positive_conflict_and_scope_and_unchanged():
    source = row()
    before = deepcopy(source)
    result = decide_candidate_points(source)
    assert source == before
    assert result["status"] == "PROVISIONAL_POINT_SUPPORT_ONLY"
    assert result["provisional_candidate_id"] == "c01"
    assert result["candidate_results"][0]["conflicts"] == ["POSITIVE_POINT_CONFLICT"]
    assert result["supported_observations"] == source["positive_evidence"]
    result["supported_observations"][0]["track_id"] = 99
    assert source == before
    assert result["scope"] == "EXISTING_MEASURED_POINTS_ONLY"
    assert result["clipped_boundary_status"] == "UNKNOWN"
    assert all(result[k] is False for k in ("full_mask_accepted", "new_point_seeding_allowed", "publishable_for_geometry"))


def test_nested_both_compatible_ambiguous_and_scores_ignored():
    source = row()
    first, second = source["assessment"]["candidates"]
    first["positive_evidence"] = deepcopy(second["positive_evidence"])
    first["reasons"] = []
    first.update(area_px=99999, score=1., boundary={"image_edge_alignment": 1.})
    second.update(area_px=1, score=0., boundary={"image_edge_alignment": 0.})
    assert decide_candidate_points(source)["status"] == "AMBIGUOUS"
    first["positive_evidence"] = row()["assessment"]["candidates"][0]["positive_evidence"]
    first["reasons"] = ["POSITIVE_POINT_CONFLICT"]
    assert decide_candidate_points(source)["provisional_candidate_id"] == "c01"


def test_exterior_conflict_and_identity_hold():
    source = row()
    second = source["assessment"]["candidates"][1]
    evidence = second["exterior_evidence"]
    evidence.update(inside_count=1, inside_fraction=1.)
    evidence["points"][0]["inside"] = True
    second["reasons"].append("REFERENCE_EXTERIOR_CONFLICT")
    assert decide_candidate_points(source)["status"] == "REFERENCE_CONFLICT_HOLD"
    source["exterior_evidence"][0]["verified_background"] = True
    assert decide_candidate_points(source)["status"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize("kind", ["positive_evidence", "exterior_evidence"])
def test_no_evidence(kind):
    source = row()
    source[kind] = []
    for candidate in source["assessment"]["candidates"]:
        candidate[kind] = dict(count=0, inside_count=0, inside_fraction=None, status="UNAVAILABLE", points=[])
        if kind == "positive_evidence":
            candidate["reasons"] = [r for r in candidate["reasons"] if r != "POSITIVE_POINT_CONFLICT"]
    assert decide_candidate_points(source)["status"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("mutation", [
    lambda r: r["candidates"][1].update(candidate_id="c00"),
    lambda r: r.update(input_index=True),
    lambda r: r["positive_evidence"][0].update(track_id=True),
    lambda r: r["positive_evidence"][0].update(query_point_xy=[float("nan"), 1]),
    lambda r: r["positive_evidence"][0].update(query_point_xy=[True, 1]),
    lambda r: r["assessment"]["candidates"][1]["positive_evidence"].update(count=2),
    lambda r: r["assessment"]["candidates"][1]["positive_evidence"].update(inside_count=True),
    lambda r: r["assessment"]["candidates"][1]["positive_evidence"]["points"][0].update(inside=1),
    lambda r: r["assessment"]["candidates"][1]["positive_evidence"]["points"][0].update(sample_xy=[True, 9]),
    lambda r: r["assessment"]["candidates"][0].update(reasons=[]),
    lambda r: r["assessment"]["candidates"][1].update(candidate_id="missing"),
    lambda r: r["assessment"].update(status="ACCEPTED"),
])
def test_tampering_fails_closed(mutation):
    source = row()
    mutation(source)
    result = decide_candidate_points(source)
    assert result["status"] == "INVALID_EVIDENCE"
    assert result["provisional_candidate_id"] is None
    assert result["supported_observations"] == []


@pytest.mark.parametrize("original, expected", [("NO_MATCH", "NO_CANDIDATE"), ("OK", "ORIGINAL_OK_UNCHANGED")])
def test_original_status_preserved(original, expected):
    source = row()
    source["original_status"] = original
    result = decide_candidate_points(source)
    assert result["status"] == expected
    assert result["original_status"] == original
    assert result["provisional_candidate_id"] is None


def test_both_cover_positive_one_exterior_conflict_does_not_select_other():
    source = row()
    first, second = source["assessment"]["candidates"]
    first["positive_evidence"] = deepcopy(second["positive_evidence"])
    first["exterior_evidence"].update(inside_count=1, inside_fraction=1.)
    first["exterior_evidence"]["points"][0]["inside"] = True
    first["reasons"] = ["REFERENCE_EXTERIOR_CONFLICT"]
    result = decide_candidate_points(source)
    assert result["status"] == "REFERENCE_CONFLICT_HOLD"
    assert result["provisional_candidate_id"] is None
    assert result["supported_observations"] == []
    assert result["candidate_results"][0]["point_status"] == "REFERENCE_EXTERIOR_HELD"


def test_positive_rejected_with_exterior_conflict_does_not_hold():
    source = row()
    first = source["assessment"]["candidates"][0]
    first["exterior_evidence"].update(inside_count=1, inside_fraction=1.)
    first["exterior_evidence"]["points"][0]["inside"] = True
    first["reasons"].append("REFERENCE_EXTERIOR_CONFLICT")
    result = decide_candidate_points(source)
    assert result["status"] == "PROVISIONAL_POINT_SUPPORT_ONLY"
    assert result["candidate_results"][0]["point_status"] == "POSITIVE_REJECTED"


@pytest.mark.parametrize("original", ["OK", "NO_MATCH"])
@pytest.mark.parametrize("field,value", [("input_index", True), ("input_index", -1),
                                         ("frame_id", True), ("frame_id", ""),
                                         ("before_index", False), ("after_index", -1)])
def test_preserved_status_still_validates_identity(original, field, value):
    source = row()
    source["original_status"] = original
    source[field] = value
    result = decide_candidate_points(source)
    assert result["status"] == "INVALID_EVIDENCE"
    assert result["original_status"] == original


def test_saved_frame81_schema():
    path = Path(__file__).resolve().parents[1] / "outputs/delegation/PPP_VIDEO/20261014_object_first01/boundary_support_20261017/real01/boundary_support/DECISIONS.json"
    if not path.exists():
        pytest.skip("Local immutable real evidence fixture is not distributed")
    source = next(r for r in json.loads(path.read_text(encoding="utf-8")) if r["input_index"] == 81)
    result = decide_candidate_points(source)
    assert result["status"] == "PROVISIONAL_POINT_SUPPORT_ONLY"
    assert result["provisional_candidate_id"] == "c01"
    assert result["positive_evidence_count"] == 3
