"""TEST_DATA only: pure selection policy, no images, models, GPU, or native run."""
from copy import deepcopy
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.video_masks.object_region_policy import POLICY, select_intermediates


def timeline(decode_ids=range(13), endpoint_ids=(0, 6, 12), offset=0, name="test object"):
    endpoints = {decode: index for index, decode in enumerate(endpoint_ids)}
    return [dict(decode_index=decode + offset, object_name=name,
                 **({"input_index": endpoints[decode] + offset} if decode in endpoints else {}))
            for decode in decode_ids]


def model(mid, views, status="CAMERA_DOMAIN_VALID", label="new"):
    return dict(model_id=mid, registered_indices=views, camera_domain_status=status, label=label)


def test_translation_and_object_name_invariance_and_no_input_mutation():
    models = [model(2, [0]), model(8, [1, 2])]
    source = timeline(name="ceramic vessel")
    before = deepcopy((source, models))
    chosen, ledger = select_intermediates(source, models)
    shifted = [model(102, [100]), model(108, [101, 102])]
    other, other_ledger = select_intermediates(timeline(offset=100, name="mechanical housing"), shifted)
    assert chosen == [2, 4]
    assert [value - 100 for value in other] == chosen
    assert [g["reason"] for g in ledger["gaps"]] == [g["reason"] for g in other_ledger["gaps"]]
    assert (source, models) == before
    assert ledger["policy"] == POLICY
    assert ledger["quality_status"] == "NOT_EVALUATED"
    assert ledger["geometry_executed"] is False
    assert ledger["resource_limits_are_quality_criteria"] is False
    json.dumps(ledger)


def test_irregular_nonconsecutive_decode_uses_actual_nearest_rows():
    rows = timeline([10, 12, 17, 25, 38], (10, 38))
    chosen, ledger = select_intermediates(rows, [model(0, [0]), model(1, [1])])
    assert chosen == [17, 25]
    assert ledger["coordinate"] == "decode_index"


def test_irregular_timestamp_coordinates_change_nearest_selection():
    rows = timeline([0, 1, 2, 3, 4, 5], (0, 5))
    for row, time in zip(rows, [0, 3.1, 4.1, 6.8, 9.9, 10]):
        row["timestamp_seconds"] = time
    chosen, ledger = select_intermediates(rows, [model(0, [0]), model(1, [1])])
    assert chosen == [1, 3]
    assert ledger["coordinate"] == "timestamp_seconds"


def test_integer_coordinates_do_not_lose_precision_or_overflow():
    offset = 10 ** 400
    rows = timeline([0, 2, 4, 6], (0, 6), offset=offset)
    for row in rows:
        row["timestamp_seconds"] = row["decode_index"]
    chosen, _ = select_intermediates(rows, [model(0, [offset]), model(1, [offset + 1])])
    assert chosen == [offset + 2, offset + 4]


def test_one_candidate_is_never_reused_for_second_third():
    rows = timeline([0, 5, 9], (0, 9))
    chosen, ledger = select_intermediates(rows, [model(0, [0]), model(1, [1])])
    assert chosen == [5]
    assert [d["decision"] for d in ledger["decisions"]] == ["SELECTED", "NO_UNUSED_INTERMEDIATE_FRAME"]


def test_tie_breaks_by_lower_decode_and_selects_next_unused_candidate():
    rows = timeline([0, 2, 4, 8, 9], (0, 9))
    chosen, _ = select_intermediates(rows, [model(0, [0]), model(1, [1])])
    assert chosen == [2, 4]


def test_same_component_gap_proposes_every_constituent_original_interval():
    rows = timeline()
    chosen, ledger = select_intermediates(rows, [model(7, [0, 2])])
    assert chosen == [2, 4, 8, 10]
    assert ledger["gaps"][0]["reason"] == "SAME_COMPONENT_ENCLOSES_UNREGISTERED_ENDPOINTS"
    assert ledger["unregistered_original_positions"] == [1]
    assert ledger["group_by_original_position"][1] is None


def test_fully_registered_single_component_has_no_eligible_gap():
    chosen, ledger = select_intermediates(timeline(), [model(7, [0, 1, 2])])
    assert chosen == []
    assert ledger["reasons"] == ["NO_ELIGIBLE_GAPS"]


def test_shorter_span_priority_and_determinism_under_model_reordering():
    rows = timeline(range(25), (0, 6, 12, 18, 24))
    models = [model(2, [0]), model(3, [3]), model(4, [4])]
    chosen, ledger = select_intermediates(rows, models, policy={"max_added": 2})
    assert chosen == [20, 22]
    assert [g["span"] for g in ledger["gaps"]] == [1, 3]
    assert select_intermediates(rows, list(reversed(models)), policy={"max_added": 2}) == (chosen, ledger)


def test_largest_valid_assignment_overlaps_ignore_unknown_and_rejected():
    models = [model(9, [0]), model(5, [0, 1]), model(2, [1, 2]),
              model(0, [0, 1, 2], status="UNKNOWN"),
              model(1, [0, 1, 2], status="CAMERA_DOMAIN_REJECTED")]
    _, ledger = select_intermediates(timeline(), models)
    assert ledger["valid_model_order"] == [2, 5, 9]
    assert ledger["group_by_original_position"] == {0: 5, 1: 2, 2: 2}
    assert [m["model_id"] for m in ledger["rejected_models"]] == [0, 1]


@pytest.mark.parametrize("status", ["UNKNOWN", "NOT_EVALUATED", "CAMERA_DOMAIN_REJECTED", "unexpected"])
def test_nonvalid_status_never_creates_registration(status):
    chosen, ledger = select_intermediates(timeline(), [model(0, [0, 1, 2], status)])
    assert chosen == []
    assert ledger["unregistered_original_positions"] == [0, 1, 2]
    assert "NO_VALID_CAMERA_DOMAIN_MODEL" in ledger["reasons"]


@pytest.mark.parametrize("rows,models,reason", [
    ([], [], "INSUFFICIENT_ENDPOINT_DOMAIN"),
    ([{"decode_index": 3}], [], "INSUFFICIENT_ENDPOINT_DOMAIN"),
    (timeline(), [model(0, [0])], "INSUFFICIENT_REGISTERED_ENDPOINTS"),
    (timeline([0, 6], (0, 6)), [model(0, [0]), model(1, [1])], "NO_AVAILABLE_INTERMEDIATE_FRAMES"),
])
def test_insufficient_domains_report_reasons(rows, models, reason):
    chosen, ledger = select_intermediates(rows, models)
    assert chosen == []
    assert reason in ledger["reasons"]
    assert ledger["status"] == "NO_CANDIDATES"


@pytest.mark.parametrize("endpoint_count", [2, 5, 13, 37, 94, 95, 96, 97])
def test_different_video_lengths_respect_both_default_resource_caps(endpoint_count):
    rows = timeline(range((endpoint_count - 1) * 3 + 1), tuple(range(0, endpoint_count * 3, 3)))
    models = [model(index, [index]) for index in range(endpoint_count)]
    chosen, ledger = select_intermediates(rows, models)
    assert len(chosen) == min(24, max(0, 96 - endpoint_count), 2 * (endpoint_count - 1))
    assert len(chosen) == len(set(chosen))
    assert not set(chosen) & {row["decode_index"] for row in rows if "input_index" in row}
    if endpoint_count > 96:
        assert ledger["resource_limit_satisfied"] is False
        assert "EXISTING_ENDPOINTS_EXCEED_NATIVE_RESOURCE_LIMIT" in ledger["reasons"]
    else:
        assert ledger["resource_limit_satisfied"] is True


def test_resource_caps_can_be_tightened_and_zero_added_disables_proposal():
    models = [model(0, [0]), model(1, [1]), model(2, [2])]
    chosen, ledger = select_intermediates(timeline(), models, policy={"native_max_frames": 4})
    assert chosen == [2]
    assert ledger["total_frame_count"] == 4
    chosen, ledger = select_intermediates(timeline(), models, policy={"max_added": 0})
    assert chosen == []
    assert ledger["reasons"] == ["RESOURCE_BUDGET_EXHAUSTED"]


def test_all_unlabelled_models_are_explicit_caller_filtered_set():
    models = [model(0, [0]), model(1, [1, 2])]
    for item in models:
        item.pop("label")
    chosen, ledger = select_intermediates(timeline(), models)
    assert chosen == [2, 4]
    assert ledger["model_set"] == "CALLER_FILTERED"


def test_bridge_label_is_accepted_independently():
    chosen, ledger = select_intermediates(timeline(), [model(0, [0], label="bridge"), model(1, [1, 2], label="bridge")])
    assert chosen == [2, 4]
    assert ledger["model_set"] == "bridge"


@pytest.mark.parametrize("policy", [True, [], {"unexpected": 2}, {"schema": "bad"},
    {"native_max_frames": 97}, {"max_added": 25}, {"native_max_frames": 0},
    {"max_added": -1}, {"max_added": True}, {"native_max_frames": 2.0}])
def test_reject_invalid_or_loosened_policy(policy):
    with pytest.raises(ValueError):
        select_intermediates([], [], policy=policy)


@pytest.mark.parametrize("rows", [
    None, {}, "bad", [1], [{}], [{"decode_index": True}], [{"decode_index": -1}],
    [{"decode_index": 1.0}], [{"decode_index": "1"}],
    [{"decode_index": 1}, {"decode_index": 1}],
    [{"decode_index": 2}, {"decode_index": 1}],
    [{"decode_index": 0, "input_index": True}],
    [{"decode_index": 0, "input_index": 2}, {"decode_index": 1, "input_index": 1}],
    [{"decode_index": 0, "input_index": 1}, {"decode_index": 1, "input_index": 1}],
    [{"decode_index": 0, "timestamp_seconds": True}],
    [{"decode_index": 0, "timestamp_seconds": float("nan")}],
    [{"decode_index": 0, "timestamp_seconds": float("inf")}],
    [{"decode_index": 0, "timestamp_seconds": -0.1}],
    [{"decode_index": 0, "timestamp_seconds": 0}, {"decode_index": 1}],
    [{"decode_index": 0, "timestamp_seconds": 0}, {"decode_index": 1, "timestamp_seconds": 0}],
])
def test_reject_malformed_timeline(rows):
    with pytest.raises(ValueError):
        select_intermediates(rows, [])


@pytest.mark.parametrize("models", [
    None, {}, "bad", [1], [{}], [model(True, [0])], [model(-1, [0])],
    [model(0, [True])], [model(0, [0.0])], [model(0, [0, 0])], [model(0, [1, 0])],
    [model(0, [9])], [model(0, [9], status="UNKNOWN")], [model(0, "0")],
    [model(0, [0], status=True)], [model(0, [0], status="")],
    [model(0, [0]), model(0, [1])], [model(0, [0], label="old")],
    [model(0, [0]), model(1, [1], label="old")],
    [model(0, [0]), model(1, [1], label="bridge")],
    [model(0, [0]), model(1, [1], label=None)],
    [model(0, [0], label=True)],
])
def test_reject_malformed_models_or_mixed_domains(models):
    with pytest.raises(ValueError):
        select_intermediates(timeline(), models)
