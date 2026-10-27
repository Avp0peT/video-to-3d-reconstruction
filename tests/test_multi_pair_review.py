"""Independent review counterexamples for preserving stage attribution."""
import numpy as np
from integrations.demo_pipeline import multi_pair_ablation as batch
from kozlov_lab.stereo.baseline import rectified_ablation as ab
from kozlov_lab.stereo.baseline.rectified_parallel import calibrated_rectification


def test_construction_count_survives_verifier_budget(monkeypatch):
    k = np.array(((700, 0, 320), (0, 700, 240), (0, 0, 1)), float)
    rec = calibrated_rectification(k, np.eye(3), [0, 0, 0], k, np.eye(3), [-1, 0, 0])
    left = np.array(((.125,.125),(.25,.125),(.125,.25),(.3,.3),(.4,.2),(.2,.4)))
    left = np.round(left*262144)/262144
    right = left.copy()
    right[:, 0] -= np.array((.125,.125,.125,.25,.0625,.1875))
    monkeypatch.setattr(ab.given, 'assess_given_parallel_certificate',
                        lambda *a, **k: dict(status='UNKNOWN', reason='review budget refusal'))
    result = ab.evaluate_six(left*700+[320,240], right*700+[320,240], rec, range(6))
    assert result['exact_result']['status'] == 'CONSTRUCTED'
    assert result['C']['independent']['status'] == 'UNKNOWN'
    assert result['C']['world_xyz'] is None
    counts = batch._counts([dict(status='EVALUATED_WITH_STAGE_STATUSES', ablation=result)])
    assert counts['requested'] == 1
    assert counts['C_certificate_verified'] == counts['C_world_ok'] == 0
    assert counts['C_constructed'] == 1


def test_selected_and_unselected_blocks_cover_all_complete_tracks():
    ids = list(range(253))
    groups = batch.select_groups(ids)
    blocks = groups['selected'] + groups['unselected_blocks']
    assert sorted(p for b in blocks for p in b['point_ids']) == ids[:252]
    assert groups['tail_point_ids'] == [252]
    assert len(groups['selected']) == 40
    assert len({b['block_index'] for b in blocks}) == 42
