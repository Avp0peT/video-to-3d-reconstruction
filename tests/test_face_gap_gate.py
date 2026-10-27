import numpy as np
import pytest

from integrations.video_masks import face_gap_gate as gap
from integrations.video_masks import face_mask_gate as old


def fixture():
    mask = np.zeros((96, 96), bool)
    mask[:, 12:40] = True
    mask[:, 56:84] = True
    mask[10:30, 12:84] = True
    return mask, np.ones_like(mask)


def view(uv, mask=None, validity=None):
    m, v = fixture()
    m = m if mask is None else mask
    v = v if validity is None else validity
    xyz = np.column_stack((np.asarray(uv)*10, np.full(3, 10)))
    return gap.evaluate_gap_view(xyz, np.array([[0, 1, 2]]), K=np.eye(3), R=np.eye(3),
                                C=np.zeros(3), mask=m, validity=v, view_id='a'), xyz


def test_subdivided_curtain_all_vertices_background_is_not_exempt():
    result, vertices = view([[44, 45], [51, 45], [44, 60]])
    mask, valid = fixture()
    assert not mask[45, 44] and result['status'][0] == gap.CONFLICT
    assert result['background_pixels'][0] >= 4
    # This is the previous structural failure, not a lowered depth tolerance.
    previous = old.evaluate_view(vertices, np.array([[0, 1, 2]]), K=np.eye(3), R=np.eye(3), C=np.zeros(3),
        mask=mask, validity=valid, depth=np.where(mask, 10., 0.), view_id='a')
    assert previous['status'][0] == old.UNKNOWN
    assert previous['reason_names'][int(previous['reason'][0])] == 'VERTEX_OUTSIDE_FOREGROUND_MARGIN'


def test_outer_background_triangle_has_no_bank_pair_and_no_negative_vote():
    result, _ = view([[2, 40], [6, 40], [2, 60]])
    assert result['status'][0] == gap.UNKNOWN
    assert result['reason_names'][int(result['reason'][0])] == 'NO_BRACKETED_GAP'


def test_banks_are_from_mask_not_face_endpoints_and_mask_is_unchanged():
    mask, valid = fixture(); before = mask.copy()
    evidence, info = gap.bracketed_gap_mask(mask, valid)
    assert evidence[50, 47] and not evidence[50, 5]
    assert np.array_equal(before, mask)
    assert info['source_mask_changed'] is False
    assert info['physical_visibility'] == 'NOT_EVALUATED'


def test_foreground_face_and_too_wide_separation_do_not_vote():
    result, _ = view([[20, 45], [30, 45], [20, 60]])
    assert result['status'][0] != gap.CONFLICT
    mask, valid = fixture()
    mask[:, 56:84] = False
    evidence, _ = gap.bracketed_gap_mask(mask, valid)
    assert not evidence[50, 47]


def test_invalid_corridor_does_not_supply_a_foreground_pair():
    mask, valid = fixture()
    valid[30:, 46:50] = False
    evidence, _ = gap.bracketed_gap_mask(mask, valid)
    assert not evidence[50, 44]
    result, _ = view([[44, 45], [51, 45], [44, 60]], validity=valid)
    assert result['status'][0] != gap.CONFLICT


def test_boundary_tolerance_still_keeps_narrow_gap_unknown():
    mask, valid = fixture()
    mask[30:80, 40:46] = True
    mask[30:80, 50:56] = True
    evidence, _ = gap.bracketed_gap_mask(mask, valid)
    assert not evidence[50, 48]


def test_rotation_of_mask_preserves_gap_evidence():
    mask, valid = fixture()
    first, _ = gap.bracketed_gap_mask(mask, valid)
    rotated, _ = gap.bracketed_gap_mask(np.rot90(mask), np.rot90(valid))
    assert np.array_equal(np.rot90(first), rotated)


def test_negative_vote_does_not_cause_single_view_quarantine():
    result, vertices = view([[44, 45], [51, 45], [44, 60]])
    f = np.array([[0, 1, 2]])
    assert not old.aggregate_views(vertices, f, [result], {'a': [0, 0, 0]})['quarantine'][0]
    duplicate = dict(result, view_id='b')
    assert not old.aggregate_views(vertices, f, [result, duplicate], {'a': [0, 0, 0], 'b': [0, 0, 0]})['quarantine'][0]


def test_offscreen_and_deadline_do_not_vote():
    result, _ = view([[-20, 45], [51, 45], [44, 60]])
    assert result['status'][0] == gap.UNKNOWN
    mask, valid = fixture()
    def stop():
        raise TimeoutError('budget')
    with pytest.raises(TimeoutError):
        gap.bracketed_gap_mask(mask, valid, check=stop)


@pytest.mark.parametrize('value', [True, float('nan'), 0, 1])
def test_unbounded_gap_span_refused(value):
    with pytest.raises(ValueError):
        gap.GapPolicy(max_gap_span_fraction=value)
