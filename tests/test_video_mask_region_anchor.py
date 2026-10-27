"""Small CPU TEST_DATA; endpoint admission is not physical identity truth."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.video_masks.region_tracks import load_cv
from integrations.video_masks.region_anchor import audit_pair


@pytest.fixture(scope='module')
def cv():
    cv = load_cv([])
    if cv is None:
        pytest.skip('bound OpenCV unavailable')
    return cv


def images(cv):
    a = cv.GaussianBlur(np.random.default_rng(11).integers(30, 180, (128, 128), dtype=np.uint8), (3, 3), .7)
    b = cv.warpAffine(a, np.float32([[1, 0, 3], [0, 1, 2]]), (128, 128))
    return a, b, np.full_like(a, 255)


def test_translation_admitted(cv):
    a, b, mask = images(cv)
    result = audit_pair(a, b, mask, mask, [dict(track_id=9, xy=[50, 60])], [dict(track_id=9, xy=[53, 62])], cv)
    row = result['candidates'][0]
    assert row['track_id'] == 9 and row['status'] == 'admitted'
    for direction in ('forward', 'backward'):
        assert row[direction]['endpoint_error_pixels'] < .3
        assert row[direction]['patch_zncc'] > .98
    assert result['identity_verified'] is False


def test_repeated_texture_serial_offset_is_unknown(cv):
    tile = np.random.default_rng(5).integers(40, 180, (16, 16), dtype=np.uint8)
    gray = cv.GaussianBlur(np.tile(tile, (8, 8)), (3, 3), .7)
    mask = np.full_like(gray, 255)
    result = audit_pair(gray, gray, mask, mask, [dict(track_id=1, xy=[48, 64])], [dict(track_id=1, xy=[64, 64])], cv)
    row = result['candidates'][0]
    assert row['status'] == 'unknown'
    assert 'forward:serial_endpoint_disagreement' in row['reasons']
    assert row['forward']['patch_zncc'] > .99
    assert row['forward']['endpoint_error_pixels'] > 15


def test_mask_rejection_remains_unknown(cv):
    a, b, mask = images(cv)
    result = audit_pair(a, b, mask, np.zeros_like(mask), [dict(track_id=4, xy=[50, 60])], [dict(track_id=4, xy=[53, 62])], cv)
    row = result['candidates'][0]
    assert row['status'] == 'unknown'
    assert row['forward']['measured_xy'] is None
    assert row['forward']['reason'] == 'target_mask_or_bounds'
    assert row['backward']['reason'] == 'source_mask_or_bounds'


def test_inputs_unchanged_and_shared_ids_preserved(cv):
    a, b, mask = images(cv)
    source = [dict(track_id=7, xy=[50, 60]), dict(track_id=8, xy=[70, 70])]
    target = [dict(track_id=7, xy=[53, 62]), dict(track_id=10, xy=[80, 80])]
    copies = deepcopy((source, target)); arrays = [v.copy() for v in (a, b, mask)]
    result = audit_pair(a, b, mask, mask, source, target, cv)
    assert (source, target) == copies
    assert all(np.array_equal(v, old) for v, old in zip((a, b, mask), arrays))
    assert [r['track_id'] for r in result['candidates']] == [7]
    assert result['source_only_ids'] == [8] and result['target_only_ids'] == [10]
