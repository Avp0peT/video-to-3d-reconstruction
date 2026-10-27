"""TEST_DATA only: synthetic patch admission and identity discontinuity."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.video_masks import region_tracks as rt


@pytest.fixture(scope='module')
def cv():
    result = rt.load_cv([])
    if result is None:
        pytest.skip('bound OpenCV unavailable')
    return result


def texture(cv):
    rng = np.random.default_rng(48)
    return cv.GaussianBlur(rng.integers(30, 170, (128, 128), dtype=np.uint8), (3, 3), .7)


def test_translation_and_gain_bias_are_measured(cv):
    a = texture(cv)
    b = cv.warpAffine(a, np.float32([[1, 0, 3], [0, 1, 2]]), (128, 128))
    b = np.clip(b.astype(float)*1.15+12, 0, 255).astype(np.uint8)
    mask = np.full_like(a, 255)
    p = [[40, 40], [70, 70], [90, 45]]
    q, reasons, scores = rt.step(a, b, mask, mask, p, cv)
    assert reasons == ['accepted']*3
    assert np.max(np.linalg.norm(q-np.asarray(p)-[3, 2], axis=1)) < .4
    assert min(scores) > .95


def test_mask_occlusion_and_wrong_texture_reject(cv):
    a = texture(cv); full = np.full_like(a, 255)
    empty = np.zeros_like(a)
    assert rt.step(a, a, full, empty, [[64, 64]], cv)[1] == ['target_mask_or_bounds']
    occluded = a.copy(); occluded[45:84, 45:84] = 110
    assert rt.step(a, occluded, full, full, [[64, 64]], cv)[1][0] != 'accepted'
    wrong = np.random.default_rng(100).integers(30, 170, a.shape, dtype=np.uint8)
    assert rt.step(a, wrong, full, full, [[64, 64]], cv)[1][0] != 'accepted'


def test_flat_produces_no_reliable_points(cv):
    gray = np.full((128, 128), 110, np.uint8)
    mask = np.full_like(gray, 255)
    assert rt.seed(gray, mask, [], cv) == []
    assert rt.step(gray, gray, mask, mask, [[64, 64]], cv)[1][0] != 'accepted'


def test_identity_lost_then_new_endpoint_birth_and_sha(tmp_path, cv):
    gray = texture(cv); full = np.full_like(gray, 255); empty = np.zeros_like(gray)
    rows = []
    for i, mask in enumerate((full, full, empty, full)):
        row = dict(decode_index=i, width=128, height=128)
        if i != 1:
            row['input_index'] = i
        for kind, array in (('image', gray), ('mask', mask)):
            path = tmp_path/f'{kind}{i}.png'; Image.fromarray(array).save(path)
            row[kind+'_path'] = str(path)
            row[kind+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(schema='glove-region-plan-v1', timeline=rows)))
    result = rt.run(plan, tmp_path/'out')
    ids0 = {o['track_id'] for o in result['frames'][0]['observations']}
    ids2 = {o['track_id'] for o in result['frames'][2]['observations']}
    assert ids0 and ids2 and ids0.isdisjoint(ids2)
    assert result['frames'][1]['observations'] == []
    assert result['timeline_stats'][2]['lost']
    assert len(result['frames']) == 3 and len(result['timeline_stats']) == 4
    assert all(o['decode_index'] != 1 for t in result['tracks'] for o in t['observations'])
    progress = [json.loads(line) for line in (tmp_path/'out'/'progress.jsonl').read_text().splitlines()]
    assert progress[0]['event'] == 'START' and progress[-1]['event'] == 'FINAL'
    assert sum(r['event'] == 'FRAME' for r in progress) == 4
    assert (tmp_path/'out'/'REGION_TRACKS.json').exists()
    rows[0]['image_sha256'] = '0'*64
    plan.write_text(json.dumps(dict(schema='glove-region-plan-v1', timeline=rows)))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        rt.run(plan, tmp_path/'bad')
    rows[0]['image_sha256'] = hashlib.sha256(Path(rows[0]['image_path']).read_bytes()).hexdigest()
    bad_mask = Path(rows[0]['mask_path'])
    Image.fromarray(np.full_like(gray, 1)).save(bad_mask)
    rows[0]['mask_sha256'] = hashlib.sha256(bad_mask.read_bytes()).hexdigest()
    plan.write_text(json.dumps(dict(schema='glove-region-plan-v1', timeline=rows)))
    with pytest.raises(ValueError, match='only labels 0/255'):
        rt.run(plan, tmp_path/'invalid_mask')
