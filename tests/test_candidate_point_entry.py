"""Synthetic masks and recorded tracks exercise the complete point-only wiring."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import apply_candidate_point_support as entry


def world(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('RGB', (12, 12), 'white').save(image)
    frame = dict(input_index=81, frame_id='fixture81', original_mask_status='NEEDS_SELECTION',
                 path=str(image), sha256=entry.bind(image)['sha256'], width=12, height=12)
    positive = dict(track_id=66, query_point_xy=[3., 9.])
    exterior = dict(witness_id='fixture:1', query_point_xy=[1., 5.],
                    reference_exterior_witness=True, verified_background=False)

    def evidence(point, inside):
        return dict(count=1, inside_count=int(inside), inside_fraction=float(inside),
                    status='OBSERVED', points=[dict(point, sample_xy=[int(v) for v in point['query_point_xy']], inside=inside)])

    candidates, assessed = [], []
    for cid, value in [('c00', 0), ('c01', 255)]:
        mask = np.zeros((12, 12), np.uint8)
        mask[2, 2] = 255
        mask[9, 3] = value
        path = tmp_path / (cid + '.png')
        Image.fromarray(mask).save(path)
        candidates.append(dict(candidate_id=cid, binding=entry.bind(path), original_metadata={}))
        assessed.append(dict(candidate_id=cid, area_px=int(np.count_nonzero(mask)),
            positive_evidence=evidence(positive, bool(value)), exterior_evidence=evidence(exterior, False),
            boundary=dict(clipped_px=0), reasons=[] if value else ['POSITIVE_POINT_CONFLICT']))
    row = dict(input_index=81, frame_id='fixture81', original_status='NEEDS_SELECTION',
        positive_evidence=[positive], exterior_evidence=[exterior], candidates=candidates,
        status='EVIDENCE_ONLY', selected_candidate_id=None, publishable_for_geometry=False,
        assessment=dict(status='EVIDENCE_ONLY', selected_candidate_id=None,
                        publishable=False, publishable_for_geometry=False, candidates=assessed))
    tracks = {'49_62': dict(tracks=[]), '77_90': dict(tracks=[dict(track_id=66,
        observations=[dict(frame_index=4, identity_segment=7, xy=[3.5, 9.5], label='unknown',
                           target_verified=False, strong=True, fb_error_px=.1, lk_error=2.)])])}
    return [row], {81: frame}, tracks


def test_pixels_policy_and_consumer_connected(tmp_path):
    rows, frames, tracks = world(tmp_path)
    originals = deepcopy((rows, frames, tracks))
    decisions, ledger, annotations = entry.process_rows(rows, frames, tracks)
    assert decisions[0]['status'] == 'PROVISIONAL_POINT_SUPPORT_ONLY'
    assert ledger[0]['candidate_binding'] == rows[0]['candidates'][1]['binding']
    selected = annotations['77_90']['rows'][0]
    assert selected['derived_point_support'] == 'PROVISIONAL_POINT_SUPPORT_ONLY'
    assert selected['source_xy'] == [3.5, 9.5]
    assert selected['source_identity_segment'] == 7
    assert selected['source_target_verified'] is False
    assert selected['geometry_eligible'] is False
    assert (rows, frames, tracks) == originals


@pytest.mark.parametrize('change', ['file_drift', 'membership', 'frame_id', 'original_status',
                                   'pixel_center', 'area', 'labels', 'source_coordinate'])
def test_wiring_rejects_actual_mismatch(tmp_path, change):
    rows, frames, tracks = world(tmp_path)
    candidate = rows[0]['candidates'][1]
    assessment = rows[0]['assessment']['candidates'][1]
    if change == 'file_drift':
        with Path(candidate['binding']['path']).open('ab') as f:
            f.write(b'drift')
    elif change == 'membership':
        assessment['positive_evidence']['points'][0]['inside'] = False
    elif change == 'frame_id':
        frames[81]['frame_id'] = 'another'
    elif change == 'original_status':
        frames[81]['original_mask_status'] = 'OK'
    elif change == 'pixel_center':
        assessment['positive_evidence']['points'][0]['sample_xy'] = [4, 9]
    elif change == 'area':
        assessment['area_px'] += 1
    elif change == 'labels':
        path = Path(candidate['binding']['path'])
        Image.new('L', (12, 12), 127).save(path)
        candidate['binding'] = entry.bind(path)
    else:
        tracks['77_90']['tracks'][0]['observations'][0]['xy'][0] += .01
    with pytest.raises(ValueError):
        entry.process_rows(rows, frames, tracks)


def test_untrusted_seal_rejected_before_loading_records(tmp_path):
    task = tmp_path / 'old'
    source = task / 'real01/boundary_support'
    source.mkdir(parents=True)
    (task / 'SHA256SUMS.txt').write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='drift'):
        entry.sealed_boundary(source, 'f' * 64)


def test_checksum_path_escape_and_alias_rejected(tmp_path):
    task = tmp_path / 'old'
    source = task / 'real01/boundary_support'
    source.mkdir(parents=True)
    outside = tmp_path / 'outside'
    outside.write_text('not an instruction', encoding='utf-8')
    seal = task / 'SHA256SUMS.txt'
    seal.write_text(entry.bind(outside)['sha256'] + '  ../outside\n', encoding='utf-8')
    with pytest.raises(ValueError, match='sealed path'):
        entry.sealed_boundary(source, entry.bind(seal)['sha256'])


def test_output_must_be_new_and_nonoverlapping(tmp_path):
    source = tmp_path / 'old'
    source.mkdir()
    for output in (source, source / 'new', tmp_path):
        with pytest.raises(ValueError):
            entry.fresh_output(output, (source,))
