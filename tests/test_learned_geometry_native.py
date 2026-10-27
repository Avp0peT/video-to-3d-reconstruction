import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from integrations.demo_pipeline import learned_geometry_native as native


def fixture_source(tmp_path):
    root=tmp_path/'source';root.mkdir()
    image=root/'rgb.png';mask=root/'mask.png'
    Image.new('RGB',(576,1024)).save(image)
    Image.new('L',(576,1024),255).save(mask)
    sha=native._hash
    ids=list(range(16));points=[dict(track_id=t,xy=[20.+t,30.],visibility_probability=1.,certainty=1.,status='VISIBLE_CANDIDATE',supported=True) for t in ids]
    frames=[dict(decode_index=555+i,path=str(image),sha256=sha(image)['sha256']) for i in range(3)]
    masks=[dict(decode_index=f['decode_index'],current_observed=True,status='OK',raw_path=str(image),raw_sha256=sha(image)['sha256'],mask_path=str(mask),mask_sha256=sha(mask)['sha256']) for f in frames]
    docs=dict(INPUTS=dict(frames=frames),MASKS=dict(frames=masks),
        TRACKS=dict(status='TRACKS_GENERATED_NOT_IDENTITY_VERIFIED',physical_identity_verified=False,geometry_accepted=False,
            frames=[dict(decode_index=f['decode_index'],points=points) for f in frames]),
        QUERIES=dict(points=[dict(track_id=t) for t in ids]),
        PLAN=dict(schema='learned-point-plan-v1',model='TAPNext++',frames=3,dimensions=[576,1024],query_count=16),
        RESULT=dict(status='EXPERIMENT_COMPLETE_NOT_IDENTITY_VERIFIED'))
    for n,d in docs.items(): (root/(n+'.json')).write_text(json.dumps(d),encoding='utf-8')
    return root


def mutate(root,name,change):
    p=root/(name+'.json');doc=json.loads(p.read_text());change(doc);p.write_text(json.dumps(doc))


def test_prepare_candidate_rows_and_exact_half_pixel(tmp_path):
    root=fixture_source(tmp_path);before=set(root.iterdir());plan=native.prepare(root)
    assert set(root.iterdir())==before
    assert plan['schema']=='learned-candidate-geometry-source-v1'
    assert plan['physical_identity_verified'] is False
    assert len(plan['pairs'])==3
    np.testing.assert_array_equal(plan['frames'][0]['keypoints'][0],[20.5,30.5])
    np.testing.assert_array_equal(plan['pairs'][0]['matches'],np.column_stack((np.arange(16),np.arange(16))))
    assert 'measured' not in repr(plan)


def test_selection_is_bounded_endpoints_and_outcome_independent():
    assert native.keyframe_indices(64)==list(range(64))
    rows=native.keyframe_indices(1000)
    assert len(rows)==len(set(rows))==64 and rows[0]==0 and rows[-1]==999
    with pytest.raises(ValueError): native.keyframe_indices(1001)


def test_invisible_candidate_is_excluded(tmp_path):
    root=fixture_source(tmp_path)
    def hide(d):
        for f in d['frames']:
            f['points'][0].update(visibility_probability=0.,status='OCCLUDED_OR_UNCERTAIN',supported=False)
    mutate(root,'TRACKS',hide)
    plan=native.prepare(root)
    assert plan['excluded_candidate_counts']=={'OCCLUDED_OR_UNCERTAIN':3}
    assert all(0 not in p['track_ids'] for p in plan['pairs'])


def test_claimed_visible_outside_mask_is_rejected(tmp_path):
    root=fixture_source(tmp_path)
    def move(d): d['frames'][0]['points'][0]['xy']=[575.8,30.]
    mutate(root,'TRACKS',move)
    with pytest.raises(ValueError,match='classification drift'): native.prepare(root)


def test_hash_drift_and_frame_identity_fail_closed(tmp_path):
    root=fixture_source(tmp_path)
    mutate(root,'MASKS',lambda d:d['frames'][0].update(raw_sha256='0'*64))
    with pytest.raises(ValueError,match='binding mismatch'): native.prepare(root)


def test_old_measured_schema_is_not_admitted(tmp_path):
    root=fixture_source(tmp_path)
    mutate(root,'TRACKS',lambda d:d.update(status='OK',schema='balanced-object-observations-v1'))
    with pytest.raises(ValueError,match='unverified learned source'): native.prepare(root)


def test_no_match_mask_never_supplies_candidates(tmp_path):
    root=fixture_source(tmp_path)
    mutate(root,'MASKS',lambda d:d['frames'][0].update(current_observed=False,status='NO_MATCH'))
    with pytest.raises(ValueError,match='classification drift'): native.prepare(root)
