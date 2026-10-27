"""Real CPU OpenCV synthetic-image tracking, never video/model acceptance."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/toolchains/opencv-4.13.0.92'))
import cv2 as cv
from kozlov_lab.stereo.temporal_tracks import track_sequence


def texture():
    image=np.random.default_rng(33).integers(0,256,(160,192),dtype=np.uint8)
    return cv.GaussianBlur(image,(3,3),.6)


def shifted(image,x,y):
    return cv.warpAffine(image,np.array([[1.,0.,x],[0.,1.,y]],dtype=np.float32),
        (image.shape[1],image.shape[0]),flags=cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT)


def full(image):return np.full(image.shape,255,dtype=np.uint8)


def test_real_translation_fb_and_canonical_pixel_convention():
    base=texture();grays=[shifted(base,2*i,i) for i in range(4)];before=[g.copy() for g in grays]
    result=track_sequence(grays,[full(base)]*4,cv,dict(max_active=32,directions=['forward']))
    long=[t for t in result['tracks'] if len(t['observations'])==4]
    assert len(long)>=10
    for track in long:
        obs=track['observations'];start=np.array(obs[0]['xy'])
        assert np.array_equal(start-.5,np.rint(start-.5))
        for i,o in enumerate(obs):
            assert np.linalg.norm(np.array(o['xy'])-start-[2*i,i])<.15
            assert o['target_verified'] and o['frame']==i
            if i:assert o['fb_error']<=1
    assert all(np.array_equal(g,old) for g,old in zip(grays,before))
    assert result['passes'][0]['max_active_observed']<=32
    assert result['backend']==dict(name='OpenCV CPU',version='4.13.0',opencl_enabled=False,threads=2)
    json.dumps(result,allow_nan=False)


def test_real_translation_return_loop():
    base=texture();grays=[shifted(base,dx,0) for dx in [0,2,4,2,0]]
    result=track_sequence(grays,[full(base)]*5,cv,dict(max_active=24,directions=['forward']))
    loop=[t for t in result['tracks'] if len(t['observations'])==5]
    assert len(loop)>=8
    assert max(np.linalg.norm(np.array(t['observations'][-1]['xy'])-t['observations'][0]['xy']) for t in loop)<.15


def test_unknown_mask_limit_and_no_unknown_seeding():
    base=texture();result=track_sequence([base]*5,[full(base),None,None,None,full(base)],cv,
        dict(max_active=16,unknown_mask_max_steps=2,directions=['forward']))
    first=[t for t in result['tracks'] if t['seed_frame']==0]
    assert first and all(t['termination']['reason']=='UNKNOWN_MASK_LIMIT' for t in first)
    assert all([o['target_verified'] for o in t['observations']]==[True,False,False] for t in first)
    assert all(t['observations'][-1]['target_status']=='TARGET_UNVERIFIED' for t in first)
    assert all(step['seeded']==0 for step in result['steps'][1:4])
    assert result['steps'][4]['seeded']>0
    assert all(not any(o['frame']==4 for o in t['observations']) for t in first)


def test_known_occlusion_terminates_old_ids_and_restoration_uses_new_ids():
    base=texture();zero=np.zeros_like(base)
    result=track_sequence([base,zero,base],[full(base),zero,full(base)],cv,dict(max_active=16,directions=['forward']))
    first=[t for t in result['tracks'] if t['seed_frame']==0];later=[t for t in result['tracks'] if t['seed_frame']==2]
    assert first and later
    assert not {t['track_id'] for t in first}&{t['track_id'] for t in later}
    assert all(t['termination']['frame']==1 and len(t['observations'])==1 for t in first)


def test_real_large_motion_boundary_failures_are_explicit():
    base=texture();result=track_sequence([base,shifted(base,190,0)],[full(base)]*2,cv,
        dict(max_active=24,directions=['forward']))
    first=[t for t in result['tracks'] if t['seed_frame']==0]
    assert first and any(t['termination']['reason']!='DIRECTION_BOUNDARY' for t in first)
    assert all(all(.5<=o['xy'][0]<=191.5 and .5<=o['xy'][1]<=159.5 for o in t['observations']) for t in result['tracks'])


def test_two_direction_ids_are_disjoint_and_labels_retain_input_frames():
    base=texture();result=track_sequence([base,shifted(base,2,1)],[full(base)]*2,cv,dict(max_active=16))
    forward,backward=result['passes'];f={t['track_id'] for t in forward['tracks']};b={t['track_id'] for t in backward['tracks']}
    assert f and b and not f&b
    assert forward['frame_order']==[0,1] and backward['frame_order']==[1,0]
    assert all(t['direction']=='backward' for t in backward['tracks'])
    assert not result['correspondence_verified'] and not result['geometry_acceptance']


def test_no_texture_or_unknown_masks_create_no_seeds():
    zero=np.zeros((40,40),dtype=np.uint8)
    assert track_sequence([zero]*2,[full(zero)]*2,cv)['tracks']==[]
    base=texture();result=track_sequence([base]*2,[None,None],cv)
    assert result['tracks']==[] and all(s['seeded']==0 for s in result['steps'])


@pytest.mark.parametrize('mutate',[
    lambda g,m:([],[]),lambda g,m:([g[0].astype(float)],m[:1]),
    lambda g,m:([np.zeros((1025,12),np.uint8)],[None]),
    lambda g,m:(g,[np.full(g[0].shape,1,np.uint8)]*len(g)),
    lambda g,m:(g,[None]),lambda g,m:([g[0],g[0][:10]],[None,None])])
def test_illegal_inputs_refused(mutate):
    base=texture();g,m=mutate([base]*2,[full(base)]*2)
    with pytest.raises(ValueError):track_sequence(g,m,cv)


def test_profile_cannot_exceed_limits_or_retune_lk():
    base=texture()
    for profile in [dict(max_active=257),dict(fb_max_pixels=2),dict(max_active=True),dict(directions=['forward','forward'])]:
        with pytest.raises(ValueError):track_sequence([base],[full(base)],cv,profile)


def test_default_twelve_unknown_steps_are_allowed_then_thirteenth_terminates():
    base=texture();result=track_sequence([base]*14,[full(base)]+[None]*13,cv,
        dict(max_active=8,directions=['forward']))
    assert result['tracks']
    for track in result['tracks']:
        assert len(track['observations'])==13 and track['observations'][-1]['consecutive_unknown_steps']==12
        assert track['termination']['frame']==13 and track['termination']['reason']=='UNKNOWN_MASK_LIMIT'


def test_limited_corner_cover_is_partial_and_seeds_are_original_candidates():
    base=texture();result=track_sequence([base],[full(base)],cv,dict(max_active=1,directions=['forward']))
    step=result['steps'][0]
    assert step['seed_cover']['status']=='PARTIAL' and not step['seed_cover']['complete']
    assert step['seeded']==step['active_count']==1
    track=result['tracks'][0]
    assert track['seed_xy']==step['cover_candidate_xy'][track['seed_candidate_index']]
