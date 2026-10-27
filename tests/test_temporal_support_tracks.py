"""CPU-only temporal tracker integration; no PPP execution or geometry acceptance."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'outputs/toolchains/opencv-4.13.0.92'))
import cv2 as cv
from kozlov_lab.stereo import coverage_tracks as c


def texture():
    return cv.GaussianBlur(np.random.default_rng(9).integers(0,256,(120,160),dtype=np.uint8),(3,3),.6)


def run(images,masks):
    return c.track_coverage(images,masks,cv,profile_id=c.TEMPORAL_SUPPORT_PROFILE_ID)


def test_v1_and_default_equal_frozen_snapshot():
    path=Path(__file__).resolve().parents[1]/'outputs/delegation/PPP_VIDEO/20261014_object_first01/temporal_support_20261017/snapshotbeforeedits/coverage_tracks.py'
    spec=importlib.util.spec_from_file_location('frozen_coverage',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    image=texture();mask=np.full_like(image,255)
    for profile in ('default',c.INCREMENTAL_PROFILE_ID):
        assert c.track_coverage([image]*4,[mask,None,None,mask],cv,profile_id=profile)==old.track_coverage([image]*4,[mask,None,None,mask],cv,profile_id=profile)


def test_profile_preserves_flow_thresholds():
    old=c.profile_for(profile_id=c.INCREMENTAL_PROFILE_ID)
    new=c.profile_for(profile_id=c.TEMPORAL_SUPPORT_PROFILE_ID)
    for key in ('win_size','max_level','fb_max','error_max','strong_fb_max','strong_error_max','max_active','unknown_mask_max_steps'):
        assert new[key]==old[key]
    with pytest.raises(ValueError): c.profile_for('whole-frame',c.TEMPORAL_SUPPORT_PROFILE_ID)


def test_missing_mask_support_never_verifies_and_return_splits_segment():
    image=texture();mask=np.full_like(image,255)
    result=run([image]*5,[mask,mask,mask,None,mask])
    supported=[t for t in result['tracks'] if any(o['mask_status']=='temporal_supported' for o in t['observations'])]
    assert supported
    for t in supported:
        gap=next(o for o in t['observations'] if o['frame_index']==3)
        assert gap['label']=='unknown' and gap['target_verified'] is False
        assert gap['tracking_status']=='flow_tracked' and gap['temporal_support']['supported']
        returned=next(o for o in t['observations'] if o['frame_index']==4)
        assert returned['identity_segment']==gap['identity_segment']+1
    assert result['coverage'][3]['seeded']==0
    assert all(e['cross_gap_identity']=='UNVERIFIED' for e in result['events'] if e['kind']=='mask_gap_recovered')


def test_seed_only_history_cannot_support_missing_mask():
    image=texture();mask=np.full_like(image,255)
    result=run([image,image],[mask,None])
    assert result['coverage'][1]['active']>0
    assert all(o['mask_status']=='unknown' for t in result['tracks'] for o in t['observations'] if o['frame_index']==1)


def test_boundary_support_does_not_promote_or_birth_outside(monkeypatch):
    from kozlov_lab.stereo import temporal_support as kernel
    monkeypatch.setattr(kernel,'assess_temporal_support',lambda a,b,p,q,*args,**kw:[{'supported':True} for _ in p])
    image=texture();full=np.full_like(image,255);mask=full.copy();mask[:,:80]=0
    result=run([image]*5,[full,full,full,mask,full])
    boundary=[(t,o) for t in result['tracks'] for o in t['observations'] if o['mask_status']=='boundary_supported']
    assert boundary
    for t,o in boundary:
        assert 74<=o['xy'][0]-.5<80
        assert o['label']=='background' and not o['target_verified']
        assert t['seed_frame']<3
        after=next((v for v in t['observations'] if v['frame_index']==4),None)
        if after: assert after['identity_segment']==o['identity_segment']+1
    assert any(t['termination']['reason']=='LEFT_FOREGROUND' for t in result['tracks'])
    for t in result['tracks']:
        if t['seed_frame']==3:
            assert t['observations'][0]['label']=='foreground'


def test_unsupported_unknown_survives_without_support_metadata(monkeypatch):
    from kozlov_lab.stereo import temporal_support as kernel
    monkeypatch.setattr(kernel,'assess_temporal_support',lambda a,b,p,q,*args,**kw:[{'supported':False,'reasons':['PATCH_NCC']} for _ in p])
    image=texture();mask=np.full_like(image,255)
    result=run([image]*4,[mask,mask,mask,None])
    gaps=[o for t in result['tracks'] for o in t['observations'] if o['frame_index']==3]
    assert gaps and all(o['mask_status']=='unknown' and not o['target_verified'] for o in gaps)

@pytest.mark.parametrize('failure',['fb','error'])
def test_support_cannot_override_ordinary_flow_threshold(monkeypatch,failure):
    from kozlov_lab.stereo import temporal_support as kernel
    monkeypatch.setattr(kernel,'assess_temporal_support',lambda a,b,p,q,*args,**kw:[{'supported':True} for _ in p])
    class Flow:
        calls=0
        def __getattr__(self,name): return getattr(cv,name)
        def calcOpticalFlowPyrLK(self,a,b,p,d,**kw):
            self.calls+=1
            assert kw['winSize']==(21,21) and kw['maxLevel']==3
            q=p.copy();status=np.ones((len(p),1),np.uint8);errors=np.zeros((len(p),1),np.float32)
            if self.calls==2 and failure=='fb':q[:,:,0]+=3.1
            if self.calls==1 and failure=='error':errors[:]=40.1
            return q,status,errors
    image=texture();mask=np.full_like(image,255)
    result=c.track_coverage([image,image],[mask,None],Flow(),profile_id=c.TEMPORAL_SUPPORT_PROFILE_ID)
    assert result['coverage'][1]['active']==0
    assert all(t['termination']['reason']=='FLOW_THRESHOLD' for t in result['tracks'])


def test_thirteenth_unknown_step_and_empty_mask_still_terminate():
    image=texture();mask=np.full_like(image,255)
    result=run([image]*15,[mask]+[None]*13+[mask])
    assert result['coverage'][13]['active']==0
    assert any(t['termination']['reason']=='MASK_GAP_EXCEEDED' for t in result['tracks'])
    result=run([image]*4,[mask,mask,mask,np.zeros_like(mask)])
    assert result['coverage'][3]['active']==0 and result['coverage'][3]['seeded']==0
    assert all(t['termination']['reason']=='EMPTY_FOREGROUND' for t in result['tracks'])


def anchor_history():
    return [dict(frame_index=0,label='foreground',strong=False,fb_error_px=None,interior_distance_px=10),
        dict(frame_index=1,label='foreground',strong=True,fb_error_px=0.,interior_distance_px=10),
        dict(frame_index=2,label='foreground',strong=True,fb_error_px=0.,interior_distance_px=10),
        dict(frame_index=3,label='unknown',strong=True,fb_error_px=0.,mask_status='temporal_supported',
             temporal_support=dict(supported=True,patch_ncc=.9,patch_std=10.,reasons=[]))]


def test_anchor_qualification_never_depends_on_prior_supported():
    from copy import deepcopy
    cfg=c.profile_for(profile_id=c.TEMPORAL_SUPPORT_PROFILE_ID)
    original=anchor_history()
    expected=c._original_foreground_anchor(original,4,12,6,cfg)
    assert expected==dict(track_id=12,last_foreground_frame=2,age=2,source='original_foreground_with_bounded_patch_chain')
    for status in ('unknown','temporal_supported','boundary_supported'):
        changed=deepcopy(original);changed[-1]['mask_status']=status
        changed[-1]['temporal_support']['supported']=False
        changed[-1]['temporal_support']['reasons']=['insufficient_neighbors','local_motion_disagreement']
        assert c._original_foreground_anchor(changed,4,12,6,cfg)==expected


@pytest.mark.parametrize('change',['ncc','std','flow','patch','background','weak'])
def test_independent_patch_or_flow_break_disqualifies_anchor(change):
    history=anchor_history();last=history[-1]
    if change=='ncc':last['temporal_support']['patch_ncc']=.74
    if change=='std':last['temporal_support']['patch_std']=4.9
    if change=='flow':last['temporal_support']['reasons']=['flow_failed']
    if change=='patch':last['temporal_support']['reasons']=['patch_appearance']
    if change=='background':last['label']='background'
    if change=='weak':last['strong']=False
    cfg=c.profile_for(profile_id=c.TEMPORAL_SUPPORT_PROFILE_ID)
    assert c._original_foreground_anchor(history,4,1,6,cfg) is None


def test_unknown_chain_does_not_reset_original_anchor_age():
    from copy import deepcopy
    history=anchor_history()
    for frame in range(4,10):
        obs=deepcopy(history[-1]);obs['frame_index']=frame;history.append(obs)
    cfg=c.profile_for(profile_id=c.TEMPORAL_SUPPORT_PROFILE_ID)
    assert c._original_foreground_anchor(history[:8],8,1,6,cfg)['last_foreground_frame']==2
    assert c._original_foreground_anchor(history[:8],8,1,6,cfg)['age']==6
    assert c._original_foreground_anchor(history,10,1,6,cfg) is None


def test_proof_anchor_ids_map_kernel_indices_without_self():
    image=texture();mask=np.full_like(image,255)
    result=run([image]*5,[mask,mask,mask,None,None])
    proofs=[(t['track_id'],o['temporal_support']) for t in result['tracks'] for o in t['observations']
            if o['temporal_support'] and o['mask_status']=='temporal_supported']
    assert proofs
    for ident,proof in proofs:
        assert ident not in proof['neighbor_anchor_track_ids']
        assert ident not in proof['agreeing_anchor_track_ids']
        assert set(proof['agreeing_anchor_track_ids'])<=set(proof['neighbor_anchor_track_ids'])
        assert all(a['track_id']!=ident and a['source']=='original_foreground_with_bounded_patch_chain' for a in proof['eligible_anchors'])
