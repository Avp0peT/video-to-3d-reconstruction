"""CPU incremental motion evidence; no SAM/model or real-video execution."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'outputs/toolchains/opencv-4.13.0.92'))
import cv2 as cv
from kozlov_lab.stereo import coverage_tracks as c

NEW=c.INCREMENTAL_PROFILE_ID

def texture():
    return cv.GaussianBlur(np.random.default_rng(9).integers(0,256,(120,160),dtype=np.uint8),(3,3),.6)

def run(images,masks,engine=cv):
    return c.track_coverage(images,masks,engine,profile_id=NEW)


def test_profile_is_explicit_bounded_and_default_unchanged():
    old=c.profile_for()
    assert old['max_frames']==96 and old['missing_mask_policy']=='end_tracks_no_seeds'
    assert 'profile_id' not in old
    new=c.profile_for(profile_id=NEW)
    assert new['max_frames']==192 and new['unknown_mask_max_steps']==12 and new['max_unknown_active']==768
    for k,v in c.PROFILE.items():
        assert new[k]==(192 if k=='max_frames' else v)
    with pytest.raises(ValueError):c.profile_for('whole-frame',NEW)
    with pytest.raises(ValueError):c.profile_for(profile_id='bad')
    small=np.zeros((20,20),np.uint8)
    assert len(run([small]*192,[None]*192)['coverage'])==192
    with pytest.raises(ValueError,match='192'):run([small]*193,[None]*193)
    with pytest.raises(ValueError,match='96'):c.track_coverage([small]*97,[None]*97,cv)


def test_new_observed_mask_area_seeds_every_frame_without_physical_claim():
    image=texture();half=np.zeros_like(image);half[:,:80]=255;full=np.full_like(image,255)
    originals=[image.copy(),half.copy(),full.copy()]
    r=run([image,image,image],[half,full,full])
    assert r['coverage'][1]['seeded']>0 and r['coverage'][1]['tracked']>0
    births=[t for t in r['tracks'] if t['seed_frame']==1]
    new=[t for t in births if t['birth']['reason']=='NEWLY_VISIBLE_REGION']
    assert new
    for t in new:
        x,y=np.floor(np.array(t['observations'][0]['xy'])).astype(int)
        assert half[y,x]==0 and full[y,x]==255
        assert t['birth']['reason_scope']=='observed-new-mask-area'
        assert t['birth']['physical_surface_visibility_inferred'] is False
    assert all(t['birth']['reason']=='COVERAGE_REPLENISHMENT' for t in r['tracks'] if t['seed_frame']==0)
    assert all(o['label']=='foreground' and o['target_verified'] for t in r['tracks'] for o in t['observations'])
    assert all(np.array_equal(a,b) for a,b in zip([image,half,full],originals))
    json.dumps(r,allow_nan=False)


def test_short_unknown_gap_retains_id_and_never_upgrades_unknown():
    image=texture();mask=np.full_like(image,255)
    r=run([image]*4,[mask,None,None,mask])
    long=[t for t in r['tracks'] if len(t['observations'])==4]
    assert long
    for t in long:
        assert [o['label'] for o in t['observations']]==['foreground','unknown','unknown','foreground']
        assert [o['target_verified'] for o in t['observations']]==[True,False,False,True]
        assert all(o['fb_error_px'] is not None for o in t['observations'][1:])
    for frame in (1,2):
        s=r['coverage'][frame]
        assert s['seeded']==0 and s['suspended']==s['active']==s['unknown'] and s['foreground']==0
        assert s['active']<=768
    recovered=[e for e in r['events'] if e['kind']=='mask_gap_recovered']
    assert recovered and all(e['unknown_steps']==2 for e in recovered)
    assert r['coverage'][3]['recovered']==len(recovered)
    old=c.track_coverage([image,image],[mask,None],cv)
    assert all(len(t['observations'])==1 for t in old['tracks'])
    assert old['coverage'][1]['active']==0


def test_thirteenth_unknown_step_terminates_and_return_uses_new_ids():
    image=texture();mask=np.full_like(image,255)
    r=run([image]*15,[mask]+[None]*13+[mask])
    first=[t for t in r['tracks'] if t['seed_frame']==0]
    assert first and all(len(t['observations'])<=13 for t in first)
    ended=[t for t in first if t['termination']['reason']=='MASK_GAP_EXCEEDED']
    assert ended and all(t['termination']['frame_index']==13 for t in ended)
    assert r['coverage'][13]['active']==0 and r['coverage'][14]['recovered']==0
    assert any(t['seed_frame']==14 for t in r['tracks'])
    assert not any(e['kind']=='mask_gap_recovered' for e in r['events'])


def test_known_background_terminates_without_reassociation():
    image=texture();mask=np.full_like(image,255);background=np.zeros_like(image)
    r=run([image]*4,[mask,None,background,mask])
    first=[t for t in r['tracks'] if t['seed_frame']==0]
    assert first and all(len(t['observations'])<=2 for t in first)
    assert r['coverage'][2]['active']==0 and r['coverage'][2]['seeded']==0
    assert any(t['seed_frame']==3 for t in r['tracks'])
    assert not any(e['kind']=='mask_gap_recovered' for e in r['events'])


@pytest.mark.parametrize('failure',['backward_status','fb_threshold','forward_status'])
def test_measured_flow_failures_end_ids_never_nearest_rejoin(failure):
    class Flow:
        calls=0
        def __getattr__(self,name):return getattr(cv,name)
        def calcOpticalFlowPyrLK(self,a,b,points,d,**kw):
            self.calls+=1
            q=points.copy();status=np.ones((len(points),1),np.uint8);error=np.zeros((len(points),1),np.float32)
            if failure=='forward_status' and self.calls==1:status[:]=0
            if failure=='backward_status' and self.calls==2:status[:]=0
            if failure=='fb_threshold' and self.calls==2:q[:,:,0]+=4
            return q,status,error
    image=texture();mask=np.full_like(image,255)
    r=run([image]*3,[mask,None,mask],Flow())
    first=[t for t in r['tracks'] if t['seed_frame']==0]
    assert first and all(len(t['observations'])==1 for t in first)
    expected={'backward_status':'BACKWARD_LK_FAILED','fb_threshold':'FLOW_THRESHOLD','forward_status':'FORWARD_LK_FAILED'}[failure]
    assert all(t['termination']['reason']==expected for t in first)
    assert r['coverage'][2]['recovered']==0 and any(t['seed_frame']==2 for t in r['tracks'])


def test_real_occlusion_cannot_carry_old_ids_through_blank_frame():
    image=texture();mask=np.full_like(image,255)
    r=run([image,np.zeros_like(image),image],[mask,None,mask])
    first=[t for t in r['tracks'] if t['seed_frame']==0]
    assert first and all(len(t['observations'])==1 for t in first)
    assert r['coverage'][2]['recovered']==0
    assert any(t['seed_frame']==2 for t in r['tracks'])

@pytest.mark.parametrize('gap',[11,12,13])
def test_gap_identity_segments_and_exact_budget_boundary(gap):
    image=texture();mask=np.full_like(image,255)
    r=run([image]*(gap+3),[mask]+[None]*gap+[mask,mask])
    first=[t for t in r['tracks'] if t['seed_frame']==0]
    assert first
    for t in first:
        obs=t['observations']
        assert obs[0]['identity_segment']==0
        unknown=[o for o in obs if o['label']=='unknown']
        assert all(o['identity_segment']==0 and o['target_verified'] is False for o in unknown)
    recovered=[e for e in r['events'] if e['kind']=='mask_gap_recovered']
    if gap<=12:
        assert recovered and all(e['unknown_steps']==gap for e in recovered)
        for event in recovered:
            assert event['cross_gap_identity']=='UNVERIFIED'
            assert event['previous_identity_segment']==0 and event['identity_segment']==1
            obs=r['tracks'][event['track_id']]['observations']
            assert all(o['identity_segment']==1 for o in obs if o['frame_index']>=gap+1)
            assert all(o['identity_segment']==0 for o in obs if o['frame_index']<=gap)
        assert all(t['observations'][0]['identity_segment']==0 for t in r['tracks'])
    else:
        assert not recovered
        assert all(t['termination']['frame_index']<=13 for t in first)
        assert not any(o['frame_index']>=14 for t in first for o in t['observations'])


def test_default_result_matches_frozen_original_implementation():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'outputs/delegation/PPP_VIDEO/20261014_object_first01/incremental_tracks_20261016/tracker_original/coverage_tracks.py'
    spec=importlib.util.spec_from_file_location('frozen_coverage_original',path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    image=texture();mask=np.full_like(image,255)
    for scope in ('object','whole-frame'):
        expected=original.track_coverage([image]*3,[mask,None,mask],cv,selection_scope=scope)
        actual=c.track_coverage([image]*3,[mask,None,mask],cv,selection_scope=scope)
        assert actual==expected
