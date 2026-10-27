from pathlib import Path
import sys
import json
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'outputs/toolchains/opencv-4.13.0.92'))
import cv2
from integrations.demo_pipeline.temporal_geometry import analyse_track_pairs


@pytest.fixture(autouse=True)
def controlled_cv():
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(0)


def data(planar=False):
    rng = np.random.default_rng(4)
    xyz = rng.uniform([-1,-1,4], [1,1,8], (60,3))
    if planar:
        xyz[:,2] = 6
    projected = []
    for translation in (0, .6, 1.2):
        q = xyz.copy();q[:,0] -= translation
        projected.append(q[:,:2]/q[:,2,None]*500 + [320.5,240.5])
    tracks = [dict(track_id=f'{i:03}', observations=[dict(frame=f,xy=projected[f][i].tolist(),target_verified=True,fb_error=0) for f in range(3)]) for i in range(len(xyz))]
    frames = [dict(input_index=i,time_seconds=i,width=640,height=480,mask_pixels=150000,model_ids=[str(i)]) for i in range(3)]
    return tracks,frames


def test_synthetic_camera_candidate_and_no_merge_authority():
    tracks,frames=data()
    result=analyse_track_pairs(tracks,frames,cv2)
    assert all(p['status']=='CANDIDATE_SUPPORT_NOT_VERIFIED' for p in result['pairs'])
    assert all(len(p['holdout_ids'])>=8 for p in result['pairs'])
    assert result['proposed_bridges'] and not result['components_merged'] and not result['stitch_authorized']
    assert result['top_three_frame_candidates'][0]['common_unique_target_verified_tracks']==60
    json.dumps(result,allow_nan=False)


def test_holdout_mismatches_are_not_train_inlier_validation():
    tracks,frames=data()
    for index,track in enumerate(tracks):
        if index%3==0:
            track['observations'][1]['xy'][1]+=40
    result=analyse_track_pairs(tracks,frames,cv2)
    row=next(p for p in result['pairs'] if p['frames']==[0,1])
    assert row['status']=='HOLDOUT_RESIDUAL_LARGE'
    assert row['train_inlier_count']>=30


def test_planar_homography_ambiguity_is_explicit():
    tracks,frames=data(planar=True)
    result=analyse_track_pairs(tracks,frames,cv2)
    assert all(p.get('planar_or_low_parallax_ambiguity') for p in result['pairs'])


def test_forward_reverse_replays_not_independent():
    tracks,frames=data()
    duplicate=[dict(track_id='repeat'+t['track_id'],observations=t['observations']) for t in tracks]
    result=analyse_track_pairs(tracks+duplicate,frames,cv2)
    assert all(p['unique_candidates']==60 and p['duplicate_alias_count']==60 for p in result['pairs'])
    assert all(len(c['aliases'])==2 for c in result['pairs'][0]['candidates'])


def test_unverified_endpoints_and_frame_budget():
    tracks,frames=data()
    for t in tracks:
        t['observations'][1]['target_verified']=False
    result=analyse_track_pairs(tracks,frames,cv2)
    assert next(p for p in result['pairs'] if p['frames']==[0,1])['unique_candidates']==0
    with pytest.raises(ValueError,match='budget'):
        analyse_track_pairs([],frames*33,cv2)


def test_nonfinite_refused_and_degenerate_explicit():
    tracks,frames=data()
    tracks[0]['observations'][0]['xy'][0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        analyse_track_pairs(tracks,frames,cv2)
    tracks=[dict(track_id=str(i),observations=[dict(frame=f,xy=[i*4+.5,2*i+.5],target_verified=True) for f in range(2)]) for i in range(30)]
    result=analyse_track_pairs(tracks,frames[:2],cv2)
    assert result['pairs'][0]['status']=='DEGENERATE_COORDINATES'


def test_mask_frame_subset_recorded_without_full_coverage_claim():
    frames=[dict(input_index=i,mask_pixels=1,time_seconds=i,model_ids=[]) for i in range(30)]
    result=analyse_track_pairs([],frames,cv2)
    assert len(result['selected_frames'])==24 and len(result['omitted_mask_eligible_frames'])==6
    assert result['selected_frames'][0]==0 and result['selected_frames'][-1]==29


@pytest.mark.parametrize('invalid', ['1.5', True, np.bool_(False), None, object()])
def test_non_numeric_coordinate_types_refused(invalid):
    tracks,frames=data()
    tracks[0]['observations'][0]['xy'][0]=invalid
    with pytest.raises(ValueError,match='invalid canonical'):
        analyse_track_pairs(tracks,frames,cv2)


@pytest.mark.parametrize('matrix', [np.zeros((3,3)), np.ones((2,3)), np.full((3,3),np.nan)])
def test_degenerate_h_does_not_claim_planar_ambiguity_or_block_f(monkeypatch,matrix):
    tracks,frames=data()
    monkeypatch.setattr(cv2,'findHomography',lambda *a,**k:(matrix,None))
    result=analyse_track_pairs(tracks,frames,cv2)
    assert all(p['H_status']=='DEGENERATE_OR_UNAVAILABLE' for p in result['pairs'])
    assert all(p['planar_or_low_parallax_ambiguity'] is None for p in result['pairs'])
    assert all(p['status']=='CANDIDATE_SUPPORT_NOT_VERIFIED' for p in result['pairs'])
