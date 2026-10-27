"""Real CPU OpenCV synthetic gap tests; no physical identity acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/toolchains/opencv-4.13.0.92'))
import cv2 as cv
from kozlov_lab.stereo.temporal_reassociation import reassociate_tracklets, PROFILE
from kozlov_lab.stereo import temporal_reassociation as module


def scene():
    a=np.random.default_rng(33).integers(0,256,(160,192),dtype=np.uint8)
    a=cv.GaussianBlur(a,(3,3),.6)
    shifted=cv.warpAffine(a,np.array([[1.,0.,2],[0.,1.,1]],dtype=np.float32),(192,160))
    return [a,np.zeros_like(a),shifted],[np.full_like(a,255)]*3


def track(ident,frame,xy=(70.5,70.5),direction='forward'):
    return dict(track_id=ident,direction=direction,seed_xy=list(xy),observations=[
        dict(frame_index=frame,frame=frame,xy=list(xy),target_verified=True)])


def pair():return [track('old',0),track('new',2,(72.5,71.5))]


def test_blank_gap_reappearance_real_seed_only_unchanged():
    images,masks=scene(); tracks=pair();old=deepcopy(tracks);pixels=[a.copy() for a in images]
    result=reassociate_tracklets(tracks,images,masks,cv)
    assert result['accepted_link_count']==1
    link=result['links'][0]
    assert link['from_track_id']=='old' and link['to_track_id']=='new'
    assert link['missing_frame_indices']==[1] and link['missing_count']==1 and link['frame_delta']==2
    assert link['gap_status']=='UNKNOWN' and link['metrics']['seed_back_to_source_pixels']<=1
    joined=result['joined_tracks'][0]
    assert joined['source_track_ids']==['old','new']
    assert [o['frame'] for o in joined['observations']]==[0,2]
    assert joined['observations']==old[0]['observations']+old[1]['observations']
    assert tracks==old and all(np.array_equal(a,b) for a,b in zip(images,pixels))
    assert not result['correspondence_verified'] and not result['geometry_acceptance']
    json.dumps(result,allow_nan=False)


def test_new_surface_and_wrong_texture_stay_new():
    images,masks=scene();tracks=pair()+[track('other',2,(130.5,110.5))]
    r=reassociate_tracklets(tracks,images,masks,cv)
    assert r['accepted_link_count']==1 and any(t['source_track_ids']==['other'] for t in r['joined_tracks'])
    images[2]=np.random.default_rng(9).integers(0,256,images[2].shape,dtype=np.uint8)
    assert not reassociate_tracklets(pair(),images,masks,cv)['links']


def test_close_ambiguous_seed_rejected():
    images,masks=scene()
    r=reassociate_tracklets(pair()+[track('near',2,(73.5,71.5))],images,masks,cv)
    assert not r['links'] and r['rejection_counts']['AMBIGUOUS_ENDPOINT']>=1


def test_actual_seed_reverse_disagreement():
    images,masks=scene();tracks=[track('old',0),track('offseed',2,(74.,71.5))]
    r=reassociate_tracklets(tracks,images,masks,cv)
    assert not r['links']
    assert r['rejection_counts'].get('MEASURED_SEED_BACKWARD_DISAGREEMENT',0)>=1


@pytest.mark.parametrize('mode',['unknown','empty','unverified','patch_edge'])
def test_mask_support_required(mode):
    images,masks=scene();tracks=pair();masks=[m.copy() for m in masks]
    if mode=='unknown':masks[2]=None
    elif mode=='empty':masks[2][:]=0
    elif mode=='unverified':tracks[1]['observations'][0]['target_verified']=False
    else:masks[2][65,72]=0
    r=reassociate_tracklets(tracks,images,masks,cv)
    assert not r['links'] and r['joined_track_count']==2


def test_forward_backward_never_join_and_backward_can_link():
    images,masks=scene();tracks=pair();tracks[1]['direction']='backward'
    assert not reassociate_tracklets(tracks,images,masks,cv)['links']
    tracks=[track('back2',2,(72.5,71.5),'backward'),track('back0',0,(70.5,70.5),'backward')]
    r=reassociate_tracklets(tracks,images,masks,cv)
    assert len(r['links'])==1 and r['links'][0]['missing_frame_indices']==[1]


def test_global_gap_competitors_are_not_greedily_attached():
    images,masks=scene();images.append(images[2].copy());masks.append(masks[2])
    r=reassociate_tracklets(pair()+[track('later',3,(72.5,71.5))],images,masks,cv)
    assert not r['links'] and r['rejection_counts']['GLOBAL_COMPETING_ENDPOINT']==2


def test_six_step_limit_and_adjacent_not_gap():
    images,masks=scene();a=images[0];b=images[2];mask=masks[0]
    grays=[a]+[np.zeros_like(a)]*6+[b];ms=[mask]*8
    assert not reassociate_tracklets([track('a',0),track('b',7,(72.5,71.5))],grays,ms,cv)['links']
    assert len(reassociate_tracklets([track('a',0),track('b',6,(72.5,71.5))],grays[:6]+[b],ms[:7],cv)['links'])==1
    assert not reassociate_tracklets([track('a',0),track('b',1,(72.5,71.5))],[a,b],[mask]*2,cv)['links']


def test_resource_and_identity_boundaries():
    images,masks=scene()
    with pytest.raises(ValueError):reassociate_tracklets([track('x',0),track('x',2)],images,masks,cv)
    with pytest.raises(ValueError):reassociate_tracklets([],images*33,masks*33,cv)
    bad=pair();bad[1]['observations'][0]['xy'][0]=float('nan')
    with pytest.raises(ValueError):reassociate_tracklets(bad,images,masks,cv)
    many=[track('a'+str(i),0) for i in range(257)]+[track('b',2)]
    with pytest.raises(ValueError,match='endpoint budget'):reassociate_tracklets(many,images,masks,cv)


def test_repeated_close_texture_seeds_are_ambiguous():
    y,x=np.indices((160,192));a=(((x//2+y//2)%2)*200+20).astype(np.uint8)
    masks=[np.full_like(a,255)]*3
    r=reassociate_tracklets([track('a',0),track('b',2),track('c',2,(74.5,70.5))],
        [a,np.zeros_like(a),a],masks,cv)
    assert not r['links']


def test_index_matches_dense_radius_oracle_thresholds_ties_negative_random():
    rng=np.random.default_rng(42)
    points=np.concatenate([rng.uniform(-40,40,(300,2)),np.array([[0,0],[2,0],[-2,0],[6,0],[0,6],[-5,0],[16,0],[16.000001,0],[0,0]])])
    budget=module._DistanceBudget(2000000);index=module._SpatialIndex(points,budget)
    for query in [np.array([0.,0.]),np.array([-16.,-16.]),*rng.uniform(-45,45,(40,2))]:
        for radius in (1.,2.,4.,5.,6.,16.):
            dense=np.linalg.norm(points-query,axis=1)
            expected=sorted([(float(d),i) for i,d in enumerate(dense) if d<=radius],key=lambda item:(item[0],item[1]))
            assert index.query(query,radius)==expected


def test_index_counts_actual_norms_and_keeps_budget():
    budget=module._DistanceBudget(3)
    index=module._SpatialIndex([[0,0],[15,15],[1000,1000]],budget)
    assert index.query([0,0],1)==[(0.,0)]
    # Same bin contains one candidate outside the exact radius; it still costs a norm.
    assert budget.count==2
    with pytest.raises(ValueError,match='comparison budget'):index.query([0,0],1)
    assert budget.count==2


def test_neighbor_acceptance_matches_full_unbounded_dense_decision():
    rng=np.random.default_rng(51)
    cases=[np.array([[0,0],[4,0]]),np.array([[2,0],[6,0]]),np.array([[1,0],[5,0]]),
           np.array([[0,0],[3.99999,0]]),np.array([[0,0],[0,0]]),np.array([[-2,0],[2,0]])]
    cases.extend(rng.uniform(-20,20,(80,40,2)))
    for points in cases:
        for best_limit,radius in ((2.,6.),(1.,5.)):
            dense=sorted([(float(np.linalg.norm(p)),i) for i,p in enumerate(points)])
            dense_accept=dense[0][0]<=best_limit and (len(dense)<2 or dense[1][0]-dense[0][0]>=4)
            local=module._SpatialIndex(points,module._DistanceBudget(2000000)).query([0,0],radius)
            indexed_accept=bool(local) and local[0][0]<=best_limit and (len(local)<2 or local[1][0]-local[0][0]>=4)
            assert dense_accept==indexed_accept
            if indexed_accept:assert local[0][1]==dense[0][1]


class DenseOracle:
    def __init__(self,points,budget,cell_size=16.):self.points=np.asarray(points,dtype=float);self.budget=budget
    def query(self,xy,radius):
        self.budget.add(len(self.points))
        ds=np.linalg.norm(self.points-np.asarray(xy),axis=1)
        return sorted([(float(d),i) for i,d in enumerate(ds) if d<=radius],key=lambda item:(item[0],item[1]))


@pytest.mark.parametrize('variant',['simple','ambiguous','dense','competing'])
def test_index_same_links_as_independent_dense_oracle(monkeypatch,variant):
    images,masks=scene();tracks=pair()
    if variant=='ambiguous':tracks.append(track('near',2,(73.5,71.5)))
    elif variant=='competing':
        images.append(images[2].copy());masks.append(masks[2]);tracks.append(track('later',3,(72.5,71.5)))
    elif variant=='dense':
        for n,(x,y) in enumerate([(x,y) for y in range(20,145,16) for x in range(20,175,16)]):
            if abs(x-70)<10 and abs(y-70)<10:continue
            tracks.extend([track('old'+str(n),0,(x+.5,y+.5)),track('new'+str(n),2,(x+2.5,y+1.5))])
    indexed=reassociate_tracklets(tracks,images,masks,cv)
    monkeypatch.setattr(module,'_SpatialIndex',DenseOracle)
    dense=reassociate_tracklets(tracks,images,masks,cv)
    assert indexed['links']==dense['links']
    assert indexed['joined_tracks']==dense['joined_tracks']
    assert indexed['rejection_counts']==dense['rejection_counts']
    if variant=='dense':assert indexed['endpoint_comparisons']<dense['endpoint_comparisons']


def test_real_kernel_distance_budget_fail_closed(monkeypatch):
    images,masks=scene()
    monkeypatch.setitem(PROFILE,'max_comparisons',0)
    with pytest.raises(ValueError,match='comparison budget'):reassociate_tracklets(pair(),images,masks,cv)
