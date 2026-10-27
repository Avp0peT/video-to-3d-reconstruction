from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.video_masks.motion_window_linking import associate_pair


def windows(n=2,count=4):
    left=dict(window_id=0,frames=[]); right=dict(window_id=1,frames=[])
    coords=[[10,10],[20,10],[10,20],[20,20]][:count]
    for f in range(n):
        mask=np.zeros((50,70),bool); mask[5:30,5:30]=True
        metadata=dict(frame_id='frame_'+str(f),time_seconds=float(f),image_sha256='a'*64,whole_sha256='b'*64)
        left['frames'].append(dict(metadata,parts={'left_part':mask.copy()},tracks={'left_part':{i:list(xy) for i,xy in enumerate(coords)}}))
        right['frames'].append(dict(metadata,parts={'other_local_id':mask.copy()},tracks={'other_local_id':{100+i:list(xy) for i,xy in enumerate(coords)}}))
    return left,right


def test_different_local_ids_coordinate_evidence_and_no_mutation():
    left,right=windows(); a=deepcopy(left); b=deepcopy(right)
    r=associate_pair(left,right)
    assert r['accepted_links'][0]['track_pairs']==[[0,100],[1,101],[2,102],[3,103]]
    assert not r['geometry_accepted']
    json.dumps(r,allow_nan=False)
    for current,original in [(left,a),(right,b)]:
        for f,of in zip(current['frames'],original['frames']):
            assert f['tracks']==of['tracks']
            for part in f['parts']: assert np.array_equal(f['parts'][part],of['parts'][part])


def test_same_local_ids_do_not_mean_same_identity():
    left,right=windows()
    for f in right['frames']:
        f['parts']={'left_part':f['parts']['other_local_id']}
        f['tracks']={'left_part':{i:[x+8,y] for i,(x,y) in enumerate(f['tracks']['other_local_id'].values())}}
    assert not associate_pair(left,right)['accepted_links']


def test_one_shared_frame_is_insufficient():
    assert not associate_pair(*windows(1))['accepted_links']


@pytest.mark.parametrize('key,value',[('time_seconds',.1),('image_sha256','c'*64),('whole_sha256','c'*64)])
def test_shared_identity_drift_is_error(key,value):
    left,right=windows(); right['frames'][0][key]=value
    with pytest.raises(ValueError): associate_pair(left,right)


def test_temporal_track_identity_exchange_is_not_merged():
    left,right=windows()
    points=right['frames'][1]['tracks']['other_local_id']
    points[100],points[101]=points[101],points[100]
    assert not associate_pair(left,right)['accepted_links']


@pytest.mark.parametrize('count',[2,3])
def test_too_few_pairs(count):
    assert not associate_pair(*windows(count=count))['accepted_links']


def test_collinear_tracks_rejected():
    left,right=windows()
    for w in [left,right]:
        for f in w['frames']:
            for i,xy in enumerate(next(iter(f['tracks'].values())).values()): xy[:]=[10+i*3,15]
    r=associate_pair(left,right)
    assert r['pairs'][0]['reason']=='DEGENERATE_TRACK_GEOMETRY'


def test_same_frame_nearest_tie_rejects_pair():
    left,right=windows()
    right['frames'][0]['tracks']['other_local_id'][104]=[10,10]
    r=associate_pair(left,right)
    assert r['pairs'][0]['reason']=='AMBIGUOUS_NEAREST_POINTS'


def test_one_to_many_all_eligible_links_rejected():
    left,right=windows()
    for lf,rf in zip(left['frames'],right['frames']):
        whole=np.zeros((50,70),bool); whole[5:30,5:55]=True
        a=whole.copy(); a[:,30:]=False
        b=whole&~a
        leftpoints=[[10,10],[20,10],[10,20],[20,20],[35,10],[45,10],[35,20],[45,20]]
        lf['parts']={'left_part':whole}; lf['tracks']={'left_part':{i:xy for i,xy in enumerate(leftpoints)}}
        rf['parts']={'r1':a,'r2':b}; rf['tracks']={'r1':{i+100:xy for i,xy in enumerate(leftpoints[:4])},'r2':{i+200:xy for i,xy in enumerate(leftpoints[4:])}}
    r=associate_pair(left,right)
    assert not r['accepted_links']
    assert len(r['pairs'])==2
    assert all(p['reason']=='AMBIGUOUS_PART_ASSOCIATION' for p in r['pairs'])


def test_near_coordinates_but_inconsistent_displacements_rejected():
    left,right=windows()
    for f,frame in enumerate(right['frames']):
        for xy in frame['tracks']['other_local_id'].values(): xy[0]+=-1 if f==0 else 1
    r=associate_pair(left,right)
    assert r['pairs'][0]['reason']=='DISPLACEMENT_INCONSISTENT'
    assert all(c['max_displacement_difference']==2 for c in r['pairs'][0]['displacement_checks'])


@pytest.mark.parametrize('mutation',['bool_window','bool_part','bool_track','nan','outside','badshape','missing_tracks','repeat_frame','unordered','bad_hash','overlap','too_many_parts'])
def test_invalid_inputs(mutation):
    left,right=windows(); frame=left['frames'][0]
    if mutation=='bool_window': left['window_id']=True
    elif mutation=='bool_part': frame['parts']={True:frame['parts']['left_part']}; frame['tracks']={True:frame['tracks']['left_part']}
    elif mutation=='bool_track': frame['tracks']['left_part']={True:[10,10]}
    elif mutation=='nan': frame['tracks']['left_part'][0]=[float('nan'),10]
    elif mutation=='outside': frame['tracks']['left_part'][0]=[0,0]
    elif mutation=='badshape': frame['parts']['left_part']=np.zeros((50,),bool)
    elif mutation=='missing_tracks': frame.pop('tracks')
    elif mutation=='repeat_frame': left['frames'][1]['frame_id']=frame['frame_id']
    elif mutation=='unordered': left['frames'][1]['time_seconds']=-1
    elif mutation=='bad_hash': frame['image_sha256']='nope'
    elif mutation=='overlap': frame['parts']['extra']=frame['parts']['left_part'].copy(); frame['tracks']['extra']={}
    else:
        frame['parts']={str(i):np.zeros((50,70),bool) for i in range(13)}; frame['tracks']={str(i):{} for i in range(13)}
    with pytest.raises(ValueError): associate_pair(left,right)


@pytest.mark.parametrize('policy',[{'min_track_pairs':3},{'min_shared_frames':True},{'point_radius':3},{'max_displacement_difference':2},{'min_mask_iou':.1},{'bad':1}])
def test_invalid_policy(policy):
    with pytest.raises(ValueError): associate_pair(*windows(),policy=policy)


def test_string_track_ids_are_local_and_grid_mismatch_rejected():
    left,right=windows()
    for f in right['frames']:
        f['tracks']['other_local_id']={str(k):v for k,v in f['tracks']['other_local_id'].items()}
    r=associate_pair(left,right)
    assert len(r['accepted_links'])==1
    assert r['left_window_id']==0 and r['right_window_id']==1
    assert r['pairs'][0]['frames'][0]['image_sha256']=='a'*64
    right['frames'][0]['parts']['other_local_id']=np.ones((51,70),bool)
    with pytest.raises(ValueError): associate_pair(left,right)

def test_missing_frame_identity_and_excess_tracks_rejected():
    left,right=windows()
    del left['frames'][0]['frame_id']
    with pytest.raises(ValueError): associate_pair(left,right)
    left,right=windows()
    left['frames'][0]['tracks']['left_part']={i:[10,10] for i in range(1201)}
    with pytest.raises(ValueError): associate_pair(left,right)

def test_every_common_active_frame_must_pass():
    left,right=windows(3)
    for xy in right['frames'][2]['tracks']['other_local_id'].values(): xy[0]+=4
    r=associate_pair(left,right)
    assert not r['accepted_links']
    assert len(r['pairs'][0]['frames'])==3

def test_multiple_unequal_neighbors_within_radius_allow_mutual_unique_nearest():
    from integrations.video_masks.motion_window_linking import _matching
    # Both right points lie within radius; only the closer one is reciprocal nearest.
    matches,ties=_matching({0:[10.0,10.0]},{100:[10.25,10.0],101:[11.5,10.0]},2.0)
    assert matches==[(0,100)]
    assert ties==[]
