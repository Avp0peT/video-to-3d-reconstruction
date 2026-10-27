import json
import cv2
import numpy as np
import pytest
from integrations.video_masks.motion_parts import propose_motion_parts, _domain


def scene(independent=False):
    rng=np.random.default_rng(42)
    base=rng.integers(0,256,(192,240,3),dtype=np.uint8)
    images=[]; masks=[]
    texture=rng.integers(0,256,(52,52,3),dtype=np.uint8)
    for t in range(4):
        im=base.copy(); mask=np.zeros((192,240),bool)
        for x,y,dx in ((25,30,4),(140,105,-4 if independent else 4)):
            xx=x+dx*t; im[y:y+52,xx:xx+52]=texture
            mask[y:y+52,xx:xx+52]=True
        images.append(im); masks.append(mask)
    return images,masks


def test_independent_tracks_and_bounds():
    images,masks=scene(True)
    result=propose_motion_parts(images,masks)
    assert result['status']=='MOTION_CANDIDATES_NOT_PHYSICAL_PARTS'
    assert 1 <= len(result['part_masks']) <= 4
    assert result['ledger']['seed_count']<=1200
    json.dumps(result['ledger'])
    for f,m in enumerate(masks):
        acc=np.zeros(m.shape,bool)
        for arrays in result['part_masks'].values():
            assert not (acc & arrays[f]).any()
            assert not (arrays[f] & ~m).any()
            acc |= arrays[f]
        assert np.array_equal(acc | result['fallback_masks'][f],m)
        assert np.array_equal(result['global_masks'][f],m)
        assert np.array_equal(result['unknown_masks'][f],m & ~result['evidence_domain_masks'][f])


def test_uniform_entire_scene_does_not_split():
    rng=np.random.default_rng(1)
    im=rng.integers(0,256,(160,192,3),dtype=np.uint8)
    images=[cv2.warpAffine(im,np.float32([[1,0,t*2],[0,1,t]]),(192,160)) for t in range(4)]
    masks=[np.ones((160,192),bool) for _ in images]
    result=propose_motion_parts(images,masks)
    assert not result['part_masks']


def test_low_texture_is_unknown_fallback():
    images=[np.zeros((48,64,3),np.uint8) for _ in range(3)]
    mask=np.ones((48,64),bool); mask[20:25,20:25]=False
    result=propose_motion_parts(images,[mask]*3)
    assert not result['part_masks']
    for f in range(3):
        assert np.array_equal(result['unknown_masks'][f],mask)
        assert np.array_equal(result['fallback_masks'][f],mask)


def test_geodesic_domain_preserves_holes_and_disconnection():
    mask=np.ones((25,30),bool); mask[:,14:16]=False
    owner=_domain(mask,[(10,12,0)],40)
    assert not (owner[:,16:]>=0).any()
    assert not (owner[~mask]>=0).any()


@pytest.mark.parametrize('policy',[{'max_points':True},{'max_frames':7},{'max_parts':5},{'fb_error':float('nan')},{'relative_motion':0},{'min_pair_support':1},{'bad':1}])
def test_invalid_policy(policy):
    with pytest.raises(ValueError):
        propose_motion_parts([np.zeros((32,32,3),np.uint8)],[np.ones((32,32),bool)],policy=policy)


def test_invalid_inputs():
    with pytest.raises(ValueError): propose_motion_parts([],[])
    with pytest.raises(ValueError): propose_motion_parts([np.zeros((32,32,3))],[np.ones((32,32),bool)])
    with pytest.raises(ValueError): propose_motion_parts([np.zeros((32,32,3),np.uint8)],[np.ones((32,32),np.uint8)])


def test_single_pair_not_confirmed_and_deterministic():
    images,masks=scene(True)
    result=propose_motion_parts(images[:2],masks[:2])
    assert not result['part_masks']
    assert result['status']=='NOT_TEMPORALLY_CONFIRMED'
    a=propose_motion_parts(images,masks,policy={'max_points':180,'max_parts':1})
    b=propose_motion_parts(images,masks,policy={'max_points':180,'max_parts':1})
    assert a['ledger']==b['ledger']
    assert a['ledger']['seed_count']<=180
    assert len(a['part_masks'])<=1


def test_common_local_movement_not_forced_into_two_parts():
    result=propose_motion_parts(*scene(False))
    assert len(result['part_masks'])<=1


def test_ids_are_track_bound_and_temporally_supported():
    result=propose_motion_parts(*scene(True))
    for entry in result['ledger']['parts']:
        assert entry['id']=='motion_%04d'%min(entry['track_ids'])
        assert len(entry['support_pairs'])>=2
    assert result['geometry_accepted'] is False


def test_global_fit_coordinates_within_masks():
    images,masks=scene(True)
    result=propose_motion_parts(images,masks)
    for pair in result['ledger']['pairs']:
        for key,f in [('source_points',pair['pair'][0]),('destination_points',pair['pair'][1])]:
            xy=np.rint(pair[key]).astype(int).reshape(-1,2)
            assert masks[f][xy[:,1],xy[:,0]].all()

def test_small_input_and_single_frame_unknown():
    image=np.zeros((8,8,3),np.uint8); mask=np.ones((8,8),bool)
    result=propose_motion_parts([image],[mask])
    assert result['unknown_masks'][0].all()
