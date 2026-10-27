from copy import deepcopy
import cv2
import numpy as np
import pytest
from integrations.mvs_surface.local_depth_search import propose_local_depth,validate_depth


def scene(count=2,uniform=False):
    h,w=64,96; rng=np.random.default_rng(45)
    image=rng.integers(0,256,(h,w,3),dtype=np.uint8)
    if uniform: image[:]=127
    K=np.array([[80.,0,48],[0,80,32],[0,0,1]])
    def view(vid,center,img):
        return dict(view_id=vid,K=K.copy(),R=np.eye(3),C=np.array([center,0.,0.]),image=img,mask=np.ones((h,w),bool),depth=np.full((h,w),2.,np.float32))
    reference=view(0,0,image)
    sources=[]
    for i,b in enumerate([.2,-.2,.4,-.4][:count]):
        im=cv2.warpAffine(image,np.float32([[1,0,-80*b/2],[0,1,0]]),(w,h))
        sources.append(view(i+1,b,im))
    roi=np.zeros((h,w),bool); roi[24:40,36:60]=True
    return reference,sources,roi


def test_textured_plane_nonzero_disparity_recovery_and_no_mutation():
    ref,sources,roi=scene(); old=deepcopy(ref); sourceold=deepcopy(sources)
    r=propose_local_depth(ref,sources,roi,[1.,4.])
    assert r['accepted'].sum()>100
    assert np.max(np.abs(r['proposal_depth'][r['accepted']]-2)/2)<.02
    assert not r['proposal_depth'][~r['accepted']].any()
    assert r['prior_used'][roi].all()
    for k in ['K','R','C','image','mask','depth']: assert np.array_equal(ref[k],old[k])
    for s,o in zip(sources,sourceold):
        for k in ['image','mask','depth']: assert np.array_equal(s[k],o[k])


def test_uniform_texture_and_missing_mask_patch_rejected():
    ref,sources,roi=scene(uniform=True)
    r=propose_local_depth(ref,sources,roi,[1.,4.])
    assert not r['accepted'].any()
    ref,sources,roi=scene()
    for s in sources: s['mask'][:]=False
    assert not propose_local_depth(ref,sources,roi,[1.,4.])['accepted'].any()


def test_holes_and_whole_depth_unchanged():
    ref,sources,roi=scene()
    ref['mask'][30:34,45:49]=False; roi &= ref['mask']
    old=ref['depth'].copy()
    r=propose_local_depth(ref,sources,roi,[1.,4.])
    assert not r['accepted'][28:36,43:51].any()
    assert np.array_equal(ref['depth'],old)


def test_bad_depth_and_two_conflicts_override_two_supports():
    ref,sources,roi=scene(4)
    for s in sources[2:]: s['depth'][:]=3
    v=validate_depth(ref,sources,ref['depth'],roi)
    assert (v['support_count'][roi]==2).all()
    assert (v['conflict_count'][roi]==2).all()
    r=propose_local_depth(ref,sources,roi,[1.,4.])
    assert not r['accepted'].any()
    assert (r['conflict_count'][roi]>=2).all()


def test_wrong_prior_does_not_limit_wide_search():
    ref,sources,roi=scene(); ref['depth'][:]=1.
    r=propose_local_depth(ref,sources,roi,[1.,4.])
    assert not r['prior_used'].any()
    assert r['accepted'].sum()>100
    assert np.max(np.abs(r['raw_best_depth'][r['accepted']]-2)/2)<.02


def test_missing_source_depth_unknown_not_conflict():
    ref,sources,roi=scene()
    for s in sources: s['depth'][:]=0
    v=validate_depth(ref,sources,ref['depth'],roi)
    assert not v['conflict_count'].any() and not v['support_count'].any()
    assert (v['unknown_count'][roi]==2).all()
    assert not propose_local_depth(ref,sources,roi,[1.,4.])['accepted'].any()


@pytest.mark.parametrize('bad',['bool_id','dup','same_id','badR','nanK','boolK','outside_roi','nan_depth','wrongmask'])
def test_invalid_inputs(bad):
    ref,sources,roi=scene()
    if bad=='bool_id': ref['view_id']=True
    elif bad=='dup': sources[1]['view_id']=sources[0]['view_id']
    elif bad=='same_id': sources[0]['view_id']=0
    elif bad=='badR': ref['R'][0,0]=2
    elif bad=='nanK': ref['K'][0,0]=np.nan
    elif bad=='boolK': ref['K']=np.eye(3,dtype=bool)
    elif bad=='outside_roi': ref['mask'][25,40]=False
    elif bad=='nan_depth': sources[0]['depth'][0,0]=np.nan
    else: sources[0]['mask']=np.zeros((3,4),bool)
    with pytest.raises(ValueError): propose_local_depth(ref,sources,roi,[1.,4.])


@pytest.mark.parametrize('bounds',[[True,4.],[0.,4.],[4.,1.],[1.,float('inf')]])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError): propose_local_depth(*scene(),bounds)


def test_check_called_and_plane_bound():
    calls=[]
    r=propose_local_depth(*scene(),[1.,4.],policy={'planes':16},check=lambda:calls.append(1))
    assert len(calls)>=16*3
    assert r['counters']['photo_volume_elements']==16*384
    with pytest.raises(ValueError): propose_local_depth(*scene(),[1.,4.],policy={'planes':129})


def test_source_grids_can_differ_and_unavailable_patch_is_not_black_evidence():
    ref,sources,roi=scene()
    for source in sources:
        source['image']=source['image'][:,8:88].copy()
        source['mask']=source['mask'][:,8:88].copy()
        source['depth']=source['depth'][:,8:88].copy()
        source['K'][0,2]-=8
    r=propose_local_depth(ref,sources,roi,[1.,4.])
    assert r['accepted'].sum()>100


def test_insufficient_alternative_and_mixed_boolean_camera_rejected():
    ref,sources,roi=scene()
    r=propose_local_depth(ref,sources,roi,[1.,4.],policy={'planes':2})
    assert not r['accepted'].any()
    ref['K']=ref['K'].tolist(); ref['K'][0][1]=False
    with pytest.raises(ValueError): propose_local_depth(ref,sources,roi,[1.,4.])


def test_timeout_check_propagates_without_mutation():
    ref,sources,roi=scene(); old=ref['depth'].copy()
    def stop(): raise TimeoutError('test budget')
    with pytest.raises(TimeoutError): propose_local_depth(ref,sources,roi,[1.,4.],check=stop)
    assert np.array_equal(ref['depth'],old)


def test_repeated_textured_pattern_rejected_as_non_unique():
    ref,sources,roi=scene()
    rng=np.random.default_rng(9)
    stripe=rng.integers(0,256,(64,4,3),dtype=np.uint8)
    ref['image']=np.tile(stripe,(1,24,1))
    for source in sources: source['image']=ref['image'].copy()
    r=propose_local_depth(ref,sources,roi,[1.,4.])
    assert not r['accepted'].any()
    assert (r['reason'][roi]==2).any()


def test_isolated_strong_peak_with_measured_low_correlation_alternatives_passes():
    ref,sources,roi=scene(); ref['depth'][:]=0
    # Four inverse-depth planes include z=2 exactly. Other planes have real
    # textured patches, but their disparities have very low correlation.
    r=propose_local_depth(ref,sources,roi,[1.,4.],policy={'planes':4})
    assert r['accepted'].sum()>100
    assert np.allclose(r['proposal_depth'][r['accepted']],2.)
    assert np.isfinite(r['uniqueness_gap'][r['accepted']]).all()
    assert (r['uniqueness_gap'][r['accepted']]>=.01).all()


def test_unmeasurable_alternatives_are_not_low_correlation_evidence():
    ref,sources,roi=scene()
    roi[:]=False; roi[32,48]=True
    for source,center in zip(sources,[40,56]):
        source['mask'][:]=False
        source['mask'][28:37,center-4:center+5]=True
    r=propose_local_depth(ref,sources,roi,[1.,4.],policy={'planes':4})
    assert r['raw_best_depth'][32,48]==2.
    assert not r['accepted'].any()
    assert r['reason'][32,48]==2
    assert np.isneginf(r['uniqueness_gap'][32,48])


def test_same_source_photo_geometry_intersection_is_required():
    ref,sources,roi=scene(4)
    # Views 1/2 supply photometric matches but no valid geometric agreement.
    # Views 3/4 supply geometry while their constant images cannot supply NCC.
    sources[0]['depth'][:]=3
    sources[1]['depth'][:]=0
    for source in sources[2:]: source['image'][:]=127
    r=propose_local_depth(ref,sources,roi,[1.,4.],policy={'planes':4})
    assert (r['photo_support_count'][roi]==2).all()
    assert (r['support_count'][roi]==2).all()
    assert (r['conflict_count'][roi]==1).all()
    assert not r['joint_support_count'].any()
    assert not r['accepted'].any()
    assert (r['reason'][roi]==6).all()
    assert r['source_ids']==[1,2,3,4]
    assert (r['validation']['support_bits'][roi]==12).all()
    assert (r['photo_support_bits'][roi]==3).all()


def test_joint_bits_and_counts_for_true_plane_and_two_conflicts():
    ref,sources,roi=scene()
    r=propose_local_depth(ref,sources,roi,[1.,4.])
    assert (r['joint_support_count'][r['accepted']]>=2).all()
    assert (r['photo_support_count'][r['accepted']]>=2).all()
    assert r['source_ids']==r['validation']['source_ids']
    ref,sources,roi=scene(4)
    for source in sources[:2]: source['depth'][:]=3
    for source in sources[2:]: source['image'][:]=127
    assert not propose_local_depth(ref,sources,roi,[1.,4.],policy={'planes':4})['accepted'].any()
