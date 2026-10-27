from copy import deepcopy
import numpy as np
import pytest
from integrations.mvs_surface.depth_coverage import classify_depth_coverage


def views():
    out=[]
    for i,c in enumerate((-.08,0.,.08)):
        out.append(dict(view_id=i,image=np.full((32,32,3),100,np.uint8),mask=np.ones((32,32),bool),validity=np.ones((32,32),bool),depth=np.full((32,32),2.,np.float32),depth_available=True,K=np.array([[40.,0,16],[0,40.,16],[0,0,1]]),R=np.eye(3),C=np.array([c,0,0])))
    return out


def test_plane_native_samples_and_origin_exclusion_immutable():
    v=views();before=deepcopy(v)
    result=classify_depth_coverage(v)
    assert result['status']=='NATIVE_DEPTH_COVERAGE_CLASSIFIED_NOT_GEOMETRY_ACCEPTED'
    assert result['cross_view_evaluated']
    assert result['coverage_view_1'][15,15]==5
    assert result['coverage_view_1'][14,14]==3
    assert result['summary']['total_valid_mask_foreground_pixels']==3072
    assert result['summary']['total_foreground_without_native_depth']==0
    assert result['summary']['processed_sampled_native_pixels']==363
    assert result['source_uv'].shape==(363,2)
    for i,iid in enumerate(result['view_ids']):
        own=result['source_view_ids']==iid
        assert (result['audit_state'][own,i]==0).all()
        xy=result['source_uv'][own]
        rays=np.column_stack((xy,np.ones(len(xy))))@np.linalg.inv(v[i]['K']).T
        np.testing.assert_array_equal(result['points'][own],(rays*2)@v[i]['R']+v[i]['C'])
    for a,b in zip(v,before):
        for key in ('image','mask','validity','depth','K','R','C'):np.testing.assert_array_equal(a[key],b[key])


def test_validity_background_and_foreground_missing_have_different_denominators():
    v=views();v[0]['mask'][:8]=False;v[0]['validity'][8:10]=False;v[0]['depth'][10:12]=0
    r=classify_depth_coverage(v)
    image=r['coverage_view_0'];s=r['summary']['per_view'][0]
    assert (image[:8]==0).all()
    assert (image[8:10]==1).all()
    assert (image[10:12]==2).all()
    assert s['foreground_without_native_depth']==64
    assert s['valid_mask_foreground_pixels']==704
    assert s['background_pixels']==256
    assert sum(s['coverage_map_counts'].values())==1024


def test_missing_native_depth_no_source_samples_and_unknown_evidence():
    v=views();v[2]['depth_available']=False;v[2]['depth'][:]=0
    r=classify_depth_coverage(v)
    assert (r['coverage_view_2']==2).all()
    assert not (r['source_view_ids']==2).any()
    assert not r['audit_supported'].any()
    assert r['summary']['total_foreground_without_native_depth']==1024
    assert r['summary']['per_view'][2]['processed_sampled_native_pixels']==0


def test_off_plane_depth_distinguishes_free_space_conflict_and_occlusion():
    v=views();v[2]['depth'][:]=3.
    r=classify_depth_coverage(v)
    assert r['coverage_view_1'][15,15]==6
    assert r['coverage_view_2'][15,15]==7
    assert r['audit_free_space_contradiction_count'].max()>0
    assert r['audit_occluded_count'].max()>0


def test_capacity_fails_explicitly_without_truncating_a_sample_set():
    r=classify_depth_coverage(views(),max_points=10)
    assert r['status']=='CAPACITY_LIMIT'
    assert not r['cross_view_evaluated']
    assert r['capacity_required']==363
    assert r['summary']['processed_sampled_native_pixels']==0
    assert r['points'].shape==(0,3)
    assert (r['coverage_view_0']==3).all()


def test_duplicate_ids_explicit_masks_and_parameter_validation():
    v=views()
    with pytest.raises(ValueError,match='duplicate'):classify_depth_coverage([v[0],v[0]])
    del v[0]['validity']
    with pytest.raises(ValueError,match='validity'):classify_depth_coverage(v)
    with pytest.raises(ValueError):classify_depth_coverage(views(),stride=True)
    with pytest.raises(ValueError):classify_depth_coverage(views(),max_points=200001)


def test_empty_native_depth_and_single_view_are_not_supported_or_missing_background():
    v=views();v[0]['mask'][:]=False
    r=classify_depth_coverage([v[0]])
    assert r['points'].shape==(0,3)
    assert r['summary']['total_valid_mask_foreground_pixels']==0
    assert r['summary']['total_foreground_without_native_depth']==0
    r=classify_depth_coverage([v[1]])
    assert (r['sampled_coverage_classification']==4).all()
    assert not r['audit_supported'].any()


def test_67_view_coverage_explicit_capacity_and_no_truncation():
    base=views()[1];selected=[]
    for i in range(67):
        v=deepcopy(base);v['view_id']=i;selected.append(v)
    limited=classify_depth_coverage(selected,stride=32,max_points=10)
    assert limited['status']=='CAPACITY_LIMIT' and limited['capacity_required']==67
    assert not limited['cross_view_evaluated'] and not len(limited['points'])
    r=classify_depth_coverage(selected,stride=32)
    assert len(r['view_ids'])==67 and len(r['points'])==67
    assert r['audit_state'].shape==(67,67)
    assert r['summary']['processed_sampled_native_pixels']==67
    for index in range(67):assert 'coverage_view_'+str(index) in r


def test_129_coverage_views_rejected():
    with pytest.raises(ValueError,match='128'):classify_depth_coverage([views()[0]]*129)
