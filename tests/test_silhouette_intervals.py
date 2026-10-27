import copy
import numpy as np
import pytest
from integrations.mvs_surface.silhouette_intervals import estimate_silhouette_intervals as estimate


def scene():
    shape = (101,101)
    k = np.array([[50.,0,48.],[0,50.,48.],[0,0,1.]])
    ref = dict(view_id=0,image=np.zeros((*shape,3),np.uint8),mask=np.ones(shape,bool),validity=np.ones(shape,bool),depth=np.full(shape,5.,np.float32),K=k,R=np.eye(3),C=np.zeros(3))
    sources=[]
    for i in range(1,5):
        s=copy.deepcopy(ref); s['view_id']=i; s['C']=np.array([2.,0,0])
        s['mask'][:]=False; s['mask'][:,23:34]=True
        sources.append(s)
    roi=np.zeros(shape,bool); roi[48,48]=True
    return ref,sources,roi


def run(ref,src,roi):
    return estimate(ref,src,roi,[2.,10.])


def test_translated_box_silhouette_bounds():
    ref,src,roi=scene(); result=run(ref,src,roi)
    assert result['bounded_mask'][48,48]
    # Pixel-centre nearest sampling gives angular bounds u=22.5 and 33.5.
    assert abs(result['front_depth'][48,48]-100/25.5)<.13
    assert abs(result['back_depth'][48,48]-100/14.5)<.23
    assert result['counts']['bounded_pixels']==1
    assert not result['geometry_accepted']
    assert result['admission_source_ids']==[1,2,3]
    for name in ['front','back']:
        assert result[name+'_transition_low'][48,48] <= result[name+'_depth'][48,48] <= result[name+'_transition_high'][48,48]
    assert 'whole_depth_outside_midpoint_estimate' in result['counts']


def test_uniform_foreground_unbounded():
    ref,src,roi=scene()
    for s in src: s['mask'][:]=True
    result=run(ref,src,roi)
    assert result['reason'][48,48]==3
    assert not result['bounded_mask'].any()


def test_empty_masks():
    ref,src,roi=scene()
    for s in src: s['mask'][:]=False
    assert run(ref,src,roi)['reason'][48,48]==1


def test_invalid_validity_is_not_pass():
    ref,src,roi=scene(); src[0]['validity'][:]=False
    assert not run(ref,src,roi)['bounded_mask'].any()


def test_disjoint_segments_rejected():
    ref,src,roi=scene()
    for s in src: s['mask'][:,27:30]=False
    assert run(ref,src,roi)['reason'][48,48]==2


def test_withheld_only_changes_diagnostic_and_no_mutation():
    ref,src,roi=scene(); before=copy.deepcopy((ref,src,roi))
    a=run(ref,src,roi)
    for old,new in zip([before[0]]+before[1],[ref]+src):
        for key in old:
            if isinstance(old[key],np.ndarray): np.testing.assert_array_equal(old[key],new[key])
    np.testing.assert_array_equal(before[2],roi)
    src[-1]['mask'][:]=False
    b=run(ref,src,roi)
    for key in ['front_depth','back_depth','interval_width','bounded_mask','reason','sample_counts']:
        np.testing.assert_array_equal(a[key],b[key])
    assert a['withheld_sample_counts']['support']>0
    assert b['withheld_sample_counts']['support']==0
    assert b['withheld_sample_counts']['known_background']>0
    for endpoint in b['withheld_endpoint_counts'].values():
        assert sum(endpoint[k] for k in ['foreground','background','unknown'])==endpoint['evaluated']==1


def test_reference_background_never_output_and_no_interior_fill():
    ref,src,roi=scene(); roi[48,49]=True
    result=run(ref,src,roi)
    assert not result['sampled_mask'][48,49]
    assert result['front_depth'][48,49]==0
    ref['mask'][48,48]=False; roi[48,48]=False
    result=run(ref,src,roi)
    assert not result['bounded_mask'].any()


def test_unknown_bracket_not_background():
    ref,src,roi=scene()
    for s in src: s['validity'][:,22]=False
    assert run(ref,src,roi)['reason'][48,48]==4


@pytest.mark.parametrize('count',[2,3])
def test_source_count(count):
    ref,src,roi=scene()
    with pytest.raises(ValueError): run(ref,src[:count],roi)


def test_varied_translations_and_rotation():
    from integrations.mvs_surface.local_depth_search import _project, _rays
    ref,src,roi=scene()
    for s,translation,angle in zip(src,[1.6,2.0,2.4,1.8],[.02,-.03,.04,-.01]):
        s['C']=np.array([translation,0.,0.])
        c,t=np.cos(angle),np.sin(angle)
        s['R']=np.array([[c,0.,t],[0.,1.,0.],[-t,0.,c]])
        rays=np.repeat(_rays(ref,[48],[48]),2,axis=0)
        uv,_,_= _project(ref,s,rays,np.array([4.,7.]))
        s['mask'][:]=False
        s['mask'][:,int(np.ceil(uv[:,0].min())):int(np.floor(uv[:,0].max()))+1]=True
    result=run(ref,src,roi)
    assert result['bounded_mask'][48,48]
    assert abs(result['front_depth'][48,48]-4.)<.35
    assert abs(result['back_depth'][48,48]-7.)<.55
    assert result['withheld_endpoint_counts']['front']['evaluated']==1
    assert result['withheld_endpoint_counts']['back']['evaluated']==1
    for endpoint in result['withheld_endpoint_counts'].values():
        assert sum(endpoint[k] for k in ['foreground','background','unknown'])==1


def test_missing_validity_defaults_without_mutating_input_keys_or_arrays():
    ref,src,roi=scene()
    for view in [ref]+src: del view['validity']
    before=copy.deepcopy((ref,src,roi))
    result=run(ref,src,roi)
    assert result['bounded_mask'][48,48]
    for old,new in zip([before[0]]+before[1],[ref]+src):
        assert old.keys()==new.keys()
        assert 'validity' not in new
        for key in old:
            if isinstance(old[key],np.ndarray): np.testing.assert_array_equal(old[key],new[key])
    np.testing.assert_array_equal(before[2],roi)
