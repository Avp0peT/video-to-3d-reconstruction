import copy
import numpy as np
import pytest
from integrations.mvs_surface.silhouette_multiview import tighten_silhouette_intervals as tighten
from integrations.mvs_surface.silhouette_intervals import estimate_silhouette_intervals


def scene():
    shape=(101,101)
    ref=dict(view_id=0,image=np.zeros((*shape,3),np.uint8),mask=np.ones(shape,bool),depth=np.full(shape,5.,np.float32),K=np.array([[50.,0,48.],[0,50.,48.],[0,0,1.]]),R=np.eye(3),C=np.zeros(3))
    src=[]
    for i in range(1,6):
        s=copy.deepcopy(ref); s['view_id']=i; s['C']=np.array([2.,0,0]); s['mask'][:]=False; s['mask'][:,23:34]=True
        src.append(s)
    roi=np.zeros(shape,bool); roi[48,48]=True
    return ref,src,roi


def run(ref,src,roi): return tighten(ref,src,roi,[2.,10.])


def test_added_view_narrows_with_exact_baseline_counterfactual():
    ref,src,roi=scene(); src[3]['mask'][:]=False; src[3]['mask'][:,25:32]=True
    result=run(ref,src,roi)
    old=estimate_silhouette_intervals(ref,src[:3]+[src[-1]],roi,[2.,10.])
    for key in ['reason','front_depth','back_depth','sample_counts']:
        assert result['baseline_'+key].tobytes()==old[key].tobytes()
    assert result['front_depth'][48,48]>old['front_depth'][48,48]
    assert result['back_depth'][48,48]<old['back_depth'][48,48]
    stages=result['stage_counts']
    assert stages[1]['surviving_samples']<stages[0]['surviving_samples']
    assert stages[1]['bounded']==1
    assert result['leave_one_out']['4']['background_samples']>0
    assert result['whole_depth_source_classifications']['4']['evaluated']==1


def test_added_view_eliminates():
    ref,src,roi=scene(); src[3]['mask'][:]=False
    result=run(ref,src,roi)
    assert result['baseline_summary']['bounded']==1
    assert result['counts']['bounded_pixels']==0
    assert result['stage_counts'][-1]['no_survivor']==1


def test_unbounded_to_bounded():
    ref,src,roi=scene()
    for s in src[:3]: s['mask'][:]=True
    result=run(ref,src,roi)
    assert result['baseline_summary']['unbounded']==1
    assert result['stage_counts'][-1]['bounded']==1
    assert result['stage_counts'][-1]['surviving_samples']<128


def test_unknown_does_not_pass_and_loo_distinguishes_unknown():
    ref,src,roi=scene(); src[3]['validity']=np.zeros(roi.shape,bool)
    result=run(ref,src,roi)
    assert not result['bounded_mask'].any()
    assert result['leave_one_out']['4']['unknown_samples']>0
    assert result['leave_one_out']['4']['background_samples']==0


def test_disjoint_rejected():
    ref,src,roi=scene(); src[3]['mask'][:,27:30]=False
    result=run(ref,src,roi)
    assert result['reason'][48,48]==2
    assert result['stage_counts'][-1]['disjoint']==1


def test_final_source_only_diagnostic():
    ref,src,roi=scene(); a=run(ref,src,roi); src[-1]['mask'][:]=False; b=run(ref,src,roi)
    for key in a:
        if isinstance(a[key],np.ndarray): np.testing.assert_array_equal(a[key],b[key])
    assert a['stage_counts']==b['stage_counts']
    assert a['leave_one_out']==b['leave_one_out']
    assert a['withheld_sample_counts']['support']>0
    assert b['withheld_sample_counts']['support']==0
    for endpoint in b['withheld_endpoint_counts'].values():
        assert endpoint['foreground']+endpoint['background']+endpoint['unknown']==endpoint['evaluated']==1


def test_input_immutability_zero_native_depth_and_grid():
    ref,src,roi=scene(); roi[48,49]=True
    for s in src: s['depth'][:]=0
    before=copy.deepcopy((ref,src,roi)); result=run(ref,src,roi)
    assert result['bounded_mask'][48,48]
    assert not result['sampled_mask'][48,49]
    assert 'unknown geometry' in result['depth_semantics']
    for old,new in zip([before[0]]+before[1],[ref]+src):
        assert old.keys()==new.keys()
        for key in old:
            if isinstance(old[key],np.ndarray): np.testing.assert_array_equal(old[key],new[key])
    np.testing.assert_array_equal(before[2],roi)
    assert not result['surface_accepted'] and not result['geometry_accepted']


@pytest.mark.parametrize('mode',['duplicate','too_many','too_few'])
def test_sources_rejected(mode):
    ref,src,roi=scene()
    if mode=='duplicate': src[3]['view_id']=src[0]['view_id']
    elif mode=='too_few': src=src[:4]
    else:
        src=[copy.deepcopy(src[0]) for _ in range(16)]
        for i,s in enumerate(src): s['view_id']=i+1
    with pytest.raises(ValueError): run(ref,src,roi)


def test_fifteen_sources_monotonic_samples():
    ref,src,roi=scene()
    src=[copy.deepcopy(src[0]) for _ in range(15)]
    for i,s in enumerate(src): s['view_id']=i+1
    result=run(ref,src,roi)
    samples=[s['surviving_samples'] for s in result['stage_counts']]
    assert len(samples)==12 and samples==sorted(samples,reverse=True)
    assert result['admission_source_ids']==list(range(1,15))
    assert result['withheld_source_id']==15


def test_inspectable_matched_sample_sets_and_spans():
    ref,src,roi=scene(); roi[45:52,45:52]=True
    src[3]['mask'][:]=False; src[3]['mask'][:,25:32]=True
    result=run(ref,src,roi)
    base=result['baseline_surviving_samples']; final=result['final_surviving_samples']
    assert base.dtype==bool and base.shape==final.shape==(len(result['ray_x']),128)
    assert not (final & ~base).any()
    x,y=result['ray_x'],result['ray_y']
    np.testing.assert_array_equal(final.sum(axis=1),result['sample_counts'][y,x])
    np.testing.assert_array_equal(base.sum(axis=1),result['baseline_sample_counts'][y,x])
    assert result['monotonicity']==dict(no_sample_gains=True,sample_gains_across_stages=0,final_samples_outside_baseline=0)
    assert (result['final_surviving_span'][y,x] <= result['baseline_surviving_span'][y,x]).all()
    for index,(px,py) in enumerate(zip(x,y)):
        ids=np.flatnonzero(final[index])
        expected=0 if len(ids)<2 else result['depth_samples'][ids[-1]]-result['depth_samples'][ids[0]]
        assert result['final_surviving_span'][py,px]==expected
    src[-1]['validity']=np.zeros(roi.shape,bool)
    changed=run(ref,src,roi)
    for key in ['ray_x','ray_y','baseline_surviving_samples','final_surviving_samples','baseline_surviving_span','final_surviving_span']:
        np.testing.assert_array_equal(result[key],changed[key])
    assert result['monotonicity']==changed['monotonicity']
