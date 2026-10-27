import json
import pytest
from integrations.demo_pipeline import overlap_review as module
from integrations.demo_pipeline.temporal_review import fingerprint


def rig(monkeypatch,n=9):
    roots=[dict(region_id=f'r{i}') for i in range(n)]
    plan=dict(regions=roots,frame_domains={},sources=[],planned_connected_components=[],selection_counts={})
    monkeypatch.setattr(module,'build_connected_plan',lambda *a,**k:plan)
    monkeypatch.setattr(module,'prefix_plan',lambda p,n,**k:dict(p,regions=p['regions'][:n]))
    monkeypatch.setattr(module,'render_area_tracking',lambda *a,**k:[])
    monkeypatch.setattr(module,'render_overlap',lambda *a,**k:[])
    monkeypatch.setattr(module,'review_neighborhood',lambda *a,**k:(dict(status='NEIGHBORHOOD_REVIEW_COMPLETE',counts={},native_calls=dict(matchTemplate=0,findTransformECCWithMask=0)),[]))
    monkeypatch.setattr(module,'assess_overlap',lambda p,r,**k:dict(edges=[],coverage=[],regions=[dict(region_id=x['region_id'],network_tracklet=False) for x in r]))
    return plan


def test_frozen_full_plan_then_batches_of_eight_without_neighbor_motion(tmp_path,monkeypatch):
    plan=rig(monkeypatch);calls=[]
    def measured(p,decoded,cv,*,budget):
        assert json.loads((tmp_path/'run/PLAN.json').read_text())==plan
        assert json.loads((tmp_path/f'run/batch_{len(calls):03d}/PLAN.json').read_text())==p
        assert all(set(r)=={'region_id'} for r in p['regions'])
        calls.append(len(p['regions']))
        return dict(regions=[dict(**r,verified_targets=0,region_legs=[]) for r in p['regions']],native_calls=dict(matchTemplate=1,findTransformECCWithMask=1))
    monkeypatch.setattr(module,'track_areas',measured)
    response=module.review_overlap_sequence({}, {},tmp_path/'run',[],None)
    assert calls==[8,1] and response['counts']['regions']==9
    assert response['native_calls']==dict(matchTemplate=2,findTransformECCWithMask=2)
    assert response['geometry_accepted'] is False and response['new_model_points']==0
    assert all(fingerprint(row['path'])==row for row in response['source_bindings'])


def test_drift_before_planning_has_no_measurement(tmp_path,monkeypatch):
    p=tmp_path/'source';p.write_bytes(b'a');ref=fingerprint(p);p.write_bytes(b'b')
    monkeypatch.setattr(module,'build_connected_plan',lambda *a,**k:pytest.fail('planned after drift'))
    with pytest.raises(ValueError,match='overlap source drift'):
        module.review_overlap_sequence({}, {},tmp_path/'run',[ref],None)
    failed=json.loads((tmp_path/'run/FAILURE.json').read_text())
    assert failed['status']=='FAILED' and not failed['regions']


def test_failure_preserves_complete_batch_and_interrupted_measurements(tmp_path,monkeypatch):
    rig(monkeypatch);calls=[];error=TimeoutError('shared deadline')
    error.partial_result=dict(status='INTERRUPTED',native_calls=dict(matchTemplate=1,findTransformECCWithMask=0))
    def measured(p,*a,**k):
        calls.append(1)
        if len(calls)==2:raise error
        return dict(regions=[dict(**r,verified_targets=0,region_legs=[]) for r in p['regions']],native_calls=dict(matchTemplate=3,findTransformECCWithMask=2))
    monkeypatch.setattr(module,'track_areas',measured)
    with pytest.raises(TimeoutError) as caught:
        module.review_overlap_sequence({}, {},tmp_path/'run',[],None)
    assert caught.value is error
    failed=json.loads((tmp_path/'run/FAILURE.json').read_text())
    assert failed['status']=='RESOURCE_LIMIT' and len(failed['regions'])==8
    assert failed['native_calls']['matchTemplate']==4
    assert failed['interrupted_batch']['measurement']==error.partial_result
    assert json.loads((tmp_path/'run/batch_001/FAILURE.json').read_text())==error.partial_result
    assert not (tmp_path/'run/RESULT.json').exists()


def test_baseline_union_counts_foreground_once_without_fitting(tmp_path):
    p=tmp_path/'baseline.json'
    source=dict(source_input_index=1,source_decode_index=10)
    plan=dict(sources=[source],frame_domains={'10':dict(width=4,height=3,original_mask_status='OK',mask_rle=[[[0,4]],[[0,4]],[]])})
    regions=[dict(**source,verified_targets=2,template_roi=dict(left=x,top=0,width=3,height=3)) for x in (0,1)]
    p.write_text(json.dumps(dict(regions=regions,counts={},measurement_wall_seconds=.1)))
    baseline=module._baseline(plan,p,lambda:None)
    row=baseline['source_coverage'][0]
    assert row['foreground_pixels']==row['local_union_pixels']==8
    assert row['local_fraction']==1 and baseline['same_estimator'] is True


def test_sixty_four_comparison_reuses_only_frozen_prefix_without_extra_fits(tmp_path,monkeypatch):
    rig(monkeypatch,n=72);native=[];assessed=[]
    def measured(p,decoded,cv,*,budget):
        prefix=json.loads((tmp_path/'run/PREFIX64_PLAN.json').read_text())
        assert [r['region_id'] for r in prefix['regions']]==[f'r{i}' for i in range(64)]
        native.extend(r['region_id'] for r in p['regions'])
        return dict(regions=[dict(**r,verified_targets=0,region_legs=[]) for r in p['regions']],native_calls=dict(matchTemplate=0,findTransformECCWithMask=0))
    def assessment(p,rows,**kwargs):
        assert [r['region_id'] for r in rows]==[r['region_id'] for r in p['regions']]
        assessed.append(len(rows))
        return dict(edges=[],coverage=[],regions=[dict(region_id=r['region_id'],network_tracklet=False) for r in rows])
    monkeypatch.setattr(module,'track_areas',measured)
    monkeypatch.setattr(module,'assess_overlap',assessment)
    module.review_overlap_sequence({}, {},tmp_path/'run',[],None)
    assert native==[f'r{i}' for i in range(72)] and assessed==[72,64]
    result=json.loads((tmp_path/'run/RESULT.json').read_text())
    assert result['same_budget64']['counts']['regions']==64 and result['counts']['regions']==72


def test_neighborhood_is_after_strict_fit_and_separately_counted(tmp_path,monkeypatch):
    rig(monkeypatch,n=1)
    monkeypatch.setattr(module,'track_areas',lambda p,*a,**k:dict(regions=[dict(region_id='r0',verified_targets=0,region_legs=[])],native_calls=dict(matchTemplate=1,findTransformECCWithMask=1)))
    def review(plan,regions,decoded,output,cv,*,budget):
        assert (tmp_path/'run/batch_000/RESULT.json').is_file()
        assert (tmp_path/'run/PREFIX64_ASSESSMENT.json').is_file()
        assert regions==[dict(region_id='r0',verified_targets=0,region_legs=[])]
        return dict(status='NEIGHBORHOOD_REVIEW_COMPLETE',counts=dict(continuation_extra_accepted=2),native_calls=dict(matchTemplate=4,findTransformECCWithMask=4)),[]
    monkeypatch.setattr(module,'review_neighborhood',review)
    result=module.review_overlap_sequence({}, {},tmp_path/'run',[],None)
    assert result['counts']['targets_accepted']==0
    assert result['strict_native_calls']['matchTemplate']==1
    assert result['native_calls']['matchTemplate']==5
    assert result['neighborhood']['counts']['continuation_extra_accepted']==2
    assert result['new_model_points']==0
