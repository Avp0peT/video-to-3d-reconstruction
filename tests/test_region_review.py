import json
import pytest
from integrations.demo_pipeline import region_review as module
from integrations.demo_pipeline.temporal_review import fingerprint


def test_empty_real_chain_writes_honest_report(tmp_path):
    refs=[]
    result=module.review_regions({'records':[]},[],[],{},tmp_path/'run',refs,None)
    assert result['status']=='REGION_GROWTH_REVIEW_COMPLETE'
    assert result['regions']==0 and result['flow_calls']==0 and result['counts']=={}
    assert result['new_model_points']==0 and not result['geometry_accepted']
    assert all(fingerprint(b['path'])==b for b in result['source_bindings'])
    assert (tmp_path/'run/index.html').is_file()


def test_binding_drift_cannot_start_measurement(tmp_path,monkeypatch):
    source=tmp_path/'input';source.write_bytes(b'a');b=fingerprint(source)
    source.write_bytes(b'b')
    monkeypatch.setattr(module,'build_region_plan',lambda *a,**k:pytest.fail('called after drift'))
    with pytest.raises(ValueError,match='source drift'):
        module.review_regions({'records':[]},[],[],{},tmp_path/'run',[b],None)
    result=json.loads((tmp_path/'run/FAILURE.json').read_text())
    assert result['status']=='FAILED' and result['flow_calls']==0


def test_three_levels_are_measured_after_each_plan_is_written(tmp_path,monkeypatch):
    def plan(level):
        return dict(regions=[dict(region_id=str(level),depth=level,points=[])] if level<3 else [],
                    frame_domains={},growth_statistics=dict(level=level))
    monkeypatch.setattr(module,'build_region_plan',lambda *a,**k:plan(0))
    seen=[]
    def measured(p,decoded,cv,*,budget):
        level=p['regions'][0]['depth']
        assert json.loads((tmp_path/f'run/PLAN_{level:02d}.json').read_text())==p
        seen.append(level)
        return dict(status='REGION_MEASUREMENTS_COMPLETE',regions=p['regions'],counts={},flow_calls=2)
    monkeypatch.setattr(module,'measure_regions',measured)
    monkeypatch.setattr(module,'next_region_plan',lambda p,results,**k:plan(len(results)))
    monkeypatch.setattr(module,'render_regions',lambda *a,**k:[])
    area_plans=[]
    def area(p,*a,**k):
        area_plans.append(p)
        return dict(status='AREA_ALIGNMENT_COMPLETE',source_bindings=[])
    monkeypatch.setattr(module,'review_area_sequence',area)
    overlap_plans=[]
    def overlap(p,*a,**k):
        overlap_plans.append(p)
        return dict(status='OVERLAP_REVIEW_COMPLETE',source_bindings=[])
    monkeypatch.setattr(module,'review_overlap_sequence',overlap)
    result=module.review_regions({'records':[]},[],[],{},tmp_path/'run',[],None)
    assert seen==[0,1,2] and result['regions']==3 and result['flow_calls']==6
    full=json.loads((tmp_path/'run/RESULT.json').read_text())
    assert full['unexecuted_frontier_count']==0 and len(full['rounds'])==3
    assert area_plans==[plan(0)]
    assert overlap_plans==[plan(0)]


def test_partial_measurement_is_preserved_without_success(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'build_region_plan',lambda *a,**k:dict(regions=[{}]))
    error=TimeoutError('shared budget')
    error.partial_result={'status':'INTERRUPTED','flow_calls':1}
    def stop(*a,**k):raise error
    monkeypatch.setattr(module,'measure_regions',stop)
    with pytest.raises(TimeoutError) as caught:
        module.review_regions({'records':[]},[],[],{},tmp_path/'run',[],None)
    assert caught.value is error
    out=json.loads((tmp_path/'run/FAILURE.json').read_text())
    assert out['status']=='RESOURCE_LIMIT' and out['interrupted_measurement']['flow_calls']==1
    assert not (tmp_path/'run/RESULT.json').exists()


def test_native_failure_is_not_a_completed_empty_region(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'build_region_plan',lambda *a,**k:dict(regions=[{}]))
    def stop(*a,**k):raise RuntimeError('native problem')
    monkeypatch.setattr(module,'measure_regions',stop)
    with pytest.raises(RuntimeError,match='native problem'):
        module.review_regions({'records':[]},[],[],{},tmp_path/'run',[],None)
    assert json.loads((tmp_path/'run/FAILURE.json').read_text())['status']=='FAILED'
