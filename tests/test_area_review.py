import json
import pytest
from integrations.demo_pipeline import area_review as module
from integrations.demo_pipeline.temporal_review import fingerprint


def test_empty_real_chain_reports_no_alignment(tmp_path):
    out=tmp_path/'run'
    r=module.review_area_sequence(dict(regions=[],frame_domains={}),{},out,[],None)
    assert r['regions']==0 and r['new_model_points']==0 and not r['geometry_accepted']
    assert all(fingerprint(b['path'])==b for b in r['source_bindings'])
    assert (out/'index.html').is_file()


def test_source_drift_rejected_before_native(tmp_path,monkeypatch):
    p=tmp_path/'input';p.write_bytes(b'a');ref=fingerprint(p);p.write_bytes(b'b')
    monkeypatch.setattr(module,'track_areas',lambda *a,**k:pytest.fail('must not run'))
    with pytest.raises(ValueError,match='whole-area source drift'):
        module.review_area_sequence({}, {},tmp_path/'run',[ref],None)
    assert json.loads((tmp_path/'run/FAILURE.json').read_text())['status']=='FAILED'


def test_frozen_plan_precedes_alignment_and_point_results_do_not_enter_estimator(tmp_path,monkeypatch):
    plan=dict(regions=[dict(region_id='TEST_DATA')])
    point=dict(region_id='TEST_DATA',center_baseline={'total':1},expanded_grid={'total':80},counts={})
    seen=[]
    def measure(p,d,cv,*,budget):
        assert json.loads((tmp_path/'run/PLAN.json').read_text())==plan
        assert p is plan and 'point_baseline' not in p
        seen.append(True)
        return dict(status='AREA_ALIGNMENT_COMPLETE',regions=[],counts={},native_calls={})
    monkeypatch.setattr(module,'track_areas',measure)
    monkeypatch.setattr(module,'render_area_tracking',lambda *a,**k:[])
    module.review_area_sequence(plan,{},tmp_path/'run',[],None,point_baseline=[point])
    saved=json.loads((tmp_path/'run/RESULT.json').read_text())
    assert seen==[True] and saved['point_baseline']==[point]


def test_partial_native_evidence_retained_on_timeout(tmp_path,monkeypatch):
    error=TimeoutError('TEST_DATA shared budget');error.partial_result={'native_calls':{'ecc':1}}
    def stop(*a,**k):raise error
    monkeypatch.setattr(module,'track_areas',stop)
    with pytest.raises(TimeoutError) as caught:
        module.review_area_sequence({'regions':[]},{},tmp_path/'run',[],None)
    assert caught.value is error
    r=json.loads((tmp_path/'run/FAILURE.json').read_text())
    assert r['status']=='RESOURCE_LIMIT' and r['interrupted_measurement']==error.partial_result
    assert not (tmp_path/'run/RESULT.json').exists()
