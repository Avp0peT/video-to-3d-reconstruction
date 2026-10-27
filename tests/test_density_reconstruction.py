"""TEST_DATA; plan identity and fail-stop controller, no native execution."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.video_masks import density_reconstruction as dr


def test_derivation_preserves_all_sources_and_changes_only_density_policy():
    old=dict(schema='glove-region-plan-v1',timeline=[dict(decode_index=i,input_index=i if i%2==0 else None,image_path='TEST_DATA',image_sha256='a'*64) for i in range(7)],baseline_sfm='TEST_DATA',source_bindings={'old':'hash'})
    frozen=deepcopy(old);new=dr.derived_plan(old)
    assert old==frozen and new['timeline']==old['timeline']
    assert new.pop('density_policy')==dr.default_policy()
    assert new.pop('identity_policy')==dr.default_identity_policy()
    assert new==old


def test_large_endpoint_set_requires_explicit_expansion():
    old=dict(schema='glove-region-plan-v1',timeline=[dict(decode_index=i,input_index=i) for i in range(88)])
    with pytest.raises(ValueError,match='expanded'):
        dr.derived_plan(old)
    old['endpoint_expansion_policy']=dict(schema='object-region-bridge-policy-v1',native_max_frames=96,max_added=24)
    assert len(dr.derived_plan(old)['timeline'])==88


def test_failed_stage_stops_and_never_retries(tmp_path,monkeypatch):
    out=tmp_path/'new';out.mkdir()
    dr.save(out/'PLAN.json',dict(source_bindings=[],code_bindings=[],density_policy=dr.default_policy()))
    dr.save(out/'RESULT.json',dict(status='PREPARED',stages=[]))
    calls=[]
    def invoke(root,run,name,argv,**kw):
        calls.append(name)
        (run/'logs').mkdir()
        return dict(status='completed',root_exit_code=1,reclaim_confirmed=True,errors=[],log_errors=[])
    monkeypatch.setattr(dr,'_invoke',invoke)
    result=dr.execute(out)
    assert calls==['tracking'] and result['status']=='PARTIAL_OR_FAILED'
    with pytest.raises(ValueError,match='already attempted'):dr.execute(out)
    assert calls==['tracking']


def test_public_entry_routes_to_prepare_then_execute(tmp_path,monkeypatch):
    from scripts import run_photo_demo as cli
    calls=[]
    monkeypatch.setattr(dr,'prepare',lambda source,out:calls.append(('prepare',source,out)))
    def execute(out):
        calls.append(('execute',out));return dict(status='SURFACE_GENERATED_NOT_ACCEPTED')
    monkeypatch.setattr(dr,'execute',execute)
    assert cli.main(['rebuild-density-model','--run',str(tmp_path/'old'),'--output',str(tmp_path/'new')])==0
    assert calls==[('prepare',tmp_path/'old',tmp_path/'new'),('execute',tmp_path/'new')]
