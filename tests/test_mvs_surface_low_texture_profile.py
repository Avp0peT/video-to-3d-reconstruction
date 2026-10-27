"""TEST_DATA only: fixed-profile admission and mocked native config rewrite."""
import copy
import hashlib
import sys
from pathlib import Path
import pytest
from integrations.mvs_surface import runner
from test_mvs_surface_runner import inputs, _install_supervisor, _snapshot


def plan(inputs, **changes):
    return runner.make_plan(**(inputs.arguments | changes))


def test_default_recipe_unchanged_and_low_texture_only_adds_fixed_config(inputs,tmp_path):
    before=_snapshot(inputs.root)
    old=plan(inputs); explicit=plan(inputs,profile=runner.PROFILE)
    assert explicit==old and 'profile' not in old['arguments'] and 'config_initial' not in old
    new=plan(inputs,profile=runner.LOW_TEXTURE_PROFILE)
    stripped=copy.deepcopy(new);stripped['profile']=old['profile'];stripped['arguments'].pop('profile');initial=stripped.pop('config_initial')
    assert stripped==old
    assert runner.commands(new,tmp_path)==runner.commands(old,tmp_path)
    data=b'Descriptor Min Magnitude Threshold = 0.01\n'
    assert initial==dict(filename='__mvs_dense.cfg',text=data.decode(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    assert _snapshot(inputs.root)==before


@pytest.mark.parametrize('profile',['',None,'custom','openmvs-2.4.0-cpu-low-texture-v2'])
def test_unrecognized_profile_is_rejected(inputs,profile):
    with pytest.raises(ValueError,match='profile'):plan(inputs,profile=profile)


def test_low_texture_config_not_in_immutable_initial_staged_set(inputs,tmp_path):
    p=plan(inputs,profile=runner.LOW_TEXTURE_PROFILE);w=tmp_path/'work';w.mkdir()
    staged=runner.stage_inputs(p,w,float('inf'))
    assert (w/'__mvs_dense.cfg').read_text()==runner.LOW_TEXTURE_CONFIG
    assert all(Path(x['path']).name!='__mvs_dense.cfg' for x in staged)
    (w/'__mvs_dense.cfg').write_text('Descriptor Min Magnitude Threshold = 0.1\n')
    with pytest.raises(ValueError,match='initial dense configuration'):runner._verify_initial_dense_config(p,w)


def test_old_profile_does_not_prewrite_dense_config(inputs,tmp_path):
    w=tmp_path/'work';w.mkdir();runner.stage_inputs(plan(inputs),w,float('inf'))
    assert not (w/'__mvs_dense.cfg').exists()


def test_native_rewrite_records_effective_config_without_false_drift(inputs,tmp_path,monkeypatch):
    before=_snapshot(inputs.root)
    def rewrite(label,cwd):
        if label=='dense':(cwd/'__mvs_dense.cfg').write_text('# Native complete effective settings\nDescriptor Min Magnitude Threshold = 0.010000\nNCC Threshold = 0.9\n')
    calls=_install_supervisor(monkeypatch,mutate=rewrite)
    p=plan(inputs,profile=runner.LOW_TEXTURE_PROFILE);r=runner.execute_plan(p,tmp_path/'run')
    assert r['status']=='SURFACE_GENERATED_NOT_ACCEPTED' and len(calls)==2
    assert r['profile']==runner.LOW_TEXTURE_PROFILE and r['input_drift']==[]
    effective=r['stages'][0]['dense_config_effective']
    assert effective['descriptor_min_magnitude_threshold']==.01
    assert effective['native_config']['sha256']!=p['config_initial']['sha256']
    assert Path(effective['effective_snapshot']['path']).read_bytes()==Path(effective['native_config']['path']).read_bytes()
    assert _snapshot(inputs.root)==before


@pytest.mark.parametrize('text',['Descriptor Min Magnitude Threshold = 0.1\n','NCC Threshold = 0.9\n','Descriptor Min Magnitude Threshold = nan\n','Descriptor Min Magnitude Threshold = 0.01\nDescriptor Min Magnitude Threshold = 0.01\n'])
def test_effective_wrong_missing_or_duplicate_value_stops_mesh(inputs,tmp_path,monkeypatch,text):
    def rewrite(label,cwd):
        if label=='dense':(cwd/'__mvs_dense.cfg').write_text(text)
    calls=_install_supervisor(monkeypatch,mutate=rewrite)
    r=runner.execute_plan(plan(inputs,profile=runner.LOW_TEXTURE_PROFILE),tmp_path/'run')
    assert r['status']=='FAILED' and r['exit_code']!=0 and len(calls)==1


def test_stage_initial_bytes_drift_stops_before_native(inputs,tmp_path,monkeypatch):
    original=runner.stage_inputs
    def drift(p,w,deadline):
        staged=original(p,w,deadline);(w/'__mvs_dense.cfg').write_bytes(b'changed');return staged
    monkeypatch.setattr(runner,'stage_inputs',drift)
    calls=_install_supervisor(monkeypatch)
    r=runner.execute_plan(plan(inputs,profile=runner.LOW_TEXTURE_PROFILE),tmp_path/'run')
    assert r['status']=='FAILED' and r['native_launches']==0 and not calls


def test_plan_arbitrary_config_override_rejected_before_output(inputs,tmp_path):
    p=plan(inputs,profile=runner.LOW_TEXTURE_PROFILE);p['config_initial']['text']='Descriptor Min Magnitude Threshold = 0\n'
    output=tmp_path/'refused'
    with pytest.raises(ValueError,match='plan does not match'):runner.execute_plan(p,output)
    assert not output.exists()


def test_cli_profile_reaches_plan_api(tmp_path,monkeypatch):
    from scripts import build_mvs_surface as cli
    captured={}
    monkeypatch.setattr(runner,'make_plan',lambda **kwargs:captured.update(kwargs) or {'TEST_DATA':True})
    args=['surface','plan']
    for key in ('scene-path','image-root','mask-root','validity-root','tool-root'):args+=['--'+key,str(tmp_path/key)]
    for key,value in [('total-seconds',30),('stage-seconds',10),('memory-bytes',128<<20),('output-bytes',16<<20)]:args+=['--'+key,str(value)]
    args+=['--profile',runner.LOW_TEXTURE_PROFILE,'--plan-output',str(tmp_path/'plan.json')]
    monkeypatch.setattr(sys,'argv',args)
    assert cli.main()==0 and captured['profile']==runner.LOW_TEXTURE_PROFILE
