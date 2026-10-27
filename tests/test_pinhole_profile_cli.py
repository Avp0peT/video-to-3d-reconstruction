"""Pinhole CLI wiring tests; synthetic inputs, no reconstruction runs."""
import json
import pytest
from kozlov_lab.reconstruction import cli, profiles, colmap_adapter as adapter
from test_colmap_adapter import make_upstream_run
from test_colmap_profiles import _shared_extract_result

PROFILE=profiles.PROFILE_CAD_OBJECT_PINHOLE_V1

def test_shared_pinhole_checks_actual_dimensions_and_three_params():
    result=_shared_extract_result(['images/000000.png'])
    camera=result['cameras']['7']
    camera.update(model_name='SIMPLE_PINHOLE',width=800,height=600,params=[960.,400.,300.])
    expected={'images/000000.png':(800,600)}
    cli._check_shared_camera(result,None,expected,profile=PROFILE)
    for updates,match in [({'params':[960.,400.,300.,0.]},'parameter'),
                          ({'model_name':'SIMPLE_RADIAL'},'SIMPLE_PINHOLE'),
                          ({'width':801},'expected 800x600'),
                          ({'params':[float('nan'),400.,300.]},'finite')]:
        saved=dict(camera);camera.update(updates)
        with pytest.raises(adapter.BackendError,match=match):
            cli._check_shared_camera(result,None,expected,profile=PROFILE)
        camera.clear();camera.update(saved)

def test_pinhole_requires_thread_policy_before_native(tmp_path,monkeypatch):
    for key in profiles.RUNTIME_THREAD_ENV:monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(cli,'load_upstream_run',lambda *a:pytest.fail('input read before thread check'))
    out=tmp_path/'out';out.mkdir()
    assert cli.run_reconstruction(io_run='unused',run_dir=out,project_root=tmp_path,profile=PROFILE)==2
    assert 'thread environment' in json.loads((out/'run.json').read_text())['errors'][0]['message']

def test_worker_spec_uses_pinhole_and_selected_profile(tmp_path,monkeypatch):
    source=make_upstream_run(tmp_path/'input')
    for key in profiles.RUNTIME_THREAD_ENV:monkeypatch.setenv(key,'2')
    monkeypatch.setattr(cli,'backend_info',lambda *a,**k:dict(name='pycolmap',version=adapter.BACKEND_PIN,runtime='cpu'))
    calls=[]
    def stop(stage,spec,**kwargs):
        calls.append((stage,spec))
        return adapter.NativeStageResult(stage=stage,started_utc='T',finished_utc='T',returncode=None,timed_out=False,
            stdout_path='',stderr_path='',result=None,error='synthetic stop',started=False)
    monkeypatch.setattr(cli,'run_native_stage',stop)
    out=tmp_path/'out';out.mkdir()
    cli.run_reconstruction(io_run=str(source),run_dir=out,project_root=tmp_path,profile=PROFILE)
    assert calls and calls[0][1]['camera_model']=='SIMPLE_PINHOLE'
    assert calls[0][1]['camera_mode']=='SINGLE'
    assert calls[0][1]['reconstruction_profile']==PROFILE
    record=json.loads((out/'run.json').read_text())
    assert record['contract_version']=='sfm-native-object-pinhole-v1'
    assert record['effective_options']['max_distinct_images']==96

def test_foreground_and_rcover_receive_profile(tmp_path,monkeypatch):
    from kozlov_lab.reconstruction import foreground
    source=make_upstream_run(tmp_path/'input')
    for key in profiles.RUNTIME_THREAD_ENV:monkeypatch.setenv(key,'2')
    called=[]
    def stop(upstream,path,out,*,profile):
        called.append(profile);raise ValueError('intentional foreground stop')
    monkeypatch.setattr(foreground,'select_foreground',stop)
    out=tmp_path/'out';out.mkdir()
    assert cli.run_reconstruction(io_run=str(source),run_dir=out,project_root=tmp_path,
        profile=PROFILE,point_selection='rcover-v1',object_mask_manifest=tmp_path/'unused.json')==2
    assert called==[PROFILE]
