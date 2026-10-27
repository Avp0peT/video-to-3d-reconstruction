import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest
from integrations.blender_export import isolated as isolation
from integrations.blender_export import isolated_launcher as launcher


def setup(tmp_path,monkeypatch):
    outputs=tmp_path/'outputs';work=outputs/'run';stage=work/'verify';stage.mkdir(parents=True)
    scratch=outputs/'RUNTIME_TEMP'/'task'
    exe=outputs/'fake_blender.exe';exe.write_bytes(b'FAKE; NEVER EXECUTED')
    worker=Path(isolation.__file__).with_name('worker.py')
    monkeypatch.setattr(isolation,'OUTPUTS',outputs)
    return dict(argv=[str(exe),'--background','--python',str(worker),'--','verify'],cwd=work,output=stage,scratch_root=scratch,wall_seconds=90,memory_bytes=4<<30,output_bytes=128<<20,label='blender_verify')


def test_supervisor_forwarding_scope_and_retained_scratch(tmp_path,monkeypatch):
    args=setup(tmp_path,monkeypatch);calls=[]
    def fake(argv,**kwargs):
        calls.append((argv,kwargs));request=json.loads(Path(argv[2]).read_text())
        assert request['blender_argv']==args['argv']
        assert Path(request['scratch']).is_relative_to(args['scratch_root'])
        assert not Path(request['scratch']).is_relative_to(args['cwd'])
        (Path(request['scratch'])/'native.log').write_text('retained')
        return dict(status='completed',root_exit_code=7,reclaim_confirmed=True)
    monkeypatch.setattr(isolation,'run_supervised',fake)
    original_env=dict(os.environ);result=isolation.run_blender_isolated(**args)
    assert dict(os.environ)==original_env
    assert result['root_exit_code']==7
    assert calls[0][1]['cwd']==str(args['cwd'])
    assert calls[0][1]['output']==str(args['output'])
    assert calls[0][1]['memory_bytes']==4<<30 and calls[0][1]['wall_seconds']==90
    record=result['temp_isolation']
    assert record['scratch_before']['files']==0 and record['scratch_after']['files']==1
    assert record['environment_overrides']['TMP']==record['environment_overrides']['TEMP']
    assert record['limits']['child_timeout_seconds']<=80
    assert 'not continuously' in record['scratch_observation']
    assert set(record['bindings'])=={'blender_executable','helper','legacy_export','worker','adapter','python_executable'}
    assert (args['scratch_root']/args['label']/'native.log').exists()


@pytest.mark.parametrize('mode',['inside','ancestor','outside_repo','reused'])
def test_scratch_rejected_before_native_start(tmp_path,monkeypatch,mode):
    args=setup(tmp_path,monkeypatch)
    if mode=='inside':args['scratch_root']=args['cwd']/'scratch'
    elif mode=='ancestor':args['scratch_root']=args['cwd'].parent
    elif mode=='outside_repo':args['scratch_root']=tmp_path/'outside'
    else:(args['scratch_root']/args['label']).mkdir(parents=True)
    monkeypatch.setattr(isolation,'run_supervised',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(ValueError):isolation.run_blender_isolated(**args)


def test_launcher_forwards_stdout_stderr_cwd_argv_exit_and_environment(tmp_path):
    result=tmp_path/'result.json';seen={};before=dict(os.environ)
    request=dict(blender_argv=['fake','--arg'],cwd=str(tmp_path),scratch=str(tmp_path/'scratch'),child_timeout_seconds=80,result_path=str(result))
    def fake(argv,**kwargs):
        seen.update(kwargs);assert argv==request['blender_argv'];return SimpleNamespace(returncode=17)
    assert launcher.execute(request,runner=fake)==17
    assert seen['env']['TMP']==seen['env']['TEMP']==request['scratch']
    assert not seen['shell'] and seen['timeout']==80 and seen['cwd']==str(tmp_path)
    assert 'stdout' not in seen and 'stderr' not in seen and 'creationflags' not in seen
    assert dict(os.environ)==before
    assert json.loads(result.read_text())['child_exit_code']==17


def test_launcher_timeout_is_failure(tmp_path):
    request=dict(blender_argv=['fake'],cwd=str(tmp_path),scratch=str(tmp_path/'scratch'),child_timeout_seconds=80,result_path=str(tmp_path/'result.json'))
    def timeout(*a,**k):raise subprocess.TimeoutExpired('fake',80)
    assert launcher.execute(request,runner=timeout)==124
    assert json.loads(Path(request['result_path']).read_text())['status']=='CHILD_TIMEOUT'


def test_private_export_module_never_patches_shared_legacy(tmp_path,monkeypatch):
    legacy=importlib.import_module('integrations.blender_export.export')
    before=legacy.run_supervised;modules=[];observed=[]
    def execute(module):
        modules.append(module)
        def export(spec,output,profile):
            observed.append((spec,output,profile));return module.run_supervised(['fake'],cwd='x',output='y',wall_seconds=90,memory_bytes=1,output_bytes=1,label='test')
        module.export_blend=export
    monkeypatch.setattr(isolation.importlib.util,'spec_from_file_location',lambda *a:SimpleNamespace(loader=SimpleNamespace(exec_module=execute)))
    monkeypatch.setattr(isolation.importlib.util,'module_from_spec',lambda spec:SimpleNamespace())
    calls=[]
    monkeypatch.setattr(isolation,'run_blender_isolated',lambda argv,**kw:calls.append(kw) or {'status':'FAKE'})
    assert isolation.export_blend_isolated('spec','out',scratch_root='scratch',profile='surface400k')['status']=='FAKE'
    assert calls[0]['scratch_root']=='scratch'
    assert observed==[('spec','out','surface400k')]
    assert modules[0].run_supervised is not before and legacy.run_supervised is before
