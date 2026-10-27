"""Narrow Windows runtime environment; no model, GPU or native process."""
from types import SimpleNamespace
from pathlib import Path
import os
import pytest
from integrations.mvs_surface import supervision


def test_explicit_platform_path_is_the_only_retained_host_value(monkeypatch,tmp_path):
    monkeypatch.setenv('ProgramW6432',str(tmp_path))
    monkeypatch.setenv('TEST_SECRET_NOT_FOR_CHILD','do-not-inherit')
    monkeypatch.setattr(supervision,'_registry_program_w6432',lambda:pytest.fail('unexpected registry fallback'))
    assert supervision._windows_platform_environment()=={
        'ProgramW6432':str(tmp_path.resolve()),'PATHEXT':'.COM;.EXE;.BAT;.CMD'}


def test_already_isolated_parent_uses_machine_path(monkeypatch,tmp_path):
    monkeypatch.delenv('ProgramW6432',raising=False)
    monkeypatch.setattr(supervision,'_registry_program_w6432',lambda:str(tmp_path))
    assert supervision._windows_platform_environment()=={
        'ProgramW6432':str(tmp_path.resolve()),'PATHEXT':'.COM;.EXE;.BAT;.CMD'}


@pytest.mark.parametrize('value',['','relative/path'])
def test_invalid_override_is_rejected_not_silently_replaced(monkeypatch,value):
    monkeypatch.setenv('ProgramW6432',value)
    monkeypatch.setattr(supervision,'_registry_program_w6432',lambda:pytest.fail('must not hide bad override'))
    with pytest.raises(ValueError):supervision._windows_platform_environment()


def test_missing_platform_path_refuses_launch(monkeypatch,tmp_path):
    monkeypatch.setenv('ProgramW6432',str(tmp_path/'missing'))
    with pytest.raises(OSError):supervision._windows_platform_environment()


def test_supervised_entry_receives_path_without_host_secrets(monkeypatch,tmp_path):
    if os.name!='nt':pytest.skip('Windows launch environment')
    evidence=tmp_path/'evidence';evidence.mkdir()
    exe=tmp_path/'test.exe';exe.write_bytes(b'TEST_DATA no execution')
    program_files=tmp_path/'program-files';program_files.mkdir()
    monkeypatch.setenv('ProgramW6432',str(program_files))
    monkeypatch.setenv('TEST_SECRET_NOT_FOR_CHILD','do-not-inherit')
    calls=[]
    def supervise(argv,**kwargs):
        calls.append(kwargs)
        return {'status':'completed','root_exit_code':0,'pid':123}
    fake=SimpleNamespace(supervise=supervise,_mvs_eventlog_failures=[])
    monkeypatch.setattr(supervision,'_load_supervisor',lambda:(fake,{'test':'fixture'}))
    result=supervision.run_supervised([str(exe)],cwd=str(tmp_path),output=str(evidence),
         wall_seconds=1,memory_bytes=128<<20,output_bytes=1<<20,label='test_environment')
    assert calls[0]['inherit_env'] is False
    assert calls[0]['env_overrides']['ProgramW6432']==str(program_files.resolve())
    assert 'TEST_SECRET_NOT_FOR_CHILD' not in calls[0]['env_overrides']
    assert calls[0]['env_overrides']['PATHEXT']=='.COM;.EXE;.BAT;.CMD'
    assert calls[0]['job_memory_limit_bytes']==128<<20
    assert result['mvs_surface_adapter']['environment']['ProgramW6432']==str(program_files.resolve())
