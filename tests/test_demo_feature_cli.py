"""Public entry integration; no subprocess or model is started."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('photo_feature_cli', ROOT / 'scripts/run_photo_demo.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


@pytest.fixture
def photos(tmp_path):
    folder = tmp_path / 'photos'
    folder.mkdir()
    (folder / 'a.jpg').write_bytes(b'fixture-a')
    (folder / 'b.jpg').write_bytes(b'fixture-b')
    return folder


def test_old_cli_defaults_preserved(photos, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, 'plan', lambda options: seen.append(options) or {'status': 'TEST_ONLY'})
    assert cli.main(['plan', '--source', str(photos), '--prompt', 'cup']) == 0
    assert seen[0].target == 'blender'
    assert seen[0].checkpoint == Path(r'C:\sam3\sam3.pt')


def test_deployment_parameters_enter_actual_options(photos, tmp_path, monkeypatch):
    from integrations.demo_pipeline import deployment
    calls, seen = [], []
    def options(path, digest, *, expected_input_kind):
        calls.append((path, digest, expected_input_kind))
        return {'target': 'cad', 'python': tmp_path / 'python.exe', 'checkpoint': tmp_path / 'weights.pt'}
    monkeypatch.setattr(deployment, 'options_overrides', options)
    monkeypatch.setattr(cli, 'plan', lambda value: seen.append(value) or {})
    config = tmp_path / 'deploy.json'
    assert cli.main(['plan', '--source', str(photos), '--prompt', 'cup', '--deployment-config', str(config), '--deployment-config-sha256', 'a'*64]) == 0
    assert calls == [(config, 'a'*64, 'photos')]
    assert seen[0].target == 'cad' and seen[0].checkpoint == tmp_path / 'weights.pt'


@pytest.mark.parametrize('flags', [[], ['--target', 'report']])
def test_incomplete_or_conflicting_deployment_never_launches(photos, tmp_path, monkeypatch, flags):
    from integrations.demo_pipeline import deployment
    monkeypatch.setattr(deployment, 'options_overrides', lambda *a, **k: {'target': 'cad'})
    monkeypatch.setattr(cli, 'run_pipeline', lambda *a, **k: pytest.fail('native pipeline started'))
    argv = ['run', '--source', str(photos), '--prompt', 'cup', '--deployment-config', str(tmp_path / 'd.json')]
    if flags:
        argv += ['--deployment-config-sha256', 'a'*64] + flags
    assert cli.main(argv) == 2


def test_deployment_kind_is_checked_in_same_audit(monkeypatch):
    from integrations.demo_pipeline import deployment
    seen = []
    monkeypatch.setattr(deployment, 'audit_deployment', lambda *a: seen.append(a) or {
        'status': 'STATIC_BOUND_RUNTIME_UNVERIFIED', 'spec': {'input_kind': 'photos', 'target': 'blender'}})
    with pytest.raises(ValueError, match='input_kind'):
        deployment.options_overrides('d.json', 'a'*64, expected_input_kind='video')
    assert len(seen) == 1


def test_resume_plan_has_no_execution_and_resume_uses_the_same_plan(tmp_path, monkeypatch):
    from integrations.demo_pipeline import resume
    from integrations.blender_export import export
    planned, executed = [], []
    def make_plan(*args, **kwargs):
        planned.append((args, kwargs));return {'schema':'fixture-continuation'}
    monkeypatch.setattr(resume,'plan',make_plan)
    monkeypatch.setattr(resume,'execute',lambda frozen: executed.append(frozen) or {'exit_code':1,'status':'PARTIAL'})
    monkeypatch.setattr(export,'find_blender',lambda explicit: tmp_path/'blender.exe')
    args=['--run',str(tmp_path/'old'),'--output',str(tmp_path/'new')]
    assert cli.main(['plan-resume']+args)==0 and not executed
    assert cli.main(['resume']+args)==1
    assert len(planned)==2 and executed==[{'schema':'fixture-continuation'}]


def test_cad_plan_and_execution_share_bound_source_without_reconstruction(tmp_path,monkeypatch):
    from integrations.demo_pipeline import cad
    planned=[];executed=[]
    def make_plan(**kw):
        planned.append(kw);return {'schema':'fixture-cad','source':str(kw['run'])}
    monkeypatch.setattr(cad,'plan',make_plan)
    monkeypatch.setattr(cad,'execute',lambda frozen:executed.append(frozen) or {'exit_code':0})
    monkeypatch.setattr(cli,'run_pipeline',lambda *a,**k:pytest.fail('reconstruction started'))
    args=['--run',str(tmp_path/'continued'),'--output',str(tmp_path/'cad')]
    assert cli.main(['plan-cad']+args)==0 and not executed
    assert cli.main(['continue-cad']+args)==0 and len(executed)==1
    assert planned[0]==planned[1] and executed[0]['source']==str(tmp_path/'continued')


def test_cad_precondition_error_does_not_execute(tmp_path,monkeypatch):
    from integrations.demo_pipeline import cad
    def reject(**kw):raise ValueError('continuation is unbound')
    monkeypatch.setattr(cad,'plan',reject)
    monkeypatch.setattr(cad,'execute',lambda *a,**k:pytest.fail('CAD started'))
    assert cli.main(['continue-cad','--run',str(tmp_path/'bad'),'--output',str(tmp_path/'out')])==2
