import json
import socket
import subprocess
import sys
from pathlib import Path
import pytest
from integrations.demo_pipeline import deployment as dep


def fixture(root):
    root.mkdir(exist_ok=True)
    for name in ('python.exe', 'heavy.exe', 'git.exe', 'model.pt', 'pyproject.toml',
                 'scripts/run_photo_demo.py', 'integrations/demo_pipeline/pipeline.py',
                 'integrations/demo_pipeline/runtime.py'):
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'fixture')
    (root / 'package').mkdir(); (root / 'integrations/sam3/src/sam3-repo').mkdir(parents=True)
    return dict(project_root=str(root), project_python=str(root / 'python.exe'),
                sam3_python=str(root / 'heavy.exe'), git=str(root / 'git.exe'),
                checkpoint=str(root / 'model.pt'), checkpoint_sha256=dep.runtime.sha256(root / 'model.pt'),
                sam3_source=str(root / 'integrations/sam3/src/sam3-repo'),
                sam3_package=str(root / 'package'), input_kind='photos', target='report')


def test_new_record_and_audit_keep_blockers_no_authority(tmp_path):
    spec = fixture(tmp_path / 'project with spaces')
    path = Path(spec['project_root']) / 'outputs/deployment/config.json'
    result = dep.create_deployment(spec, path)
    assert result['status'] == 'STATIC_BLOCKED'
    record = dep.audit_deployment(path, result['sha256'])
    assert not record['execution_authorized'] and not record['historical_pass_reused']
    assert not record['current_runtime_evidence']
    assert any(b['category'] == 'REBINDING_REQUIRES_REVIEWED_RUNTIME_PROFILE' for b in record['blockers'])
    with pytest.raises(ValueError, match='deployment blocked'): dep.options_overrides(path, result['sha256'])
    with pytest.raises(FileExistsError): dep.create_deployment(spec, path)


def test_dependency_and_config_tampering(tmp_path):
    spec = fixture(tmp_path / 'root'); path = Path(spec['project_root']) / 'outputs/config.json'
    result = dep.create_deployment(spec, path)
    Path(spec['checkpoint']).write_bytes(b'changed')
    report = dep.audit_deployment(path, result['sha256'])
    assert {'DEPLOYMENT_BINDING_DRIFT', 'CHECKPOINT_DIGEST_MISMATCH'} <= {b['category'] for b in report['blockers']}
    path.write_text('{}')
    with pytest.raises(ValueError, match='record digest mismatch'): dep.audit_deployment(path, result['sha256'])


def test_explicit_paths_target_requirements_and_protected_output(tmp_path):
    spec = fixture(tmp_path / 'root')
    spec['target'] = 'cad'
    with pytest.raises(ValueError, match='explicit paths'): dep.paths(spec)
    spec['target'] = 'report'; spec['git'] = 'relative.exe'
    with pytest.raises(ValueError, match='absolute'): dep.paths(spec)
    spec['git'] = str(Path(spec['project_root']) / 'git.exe')
    with pytest.raises(ValueError, match='outputs'): dep.create_deployment(spec, tmp_path / 'unsafe.json')
    spec['sam3_package'] = str(Path(spec['project_root']) / 'outputs/package')
    with pytest.raises(ValueError, match='disjoint'): dep.create_deployment(spec, Path(spec['sam3_package']) / 'config.json')


def test_no_native_network_import_or_authority_mutation(tmp_path, monkeypatch):
    spec = fixture(tmp_path / 'root')
    def forbidden(*a, **k): raise AssertionError('forbidden')
    monkeypatch.setattr(subprocess, 'run', forbidden); monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    before = {str(p): p.read_bytes() for p in Path(spec['project_root']).rglob('*') if p.is_file()}
    modules = set(sys.modules)
    dep.inspect(spec)
    assert before == {str(p): p.read_bytes() for p in Path(spec['project_root']).rglob('*') if p.is_file()}
    assert not any(n.split('.')[0] in {'sam3', 'torch'} for n in set(sys.modules) - modules)


def test_duplicate_json_and_cli_failure_record(tmp_path, capsys):
    p = tmp_path / 'bad.json'; p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match='duplicate'): dep.read(p)
    spec = fixture(tmp_path / 'root'); p.write_text(json.dumps(spec))
    out = Path(spec['project_root']) / 'outputs/config.json'
    assert dep.main(['create', '--spec', str(p), '--output', str(out)]) == 2
    assert out.is_file() and json.loads(capsys.readouterr().out)['status'] == 'STATIC_BLOCKED'


def test_product_options_mapping_requires_successful_fresh_audit(monkeypatch):
    # Mapping only, synthetic audit; never claims an accepted real deployment.
    monkeypatch.setattr(dep, 'audit_deployment', lambda *a: dict(status='STATIC_BOUND_RUNTIME_UNVERIFIED',
        spec={'project_python': 'C:/python.exe', 'git': 'C:/git.exe', 'target': 'report', 'ffmpeg': 'C:/ffmpeg.exe', 'input_kind': 'photos'}))
    options = dep.options_overrides('unused', 'unused')
    assert options == {'python': Path('C:/python.exe'), 'git_exe': Path('C:/git.exe'), 'target': 'report', 'ffmpeg': 'C:/ffmpeg.exe'}
    assert dep.options_overrides('unused', 'unused', expected_input_kind='photos') == options
    with pytest.raises(ValueError, match='input_kind'): dep.options_overrides('unused', 'unused', expected_input_kind='video')
