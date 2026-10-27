import json
from pathlib import Path
import subprocess
import socket
import sys

from integrations.demo_pipeline.preflight import check_environment, main, LIGHT, HEAVY, PACKAGE


def fixture(root):
    for name in ('pyproject.toml', LIGHT, HEAVY, 'model.pt', 'git.exe',
                 'scripts/prepare_object.py', 'scripts/reconstruct_object.py', 'scripts/demo_segment_worker.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    for name in ('BUNDLE.json', 'AUTHORITY_BINDINGS.json', 'START_BASELINE.json'):
        path = root / PACKAGE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'project_root': str(root)}), encoding='utf-8')
    return {'checkpoint': 'model.pt', 'git': 'git.exe'}


def test_missing_and_wrong_type_are_actionable(tmp_path):
    args = fixture(tmp_path)
    (tmp_path / 'model.pt').unlink()
    (tmp_path / 'model.pt').mkdir()
    result = check_environment(tmp_path, target='report', **args)
    item = next(x for x in result['checks'] if x['name'] == 'checkpoint')
    assert item['status'] == 'WRONG_TYPE'
    assert '--checkpoint' in item['next_step']
    assert result['status'] == 'STATIC_BLOCKED'


def test_report_does_not_require_geometry_cad_or_video(tmp_path):
    args = fixture(tmp_path)
    result = check_environment(tmp_path, target='report', **args)
    assert result['missing_required'] == []
    assert result['status'] == 'STATIC_ITEMS_PRESENT_RUNTIME_UNVERIFIED'
    assert result['current_runtime_evidence'] == []
    assert result['clean_machine_acceptance'] == 'NOT_PERFORMED'


def test_target_matrix(tmp_path):
    args = fixture(tmp_path)
    photo = check_environment(tmp_path, target='blender', **args)
    video = check_environment(tmp_path, input_kind='video', target='cad', **args)
    required = lambda r: {x['name'] for x in r['checks'] if x['required']}
    assert 'freecad_python' not in required(photo)
    assert {'ffmpeg', 'ffprobe', 'freecad_python', 'blender'} <= required(video)
    assert 'openmvs/DensifyPointCloud.exe' in required(photo)


def test_paths_spaces_and_static_identity(tmp_path, capsys):
    root = tmp_path / 'relocated project with spaces'
    args = fixture(root)
    cfg = root / Path(LIGHT).parent.parent / 'pyvenv.cfg'
    cfg.write_text('version = 3.13.2\n', encoding='utf-8')
    assert main(['--project-root', str(root), '--target', 'report', '--checkpoint', 'model.pt', '--git-exe', 'git.exe']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['project_root'] == str(root.resolve())
    python = next(x for x in result['checks'] if x['name'] == 'project_python')
    assert python['version'] == '3.13.2'
    assert python['runtime_compatibility'] == 'UNVERIFIED'


def test_foreign_root_and_override_do_not_reuse_pass(tmp_path):
    args = fixture(tmp_path)
    path = tmp_path / PACKAGE / 'AUTHORITY_BINDINGS.json'
    path.write_text(json.dumps({'project_root': str(tmp_path / 'old root'), 'status': 'PASS'}))
    args['sam3_python'] = 'git.exe'
    result = check_environment(tmp_path, target='report', **args)
    categories = {x['category'] for x in result['known_contract_blockers']}
    assert {'BOUND_TO_OTHER_PROJECT_ROOT', 'RUNTIME_PATH_CONTRACT'} <= categories
    assert result['historical_pass_reused'] is False
    assert not result['current_runtime_evidence']


def test_no_process_network_heavy_import_or_file_mutation(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('side effect')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    modules = set(sys.modules)
    check_environment(tmp_path, target='report', **args)
    assert not any(n.split('.')[0] in {'torch', 'sam3'} for n in set(sys.modules) - modules)
    after = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after


def test_invalid_and_empty_metadata_block(tmp_path):
    args = fixture(tmp_path)
    (tmp_path / PACKAGE / 'BUNDLE.json').write_text('[]')
    (tmp_path / 'model.pt').write_bytes(b'')
    result = check_environment(tmp_path, target='report', **args)
    assert 'checkpoint' in result['missing_required']
    assert any(x['category'] == 'INVALID_STATIC_METADATA' for x in result['known_contract_blockers'])
