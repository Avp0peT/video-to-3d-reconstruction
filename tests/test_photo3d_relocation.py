import json
from pathlib import Path
import subprocess
import socket
import pytest
from integrations.demo_pipeline import relocation as m


def fixture(tmp_path, monkeypatch):
    source = tmp_path / 'source with spaces'; source.mkdir()
    package = source / m.runtime.PACKAGE_RELATIVE; package.mkdir(parents=True)
    model = source / 'model.pt'; model.write_bytes(b'weights-fixture-not-real')
    for rel in (m.preflight.HEAVY, m.preflight.LIGHT):
        p = source / rel; p.parent.mkdir(parents=True); p.write_bytes(b'python-fixture')
        (p.parent.parent / 'pyvenv.cfg').write_text('home = OLD_BASE\nversion = 3.12.13\n')
    dependency = source / 'integrations/sam3/sam3_adapter/__init__.py'
    dependency.parent.mkdir(parents=True); dependency.write_bytes(b'fixture')
    row = {'path': dependency.relative_to(source).as_posix(), 'sha256': m.runtime.sha256(dependency), 'bytes': dependency.stat().st_size}
    stat = model.stat(); sha = m.runtime.sha256(model)
    documents = {'PROFILE.json': {'schema': 'sam3-demo-text-runtime-profile-v2', 'mode': 'text_only'},
        'AUTHORITY_BINDINGS.json': {'project_root': str(source), 'checkout_head': 'abc', 'files': [row], 'checkpoint_stat_only': {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}},
        'START_BASELINE.json': {'project_root': str(source), 'checkout_head': 'abc', 'files': [row], 'checkpoint': {'path': str(model), 'sha256': sha}, 'package_inventory': []}}
    for name, doc in documents.items(): (package / name).write_text(json.dumps(doc))
    worker = package / 'worker_entry/mmap_review_worker.py'; worker.parent.mkdir()
    worker.write_text('from pathlib import Path\nROOT = Path(' + repr(str(source)) + ')\nBOUND_CHECKPOINT = ' + repr(str(model)) + '\nBOUND_SHA256 = ' + repr(sha) + '\n')
    rows = [{'path': p.relative_to(package).as_posix(), 'sha256': m.runtime.sha256(p), 'bytes': p.stat().st_size} for p in package.rglob('*') if p.is_file()]
    (package / 'BUNDLE.json').write_text(json.dumps({'files': rows}))
    seal = m.runtime.sha256(package / 'BUNDLE.json'); monkeypatch.setattr(m.runtime, 'BUNDLE_SHA256', seal)
    target = tmp_path / 'TARGET_NOT_CREATED'
    paths = dict(project_python=str(target / 'light/Scripts/python.exe'), sam3_python=str(target / m.preflight.HEAVY),
        sam3_base_python=str(target / 'integrations/sam3/python/cpython-3.12-windows-x86_64-none/python.exe'),
        sam3_source=str(target / 'integrations/sam3/src/sam3-repo'), sam3_package=str(target / m.runtime.PACKAGE_RELATIVE),
        checkpoint=str(target / 'model.pt'), checkpoint_sha256=sha, git=str(target / 'tools/git.exe'), input_kind='photos', target='report')
    return dict(source_root=str(source), source_package=str(package), expected_source_bundle_sha256=seal, target_root=str(target), target_paths=paths)


def test_real_fixture_plan_and_exclusive_write_never_creates_target(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch); plan = m.plan_relocation(**args)
    assert plan['status'] == 'PLANNED_REVIEW_REQUIRED' and not plan['execution_authorized'] and plan['run_argv'] is None
    assert not plan['checkpoint_identity']['bytes_rehashed'] and not Path(args['target_root']).exists()
    out = Path(args['source_root']) / 'outputs/new_plan/PLAN.json'
    saved = m.write_plan(plan, out)
    assert saved['sha256'] == m.runtime.sha256(out) and not Path(args['target_root']).exists()
    with pytest.raises(ValueError, match='new directory'): m.write_plan(plan, out)


@pytest.mark.parametrize('mutation', ['digest', 'missing', 'duplicate', 'overlap', 'layout', 'drift', 'escape'])
def test_rejections(tmp_path, monkeypatch, mutation):
    args = fixture(tmp_path, monkeypatch)
    if mutation == 'digest': args['expected_source_bundle_sha256'] = '0' * 64
    elif mutation == 'missing': args['target_paths'].pop('git')
    elif mutation == 'duplicate': args['target_paths']['git'] = args['target_paths']['project_python']
    elif mutation == 'overlap': args['target_root'] = args['source_root']
    elif mutation == 'layout': args['target_paths']['sam3_python'] = str(Path(args['target_root']) / 'wrong.exe')
    elif mutation == 'drift': (Path(args['source_root']) / 'integrations/sam3/sam3_adapter/__init__.py').write_bytes(b'drift')
    elif mutation == 'escape':
        bundle = Path(args['source_package']) / 'BUNDLE.json'; data = json.loads(bundle.read_text()); data['files'][0]['path'] = '../escape'
        bundle.write_text(json.dumps(data)); args['expected_source_bundle_sha256'] = m.runtime.sha256(bundle)
        monkeypatch.setattr(m.runtime, 'BUNDLE_SHA256', args['expected_source_bundle_sha256'])
    with pytest.raises(ValueError): m.plan_relocation(**args)
    assert not Path(tmp_path / 'TARGET_NOT_CREATED').exists()


def test_4mib_metadata_limit_and_no_process_network_weight_hash(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch); original = m.runtime.sha256
    def safe_sha(path):
        assert Path(path).name != 'model.pt', 'weight bytes must not be rehashed'
        return original(path)
    def forbidden(*a, **k): raise AssertionError('native/network forbidden')
    monkeypatch.setattr(m.runtime, 'sha256', safe_sha); monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden); monkeypatch.setattr(socket, 'socket', forbidden)
    before = {p: p.read_bytes() for p in Path(args['source_root']).rglob('*') if p.is_file()}
    m.plan_relocation(**args)
    assert before == {p: p.read_bytes() for p in Path(args['source_root']).rglob('*') if p.is_file()}
    p = Path(args['source_package']) / 'PROFILE.json'; p.write_bytes(b' ' * (4 << 20) + b'{}')
    with pytest.raises(ValueError, match='metadata limit'): m.plan_relocation(**args, max_metadata_bytes=4 << 20)


def test_cli_and_output_guard(tmp_path, monkeypatch, capsys):
    args = fixture(tmp_path, monkeypatch)
    plan = m.plan_relocation(**args)
    with pytest.raises(ValueError, match='source outputs'): m.write_plan(plan, Path(args['target_root']) / 'PLAN.json')
    p = tmp_path / 'paths.json'; p.write_text(json.dumps(args.pop('target_paths')))
    argv = []
    for key, value in args.items(): argv += ['--' + key.replace('_', '-'), value]
    argv += ['--target-paths', str(p), '--output', str(Path(args['source_root']) / 'outputs/cli-new/PLAN.json')]
    assert m.main(argv) == 0
    assert json.loads(capsys.readouterr().out)['execution_authorized'] is False
    assert m.relative_file(Path(args['source_root']), 'integrations\\sam3\\sam3_adapter\\__init__.py') == Path(args['source_root']) / 'integrations/sam3/sam3_adapter/__init__.py'
    with pytest.raises(ValueError): m.relative_file(Path(args['source_root']), '..\\escape')
