"""Offline selector wiring: native extraction, DB ordering and fail-closed gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kozlov_lab.reconstruction import cli, colmap_adapter as adapter, foreground, point_selection
from test_colmap_adapter import make_upstream_run
from test_foreground_reconstruction import database, selected_masks, staged_fixture
from test_object_first_pipeline import rig, injected, renderer
from integrations.demo_pipeline import pipeline as p


@pytest.mark.parametrize('selector,profile,manifest', [
    ('rcover-v1', 'cad-object-v1', None),
    ('rcover-v1', 'default', 'unused.json'),
    ('unknown-selector', 'cad-object-v1', 'unused.json'),
])
def test_invalid_selector_request_stops_before_backend(tmp_path, monkeypatch, selector, profile, manifest):
    # Use the real default profile identifier rather than relying on its spelling.
    if profile == 'default':
        profile = cli.profiles.DEFAULT_PROFILE
    monkeypatch.setattr(cli, 'backend_info', lambda *a: pytest.fail('backend started'))
    out = tmp_path / 'out'
    out.mkdir()
    assert cli.run_reconstruction(io_run='not-read', run_dir=out, project_root=tmp_path,
        profile=profile, object_mask_manifest=manifest, point_selection=selector) == 2
    report = json.loads((out / 'run.json').read_text())
    assert report['status'] == 'INPUT_ERROR'
    assert report['errors'][0]['code'] == 'INPUT_ERROR'
    assert ('unknown point selection' if selector.startswith('unknown') else 'requires cad-object-v1') in report['errors'][0]['message']


class ReadbackReached(RuntimeError):
    pass


@pytest.mark.parametrize('selector,fail_selection', [('native', False), ('rcover-v1', False), ('rcover-v1', True)])
def test_selector_runs_after_native_write_before_database_readback(tmp_path, monkeypatch, selector, fail_selection):
    _, staged = staged_fixture(tmp_path)
    events = []
    db = tmp_path / 'database.db'

    def native_extract(*args, **kwargs):
        events.append('native_extract')
        assert not db.exists()
        # Actual SQLite feature storage, with centres inside the bound masks.
        database(db, staged.mask_bindings)

    def select(path, bindings, evidence):
        events.append('select')
        assert events == ['native_extract', 'select']
        assert adapter.audit_foreground_keypoints(path, bindings)['checked_keypoints'] == 2
        assert evidence == tmp_path / 'point_selection.json'
        if fail_selection:
            raise ValueError('intentional selector failure')
        return {'policy_id': 'rcover-v1'}

    def readback(path):
        events.append('database_readback')
        raise ReadbackReached('readback reached')

    monkeypatch.setattr(point_selection, 'apply_point_selection', select if selector != 'native'
        else lambda *a: pytest.fail('native default called selector'))
    fake = SimpleNamespace(FeatureExtractionOptions=lambda: SimpleNamespace(sift=SimpleNamespace()),
        ImageReaderOptions=lambda: SimpleNamespace(), CameraMode=SimpleNamespace(AUTO='auto'),
        Device=SimpleNamespace(cpu='cpu'), extract_features=native_extract,
        Database=SimpleNamespace(open=readback))
    spec = dict(database_path=str(db), image_path=str(staged.images_dir),
        image_names=[b['image_name'] for b in staged.mask_bindings], num_threads=2,
        max_long_side=1024, max_sift_features=4096, camera_model='SIMPLE_RADIAL',
        camera_mode='AUTO', focal_length_factor=1.2,
        foreground_mask_path=str(staged.masks_dir), foreground_mask_bindings=staged.mask_bindings)
    # Omit the key for the default case: the historical spec stays compatible.
    if selector != 'native':
        spec['point_selection'] = selector
    error = ValueError if fail_selection else ReadbackReached
    with pytest.raises(error, match='intentional selector failure' if fail_selection else 'readback reached'):
        adapter._stage_extract(fake, spec)
    assert events == (['native_extract', 'select'] if fail_selection else
        ['native_extract', 'select', 'database_readback'] if selector != 'native' else
        ['native_extract', 'database_readback'])


@pytest.mark.parametrize('case', ['missing_audit', 'unverified', 'modified_coordinates',
                                  'verified', 'drifted_evidence'])
def test_selection_evidence_gate_controls_match(tmp_path, monkeypatch, case):
    audit = {'status': 'VERIFIED', 'policy_id': 'rcover-v1', 'after_total': 2,
             'coordinates_modified': False, 'descriptors_preserved': True}
    if case == 'missing_audit':
        audit = None
    elif case == 'unverified':
        audit = {'status': 'NOT_VERIFIED'}
    elif case == 'modified_coordinates':
        audit['coordinates_modified'] = True
    for key in cli.profiles.RUNTIME_THREAD_ENV:
        monkeypatch.setenv(key, '2')
    source = make_upstream_run(tmp_path / 'input', entries=[dict(value=v, mask=None) for v in [10, 20, 30]])
    masks = selected_masks(tmp_path, [0, 2])
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')

    def foreground_selection(upstream, path, out):
        out.mkdir()
        ledger = out / 'ledger.json'
        ledger.write_text('{}')
        return dict(selected_indices=[0, 2], masks_by_index=masks, ledger_path=ledger,
            manifest_sha256=adapter.sha256_file(manifest), coverage={'selected': 2})

    monkeypatch.setattr(foreground, 'select_foreground', foreground_selection)
    monkeypatch.setattr(cli, 'backend_info', lambda *a: dict(name='pycolmap', version=adapter.BACKEND_PIN, runtime='cpu'))
    calls = []

    def stage(name, spec, **kwargs):
        calls.append(name)
        if name != 'extract_features':
            assert case == 'verified', 'unverified selector reached matching'
            assert name == cli.STAGES[1]
            return adapter.NativeStageResult(stage=name, started_utc='T', finished_utc='T',
                returncode=4, timed_out=False, stdout_path='', stderr_path='', result=None,
                error='intentional match stop after verified selection')
        assert spec['point_selection'] == 'rcover-v1'
        if case in ('verified', 'drifted_evidence'):
            persisted = dict(audit)
            if case == 'drifted_evidence':
                persisted['after_total'] = 999
            (out / 'point_selection.json').write_text(json.dumps(persisted), encoding='utf-8')
        database(Path(spec['database_path']), spec['foreground_mask_bindings'])
        names = spec['image_names']
        result = dict(num_images=2, num_cameras=1, num_keypoints_total=2,
            cameras={'1': dict(model_name='SIMPLE_RADIAL', width=16, height=12,
                params=[19.2, 8., 6., 0.], has_prior_focal_length=False)},
            database_image_ids=[1, 2], image_names_to_ids=dict(zip(names, [1, 2])),
            image_camera_ids=dict.fromkeys(names, 1), point_selection_audit=audit,
            actual_options={'reader': {'mask_path': spec['foreground_mask_path']}},
            foreground_mask_audit=adapter.audit_foreground_keypoints(spec['database_path'],
                spec['foreground_mask_bindings']))
        return adapter.NativeStageResult(stage=name, started_utc='T', finished_utc='T',
            returncode=0, timed_out=False, stdout_path='', stderr_path='', result=result, error=None)

    monkeypatch.setattr(cli, 'run_native_stage', stage)
    out = tmp_path / 'out'
    out.mkdir()
    assert cli.run_reconstruction(io_run=str(source), run_dir=out, project_root=tmp_path,
        profile='cad-object-v1', object_mask_manifest=manifest, point_selection='rcover-v1') == 4
    report = json.loads((out / 'run.json').read_text())
    if case == 'verified':
        assert calls == ['extract_features', cli.STAGES[1]]
        assert 'intentional match stop' in report['errors'][0]['message']
        assert report['point_selection']['status'] == 'VERIFIED'
        assert report['point_selection']['evidence_sha256'] == adapter.sha256_file(out / 'point_selection.json')
        assert report['foreground']['mask_consumption_verified'] is True
    else:
        assert calls == ['extract_features']
        expected = 'evidence differs from worker result' if case == 'drifted_evidence' else 'missing verified'
        assert expected in report['errors'][0]['message']


@pytest.mark.parametrize('selector', ['native', 'rcover-v1'])
def test_pipeline_freezes_and_passes_selector(rig, selector):
    root, options = rig
    options.point_selection = selector
    options.post_review = False
    executor, calls, argv = injected()
    run, code = p.run_pipeline(options, root=root, executor=executor, renderer=renderer)
    assert calls == ['input', 'segmentation', 'sfm']
    assert code != 0  # Injected NO_MODEL must remain incomplete.
    assert p.read(run / 'PLAN.json')['point_selection'] == selector
    assert argv[0][argv[0].index('--point-selection') + 1] == selector


def test_cli_selector_default_and_explicit_value():
    parser = cli.build_parser()
    assert parser.parse_args(['--io-run', 'in', '--output', 'out']).point_selection == 'native'
    assert parser.parse_args(['--io-run', 'in', '--output', 'out',
        '--point-selection', 'rcover-v1']).point_selection == 'rcover-v1'


@pytest.mark.parametrize('selector,selection_status,fail_report', [
    ('native', None, False),
    ('rcover-v1', 'NOT_VERIFIED', False),
    ('rcover-v1', 'VERIFIED', False),
    ('rcover-v1', 'VERIFIED', True),
])
def test_pipeline_selected_points_report_gate_and_failure(rig, monkeypatch, selector,
                                                         selection_status, fail_report):
    from integrations.demo_pipeline import selected_point_report
    root, options = rig
    options.point_selection = selector
    options.post_review = False
    executor, stage_calls, _ = injected()
    report_calls = []
    sfm_runs = []

    def execute(root, run, stage, argv, **limits):
        outcome = executor(root, run, stage, argv, **limits)
        if stage == 'sfm' and selection_status is not None:
            sfm_run = Path(argv[argv.index('--output') + 1]) / 'fixture'
            sfm_runs.append(sfm_run)
            record = p.read(sfm_run / 'run.json')
            record['point_selection'] = {'status': selection_status, 'policy_id': selector,
                                         'geometry_acceptance': False}
            p.save(sfm_run / 'run.json', record)
        return outcome

    def build(sfm_run, output):
        report_calls.append((Path(sfm_run), Path(output)))
        assert stage_calls == ['input', 'segmentation', 'sfm']
        assert p.read(Path(sfm_run) / 'run.json')['point_selection']['status'] == 'VERIFIED'
        if fail_report:
            raise ValueError('intentional selected point receipt failure')
        output.mkdir()
        (output / 'index.html').write_text('EXPLICIT_TEST_DIAGNOSTIC', encoding='utf-8')
        result = {'status': 'DIAGNOSTIC_READY', 'geometry_acceptance': False,
                  'test_receipt': 'explicit injected diagnostic'}
        p.save(output / 'RESULT.json', result)
        return result

    monkeypatch.setattr(selected_point_report, 'build_report', build)
    run, code = p.run_pipeline(options, root=root, executor=execute, renderer=renderer)
    assert code != 0  # The injected SfM NO_MODEL remains authoritative.
    state = p.read(run / 'pipeline.json')
    pointcloud = p.read(run / 'REPORT_DATA.json')['pointcloud']
    stage = state['stages'].get('point_selection_report', {})
    expected_index = run / 'selected_points/index.html'
    if selection_status != 'VERIFIED':
        assert not report_calls
        assert stage.get('status') != 'DIAGNOSTIC_READY'
        assert not expected_index.exists()
        assert str(expected_index.resolve()) not in pointcloud.get('evidence_paths', [])
    else:
        assert report_calls == [(sfm_runs[0], run / 'selected_points')]
        if fail_report:
            assert stage['status'] == 'FAILED'
            assert 'intentional selected point receipt failure' in stage['error']
            assert not expected_index.exists()
            assert str(expected_index.resolve()) not in pointcloud.get('evidence_paths', [])
        else:
            assert stage['status'] == 'DIAGNOSTIC_READY'
            assert stage['geometry_acceptance'] is False
            assert stage['index'] == 'selected_points/index.html'
            assert stage['result']
            assert expected_index.read_text() == 'EXPLICIT_TEST_DIAGNOSTIC'
            assert str(expected_index.resolve()) in pointcloud['evidence_paths']
