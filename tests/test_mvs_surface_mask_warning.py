"""TEST_DATA runner flow; fabricated supervision/logs, no native launch."""
import json
from pathlib import Path

import pytest
import test_mvs_surface_runner as base

from integrations.mvs_surface import runner, supervision

inputs = base.inputs


def install(monkeypatch, mode, channel):
    calls = base._install_supervisor(monkeypatch)
    original = supervision.run_supervised
    def fake(argv, **kwargs):
        native = original(argv, **kwargs)
        evidence, work, label = kwargs['output'], kwargs['cwd'], kwargs['label']
        # Exact field schema/path written by real run_supervised adapter.
        native['mvs_surface_adapter'] = {'schema': 'mvs-surface-supervision-v1',
                                         'output': str(evidence)}
        if label == 'dense':
            paths = {'stdout': evidence / 'stdout.txt', 'stderr': evidence / 'stderr.txt',
                     'native': work / 'DensifyPointCloud-TEST_DATA.log'}
            paths['stdout'].write_bytes(b'')
            paths['stderr'].write_bytes(b'')
            paths['native'].write_bytes(b'TEST_DATA OpenMVS log: normal completion\n')
            if mode == 'warning':
                paths[channel].write_bytes(b"18:00:00 warning: can not load the segmentation mask 'x.mask.png'\n")
            elif mode == 'decode_error':
                paths[channel].write_bytes(b'18:00:00 error: cannot load mask image x.mask.png\n')
            elif mode == 'missing':
                paths[channel].unlink()
            elif mode == 'unreadable':
                paths[channel].unlink(); paths[channel].mkdir()
            elif mode == 'empty_native': paths['native'].write_bytes(b'')
            elif mode == 'nul_native': paths['native'].write_bytes(b'corrupt\x00log')
            elif mode == 'bad_adapter': native['mvs_surface_adapter']['output'] = str(work)
            elif mode == 'ansi_warning':
                paths[channel].write_bytes(b"\xff\x81\xfe warning: CAN NOT LOAD THE SEGMENTATION MASK 'x'\n")
        return native
    monkeypatch.setattr(supervision, 'run_supervised', fake)
    return calls


def run(inputs, tmp_path):
    return runner.execute_plan(base._plan(inputs), tmp_path / 'run')


@pytest.mark.parametrize('mode', ['warning', 'decode_error', 'ansi_warning'])
@pytest.mark.parametrize('channel', ['stdout', 'stderr', 'native'])
def test_actual_runner_stops_before_mesh_on_mask_warning(inputs, tmp_path, monkeypatch, mode, channel):
    calls = install(monkeypatch, mode, channel)
    result = run(inputs, tmp_path)
    assert result['status'] == 'FAILED' and result['exit_code'] == 4
    assert [c['label'] for c in calls] == ['dense']
    assert result['native_launches'] == 1
    guard = result['stages'][0]['mask_log_guard']
    assert guard['status'] == 'MASK_LOAD_DIAGNOSTIC_FOUND' and guard['matches']
    assert guard['matches'][0]['path'].endswith({'stdout': 'stdout.txt', 'stderr': 'stderr.txt',
                                              'native': 'DensifyPointCloud-TEST_DATA.log'}[channel])
    assert result['raw_mesh_status'] == 'NOT_RUN'
    assert result['mask_consumption'] == 'NOT_VERIFIED'
    saved = json.loads((tmp_path / 'run/result.json').read_text(encoding='utf-8'))
    assert saved['stages'][0]['mask_log_guard'] == guard


@pytest.mark.parametrize('mode', ['missing', 'unreadable'])
@pytest.mark.parametrize('channel', ['stdout', 'stderr', 'native'])
def test_actual_runner_missing_unreadable_logs_prevent_mesh(inputs, tmp_path, monkeypatch, mode, channel):
    calls = install(monkeypatch, mode, channel)
    result = run(inputs, tmp_path)
    assert result['exit_code'] == 4
    assert [c['label'] for c in calls] == ['dense']
    assert result['stages'][0]['mask_log_guard']['status'] == 'LOG_EVIDENCE_UNAVAILABLE'
    assert result['raw_mesh_status'] == 'NOT_RUN'


@pytest.mark.parametrize('mode', ['empty_native', 'nul_native', 'bad_adapter'])
def test_actual_runner_invalid_log_evidence_prevents_mesh(inputs, tmp_path, monkeypatch, mode):
    calls = install(monkeypatch, mode, 'native')
    result = run(inputs, tmp_path)
    assert result['exit_code'] == 4
    assert [c['label'] for c in calls] == ['dense']
    assert result['stages'][0]['mask_log_guard']['status'] == 'LOG_EVIDENCE_UNAVAILABLE'


def test_normal_logs_allow_original_recipe_without_promoting_consumption(inputs, tmp_path, monkeypatch):
    calls = install(monkeypatch, 'normal', 'native')
    result = run(inputs, tmp_path)
    assert result['status'] == 'SURFACE_GENERATED_NOT_ACCEPTED' and result['exit_code'] == 0
    assert [c['label'] for c in calls] == ['dense', 'mesh']
    assert result['stages'][0]['mask_log_guard']['status'] == 'NO_MASK_LOAD_DIAGNOSTIC_FOUND'
    assert len(result['stages'][0]['mask_log_guard']['files']) == 3
    assert result['mask_consumption'] == 'NOT_VERIFIED'
    assert runner.DENSE_OPTIONS['resolution-level'] == 1
    assert runner.DENSE_OPTIONS['sub-resolution-levels'] == 2
    assert runner.DENSE_OPTIONS['geometric-iters'] == 2
