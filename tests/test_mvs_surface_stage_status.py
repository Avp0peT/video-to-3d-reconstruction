"""TEST_DATA stage reporting only; fabricated supervision, no native process."""
import json

import pytest

from test_mvs_surface_runner import inputs, _plan, _install_supervisor, EMPTY_CLOUD
from integrations.mvs_surface import runner, supervision


def test_staging_failure_keeps_dense_not_run(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)
    calls = _install_supervisor(monkeypatch)

    def fail(*args):
        raise ValueError('TEST_DATA staging failed before launch')

    monkeypatch.setattr(runner, 'stage_inputs', fail)
    result = runner.execute_plan(plan, tmp_path / 'staging-failed')
    assert result['status'] == 'FAILED' and result['exit_code'] == 4
    assert result['dense_status'] == result['raw_mesh_status'] == 'NOT_RUN'
    assert result['native_launches'] == 0 and result['stages'] == [] and calls == []


def test_confirmed_no_creation_keeps_dense_not_run(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)
    calls = _install_supervisor(monkeypatch, response={'pid': None, 'status': 'failed'})
    result = runner.execute_plan(plan, tmp_path / 'not-created')
    assert [call['label'] for call in calls] == ['dense']
    assert result['native_launches'] == 0
    assert result['dense_status'] == result['raw_mesh_status'] == 'NOT_RUN'
    assert result['exit_code'] == 4


def test_proven_launch_failed_supervision_keeps_output_unverified(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)
    _install_supervisor(monkeypatch, response={'root_exit_code': 7})
    result = runner.execute_plan(plan, tmp_path / 'failed-supervision')
    assert result['dense_status'] == 'PROCESS_CREATED_OUTPUT_UNVERIFIED'
    assert result['native_launches'] == 1 and result['exit_code'] == 4
    assert result['stages'][0]['returncode'] == 7
    assert result['raw_mesh_status'] == 'NOT_RUN'


@pytest.mark.parametrize('geometry', ['missing', 'empty', 'invalid'])
def test_executed_dense_readback_failure_is_not_not_run(inputs, tmp_path, monkeypatch, geometry):
    plan = _plan(inputs)

    def mutate(label, work):
        assert label == 'dense'
        path = work / '__mvs_dense.ply'
        if geometry == 'missing':
            path.unlink()
        else:
            path.write_bytes(EMPTY_CLOUD if geometry == 'empty' else b'TEST_DATA invalid PLY')

    calls = _install_supervisor(monkeypatch, mutate=mutate)
    target = tmp_path / ('readback-' + geometry)
    result = runner.execute_plan(plan, target)
    assert [call['label'] for call in calls] == ['dense']
    assert result['status'] == 'FAILED' and result['exit_code'] == 4
    assert result['native_launches'] == result['known_native_launches'] == 1
    assert result['dense_status'] == 'NO_VERIFIED_GEOMETRY'
    assert result['raw_mesh_status'] == 'NOT_RUN'
    assert result['stages'][0]['status'] == 'OUTPUT_READBACK_FAILED'
    assert result['stages'][0]['returncode'] == 0
    assert json.loads((target / 'result.json').read_text(encoding='utf-8')) == result


def test_successful_geometry_status_preserved_despite_zero_point_log(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)

    def mutate(label, work):
        if label == 'dense':
            (work / 'DensifyPointCloud-TEST_DATA.log').write_bytes(b'TEST_DATA 0 points\n')

    calls = _install_supervisor(monkeypatch, mutate=mutate)
    result = runner.execute_plan(plan, tmp_path / 'success')
    assert [call['label'] for call in calls] == ['dense', 'mesh']
    assert result['exit_code'] == 0
    assert result['dense_status'] == result['raw_mesh_status'] == 'NONEMPTY_FINITE_GEOMETRY'


def test_later_mesh_failure_preserves_verified_dense(inputs, tmp_path, monkeypatch):
    plan = _plan(inputs)
    calls = _install_supervisor(monkeypatch)
    dense_supervisor = supervision.run_supervised

    def fake(argv, **kwargs):
        result = dense_supervisor(argv, **kwargs)
        if kwargs['label'] == 'mesh':
            result['root_exit_code'] = 7
        return result

    monkeypatch.setattr(supervision, 'run_supervised', fake)
    result = runner.execute_plan(plan, tmp_path / 'mesh-failed')
    assert [call['label'] for call in calls] == ['dense', 'mesh']
    assert result['exit_code'] == 4
    assert result['dense_status'] == 'NONEMPTY_FINITE_GEOMETRY'
