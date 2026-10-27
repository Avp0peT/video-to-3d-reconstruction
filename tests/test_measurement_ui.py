"""Scale UI integration with explicit synthetic callbacks; no native processes."""
from pathlib import Path
import hashlib
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'app')]
import measurement_ui as ui
from integrations.demo_pipeline import measurement


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    source = tmp_path / 'outputs/source'; source.mkdir(parents=True)
    cad = source / 'cad'; cad.mkdir()
    proof = tmp_path / 'synthetic_measurement.txt'; proof.write_text('SYNTHETIC UI TEST; not a physical measurement')
    mesh = {'path': str(source / 'prepared.ply'), 'bytes': 4, 'sha256': 'a' * 64}
    data = {'schema': 'photo-measurement-source-v1', 'run': str(source), 'cad_output': str(cad),
            'mesh': mesh, 'vertex_count': 4, 'face_count': 4}
    calls = []
    monkeypatch.setattr(measurement, 'inspect_source', lambda *args: data)
    def request(*args, **kwargs):
        calls.append(('request', args, kwargs))
        return {'schema_version': 'mesh-scale-request-v1', 'data_kind': 'synthetic_ui_fixture',
                'reference': {'length': args[2], 'unit': args[3]}}
    monkeypatch.setattr(measurement, 'create_request', request)
    def plan(run, reference, output, **kwargs):
        assert Path(reference).is_file() and not Path(output).exists()
        assert json.loads(Path(reference).read_text())['data_kind'] == 'synthetic_ui_fixture'
        calls.append(('plan', run, reference, output))
        return {'output': str(output)}
    monkeypatch.setattr(measurement, 'plan', plan)
    monkeypatch.setattr(measurement, 'execute', lambda frozen: calls.append(('execute', frozen)) or {'status': 'SUCCESS', 'exit_code': 0})
    return source, cad, proof, data, calls


def test_click_path_writes_new_reference_and_no_source_changes(fixture):
    run, cad, proof, _, calls = fixture
    source_before = sorted(run.rglob('*'))
    kwargs = dict(vertex_indices=[0, 1], length=25., unit='mm', label='SYNTHETIC', correspondence='test correspondence', evidence_path=proof)
    one = ui.execute_reference(run, cad, **kwargs)
    two = ui.execute_reference(run, cad, **kwargs)
    assert one['task'] != two['task']
    assert len([x for x in calls if x[0] == 'execute']) == 2
    assert sorted(run.rglob('*')) == source_before
    assert json.loads((Path(one['task']) / 'UI_RESULT.json').read_text()) == one


def test_invalid_reference_does_not_create_output(fixture, monkeypatch):
    run, cad, proof, _, calls = fixture
    def reject(*args, **kwargs): raise ValueError('invalid reference')
    monkeypatch.setattr(measurement, 'create_request', reject)
    with pytest.raises(ValueError, match='invalid reference'):
        ui.execute_reference(run, cad, vertex_indices=[0, 0], length=-1, unit='mm', label='', correspondence='', evidence_path=proof)
    assert not (ui.ROOT / 'outputs/photo_measurement_ui').exists() and not calls


def test_failed_result_keeps_actual_code(fixture, monkeypatch):
    run, cad, proof, _, _ = fixture
    monkeypatch.setattr(measurement, 'execute', lambda _: {'status': 'FAILED', 'exit_code': 5})
    saved = ui.execute_reference(run, cad, vertex_indices=[0, 1], length=25., unit='mm', label='SYNTHETIC', correspondence='test', evidence_path=proof)
    assert saved['status'] == 'FAILED' and saved['exit_code'] == 5
    assert ui.verified_result(saved, run, cad) is None


@pytest.mark.parametrize('change', [{'exit_code': False}, {'exit_code': 1}, {'source_run': 'foreign'}, {'source_cad_output': 'foreign'}])
def test_download_rejects_failed_or_foreign_result(fixture, monkeypatch, change):
    run, cad, _, _, _ = fixture
    monkeypatch.setattr(measurement, 'verify_result', lambda _: pytest.fail('invalid result must not reach verifier'))
    saved = {'source_run': str(run), 'source_cad_output': str(cad), 'status': 'SUCCESS', 'exit_code': 0, 'output': 'unused'}
    saved.update(change)
    assert ui.verified_result(saved, run, cad) is None


def test_payload_checks_the_bytes_sent(tmp_path):
    path = tmp_path / 'fixture.step'; path.write_bytes(b'SYNTHETIC')
    row = {'path': str(path), 'bytes': 9, 'sha256': hashlib.sha256(b'SYNTHETIC').hexdigest()}
    assert ui._payload(row) == b'SYNTHETIC'
    path.write_bytes(b'ALTERED!!')
    with pytest.raises(ValueError): ui._payload(row)


def test_verified_backend_cannot_return_another_cad(fixture, monkeypatch):
    run, cad, _, _, _ = fixture
    monkeypatch.setattr(measurement, 'verify_result', lambda _: {'source_run': str(run), 'source_cad_output': str(cad / 'foreign')})
    saved = {'source_run': str(run), 'source_cad_output': str(cad), 'status': 'SUCCESS', 'exit_code': 0, 'output': 'unused'}
    with pytest.raises(ValueError, match='另一个源任务'):
        ui.verified_result(saved, run, cad)


def test_independent_check_binds_actual_evidence(fixture):
    _, _, proof, _, _ = fixture
    check = ui.independent_check(2, 3, '12.5', 'mm', 'SYNTHETIC', 'fixture only', str(proof), '0.1')
    assert check['vertex_indices'] == [2, 3] and check['length'] == 12.5
    assert check['allowed_absolute_error_mm'] == 0.1
    assert check['evidence']['sha256'] == hashlib.sha256(proof.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='请填写'):
        ui.independent_check(2, 3, '', 'mm', 'SYNTHETIC', 'fixture only', str(proof), '0.1')


def test_real_panel_requires_explicit_submit_and_rerun_never_launches(fixture, monkeypatch):
    from streamlit.testing.v1 import AppTest
    run, cad, proof, data, calls = fixture
    artifacts = []
    for name in ('model.FCStd', 'model.step'):
        p = ui.ROOT / name; p.write_bytes(b'SYNTHETIC DOWNLOAD')
        artifacts.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    monkeypatch.setattr(measurement, 'verify_result', lambda _: {'source_run': str(run), 'source_cad_output': str(cad), 'artifacts': artifacts})
    script = 'from measurement_ui import render_measurement\nfrom pathlib import Path\nrender_measurement(Path(' + repr(str(run)) + '),Path(' + repr(str(cad)) + '))'
    app = AppTest.from_string(script, default_timeout=15).run()
    assert not app.exception and not calls
    next(b for b in app.button if b.label == '载入可定尺度网格').click().run()
    next(b for b in app.button if b.label == '生成带参考尺度的 FCStd / STEP').click().run()
    assert any('请填写' in x.value for x in app.error) and not calls
    for label, value in [('实测参考长度', '25'), ('测量段名称', 'SYNTHETIC'), ('本机测量记录文件路径', str(proof))]:
        next(x for x in app.text_input if x.label == label).set_value(value)
    next(x for x in app.text_area if x.label == '实物与两个网格端点如何对应').set_value('SYNTHETIC test endpoints')
    next(b for b in app.button if b.label == '生成带参考尺度的 FCStd / STEP').click().run()
    assert not app.exception and len([x for x in calls if x[0] == 'execute']) == 1
    assert len(app.get('download_button')) == 2
    app.run()
    assert not app.exception and len([x for x in calls if x[0] == 'execute']) == 1
    assert len(app.get('download_button')) == 2
    assert any('真实性' in x.value and '精度尚未验收' in x.value for x in app.success)


def test_panel_check_rejects_reused_fit_pair_and_forwards_distinct_pair(fixture, monkeypatch):
    from streamlit.testing.v1 import AppTest
    run, cad, proof, _, calls = fixture
    monkeypatch.setattr(measurement, 'verify_result', lambda _: {'source_run': str(run), 'source_cad_output': str(cad), 'artifacts': [], 'independent_check_status': 'PASS'})
    script = 'from measurement_ui import render_measurement\nfrom pathlib import Path\nrender_measurement(Path(' + repr(str(run)) + '),Path(' + repr(str(cad)) + '))'
    app = AppTest.from_string(script, default_timeout=15).run()
    next(b for b in app.button if b.label == '载入可定尺度网格').click().run()
    fields = [('实测参考长度', '25'), ('测量段名称', 'SYNTHETIC'), ('本机测量记录文件路径', str(proof)),
              ('独立校验实测长度', '15'), ('校验段名称', 'SYNTHETIC CHECK'), ('校验允许绝对误差（毫米）', '0.2')]
    for label, value in fields:
        next(x for x in app.text_input if x.label == label).set_value(value)
    for label in ('实物与两个网格端点如何对应', '校验段实物与网格端点对应'):
        next(x for x in app.text_area if x.label == label).set_value('SYNTHETIC TEST ONLY')
    next(x for x in app.checkbox if x.label == '增加独立校验尺寸（不参与定尺度）').set_value(True)
    next(b for b in app.button if b.label == '生成带参考尺度的 FCStd / STEP').click().run()
    assert any('不同于定尺度' in x.value for x in app.error) and not calls
    next(x for x in app.number_input if x.label == '校验段端点 A 编号').set_value(2)
    next(x for x in app.number_input if x.label == '校验段端点 B 编号').set_value(3)
    next(b for b in app.button if b.label == '生成带参考尺度的 FCStd / STEP').click().run()
    assert not app.exception and len([c for c in calls if c[0] == 'execute']) == 1
    request = next(c for c in calls if c[0] == 'request')
    assert request[2]['checks'][0]['vertex_indices'] == [2, 3]
    assert request[2]['checks'][0]['evidence']['sha256'] == hashlib.sha256(proof.read_bytes()).hexdigest()
    assert any('符合本次填写' in x.value for x in app.caption)
    app.run()
    assert len([c for c in calls if c[0] == 'execute']) == 1
