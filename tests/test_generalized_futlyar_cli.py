import json
from pathlib import Path
import importlib.util
from fractions import Fraction

import pytest

SPEC = importlib.util.spec_from_file_location('futlyar_cli', Path(__file__).resolve().parents[1] / 'scripts/verify_generalized_futlyar.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


@pytest.fixture
def source(tmp_path):
    value = {'schema': 'generalized-futlyar-input-v1',
             'points': [{'label': 7, 'xy': [0, 0]}, {'label': 9, 'xy': [cli.encode(Fraction(1, 2)), 0]}, {'label': 11, 'xy': [3, 0]}],
             'centers': [{'label': 20, 'xy': [0, 0]}, {'label': 30, 'xy': [3, 0]}], 'radius': 1}
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_real_cli_certificate_roundtrip_and_tamper(source, tmp_path):
    output, check = tmp_path / 'result.json', tmp_path / 'check.json'
    assert cli.main(['--input', str(source), '--output', str(output)]) == 0
    saved = json.loads(output.read_text())
    assert saved['independent_check'] == 'TRUE'
    assert saved['certificate']['blocks'] == [[20, [7, 9]], [30, [11]]]
    assert cli.main(['--input', str(source), '--verify', str(output), '--output', str(check)]) == 0
    saved['certificate']['blocks'][0][1].append(11)
    output.write_text(json.dumps(saved))
    assert cli.main(['--input', str(source), '--verify', str(output), '--output', str(tmp_path / 'bad.json')]) == 1
    assert json.loads((tmp_path / 'bad.json').read_text())['status'] == 'FALSE'


def test_unknown_not_false_and_existing_output_is_preserved(source, tmp_path):
    output = tmp_path / 'result.json'
    assert cli.main(['--input', str(source), '--output', str(output), '--max-work', '0']) == 1
    before = output.read_bytes()
    assert json.loads(before)['status'] == 'UNKNOWN'
    assert cli.main(['--input', str(source), '--output', str(output)]) == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize('corrupt', ['duplicate-key', 'bool-label', 'float-coordinate', 'duplicate-label', 'unreduced-rational'])
def test_strict_source_rejection(source, tmp_path, corrupt):
    data = json.loads(source.read_text())
    if corrupt == 'bool-label': data['points'][0]['label'] = True
    if corrupt == 'float-coordinate': data['points'][0]['xy'][0] = .1
    if corrupt == 'duplicate-label': data['points'][0]['label'] = 9
    if corrupt == 'unreduced-rational': data['radius'] = {'numerator': '2', 'denominator': '2'}
    raw = json.dumps(data)
    if corrupt == 'duplicate-key': raw = raw.replace('"radius": 1', '"radius": 1, "radius": 2')
    source.write_text(raw)
    output = tmp_path / 'out.json'
    assert cli.main(['--input', str(source), '--output', str(output)]) == 2
    assert not output.exists()


def test_verification_binds_input_bytes(source, tmp_path):
    output = tmp_path / 'result.json'
    assert cli.main(['--input', str(source), '--output', str(output)]) == 0
    source.write_text(source.read_text() + '\n')
    assert cli.main(['--input', str(source), '--verify', str(output), '--output', str(tmp_path / 'check.json')]) == 2
