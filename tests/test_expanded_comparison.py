"""Real COLMAP binary fixture, complete exactly-once candidate provenance."""
import importlib
import json
import hashlib
import struct
from test_multi_pair_ablation import multi_camera_fixture


def module():
    return importlib.import_module('integrations.demo_pipeline.expanded_comparison')


def digests(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_greedy_uncovered_no_regrouping():
    points = {i: dict(track={1: 0, 2: 0, 3: 0}) for i in range(1, 14)}
    points.update({i: dict(track={3: 0, 4: 0}) for i in range(14, 20)})
    plan = module().assign(points)
    assert [p['image_pair'] for p in plan['selected']] == [[1, 2], [3, 4]]
    assert plan['selected'][0]['tail_point_ids'] == [13]
    ids = [i for p in plan['selected'] for i in p['point_ids']]
    assert sorted(ids) == list(range(1, 20)) and len(set(ids)) == len(ids)
    assert module().assign(dict(reversed(list(points.items())))) == plan


def test_real_binary_fixture_same_ids_and_source_unchanged(tmp_path):
    source, binary, frozen, truth = multi_camera_fixture(tmp_path, count=13)
    # Change old point XYZ without changing original raw measurements.
    p = binary/'points3D.bin'; data = bytearray(p.read_bytes())
    for offset in range(8, len(data), 51+6*8):
        struct.pack_into('<ddd', data, offset+8, 1e8, -1e8, 1e8)
    p.write_bytes(data)
    # Rebind the prior after the deliberate fixture edit, proving old XYZ is unused.
    native = importlib.import_module('integrations.demo_pipeline.parallel_photo_audit')
    prior = json.loads(frozen.read_text(encoding='utf-8'))
    prior['source_bindings'] = [native.binding(r['path']) for r in prior['source_bindings']]
    frozen.write_text(json.dumps(prior), encoding='utf-8')
    before = digests(source); out = tmp_path/'expanded'
    result = module().generate(frozen, out)
    assert digests(source) == before
    assert result['paired_count'] == 12 and result['unpaired_count'] == 1
    records = json.loads((out/'PAIRED_POINTS.json').read_text())['points']
    assert [r['point_id'] for r in records] == list(range(1, 13))
    for r in records:
        assert len(r['observations']) == 6 and r['C_success']
        assert max(abs(v) for v in r['C_xyz']+r['J_xyz']) < 2
        assert r['J_details']['train_image_ids'] == list(range(1, 7))
        evidence = json.loads((out/r['evidence_file']).read_text())
        assert evidence['ablation']['C']['independent']['status'] == 'VERIFIED'
        assert evidence['ablation']['C']['projection_readback_exact']
    ledger = json.loads((out/'UNPAIRED_LEDGER.json').read_text())['points']
    assert ledger[0]['point_id'] == 13
    assert result['old_xyz_used'] is False


def test_exclusive_output(tmp_path):
    import pytest
    _, _, frozen, _ = multi_camera_fixture(tmp_path)
    output = tmp_path/'exists'; output.mkdir()
    with pytest.raises(ValueError, match='exclusive'):
        module().generate(frozen, output)
    assert not list(output.iterdir())
