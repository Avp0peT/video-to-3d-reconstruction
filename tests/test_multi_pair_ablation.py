"""Native six-camera controls for the frozen multi-pair engineering ablation."""
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integrations.demo_pipeline import parallel_photo_audit as prior


def multi_camera_fixture(tmp_path, count=29, camera_model="SIMPLE_RADIAL"):
    """Extend camera_fixture's real binary encoding to six known poses."""
    root = tmp_path / 'source'
    binary = root / 'models/0/binary'
    binary.mkdir(parents=True)
    (root / 'images').mkdir()
    xyz = np.random.default_rng(1996).uniform(
        (-0.7, -0.6, -0.6), (0.7, 0.6, 0.6), (count, 3))
    image_bytes = struct.pack('<Q', 6)
    camera_bytes = struct.pack('<Q', 6)
    frames = []
    for ident in range(1, 7):
        angle = -0.07 + 0.04 * (ident - 1)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array(((cosine, 0., sine), (0., 1., 0.),
                             (-sine, 0., cosine)))
        center = np.array((0.6 * (ident - 1), 0.03 * (ident - 1), -8.))
        translation = -rotation @ center
        quaternion = (float(np.cos(angle / 2)), 0.,
                      float(np.sin(angle / 2)), 0.)
        params = (1000. - 10. * (ident - 1), 512., 341.5,
                  0. if camera_model == "SIMPLE_PINHOLE" else 0.01 if ident % 2 else -0.005)
        f, cx, cy, k = params
        if camera_model == 'SIMPLE_PINHOLE':
            camera_bytes += struct.pack('<iiQQddd', ident, 0, 1024, 683, *params[:3])
        else:
            camera_bytes += struct.pack('<iiQQdddd', ident, 2, 1024, 683, *params)
        name = f'images/{ident}.png'
        photo = root / name
        photo.write_bytes(b'fixed synthetic photograph byte fixture ' + str(ident).encode())
        image_bytes += (struct.pack('<idddddddi', ident, *quaternion,
                                    *translation, ident) + name.encode() + b'\0'
                        + struct.pack('<Q', count))
        for pid, point in enumerate(xyz, 1):
            cp = rotation @ point + translation
            u, v = cp[:2] / cp[2]
            factor = 1 + k * (u * u + v * v)
            image_bytes += struct.pack('<ddq', f * u * factor + cx,
                                       f * v * factor + cy, pid)
        frames.append(dict(upstream_io_run_id='synthetic_multi_camera_io',
            input_index=ident - 1, frame_id=f'frame{ident}', staged_image_name=name,
            canonical_image_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),
            registrations=[dict(model_id='0', image_id=ident, camera_id=ident)],
            status='REGISTERED'))
    point_bytes = struct.pack('<Q', count)
    for pid, point in enumerate(xyz, 1):
        point_bytes += struct.pack('<QdddBBBdQ', pid, *point, 1, 2, 3, 0., 6)
        for ident in range(1, 7):
            point_bytes += struct.pack('<ii', ident, pid - 1)
    for name, payload in [('images.bin', image_bytes), ('points3D.bin', point_bytes),
                          ('cameras.bin', camera_bytes)]:
        (binary / name).write_bytes(payload)
    (root / 'frames_map.jsonl').write_text(
        '\n'.join(json.dumps(row) for row in frames), encoding='utf-8')
    (root / 'run.json').write_text(json.dumps(dict(
        contract_version='sfm-native-object-v1', status='SUCCESS',
        upstream_io_run_id='synthetic_multi_camera_io', models=[dict(
            model_id='0', binary_path='models/0/binary', registered_image_ids=list(range(1, 7)),
            num_points3D=count, usable=True, roundtrip_verified=True)])), encoding='utf-8')
    old = tmp_path / 'prior'
    prior.audit(root, '0', old)
    frozen = old / 'INPUT.json'
    data = json.loads(frozen.read_text(encoding='utf-8'))
    assert data['image_pair'] == [1, 2]
    assert data['point_ids'] == list(range(1, 7))
    assert data['common_tracks'] == count
    return root, binary, frozen, xyz


def ablation():
    return importlib.import_module('integrations.demo_pipeline.multi_pair_ablation')


def test_disjoint_pairs_and_ties_are_deterministic(tmp_path):
    _, binary, _, _ = multi_camera_fixture(tmp_path)
    images, points = prior.read_native(binary / 'images.bin', binary / 'points3D.bin')
    first = ablation().select_pairs(images, points, (1, 2))
    # Insertion order cannot change a common-track tie.
    second = ablation().select_pairs(dict(reversed(list(images.items()))),
                                    dict(reversed(list(points.items()))), (1, 2))
    assert first == second
    assert [row['image_pair'] for row in first['selected']] == [[1, 2], [3, 4], [5, 6]]
    assert [row['role'] for row in first['selected']] == ['reference', 'additional', 'additional']
    flattened = [ident for row in first['selected'] for ident in row['image_pair']]
    assert len(flattened) == len(set(flattened)) == 6
    assert all(row['common_ids'] == list(range(1, 30)) for row in first['selected'])


@pytest.mark.parametrize('count', [29, 6 * 47 + 5])
def test_groups_stay_whole_without_repacking_or_duplication(count):
    common = list(range(1, count + 1))
    selection = ablation().select_groups(common)
    blocks = [dict(block_index=index, point_ids=common[index * 6:(index + 1) * 6])
              for index in range(count // 6)]
    expected_indices = (list(range(len(blocks))) if len(blocks) <= 40 else
                        [index * (len(blocks) - 1) // 39 for index in range(40)])
    assert selection['selected'] == [blocks[index] for index in expected_indices]
    assert selection['unselected_blocks'] == [block for block in blocks
        if block['block_index'] not in expected_indices]
    assert selection['tail_point_ids'] == common[(count // 6) * 6:]
    selected_ids = [pid for block in selection['selected'] for pid in block['point_ids']]
    assert len(selected_ids) == len(set(selected_ids))
    assert not set(selected_ids).intersection(selection['tail_point_ids'])
    assert ablation().select_groups(common) == selection


@pytest.mark.parametrize('changed', ['photo', 'native'])
def test_multi_pair_source_drift_rejected_without_output(tmp_path, changed):
    root, binary, frozen, _ = multi_camera_fixture(tmp_path)
    target = root / 'images/1.png' if changed == 'photo' else binary / 'cameras.bin'
    target.write_bytes(target.read_bytes() + b'changed bound source')
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        ablation().experiment(frozen, output)
    assert not output.exists()


def test_multi_pair_output_never_overwritten(tmp_path):
    _, _, frozen, _ = multi_camera_fixture(tmp_path)
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'evidence.bin'
    marker.write_bytes(b'original evidence')
    with pytest.raises(ValueError):
        ablation().experiment(frozen, output)
    assert marker.read_bytes() == b'original evidence'
    assert list(output.iterdir()) == [marker]


def test_tiny_native_multi_pair_actual_algorithm_and_readback(tmp_path):
    _, _, frozen, truth = multi_camera_fixture(tmp_path)
    frozen_bytes = frozen.read_bytes()
    output = tmp_path / 'experiment'
    result = ablation().experiment(frozen, output)
    assert frozen.read_bytes() == frozen_bytes
    assert result['selected_pairs_count'] == 3
    assert result['group_count'] == 12
    assert result['counts']['positive']['requested'] == 12
    assert result['counts']['positive']['C_world_ok'] == 12
    assert result['counts']['positive']['C_certificate_verified'] == 12
    assert result['counts']['negative_control']['requested'] == 12
    assert json.loads((output / 'RESULT.json').read_text(encoding='utf-8')) == result
    plan = json.loads((output / 'PLAN.json').read_text(encoding='utf-8'))
    assert [row['image_pair'] for row in plan['pairs']] == [[1, 2], [3, 4], [5, 6]]
    for pair in plan['pairs']:
        selection = pair['group_selection']
        assert selection['tail_point_ids'] == [25, 26, 27, 28, 29]
        assert selection['unselected_blocks'] == []
        assert [row['point_ids'] for row in selection['selected']] == [
            list(range(start, start + 6)) for start in (1, 7, 13, 19)]
    text = (output / 'candidate_samples.ply').read_text(encoding='ascii')
    assert 'element vertex 72' in text
    candidate = np.array([[float(value) for value in row.split()]
        for row in text.split('end_header\n', 1)[1].splitlines()])
    assert candidate.shape == (72, 3)
    # Three pair samples preserve duplicate scene IDs; no cross-pair averaging.
    assert np.max(np.linalg.norm(candidate - np.tile(truth[:24], (3, 1)), axis=1)) < 0.02
    samples = json.loads((output / 'CANDIDATE_SAMPLES.json').read_text(encoding='utf-8'))['samples']
    assert len(samples) == 72
    assert [(row['pair_index'], row['point_id']) for row in samples] == [
        (pair, pid) for pair in range(3) for pid in range(1, 25)]
    assert np.array_equal(candidate, np.array([row['world_xyz'] for row in samples]))
    for pair in range(3):
        for group in range(4):
            labels = list(range(group * 6 + 1, group * 6 + 7))
            positive = json.loads((output / 'groups' /
                f'p{pair}_g{group:04d}_positive.json').read_text(encoding='utf-8'))
            methods = positive['ablation']
            assert positive['point_ids'] == labels
            assert methods['C']['algorithm_called'] == 'AlgT_prime_both_views'
            assert methods['C']['independent']['status'] == 'VERIFIED'
            assert methods['C']['projection_readback_exact'] is True
            assert methods['C']['projection_metrics']['status'] == 'NUMERIC_ROUNDTRIP_VERIFIED'
            assert methods['errors']['B_vs_C']['max_abs'] == 0.0
            assert methods['B']['world_xyz'] == methods['C']['world_xyz']
            negative = json.loads((output / 'groups' /
                f'p{pair}_g{group:04d}_negative_control.json').read_text(encoding='utf-8'))
            assert negative['left_source_track_ids'] == labels
            assert negative['right_source_track_ids'] == labels[1:] + labels[:1]
    payloads = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
    with pytest.raises(ValueError):
        ablation().experiment(frozen, output)
    assert all(path.read_bytes() == payload for path, payload in payloads.items())
