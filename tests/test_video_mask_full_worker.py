"""Full-window contracts with a fake backend and real PNG publication."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from integrations.video_masks import full_worker as full


def config(tmp_path, count=3, frames=2):
    repo = tmp_path / 'repo'
    (repo / 'sam3').mkdir(parents=True)
    (repo / 'sam3/model_builder.py').write_text('# TEST_DATA')
    seed = tmp_path / 'seed.png'
    from PIL import Image
    Image.new('L', (2, 2), 255).save(seed)
    tasks = []
    for i in range(count):
        task_id = f'task_{i:03d}'
        folder = tmp_path / 'run/windows' / task_id
        images = folder / 'frames'
        images.mkdir(parents=True)
        for n in range(frames):
            (images / f'{n:06d}.jpg').write_bytes(b'TEST_DATA')
        task = dict(task_id=task_id, frames_dir=str(images), output_dir=str(folder / 'worker'),
                    frame_count=frames, max_frames=96, direction='forward',
                    start_decode_index=17 + i * (frames - 1), end_decode_index=17 + (i + 1) * (frames - 1))
        if i:
            task['seed_from'] = dict(task_id=tasks[-1]['task_id'], frame_id=frames - 1, frame_index=0)
        else:
            task['seed'] = dict(frame_index=0, mask_path=str(seed), mask_sha256=full.worker.sha256(seed), root_seed_decode_index=17)
        tasks.append(task)
    return dict(repo_root=str(repo), checkpoint_path=str(seed), bpe_path=str(seed),
                device='cuda:0', threads=2, output_dir=str(tmp_path / 'run/worker'), tasks=tasks)


def runtime(monkeypatch, raw, fail=None, change_parent=False):
    loads, calls = [], []
    original = SimpleNamespace(load=lambda *a, **k: loads.append((a, k)))
    builder = SimpleNamespace(torch=original, __file__=str(Path(raw['repo_root']) / 'sam3/model_builder.py'))
    model = object()
    def build(**kwargs):
        assert kwargs['strict_state_dict_loading'] and not kwargs['load_from_HF'] and not kwargs['compile']
        with open(kwargs['checkpoint_path'], 'rb') as stream:
            builder.torch.load(stream, map_location='cpu', weights_only=True)
        return model
    builder.build_sam3_video_model = build
    torch = SimpleNamespace(set_num_threads=lambda n: None)
    monkeypatch.setattr(full.importlib, 'import_module', lambda n: torch if n == 'torch' else builder)
    monkeypatch.setattr(full.worker, 'model_audit', lambda m, d: {'parameter_count': 1})
    def propagate(cfg, t, m, emit):
        import numpy as np
        calls.append((cfg['task_id'], cfg['direction'], id(m)))
        for n in full._order(cfg):
            emit('frame_begin', frame_id=n)
            metadata = full.worker.commit_mask(np.array([[True, False], [False, False]]), Path(cfg['output_dir']) / f'{n:06d}.png')
            emit('frame_commit', frame_id=n, status='propagated', **metadata)
            if cfg['task_id'] == fail:
                raise RuntimeError('TEST_DATA interrupted after commit')
        if change_parent and cfg['task_id'] == 'task_000':
            Path(metadata['mask_path']).write_bytes(b'TEST_DATA changed')
    monkeypatch.setattr(full.worker, 'propagate', propagate)
    return loads, calls, builder, original


def test_single_mmap_load_chain_and_real_png(tmp_path, monkeypatch):
    raw = config(tmp_path)
    loads, calls, builder, original = runtime(monkeypatch, raw)
    assert full.run(raw) == 0
    assert len(loads) == 1 and loads[0][1]['mmap'] is True and builder.torch is original
    assert len(calls) == 3 and len({c[2] for c in calls}) == 1
    result = json.loads((Path(raw['output_dir']) / 'result.json').read_text())
    assert result['load_audit'] == {'mmap_load_count': 1, 'builder_torch_restored': True}
    for i, record in enumerate(result['tasks']):
        assert json.loads((Path(record['output_dir']) / 'result.json').read_text()) == record
        assert len(record['frames']) == 2
        assert record['seed_lineage']['root_seed_decode_index'] == 17
        assert record['seed_lineage']['kind'] == ('propagated_parent' if i else 'independent_anchor')
        for item in record['frames']:
            assert item['direction'] == 'forward' and item['foreground_pixels'] == 1
            assert full.worker.sha256(item['mask_path']) == item['mask_sha256']
        if i:
            assert record['actual_seed']['mask_path'] == result['tasks'][i - 1]['frames'][-1]['mask_path']
    events = [json.loads(s) for s in (Path(raw['output_dir']) / 'events.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'ready' and events[-1]['event'] == 'done'
    assert [e['seq'] for e in events] == list(range(len(events)))
    assert all(e['pid'] > 0 and e['monotonic_ns'] > 0 for e in events)
    assert all('task_id' in e and 'direction' in e for e in events if e['event'].startswith('frame_'))


@pytest.mark.parametrize('change_parent', [False, True])
def test_failure_preserves_active_and_stops(tmp_path, monkeypatch, change_parent):
    raw = config(tmp_path)
    runtime(monkeypatch, raw, fail=None if change_parent else 'task_001', change_parent=change_parent)
    assert full.run(raw) == 1
    result = json.loads((Path(raw['output_dir']) / 'result.json').read_text())
    assert [r['status'] for r in result['tasks']] == ['PROPAGATED_NOT_SEMANTICALLY_EVALUATED', 'FAILED', 'NOT_RUN']
    failed = result['tasks'][1]
    assert len(failed['frames']) == (0 if change_parent else 1)
    assert ('hash changed' if change_parent else 'interrupted') in failed['error']
    assert json.loads((Path(failed['output_dir']) / 'result.json').read_text()) == failed
    assert not Path(raw['tasks'][2]['output_dir']).exists()


@pytest.mark.parametrize('mutation', ['future', 'missing', 'frame', 'index', 'both', 'override', 'scope', 'existing', 'frames', 'source', 'count', 'budget', 'window_bound', 'lineage', 'interval_type', 'interval_count', 'parent_direction', 'decode_identity', 'ref_decode', 'anchor_decode'])
def test_rejected_before_load(tmp_path, mutation):
    raw = config(tmp_path, count=28 if mutation == 'budget' else 3, frames=96 if mutation == 'budget' else 2)
    if mutation == 'future': raw['tasks'][1]['seed_from']['task_id'] = 'task_002'
    if mutation == 'missing': raw['tasks'][1]['seed_from']['task_id'] = 'absent'
    if mutation == 'frame': raw['tasks'][1]['seed_from']['frame_id'] = 2
    if mutation == 'index': raw['tasks'][1]['seed_from']['frame_index'] = 2
    if mutation == 'both': raw['tasks'][1]['direction'] = 'both'
    if mutation == 'override': raw['tasks'][1]['threads'] = 2
    if mutation == 'scope': raw['tasks'][1]['output_dir'] = str(tmp_path / 'escape/worker')
    if mutation == 'existing': Path(raw['tasks'][1]['output_dir']).mkdir()
    if mutation == 'frames': raw['tasks'][1]['frames_dir'] = raw['tasks'][0]['frames_dir']
    if mutation == 'source': raw['bpe_path'] = raw['output_dir'] + '/x'
    if mutation == 'count': raw['tasks'] *= 22
    if mutation == 'window_bound': raw['tasks'][0]['frame_count'] = 97
    if mutation == 'lineage': raw['tasks'][1]['root_seed_decode_index'] = 999
    if mutation == 'interval_type': raw['tasks'][0]['start_decode_index'] = 17.0
    if mutation == 'interval_count': raw['tasks'][0]['end_decode_index'] += 1
    if mutation == 'parent_direction': raw['tasks'][1]['direction'] = 'reverse'
    if mutation == 'decode_identity':
        raw['tasks'][1]['start_decode_index'] += 1
        raw['tasks'][1]['end_decode_index'] += 1
    if mutation == 'ref_decode': raw['tasks'][1]['seed_from']['decode_index'] = 999
    if mutation == 'anchor_decode': raw['tasks'][0]['seed']['root_seed_decode_index'] = 999
    with pytest.raises((ValueError, FileNotFoundError)):
        full.validate_config(raw)
    assert not Path(raw['output_dir']).exists()


def test_uncommitted_reference_rejected(tmp_path):
    cfg = full.validate_config(config(tmp_path))
    with pytest.raises(ValueError, match='not COMMITTED'):
        full.resolve_seed(cfg['tasks'][1], {})


def test_reverse_window(tmp_path, monkeypatch):
    raw = config(tmp_path, count=1)
    raw['tasks'][0]['direction'] = 'reverse'
    raw['tasks'][0]['seed']['frame_index'] = 1
    raw['tasks'][0]['seed']['root_seed_decode_index'] = 18
    runtime(monkeypatch, raw)
    assert full.run(raw) == 0
    result = json.loads((Path(raw['output_dir']) / 'result.json').read_text())
    assert [f['frame_id'] for f in result['tasks'][0]['frames']] == [1, 0]
