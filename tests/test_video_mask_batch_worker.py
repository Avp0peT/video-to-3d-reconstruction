"""Mocked batch contracts; no torch, GPU, or model execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from integrations.video_masks import batch_worker as batch


def config(tmp_path, count=3, frames=2):
    repo = tmp_path / 'repo'
    (repo / 'sam3').mkdir(parents=True)
    (repo / 'sam3/model_builder.py').write_text('# TEST_DATA')
    seed = tmp_path / 'seed.png'
    seed.write_bytes(b'TEST_DATA')
    clips = []
    for i in range(count):
        clip_id = f'clip_{i:03d}'
        folder = tmp_path / 'run/clips' / clip_id
        images = folder / 'frames'
        images.mkdir(parents=True)
        for n in range(frames):
            (images / f'{n:06d}.jpg').write_bytes(b'TEST_DATA')
        clips.append(dict(clip_id=clip_id, frames_dir=str(images), output_dir=str(folder / 'worker'),
            frame_count=frames, max_frames=frames, direction='both',
            seeds=[dict(frame_index=n, mask_path=str(seed), mask_sha256=batch.worker.sha256(seed))
                   for n in (0, frames - 1)]))
    return dict(repo_root=str(repo), checkpoint_path=str(seed), bpe_path=str(seed),
                device='cuda:0', threads=2, output_dir=str(tmp_path / 'run/worker'), clips=clips)


def runtime(monkeypatch, raw, fail=None):
    loads, models, calls = [], [], []
    original = SimpleNamespace(load=lambda *a, **k: loads.append((a, k)))
    builder = SimpleNamespace(torch=original, __file__=str(Path(raw['repo_root']) / 'sam3/model_builder.py'))
    model = object()
    def build(**kwargs):
        assert kwargs['strict_state_dict_loading'] and not kwargs['load_from_HF'] and not kwargs['compile']
        with open(kwargs['checkpoint_path'], 'rb') as stream:
            builder.torch.load(stream, map_location='cpu', weights_only=True)
        models.append(model)
        return model
    builder.build_sam3_video_model = build
    torch = SimpleNamespace(set_num_threads=lambda n: calls.append(('threads', n)))
    monkeypatch.setattr(batch.importlib, 'import_module', lambda n: torch if n == 'torch' else builder)
    monkeypatch.setattr(batch.worker, 'model_audit', lambda m, d: {'parameter_count': 1})
    def propagate(cfg, t, m, emit):
        import numpy as np
        calls.append((cfg['clip_id'], cfg['direction'], id(m)))
        order = range(cfg['frame_count']) if cfg['direction'] == 'forward' else range(cfg['frame_count'] - 1, -1, -1)
        committed = []
        for n in order:
            emit('frame_begin', frame_id=n)
            metadata = batch.worker.commit_mask(np.array([[True, False], [False, False]]),
                Path(cfg['output_dir']) / f'{n:06d}.png')
            item = dict(frame_id=n, status='propagated', **metadata)
            emit('frame_commit', **item)
            committed.append(item)
            if cfg['clip_id'] == fail:
                raise RuntimeError('TEST_DATA interrupted after commit')
        return committed
    monkeypatch.setattr(batch.worker, 'propagate', propagate)
    return loads, models, calls, builder, original


def test_one_load_independent_directions_and_complete_evidence(tmp_path, monkeypatch):
    raw = config(tmp_path)
    loads, models, calls, builder, original = runtime(monkeypatch, raw)
    assert batch.run(raw) == 0
    assert len(loads) == len(models) == 1
    assert loads[0][1]['mmap'] is True
    assert builder.torch is original
    assert calls[0] == ('threads', 2)
    assert [(x[0], x[1]) for x in calls[1:]] == [(c['clip_id'], d) for c in raw['clips'] for d in ('forward', 'reverse')]
    assert len({x[2] for x in calls[1:]}) == 1
    result = json.loads((Path(raw['output_dir']) / 'result.json').read_text())
    assert result['load_audit'] == {'mmap_load_count': 1, 'builder_torch_restored': True}
    for clip, record in zip(raw['clips'], result['clips']):
        persisted = json.loads((Path(clip['output_dir']) / 'result.json').read_text())
        assert persisted == record
        assert len(persisted['frames']) == 4
        assert set(persisted['directions']) == {'forward', 'reverse'}
        assert all(x['mask_sha256'] == batch.worker.sha256(x['mask_path']) for x in persisted['frames'])
        assert all(x['foreground_pixels'] == 1 for x in persisted['frames'])
        assert [x['frame_id'] for x in persisted['directions']['forward']['frames']] == [0, 1]
        assert [x['frame_id'] for x in persisted['directions']['reverse']['frames']] == [1, 0]
    events = [json.loads(x) for x in (Path(raw['output_dir']) / 'events.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'ready' and events[-1]['event'] == 'done'
    assert [e['seq'] for e in events] == list(range(len(events)))
    assert all(e['pid'] > 0 and e['time_ns'] > 0 and e['monotonic_ns'] > 0 for e in events)
    assert all('clip_id' in e and 'direction' in e for e in events if e['event'].startswith('frame_'))


def test_first_failure_preserves_commits_and_stops_batch(tmp_path, monkeypatch):
    raw = config(tmp_path)
    loads, models, calls, builder, original = runtime(monkeypatch, raw, fail='clip_001')
    assert batch.run(raw) == 1
    result = json.loads((Path(raw['output_dir']) / 'result.json').read_text())
    assert [c['status'] for c in result['clips']] == ['PROPAGATED_NOT_SEMANTICALLY_EVALUATED', 'FAILED', 'NOT_RUN']
    failed = json.loads((Path(raw['clips'][1]['output_dir']) / 'result.json').read_text())
    assert len(failed['frames']) == 1
    commit = failed['directions']['forward']['frames'][0]
    assert commit['mask_sha256'] == batch.worker.sha256(commit['mask_path'])
    assert not Path(raw['clips'][2]['output_dir']).exists()
    assert len(loads) == 1 and builder.torch is original
    events = [json.loads(x) for x in (Path(raw['output_dir']) / 'events.jsonl').read_text().splitlines()]
    assert events[-2]['event'] == 'fatal' and events[-2]['clip_id'] == 'clip_001'
    assert events[-1]['event'] == 'done' and events[-1]['exit_code'] == 1


@pytest.mark.parametrize('mutation', ['root', 'scope', 'ids', 'frames', 'direction', 'existing', 'override', 'source'])
def test_reject_scope_and_bindings(tmp_path, mutation):
    raw = config(tmp_path)
    if mutation == 'root': raw['output_dir'] = str(tmp_path / 'run/other')
    if mutation == 'scope': raw['clips'][0]['output_dir'] = str(tmp_path / 'escape/worker')
    if mutation == 'ids': raw['clips'][1]['clip_id'] = raw['clips'][0]['clip_id']
    if mutation == 'frames': raw['clips'][1]['frames_dir'] = raw['clips'][0]['frames_dir']
    if mutation == 'direction': raw['clips'][0]['direction'] = 'forward'
    if mutation == 'existing': Path(raw['clips'][0]['output_dir']).mkdir()
    if mutation == 'override': raw['clips'][0]['threads'] = 2
    if mutation == 'source': raw['bpe_path'] = str(Path(raw['output_dir']).parent)
    with pytest.raises(ValueError): batch.validate_config(raw)
    assert not Path(raw['output_dir']).exists()


@pytest.mark.parametrize('count,frames', [(7, 2), (5, 96), (1, 97)])
def test_budget_limits(tmp_path, count, frames):
    with pytest.raises(ValueError): batch.validate_config(config(tmp_path, count=count, frames=frames))


def test_load_failure_keeps_all_clips_not_run(tmp_path, monkeypatch):
    raw = config(tmp_path)
    monkeypatch.setattr(batch.importlib, 'import_module', lambda name: (_ for _ in ()).throw(RuntimeError('TEST_DATA load failed')))
    assert batch.run(raw) == 1
    result = json.loads((Path(raw['output_dir']) / 'result.json').read_text())
    assert all(c['status'] == 'NOT_RUN' for c in result['clips'])
