"""Sequential bounded directional windows sharing one strict offline SAM3 load."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time

try:
    from . import worker
except ImportError:
    import worker


def _order(task):
    start = task['seeds'][0]['frame_index']
    return range(start, task['frame_count']) if task['direction'] == 'forward' else range(start, -1, -1)


def validate_config(raw):
    cfg = dict(raw)
    output = Path(cfg['output_dir']).resolve()
    if output.name != 'worker' or output.exists():
        raise ValueError('root output_dir must be absent run/worker')
    tasks = cfg.get('tasks')
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 64:
        raise ValueError('requires 1..64 tasks')
    normalized, earlier = [], {}
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError('task must be an object')
        if any(k in task for k in ('repo_root', 'checkpoint_path', 'bpe_path', 'device', 'threads', 'seeds')):
            raise ValueError('task cannot override common bindings or use seeds')
        task_id = task.get('task_id')
        if not isinstance(task_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', task_id) or task_id in earlier:
            raise ValueError('task_id must be unique and path-safe')
        if task.get('direction') not in ('forward', 'reverse'):
            raise ValueError('task direction must be forward or reverse')
        if task.get('max_frames') != 96:
            raise ValueError('max_frames must be 96')
        start, end = task.get('start_decode_index'), task.get('end_decode_index')
        if (type(start) is not int or type(end) is not int or start < 0 or
                type(task.get('frame_count')) is not int or end - start + 1 != task['frame_count']):
            raise ValueError('task decode interval must be integer, nonnegative and match frame_count')
        if ('seed' in task) == ('seed_from' in task):
            raise ValueError('requires exactly one seed or seed_from')
        if 'seed' in task:
            seed = dict(task['seed'])
            root = seed.get('root_seed_decode_index', task.get('root_seed_decode_index'))
            if (type(seed.get('frame_index')) is not int or type(root) is not int or
                    root != start + seed['frame_index']):
                raise ValueError('original root anchor must equal task start plus local seed index')
            lineage = {'kind': 'independent_anchor', 'root_seed_decode_index': root}
        else:
            ref = task['seed_from']
            if not isinstance(ref, dict):
                raise ValueError('seed_from must be an object')
            parent = earlier.get(ref.get('task_id'))
            if parent is None:
                raise ValueError('seed_from must reference an earlier task')
            if type(ref.get('frame_id')) is not int or ref['frame_id'] not in _order(parent):
                raise ValueError('seed_from frame must be in parent committed processing range')
            if parent['direction'] != task['direction']:
                raise ValueError('parent and child directions must match')
            decode_index = parent['start_decode_index'] + ref['frame_id']
            if (type(ref.get('frame_index')) is not int or decode_index != start + ref['frame_index'] or
                    ('decode_index' in ref and (type(ref['decode_index']) is not int or ref['decode_index'] != decode_index))):
                raise ValueError('seed_from must bind the identical decoded frame in parent and child')
            seed = dict(parent['seeds'][0], frame_index=ref.get('frame_index'))
            lineage = {'kind': 'propagated_parent', 'parent_task_id': ref['task_id'],
                       'parent_frame_id': ref['frame_id'],
                       'root_seed_decode_index': parent['seed_lineage']['root_seed_decode_index']}
            if ('root_seed_decode_index' in task and
                    task['root_seed_decode_index'] != lineage['root_seed_decode_index']):
                raise ValueError('dependent task cannot change root anchor lineage')
        child = worker.validate_config({**cfg, **task, 'seeds': [seed]})
        expected = output.parent / 'windows' / task_id / 'worker'
        if Path(child['output_dir']) != expected or expected.exists():
            raise ValueError('task output must be absent run/windows/<task_id>/worker')
        child['seed_lineage'] = lineage
        normalized.append(child)
        earlier[task_id] = child
    if sum(len(_order(t)) for t in normalized) > 2600:
        raise ValueError('directional frame budget exceeds 2600')
    frames = [Path(t['frames_dir']) for t in normalized]
    outputs = [output] + [Path(t['output_dir']) for t in normalized]
    sources = frames + [Path(normalized[0][k]) for k in ('repo_root', 'checkpoint_path', 'bpe_path')]
    sources += [Path(t['seeds'][0]['mask_path']) for t in normalized]
    def overlap(a, b):
        return a == b or a in b.parents or b in a.parents
    if any(overlap(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1:]):
        raise ValueError('outputs overlap')
    if any(overlap(a, b) for a in outputs for b in sources):
        raise ValueError('output overlaps immutable source')
    if any(overlap(a, b) for i, a in enumerate(frames) for b in frames[i + 1:]):
        raise ValueError('frame paths must be unique and disjoint')
    return {**cfg, **{k: normalized[0][k] for k in ('repo_root', 'checkpoint_path', 'bpe_path', 'device', 'threads')},
            'output_dir': str(output), 'tasks': normalized}


def resolve_seed(task, committed):
    if 'seed_from' not in task:
        return dict(task['seeds'][0])
    ref = task['seed_from']
    item = committed.get((ref['task_id'], ref['frame_id']))
    if item is None:
        raise ValueError('seed_from frame is not COMMITTED in this run')
    path = Path(item['mask_path']).resolve(strict=True)
    digest = worker.sha256(path)
    if digest != item['mask_sha256']:
        raise ValueError('seed_from committed mask hash changed')
    return {'frame_index': ref['frame_index'], 'mask_path': str(path), 'mask_sha256': digest}


def run(raw):
    cfg = validate_config(raw)
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    audit, runtime_audit, committed = {}, {}, {}
    records = [dict(task_id=t['task_id'], status='NOT_RUN', direction=t['direction'],
                    frame_count=t['frame_count'], output_dir=t['output_dir'],
                    start_decode_index=t['start_decode_index'], end_decode_index=t['end_decode_index'],
                    seed_lineage=t['seed_lineage'], frames=[]) for t in cfg['tasks']]
    active = None
    with (output / 'events.jsonl').open('x', encoding='utf-8') as events:
        sequence = 0
        def emit(event, **values):
            nonlocal sequence
            events.write(json.dumps(dict(event=event, time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                                         pid=os.getpid(), seq=sequence, **values)) + '\n')
            events.flush()
            os.fsync(events.fileno())
            sequence += 1
        try:
            emit('ready', frame_count=sum(len(_order(t)) for t in cfg['tasks']), task_count=len(records), direction='windows')
            for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
                os.environ[key] = '1'
            os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
            sys.path.insert(0, cfg['repo_root'])
            emit('load_begin')
            torch = importlib.import_module('torch')
            torch.set_num_threads(2)
            builder = importlib.import_module('sam3.model_builder')
            if Path(builder.__file__).resolve() != Path(cfg['repo_root']) / 'sam3/model_builder.py':
                raise RuntimeError('SAM3 builder import provenance mismatch')
            with worker.builder_mmap(builder, cfg['checkpoint_path']) as audit:
                model = builder.build_sam3_video_model(checkpoint_path=cfg['checkpoint_path'],
                    bpe_path=cfg['bpe_path'], load_from_HF=False, compile=False,
                    strict_state_dict_loading=True, device=cfg['device'])
            runtime_audit = worker.model_audit(model, cfg['device'])
            emit('load_end', **audit, strict_state_dict_loading=True, model_audit=runtime_audit)
            for task, record in zip(cfg['tasks'], records):
                active = record
                Path(task['output_dir']).mkdir(parents=True, exist_ok=False)
                record.update(status='FAILED', load_audit=audit, model_audit=runtime_audit)
                try:
                    seed = resolve_seed(task, committed)
                    record['actual_seed'] = seed
                    child = worker.validate_config(dict(task, seeds=[seed]))
                    def task_emit(event, **values):
                        emit(event, task_id=task['task_id'], direction=task['direction'], **values)
                        if event == 'frame_commit':
                            item = dict(values, direction=task['direction'])
                            record['frames'].append(item)
                            committed[(task['task_id'], item['frame_id'])] = item
                    worker.propagate(child, torch, model, task_emit)
                    if [f['frame_id'] for f in record['frames']] != list(_order(child)):
                        raise RuntimeError('committed frame coverage differs from bounded order')
                    record['status'] = 'PROPAGATED_NOT_SEMANTICALLY_EVALUATED'
                except Exception as exc:
                    record['error'] = str(exc)
                    raise
                finally:
                    worker.write_result(Path(task['output_dir']) / 'result.json', record)
                active = None
            result = dict(status='PROPAGATED_NOT_SEMANTICALLY_EVALUATED', tasks=records,
                          load_audit=audit, model_audit=runtime_audit)
            code = 0
        except Exception as exc:
            emit('fatal', error_type=type(exc).__name__, error=str(exc), load_audit=audit,
                 **({'task_id': active['task_id']} if active else {}))
            result = dict(status='FAILED', tasks=records, error=str(exc), load_audit=audit, model_audit=runtime_audit)
            code = 1
        worker.write_result(output / 'result.json', result)
        emit('done', status=result['status'], exit_code=code)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    with open(args.config, encoding='utf-8-sig') as stream:
        return run(json.load(stream))


if __name__ == '__main__':
    raise SystemExit(main())
