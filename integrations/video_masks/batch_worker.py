"""Bounded offline multi-clip propagation sharing one SAM3 video model."""
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


def validate_config(raw):
    cfg = dict(raw)
    output = Path(cfg['output_dir']).resolve()
    if output.name != 'worker' or output.exists():
        raise ValueError('root output_dir must be an absent run/worker directory')
    clips = cfg.get('clips')
    if not isinstance(clips, list) or not 1 <= len(clips) <= 6:
        raise ValueError('batch requires 1..6 clips')
    normalized = []
    ids = set()
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError('clip must be an object')
        if any(key in clip for key in ('repo_root', 'checkpoint_path', 'bpe_path', 'device', 'threads')):
            raise ValueError('clip cannot override common model bindings')
        clip_id = clip.get('clip_id')
        if not isinstance(clip_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', clip_id) or clip_id in ids:
            raise ValueError('clip_id must be unique and path-safe')
        ids.add(clip_id)
        if clip.get('direction') != 'both':
            raise ValueError('batch clip direction must be both')
        child = worker.validate_config({**cfg, **clip})
        expected = output.parent / 'clips' / clip_id / 'worker'
        if Path(child['output_dir']) != expected or Path(child['output_dir']).exists():
            raise ValueError('clip output_dir must be absent run/clips/<clip_id>/worker')
        normalized.append(child)
    if sum(c['frame_count'] for c in normalized) > 384:
        raise ValueError('batch frame budget exceeds 384')
    frames = [Path(c['frames_dir']) for c in normalized]
    if len(set(frames)) != len(frames):
        raise ValueError('clip frames_dir paths must be unique')
    outputs = [output] + [Path(c['output_dir']) for c in normalized]
    sources = frames + [Path(normalized[0][key]) for key in ('repo_root', 'checkpoint_path', 'bpe_path')]
    sources += [Path(s['mask_path']) for c in normalized for s in c['seeds']]
    def overlap(a, b):
        return a == b or a in b.parents or b in a.parents
    if any(overlap(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1:]):
        raise ValueError('output paths overlap')
    if any(overlap(a, b) for a in outputs for b in sources):
        raise ValueError('output overlaps immutable source')
    if any(overlap(a, b) for i, a in enumerate(frames) for b in frames[i + 1:]):
        raise ValueError('clip frame paths overlap')
    return {**cfg, **{k: normalized[0][k] for k in ('repo_root', 'checkpoint_path', 'bpe_path', 'device', 'threads')},
            'output_dir': str(output), 'clips': normalized}


def run(raw):
    cfg = validate_config(raw)
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    audit, runtime_audit = {}, {}
    records = [{'clip_id': c['clip_id'], 'status': 'NOT_RUN', 'frame_count': c['frame_count'],
                'output_dir': c['output_dir'], 'frames': []} for c in cfg['clips']]
    active = None
    with (output / 'events.jsonl').open('x', encoding='utf-8') as events:
        sequence = 0
        def emit(event, **values):
            nonlocal sequence
            record = {'event': event, 'time_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
                      'pid': os.getpid(), 'seq': sequence, **values}
            events.write(json.dumps(record) + '\n')
            events.flush()
            os.fsync(events.fileno())
            sequence += 1
        try:
            emit('ready', frame_count=sum(c['frame_count'] for c in cfg['clips']), direction='both', clip_count=len(records))
            for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
                os.environ[key] = '1'
            os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
            sys.path.insert(0, cfg['repo_root'])
            emit('load_begin')
            torch = importlib.import_module('torch')
            torch.set_num_threads(2)
            builder = importlib.import_module('sam3.model_builder')
            if Path(builder.__file__).resolve() != Path(cfg['repo_root']) / 'sam3' / 'model_builder.py':
                raise RuntimeError('SAM3 builder import provenance mismatch')
            with worker.builder_mmap(builder, cfg['checkpoint_path']) as audit:
                model = builder.build_sam3_video_model(checkpoint_path=cfg['checkpoint_path'],
                    bpe_path=cfg['bpe_path'], load_from_HF=False, compile=False,
                    strict_state_dict_loading=True, device=cfg['device'])
            runtime_audit = worker.model_audit(model, cfg['device'])
            emit('load_end', **audit, strict_state_dict_loading=True, model_audit=runtime_audit)
            for child, record in zip(cfg['clips'], records):
                active = record
                Path(child['output_dir']).mkdir(parents=True, exist_ok=False)
                record.update(status='FAILED', load_audit=audit, model_audit=runtime_audit)
                def clip_emit(event, **values):
                    emit(event, clip_id=child['clip_id'], **values)
                    if event == 'frame_commit':
                        record['frames'].append(dict(values))
                try:
                    record['directions'] = worker.propagate_directions(child, torch, model, clip_emit)
                    record['status'] = 'PROPAGATED_NOT_SEMANTICALLY_EVALUATED'
                except Exception as exc:
                    record['error'] = str(exc)
                    # Include completed and failed direction evidence published by the shared helper.
                    record['directions'] = {p.name: json.loads((p / 'result.json').read_text(encoding='utf-8'))
                        for p in Path(child['output_dir']).iterdir() if p.is_dir() and (p / 'result.json').is_file()}
                    raise
                finally:
                    worker.write_result(Path(child['output_dir']) / 'result.json', record)
                active = None
            result = {'status': 'PROPAGATED_NOT_SEMANTICALLY_EVALUATED', 'clips': records,
                      'load_audit': audit, 'model_audit': runtime_audit}
            code = 0
        except Exception as exc:
            emit('fatal', error_type=type(exc).__name__, error=str(exc), load_audit=audit,
                 **({'clip_id': active['clip_id']} if active else {}))
            result = {'status': 'FAILED', 'clips': records, 'error': str(exc),
                      'load_audit': audit, 'model_audit': runtime_audit}
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
