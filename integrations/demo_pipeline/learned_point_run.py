"""Bounded, provenance-preserving TAPNext++ experiment on decoded object video."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections import Counter

from .balanced_point_run import read, save, binding, check_bindings, sha

ROOT = Path(__file__).resolve().parents[2]
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def implementation():
    return [binding(ROOT / p) for p in (
        'integrations/demo_pipeline/learned_point_run.py',
        'integrations/demo_pipeline/learned_point_contract.py',
        'integrations/demo_pipeline/learned_point_report.py',
        'integrations/demo_pipeline/masked_trace_run.py',
        'integrations/demo_pipeline/supervision.py',
        'scripts/track_learned_object.py')]


def copy_identical(source, destination):
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError('existing input differs: ' + str(destination))
    else:
        shutil.copyfile(source, destination)


def prepare(args):
    import numpy as np
    from PIL import Image, ImageFilter
    out = args.output.resolve()
    if (out / 'PLAN.json').exists():
        raise FileExistsError('a prepared experiment is immutable')
    out.mkdir(parents=True, exist_ok=True)
    copy_identical(args.source_run / 'INPUTS.json', out / 'INPUTS.json')
    copy_identical(args.mask_run / 'MASKS.json', out / 'MASKS.json')
    inputs, masks = read(out / 'INPUTS.json'), read(out / 'MASKS.json')
    frames = inputs['frames']
    if not 3 <= len(frames) <= 1000 or not 1 <= args.points <= 512:
        raise ValueError('bounded frames/query count required')
    if [f['decode_index'] for f in frames] != [m['decode_index'] for m in masks['frames']]:
        raise ValueError('mask/frame ordered identity mismatch')
    refs = [binding(out / 'INPUTS.json'), binding(out / 'MASKS.json')]
    dimensions = None
    for f, m in zip(frames, masks['frames']):
        ref = binding(f['path'])
        if ref['sha256'] != f['sha256']: raise ValueError('image drift')
        if (m.get('raw_sha256') != ref['sha256'] or
                Path(m['raw_path']).resolve() != Path(f['path']).resolve()):
            raise ValueError('mask/raw-frame identity mismatch')
        refs.append(ref)
        with Image.open(f['path']) as im:
            if im.mode != 'RGB': raise ValueError('RGB input required')
            if dimensions is None: dimensions = im.size
            if im.size != dimensions: raise ValueError('mixed input dimensions')
        if m['current_observed']:
            ref = binding(m['mask_path'])
            if ref['sha256'] != m['mask_sha256']: raise ValueError('mask drift')
            refs.append(ref)
            with Image.open(m['mask_path']) as im:
                if im.mode != 'L' or im.size != dimensions: raise ValueError('mask domain mismatch')
                if not np.isin(np.asarray(im), [0, 255]).all(): raise ValueError('mask labels')
    if not masks['frames'][0]['current_observed']: raise ValueError('observed seed mask required')
    seed = inputs['seed']
    points = [dict(track_id=0, xy=seed['xy'], identity=seed['identity'])]
    with Image.open(masks['frames'][0]['mask_path']) as im:
        interior = np.asarray(im.filter(ImageFilter.MinFilter(17))) == 255
    x, y = seed['xy']
    if not interior[int(y + .5), int(x + .5)]: raise ValueError('seed lacks interior support')
    if args.points > 1:
        yy, xx = np.mgrid[8:dimensions[1]:16, 8:dimensions[0]:16]
        pool = np.column_stack((xx.ravel(), yy.ravel()))
        pool = pool[interior[pool[:, 1], pool[:, 0]]]
        pool = pool[np.linalg.norm(pool - np.asarray(seed['xy']), axis=1) >= 8]
        if len(pool) < args.points - 1: raise ValueError('insufficient independent interior queries')
        for n, i in enumerate(np.linspace(0, len(pool)-1, args.points-1).round().astype(int), 1):
            points.append(dict(track_id=n, xy=pool[i].tolist(), identity=f'onion_query_{n:04d}'))
    save(out / 'QUERIES.json', dict(points=points, selection='first-frame mask interior, fixed before inference',
                                   coordinate_convention='OpenCV integer pixel centers'))
    refs.append(binding(out / 'QUERIES.json'))
    upstream = args.upstream.resolve(strict=True)
    source_refs = [binding(upstream / p) for p in (
        'tapnet/__init__.py', 'tapnet/tapnext/tapnext_torch.py',
        'tapnet/tapnext/tapnext_lru_modules.py', 'tapnet/tapnext/pscan.py',
        'tapnet/tapnext/tapnext_torch_utils.py',
        'tapnet/tapnextpp/votsp2026/utils.py')]
    checkpoint = binding(args.checkpoint.resolve(strict=True))
    python = (ROOT / 'integrations/sam3/env/Scripts/python.exe').resolve(strict=True)
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions(
        path=[str(python.parent.parent / 'Lib/site-packages')])}
    save(out / 'PLAN.json', dict(schema='learned-point-plan-v1', model='TAPNext++',
        upstream=str(upstream), upstream_revision='730cda1c730877cfedbe01bf87fb1cadb78a565d',
        upstream_bindings=source_refs, checkpoint=checkpoint, implementation=implementation(),
        python=str(python), packages=packages, device='cuda:0', input_resolution=256,
        model_coordinate_resolution=256, autocast='float16', compile=False,
        checkpoint_mmap=True, checkpoint_weights_only=True, strict_load=True,
        visibility_policy='sigmoid(visibility_logits) * tracker_certainty(radius=8) > 0.5',
        query_count=len(points), frames=len(frames), dimensions=dimensions,
        wall_seconds=600, job_commit_bytes=12 << 30, torch_reserved_limit_bytes=7 << 30,
        output_threshold_bytes=2 << 30, geometry_accepted=False))
    save(out / 'SOURCE_BINDINGS.json', refs)
    save(out / 'RESULT.json', dict(status='PREPARED', quality_status='NOT_EVALUATED',
                                  geometry_accepted=False))
    print('prepared', len(frames), 'frames', len(points), 'queries', flush=True)


def check(out, include_checkpoint=True):
    plan = read(out / 'PLAN.json')
    refs = read(out / 'SOURCE_BINDINGS.json') + plan['implementation'] + plan['upstream_bindings']
    if include_checkpoint: refs.append(plan['checkpoint'])
    drift = check_bindings(refs)
    if drift: raise ValueError('bound input or code drift: ' + repr(drift))
    return plan


def worker(out):
    plan = check(out)
    started = time.monotonic()
    records = []
    result = dict(status='RUNNING', model=plan['model'], frames=records,
                  metrics={}, quality_status='NOT_EVALUATED', physical_identity_verified=False,
                  geometry_accepted=False)
    save(out / 'TRACKS.json', result)
    def emit(event):
        with (out / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(monotonic=time.monotonic(), pid=os.getpid(), **event),
                                    allow_nan=False) + '\n')
            stream.flush()
    try:
        import numpy as np
        from PIL import Image
        import torch
        import torchvision
        from .learned_point_contract import cv_to_model_queries, model_to_cv, classify_points
        sys.path.insert(0, plan['upstream'])
        from tapnet.tapnext.tapnext_torch import TAPNext
        from tapnet.tapnext.tapnext_torch_utils import tracker_certainty
        from tapnet.tapnextpp.votsp2026.utils import preprocess_frame
        expected_module = Path(plan['upstream']) / 'tapnet/tapnext/tapnext_torch.py'
        if Path(sys.modules[TAPNext.__module__].__file__).resolve() != expected_module.resolve():
            raise RuntimeError('unexpected TAPNext module origin')
        for name, version in [('torch', torch.__version__), ('torchvision', torchvision.__version__),
                              ('numpy', np.__version__)]:
            if version != plan['packages'][name]: raise RuntimeError('package version drift: '+name)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        device = torch.device(plan['device'])
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < 4 << 30: raise RuntimeError('less than 4GiB device headroom before load')
        save(out / 'ENVIRONMENT.json', dict(python=sys.version, torch=torch.__version__,
            torchvision=torchvision.__version__, numpy=np.__version__,
            device=torch.cuda.get_device_name(device), initial_free_bytes=free,
            total_bytes=total, threads=torch.get_num_threads(),
            imported_TAPNext_module=sys.modules[TAPNext.__module__].__file__))
        emit(dict(stage='load', phase='begin'))
        t0 = time.monotonic()
        model = TAPNext(image_size=(256, 256))
        ckpt = torch.load(plan['checkpoint']['path'], map_location='cpu',
                          weights_only=True, mmap=True)
        weights = ckpt.get('state_dict', ckpt)
        weights = {k.removeprefix('tapnext.'): v for k, v in weights.items()}
        loaded = model.load_state_dict(weights, strict=True)
        model = model.to(device).eval()
        del weights, ckpt
        torch.cuda.synchronize(device)
        result['metrics']['load_seconds'] = time.monotonic() - t0
        result['load'] = dict(missing_keys=loaded.missing_keys, unexpected_keys=loaded.unexpected_keys,
                              parameters=sum(p.numel() for p in model.parameters()))
        emit(dict(stage='load', phase='end', **result['load']))
        inputs, masks = read(out/'INPUTS.json'), read(out/'MASKS.json')
        queries = read(out/'QUERIES.json')['points']
        w, h = plan['dimensions']
        query = torch.from_numpy(cv_to_model_queries([p['xy'] for p in queries], w, h).astype(np.float32)).to(device)[None]
        state = None
        model_seconds = 0.
        loop_start = time.monotonic()
        for frame, mask_row in zip(inputs['frames'], masks['frames']):
            if time.monotonic() - started > plan['wall_seconds'] - 15:
                raise TimeoutError('worker deadline')
            if torch.cuda.memory_reserved(device) > plan['torch_reserved_limit_bytes']:
                raise MemoryError('torch reserved budget')
            emit(dict(stage='frame', phase='begin', decode_index=frame['decode_index']))
            with Image.open(frame['path']) as im: rgb = np.array(im)
            mask = None
            if mask_row['current_observed']:
                with Image.open(mask_row['mask_path']) as im: mask = np.array(im)
            t0 = time.monotonic()
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                video = preprocess_frame(rgb[..., ::-1].copy(), device, 256)
                tracks, logits, visibility, state = model(video=video,
                    query_points=query if state is None else None, state=state)
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
                certainty = tracker_certainty(tracks.float(), logits.float(), radius=8)
                pv = torch.sigmoid(visibility.float())
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - t0
            model_seconds += elapsed
            coords = model_to_cv(tracks[0, 0].float().cpu().numpy(), w, h)
            pv = pv[0, 0, :, 0].cpu().numpy()
            cert = certainty[0, 0, :, 0].float().cpu().numpy()
            statuses = classify_points(coords, pv, cert, mask, w, h)
            points = []
            for i, q in enumerate(queries):
                status = statuses[i]
                points.append(dict(track_id=q['track_id'], xy=coords[i].tolist(),
                    visibility_probability=float(pv[i]), certainty=float(cert[i]),
                    status=status, supported=status == 'VISIBLE_CANDIDATE'))
            row = dict(decode_index=frame['decode_index'], time_seconds=frame['time_seconds'],
                       points=points, inference_seconds=elapsed,
                       torch_allocated_bytes=torch.cuda.memory_allocated(device),
                       torch_reserved_bytes=torch.cuda.memory_reserved(device))
            records.append(row)
            with (out/'frames.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            emit(dict(stage='frame', phase='end', decode_index=frame['decode_index'],
                      candidate_count=sum(p['supported'] for p in points)))
            print('frame', frame['decode_index'], 'candidates', sum(p['supported'] for p in points), flush=True)
            if torch.cuda.memory_reserved(device) > plan['torch_reserved_limit_bytes']:
                raise MemoryError('torch reserved budget after frame')
        result['metrics'].update(frames_completed=len(records), queries=len(queries),
            model_seconds=model_seconds, loop_wall_seconds=time.monotonic()-loop_start,
            torch_max_allocated_bytes=torch.cuda.max_memory_allocated(device),
            torch_max_reserved_bytes=torch.cuda.max_memory_reserved(device))
        result['counts'] = dict(Counter(p['status'] for row in records for p in row['points']))
        result['primary_counts'] = dict(Counter(row['points'][0]['status'] for row in records))
        result['status'] = 'TRACKS_GENERATED_NOT_IDENTITY_VERIFIED'
        return 0
    except Exception as error:
        result.update(status='FAILED', error=type(error).__name__ + ': ' + str(error))
        emit(dict(stage='worker', phase='error', error=result['error']))
        import traceback
        traceback.print_exc()
        return 1
    finally:
        result['metrics']['total_worker_seconds'] = time.monotonic() - started
        result['metrics']['frames_completed'] = len(records)
        save(out / 'TRACKS.json', result)


def run(out):
    from .masked_trace_run import invoke
    plan = check(out)
    argv = [plan['python'], '-B', '-X', 'utf8', str(ROOT/'scripts/track_learned_object.py'),
            'worker', '--output', str(out)]
    entry = invoke(out, 'inference', argv, plan['wall_seconds'], plan['job_commit_bytes'])
    result = dict(status='FAILED', inference=entry, quality_status='NOT_EVALUATED', geometry_accepted=False)
    if (out / 'TRACKS.json').exists():
        result['tracking_status'] = read(out/'TRACKS.json')['status']
    result['source_drift'] = check_bindings(read(out/'SOURCE_BINDINGS.json'))
    result['implementation_drift'] = check_bindings(plan['implementation'] + plan['upstream_bindings'])
    if (entry.get('root_exit_code') == 0 and entry.get('reclaim_confirmed')
            and not result['source_drift'] and not result['implementation_drift']):
        from .learned_point_report import render
        render(out)
        result['status'] = 'EXPERIMENT_COMPLETE_NOT_IDENTITY_VERIFIED'
        save(out/'OUTPUT_BINDINGS.json', [binding(out/n) for n in (
            'TRACKS.json','ENVIRONMENT.json','frames.jsonl','events.jsonl','index.html',
            'motion.gif','contact_sheet.png','overview.png','LEARNED_POINT_REPORT.json')])
    save(out/'RESULT.json', result)
    return 0 if result['status'].startswith('EXPERIMENT_COMPLETE') else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run','worker','report'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-run',type=Path); p.add_argument('--mask-run',type=Path)
    p.add_argument('--upstream',type=Path); p.add_argument('--checkpoint',type=Path)
    p.add_argument('--points',type=int,default=1)
    args=p.parse_args(argv)
    if args.action=='prepare':
        if not all((args.source_run,args.mask_run,args.upstream,args.checkpoint)):
            p.error('prepare requires source-run, mask-run, upstream and checkpoint')
        prepare(args); return 0
    out=args.output.resolve(strict=True)
    if args.action=='worker': return worker(out)
    if args.action=='run': return run(out)
    from .learned_point_report import render
    render(out); return 0
