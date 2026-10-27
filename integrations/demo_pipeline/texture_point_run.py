"""Bounded, immutable-input single texture-point experiment; no SAM execution."""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

from .balanced_point_run import ROOT, read, save, binding, check_bindings
from .masked_trace_run import invoke


def code_bindings():
    return [binding(ROOT / p) for p in (
        'scripts/trace_texture_object_point.py',
        'integrations/demo_pipeline/texture_point_trace.py',
        'integrations/demo_pipeline/texture_point_run.py',
        'integrations/demo_pipeline/texture_point_report.py',
        'integrations/demo_pipeline/area_alignment.py',
        'integrations/demo_pipeline/temporal_review.py',
        'integrations/demo_pipeline/balanced_point_run.py',
        'integrations/demo_pipeline/masked_trace_run.py',
        'integrations/demo_pipeline/supervision.py',
        'tests/test_texture_point_trace.py',
    )]


def validate_images(inputs, masks):
    import numpy as np
    from PIL import Image
    fs = inputs['frames']
    ms = masks['frames']
    if len(fs) != len(ms) or not 2 <= len(fs) <= 128:
        raise ValueError('aligned bounded input and mask sequences required')
    previous_index, previous_time = None, None
    refs = []
    for f, m in zip(fs, ms):
        i, t = f['decode_index'], f['time_seconds']
        if type(i) is not int or type(t) not in (int, float) or not math.isfinite(t):
            raise ValueError('invalid timeline')
        if previous_index is not None and (i != previous_index + 1 or t <= previous_time):
            raise ValueError('raw frames must be contiguous and real times increasing')
        previous_index, previous_time = i, t
        r = binding(f['path']); refs.append(r)
        if r['sha256'] != f['sha256'] or r['bytes'] != f['bytes']:
            raise ValueError('raw frame drift')
        if m['decode_index'] != i or Path(m['raw_path']).resolve() != Path(f['path']).resolve() or m['raw_sha256'] != r['sha256']:
            raise ValueError('mask/raw identity mismatch')
        with Image.open(f['path']) as im:
            rgb = np.asarray(im.convert('RGB'))
        h, w = rgb.shape[:2]
        digest = hashlib.sha256(f'{w}x{h}:RGB:'.encode('ascii') + rgb.tobytes()).hexdigest()
        if digest != m['decoded_rgb_sha256'] or [w, h] != [m['width'], m['height']]:
            raise ValueError('decoded identity mismatch')
        if m['current_observed'] is not True:
            if m.get('mask_path') is not None:
                raise ValueError('unobserved mask cannot be supplied')
            continue
        if m['status'] != 'OK':
            raise ValueError('current observation must be OK')
        mask_ref = binding(m['mask_path']); refs.append(mask_ref)
        if mask_ref['sha256'] != m['mask_sha256']:
            raise ValueError('mask drift')
        with Image.open(m['mask_path']) as im:
            if im.mode != 'L' or im.size != (w, h):
                raise ValueError('mask shape or mode invalid')
            a = np.asarray(im)
        if not np.isin(a, [0, 255]).all():
            raise ValueError('nonbinary mask')
        refs.extend([binding(m['source_image']), binding(m['manifest'])])
        if refs[-2]['sha256'] != m['source_image_sha256']:
            raise ValueError('canonical drift')
        with Image.open(m['source_image']) as im:
            if not np.array_equal(np.asarray(im.convert('RGB')), rgb):
                raise ValueError('canonical/raw pixels differ')
    return list({r['path']: r for r in refs}.values())


def prepare(source, out, seed):
    from PIL import Image, ImageDraw
    from .texture_point_trace import PROFILE
    if len(seed) != 2 or not all(math.isfinite(v) for v in seed):
        raise ValueError('finite seed x y required')
    if out.exists():
        raise FileExistsError('fresh output directory required')
    inputs, masks = read(source / 'INPUTS.json'), read(source / 'MASKS.json')
    refs = validate_images(inputs, masks)
    code = code_bindings()
    refs += [binding(source / 'INPUTS.json'), binding(source / 'MASKS.json')]
    # Point is frozen before any tracking of future frames. It is a NEW identity.
    inputs = dict(inputs)
    inputs.pop('comparison', None)
    inputs['seed'] = dict(xy=list(seed), decode_index=inputs['frames'][0]['decode_index'],
        time_seconds=inputs['frames'][0]['time_seconds'], coordinate_convention='OpenCV integer pixel centers',
        selection='first-frame-only visual choice; surface peel/crease; no trajectory-based reselection',
        identity='new_texture_point_01', previous_track_id_reused=False)
    with Image.open(inputs['frames'][0]['path']) as im:
        image = im.convert('RGB')
    x, y = seed
    if x < 17 or y < 17 or x >= image.width - 18 or y >= image.height - 18:
        raise ValueError('seed patch outside image')
    out.mkdir(parents=True)
    save(out / 'INPUTS.json', inputs)
    save(out / 'MASKS.json', masks)
    refs += [binding(out / 'INPUTS.json'), binding(out / 'MASKS.json')]
    save(out / 'SOURCE_BINDINGS.json', refs)
    save(out / 'PLAN.json', dict(schema='fixed-texture-trace-plan-v1', source=str(source),
        profile=PROFILE, implementation=code, seed=inputs['seed'], source_frames=len(inputs['frames']),
        real_tracking_entries=1, automatic_retries=0, sam_calls=0, gpu_calls=0,
        missing_masks_preserved=[r['decode_index'] for r in masks['frames'] if not r['current_observed']],
        book_source='book_selected_pages.txt PDFPAGE 56; approximate affine correspondence, not a visibility test',
        official_reference='https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html',
        geometry_accepted=False, physical_identity_verified=False, occlusion_confirmed=False))
    annotated = image.copy(); d = ImageDraw.Draw(annotated)
    d.rectangle((x-16, y-16, x+16, y+16), outline=(50,255,80), width=2)
    d.line((x-7,y,x+7,y),fill=(50,255,80),width=2)
    d.line((x,y-7,x,y+7),fill=(50,255,80),width=2)
    annotated.save(out / 'seed_view.png')
    save(out / 'RESULT.json', dict(status='PREPARED', tracking_entries=0))
    print('PREPARED', out, flush=True)


def check(out):
    drift = check_bindings(read(out / 'SOURCE_BINDINGS.json') + read(out / 'PLAN.json')['implementation'])
    if drift:
        raise ValueError('bound source/implementation drift: ' + repr(drift))


def worker(out):
    import numpy as np
    from PIL import Image
    from .temporal_review import load_cv
    from .texture_point_trace import trace
    check(out)
    inputs, masks = read(out / 'INPUTS.json'), read(out / 'MASKS.json')
    cvrefs = []; cv = load_cv(cvrefs); cv.setNumThreads(2); cv.ocl.setUseOpenCL(False)
    gray, ms = [], []
    for f, m in zip(inputs['frames'], masks['frames']):
        with Image.open(f['path']) as im:
            gray.append(np.asarray(im.convert('L')).copy())
        if m['current_observed']:
            with Image.open(m['mask_path']) as im: ms.append(np.asarray(im).copy())
        else: ms.append(None)
    start = time.monotonic()
    def budget():
        if time.monotonic() - start > 110:
            raise TimeoutError('texture trace CPU budget 110 s')
    result = trace(gray, ms, [f['decode_index'] for f in inputs['frames']],
        [f['time_seconds'] for f in inputs['frames']], inputs['seed']['xy'], cv, budget=budget)
    result.update(wall_seconds=time.monotonic()-start, opencv_version=cv.__version__,
        opencv_bindings=cvrefs, sam_calls=0, gpu_calls=0, physical_identity_verified=False,
        occlusion_confirmed=False, geometry_accepted=False)
    save(out / 'TRACE.json', result)
    check(out)
    print('TRACE_WRITTEN', result.get('counts'), flush=True)


def execute(out):
    check(out)
    if (out / 'processes/tracking').exists():
        raise FileExistsError('tracking already attempted; no automatic retry')
    state = read(out / 'RESULT.json')
    state.update(status='RUNNING', tracking_entries=1)
    save(out / 'RESULT.json', state)
    try:
        entry = invoke(out, 'tracking', [sys.executable, '-B', '-X', 'utf8',
            str(ROOT/'scripts/trace_texture_object_point.py'), '--output', str(out), '--worker'], 130, 3<<30)
        state['tracking_process'] = entry
        if entry.get('root_exit_code') != 0 or not entry.get('reclaim_confirmed'):
            raise RuntimeError('tracking process failed or not reclaimed')
        entry = invoke(out, 'report', [sys.executable, '-B', '-X', 'utf8',
            str(ROOT/'scripts/trace_texture_object_point.py'), '--output', str(out), '--report'], 120, 3<<30)
        state['report_process'] = entry
        if entry.get('root_exit_code') != 0 or not entry.get('reclaim_confirmed'):
            raise RuntimeError('report failed or not reclaimed')
        check(out)
        state.update(status='TRACKING_EXPERIMENT_COMPLETE_NOT_IDENTITY_VERIFIED',
            source_drift=[], geometry_accepted=False, physical_identity_verified=False,
            occlusion_confirmed=False)
    except Exception as exc:
        state.update(status='FAILED', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        save(out / 'RESULT.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', nargs=2, type=float)
    modes = p.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'execute', 'worker', 'report'):
        modes.add_argument('--'+name, action='store_true')
    a = p.parse_args(); out=a.output.resolve()
    if a.prepare:
        if a.source is None or a.seed is None: p.error('prepare requires --source and --seed X Y')
        prepare(a.source.resolve(), out, a.seed)
    elif a.execute: execute(out)
    elif a.worker: worker(out)
    else:
        from .texture_point_report import render
        check(out); render(out); check(out)
    return 0
