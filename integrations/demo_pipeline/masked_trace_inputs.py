"""Identity-bound actual SAM masks for a frozen raw-frame point trace.

Preparation and collection only: this module never launches a subprocess.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def _binding(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def _check(ref):
    actual = _binding(ref['path'])
    if any(actual[k] != ref[k] for k in ('bytes', 'sha256')):
        raise ValueError('source binding drift: ' + str(ref['path']))
    return actual


def _inside(root, relative):
    p = (Path(root) / relative).resolve(strict=True)
    if not p.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError('path escapes bound root: ' + str(p))
    return p


def _rgb(path):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB')
        width, height = image.size
        digest = hashlib.sha256(f'{width}x{height}:RGB:'.encode('ascii') + image.tobytes()).hexdigest()
    return width, height, digest


def _rows(path):
    result = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({r['input_index'] for r in result}) != len(result):
        raise ValueError('duplicate IO input index')
    return {r['input_index']: r for r in result}


def _sam_frame(manifest_path, manifest, frame, raw, io_run):
    """Validate source IO and canonical pixels even for an unselected result."""
    io_run = Path(io_run).resolve(strict=True)
    source = manifest['source_io']
    if Path(source['path']).resolve(strict=True) != io_run:
        raise ValueError('SAM source IO path mismatch')
    if _read(io_run / 'run.json')['run_id'] != source['run_id']:
        raise ValueError('SAM source IO run ID mismatch')
    for name, key in (('run.json', 'run_json_sha256'), ('frames.jsonl', 'frames_jsonl_sha256')):
        if _binding(io_run / name)['sha256'] != source[key]:
            raise ValueError('SAM source IO digest mismatch: ' + name)
    io = _rows(io_run / 'frames.jsonl')[frame['input_index']]
    if io['status'] != 'OK' or io['frame_id'] != frame['frame_id']:
        raise ValueError('SAM frame identity/status differs from IO')
    image = _inside(io_run, io['image_output'])
    if Path(frame['source_image']).resolve(strict=True) != image:
        raise ValueError('SAM source image path mismatch')
    if _binding(image)['sha256'] != frame['source_image_sha256']:
        raise ValueError('SAM source image digest mismatch')
    identity = _rgb(raw['path'])
    if _rgb(image) != identity or identity != (frame['width'], frame['height'], raw['decoded_rgb_sha256']):
        raise ValueError('canonical/raw RGB identity mismatch')
    # photo-io image_sha256 identifies the original input bytes, rather than its re-encoded PNG.
    if frame['photo_io_image_sha256'] != io['image_sha256'] or io['image_sha256'] != raw['sha256']:
        raise ValueError('SAM photo IO original byte identity mismatch')
    result = dict(status=frame['status'], current_observed=False, source_image=str(image),
                  source_image_sha256=frame['source_image_sha256'], frame_id=frame['frame_id'],
                  source_io=str(io_run), manifest=str(Path(manifest_path).resolve()),
                  input_index=frame['input_index'], mask_path=None, mask_sha256=None)
    selected = frame.get('selected_mask')
    if frame['status'] != 'OK':
        if selected:
            raise ValueError('non-OK SAM frame has selected mask')
        return result
    if not selected or not frame.get('selected_candidate_id'):
        raise ValueError('OK SAM frame missing selected candidate')
    candidates = [c for c in frame['candidates'] if c['candidate_id'] == frame['selected_candidate_id']]
    if len(candidates) != 1 or candidates[0]['mask_path'] != selected['path'] or candidates[0]['mask_sha256'] != selected['sha256']:
        raise ValueError('selected candidate binding mismatch')
    mask = _inside(Path(manifest_path).parent, selected['path'])
    if _binding(mask)['sha256'] != selected['sha256']:
        raise ValueError('selected mask digest mismatch')
    from PIL import Image
    with Image.open(mask) as im:
        if im.mode != 'L' or im.size != identity[:2] or not set(im.getdata()).issubset({0, 255}):
            raise ValueError('mask must be same-size single-channel 0/255')
        if im.getbbox() is None:
            raise ValueError('selected mask is empty')
    result.update(current_observed=True, mask_path=str(mask), mask_sha256=selected['sha256'],
                  mask_semantics='ACTUAL_SAM_SELECTED_NOT_HUMAN_GROUND_TRUTH')
    return result


def prepare(old_single_run, output):
    old = Path(old_single_run).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs, plan, result = (_read(old / n) for n in ('INPUTS.json', 'PLAN.json', 'RESULT.json'))
    if (result.get('status') != 'DIAGNOSTIC_COMPLETE_NOT_IDENTITY_VERIFIED'
            or result.get('implementation_drift') or result.get('source_drift')):
        raise ValueError('completed frozen single-point diagnostic required')
    refs = [_binding(old / n) for n in ('INPUTS.json', 'PLAN.json', 'RESULT.json')]
    for ref in inputs['bindings'] + plan['implementation']:
        refs.append(_check(ref))
    source = Path(inputs['source_run']).resolve(strict=True)
    photometric = _read(source / 'INPUTS.json')
    refs.extend([_binding(source / 'INPUTS.json'), _binding(source / 'PLAN.json')])
    for ref in _read(source / 'PLAN.json')['implementation']:
        refs.append(_check(ref))
    sampled = {f['decode_index']: f for f in photometric['frames']}
    raw_frames = inputs['frames']
    if len(raw_frames) != 64 or [f['decode_index'] for f in raw_frames] != list(range(555, 619)):
        raise ValueError('expected frozen raw555..618 window')
    mask_roots = {Path(f['mask_path']).resolve(strict=True).parents[1] for f in sampled.values() if f.get('mask_path')}
    if len(mask_roots) != 1:
        raise ValueError('one unambiguous original SAM manifest root required')
    old_manifest = next(iter(mask_roots)) / 'manifest.json'
    sam = _read(old_manifest)
    samrows = {f['input_index']: f for f in sam['frames']}
    if len(samrows) != len(sam['frames']):
        raise ValueError('duplicate original SAM index')
    refs.append(_binding(old_manifest))
    dataset_id = 'masked_trace_' + uuid.uuid4().hex
    dataset = ROOT / 'data' / 'real' / dataset_id
    if dataset.exists():
        raise FileExistsError(dataset)
    mapping, missing = [], []
    for f in raw_frames:
        ref = _check(f)
        refs.append(ref)
        width, height, digest = _rgb(f['path'])
        row = dict(f, width=width, height=height, decoded_rgb_sha256=digest,
                   origin='UNSAMPLED', new_input_index=None)
        if f['decode_index'] in sampled:
            sample = sampled[f['decode_index']]
            frame = samrows[sample['input_index']]
            if sample['frame_id'] != frame['frame_id'] or sample['original_mask_status'] != frame['status']:
                raise ValueError('sampled/SAM frame binding mismatch')
            observed = _sam_frame(old_manifest, sam, frame, row, sam['source_io']['path'])
            row.update(origin='OLD_SAM', old_input_index=sample['input_index'], old_observation=observed)
            refs.append(_binding(observed['source_image']))
            if observed['current_observed']:
                refs.append(_binding(observed['mask_path']))
        else:
            row['new_input_index'] = len(missing)
            missing.append(row)
        mapping.append(row)
    if len(missing) != 54 or sum(r.get('old_observation', {}).get('current_observed', False) for r in mapping) != 9:
        raise ValueError('expected 54 unsampled frames and nine reusable SAM masks')
    no_match = [r['decode_index'] for r in mapping if r.get('old_observation', {}).get('status') == 'NO_MATCH']
    if no_match != [576]:
        raise ValueError('expected preserved raw576 NO_MATCH')
    refs.extend([_binding(Path(sam['source_io']['path']) / n) for n in ('run.json', 'frames.jsonl')])
    refs.append(_binding(__file__))
    output.mkdir(parents=True, exist_ok=False)
    (dataset / 'images').mkdir(parents=True, exist_ok=False)
    photo_frames = []
    for row in missing:
        name = f"images/frame_{row['decode_index']:06d}.png"
        destination = dataset / name
        shutil.copyfile(row['path'], destination)
        copied = _binding(destination)
        if (copied['bytes'], copied['sha256']) != (row['bytes'], row['sha256']):
            raise ValueError('copied raw bytes differ')
        row['dataset_image'] = str(destination)
        refs.append(copied)
        photo_frames.append(dict(image=name, mask=None))
    manifest = output / 'photo_manifest.json'
    _save(manifest, dict(contract_version='photo-io-v1', dataset_id=dataset_id, data_kind='real',
                         dataset_root=f'data/real/{dataset_id}', frames=photo_frames))
    _save(output / 'FRAME_MAPPING.json', dict(schema='masked-trace-frame-mapping-v1', frames=mapping,
          old_single_run=str(old), old_manifest=str(old_manifest), dataset_root=str(dataset)))
    refs.extend([_binding(manifest), _binding(output / 'FRAME_MAPPING.json')])
    unique = {r['path']: r for r in refs}
    _save(output / 'SOURCE_BINDINGS.json', list(unique.values()))
    prior = _read(Path(photometric['video_root']) / 'photo_io.process.json')
    io_output = output / 'photo_io'
    return dict(output=str(output), dataset_root=str(dataset), photo_manifest=str(manifest),
                io_output=str(io_output), mapping_path=str(output / 'FRAME_MAPPING.json'),
                source_bindings=str(output / 'SOURCE_BINDINGS.json'),
                io_argv=[prior['argv'][0], '-B', '-m', 'kozlov_lab.io', '--manifest', str(manifest),
                         '--output', str(io_output), '--data-kind', 'real'],
                io_env=dict(prior['env_overrides']), indices=list(range(len(missing))))


def collect(output, io_run=None, manifest=None):
    output = Path(output).resolve(strict=True)
    for ref in _read(output / 'SOURCE_BINDINGS.json'):
        _check(ref)
    frozen = _read(output / 'FRAME_MAPPING.json')
    if manifest is None:
        candidates = list((output / 'segmentation' / 'product_run').glob('*/manifest.json'))
        if len(candidates) != 1:
            raise ValueError('pass explicit new SAM manifest; expected exactly one')
        manifest = candidates[0]
    manifest = Path(manifest).resolve(strict=True)
    sam = _read(manifest)
    io_run = Path(io_run or sam['source_io']['path']).resolve(strict=True)
    if not io_run.is_relative_to(output) or not manifest.is_relative_to(output):
        raise ValueError('new IO and SAM results must be within new output')
    run = _read(io_run / 'run.json')
    if (run.get('status') != 'SUCCESS' or run.get('data_kind') != 'real'
            or run.get('manifest_sha256') != _binding(output / 'photo_manifest.json')['sha256']):
        raise ValueError('successful new photo IO run required')
    rows = {r['input_index']: r for r in sam['frames']}
    expected = {r['new_input_index'] for r in frozen['frames'] if r['new_input_index'] is not None}
    if len(rows) != len(sam['frames']) or set(rows) != expected:
        raise ValueError('new SAM index population mismatch')
    io_rows = _rows(io_run / 'frames.jsonl')
    if set(io_rows) != expected:
        raise ValueError('new photo IO population mismatch')
    observed = []
    for raw in frozen['frames']:
        index = raw['new_input_index']
        if index is None:
            old_path = Path(frozen['old_manifest'])
            old_sam = _read(old_path)
            frame = next(r for r in old_sam['frames'] if r['input_index'] == raw['old_input_index'])
            row = _sam_frame(old_path, old_sam, frame, raw, old_sam['source_io']['path'])
        else:
            io = io_rows[index]
            if _inside(frozen['dataset_root'], io['input_image']) != Path(raw['dataset_image']).resolve(strict=True):
                raise ValueError('new IO input-to-raw mapping mismatch')
            row = _sam_frame(manifest, sam, rows[index], raw, io_run)
        row.update(decode_index=raw['decode_index'], raw_path=raw['path'], raw_sha256=raw['sha256'],
                   decoded_rgb_sha256=raw['decoded_rgb_sha256'], width=raw['width'], height=raw['height'],
                   origin=raw['origin'] if index is None else 'NEW_SAM')
        observed.append(row)
    document = dict(schema='masked-trace-masks-v1', frames=observed,
                    current_observed_count=sum(r['current_observed'] for r in observed),
                    unknown_count=sum(not r['current_observed'] for r in observed),
                    semantics='Selected actual SAM masks are observations, not human ground truth; unknown masks remain absent.',
                    new_manifest_binding=_binding(manifest),
                    source_bindings_binding=_binding(output / 'SOURCE_BINDINGS.json'))
    if (output / 'MASKS.json').exists():
        raise FileExistsError(output / 'MASKS.json')
    _save(output / 'MASKS.json', document)
    return document
