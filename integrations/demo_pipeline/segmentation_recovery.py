"""One bounded text-prompt recovery composition; never executes inference."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import uuid

from PIL import Image

from .segmentation_batches import (_need, _read, _sha, _pairs, _float, _constant,
                                   _identity, _counts, _contained, merge_manifests)


def _load(path, expected, temporary):
    path = Path(path).resolve()
    raw = _read(path, 8 << 20)
    original = json.loads(raw, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_constant)
    _need(original.get('backend', {}).get('name') == 'sam3', 'real SAM3 backend required')
    _need(original.get('evidence_level') in ('sam3_real_inference', 'AGGREGATE_OF_SAM_BATCHES'), 'real SAM3 evidence required')
    result = merge_manifests([path], temporary, expected)
    doc = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
    for f in doc['frames']:
        image = Path(f.get('source_image', ''))
        _need(image.is_absolute() and image.is_file(), 'missing canonical source image')
        _need(_sha(_read(image, 32 << 20)) == f.get('source_image_sha256'), 'source image hash mismatch')
        _need(isinstance(f.get('photo_io_image_sha256'), str), 'missing photo IO hash')
        for c in f['candidates']:
            data = (Path(temporary) / c['mask_path']).read_bytes()
            with Image.open(io.BytesIO(data)) as mask:
                _need(mask.format == 'PNG' and mask.mode == 'L' and mask.size == (f['width'], f['height']), 'invalid canonical mask')
                pixels = mask.tobytes()
            _need(set(pixels) <= {0, 255}, 'mask must be binary')
            foreground = pixels.count(255)
            _need(type(c.get('foreground_pixels')) is int and foreground == c['foreground_pixels'], 'foreground count mismatch')
        if f['status'] == 'OK':
            chosen = next(c for c in f['candidates'] if c['candidate_id'] == f['selected_candidate_id'])
            _need(chosen['foreground_pixels'] > 0 and isinstance(f.get('selection'), dict), 'invalid selected foreground')
    return original, doc, dict(manifest=str(path), manifest_sha256=_sha(raw), run_id=original.get('run_id'))


def _request(original, origin, expected, recovery_prompt):
    _need(not original.get('recovery'), 'recovery is bounded to one attempt')
    prompt = original.get('object', {}).get('text_prompt')
    _need(isinstance(prompt, str) and prompt.strip(), 'text prompt required for recovery')
    recovery_prompt = recovery_prompt if recovery_prompt is not None else 'the ' + prompt.strip()
    _need(isinstance(recovery_prompt, str) and 0 < len(recovery_prompt.strip()) <= 512, 'invalid recovery prompt')
    indices = [f['input_index'] for f in original['frames'] if f['status'] in ('NO_MATCH', 'NEEDS_SELECTION')]
    _need(len(indices) <= 96, 'recovery exceeds one worker batch')
    return dict(contract_version='segmentation-recovery-request-v1', original=origin,
                expected_indices=list(expected), input_indices=sorted(indices),
                object_id=original['object']['object_id'], original_text_prompt=prompt,
                text_prompt=recovery_prompt.strip(), max_attempts=1, confidence_threshold=0.5)


def build_recovery_request(existing_manifest, expected_indices, recovery_prompt=None):
    """Validate source bytes and select only unresolved inputs for one SAM call."""
    expected = list(expected_indices)
    with tempfile.TemporaryDirectory(prefix='sam_recovery_validate_') as temp:
        original, _, origin = _load(existing_manifest, expected, Path(temp) / 'original')
        return _request(original, origin, expected, recovery_prompt)


def compose_recovery(existing_manifest, recovery_manifest, output, expected_indices, recovery_request=None):
    """Publish a derived manifest only after both completed runs validate.

    Partial runs retain unresolved original evidence. Error/incomplete runs raise
    and publish nothing. Existing OK selections are immutable.
    """
    output = Path(output).resolve()
    _need(not output.exists(), 'output must be a new directory')
    expected = list(expected_indices)
    with tempfile.TemporaryDirectory(prefix='sam_recovery_compose_') as temp:
        temp = Path(temp)
        original, base, origin = _load(existing_manifest, expected, temp / 'original')
        request = recovery_request or _request(original, origin, expected, None)
        _need(request == _request(original, origin, expected, request.get('text_prompt')), 'recovery request binding mismatch')
        _need(request['input_indices'], 'no unresolved frames to recover')
        attempt, recovered, recovery_origin = _load(recovery_manifest, request['input_indices'], temp / 'attempt')
        for key in ('source_io', 'backend', 'backend_bindings'):
            _need(_identity(original, key) == _identity(attempt, key), 'recovery identity mismatch: ' + key)
        wanted_object = deepcopy(original['object'])
        wanted_object['text_prompt'] = request['text_prompt']
        _need(attempt['object'] == wanted_object, 'recovery object/prompt/options mismatch')
        _need(wanted_object.get('submitted_options', {}).get('confidence_threshold') == 0.5, 'recovery threshold must remain 0.5')
        by_index = {f['input_index']: f for f in recovered['frames']}
        frames, ledger, copies = [], [], []
        for base_frame in base['frames']:
            idx = base_frame['input_index']
            r = by_index.get(idx)
            if r:
                for key in ('frame_id', 'source_image', 'source_image_sha256', 'photo_io_image_sha256', 'width', 'height', 'image_domain'):
                    _need(base_frame.get(key) == r.get(key), 'recovery frame identity mismatch: ' + key)
            promote = r is not None and r['status'] == 'OK'
            if promote:
                _need(len(r['candidates']) == 1 and r.get('selection', {}).get('method') == 'unique_candidate'
                      and r['selection'].get('source') == 'automatic'
                      and r['selection'].get('mask_source') == 'automatic_sam3', 'recovery requires automatic unique candidate')
            f = deepcopy(r if promote else base_frame)
            chosen_root = temp / ('attempt' if promote else 'original')
            for ordinal, c in enumerate(f['candidates']):
                data = (chosen_root / c['mask_path']).read_bytes()
                dest = f'masks/{idx:06d}_{ordinal:03d}.png'
                c['mask_path'] = dest
                copies.append((dest, data))
                if c['candidate_id'] == f.get('selected_candidate_id'):
                    f['selected_mask']['path'] = dest
            f['origin'] = dict(recovery_origin if promote else origin, input_index=idx)
            ledger.append(dict(input_index=idx, original_status=base_frame['status'],
                               recovery_status=r['status'] if r else 'NOT_REQUESTED',
                               decision='RECOVERY_UNIQUE_OK' if promote else 'PRESERVE_ORIGINAL',
                               selected_candidate_id=f.get('selected_candidate_id'),
                               selected_mask_sha256=(f.get('selected_mask') or {}).get('sha256')))
            frames.append(f)
        result = deepcopy(original)
        counts = _counts(frames)
        status = 'SUCCESS' if counts['success'] == len(frames) else 'PARTIAL'
        result.update(run_id='recovery_composite_' + uuid.uuid4().hex, status=status,
                      frames=frames, counts=counts, quality_status='NOT_EVALUATED', cross_view_status='NOT_EVALUATED',
                      evidence_level='AGGREGATE_OF_SAM_BATCHES', execution_mode='recovery_composition_no_model_execution',
                      recovery=dict(request=request, source_manifests=[origin, recovery_origin], selection_ledger=ledger,
                                    sam_execution=False, attempts=1))
        # Preserve both original manifests byte-for-byte; never relabel old evidence.
        for label, source, source_doc, binding in (('original', existing_manifest, original, origin), ('attempt', recovery_manifest, attempt, recovery_origin)):
            raw = _read(Path(source), 8 << 20)
            _need(_sha(raw) == binding['manifest_sha256'], 'source manifest changed during composition')
            copies.append((f'sources/{label}/manifest.json', raw))
            source_root = Path(source).resolve().parent
            for row in source_doc['frames']:
                for candidate in row['candidates']:
                    relative = candidate['mask_path'].replace('\\', '/')
                    data = _read(_contained(source_root, relative), 16 << 20)
                    _need(_sha(data) == candidate['mask_sha256'], 'source mask changed during composition')
                    copies.append((f'sources/{label}/{relative}', data))
        _need(sum(len(data) for _, data in copies) <= 128 << 20, 'recovery copies exceed byte limit')
        output.mkdir(parents=True, exist_ok=False)
        for relative, data in copies:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(data)
            _need(_sha(target.read_bytes()) == _sha(data), 'copy hash drift')
        target = output / 'manifest.json'
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return dict(status=status, exit_code=0 if status == 'SUCCESS' else 1, manifest=str(target),
                    run_dir=str(output), counts=counts, recovered=sum(r['decision'] == 'RECOVERY_UNIQUE_OK' for r in ledger))
