import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from integrations.demo_pipeline.segmentation_recovery import build_recovery_request, compose_recovery


def write(path, doc):
    path.write_text(json.dumps(doc), encoding='utf-8')


def batch(root, source, statuses, prompt='hand'):
    root.mkdir()
    frames = []
    for idx, status in statuses.items():
        candidates = []
        for ordinal in range(0 if status == 'NO_MATCH' else 2 if status == 'NEEDS_SELECTION' else 1):
            relative = f'masks/{idx}_{ordinal}.png'
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            Image.new('L', (2, 2), 255).save(path)
            candidates.append(dict(candidate_id=f'c{ordinal}', mask_path=relative,
                                   mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), foreground_pixels=4))
        frames.append(dict(input_index=idx, frame_id=f'f{idx}', source_image=str(source),
                           source_image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           photo_io_image_sha256='a' * 64, width=2, height=2, status=status, candidates=candidates,
                           selected_candidate_id='c0' if status == 'OK' else None,
                           selected_mask=dict(path=candidates[0]['mask_path'], sha256=candidates[0]['mask_sha256']) if status == 'OK' else None,
                           selection=dict(method='unique_candidate', source='automatic', mask_source='automatic_sam3') if status == 'OK' else None))
    counts = dict(requested=len(frames), success=list(statuses.values()).count('OK'),
                  no_match=list(statuses.values()).count('NO_MATCH'), needs_selection=list(statuses.values()).count('NEEDS_SELECTION'), error=0, not_run=0)
    doc = dict(contract_version='object-mask-v1', run_id=root.name, source_io=dict(path='same'),
               object=dict(object_id='object', text_prompt=prompt, submitted_options=dict(confidence_threshold=0.5)),
               backend=dict(name='sam3'), backend_bindings=dict(revision='pinned'), evidence_level='sam3_real_inference',
               frames=frames, counts=counts, status='SUCCESS' if counts['success'] == len(frames) else 'PARTIAL')
    path = root / 'manifest.json'
    write(path, doc)
    return path


def setup(tmp_path, statuses=None):
    source = tmp_path / 'image.png'
    Image.new('RGB', (2, 2)).save(source)
    original = batch(tmp_path / 'base', source, {0: 'OK', 1: 'NO_MATCH', 2: 'NEEDS_SELECTION'})
    attempt = batch(tmp_path / 'attempt', source, statuses or {1: 'OK', 2: 'NEEDS_SELECTION'}, 'the hand')
    return original, attempt


def test_recovery_preserves_history_and_ambiguity(tmp_path):
    original, attempt = setup(tmp_path)
    before = original.read_bytes()
    request = build_recovery_request(original, range(3))
    assert request['input_indices'] == [1, 2]
    result = compose_recovery(original, attempt, tmp_path / 'out', range(3), request)
    doc = json.loads(Path(result['manifest']).read_text())
    assert result['recovered'] == 1 and result['status'] == 'PARTIAL'
    assert [f['status'] for f in doc['frames']] == ['OK', 'OK', 'NEEDS_SELECTION']
    assert doc['frames'][2]['selected_mask'] is None
    assert doc['recovery']['selection_ledger'][1]['original_status'] == 'NO_MATCH'
    assert original.read_bytes() == before
    assert (tmp_path / 'out/sources/original/manifest.json').read_bytes() == before
    assert (tmp_path / 'out/sources/attempt/masks/2_1.png').is_file()
    with pytest.raises(ValueError, match='one attempt'):
        build_recovery_request(result['manifest'], range(3))


@pytest.mark.parametrize('tamper', ['missing', 'image_hash', 'frame_id', 'prompt', 'threshold', 'mask', 'foreground', 'ambiguous_ok', 'error', 'request', 'source_run', 'backend_binding'])
def test_fail_closed(tmp_path, tamper):
    original, attempt = setup(tmp_path)
    request = build_recovery_request(original, range(3))
    doc = json.loads(attempt.read_text())
    row = doc['frames'][0]
    if tamper == 'missing':
        doc['frames'].pop()
    elif tamper == 'image_hash':
        row['source_image_sha256'] = 'b' * 64
    elif tamper == 'frame_id':
        row['frame_id'] = 'other'
    elif tamper == 'prompt':
        doc['object']['text_prompt'] = 'something else'
    elif tamper == 'threshold':
        doc['object']['submitted_options']['confidence_threshold'] = 0.4
    elif tamper == 'mask':
        (attempt.parent / row['candidates'][0]['mask_path']).write_bytes(b'bad')
    elif tamper == 'foreground':
        row['candidates'][0]['foreground_pixels'] = 3
    elif tamper == 'ambiguous_ok':
        row['candidates'].append(dict(row['candidates'][0], candidate_id='other'))
    elif tamper == 'error':
        doc['status'] = 'ERROR'
    elif tamper == 'request':
        request['input_indices'] = [0, 1, 2]
    elif tamper == 'source_run':
        doc['source_io']['run_id'] = 'different-photo-io-run'
    elif tamper == 'backend_binding':
        doc['backend_bindings']['revision'] = 'different-code'
    write(attempt, doc)
    with pytest.raises(ValueError):
        compose_recovery(original, attempt, tmp_path / 'out', range(3), request)
    assert not (tmp_path / 'out').exists()
