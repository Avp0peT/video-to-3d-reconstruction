"""Adopt committed temporal union masks without inference or geometry execution.

Only the original sampled T01 rows are adopted. JPEG tracking frames and the
canonical PNGs share a bound decode index/PTS/grid; pixel equality is not claimed.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from PIL import Image

SOURCE = 'video_temporal_union'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    from integrations.mask_review.review import strict_json
    return strict_json(Path(path).read_bytes())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def index(rows, key):
    require(isinstance(rows, list), 'rows must be a list')
    result = {}
    for row in rows:
        value = row.get(key)
        require(type(value) is int and value >= 0 and value not in result, 'duplicate or invalid '+key)
        result[value] = row
    return result


def mask_pixels(path, size):
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.mode == 'L' and image.size == size,
                'mask format/grid mismatch')
        require(image.getexif().get(274, 1) == 1, 'mask orientation mismatch')
        counts = image.histogram()
        require(not any(counts[1:255]) and counts[255] > 0, 'mask must be nonempty binary 0/255')
        return counts[255], hashlib.sha256(image.tobytes()).hexdigest()


def _collect(source_run):
    """Rebind producer artifacts and exact original sampling before adoption."""
    from .planning import _validated_evidence
    source = Path(source_run).resolve(strict=True)
    plan = read(source/'PLAN.json')
    require(plan.get('schema') == 'video-mask-full-plan-v1', 'wrong full-video plan')
    ordered, original_bindings = _validated_evidence(Path(plan['pipeline_plan_path']).parent)
    require(ordered == plan['coverage_ledger'], 'coverage ledger differs from original sampled timeline')
    require(all(plan.get(k) == v for k, v in original_bindings.items()), 'full-video source binding differs')
    require(2 <= len(ordered) <= 96, 'adoption requires original 2-96 sampled inputs')
    result = read(source/'RESULT.json')
    require(result.get('status') == 'FULL_TIMELINE_FUSED_EXPORTED' and result.get('mode') == 'union'
            and result.get('model_entries') == 0 and not result.get('input_drift') and not result.get('code_drift'),
            'union producer is not a committed unchanged export')
    manifest = read(source/'MANIFEST.json')
    fusion = read(source/'FUSION.json')
    require(manifest.get('schema') == 'full-video-mask-manifest-v1'
            and fusion.get('summary', {}).get('mode') == 'union', 'wrong fusion manifest or mode')
    full = index(manifest['rows'], 'decode_index')
    fused = index(fusion['rows'], 'decode_index')
    require(set(full) == set(fused) == set(plan['expected_decode_indexes'])
            and len(full) == plan['frame_count'], 'full timeline coverage mismatch')
    io = Path(plan['io_run_path']).parent.resolve(strict=True)
    io_doc = read(io/'run.json')
    io_rows = index([json.loads(line) for line in (io/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line], 'input_index')
    old = read(Path(plan['source_manifest_path']))
    old_rows = index(old['frames'], 'input_index')
    require(set(io_rows) == set(old_rows) == {r['input_index'] for r in ordered}, 'sample coverage differs')
    bindings = {str(source/name): sha(source/name) for name in
                ('PLAN.json', 'RESULT.json', 'MANIFEST.json', 'FUSION.json')}
    start = read(source/'START_BINDINGS.json')
    decode_path = Path(start['source_run'])/'DECODE.json'
    require(sha(decode_path) == start['input_files'].get(str(decode_path)), 'tracking decode ledger hash differs')
    decoded = index(read(decode_path)['frames'], 'decode_index')
    bindings[str(source/'START_BINDINGS.json')] = sha(source/'START_BINDINGS.json')
    bindings[str(decode_path)] = sha(decode_path)
    for key, value in original_bindings.items():
        if key.endswith('_path'):
            bindings[value] = original_bindings[key[:-5]+'_sha256']
    bindings[str(io/'frames.jsonl')] = sha(io/'frames.jsonl')
    selected = []
    for ledger in ordered:
        n, decode = ledger['input_index'], ledger['decode_index']
        row, native = full[decode], io_rows[n]
        require(row.get('placeholder') is False, 'placeholder mask cannot enter reconstruction')
        require(row['pts'] == ledger['pts'] and row['time_base'] == ledger['time_base']
                and row['input_index'] == n and row['original_status'] == ledger['original_status'],
                'same-frame PTS or sampled identity mismatch')
        require(row.get('fusion') == fused[decode] and fused[decode]['mode'] == 'union'
                and fused[decode]['removed_pixels'] == 0, 'fusion row binding mismatch')
        require(native['raw_to_canonical'] == [[1,0,0],[0,1,0],[0,0,1]]
                and native['orientation_applied'] == 1, 'nonidentity canonical transform is unsupported')
        size = (ledger['width'], ledger['height'])
        require(size == (plan['width'], plan['height']), 'canonical and tracking grid differ')
        image = Path(ledger['source_image']).resolve(strict=True)
        require(image == (io/native['image_output']).resolve(strict=True)
                and sha(image) == ledger['source_image_sha256'], 'canonical image hash differs')
        with Image.open(image) as pixels:
            require(pixels.mode == 'RGB' and pixels.size == size, 'canonical RGB grid differs')
        tracking = source/'frames'/f'{decode:06d}.jpg'
        require(sha(tracking) == decoded[decode]['sha256'], 'tracking JPEG decode hash differs')
        with Image.open(tracking) as pixels:
            require(pixels.size == size, 'tracking JPEG grid differs')
        path = Path(row['path']).resolve(strict=True)
        require(path.is_relative_to(source/'final_masks') and sha(path) == row['sha256'] == row['mask_sha256'],
                'final union mask hash/path mismatch')
        require(sha(Path(row['mask_path'])) == row['mask_sha256'], 'fusion source mask hash differs')
        count, pixel_hash = mask_pixels(path, size)
        require(count == row['foreground_pixels'] and pixel_hash == row['pixel_sha256'], 'union raster readback differs')
        bindings[str(path)] = row['sha256']
        bindings[str(image)] = ledger['source_image_sha256']
        bindings[str(tracking)] = sha(tracking)
        bindings[str(Path(row['mask_path']))] = row['mask_sha256']
        selected.append(dict(input_index=n, frame_id=native['frame_id'], decode_index=decode,
            pts=ledger['pts'], time_base=ledger['time_base'], time_seconds=ledger['time_seconds'],
            original_status=old_rows[n]['status'], source_image=str(image),
            source_image_sha256=ledger['source_image_sha256'], photo_io_image_sha256=native['image_sha256'],
            width=size[0], height=size[1], source_mask=str(path), mask_sha256=row['sha256'],
            foreground_pixels=count, pixel_sha256=pixel_hash,
            tracking_image=str(tracking), tracking_image_sha256=bindings[str(tracking)]))
    return source, io, io_doc, old, selected, bindings


def validate_derivation(document, manifest_root):
    """Fail closed on missing/drifted union lineage; called by every consumer."""
    root = Path(manifest_root).resolve(strict=True)
    derivation = document.get('derivation', {})
    require(derivation.get('schema') == 'video-temporal-union-adoption-v1'
            and document.get('backend', {}).get('name') == SOURCE
            and document['backend'].get('model_executed') is False, 'invalid derived union backend')
    source, io, io_doc, old, selected, bindings = _collect(derivation['source_run'])
    require(document.get('object') == old['object']
            and document['backend'].get('source_backend') == old['backend']
            and derivation.get('historical_counts') == old['counts'],
            'derived union object or historical producer differs')
    require(derivation.get('bindings') == bindings and derivation.get('rows') == selected,
            'derived union lineage or source hash drift')
    require(document.get('source_io') == dict(path=str(io), run_id=io_doc['run_id'],
            run_json_sha256=sha(io/'run.json'), frames_jsonl_sha256=sha(io/'frames.jsonl')),
            'derived union IO binding differs')
    frames = index(document.get('frames'), 'input_index')
    require(set(frames) == {r['input_index'] for r in selected}, 'derived union sampled rows differ')
    for row in selected:
        frame = frames[row['input_index']]
        for key in ('frame_id','source_image','source_image_sha256','photo_io_image_sha256','width','height'):
            require(frame.get(key) == row[key], 'derived union frame identity differs: '+key)
        relative = f"masks/{row['input_index']:06d}.png"
        require(frame.get('status') == 'OK' and frame.get('historical_status') == row['original_status']
                and frame.get('image_domain') == 'photo_io_canonical'
                and frame.get('selection') == dict(method='explicit_candidate', source='bound_video_union_adoption', mask_source=SOURCE)
                and frame.get('selected_candidate_id') == 'temporal_union'
                and frame.get('selected_mask') == dict(path=relative, sha256=row['mask_sha256']),
                'derived union selection differs')
        require(frame.get('candidates') == [dict(candidate_id='temporal_union', mask_path=relative,
                mask_sha256=row['mask_sha256'], foreground_pixels=row['foreground_pixels'],
                score=None, score_semantics='not_a_probability_temporal_union')], 'derived union candidate differs')
        require(sha(root/relative) == row['mask_sha256'], 'adopted union mask hash drift')
        mask_pixels(root/relative, (row['width'], row['height']))
    return dict(status='BOUND', frames=len(selected), source_run=str(source), io_run=str(io))


def prepare(source_run: Path, output: Path) -> dict:
    source, io, io_doc, old, rows, bindings = _collect(source_run)
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'output must be new; no overwrite')
    resolved = output.resolve()
    require(not resolved.is_relative_to(source) and not resolved.is_relative_to(io)
            and not source.is_relative_to(resolved) and not io.is_relative_to(resolved), 'output overlaps immutable input')
    require(not any(p.is_symlink() for p in (output, *output.parents)), 'linked output forbidden')
    output.mkdir(parents=True)
    (output/'masks').mkdir()
    now = datetime.now(timezone.utc).isoformat()
    document = dict(contract_version='object-mask-v1', run_id='video_union_'+uuid.uuid4().hex,
        started_utc=now, finished_utc=now, status='SUCCESS',
        source_io=dict(path=str(io), run_id=io_doc['run_id'], run_json_sha256=sha(io/'run.json'),
                       frames_jsonl_sha256=sha(io/'frames.jsonl')),
        backend=dict(name=SOURCE, model_executed=False, source_backend=old['backend']),
        object=copy.deepcopy(old['object']), quality_status='NOT_EVALUATED', cross_view_status='NOT_EVALUATED',
        evidence_level='DERIVED_TEMPORAL_UNION_NOT_NEW_INFERENCE', frames=[],
        derivation=dict(schema='video-temporal-union-adoption-v1', source_run=str(source), bindings=bindings, rows=rows,
            coordinate_mapping='bound_decode_index_pts_identity_grid; JPEG pixels are not canonical PNG pixels',
            historical_counts=old['counts']))
    for row in rows:
        relative = f"masks/{row['input_index']:06d}.png"
        shutil.copyfile(row['source_mask'], output/relative)
        document['frames'].append(dict(input_index=row['input_index'], frame_id=row['frame_id'],
            source_image=row['source_image'], source_image_sha256=row['source_image_sha256'],
            photo_io_image_sha256=row['photo_io_image_sha256'], image_domain='photo_io_canonical',
            width=row['width'], height=row['height'], status='OK', historical_status=row['original_status'],
            candidates=[dict(candidate_id='temporal_union',mask_path=relative,mask_sha256=row['mask_sha256'],
                foreground_pixels=row['foreground_pixels'],score=None,score_semantics='not_a_probability_temporal_union')],
            selected_candidate_id='temporal_union',selected_mask=dict(path=relative,sha256=row['mask_sha256']),
            selection=dict(method='explicit_candidate',source='bound_video_union_adoption',mask_source=SOURCE),
            errors=[],not_run_reason=None))
    document['counts'] = dict(requested=len(rows),success=len(rows),no_match=0,needs_selection=0,error=0,not_run=0)
    save(output/'manifest.json', document)
    validate_derivation(document, output)
    from integrations.mvs_masks.manifest_io import parse_object_manifest
    parse_object_manifest(output/'manifest.json')
    from integrations.mask_review.review import load_inputs, Bindings
    reviewed, _, _ = load_inputs([output/'manifest.json'], Bindings())
    require(len(reviewed) == len(rows), 'consumer readback count differs')
    # Locate the original dataset declaration without manufacturing new inputs.
    input_result = read(Path(read(source/'PLAN.json')['pipeline_plan_path']).parent/'input/RESULT.json')
    dataset_record = input_result['photo_io']['manifest']
    dataset_manifest = Path(dataset_record['path']).resolve(strict=True)
    require(sha(dataset_manifest) == dataset_record['sha256'], 'dataset manifest hash differs')
    result = dict(status='PREPARED',manifest_path=str(output/'manifest.json'),io_run=str(io),
        dataset_manifest=str(dataset_manifest) if dataset_manifest else None,frames=len(rows),
        historical_counts=dict(Counter(r['original_status'] for r in rows)),model_entries=0,
        quality_status='NOT_EVALUATED',source_masks_modified=False,consumer_readback='BOUND')
    save(output/'RESULT.json', result)
    return result
