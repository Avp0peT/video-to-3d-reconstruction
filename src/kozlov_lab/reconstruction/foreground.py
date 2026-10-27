"""Strict mask provenance selection before sparse reconstruction; no inference."""
from __future__ import annotations

import json
from pathlib import Path

from .colmap_adapter import UpstreamRun, load_upstream_run


def select_foreground(upstream: UpstreamRun, manifest_path: Path, output_dir: Path, *, profile: str = "cad-object-v1") -> dict:
    """Retain the complete upstream ledger and select only producer-selected OK masks.

    Reuses the existing mask-review contract validator, including PNG encoding,
    candidate identity, source hashes, counters and failure-state consistency.
    A selected mask is provenance-verified, not a semantic quality verdict.
    """
    from integrations.mask_review import review

    root = upstream.run_dir.resolve(strict=True)
    manifest = Path(manifest_path).resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('foreground ledger output must be new; no overwrite')
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(root) or resolved_output.is_relative_to(manifest.parent):
        raise ValueError('foreground output cannot be inside immutable inputs')
    for ancestor in (output, *output.parents):
        if ancestor.is_symlink():
            raise ValueError('linked foreground output forbidden')
    bindings = review.Bindings()
    raw = bindings.read(manifest, review.MAX_JSON)
    document = review.strict_json(raw)
    # Reject numeric overflow too, not only named nonfinite JSON constants.
    json.dumps(document, allow_nan=False)
    review.require(isinstance(document, dict), 'foreground manifest root must be an object')
    source = document.get('source_io', {})
    review.require(isinstance(source, dict), 'source_io must be an object')
    review.require(isinstance(document.get('backend'), dict), 'backend must be an object')
    review.require(upstream.contract_version == 'photo-io-v1' and upstream.status == 'SUCCESS',
                   'foreground needs successful canonical upstream')
    review.require(Path(source.get('path', '')).resolve() == root
                   and source.get('run_id') == upstream.run_id
                   and source.get('run_json_sha256') == upstream.run_json_sha256
                   and source.get('frames_jsonl_sha256') == upstream.frames_jsonl_sha256,
                   'manifest does not bind the exact upstream run')
    if document.get('backend', {}).get('name') == 'video_temporal_union':
        from integrations.video_masks.reconstruction_input import validate_derivation
        validate_derivation(document, manifest.parent)
    else:
        review.require(document.get('backend', {}).get('name') == 'sam3', 'SAM3 producer required')
    from . import profiles
    profiles.require_valid_profile(profile)
    limits = "whole-video-192-v1" if profile == profiles.PROFILE_CAD_OBJECT_VIDEO_V1 else "default"
    frames, identity, _ = review.load_inputs([manifest], bindings, limits_profile=limits)
    rebound = load_upstream_run(root)
    review.require(rebound.run_id == upstream.run_id
                   and rebound.run_json_sha256 == upstream.run_json_sha256
                   and rebound.frames_jsonl_sha256 == upstream.frames_jsonl_sha256,
                   'upstream source binding drift')
    keys = ('input_index', 'frame_id', 'image_output', 'upstream_image_sha256',
            'canonical_width', 'canonical_height')
    signature = lambda rows: sorted(tuple(getattr(row, key) for key in keys) for row in rows)
    review.require(signature(rebound.rows) == signature(upstream.rows), 'upstream in-memory row drift')
    by_index = {row.input_index: row for row in upstream.rows}
    review.require(len(by_index) == len(upstream.rows), 'duplicate upstream index')
    requested = {row['input_index']: row for row in frames}
    review.require(set(requested).issubset(by_index), 'requested index outside upstream')
    # Bind every canonical image, including rows excluded or not requested.
    for row in upstream.rows:
        path = review.contained(root, row.image_output)
        imagebytes = bindings.read(path)
        review.decode_png(imagebytes, 'RGB', row.canonical_width, row.canonical_height)
        if row.canonical_image_sha256 is not None:
            review.require(review.digest(imagebytes) == row.canonical_image_sha256,
                           'upstream canonical hash drift')
    masks = {}; ledger = []
    object_id = identity['object_id']
    for frame in document['frames']:
        if 'object_id' in frame:
            review.require(frame['object_id'] == object_id, 'frame object identity mismatch')
        for candidate in frame.get('candidates', []):
            if 'object_id' in candidate:
                review.require(candidate['object_id'] == object_id, 'candidate object identity mismatch')
    for row in sorted(upstream.rows, key=lambda value: value.input_index):
        frame = requested.get(row.input_index)
        image = review.contained(root, row.image_output)
        status = frame['status'] if frame else 'NOT_REQUESTED'
        if frame:
            review.require(frame.get('frame_id') in (None, row.frame_id), 'upstream frame ID mismatch')
            if frame.get('binding_status') == 'BOUND':
                review.require(frame['image_output'] == row.image_output
                               and frame['photo_io_image_sha256'] == row.upstream_image_sha256
                               and (frame['width'], frame['height']) == (row.canonical_width, row.canonical_height),
                               'upstream row identity mismatch')
        selected = bool(frame and status == 'OK')
        mask_binding = None
        if selected:
            candidate = next(c for c in frame['candidates'] if c['candidate_id'] == frame['selected_candidate_id'])
            path = review.contained(manifest.parent, candidate['mask_path'])
            mask_binding = bindings.rows[str(path)]
            masks[row.input_index] = path
        ledger.append(dict(input_index=row.input_index, frame_id=row.frame_id,
                           canonical_image=bindings.rows[str(image)], object_id=object_id,
                           producer_status=status, selected=selected,
                           exclusion_reason=None if selected else status,
                           selected_candidate_id=frame.get('selected_candidate_id') if frame else None,
                           mask_binding=mask_binding, selection=frame.get('selection') if frame else None,
                           producer_errors=frame.get('errors', []) if frame else [],
                           not_run_reason=frame.get('not_run_reason') if frame else None))
    bindings.verify()
    coverage = dict(upstream_count=len(ledger), requested_count=len(requested),
                    selected_count=len(masks), excluded_count=len(ledger)-len(masks),
                    not_requested_count=len(ledger)-len(requested),
                    producer_counts=document['counts'], quality_status='NOT_EVALUATED')
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = output / 'FOREGROUND_SELECTION_LEDGER.json'
    payload = dict(schema='sfm-foreground-selection-v1', source_io=source,
                   object=document['object'], manifest=bindings.rows[str(manifest)],
                   rows=ledger, coverage=coverage, source_bindings=list(bindings.rows.values()),
                   selection_rule='only producer-selected nonempty OK masks; no largest-candidate selection',
                   source_unchanged=True, quality_status='NOT_EVALUATED')
    ledger_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    bindings.verify()
    return dict(selected_indices=sorted(masks), masks_by_index=masks, ledger_path=ledger_path,
                manifest_sha256=review.digest(raw), coverage=coverage)
