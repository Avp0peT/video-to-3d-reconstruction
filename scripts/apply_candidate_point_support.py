"""Consume sealed boundary evidence into point-only, provisional annotations."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src'), str(ROOT)]
import numpy as np
from PIL import Image
from review_mask_association import read, save, bind, verify, index_frames, verified_tracks, CLIPS
from review_local_motion import fresh_output
from kozlov_lab.stereo.candidate_point_policy import decide_candidate_points
from kozlov_lab.stereo.candidate_point_consumer import consume_point_decisions

VERSION = 'candidate-point-support-v1'


def sealed_boundary(source: Path, digest: str) -> tuple[dict, list, list]:
    """Digest is an explicit authority input, not accepted from the source itself."""
    source = source.resolve(strict=True)
    if source.name != 'boundary_support' or source.parent.name != 'real01':
        raise ValueError('sealed boundary result layout required')
    task = source.parent.parent
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('lowercase SHA256 required')
    seal = verify(dict(path=str(task / 'SHA256SUMS.txt'), sha256=digest))
    refs = [seal]
    entries = {}
    for line in Path(seal['path']).read_text(encoding='utf-8').splitlines():
        expected, rel = line.split('  ', 1)
        p = (task / rel).resolve(strict=True)
        key = str(p).casefold()
        if not p.is_relative_to(task) or key in entries:
            raise ValueError('invalid or duplicate sealed path')
        entries[key] = expected
        refs.append(verify(dict(path=str(p), sha256=expected)))
    required = [task / 'DELIVERY_CHECK.json', task / 'INDEPENDENT_AUDIT.json',
                source / 'PLAN.json', source / 'RESULT.json', source / 'DECISIONS.json',
                source / 'SELECTION_LEDGER.json', source / 'EXTERIOR_PAIRS.json']
    if any(str(p.resolve()).casefold() not in entries for p in required):
        raise ValueError('incomplete boundary seal')
    for path in required[:2]:
        check = read(path)
        if check.get('status') != 'PASS' or check.get('failures') != []:
            raise ValueError('boundary audit did not pass')
    plan, result, rows = (read(source / (name + '.json')) for name in ('PLAN', 'RESULT', 'DECISIONS'))
    index_frames(rows)
    if (plan['status'] != 'FROZEN_BEFORE_BOUNDARY_SUPPORT'
            or result['status'] != 'BOUNDARY_SUPPORT_DIAGNOSTIC_ONLY'
            or result['frames'] != rows or result['selected_candidates'] != 0
            or result['source_unchanged'] is not True
            or read(source / 'SELECTION_LEDGER.json') != []):
        raise ValueError('boundary result identity or permissions differ')
    for group in ('source_bindings', 'implementation', 'runtime_bindings'):
        for ref in plan[group]:
            refs.append(verify(ref))
    return plan, rows, refs


def check_pixel_evidence(row: dict, frame: dict) -> None:
    """Read actual old mask pixels; do not trust an inside flag in isolation."""
    if row['input_index'] != frame['input_index'] or row['frame_id'] != frame['frame_id']:
        raise ValueError('frame identity differs')
    if row['original_status'] != frame['original_mask_status']:
        raise ValueError('original frame status differs')
    verify(dict(path=frame['path'], sha256=frame['sha256']))
    if not row['candidates']:
        if row['original_status'] != 'NO_MATCH' or row['assessment'] is not None:
            raise ValueError('missing candidates on a non-NO_MATCH frame')
        return
    assessments = {r['candidate_id']: r for r in row['assessment']['candidates']}
    if len(assessments) != len(row['candidates']):
        raise ValueError('candidate coverage differs')
    for candidate in row['candidates']:
        verify(candidate['binding'])
        with Image.open(candidate['binding']['path']) as image:
            if image.mode != 'L' or image.size != (frame['width'], frame['height']):
                raise ValueError('candidate image format differs')
            mask = np.array(image)
        if not np.isin(mask, [0, 255]).all():
            raise ValueError('candidate labels differ')
        assessment = assessments[candidate['candidate_id']]
        if assessment['area_px'] != int(np.count_nonzero(mask)):
            raise ValueError('candidate area differs')
        for field in ('positive_evidence', 'exterior_evidence'):
            for point in assessment[field]['points']:
                xy = point['query_point_xy']
                if not (0 <= xy[0] <= frame['width'] - 1 and 0 <= xy[1] <= frame['height'] - 1):
                    raise ValueError('point outside closed pixel-center range')
                x, y = np.floor(np.asarray(xy) + .5).astype(int)
                if point['sample_xy'] != [int(x), int(y)] or point['inside'] is not bool(mask[y, x]):
                    raise ValueError('saved membership differs from actual candidate bytes')


def process_rows(rows: list, frames: dict, tracking: dict) -> tuple[list, list, dict]:
    decisions, ledger = [], []
    for row in rows:
        check_pixel_evidence(row, frames[row['input_index']])
        decision = decide_candidate_points(row)
        if decision['status'] == 'INVALID_EVIDENCE':
            raise ValueError('invalid evidence: ' + decision['reason'])
        decision.update(input_index=row['input_index'], frame_id=row['frame_id'],
                        image_binding=dict(path=frames[row['input_index']]['path'],
                                           sha256=frames[row['input_index']]['sha256']),
                        source_candidate_bindings=row['candidates'])
        decisions.append(decision)
        if decision['status'] == 'PROVISIONAL_POINT_SUPPORT_ONLY':
            selected = next(c for c in row['candidates']
                            if c['candidate_id'] == decision['provisional_candidate_id'])
            ledger.append(dict(decision, candidate_binding=selected['binding'],
                               mask_quality_status='NOT_EVALUATED',
                               boundary_unknown=next(c['boundary'] for c in row['assessment']['candidates']
                                                     if c['candidate_id'] == decision['provisional_candidate_id'])))
    annotations = {}
    for lo, hi in CLIPS:
        name = f'{lo}_{hi}'
        applicable = [d for d in decisions if lo <= d['input_index'] <= hi]
        annotations[name] = consume_point_decisions(tracking[name], applicable, lo)
        # Count the original source labels, not a binary "green/yellow" inference.
        annotations[name]['provisional_source_labels'] = dict(Counter(
            r['source_label'] for r in annotations[name]['rows']
            if r['derived_point_support'] == 'PROVISIONAL_POINT_SUPPORT_ONLY'))
    return decisions, ledger, annotations


def build(source: Path, seal_sha256: str, tracks_review: Path, output: Path) -> dict:
    started = time.monotonic()
    output = fresh_output(output, (source, tracks_review))
    plan, rows, refs = sealed_boundary(source, seal_sha256)
    frames = {r['input_index']: r['original_frame'] for r in plan['frames']}
    if set(frames) != {r['input_index'] for r in rows}:
        raise ValueError('frozen frame coverage differs')
    tracking = {}
    for lo, hi in CLIPS:
        name = f'{lo}_{hi}'
        data, bound = verified_tracks(tracks_review, name)
        # The tracker must be a dependency of the already sealed boundary source.
        for ref in bound:
            if not any(Path(r['path']).resolve() == Path(ref['path']).resolve()
                       and r['sha256'] == ref['sha256'] for r in refs):
                raise ValueError('tracker is not bound by boundary evidence')
        if data['profile']['profile_id'] != 'whole-video-temporal-support-v2':
            raise ValueError('wrong tracking source')
        tracking[name] = data
        refs.extend(bound)
    implementation = [bind(ROOT / p) for p in (
        'scripts/apply_candidate_point_support.py',
        'src/kozlov_lab/stereo/candidate_point_policy.py',
        'src/kozlov_lab/stereo/candidate_point_consumer.py')]
    output.mkdir(parents=True)
    try:
        save(output / 'PLAN.json', dict(status='FROZEN_BEFORE_POINT_CONSUMPTION', version=VERSION,
            source=bind(source / 'DECISIONS.json'), source_bindings=refs, implementation=implementation,
            boundary_seal_sha256=seal_sha256, frames=list(frames.values()), clips=CLIPS,
            rule='unique positive-compatible candidate; any positive-compatible exterior conflict holds; existing points only',
            no_edge_area_or_score_tiebreak=True, full_mask_accepted=False,
            new_point_seeding_allowed=False, publishable_for_geometry=False))
        decisions, ledger, annotations = process_rows(rows, frames, tracking)
        save(output / 'DECISIONS.json', decisions)
        save(output / 'POINT_SUPPORT_LEDGER.json', ledger)
        save(output / 'MASK_SELECTION_LEDGER.json', [])
        for name, data in annotations.items():
            save(output / (name + '_POINT_ANNOTATIONS.json'), data)
        for ref in refs + implementation:
            verify(ref)
        selected = [r for a in annotations.values() for r in a['rows']
                    if r['derived_point_support'] == 'PROVISIONAL_POINT_SUPPORT_ONLY']
        result = dict(status='POINT_SUPPORT_CONSUMED_MASK_UNACCEPTED', version=VERSION,
            original_status_counts=dict(Counter(r['original_status'] for r in rows)),
            decision_status_counts=dict(Counter(d['status'] for d in decisions)),
            frames=len(rows), provisional_frames=len(ledger), provisional_observations=len(selected),
            provisional_source_labels=dict(Counter(r['source_label'] for r in selected)),
            unique_clip_track_ids=len({(name, r['track_id']) for name, a in annotations.items()
                                     for r in a['rows'] if r['derived_point_support'] != 'UNCHANGED'}),
            annotation_files={name: bind(output / (name + '_POINT_ANNOTATIONS.json')) for name in annotations},
            source_unchanged=True, new_observations=0, new_masks=0, tracking_calls=0,
            model_calls=0, geometry_calls=0, downstream_3d_calls=0,
            full_mask_accepted=False, publishable_for_geometry=False,
            quality_status='NOT_EVALUATED', wall_seconds=time.monotonic() - started)
        save(output / 'RESULT.json', result)
        return result
    except Exception as exc:
        save(output / 'FAILURE.json', dict(status='FAILED', error=type(exc).__name__,
             message=str(exc), publishable_for_geometry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--boundary', required=True, type=Path)
    parser.add_argument('--boundary-seal-sha256', required=True)
    parser.add_argument('--tracks-review', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.boundary, args.boundary_seal_sha256, args.tracks_review, args.output)
