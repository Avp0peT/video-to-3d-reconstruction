"""Bind measured short prefixes to conditional XYZ, without admitting cameras."""
from collections import Counter, defaultdict
from copy import deepcopy
import math

from .original_anchor_review import _camera
from .tracklet_camera_math import assess_projection

PROFILE = dict(id='short-tracklet-camera-diagnostic-v1', max_records=32,
               max_classification_rows=10000, max_candidates=96, max_tracklets=1536, max_points_per_record=768,
               minimum_dlt_points=6, minimum_holdout_points=9,
               cohorts=['STRICT', 'STRICT_PLUS_RELAXED'])
RIGHTS = dict(geometry_accepted=False, camera_pose_established=False,
              new_cameras=0, new_model_points=0, native_injected=False)


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _vector(value, size):
    _need(isinstance(value, (list, tuple)) and len(value) == size and
          all(type(v) in (int, float) and math.isfinite(v) for v in value),
          'INVALID_FINITE_COORDINATES')
    return list(value)


def assess_cameras(ledger, classification, frames, camera_result, *, budget=lambda: None):
    """Exact identity + exact source observation, no nearest-neighbour joining.

    These XYZ inherit the original estimated-camera/central-radial hypotheses.
    A fitted unrestricted pinhole matrix does not resolve distortion or K/R/t.
    The heldout split tests only consistency within this selected evidence.
    """
    result = dict(status='TRACKLET_CAMERA_DIAGNOSTICS_COMPLETE', profile=deepcopy(PROFILE),
                  records=[], joins=[], counts={}, numeric_fits=0,
                  intrinsics_borrowed=False, source_observations_modified=False,
                  scope='conditional XYZ and tracked pixels; unknown query distortion; not calibration',
                  **RIGHTS)
    if classification is None:
        result['status'] = 'NOT_AVAILABLE_NO_CURRENT_CLASSIFICATION'
        return result
    if ledger.get('records') == []:
        result['status'] = 'NOT_APPLICABLE_NO_SHORT_TRACKLETS'
        return result
    _need(classification.get('status') == 'ANCHOR_CANDIDATES_CLASSIFIED', 'CLASSIFICATION_STATUS')
    candidates = classification.get('rows')
    records = ledger.get('records')
    _need(isinstance(candidates, list) and len(candidates) <= PROFILE['max_classification_rows'], 'CLASSIFICATION_ROW_BOUND')
    _need(isinstance(records, list) and len(records) <= PROFILE['max_tracklets'], 'TRACKLET_BOUND')
    byinput = {}; bydecode = {}; models = {}
    for frame in frames:
        budget()
        idx, dec = frame['input_index'], frame['decode_index']
        _need(type(idx) is int and idx >= 0 and idx not in byinput and
              type(dec) is int and dec >= 0 and dec not in bydecode, 'FRAME_IDENTITY')
        byinput[idx] = frame; bydecode[dec] = frame
    for model in camera_result.get('models', []) + camera_result.get('unsupported_models', []):
        budget(); key = model['model_key']
        _need(isinstance(key, str) and key and key not in models, 'MODEL_IDENTITY')
        cams = {}
        for table in ('cameras', 'unsupported_cameras'):
            for idx, cam in model.get(table, {}).items():
                idx = int(idx) if isinstance(idx, str) and idx.isdigit() else idx
                _need(type(idx) is int and idx in byinput and idx not in cams and
                      cam.get('input_index') == idx, 'CAMERA_FRAME_IDENTITY')
                _need(cam.get('frame_id') == byinput[idx]['frame_id'], 'CAMERA_FRAME_IDENTITY')
                cams[idx] = cam
        models[key] = cams
    indexed = defaultdict(list); ids = set(); eligible_count = 0
    for candidate in candidates:
        budget()
        key, tid, seg = (candidate.get(k) for k in ('model_key', 'track_id', 'identity_segment'))
        _need(key in models and type(tid) is int and tid >= 0 and type(seg) is int and seg >= 0,
              'CANDIDATE_IDENTITY')
        gid = candidate.get('group_id')
        _need(gid == f'{key}:{tid}:{seg}' and gid not in ids, 'CANDIDATE_IDENTITY')
        ids.add(gid)
        _need(all(candidate.get(k) is False for k in ('geometry_accepted', 'native_injected', 'identity_verified')),
              'CANDIDATE_RIGHTS')
        if candidate.get('tier') not in ('STRICT_CONDITIONAL', 'RELAXED_CONDITIONAL'):
            continue
        eligible_count += 1
        _need(eligible_count <= PROFILE['max_candidates'], 'CANDIDATE_BOUND')
        _vector(candidate.get('xyz'), 3)
        sources = candidate.get('sources'); source_ids = set()
        _need(isinstance(sources, list) and 3 <= len(sources) <= 12, 'THREE_SOURCE_VIEWS_REQUIRED')
        for source in sources:
            budget(); idx = source.get('input_index')
            _need(type(idx) is int and idx in models[key] and idx not in source_ids, 'SOURCE_CAMERA_IDENTITY')
            source_ids.add(idx); frame = byinput[idx]; cam = models[key][idx]
            _camera(cam, frame, idx)
            _need(source.get('frame_id') == frame['frame_id'] and source.get('image_id') == cam['image_id'],
                  'SOURCE_FRAME_IDENTITY')
            _vector(source.get('raw_pixels'), 2)
        indexed[(tid, seg)].append(candidate)
    groups = defaultdict(dict); seen = set(); domains = {}
    for record in records:
        budget()
        rid = record.get('tracklet_id')
        _need(isinstance(rid, str) and rid not in seen, 'DUPLICATE_TRACKLET')
        seen.add(rid)
        if record.get('status') != 'CANDIDATE_2D_TRACKLET':
            continue
        tid, seg, source_idx = (record.get(k) for k in ('track_id', 'identity_segment', 'source_input_index'))
        _need(all(type(v) is int and v >= 0 for v in (tid, seg, source_idx)) and source_idx in byinput,
              'TRACKLET_SOURCE_IDENTITY')
        point = record['source_point']; obs = record['observations']
        pid = f'track:{tid}:segment:{seg}:source:{source_idx}'; edge = record.get('edge_ordinal')
        _need(type(edge) is int and edge >= 0 and record.get('point_id') == pid and
              point.get('point_id') == pid and rid == f'{pid}:edge:{edge}', 'TRACKLET_IDENTITY')
        _need(all(point.get(k) == record[k] for k in ('track_id', 'identity_segment', 'source_input_index')),
              'TRACKLET_SOURCE_IDENTITY')
        legs = record.get('verified_legs')
        _need(type(legs) is int and 2 <= legs <= 8 and len(obs) == legs+1, 'VERIFIED_PREFIX_LENGTH')
        _need(point.get('source_decode_index') == byinput[source_idx]['decode_index'] and
              len(obs) >= 3 and obs[0]['decode_index'] == point['source_decode_index'], 'TRACKLET_SOURCE_DECODE')
        _need(_vector(point['source_raw_xy'], 2) == _vector(obs[0]['raw_xy'], 2), 'SOURCE_COORDINATE_MISMATCH')
        observed_ids = set()
        for ob in obs:
            budget(); dec = ob['decode_index']; xy = _vector(ob['raw_xy'], 2)
            _need(type(dec) is int and dec >= 0 and dec not in observed_ids, 'OBSERVATION_IDENTITY')
            observed_ids.add(dec)
            _need(all(type(ob.get(k)) is int and ob[k] > 0 for k in ('width', 'height')) and
                  .5 <= xy[0] <= ob['width']-.5 and .5 <= xy[1] <= ob['height']-.5, 'OBSERVATION_DOMAIN')
            domain = (ob['width'], ob['height'], ob.get('original_mask_status'))
            _need(dec not in domains or domains[dec] == domain, 'QUERY_DOMAIN_MISMATCH')
            domains[dec] = domain
            if dec in bydecode:
                frame = bydecode[dec]
                _need(domain == (frame['width'], frame['height'], frame['original_mask_status']), 'FRAME_DOMAIN_MISMATCH')
        direction = 1 if obs[-1]['decode_index'] > obs[0]['decode_index'] else -1
        _need(all(b['decode_index']-a['decode_index'] == direction for a,b in zip(obs,obs[1:])),
              'NONCONSECUTIVE_PREFIX')
        for candidate in indexed[(tid, seg)]:
            budget(); key = candidate['model_key']
            source = next((s for s in candidate['sources'] if s['input_index'] == source_idx), None)
            if source is None:
                continue
            _need(source['raw_pixels'] == point['source_raw_xy'], 'SOURCE_COORDINATE_MISMATCH')
            result['joins'].append(dict(tracklet_id=rid, group_id=candidate['group_id'], model_key=key,
                                       tier=candidate['tier'], exact_source_xy=True))
            for ob in obs[1:]:
                budget(); dec = ob['decode_index']; frame = bydecode.get(dec)
                if frame is not None and frame['input_index'] in models[key]:
                    continue
                bucket = groups[(key, dec)]; gid = candidate['group_id']
                _need(gid not in bucket, 'DUPLICATE_QUERY_ANCHOR')
                bucket[gid] = dict(point_id=gid, tracklet_id=rid, xyz=deepcopy(candidate['xyz']),
                                   raw_xy=deepcopy(ob['raw_xy']), tier=candidate['tier'],
                                   source_input_index=source_idx, original_mask_status=ob['original_mask_status'])
    # All cohorts are frozen before any camera SVD. A failure never changes membership.
    for (key, dec), points in sorted(groups.items()):
        for cohort in PROFILE['cohorts']:
            selected = [deepcopy(p) for _, p in sorted(points.items())
                        if cohort == 'STRICT_PLUS_RELAXED' or p['tier'] == 'STRICT_CONDITIONAL']
            _need(len(selected) <= PROFILE['max_points_per_record'], 'QUERY_POINT_BOUND')
            result['records'].append(dict(model_key=key, query_decode_index=dec, cohort=cohort,
                points=selected, original_mask_status=next(iter(points.values()))['original_mask_status'],
                assessment=None, **RIGHTS))
    for ordinal, row in enumerate(result['records']):
        budget()
        if ordinal >= PROFILE['max_records']:
            row['assessment'] = dict(status='NOT_EVALUATED_RECORD_BUDGET', count=len(row['points']),
                                     reasons=['FROZEN_RECORD_BOUND'], numeric_fits=0)
        else:
            row['assessment'] = assess_projection(row['points'], budget=budget)
        result['numeric_fits'] += row['assessment']['numeric_fits']
    result['counts'] = dict(Counter(r['assessment']['status'] for r in result['records']))
    return result
