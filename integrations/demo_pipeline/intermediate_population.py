"""Frozen original foreground cohorts and strictly two-dimensional summaries."""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import numpy as np

RIGHTS = dict(geometry_accepted=False, new_cameras=0, new_model_points=0)
GRID = (12, 8)  # rows, columns; same 8-column / 12-row grid as tracking


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _domain(frame, mask):
    _need(frame.get('original_mask_status') == 'OK', 'ORIGINAL_MASK_REQUIRED')
    _need(isinstance(mask, np.ndarray) and mask.dtype == np.uint8 and
          mask.shape == (frame['height'], frame['width']), 'INVALID_MASK_SHAPE')
    _need(np.isin(mask, (0, 255)).all(), 'NONBINARY_MASK')
    ys, xs = np.nonzero(mask)
    _need(len(xs) > 0, 'EMPTY_OBJECT_MASK')
    bbox = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
    cx = np.clip(((xs+.5-bbox[0])*GRID[1]/(bbox[2]-bbox[0])).astype(int), 0, GRID[1]-1)
    cy = np.clip(((ys+.5-bbox[1])*GRID[0]/(bbox[3]-bbox[1])).astype(int), 0, GRID[0]-1)
    cells = np.unique(cy*GRID[1]+cx).tolist()
    return dict(bbox=bbox, rows=GRID[0], columns=GRID[1], mask_cells=cells)


def _cell(xy, bbox):
    x0, y0, x1, y1 = bbox
    return min(11, max(0, int((xy[1]-y0)*12/(y1-y0))))*8 + min(7, max(0, int((xy[0]-x0)*8/(x1-x0))))


def build_population_plan(frames, masks, tracking, interval_plan):
    """Select once before LK; tracking frame_index is a local array position."""
    _need(len(frames) == len(masks), 'FRAME_MASK_COUNT')
    if not interval_plan['edges']:
        return dict(edges=[], decode_indices=[], skipped_observations=[],
            max_points_per_edge=768, batch_size=64, source_points_changed=False,
            new_corner_detection=False, **RIGHTS)
    registry = {}
    for pos, frame in enumerate(frames):
        idx = frame.get('input_index')
        _need(type(idx) is int and idx >= 0 and idx not in registry, 'INVALID_FRAME_IDENTITY')
        _need(type(frame.get('decode_index')) is int and frame['decode_index'] >= 0, 'INVALID_DECODE_IDENTITY')
        registry[idx] = (pos, frame)
    _need(len({f['decode_index'] for f in frames}) == len(frames), 'DUPLICATE_DECODE_IDENTITY')
    source_indices = {e['source_input_index'] for e in interval_plan['edges']}
    _need(source_indices <= set(registry), 'UNKNOWN_INTERVAL_SOURCE')
    cohorts = {}; skipped = []; seen = {}; tids = set(); domains = {}
    for track in tracking['tracks']:
        tid = track.get('track_id')
        _need(type(tid) is int and tid >= 0 and tid not in tids, 'INVALID_TRACK_IDENTITY'); tids.add(tid)
        for obs in track['observations']:
            pos = obs.get('frame_index'); seg = obs.get('identity_segment')
            _need(type(pos) is int and 0 <= pos < len(frames), 'INVALID_LOCAL_FRAME_POSITION')
            _need(type(seg) is int and seg >= 0, 'INVALID_SEGMENT_IDENTITY')
            idx = frames[pos]['input_index']; key = (tid, seg, idx)
            if idx not in source_indices: continue
            if key in seen:
                _need(seen[key] == obs, 'CONTRADICTORY_DUPLICATE_OBSERVATION')
                continue
            seen[key] = obs
            reason = None
            if obs.get('label') != 'foreground': reason = 'NON_FOREGROUND'
            elif obs.get('target_verified') is not True: reason = 'TARGET_UNVERIFIED'
            elif obs.get('tracking_status') not in ('seed', 'flow_tracked'): reason = 'UNSUPPORTED_TRACKING_STATUS'
            elif frames[pos].get('original_mask_status') != 'OK' or masks[pos] is None: reason = 'UNKNOWN_ORIGINAL_MASK'
            xy = np.asarray(obs.get('xy'), dtype=float)
            _need(xy.shape == (2,) and np.isfinite(xy).all(), 'INVALID_RAW_PIXEL_CENTER')
            if reason is None:
                if pos not in domains: domains[pos] = _domain(frames[pos], masks[pos])
                x, y = np.floor(xy).astype(int); h, w = masks[pos].shape
                if not (.5 <= xy[0] <= w-.5 and .5 <= xy[1] <= h-.5 and masks[pos][y, x] == 255):
                    reason = 'OUTSIDE_ORIGINAL_MASK'
            pid = f'track:{tid}:segment:{seg}:source:{idx}'
            if reason:
                skipped.append(dict(point_id=pid, source_input_index=idx, reason=reason)); continue
            point = dict(group_id=pid, point_id=pid, track_id=tid, identity_segment=seg,
                source_input_index=idx, source_position=pos, source_decode_index=frames[pos]['decode_index'],
                source_raw_xy=xy.tolist(), birth_kind='NEW_SEED' if obs.get('tracking_status') == 'seed' else 'CONTINUING_OBSERVATION',
                original_tracking=deepcopy(obs), original_strong=obs.get('strong'),
                original_observed=obs.get('observed'), source_mask_status='OK', is_3d=False, **RIGHTS)
            cohorts.setdefault(idx, []).append(point)
    bins = {}
    for old in interval_plan['edges']:
        a, b = old['source_input_index'], old['target_input_index']
        _need(a in registry and b in registry and a != b, 'UNKNOWN_INTERVAL_ENDPOINT')
        af, bf = registry[a][1], registry[b][1]
        direction = 1 if bf['decode_index'] > af['decode_index'] else -1
        path = list(range(af['decode_index'], bf['decode_index']+direction, direction))
        _need(old['decode_path'] == path and old['source_decode_index'] == path[0] and old['target_decode_index'] == path[-1], 'INTERVAL_DECODE_MISMATCH')
        key = (a, b)
        if key not in bins:
            edge = deepcopy(old); edge['model_keys'] = []; bins[key] = edge
        for model in old.get('model_keys', [old['model_key']]):
            _need(isinstance(model, str) and bool(model), 'INVALID_MODEL_KEY')
            if model not in bins[key]['model_keys']: bins[key]['model_keys'].append(model)
    edges = []
    for (a, b), edge in sorted(bins.items()):
        sp, sf = registry[a]; tp, tf = registry[b]
        if sp not in domains: domains[sp] = _domain(sf, masks[sp])
        if tp not in domains: domains[tp] = _domain(tf, masks[tp])
        source_domain = domains[sp]; target_domain = domains[tp]
        buckets = defaultdict(list)
        for point in cohorts.get(a, []):
            p = deepcopy(point); p['source_cell'] = _cell(p['source_raw_xy'], source_domain['bbox'])
            buckets[p['source_cell']].append(p)
        for values in buckets.values(): values.sort(key=lambda p: (p['source_raw_xy'][1], p['source_raw_xy'][0], p['track_id'], p['identity_segment']))
        ordered = []
        while any(buckets.values()):
            for cell in sorted(buckets):
                if buckets[cell]: ordered.append(buckets[cell].pop(0))
        edge.update(model_keys=sorted(edge['model_keys']), points=ordered[:768],
            source_candidate_count=len(ordered), source_domain=source_domain, target_domain=target_domain,
            all_source_candidate_cells=sorted({p['source_cell'] for p in ordered}),
            omitted_points=[dict(point_id=p['point_id'], reason='NOT_EVALUATED_POINT_BUDGET') for p in ordered[768:]],
            selection='8x12 object bbox cells, spatial order then round robin; no result filtering or refill', **RIGHTS)
        edge['batches'] = [dict(batch_index=i//64, point_ids=[p['point_id'] for p in edge['points'][i:i+64]],
                                points=deepcopy(edge['points'][i:i+64])) for i in range(0, len(edge['points']), 64)]
        edges.append(edge)
    return dict(edges=edges, decode_indices=sorted({i for e in edges for i in e['decode_path']}),
        skipped_observations=skipped, max_points_per_edge=768, batch_size=64,
        source_points_changed=False, new_corner_detection=False, **RIGHTS)


def _verified_leg(step):
    error = step.get('fb_error_px')
    if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 1: return False
    for name in ('forward', 'backward'):
        record = step.get(name, {})
        try:
            status = np.asarray(record['status']); raw_error = np.asarray(record['error']); err = raw_error.astype(float)
            xy = np.asarray(record['observed_raw_xy'], dtype=float)
            if status.dtype.kind not in 'iu' or status.shape != (1, 1) or status[0, 0] != 1 or raw_error.dtype.kind not in 'iuf' or err.shape != (1, 1) or not np.isfinite(err).all() or not 0 <= err[0, 0] <= 25 or xy.shape != (2,) or not np.isfinite(xy).all(): return False
        except (KeyError, TypeError, ValueError): return False
    return True


def summarize_edge(edge, comparison, decoded):
    """Count accepted numerical legs, never mere observed coordinates."""
    points = {p['point_id']: p for p in edge['points']}; rows = comparison['rows']
    _need(len({r['group_id'] for r in rows}) == len(rows), 'DUPLICATE_COMPARISON_ID')
    _need({r['group_id'] for r in rows} == set(points), 'INCOMPLETE_OR_FOREIGN_COMPARISON_COHORT')
    n = len(edge['decode_path'])-1; histogram = [0]*(n+1); survival = [0]*n
    keys = ('evaluated', 'direct_passed', 'dense_passed', 'RECOVERED', 'LOST', 'BOTH_PASS', 'BOTH_FAIL')
    counts = Counter({k: 0 for k in keys})
    births = {kind: Counter({k: 0 for k in keys}) for kind in ('NEW_SEED', 'CONTINUING_OBSERVATION')}
    source_pass = []; target_pass = []; durations = []
    for row in rows:
        _need(row['group_id'] in points, 'UNKNOWN_COMPARISON_ID')
        point = points[row['group_id']]
        _need(row['source_raw_xy'] == point['source_raw_xy'], 'COMPARISON_SOURCE_MISMATCH')
        direct = row['direct']['status'] == 'PASSED'; dense = row['dense']['status'] == 'PASSED'
        outcome = 'BOTH_PASS' if direct and dense else 'RECOVERED' if dense else 'LOST' if direct else 'BOTH_FAIL'
        _need(row.get('outcome') == outcome, 'COMPARISON_OUTCOME_MISMATCH')
        for counter in (counts, births[point['birth_kind']]):
            counter['evaluated'] += 1; counter['direct_passed'] += int(direct); counter['dense_passed'] += int(dense); counter[outcome] += 1
        survived = 0
        for step in row['dense']['steps'][:n]:
            if not _verified_leg(step): break
            survival[survived] += 1; survived += 1
        histogram[survived] += 1
        path = edge['decode_path']; start = decoded[path[0]].get('time_seconds'); end = decoded[path[survived]].get('time_seconds')
        durations.append(dict(point_id=point['point_id'], verified_legs=survived,
                              duration_seconds=abs(end-start) if start is not None and end is not None else None))
        if dense:
            _need(survived == n, 'PASSED_DENSE_WITHOUT_VERIFIED_LEGS')
            observations = row['dense']['observed_raw_coordinates']
            _need(observations and observations[-1]['decode_index'] == path[-1], 'DENSE_ENDPOINT_MISMATCH')
            source_pass.append(point['source_raw_xy']); target_pass.append(observations[-1]['raw_xy'])
    def coverage(domain, coordinates):
        allowed = set(domain['mask_cells']); occupied = {_cell(xy, domain['bbox']) for xy in coordinates} & allowed
        return dict(occupied=len(occupied), total=len(allowed), fraction=len(occupied)/len(allowed) if allowed else None,
                    occupied_cells=sorted(occupied), meaning='distinct nonzero original mask cells; not surface percent')
    source_all = coverage(edge['source_domain'], [p['source_raw_xy'] for p in edge['points']])
    cells = edge.get('all_source_candidate_cells', source_all['occupied_cells'])
    source_all.update(occupied=len(cells), occupied_cells=cells,
                      fraction=len(cells)/source_all['total'] if source_all['total'] else None)
    return dict(counts=dict(counts), by_birth_kind={k: dict(v) for k, v in births.items()},
        dense_per_leg_survival=[dict(from_decode_index=a, to_decode_index=b, survived=survival[i]) for i, (a,b) in enumerate(zip(edge['decode_path'], edge['decode_path'][1:]))],
        survival_histogram=[dict(verified_legs=i, count=v) for i,v in enumerate(histogram)], durations=durations,
        coverage=dict(source_all=source_all,
                      source_dense_passed=coverage(edge['source_domain'], source_pass), target_dense_passed=coverage(edge['target_domain'], target_pass)),
        continuity='2D_NUMERICAL_CONTINUITY_ONLY', **RIGHTS)
