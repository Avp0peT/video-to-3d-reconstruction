"""Retain measured consecutive LK prefixes; no camera or geometry admission."""
from copy import deepcopy
from collections import Counter
import math
import numpy as np

RIGHTS = dict(geometry_accepted=False, new_3d_points=0, new_cameras=0, new_model_points=0)


class _Rejected(Exception):
    pass


def _require(value, reason):
    if not value: raise _Rejected(reason)


def _xy(value):
    try:
        _require(isinstance(value,(list,tuple,np.ndarray)), 'INVALID_RAW_COORDINATES')
        _require(not any(isinstance(v,(bool,np.bool_)) for v in value), 'INVALID_RAW_COORDINATES')
        raw = np.asarray(value)
        _require(raw.dtype.kind in 'iuf', 'INVALID_RAW_COORDINATES')
        xy = raw.astype(float)
    except (TypeError, ValueError): raise _Rejected('INVALID_RAW_COORDINATES')
    _require(xy.shape == (2,) and np.isfinite(xy).all(), 'INVALID_RAW_COORDINATES')
    return xy


def _inside(xy, meta):
    w, h = meta.get('width'), meta.get('height')
    _require(type(w) is int and type(h) is int and w > 0 and h > 0, 'INVALID_FRAME_DIMENSIONS')
    _require(.5 <= xy[0] <= w-.5 and .5 <= xy[1] <= h-.5, 'OUTSIDE_IMAGE')


def _flow(record, a, b, start, decoded):
    _require(record.get('from_decode_index') == a and record.get('to_decode_index') == b,
             'FLOW_ENDPOINT_IDENTITY_MISMATCH')
    _require(np.array_equal(_xy(record.get('input_raw_xy')), start), 'FLOW_INPUT_COORDINATES_MISMATCH')
    _inside(start, decoded[a])
    try:
        status = np.asarray(record['status']); error = np.asarray(record['error'])
        xy = np.asarray(record['xy'])
    except (KeyError, TypeError, ValueError): raise _Rejected('MISSING_RAW_LK_OUTPUT')
    _require(status.shape == (1,1) and status.dtype.kind in 'iu', 'INVALID_LK_STATUS')
    _require(status[0,0] == 1, 'LK_STATUS_FAILED')
    _require(error.shape == (1,1) and error.dtype.kind in 'iuf' and np.isfinite(error).all()
             and 0 <= error[0,0] <= 25, 'INVALID_OR_EXCESSIVE_LK_ERROR')
    _require(xy.shape == (1,1,2) and xy.dtype.kind in 'iuf' and np.isfinite(xy).all(), 'INVALID_RAW_LK_OUTPUT')
    observed = _xy(record.get('observed_raw_xy'))
    _require(np.array_equal(xy[0,0].astype(float)+.5, observed), 'RAW_OBSERVED_COORDINATES_MISMATCH')
    _inside(observed, decoded[b])
    return observed


def _observation(index, xy, decoded):
    meta = decoded[index]
    time = meta.get('time_seconds')
    _require(time is None or type(time) in (int,float) and math.isfinite(time), 'INVALID_FRAME_TIME')
    return dict(decode_index=index, raw_xy=xy.tolist(), width=meta['width'],height=meta['height'],
                path=meta.get('path'), time_seconds=time,
                original_mask_status=meta.get('original_mask_status', 'UNKNOWN_NOT_PROVIDED'))


def extract_tracklets(population, decoded_meta, *, budget=lambda: None):
    """Each frozen source gets one record, including interrupted or absent rows."""
    plan = population['plan']; measured_edges = population.get('edges', [])
    if len(measured_edges) > len(plan['edges']):
        raise ValueError('FOREIGN_MEASURED_EDGE')
    records = []
    for ordinal, edge in enumerate(plan['edges']):
        budget()
        measured = measured_edges[ordinal] if ordinal < len(measured_edges) else {}
        rows = measured.get('rows', [])
        row_map = {}; ids = {p['point_id'] for p in edge['points']}
        for row in rows:
            pid = row.get('group_id')
            if pid not in ids: raise ValueError('FOREIGN_MEASUREMENT_IDENTITY')
            if pid in row_map: raise ValueError('DUPLICATE_MEASUREMENT_IDENTITY')
            row_map[pid] = row
        for point in edge['points']:
            budget()
            pid = point['point_id']; path = edge['decode_path']; row = row_map.get(pid)
            out = dict(tracklet_id=f'{pid}:edge:{ordinal}', point_id=pid, edge_ordinal=ordinal,
                track_id=point['track_id'], identity_segment=point['identity_segment'],
                source_input_index=point['source_input_index'], source_point=deepcopy(point),
                observations=[], verified_legs=0, status='EXCLUDED', termination_reason=None,
                wholecycle_status=None, wholecycle_reasons=[], **RIGHTS)
            records.append(out)
            try:
                _require(all(type(point.get(k)) is int and point[k] >= 0 for k in
                             ('track_id','identity_segment','source_input_index')), 'INVALID_SOURCE_IDENTITY_TYPE')
                _require(type(pid) is str, 'INVALID_SOURCE_POINT_ID_TYPE')
                _require(pid == f"track:{point['track_id']}:segment:{point['identity_segment']}:source:{point['source_input_index']}", 'SOURCE_POINT_IDENTITY_MISMATCH')
                _require(row is not None, 'NOT_MEASURED_OR_INTERRUPTED')
                _require(measured.get('source_input_index') == edge['source_input_index']
                         and measured.get('target_input_index') == edge['target_input_index']
                         and measured.get('decode_path') == path, 'MEASURED_EDGE_IDENTITY_MISMATCH')
                _require(path and path[0] == point['source_decode_index'] and path[0] == edge['source_decode_index']
                         and path[-1] == edge['target_decode_index'] and point['source_input_index'] == edge['source_input_index'], 'SOURCE_INTERVAL_IDENTITY_MISMATCH')
                direction = 1 if path[-1] > path[0] else -1
                _require(len(path) >= 3 and all(type(i) is int and i >= 0 for i in path)
                         and all(b-a == direction for a,b in zip(path,path[1:])), 'INVALID_CONSECUTIVE_PATH')
                _require(all(i in decoded_meta for i in path), 'MISSING_DECODED_FRAME_METADATA')
                current = _xy(point['source_raw_xy']); _inside(current, decoded_meta[path[0]])
                _require(np.array_equal(current, _xy(row.get('source_raw_xy'))), 'MEASUREMENT_SOURCE_MISMATCH')
                dense = row.get('dense', {})
                cycle = dense.get('fullcycle', {})
                out['wholecycle_status'] = cycle.get('status', 'NOT_RECORDED')
                out['wholecycle_reasons'] = deepcopy(dense.get('reasons', []))
                out['observations'].append(_observation(path[0],current,decoded_meta))
                steps = dense.get('steps', [])
                for i,(a,b) in enumerate(zip(path,path[1:])):
                    budget()
                    _require(i < len(steps), 'MISSING_STEP_OR_INTERRUPTED')
                    step = steps[i]
                    _require(isinstance(step,dict), 'INVALID_STEP_RECORD')
                    forward = _flow(step.get('forward', {}),a,b,current,decoded_meta)
                    backward = _flow(step.get('backward', {}),b,a,forward,decoded_meta)
                    fb = float(np.linalg.norm(backward-current))
                    _require(math.isfinite(fb) and fb <= 1, 'RECOMPUTED_FB_THRESHOLD')
                    # Stored FB is corroborating evidence, never a replacement for raw output.
                    stored = step.get('fb_error_px')
                    _require(type(stored) in (int,float) and math.isfinite(stored) and stored >= 0
                             and abs(stored-fb) <= 1e-6, 'FB_RECORD_MISMATCH')
                    out['observations'].append(_observation(b,forward,decoded_meta))
                    out['verified_legs'] += 1; current = forward
                out['termination_reason'] = 'INTERVAL_END'
            except _Rejected as exc:
                out['termination_reason'] = str(exc)
            if out['verified_legs'] >= 2: out['status'] = 'CANDIDATE_2D_TRACKLET'
    return dict(status='SHORT_TRACKLET_LEDGER_COMPLETE', records=records,
                counts=dict(Counter(r['status'] for r in records)),
                selection='consecutive measured prefix from source; at least two numerically verified legs',
                geometry_status='NOT_EVALUATED', **RIGHTS)
