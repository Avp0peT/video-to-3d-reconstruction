"""Prepare saved-camera anchor inputs, without numerical geometry or file I/O."""
from copy import deepcopy
import math

from .candidate_point_consumer import consume_point_decisions, PROVISIONAL, _integer, _xy


def build_anchor_groups(tracking, point_ledger, frames_by_index, models, clip_start):
    """Keep every provisional query and assess measured anchors in its segment.

    The caller verifies camera/file bindings and supplies per-observation original
    mask pixel checks. Two admitted anchors permit a later triangulation attempt;
    they do not establish query support, leave-one-out validity, or acceptance.
    """
    annotations = consume_point_decisions(tracking, point_ledger, clip_start)
    if type(frames_by_index) is not dict or type(models) is not list:
        raise ValueError('frame registry and models have invalid types')
    registry_frame_ids = set()
    for index, frame in frames_by_index.items():
        _integer(index, 'registry input_index')
        if type(frame) is not dict or frame.get('input_index') != index:
            raise ValueError('frame registry identity mismatch')
        if type(frame.get('frame_id')) is not str or not frame['frame_id']:
            raise ValueError('frame registry requires frame_id')
        if frame['frame_id'] in registry_frame_ids:
            raise ValueError('duplicate registry frame_id across input indices')
        registry_frame_ids.add(frame['frame_id'])
    model_keys = set()
    for model in models:
        if type(model) is not dict or type(model.get('model_key')) is not str:
            raise ValueError('invalid model key')
        key = model['model_key']
        if key in model_keys or key == '':
            raise ValueError('duplicate or empty model key')
        model_keys.add(key)
        if type(model.get('cameras')) is not dict or type(model.get('binding')) is not dict:
            raise ValueError('camera table and model binding required')
        for index, camera in model['cameras'].items():
            _integer(index, 'camera input_index')
            if type(camera) is not dict:
                raise ValueError('invalid camera')
    tracks = {t['track_id']: t for t in tracking['tracks']}
    selected = {}
    for row in annotations['rows']:
        if row['derived_point_support'] != PROVISIONAL:
            continue
        index = row['decision_input_index']
        frame = frames_by_index.get(index)
        if frame is None or frame['frame_id'] != row['decision_frame_id']:
            raise ValueError('query frame registry identity mismatch')
        key = (row['track_id'], row['source_identity_segment'])
        selected.setdefault(key, []).append(row)
    groups = []
    for (tid, segment), query_rows in selected.items():
        all_obs = tracks[tid]['observations']
        observations = [o for o in all_obs if o['identity_segment'] == segment]
        excluded = [o['frame_index']+clip_start for o in all_obs if o['identity_segment'] != segment]
        applicable = [m for m in models if any(o['frame_index']+clip_start in m['cameras']
                                               for o in observations)]
        for model in applicable or [None]:
            model_key = model['model_key'] if model else None
            cameras = model['cameras'] if model else {}
            assessments, anchors, queries = [], [], []
            for query in query_rows:
                index = query['decision_input_index']
                queries.append(dict(input_index=index, frame_id=query['decision_frame_id'],
                    track_id=tid, identity_segment=segment,
                    source_xy=deepcopy(query['source_xy']),
                    provisional_candidate_id=query['provisional_candidate_id'],
                    status='SAVED_CAMERA_PRESENT' if index in cameras else 'NO_SAVED_CAMERA',
                    query_numeric_evaluable=index in cameras,
                    geometry_accepted=False))
            for obs in observations:
                index = obs['frame_index']+clip_start
                frame = frames_by_index.get(index)
                if frame is None:
                    raise ValueError('source observation frame missing from registry')
                reasons = []
                camera = cameras.get(index)
                if camera is None:
                    reasons.append('NO_SAVED_CAMERA')
                if not (frame.get('original_mask_status') == 'OK'
                        and obs.get('label') == 'foreground'
                        and obs.get('target_verified') is True):
                    reasons.append('UNKNOWN_SOURCE')
                if obs.get('strong') is not True:
                    reasons.append('NON_STRONG')
                for field, limit in (('fb_error_px', 1), ('lk_error', 25)):
                    value = obs.get(field)
                    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                        reasons.append('INVALID_'+field.upper())
                checks = frame.get('measured_point_mask_checks', [])
                if type(checks) is not list:
                    raise ValueError('invalid mask check list')
                matching = []
                for check in checks:
                    if type(check) is not dict:
                        raise ValueError('invalid mask check')
                    _integer(check.get('track_id'), 'mask check track_id')
                    _integer(check.get('frame_index'), 'mask check frame_index')
                    if check['track_id'] == tid and check['frame_index'] == obs['frame_index']:
                        if _xy(check.get('source_xy')).tolist() != _xy(obs['xy']).tolist():
                            raise ValueError('mask check coordinate mismatch')
                        if type(check.get('inside_original_mask')) is not bool:
                            raise ValueError('mask check result must be boolean')
                        matching.append(check)
                if len(matching) > 1:
                    raise ValueError('duplicate mask pixel check')
                inside = matching[0]['inside_original_mask'] if matching else None
                if inside is None:
                    reasons.append('ORIGINAL_MASK_PIXEL_NOT_CHECKED')
                elif not inside:
                    reasons.append('OUTSIDE_ORIGINAL_MASK')
                if inside is True and not isinstance(frame.get('mask_binding'), dict):
                    reasons.append('NO_ORIGINAL_MASK_BINDING')
                assessment = dict(input_index=index, frame_index=obs['frame_index'],
                    frame_id=frame['frame_id'], track_id=tid, identity_segment=segment,
                    source_xy=deepcopy(obs['xy']), reasons=reasons,
                    status='ADMITTED_ORIGINAL_ANCHOR' if not reasons else 'EXCLUDED',
                    original_mask_inside=inside)
                assessments.append(assessment)
                if not reasons:
                    required = ('image_id', 'camera_model', 'params', 'R', 't')
                    if any(k not in camera for k in required):
                        raise ValueError('admitted anchor camera missing fields')
                    anchors.append(dict(input_index=index, image_id=camera['image_id'],
                        frame_id=frame['frame_id'], track_id=tid, identity_segment=segment,
                        model_key=model_key, raw_pixels=deepcopy(obs['xy']),
                        **{k: deepcopy(camera[k]) for k in ('camera_model','params','R','t')},
                        source_label=obs['label'], source_target_verified=obs['target_verified'],
                        strong=obs['strong'], fb_error_px=obs['fb_error_px'],
                        lk_error=obs['lk_error'], original_mask_inside=True))
            groups.append(dict(track_id=tid, identity_segment=segment, model_key=model_key,
                model_binding=deepcopy(model['binding']) if model else None,
                status='ANCHOR_INPUTS_ONLY' if model else 'NO_MODEL_WITH_ANCHORS',
                queries=queries, assessments=assessments, anchors=anchors,
                excluded_other_segment_count=len(excluded), excluded_other_segment_frames=excluded,
                ready_for_numeric_evaluation=len(anchors) >= 2,
                query_numeric_evaluable=any(q['query_numeric_evaluable'] for q in queries),
                geometry_accepted=False, physical_identity='UNVERIFIED'))
    return dict(status='SAVED_CAMERA_ANCHOR_INPUTS_ONLY', groups=groups,
        unique_query_count=sum(len(q) for q in selected.values()),
        track_segment_count=len(selected), geometry_accepted=False,
        publishable_for_geometry=False, calibration_accepted=False,
        source_unchanged=True, numerical_geometry_evaluated=False)
