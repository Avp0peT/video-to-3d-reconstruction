"""Annotate existing measured observations; never grant geometry eligibility.

Call separately for each clip. Track IDs are local to the supplied tracking file.
The returned ledger is deliberately not a replacement tracking file.
"""
from copy import deepcopy
import math

import numpy as np


PROVISIONAL = 'PROVISIONAL_POINT_SUPPORT_ONLY'


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _xy(value):
    if type(value) is not list or len(value) != 2 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in value
    ):
        raise ValueError('xy must contain two finite numeric coordinates')
    return np.asarray(value, dtype=np.float64)


def consume_point_decisions(tracking: dict, decisions: list[dict], clip_start: int) -> dict:
    """Bind provisional decisions to exact saved observations, without mutation.

    Saved xy uses COLMAP pixel centers; review queries use OpenCV coordinates
    rounded to float32. Identity segments are retained independently per row.
    """
    _integer(clip_start, 'clip_start')
    if type(tracking) is not dict or type(tracking.get('tracks')) is not list:
        raise ValueError('tracking must contain tracks')
    if type(decisions) is not list:
        raise ValueError('decisions must be a list')
    rows, sources, track_ids = [], {}, set()
    for track in tracking['tracks']:
        if type(track) is not dict:
            raise ValueError('invalid track')
        tid = _integer(track.get('track_id'), 'track_id')
        if tid in track_ids:
            raise ValueError('duplicate track_id')
        track_ids.add(tid)
        if type(track.get('observations')) is not list:
            raise ValueError('observations must be a list')
        previous_frame, previous_segment = -1, -1
        for obs in track['observations']:
            if type(obs) is not dict:
                raise ValueError('invalid observation')
            frame = _integer(obs.get('frame_index'), 'frame_index')
            segment = _integer(obs.get('identity_segment'), 'identity_segment')
            if frame <= previous_frame or segment < previous_segment:
                raise ValueError('observation frames must increase and identity segments cannot regress')
            previous_frame, previous_segment = frame, segment
            _xy(obs.get('xy'))
            key = (tid, frame)
            if key in sources:
                raise ValueError('duplicate track/frame observation')
            original_eligible = obs.get('geometry_eligible')
            if original_eligible is not None and type(original_eligible) is not bool:
                raise ValueError('invalid source geometry eligibility')
            row = dict(track_id=tid, frame_index=frame,
                source_label=deepcopy(obs.get('label')), source_identity_segment=segment,
                source_xy=deepcopy(obs['xy']), source_declared_geometry_eligible=original_eligible,
                source_original_foreground=(obs.get('label') == 'foreground'
                                            and obs.get('target_verified') is True),
                source_target_verified=deepcopy(obs.get('target_verified')),
                source_mask_status=deepcopy(obs.get('mask_status')),
                derived_point_support='UNCHANGED', provisional_candidate_id=None,
                geometry_eligible=False, derived_geometry_eligible=False,
                physical_identity='UNVERIFIED')
            sources[key] = (obs, row)
            rows.append(row)
    indices, frame_ids, requested = set(), set(), set()
    originals = {'OK', 'NO_MATCH', 'NEEDS_SELECTION', 'unknown', 'background', 'foreground',
                 'ORIGINAL_OK_UNCHANGED', 'NO_CANDIDATE', 'INSUFFICIENT_EVIDENCE',
                 'AMBIGUOUS', 'CONFLICT', 'REFERENCE_CONFLICT_HOLD'}
    for decision in decisions:
        if type(decision) is not dict:
            raise ValueError('invalid decision')
        index = _integer(decision.get('input_index'), 'input_index')
        fid = decision.get('frame_id')
        if type(fid) is not str or not fid or index in indices or fid in frame_ids:
            raise ValueError('invalid or duplicate decision frame')
        indices.add(index)
        frame_ids.add(fid)
        status = decision.get('status')
        if status != PROVISIONAL:
            if status not in originals:
                raise ValueError('unsupported decision status')
            if (decision.get('provisional_candidate_id') is not None
                or decision.get('supported_observations') != []
                or any(decision.get(k, False) is not False for k in
                       ('full_mask_accepted', 'new_point_seeding_allowed', 'publishable_for_geometry'))):
                raise ValueError('nonprovisional decision cannot confer support or rights')
            continue
        if (decision.get('scope') != 'EXISTING_MEASURED_POINTS_ONLY'
            or decision.get('original_status') != 'NEEDS_SELECTION'
            or any(decision.get(k) is not False for k in
                   ('full_mask_accepted', 'new_point_seeding_allowed', 'publishable_for_geometry'))):
            raise ValueError('provisional decision misclaims scope or eligibility')
        cid = decision.get('provisional_candidate_id')
        supports = decision.get('supported_observations')
        if type(cid) is not str or not cid or type(supports) is not list or not supports:
            raise ValueError('candidate and supported observations required')
        frame = index - clip_start
        for support in supports:
            if type(support) is not dict:
                raise ValueError('invalid support')
            tid = _integer(support.get('track_id'), 'track_id')
            query = _xy(support.get('query_point_xy'))
            key = (tid, frame)
            if key in requested or key not in sources:
                raise ValueError('duplicate or unknown requested observation')
            requested.add(key)
            obs, row = sources[key]
            if obs.get('label') == 'NO_MATCH' or obs.get('mask_status') == 'NO_MATCH':
                raise ValueError('NO_MATCH observations cannot receive candidate support')
            if obs.get('strong') is not True:
                raise ValueError('support requires a strong measured observation')
            for field, limit in (('fb_error_px', 1), ('lk_error', 25)):
                value = obs.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                    raise ValueError(f'invalid measured {field}')
            expected = np.asarray(_xy(obs['xy']) - .5, dtype=np.float32)
            # Compare to the incoming values, not an additional rounding of them.
            if not np.array_equal(expected.astype(np.float64), query):
                raise ValueError('query does not match saved observation')
            row.update(derived_point_support=PROVISIONAL, provisional_candidate_id=cid,
                       decision_frame_id=fid, decision_input_index=index)
    return dict(status='CANDIDATE_POINT_ANNOTATIONS_ONLY', rows=rows,
        counts=dict(observations=len(rows),
            prior_labeled=sum(r['source_label'] != 'unknown' for r in rows),
            prior_unknown=sum(r['source_label'] == 'unknown' for r in rows),
            provisional=len(requested)),
        publishable_for_geometry=False, original_unchanged=True,
        original_geometry_filter_evaluated=False,
        full_mask_accepted=False, new_point_seeding_allowed=False)
