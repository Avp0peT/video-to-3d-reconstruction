"""Deterministic measured-ray pair ranking for fixed ordinary cameras.

No candidate XYZ, fit, residual, identity label or ground truth is consulted.
Selection is only estimated-camera geometric conditioning, not matching
verification, physical accuracy, or geometry acceptance.
"""
import itertools
import json
import math

import numpy as np

from . import multiview_fixed as fixed
from . import radial_diagnostics as radial


NUMERICAL_TOLERANCE = 1e-10


def select_pair(observations):
    """Rank by sin(theta); exclude numerical center/ray degeneracy.

Angles use undistorted measured world rays. Baseline exclusion is relative
to the maximum separation of all estimated centers, so world units do not
set an acceptance threshold. Exact score ties use ascending image IDs.
All pairs (including excluded pairs) retain diagnostics and reasons.
"""
    result = dict(status='REFUSED', reason=None, selected_image_ids=None,
                  number_of_views=0, number_of_pairs=0, pairs=[],
                  selected_score=None, selected_ray_separation_degrees=None,
                  selected_triangulation_angle_degrees=None,
                  numerical_relative_tolerance=NUMERICAL_TOLERANCE,
                  scope='estimated-camera measured-ray conditioning only; no matching identity, physical accuracy or geometry acceptance')
    try:
        if not isinstance(observations, (list, tuple)) or not 2 <= len(observations) <= 64:
            raise ValueError('2..64 complete observations required')
        obs = sorted(fixed._observations(observations), key=lambda o: o['image_id'])
        result['number_of_views'] = len(obs)
        result['number_of_pairs'] = len(obs)*(len(obs)-1)//2
        centers, rays = [], []
        for o in obs:
            ideal = radial.undistort_simple_radial(
                o['raw_pixels'][None, :], o['params'],
                max_ideal_radius=o['max_ideal_radius'])['ideal_pixels'][0]
            uv = (ideal-o['params'][1:3])/o['params'][0]
            ray = o['R'].T @ np.array([uv[0], uv[1], 1.])
            norm = math.hypot(*ray)
            center = -o['R'].T @ o['t']
            if not np.all(np.isfinite(center)) or not math.isfinite(norm) or norm <= 0:
                raise ValueError('nonfinite estimated center or measured ray')
            centers.append(center)
            rays.append(ray/norm)
        lengths = {}
        for i, j in itertools.combinations(range(len(obs)), 2):
            with np.errstate(over='ignore', invalid='ignore'):
                length = math.hypot(*(centers[i]-centers[j]))
            if not math.isfinite(length):
                raise ValueError('nonfinite center separation')
            lengths[i, j] = length
        maximum = max(lengths.values())
        result['max_center_separation'] = maximum
        eligible = []
        for (i, j), length in lengths.items():
            sine = float(np.clip(math.hypot(*np.cross(rays[i], rays[j])), 0., 1.))
            cosine = float(np.clip(np.dot(rays[i], rays[j]), -1., 1.))
            angle = math.degrees(math.atan2(sine, cosine))
            relative = length/maximum if maximum > 0 else 0.
            reasons = []
            if relative <= NUMERICAL_TOLERANCE:
                reasons.append('coincident or numerically near-coincident estimated centers')
            if sine <= NUMERICAL_TOLERANCE:
                reasons.append('numerically parallel or antiparallel measured world rays')
            pair = dict(image_ids=[obs[i]['image_id'], obs[j]['image_id']],
                        baseline_length=length, relative_center_separation=relative,
                        ray_separation_degrees=angle,
                        triangulation_angle_degrees=min(angle, 180.-angle),
                        sin_ray_angle=sine, score=None if reasons else sine,
                        status='EXCLUDED' if reasons else 'RANKED',
                        reason='; '.join(reasons) if reasons else None, rank=None)
            result['pairs'].append(pair)
            if not reasons:
                eligible.append(pair)
        eligible.sort(key=lambda p: (-p['score'], p['image_ids']))
        for rank, pair in enumerate(eligible, 1):
            pair['rank'] = rank
        if eligible:
            best = eligible[0]
            result.update(status='SELECTED', selected_image_ids=best['image_ids'].copy(),
                          selected_score=best['score'],
                          selected_ray_separation_degrees=best['ray_separation_degrees'],
                          selected_triangulation_angle_degrees=best['triangulation_angle_degrees'])
        else:
            result['reason'] = 'no pair survives numerical center and ray degeneracy checks'
        json.dumps(result, allow_nan=False)
    except (ValueError, ArithmeticError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        result.update(status='REFUSED', reason=str(exc), selected_image_ids=None)
    return result
