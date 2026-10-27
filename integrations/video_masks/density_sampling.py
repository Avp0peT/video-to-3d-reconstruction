"""Bounded measured-corner refill; density targets never certify matches."""
from dataclasses import asdict

import numpy as np

from .point_density import DensityPolicy, assess_density


def default_policy():
    return dict(schema='object-point-density-v1', coverage=asdict(DensityPolicy()),
                quality_levels=[0.01, 0.003], candidates_per_cell=32,
                max_active=600, min_distance=7.0, refill_each_frame=True)


def validate_policy(value):
    # Freeze the first production profile; future tuning needs a new version.
    if value != default_policy():
        raise ValueError('unsupported or changed object density policy')
    return DensityPolicy(**value['coverage'])


def seed_with_density(gray, mask, existing, cv, policy=None):
    from .region_tracks import support
    policy = default_policy() if policy is None else policy
    coverage = validate_policy(policy)
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.shape != mask.shape:
        raise ValueError('same-grid uint8 gray and mask required')
    picked = np.asarray(existing, dtype=np.float32).reshape(-1, 2)
    initial = assess_density(mask, picked, coverage)
    if len(picked) > policy['max_active']:
        raise ValueError('existing active points exceed frozen cap')
    births = []
    pools = {}
    counts = {c['cell_id']: c['point_count'] for c in initial['cells']}
    attempted_levels = []
    h, w = gray.shape

    def accept(p):
        nonlocal picked
        p = np.asarray(p, np.float32)
        if len(picked) >= policy['max_active'] or not support(mask, p, cv):
            return False
        if len(picked) and np.any(np.sum((picked-p)**2, axis=1) < policy['min_distance']**2):
            return False
        patch = cv.getRectSubPix(gray, (21, 21), tuple(map(float, p)))
        if patch.std() < 2.0:
            return False
        picked = np.vstack((picked, p)); births.append(p)
        return True

    for quality in policy['quality_levels']:
        deficient = [c for c in initial['cells'] if counts[c['cell_id']] < c['target_points']]
        if not deficient or len(picked) >= policy['max_active']:
            break
        attempted_levels.append(quality)
        for cell in deficient:
            x0, y0, x1, y1 = cell['bounds']
            # A context halo avoids inventing a feature boundary at the grid edge.
            left, top, right, bottom = max(0, x0-10), max(0, y0-10), min(w, x1+10), min(h, y1+10)
            local_mask = mask[top:bottom, left:right].copy()
            yy, xx = np.indices(local_mask.shape)
            local_mask[(xx+left < x0) | (xx+left >= x1) | (yy+top < y0) | (yy+top >= y1)] = 0
            if min(local_mask.shape) < 3:
                pools[cell['cell_id']] = []
                continue
            corners = cv.goodFeaturesToTrack(gray[top:bottom, left:right],
                policy['candidates_per_cell'], quality, policy['min_distance'], mask=local_mask)
            pools[cell['cell_id']] = [] if corners is None else (corners[:, 0]+[left, top]).tolist()
        # One candidate per cell per round: texture-rich cells cannot take the floor budget first.
        for rank in range(policy['candidates_per_cell']):
            for cell in deficient:
                ident = cell['cell_id']; pool = pools[ident]
                if counts[ident] >= cell['target_points'] or rank >= len(pool):
                    continue
                if accept(pool[rank]):
                    counts[ident] += 1
    # The floor is a priority, not a new low cap. Preserve useful extra real features.
    for rank in range(policy['candidates_per_cell']):
        for cell in initial['cells']:
            pool = pools.get(cell['cell_id'], [])
            if rank < len(pool):
                accept(pool[rank])
    final = assess_density(mask, picked, coverage)
    final.update(evidence_kind='MEASURED_2D_CANDIDATES_NOT_VERIFIED_CORRESPONDENCES',
                 births=len(births), attempted_quality_levels=attempted_levels,
                 active_budget_exhausted=len(picked) >= policy['max_active'],
                 unresolved_reason='No admitted measured corner or active budget exhausted; no synthetic fill.'
                     if final['status'] != 'DENSITY_TARGET_MET' else None,
                 geometry_accepted=False)
    return births, final
