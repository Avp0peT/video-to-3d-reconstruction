"""Measured adjacent-frame texture patches; candidate identity, never truth."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from integrations.demo_pipeline.photometric_tracks import _normalize, PhotometricProfile
from integrations.demo_pipeline.temporal_review import load_cv

PROFILE = dict(id='glove-region-patch-v1', patch=21, max_active=600,
               pyramid_max_level=3, fb_max=1.5, zncc_min=.8,
               mask_support_min=.85, texture_std_min=2., grid=8,
               min_distance=7., cpu_threads=2)


def support(mask, xy, cv):
    """Require finite full patch, target centre and actual mask support."""
    x, y = map(float, xy)
    r = PROFILE['patch']//2
    if not np.isfinite([x, y]).all() or not (r <= x <= mask.shape[1]-1-r and r <= y <= mask.shape[0]-1-r):
        return False
    if not mask[int(np.floor(y+.5)), int(np.floor(x+.5))]:
        return False
    patch = cv.getRectSubPix(mask, (21, 21), (x, y))
    return float(patch.mean()/255.) >= PROFILE['mask_support_min']


def step(prev, gray, previous_mask, mask, points, cv, identity_policy=None):
    """LK proposes; raw full-patch ZNCC and masks independently admit."""
    if identity_policy is not None:
        from .track_identity import validate_policy, isolate_targets
        validate_policy(identity_policy)
    p = np.asarray(points, np.float32).reshape(-1, 2)
    reasons = ['source_mask_or_bounds']*len(p)
    q = p.copy()
    scores = [None]*len(p)
    ids = [i for i, xy in enumerate(p) if support(previous_mask, xy, cv)]
    if not ids:
        return q, reasons, scores
    a = _normalize(prev, cv, PhotometricProfile())
    b = _normalize(gray, cv, PhotometricProfile())
    opts = dict(winSize=(21, 21), maxLevel=3,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, .01), minEigThreshold=1e-4)
    flow, status, _ = cv.calcOpticalFlowPyrLK(a, b, p[ids, None], None, **opts)
    good = []
    for j, i in enumerate(ids):
        reasons[i] = 'forward_lk'
        if flow is not None and status is not None and status[j, 0] and np.isfinite(flow[j, 0]).all():
            q[i] = flow[j, 0]
            reasons[i] = 'target_mask_or_bounds'
            if support(mask, q[i], cv):
                good.append(i)
    if not good:
        return q, reasons, scores
    back, status, _ = cv.calcOpticalFlowPyrLK(b, a, q[good, None], None, **opts)
    for j, i in enumerate(good):
        reasons[i] = 'forward_backward'
        if back is None or status is None or not status[j, 0] or not np.isfinite(back[j]).all() or np.linalg.norm(back[j, 0]-p[i]) > 1.5:
            continue
        x = cv.getRectSubPix(prev, (21, 21), tuple(map(float, p[i]))).astype(float)
        y = cv.getRectSubPix(gray, (21, 21), tuple(map(float, q[i]))).astype(float)
        reasons[i] = 'patch_texture'
        if min(x.std(), y.std()) < 2.:
            continue
        x -= x.mean(); y -= y.mean()
        score = float(np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y)))
        reasons[i] = 'patch_zncc'
        if np.isfinite(score):
            scores[i] = score
            if score >= .8:
                reasons[i] = 'accepted'
    if identity_policy is not None:
        reasons = isolate_targets(q, reasons, identity_policy)
    return q, reasons, scores


def seed(gray, mask, existing, cv):
    """Spatially stratified measured corners; no blank-region artificial points."""
    candidates = []
    h, w = gray.shape
    for gy in range(8):
        for gx in range(8):
            y0, y1 = gy*h//8, (gy+1)*h//8
            x0, x1 = gx*w//8, (gx+1)*w//8
            if y1-y0 < 3 or x1-x0 < 3:
                candidates.append([])
                continue
            corners = cv.goodFeaturesToTrack(gray[y0:y1, x0:x1], 12, .01, 7,
                                             mask=mask[y0:y1, x0:x1])
            candidates.append([] if corners is None else (corners[:, 0]+[x0, y0]).tolist())
    picked = np.asarray(existing, dtype=np.float32).reshape(-1, 2)
    births = []
    for rank in range(12):
        for cell in candidates:
            if rank >= len(cell) or len(picked) >= 600:
                continue
            p = np.asarray(cell[rank], np.float32)
            if support(mask, p, cv) and (not len(picked) or np.all(np.sum((picked-p)**2, axis=1) >= 49)):
                patch = cv.getRectSubPix(gray, (21, 21), tuple(map(float, p)))
                if patch.std() >= 2.:
                    picked = np.vstack((picked, p)); births.append(p)
    return births


def run(plan_path, output):
    """Write a new REGION_TRACKS.json using only two adjacent image arrays."""
    plan_path = Path(plan_path).resolve(strict=True)
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    if plan.get('schema') != 'glove-region-plan-v1' and plan.get('contract_version') != 'glove-region-plan-v1':
        raise ValueError('glove-region-plan-v1 required')
    if 'profile' in plan and plan['profile'] != PROFILE:
        raise ValueError('frozen profile differs')
    density_policy = plan.get('density_policy')
    identity_policy = plan.get('identity_policy')
    if identity_policy is not None:
        from .track_identity import validate_policy as validate_identity
        validate_identity(identity_policy)
    if density_policy is not None:
        from .density_sampling import validate_policy, seed_with_density
        validate_policy(density_policy)
    timeline = plan['timeline']
    if not timeline or any(type(r['decode_index']) is not int or r['decode_index'] < 0 for r in timeline) or any(a['decode_index'] >= b['decode_index'] for a, b in zip(timeline, timeline[1:])):
        raise ValueError('strictly ordered decode indices required')
    endpoints = [r['input_index'] for r in timeline if r.get('input_index') is not None]
    if any(type(v) is not int or v < 0 for v in endpoints) or any(a >= b for a, b in zip(endpoints, endpoints[1:])):
        raise ValueError('nonnegative strictly ordered integer input_index required')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('new output directory required')
    output.mkdir(parents=True, exist_ok=False)
    progress = output/'progress.jsonl'
    def record(value):
        with progress.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, allow_nan=False)+'\n')
    record(dict(event='START', timeline_frames=len(timeline), endpoint_frames=len(endpoints)))
    refs = [dict(path=str(plan_path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))]
    if identity_policy is not None:
        path=Path(__file__).with_name('track_identity.py').resolve()
        blob=path.read_bytes()
        refs.append(dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
    if density_policy is not None:
        for name in ('density_sampling.py', 'point_density.py'):
            path=Path(__file__).with_name(name).resolve()
            blob=path.read_bytes()
            refs.append(dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
    cv = load_cv(refs)
    if cv is None:
        raise RuntimeError('bound CPU OpenCV unavailable')
    cv.ocl.setUseOpenCL(False); cv.setNumThreads(2); cv.setRNGSeed(0)
    frames, tracks, stats = [], {}, []
    active = {}; next_id = 0; previous = previous_mask = None
    shape = None
    for row in timeline:
        arrays = []
        bound = dict(row)
        for kind in ('image', 'mask'):
            path = Path(row[kind+'_path']).resolve(strict=True)
            blob = path.read_bytes(); digest = hashlib.sha256(blob).hexdigest()
            if digest != row[kind+'_sha256']:
                raise ValueError('SHA256 mismatch: '+str(path))
            refs.append(dict(path=str(path), sha256=digest, bytes=len(blob)))
            with Image.open(io.BytesIO(blob)) as im:
                if kind == 'mask':
                    if im.mode != 'L':
                        raise ValueError('mask must be mode L labels 0/255')
                    array = np.asarray(im).copy()
                    if not np.isin(array, [0, 255]).all():
                        raise ValueError('mask must contain only labels 0/255')
                else:
                    if im.mode not in ('RGB', 'L'):
                        raise ValueError('RGB or grayscale image required')
                    array = np.asarray(im.convert('L')).copy()
                arrays.append(array)
            bound[kind+'_path'] = str(path)
        gray, mask = arrays
        if gray.shape != mask.shape or gray.shape != (row['height'], row['width']) or (shape is not None and gray.shape != shape):
            raise ValueError('same grid image and mask required')
        shape = gray.shape
        attempted = len(active); rejected = Counter(); lost = []; births = []; approved_scores = []; conflicts = []
        if previous is not None and active:
            ids = list(active)
            if identity_policy is None:
                q, reasons, scores = step(previous, gray, previous_mask, mask, list(active.values()), cv)
            else:
                q, reasons, scores = step(previous, gray, previous_mask, mask, list(active.values()), cv, identity_policy)
            for i, ident in enumerate(ids):
                if reasons[i] == 'accepted':
                    active[ident] = q[i]
                    approved_scores.append(scores[i])
                else:
                    del active[ident]; rejected[reasons[i]] += 1
                    lost.append(dict(track_id=ident, reason=reasons[i]))
                    if reasons[i] == 'target_identity_ambiguous':
                        conflicts.append(dict(track_id=ident, proposed_xy=[float(v) for v in q[i]],
                                              status='NOT_RESOLVED', action='END_ID_NO_OBSERVATION'))
        density = None
        if density_policy is not None:
            new_points, density = seed_with_density(gray, mask, list(active.values()), cv, density_policy)
        elif previous is None or row.get('input_index') is not None:
            new_points = seed(gray, mask, list(active.values()), cv)
        else:
            new_points = []
        for xy in new_points:
            active[next_id] = xy; births.append(next_id)
            tracks[next_id] = dict(track_id=next_id, observations=[])
            next_id += 1
        observations = []
        for ident, xy in active.items():
            obs = dict(track_id=ident, xy=[float(v) for v in xy])
            observations.append(obs)
            if row.get('input_index') is not None:
                tracks[ident]['observations'].append(dict(input_index=row['input_index'], decode_index=row['decode_index'], xy=obs['xy']))
        if row.get('input_index') is not None:
            frames.append({**{k: bound.get(k) for k in ('input_index','decode_index','image_path','image_sha256','mask_path','mask_sha256','width','height')}, 'observations': observations, 'candidate_density': density})
        stats.append(dict(decode_index=row['decode_index'], attempted=attempted, accepted=attempted-len(lost), births=births, lost=lost, reject_reasons=dict(rejected), active=len(active), approved_zncc_min=min(approved_scores) if approved_scores else None, approved_zncc_mean=float(np.mean(approved_scores)) if approved_scores else None))
        if identity_policy is not None:
            stats[-1]['identity_conflicts'] = conflicts
        if density is not None:
            stats[-1]['candidate_density'] = {k: density[k] for k in ('status', 'met_area_fraction', 'largest_deficient_region_fraction', 'unique_in_mask_points', 'active_budget_exhausted')}
        record(dict(event='FRAME', **stats[-1]))
        previous, previous_mask = gray, mask
    result = dict(schema='glove-region-tracks-v1', contract_version='glove-region-tracks-v1',
                  coordinate_convention='opencv_integer_centers', profile=PROFILE,
                  identity_verified=False, geometry_accepted=False, frames=frames,
                  identity_limitations='Adjacent patch admission does not verify long-term physical identity; gradual drift remains possible.',
                  tracks=list(tracks.values()), timeline_stats=stats, source_bindings=refs,
                  metrics=dict(frames=len(frames), timeline_frames=len(timeline), endpoints=len(endpoints), tracks=len(tracks), observations=sum(len(f['observations']) for f in frames)))
    result['density_policy'] = density_policy
    result['identity_policy'] = identity_policy
    destination = output/'REGION_TRACKS.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    record(dict(event='FINAL', status='REGION_TRACKS_MEASURED_NOT_GEOMETRY_ACCEPTED', metrics=result['metrics']))
    return result
