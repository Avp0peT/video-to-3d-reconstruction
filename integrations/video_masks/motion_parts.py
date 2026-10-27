"""Conservative image-motion candidates; never a physical or geometry acceptance gate."""
from collections import deque
import math
import cv2
import numpy as np

_DEFAULTS = dict(max_frames=6, max_points=1200, max_parts=4, grid_size=6,
                 min_tracks=8, min_pair_support=2, fb_error=1.0,
                 ransac_error=1.5, relative_motion=3.0, propagation_fraction=0.08,
                 random_seed=0)


def _policy(policy):
    p = dict(_DEFAULTS)
    if policy is not None:
        if not isinstance(policy, dict) or set(policy) - set(p):
            raise ValueError('policy must be a dict containing recognized keys')
        p.update(policy)
    for k, v in p.items():
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f'invalid policy value: {k}')
        if k in ('max_frames', 'max_points', 'max_parts', 'grid_size', 'min_tracks', 'min_pair_support', 'random_seed'):
            if int(v) != v or v < (0 if k == 'random_seed' else 1):
                raise ValueError(f'invalid integer policy: {k}')
            p[k] = int(v)
        elif v <= 0:
            raise ValueError(f'policy must be positive: {k}')
    if p['max_frames'] > 6 or p['max_points'] > 1200 or p['max_parts'] > 4 or p['min_tracks'] < 3 or p['min_pair_support'] < 2 or p['propagation_fraction'] > 0.25:
        raise ValueError('policy exceeds conservative bounds')
    return p


def _domain(mask, seeds, radius):
    """Geodesic four-neighbor propagation cannot jump background or holes."""
    owner = np.full(mask.shape, -1, np.int32)
    q = deque()
    h, w = mask.shape
    for x, y, label in seeds:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < w and 0 <= y < h and mask[y, x] and owner[y, x] < 0:
            owner[y, x] = label
            q.append((x, y, 0))
    while q:
        x, y, d = q.popleft()
        if d >= radius:
            continue
        for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0 <= nx < w and 0 <= ny < h and mask[ny,nx] and owner[ny,nx] < 0:
                owner[ny,nx] = owner[y,x]
                q.append((nx,ny,d+1))
    return owner


def _affine(a, b, p):
    if len(a) < p['min_tracks']:
        return None, np.zeros(len(a), bool)
    model, inliers = cv2.estimateAffine2D(a, b, method=cv2.RANSAC,
        ransacReprojThreshold=p['ransac_error'], maxIters=1000, confidence=0.99, refineIters=10)
    if model is None:
        return None, np.zeros(len(a), bool)
    return model, inliers.ravel().astype(bool)


def propose_motion_parts(images, masks, *, policy=None):
    """Return bounded temporal motion evidence, with arrays owned by the caller.

    RGB uint8 images and boolean masks must have identical dimensions. Local
    candidates require shared track IDs across at least two adjacent pairs.
    Unknown pixels can overlap fallback: preserved segmentation is not motion
    evidence. All thresholds and the temporary confirmation gate are recorded.
    """
    p = _policy(policy)
    if not isinstance(images, list) or not isinstance(masks, list) or not images or len(images) != len(masks) or len(images) > p['max_frames']:
        raise ValueError('provide equally sized nonempty lists within max_frames')
    shape = None
    for im, mask in zip(images, masks):
        if not isinstance(im, np.ndarray) or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3 or min(im.shape[:2]) < 8:
            raise ValueError('images must be uint8 RGB arrays at least 8x8')
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != im.shape[:2]:
            raise ValueError('masks must be matching boolean arrays')
        if shape is not None and im.shape != shape:
            raise ValueError('all images must have identical shape')
        shape = im.shape
    cv2.setRNGSeed(p['random_seed'] % 2147483647)
    gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images]
    h, w = masks[0].shape
    points = []
    g = p['grid_size']
    # Uniform grid quotas prevent one richly textured object taking every seed.
    quota = max(1, p['max_points'] // (g*g))
    for gy in range(g):
        for gx in range(g):
            y0,y1 = gy*h//g,(gy+1)*h//g
            x0,x1 = gx*w//g,(gx+1)*w//g
            if min(y1-y0,x1-x0)<3:
                continue
            corners = cv2.goodFeaturesToTrack(gray[0][y0:y1,x0:x1], mask=(masks[0][y0:y1,x0:x1].astype(np.uint8)*255), maxCorners=quota, qualityLevel=0.02, minDistance=4, blockSize=3)
            if corners is not None:
                points.extend(corners.reshape(-1,2) + [x0,y0])
    first = np.asarray(points[:p['max_points']], np.float32).reshape(-1,2)
    tracks = [first]
    valid = [np.ones(len(first),bool)]
    groups, pairs = [], []
    for frame in range(len(images)-1):
        ids = np.flatnonzero(valid[-1])
        nxt = np.full(first.shape, np.nan, np.float32)
        okfull = np.zeros(len(first),bool)
        errors = []
        if len(ids):
            a = tracks[-1][ids].reshape(-1,1,2)
            b, st, _ = cv2.calcOpticalFlowPyrLK(gray[frame], gray[frame+1], a, None, winSize=(21,21), maxLevel=3)
            back, bst, _ = cv2.calcOpticalFlowPyrLK(gray[frame+1], gray[frame], b, None, winSize=(21,21), maxLevel=3)
            err = np.linalg.norm(back.reshape(-1,2)-a.reshape(-1,2),axis=1)
            xy = b.reshape(-1,2)
            good = st.ravel().astype(bool)&bst.ravel().astype(bool)&np.isfinite(xy).all(axis=1)&(err<=p['fb_error'])&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            rounded=np.rint(np.nan_to_num(xy,nan=-1.0,posinf=-1.0,neginf=-1.0)).astype(int)
            good &= masks[frame+1][rounded[:,1].clip(0,h-1),rounded[:,0].clip(0,w-1)]
            ids = ids[good]
            nxt[ids] = xy[good]
            okfull[ids] = True
            errors = err[good].tolist()
        tracks.append(nxt)
        valid.append(okfull)
        a,b = tracks[frame][ids],nxt[ids]
        model, inliers = _affine(a,b,p)
        record = dict(pair=[frame,frame+1], tracked_count=len(ids), track_ids=[int(v) for v in ids], source_points=a.tolist(), destination_points=b.tolist(), global_model=None if model is None else model.tolist(), fb_error=dict(count=len(errors), median=float(np.median(errors)) if errors else None, maximum=max(errors) if errors else None), local_models=[])
        if model is not None:
            pred = a @ model[:,:2].T + model[:,2]
            residual = np.linalg.norm(b-pred,axis=1)
            inside = np.zeros(len(ids),bool)
            if len(ids):
                aa=np.rint(a).astype(int); bb=np.rint(b).astype(int)
                inside=masks[frame][aa[:,1].clip(0,h-1),aa[:,0].clip(0,w-1)] & masks[frame+1][bb[:,1].clip(0,h-1),bb[:,0].clip(0,w-1)]
            remaining = np.flatnonzero(inside & (residual>=p['relative_motion']))
            for _ in range(p['max_parts']):
                # A disconnected set can accidentally share a fitted scaling affine.
                # First require coherent residual vectors, then fit its affine.
                if len(remaining) < p['min_tracks']:
                    break
                vectors = (b-pred)[remaining]
                counts = [int((np.linalg.norm(vectors-v,axis=1) <= p['ransac_error']).sum()) for v in vectors]
                coherent = np.linalg.norm(vectors-vectors[int(np.argmax(counts))],axis=1) <= p['ransac_error']
                selected = remaining[coherent]
                lm, local_inliers = _affine(a[selected],b[selected],p)
                li = np.zeros(len(remaining),bool)
                li[np.flatnonzero(coherent)[local_inliers]] = True
                if lm is None or li.sum()<p['min_tracks']:
                    break
                chosen=remaining[li]
                tids=set(int(v) for v in ids[chosen])
                groups.append(dict(pair=frame, ids=tids, model=lm.tolist()))
                record['local_models'].append(dict(track_ids=sorted(tids), model=lm.tolist(), relative_motion_median=float(np.median(residual[chosen]))))
                remaining=remaining[~li]
        pairs.append(record)
    # Merge only candidates with substantial actual shared track identity.
    clusters=[]
    for group in groups:
        matches=[c for c in clusters if len(c['anchor_ids'] & group['ids']) >= p['min_tracks'] and len(c['anchor_ids'] & group['ids']) / min(len(c['anchor_ids']),len(group['ids'])) >= 0.5]
        if len(matches)>1:
            for c in matches: c['ambiguous']=True
        elif matches:
            c=matches[0]
            if group['pair'] in c['pairs']:
                c['ambiguous']=True
            else:
                c['ids'] |= group['ids']; c['pairs'].add(group['pair'])
        else:
            clusters.append(dict(ids=set(group['ids']),anchor_ids=set(group['ids']),pairs={group['pair']},ambiguous=False))
    confirmed=[c for c in clusters if len(c['pairs'])>=p['min_pair_support'] and not c['ambiguous']]
    confirmed.sort(key=lambda c:(-len(c['pairs']),-len(c['ids']),min(c['ids'])))
    confirmed=confirmed[:p['max_parts']]
    names=['motion_%04d'%min(c['ids']) for c in confirmed]
    parts={name:[] for name in names}
    fallback=[]; unknown=[]; domains=[]; metrics=[]
    radius=max(1,int(round(min(h,w)*p['propagation_fraction'])))
    for f, mask in enumerate(masks):
        seeds=[]
        observed = valid[f] if f>0 else (valid[1] if len(valid)>1 else np.zeros(len(first),bool))
        for tid in np.flatnonzero(observed):
            label=next((i for i,c in enumerate(confirmed) if int(tid) in c['ids']),len(confirmed))
            x,y=tracks[f][tid]; seeds.append((float(x),float(y),label))
        owner=_domain(mask,seeds,radius)
        domain=owner>=0
        assigned=np.zeros(mask.shape,bool)
        for i,name in enumerate(names):
            pm=owner==i; parts[name].append(pm); assigned |= pm
        fallback.append(mask & ~assigned); unknown.append(mask & ~domain); domains.append(domain)
        metrics.append(dict(frame=f,global_pixels=int(mask.sum()),part_pixels=int(assigned.sum()),fallback_pixels=int(fallback[-1].sum()),unknown_pixels=int(unknown[-1].sum()),tracked_seed_count=int(valid[f].sum())))
    status='MOTION_CANDIDATES_NOT_PHYSICAL_PARTS' if confirmed else ('NOT_TEMPORALLY_CONFIRMED' if groups else 'NO_RELATIVE_MOTION_EVIDENCE')
    ledger=dict(policy=p,threshold_units='native image pixels; resolution dependent',geometry_accepted=False,interpretation='Image-motion candidates only; propagated seed evidence is not a part contour. Dominant image motion is not camera motion. Perspective and parallax can split rigid objects; no anatomical or physical acceptance.',confirmation_gate='Shared track IDs across adjacent pairs; temporary conservative heuristic.',seed_count=len(first),propagation_radius=radius,pairs=pairs,parts=[dict(id=n,track_ids=sorted(c['ids']),support_pairs=sorted(c['pairs'])) for n,c in zip(names,confirmed)],rejections=[dict(track_ids=sorted(c['ids']),reason='AMBIGUOUS_TRACK_ASSOCIATION' if c['ambiguous'] else 'INSUFFICIENT_TEMPORAL_SUPPORT',support_pairs=sorted(c['pairs'])) for c in clusters if len(c['pairs'])<p['min_pair_support'] or c['ambiguous']])
    return dict(status=status,geometry_accepted=False,global_masks=[m.copy() for m in masks],part_masks=parts,fallback_masks=fallback,unknown_masks=unknown,evidence_domain_masks=domains,metrics=metrics,ledger=ledger)
