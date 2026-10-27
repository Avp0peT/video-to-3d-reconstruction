"""Experimental inferred regional planes; shared SfM depths are not ground truth."""
import cv2
import numpy as np

from .local_depth_search import _inputs, _policy, _rays, _validate, _project

SCALES = (32, 64, 96)
REASONS = {0: 'OUTSIDE_ROI', 1: 'NO_BRACKETED_PLANE', 2: 'COMPETING_PLANES_UNKNOWN',
           3: 'INSUFFICIENT_GEOMETRIC_SUPPORT', 4: 'GEOMETRIC_CONFLICT', 5: 'ACCEPTED_INFERRED_PLANE', 6: 'KNOWN_BACKGROUND_VETO'}


def _fit(x, y, z, cx, cy, scale):
    """Fixed deterministic Huber IRLS, followed by explicit depth residual gates."""
    design = np.column_stack(((x-cx)/scale, (y-cy)/scale, np.ones(len(x))))
    if len(x) < 12 or np.linalg.matrix_rank(design) < 3:
        return None
    bins = ((x >= cx).astype(int) + 2*(y >= cy).astype(int))
    if len(np.unique(bins)) < 4:
        return None
    singular = np.linalg.svd(design, compute_uv=False)
    hull0 = cv2.convexHull(np.column_stack((x,y)).astype(np.float32))
    if singular[-1] < .02 or singular[0]/singular[-1] > 100 or cv2.contourArea(hull0)/(scale*scale) < .02:
        return None
    weights = np.ones(len(x))
    for _ in range(8):
        coef = np.linalg.lstsq(design*weights[:, None]**.5, (1/z)*weights**.5, rcond=None)[0]
        inv = design@coef
        residual = np.abs(inv*z-1)
        weights = np.minimum(1., .005/np.maximum(residual, 1e-12))
    good = (inv > 0) & (np.abs(1/np.maximum(inv, 1e-12)-z)/z <= .01)
    if good.sum() < 12 or len(np.unique(bins[good])) < 4 or np.linalg.matrix_rank(design[good]) < 3:
        return None
    coef = np.linalg.lstsq(design[good], 1/z[good], rcond=None)[0]
    predicted = design[good]@coef
    residual = np.abs(1/np.maximum(predicted, 1e-12)-z[good])/z[good]
    if (predicted <= 0).any() or (residual > .01).any():
        return None
    hull = cv2.convexHull(np.column_stack((x[good], y[good])).astype(np.float32))
    singular = np.linalg.svd(design[good], compute_uv=False)
    if singular[-1] < .02 or singular[0]/singular[-1] > 100 or cv2.contourArea(hull)/(scale*scale) < .02:
        return None
    return coef, hull, int(good.sum()), float(residual.max())


def _surface_sets(anchors, depth, check):
    labels = np.zeros(anchors.shape, np.int32)
    h,w = anchors.shape
    number = 0
    for y,x in zip(*np.nonzero(anchors)):
        if labels[y,x]:
            continue
        check()
        number += 1
        labels[y,x] = number
        stack = [(y,x)]
        popped = 0
        while stack:
            a,b = stack.pop()
            popped += 1
            if popped % 256 == 0:
                check()
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                c,d = a+dy,b+dx
                if 0 <= c < h and 0 <= d < w and anchors[c,d] and not labels[c,d] and abs(float(depth[c,d])-float(depth[a,b])) <= .02*min(float(depth[c,d]),float(depth[a,b])):
                    labels[c,d] = number
                    stack.append((c,d))
    return labels


def _hull_inside(hull, component, left, top):
    raster = np.zeros(component.shape, np.uint8)
    polygon = np.rint(hull[:,0]-[left,top]).astype(np.int32)
    cv2.fillConvexPoly(raster, polygon, 1)
    return not np.any((raster != 0) & ~component)


def _outcomes(ref, sources, depth, roi, policy, check):
    names = ('support','occluded','free_space_conflict','unknown','out_of_view','background')
    counts = {name+'_count': np.zeros(roi.shape,np.uint8) for name in names}
    y,x = np.nonzero(roi & (depth > 0))
    rays = _rays(ref,x,y)
    stats = {}
    for source in sources:
        check()
        uv,z,pos = _project(ref,source,rays,depth[y,x])
        h,w = source['mask'].shape
        inside = pos & (uv[:,0]>=0) & (uv[:,0]<=w-1) & (uv[:,1]>=0) & (uv[:,1]<=h-1)
        ix = np.rint(np.clip(uv[:,0],0,w-1)).astype(int)
        iy = np.rint(np.clip(uv[:,1],0,h-1)).astype(int)
        valid = inside & source['validity'][iy,ix]
        background = valid & ~source['mask'][iy,ix]
        known = valid & source['mask'][iy,ix] & (source['depth'][iy,ix]>0)
        ds = source['depth'][iy,ix]
        checked = _validate(ref,[source],depth,roi,policy,check)
        support = checked['support_count'][y,x]>0
        # Depth ordering distinguishes likely occlusion from free-space conflict;
        # reprojection-only disagreement remains unknown, never a pass.
        occluded = known & ~support & (ds < z*(1-policy['geometry_relative_tolerance']))
        free = known & ~support & (ds > z*(1+policy['geometry_relative_tolerance']))
        out = ~inside
        unknown = ~(support|occluded|free|out|background)
        flags = dict(support=support,occluded=occluded,free_space_conflict=free,unknown=unknown,out_of_view=out,background=background)
        for name,flag in flags.items():
            counts[name+'_count'][y,x] += flag
        available = int(known.sum())
        stats[str(source['view_id'])] = dict(**{name:int(flag.sum()) for name,flag in flags.items()}, available_depth_pixels=available, status='EVALUATED' if available else 'NOT_EVALUABLE')
    return dict(**counts, outcomes_per_source=stats)


def propose_regional_depth(reference, sources, roi, depth_bounds, *, policy=None, check=lambda: None):
    """Infer bracketed planes from qualified WHOLE anchors, then check source depths.

    Final source is held out only from this proposal-stage selection/admission;
    historical input depths may already have used that source. All returned
    depth arrays contain proposals only. Input depths are never modified.
    """
    p = _policy(policy)
    ref, src, bbox = _inputs(reference, sources, roi, check)
    if len(src) < 4:
        raise ValueError('four to five sources required including final heldout source')
    if not isinstance(depth_bounds, (list, tuple)) or len(depth_bounds) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in depth_bounds) or depth_bounds[0] >= depth_bounds[1]:
        raise ValueError('finite positive increasing axial depth bounds required')
    low, high = depth_bounds
    if low < np.finfo(np.float32).tiny or high > np.finfo(np.float32).max:
        raise ValueError('bounds exceed float32 depth range')
    for v in [ref, *src]:
        validity = v.get("validity", np.ones(v["mask"].shape, bool))
        if not isinstance(validity, np.ndarray) or validity.dtype != bool or validity.shape != v["mask"].shape:
            raise ValueError("invalid validity mask")
        v["validity"] = validity
        v["mask"] = v["mask"] & validity
    train, heldout = src[:-1], src[-1:]
    shape = roi.shape
    eroded = cv2.erode(ref['mask'].astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    context = np.zeros(shape, bool)
    yy, xx = np.nonzero(roi)
    if len(xx):
        context[max(0, yy.min()-96):min(shape[0], yy.max()+97), max(0, xx.min()-96):min(shape[1], xx.max()+97)] = True
    anchor_roi = context & eroded & (ref['depth'] > 0)
    prior = _validate(ref, train, ref['depth'], anchor_roi, p, check)
    anchors = anchor_roi & (prior['support_count'] >= 2) & (prior['conflict_count'] < 2)
    # Observed depth-connected support sets; proposals are never anchors.
    surface = _surface_sets(anchors, ref['depth'], check)
    raw = np.zeros(shape, np.float32)
    chosen = np.zeros(shape, np.uint8)
    count = np.zeros(shape, np.uint16)
    minimum = np.full(shape, np.inf)
    maximum = np.zeros(shape)
    quality = np.full(shape, -np.inf)
    fits = []
    target = roi & eroded
    for scale in SCALES:
        for top in range(0, shape[0], scale//2):
            for left in range(0, shape[1], scale//2):
                check()
                bottom, right = min(top+scale, shape[0]), min(left+scale, shape[1])
                region = np.s_[top:bottom, left:right]
                if not target[region].any():
                    continue
                _, labels = cv2.connectedComponents(eroded[region].astype(np.uint8), connectivity=8)
                # Intersect foreground components with each observed surface set.
                pairs = [(int(l), int(s)) for l in np.unique(labels[target[region]]) if l for s in np.unique(surface[region][(labels == l) & anchors[region]]) if s]
                for label, surface_id in pairs:
                    if label == 0:
                        continue
                    component = labels == label
                    ay, ax = np.nonzero(anchors[region] & component & (surface[region] == surface_id))
                    ax, ay = ax+left, ay+top
                    cx, cy = left+scale/2, top+scale/2
                    fit = _fit(ax, ay, ref['depth'][ay, ax].astype(float), cx, cy, scale)
                    if fit is None:
                        continue
                    coef, hull, support, residual = fit
                    if not _hull_inside(hull, component, left, top):
                        continue
                    ty, tx = np.nonzero(target[region] & component)
                    tx, ty = tx+left, ty+top
                    inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in zip(tx, ty)])
                    tx, ty = tx[inside], ty[inside]
                    inv = np.column_stack(((tx-cx)/scale, (ty-cy)/scale, np.ones(len(tx))))@coef
                    z = 1/np.maximum(inv, 1e-30)
                    good = (inv > 0) & (z >= low) & (z <= high)
                    tx, ty, z = tx[good], ty[good], z[good]
                    minimum[ty, tx] = np.minimum(minimum[ty, tx], z)
                    maximum[ty, tx] = np.maximum(maximum[ty, tx], z)
                    count[ty, tx] += 1
                    score = support-residual
                    take = score > quality[ty, tx]
                    raw[ty[take], tx[take]] = z[take]
                    quality[ty[take], tx[take]] = score
                    chosen[ty[take], tx[take]] = scale
                    fits.append(dict(scale=scale, window=[left, top, right, bottom], component=int(label), surface_support_set=surface_id, normalized_hull_area=float(cv2.contourArea(hull)/(scale*scale)), anchors=support, max_relative_residual=residual, inverse_depth_coefficients=coef.tolist(), center=[cx, cy], hull=hull[:, 0].tolist(), bracketed_candidates=len(tx)))
    competing = (count > 1) & ((maximum-minimum) > .02*np.maximum(minimum, 1e-12))
    candidate = _validate(ref, train, raw, roi, p, check)
    outcomes = _outcomes(ref, train, raw, roi, p, check)
    background_veto = outcomes['background_count'] > 0
    accepted = ~background_veto & target & (raw > 0) & ~competing & (candidate['support_count'] >= 2) & (candidate['conflict_count'] < 2)
    reason = np.zeros(shape, np.uint8)
    reason[roi] = 1
    reason[target & (raw > 0)] = 3
    reason[target & (raw > 0) & (candidate['conflict_count'] >= 2)] = 4
    reason[competing & roi] = 2
    reason[background_veto & roi & (raw > 0)] = 6
    reason[accepted] = 5
    proposal = np.where(accepted, raw, 0).astype(np.float32)
    held = _validate(ref, heldout, proposal, accepted, p, check)
    held_outcomes = _outcomes(ref, heldout, proposal, accepted, p, check)
    held.update(held_outcomes)
    regional_policy = dict(window_sizes=list(SCALES), stride_fraction=.5,
        min_anchors=12, min_quadrants=4, max_relative_fit_residual=.01,
        adjacent_relative_depth=.02, max_design_condition=100.,
        min_design_singular_value=.02, min_normalized_hull_area=.02,
        reference_mask_erosion_pixels=1, overlap_relative_disagreement=.02,
        geometry_relative_tolerance=p['geometry_relative_tolerance'],
        max_reprojection_error=p['max_reprojection_error'],
        min_admission_source_support=2, max_admission_conflicts_exclusive=2,
        known_background_veto=True, heldout_sources=1, min_sources=4,
        max_sources=5, irls_iterations=8, irls_huber_relative_scale=.005,
        anchor_context_margin_pixels=96, convex_hull_requires_component_containment=True,
        observed_depth_connectivity=8)
    return dict(background_veto=background_veto, surface_support_sets=surface, admission_outcomes=outcomes, heldout_scope='Proposal/admission stage only; historical depths may use this view', scale_semantics='Multiple support window sizes, not an ACMM image pyramid', proposal_depth=proposal, raw_best_depth=raw, accepted=accepted, reason=reason, reason_legend=REASONS.copy(), anchor_mask=anchors, prior_used=anchors, chosen_scale=chosen, hypothesis_count=count, competing_planes=competing, per_scale_fits={str(s): [f for f in fits if f['scale'] == s] for s in SCALES}, validation=candidate, prior_validation=prior, heldout_validation=held, heldout_source_ids=held['source_ids'], source_ids=candidate['source_ids'], support_count=candidate['support_count'], conflict_count=candidate['conflict_count'], unknown_count=candidate['unknown_count'], best_photo_cost=np.full(shape, np.inf, np.float32), uniqueness_gap=np.full(shape, -np.inf, np.float32), photo_support_bits=np.zeros(shape, np.uint8), photo_support_count=np.zeros(shape, np.uint8), joint_support_count=candidate['support_count'], policy=regional_policy, policy_version='regional-inferred-plane-v2', proposal_kind='INFERRED_PLANE_NOT_PHOTOMETRIC_OBSERVATION', geometry_accepted=False, bounds_semantics='Experimental axial depth; physical scale unknown', counters=dict(roi_pixels=int(roi.sum()), accepted_pixels=int(accepted.sum()), anchor_pixels=int(anchors.sum()), fits=len(fits), competing_pixels=int((competing & roi).sum()), roi_bbox=list(bbox)))
