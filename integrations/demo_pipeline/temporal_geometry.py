"""Bounded temporal correspondence diagnostics; no component merge authority."""
import itertools
import math
import numbers
import numpy as np

PROFILE = dict(max_frames=96, selected_frames=24, duplicate_endpoint_px=2., min_candidates=24,
               holdout_stride=3, min_holdout=8, ransac_px=1.5, confidence=.99,
               max_iterations=2000, seed=0, holdout_median_px=1.5, holdout_p90_px=3.)


def _finite_xy(xy):
    if len(xy) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) for v in xy):
        raise ValueError('invalid canonical xy')
    result = np.asarray(xy, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite canonical xy')
    return result


def _metric(values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    return dict(count=len(a), finite_count=len(finite), median=float(np.median(finite)) if len(finite) else None,
                p90=float(np.percentile(finite, 90)) if len(finite) else None,
                max=float(finite.max()) if len(finite) else None,
                values=[float(v) if math.isfinite(v) else None for v in a])


def _sampson(F, a, b):
    x = np.column_stack((a, np.ones(len(a))))
    y = np.column_stack((b, np.ones(len(b))))
    fx, fy = x @ F.T, y @ F
    den = fx[:, :2] ** 2 + fy[:, :2] ** 2
    total = den.sum(axis=1)
    result = np.full(len(a), np.inf)
    good = total > np.finfo(float).eps
    result[good] = np.abs(np.sum(y[good] * fx[good], axis=1)) / np.sqrt(total[good])
    return result


def _bbox(points, pixels):
    p = np.asarray(points)
    span = p.max(axis=0) - p.min(axis=0)
    return dict(min=p.min(axis=0).tolist(), max=p.max(axis=0).tolist(), bbox_area=float(span.prod()),
                bbox_area_over_mask_pixels=float(span.prod() / pixels) if pixels > 0 else None,
                note='bbox/mask area reference only; not spatial mask intersection or coverage truth')


def analyse_track_pairs(tracks, frames, cv):
    cv.setNumThreads(2)
    cv.ocl.setUseOpenCL(False)
    if len(frames) > PROFILE['max_frames']:
        raise ValueError('frame budget exceeded')
    eligible = [i for i, f in enumerate(frames) if f.get('mask_pixels', 0) > 0]
    selected = [eligible[i] for i in np.linspace(0, len(eligible) - 1, min(len(eligible), 24), dtype=int)] if eligible else []
    selected_set = set(selected)
    observations = []
    for ordinal, track in enumerate(tracks):
        obs = {}
        for o in track.get('observations', []):
            frame = o['frame']
            if type(frame) is not int or not 0 <= frame < len(frames):
                raise ValueError('invalid track frame')
            if o.get('target_verified') is not True or frame not in selected_set:
                continue
            if frame in obs:
                raise ValueError('duplicate observation frame in track')
            obs[frame] = _finite_xy(o['xy'])
        observations.append((str(track['track_id']), ordinal, obs))
    observations.sort(key=lambda v: (v[0], v[1]))
    pairs, bridges = [], []
    for first, second in itertools.combinations(selected, 2):
        candidates = []
        for ident, ordinal, obs in observations:
            if first not in obs or second not in obs:
                continue
            a, b = obs[first], obs[second]
            existing = next((r for r in candidates if np.linalg.norm(a-r['_a']) <= 2 and np.linalg.norm(b-r['_b']) <= 2), None)
            alias = dict(track_id=ident, source_track_ordinal=ordinal)
            if existing is not None:
                existing['aliases'].append(alias)
            else:
                candidates.append(dict(stable_id=f'{ident}@{ordinal}', original_track_id=ident, aliases=[alias], _a=a, _b=b))
        row = dict(frames=[first, second], input_indices=[frames[first]['input_index'], frames[second]['input_index']],
                   unique_candidates=len(candidates), duplicate_alias_count=sum(len(r['aliases'])-1 for r in candidates),
                   status='INSUFFICIENT_UNIQUE_CANDIDATES', reason='requires at least 24 endpoint-deduplicated candidates',
                   components_merged=False, stitch_authorized=False)
        row['candidates'] = [dict(stable_id=r['stable_id'], original_track_id=r['original_track_id'], aliases=r['aliases'], xy_first=r['_a'].tolist(), xy_second=r['_b'].tolist()) for r in candidates]
        if len(candidates) < 24:
            pairs.append(row)
            continue
        hold = list(range(0, len(candidates), 3))
        train = [i for i in range(len(candidates)) if i not in set(hold)]
        a = np.asarray([r['_a'] for r in candidates]); b = np.asarray([r['_b'] for r in candidates])
        row.update(train_ids=[candidates[i]['stable_id'] for i in train], holdout_ids=[candidates[i]['stable_id'] for i in hold],
                   split_rule='frozen sorted stable IDs: every third candidate held out; no reselection',
                   distributions=[_bbox(a, frames[first]['mask_pixels']), _bbox(b, frames[second]['mask_pixels'])])
        if min(np.linalg.matrix_rank(a[train]-a[train].mean(axis=0)), np.linalg.matrix_rank(b[train]-b[train].mean(axis=0))) < 2:
            row.update(status='DEGENERATE_COORDINATES', reason='training coordinates are collinear or repeated')
            pairs.append(row); continue
        try:
            cv.setRNGSeed(0)
            H, hmask = cv.findHomography(a[train], b[train], cv.RANSAC, 1.5, maxIters=2000, confidence=.99)
            hvalid = H is not None and np.asarray(H).shape == (3, 3) and np.isfinite(H).all() and np.linalg.matrix_rank(H) == 3
            row['H'] = H.tolist() if hvalid else None
            row['H_status'] = 'FINITE_FULL_RANK' if hvalid else 'DEGENERATE_OR_UNAVAILABLE'
            row['planar_or_low_parallax_ambiguity'] = None
            if row['H'] is not None:
                projected = np.column_stack((a[hold], np.ones(len(hold)))) @ H.T
                valid = np.abs(projected[:, 2]) > np.finfo(float).eps
                error = np.full(len(hold), np.inf)
                error[valid] = np.linalg.norm(projected[valid,:2]/projected[valid,2,None]-b[hold][valid], axis=1)
                hm = _metric(error);row['holdout_H_forward_px'] = hm
                row['planar_or_low_parallax_ambiguity'] = hm['finite_count'] == len(hold) and hm['median'] <= 1.5 and hm['p90'] <= 3
            cv.setRNGSeed(0)
            F, inliers = cv.findFundamentalMat(a[train], b[train], cv.FM_RANSAC, 1.5, .99, 2000)
            if F is None or np.asarray(F).shape != (3, 3) or not np.isfinite(F).all() or np.linalg.norm(F) == 0:
                raise ValueError('fundamental fit unavailable or nonfinite')
            row['F'] = F.tolist(); row['train_inlier_count'] = int(np.asarray(inliers).sum()) if inliers is not None else None
            metric = _metric(_sampson(F, a[hold], b[hold]));row['holdout_sampson_px'] = metric
            good = metric['finite_count'] == len(hold) and metric['median'] <= 1.5 and metric['p90'] <= 3
            row.update(status='CANDIDATE_SUPPORT_NOT_VERIFIED' if good else 'HOLDOUT_RESIDUAL_LARGE', reason='held-out Sampson diagnostic; not identity or geometry acceptance' if good else 'holdout median >1.5px, p90 >3px, or invalid projection')
            if good and frames[first].get('model_ids') and frames[second].get('model_ids') and not set(frames[first]['model_ids']) & set(frames[second]['model_ids']):
                bridges.append(dict(frames=[first, second], model_ids=[frames[first]['model_ids'], frames[second]['model_ids']], status=row['status'], stitch_authorized=False))
        except Exception as exc:
            row.update(status='FIT_REFUSED', reason=str(exc))
        pairs.append(row)
    triples = []
    for triple in itertools.combinations(selected, 3):
        unique = []
        for ident, ordinal, obs in observations:
            if all(i in obs for i in triple):
                points = [obs[i] for i in triple]
                if not any(all(np.linalg.norm(a-b) <= 2 for a,b in zip(points, old)) for old in unique):
                    unique.append(points)
        if unique:
            times = [frames[i].get('time_seconds', frames[i].get('video_time_seconds')) for i in triple]
            span = max(times)-min(times) if all(isinstance(t,(int,float)) and math.isfinite(t) for t in times) else None
            triples.append(dict(frames=list(triple), common_unique_target_verified_tracks=len(unique), time_span_seconds=span, status='TEMPORAL_CANDIDATE_NOT_GEOMETRY_ACCEPTANCE'))
    triples.sort(key=lambda r:(-r['common_unique_target_verified_tracks'], -(r['time_span_seconds'] or 0), r['frames']))
    return dict(schema='temporal-geometry-diagnostic-v1', profile=PROFILE, selected_frames=selected,
                omitted_mask_eligible_frames=[i for i in eligible if i not in selected_set], omitted_unverified_mask_frames=[i for i in range(len(frames)) if i not in eligible], pairs=pairs, proposed_bridges=bridges, top_three_frame_candidates=triples[:10], components_merged=False, stitch_authorized=False,
                limitations=['Canonical xy supplied with +0.5 pixel convention; no coordinate manufacture.', 'Training RANSAC inliers are not held-out validation.', 'H support diagnoses planar/low-parallax ambiguity, not nonrigidity or actual camera angle.', 'All support is candidate only, not quality truth or stitching authority.'])
