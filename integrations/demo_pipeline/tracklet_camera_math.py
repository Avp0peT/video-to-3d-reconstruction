"""Bounded projective resection diagnostics, without calibration or pose claims.

RMS normalization is a numerical conditioning measure following the standard
Hartley/Zisserman tutorial (CVPR 1999, tutoriala.pdf, pp. 3--4). This is an
engineering diagnostic, not a Kozlov construction or an accepted camera.
"""
import math

MAX_POINTS = 768
RIGHTS = dict(geometry_accepted=False, camera_pose_established=False)


class _Degenerate(Exception):
    pass


def assess_projection(points, *, budget=lambda: None):
    """Fit P only. Frozen holdout and shifted correspondence control use no K.

    Invalid types and identities raise ValueError. Numerical degeneracy is a
    BLOCKED_DEGENERATE result. Budget exceptions propagate unchanged.
    """
    import numpy as np

    if not isinstance(points, list) or len(points) > MAX_POINTS:
        raise ValueError('POINTS_LIST_BOUND_768')
    seen = set(); checked = []
    for point in points:
        budget()
        if not isinstance(point, dict): raise ValueError('INVALID_POINT_RECORD')
        pid = point.get('point_id')
        if type(pid) is not str or not pid or pid in seen:
            raise ValueError('INVALID_OR_DUPLICATE_POINT_ID')
        seen.add(pid)
        arrays = []
        for key, size in (('xyz', 3), ('raw_xy', 2)):
            value = point.get(key)
            try:
                if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
                    raise ValueError('BOOLEAN_GEOMETRY_INPUT')
                raw = np.asarray(value)
                if raw.shape != (size,) or raw.dtype.kind not in 'iuf':
                    raise ValueError('INVALID_NUMERIC_SHAPE_OR_TYPE')
                array = raw.astype(float, copy=True)
                if not np.isfinite(array).all(): raise ValueError('NONFINITE_GEOMETRY_INPUT')
            except (TypeError, OverflowError) as exc:
                raise ValueError('INVALID_GEOMETRY_INPUT') from exc
            arrays.append(array)
        checked.append((pid, arrays[0], arrays[1]))
    checked.sort(key=lambda p: p[0])
    ids = [p[0] for p in checked]; count = len(ids)
    hold = [i for i in range(count) if i % 4 == 3] if count >= 9 else []
    hold_set = set(hold)
    train = [i for i in range(count) if i not in hold_set]
    result = dict(status='BLOCKED_INSUFFICIENT_POINTS', count=count,
        train_ids=[ids[i] for i in train], holdout_ids=[ids[i] for i in hold],
        all_fit=None, heldout_fit=None, negative_control=None, numeric_fits=0, reasons=[],
        normalization='independent 2D and 3D centroid, RMS radius sqrt(dimension)',
        split='lexicographic point_id order; every fourth point at indices 3,7,... held out when count>=9',
        negative_control_definition='complete raw_xy array cyclic shift +1 before same frozen split',
        denominator_semantics='homogeneous projective denominator; sign consistency is not physical cheirality',
        max_points=MAX_POINTS, **RIGHTS)
    if count < 6:
        result['reasons'].append('AT_LEAST_SIX_2D_3D_CORRESPONDENCES_REQUIRED')
        return result
    X = np.stack([p[1] for p in checked]); image = np.stack([p[2] for p in checked])

    def svd(value, *, vectors=False):
        budget()
        try:
            answer = np.linalg.svd(value, full_matrices=False) if vectors else np.linalg.svd(value, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            raise _Degenerate('SVD_DID_NOT_CONVERGE') from exc
        budget()
        return answer

    def normalize(value):
        dimension = value.shape[1]; mean = value.mean(axis=0); centered = value-mean
        if not np.isfinite(centered).all(): raise _Degenerate('NORMALIZATION_OVERFLOW')
        singular = svd(centered)
        if len(singular) < dimension or singular[0] <= 0 or singular[dimension-1] <= singular[0]*1e-8:
            raise _Degenerate(f'RANK_DEFICIENT_{dimension}D_SUPPORT')
        largest = float(np.max(np.abs(centered)))
        rms = largest*float(np.sqrt(np.mean(np.sum((centered/largest)**2,axis=1))))
        if not math.isfinite(rms) or rms <= 0: raise _Degenerate('INVALID_NORMALIZATION_RADIUS')
        scale = math.sqrt(dimension)/rms
        transform = np.eye(dimension+1); transform[:dimension,:dimension] *= scale
        transform[:dimension,-1] = -mean*scale
        normalized = centered*scale
        if not np.isfinite(transform).all() or not np.isfinite(normalized).all():
            raise _Degenerate('NORMALIZATION_OVERFLOW')
        return normalized, transform, singular.tolist()

    def evaluate(P, indices, coordinates):
        rows = []; signs = []
        for index in indices:
            budget()
            homogeneous = np.append(X[index],1.)
            projected = P @ homogeneous
            scale = float(np.linalg.norm(P[2])*np.linalg.norm(homogeneous))
            if not np.isfinite(projected).all() or not math.isfinite(scale) or scale <= 0:
                raise _Degenerate('NONFINITE_PROJECTION')
            denominator = float(projected[2])
            if abs(denominator) <= 1e-12*scale:
                raise _Degenerate('UNSTABLE_PROJECTIVE_DENOMINATOR')
            predicted = projected[:2]/denominator
            error = float(np.linalg.norm(predicted-coordinates[index]))
            if not np.isfinite(predicted).all() or not math.isfinite(error):
                raise _Degenerate('NONFINITE_REPROJECTION')
            signs.append(1 if denominator > 0 else -1)
            rows.append(dict(point_id=ids[index],predicted_raw_xy=predicted.tolist(),
                             error_px=error,denominator=denominator))
        errors = [r['error_px'] for r in rows]
        return dict(rows=rows,count=len(rows),median_px=float(np.median(errors)) if errors else None,
                    max_px=max(errors) if errors else None,
                    denominator_sign_consistent=len(set(signs)) <= 1,
                    denominator_signs=signs)

    def fit(indices, coordinates):
        budget()
        xn, world_transform, world_singular = normalize(X[indices])
        yn, image_transform, image_singular = normalize(coordinates[indices])
        h = np.column_stack((xn,np.ones(len(indices)))); zero = np.zeros_like(h)
        design = np.stack((np.column_stack((h,zero,-yn[:,0,None]*h)),
                           np.column_stack((zero,h,-yn[:,1,None]*h))),axis=1).reshape(-1,12)
        _, singular, vectors = svd(design,vectors=True)
        # The least-squares design can have rank 12 in noisy data. Only the
        # penultimate singular value controls identifiability of the null direction.
        if not np.isfinite(singular).all() or singular[0] <= 0 or singular[-2] <= singular[0]*1e-10:
            raise _Degenerate('DESIGN_LACKS_ELEVEN_DIMENSIONAL_SUPPORT')
        normalized_P = vectors[-1].reshape(3,4)
        P = np.linalg.solve(image_transform, normalized_P @ world_transform)
        norm = float(np.linalg.norm(P))
        if not np.isfinite(P).all() or not math.isfinite(norm) or norm <= 0:
            raise _Degenerate('NONFINITE_PROJECTIVE_CAMERA')
        P /= norm
        if P.flat[int(np.argmax(np.abs(P)))] < 0: P *= -1
        epsilon = float(np.finfo(float).eps*max(1.,singular[0]))
        evidence = dict(P=P.tolist(),training_ids=[ids[i] for i in indices],
            design_singular_values=singular.tolist(),
            identifiability_condition=float(singular[0]/singular[-2]),
            solution_separation=float(singular[-2]/max(float(singular[-1]),epsilon)),
            separation_epsilon=epsilon, world_centered_singular_values=world_singular,
            image_centered_singular_values=image_singular)
        result['numeric_fits'] += 1
        return P, evidence

    stage = 'all_fit'
    try:
        P, evidence = fit(list(range(count)),image)
        evidence['reprojection'] = evaluate(P,list(range(count)),image)
        result['all_fit'] = evidence
        if count <= 8:
            result['status'] = 'MINIMAL_FIT_NOT_VALIDATED'
            result['reasons'].append('NO_FROZEN_HOLDOUT_FOR_SIX_TO_EIGHT_POINTS')
            return result
        stage = 'heldout_fit'
        P, evidence = fit(train,image)
        evidence['train_reprojection'] = evaluate(P,train,image)
        evidence['holdout_reprojection'] = evaluate(P,hold,image)
        result['heldout_fit'] = evidence
        stage = 'negative_control'
        shifted = np.roll(image,1,axis=0)
        try:
            P, evidence = fit(train,shifted)
            evidence['train_reprojection'] = evaluate(P,train,shifted)
            evidence['holdout_reprojection'] = evaluate(P,hold,shifted)
            evidence['status'] = 'EVALUATED'
            evidence['raw_xy_source_ids'] = [ids[(i-1) % count] for i in range(count)]
            result['negative_control'] = evidence
        except _Degenerate as exc:
            result['negative_control'] = dict(status='BLOCKED_DEGENERATE',reasons=[str(exc)])
        result['status'] = 'PROJECTIVE_FIT_DIAGNOSTIC'
    except _Degenerate as exc:
        result['status'] = 'BLOCKED_DEGENERATE'
        result['reasons'].append(stage+':'+str(exc))
    return result
