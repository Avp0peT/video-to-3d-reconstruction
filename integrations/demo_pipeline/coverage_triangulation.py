"""All-observation DLT candidates; conditional support never proves calibration."""
import numpy as np

from .partial_point_gate import assess_point


def triangulate_candidate(observations):
    """Keep every supplied observation and return finite JSON-compatible evidence."""
    out = dict(xyz=None, tier='REJECTED', reasons=[], strict_decision=None,
               relaxed_decision=None, metrics=dict(observation_count=0, loo_evaluated=False))
    def reject(reason):
        if reason not in out['reasons']:
            out['reasons'].append(reason)
    if not isinstance(observations, (list, tuple)) or not 2 <= len(observations) <= 32:
        reject('OBSERVATION_COUNT_OUT_OF_BOUNDS'); return out
    out['metrics']['observation_count'] = len(observations)
    cameras = []; ids = set(); rows = []
    def array(value, shape):
        a = np.asarray(value, dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError('invalid numeric array')
        return a
    try:
        for observation in observations:
            iid, row = observation['image_id'], observation['feature_row']
            if type(iid) is not int or type(row) is not int or min(iid, row) < 0:
                reject('INVALID_IDENTITY')
            if iid in ids: reject('DUPLICATE_IMAGE_ID')
            ids.add(iid)
            r = array(observation['R'], (3, 3)); t = array(observation['t'], (3,))
            params = array(observation['params'], (4,)); pixels = array(observation['raw_pixels'], (2,))
            if observation.get('camera_model', 'SIMPLE_RADIAL') != 'SIMPLE_RADIAL' or params[0] <= 0 or params[3] != 0:
                reject('UNSUPPORTED_INTRINSICS')
            if not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-7) or not np.isclose(np.linalg.det(r), 1, rtol=0, atol=1e-7):
                reject('INVALID_ROTATION')
            if observation['foreground'] is not True: reject('NON_FOREGROUND_OBSERVATION')
            cameras.append((r, t, params, pixels))
        if out['reasons']: return out
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            for r, t, params, pixels in cameras:
                u, v = (pixels - params[1:3]) / params[0]
                p = np.column_stack((r, t))
                rows.extend([u * p[2] - p[0], v * p[2] - p[1]])
        a = np.asarray(rows)
        if not np.isfinite(a).all(): raise ValueError('nonfinite DLT')
        _, singular, vh = np.linalg.svd(a, full_matrices=False)
        if singular[-2] <= 1e-12 * singular[0] or abs(vh[-1, 3]) <= 1e-12:
            reject('DEGENERATE_TRIANGULATION'); return out
        xyz = vh[-1, :3] / vh[-1, 3]
        if not np.isfinite(xyz).all(): raise ValueError('nonfinite point')
        rays = []; jacobians = []; depths = []; errors = []
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            for r, t, params, pixels in cameras:
                local = r @ xyz + t; depth = local[2]
                error = np.linalg.norm(params[0] * local[:2] / depth + params[1:3] - pixels)
                if not np.isfinite(depth) or depth <= 0: reject('NON_POSITIVE_DEPTH')
                if not np.isfinite(error): reject('NONFINITE_PROJECTION')
                depths.append(float(depth) if np.isfinite(depth) else None)
                errors.append(float(error) if np.isfinite(error) else None)
                ray = r.T @ np.r_[(pixels - params[1:3]) / params[0], 1.]
                norm = np.linalg.norm(ray)
                if not np.isfinite(norm) or norm <= 0: raise ValueError('invalid ray')
                rays.append(ray / norm)
                jacobians.extend(params[0] / depth * (r[:2] - local[:2, None] / depth * r[2]))
        angle = max(float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
                    for i, a in enumerate(rays) for b in rays[i+1:])
        if angle < 1.: reject('INSUFFICIENT_TRIANGULATION_ANGLE')
        jacobian = np.asarray(jacobians)
        if not np.isfinite(jacobian).all(): raise ValueError('invalid Jacobian')
        s = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.count_nonzero(s > s[0] * 1e-12))
        condition = float(s[0] / s[-1]) if s[-1] > 0 else None
        if condition is not None and not np.isfinite(condition): condition = None
        if rank < 3 or condition is None or condition > 1e4: reject('ILL_CONDITIONED_POINT')
        out['metrics'].update(depths=depths, reprojection_errors_px=errors,
            measurement_ray_angle_deg=angle, jacobian_rank=rank, jacobian_condition=condition)
        if len(cameras) == 2:
            if any(e is None or e > 4. for e in errors): reject('REPROJECTION_ERROR_EXCEEDED')
            if not out['reasons']: out.update(xyz=xyz.tolist(), tier='TWO_VIEW_PROVISIONAL')
        else:
            strict = assess_point(xyz, observations)
            relaxed = assess_point(xyz, observations, max_error_px=4., min_angle_deg=1., max_loo_error_px=6.)
            out.update(strict_decision=strict, relaxed_decision=relaxed)
            out['metrics']['loo_evaluated'] = True
            if not out['reasons']:
                if strict['status'] == 'CONDITIONAL_POINT_SUPPORTED':
                    out.update(xyz=xyz.tolist(), tier='STRICT_SUPPORTED')
                elif relaxed['status'] == 'CONDITIONAL_POINT_SUPPORTED':
                    out.update(xyz=xyz.tolist(), tier='RELAXED_SUPPORTED')
                else:
                    for reason in relaxed['reasons']: reject(reason)
    except (KeyError, ValueError, TypeError, OverflowError, np.linalg.LinAlgError, FloatingPointError):
        reject('INVALID_OBSERVATION_OR_NUMERICS')
    return out
