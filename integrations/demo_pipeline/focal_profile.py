"""Bounded necessary essential consistency; neither calibration nor validation.

Formula source: Hartley & Zisserman, Multiple View Geometry, section 9.6,
Result 9.17, https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf
The fixed numerical flags below describe this grid, not statistical confidence.
"""
import numbers

import numpy as np


def _array(value, shape):
    if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
        raise ValueError('Boolean entries are not numeric geometry inputs')
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in 'iuf':
        raise ValueError('Expected finite real numeric array with shape %s' % (shape,))
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite input')
    return result


def profile_fundamental(F, principal1, principal2, image_long_edge, *, config=3):
    """Profile a pixel-coordinate F under fixed principal points and shared f.

    config 2/3 admits a fundamental matrix, but does not establish nonplanarity.
    No E projection, optimization, correspondence fitting or held-out test occurs.
    """
    out = dict(status='FOCAL_PROFILE_REFUSED', reason_code=None,
               calibration_accepted=False, independent_truth=False,
               new_ba_authorized=False, curve=[], minima=[],
               degeneracy_status='NOT_EVALUATED_FROM_F_ALONE',
               score_semantics='Dimensionless necessary consistency; not pixel error or confidence',
               assumptions=['fixed principal points', 'square pixels', 'zero skew',
                            'shared focal length', 'zero distortion assumed'],
               policy=dict(grid_count=101, focal_over_long_edge=[0.1, 10.0],
                           rank_relative_tolerance=1e-8,
                           flat_absolute_range_tolerance=1e-12,
                           flat_relative_range_tolerance=1e-6))
    if isinstance(config, (bool, np.bool_)) or not isinstance(config, numbers.Integral) or config not in (2, 3):
        out.update(reason_code='CONFIG_AMBIGUOUS', reason='Only fundamental configurations 2 and 3 are admitted')
        return out
    try:
        matrix = _array(F, (3, 3))
        p1, p2 = _array(principal1, (2,)), _array(principal2, (2,))
        if isinstance(image_long_edge, (bool, np.bool_)) or not isinstance(image_long_edge, numbers.Real):
            raise ValueError('image_long_edge must be a positive finite real scalar')
        edge = float(image_long_edge)
        if not np.isfinite(edge) or edge <= 0:
            raise ValueError('image_long_edge must be positive and finite')
        scale = float(np.max(np.abs(matrix)))
        if scale == 0:
            out.update(reason_code='ZERO_F', reason='Zero F has no essential profile')
            return out
        matrix = matrix / scale
        matrix = matrix / np.linalg.norm(matrix)
        fs = np.linalg.svd(matrix, compute_uv=False)
        out['fundamental_singular_values_normalized'] = fs.tolist()
        out['rank_diagnostics'] = dict(second_over_first=float(fs[1]/fs[0]),
                                       third_over_first=float(fs[2]/fs[0]),
                                       tolerance=1e-8, rank2_projected=False)
        if fs[1]/fs[0] <= 1e-8 or fs[2]/fs[0] > 1e-8:
            out.update(reason_code='RANK_NOT_TWO', reason='F fails the fixed numerical rank-two gate')
            return out
        for index, ratio in enumerate(np.logspace(-1, 1, 101)):
            f = float(edge * ratio)
            k1 = np.array([[f, 0, p1[0]], [0, f, p1[1]], [0, 0, 1.]])
            k2 = np.array([[f, 0, p2[0]], [0, f, p2[1]], [0, 0, 1.]])
            E = k2.T @ matrix @ k1
            if not np.isfinite(E).all():
                raise ValueError('Non-finite essential matrix in fixed grid')
            s = np.linalg.svd(E, compute_uv=False)
            if not np.isfinite(s).all() or s[0] <= 0:
                raise ValueError('Invalid essential singular values')
            # These are singular values of raw E, before any essential projection.
            sn = s / s[0]
            q = float(((sn[0]-sn[1])**2/2 + sn[2]**2) / np.dot(sn, sn))
            out['curve'].append(dict(index=index, focal=f, focal_over_long_edge=float(ratio),
                                     singular_values=s.tolist(), qess=q,
                                     qequal=float((sn[0]-sn[1])/(sn[0]+sn[1])),
                                     qrank=float(sn[2]/np.linalg.norm(sn[:2]))))
        scores = np.array([row['qess'] for row in out['curve']])
        span = float(np.ptp(scores))
        flat_limit = max(1e-12, 1e-6 * float(np.max(scores)))
        flat = bool(span <= flat_limit)
        # Include all weak grid minima, including plateau points and endpoints.
        indices = [i for i in range(101)
                   if (i == 0 or scores[i] <= scores[i-1])
                   and (i == 100 or scores[i] <= scores[i+1])]
        out['minima'] = [dict(out['curve'][i], boundary=bool(i in (0, 100))) for i in indices]
        best = int(np.argmin(scores))
        boundary = bool(best in (0, 100))
        out.update(status='FOCAL_PROFILE_EVALUATED', reason_code=None,
                   focal_observability_status=('FLAT_PROFILE' if flat else 'BOUNDARY_MINIMUM' if boundary
                                               else 'MULTIPLE_MINIMA' if len(indices) > 1
                                               else 'CONDITIONAL_INTERIOR_MINIMUM'),
                   flat=flat, boundary_minimum=boundary, score_range=span,
                   flat_limit=flat_limit, minimum_grid_index=best,
                   minimum_qess=float(scores[best]),
                   boundary_scores=[float(scores[0]), float(scores[-1])],
                   rank_score_independent_evidence=False)
        return out
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as exc:
        out.update(reason_code='INVALID_INPUT', reason=str(exc), curve=[], minima=[])
        return out
