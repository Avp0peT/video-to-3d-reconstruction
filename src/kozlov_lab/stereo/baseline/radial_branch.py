"""Explicit central-branch hypothesis; never an inferred physical calibration."""
import math

import numpy as np

from . import radial_diagnostics as radial


DERIVATIVE_FLOOR = 1e-6


def propose_central_domain(params, width, height, raw_pixels):
    """Return JSON data for a fixed branch and separate image/observation coverage.

    Negative k uses R=sqrt((1-floor)/(-3k)), independent of observed points or
    residuals. The complete raw rectangle [0,width] x [0,height] is tested by
    its corners. Its coverage never certifies the actual camera or inverse root.
    The inherited bounded inverse and full forward roundtrip remain unchanged.
    """
    for name, value in (('width', width), ('height', height)):
        if type(value) is not int or not 1 <= value <= 16384:
            raise ValueError('bounded positive integer image '+name+' required')
    camera = radial._camera(params, 2)
    pixels = radial._pixels(raw_pixels)
    f, cx, cy, k = map(float, camera)
    radius = turn = branch_max = turning_max = derivative = None
    if k < 0:
        # Divide in two steps: -3*k can overflow even for a finite coefficient.
        turn_squared = (-1. / k) / 3.
        if not math.isfinite(turn_squared) or turn_squared <= 0:
            raise ValueError('central branch radius outside finite arithmetic domain')
        turn = math.sqrt(turn_squared)
        radius = turn * math.sqrt(1. - DERIVATIVE_FLOOR)
        derivative = 1. + 3. * (k * radius) * radius
        branch_max = radius * (1. + (k * radius) * radius)
        turning_max = 2. * turn / 3.
        if not all(math.isfinite(v) and v > 0 for v in (radius, derivative, branch_max, turning_max)):
            raise ValueError('invalid finite central branch domain')
        radial._domain(k, radius)
    corners = [(0., 0.), (float(width), 0.), (0., float(height)), (float(width), float(height))]
    corner_radii = [math.hypot((x-cx)/f, (y-cy)/f) for x, y in corners]
    if not all(math.isfinite(v) for v in corner_radii):
        raise ValueError('normalized image radius overflow')
    maximum_raw = max(corner_radii)
    covered = k >= 0 or maximum_raw < branch_max
    rows = []
    for raw in pixels:
        x, y = map(float, raw)
        rd = math.hypot((x-cx)/f, (y-cy)/f)
        if not math.isfinite(rd):
            raise ValueError('normalized observation radius overflow')
        row = dict(raw_pixels=[x, y], distorted_radius=rd,
                   inside_raw_image=0 <= x <= width and 0 <= y <= height,
                   ideal_pixels=None, forward_residual_pixels=None,
                   status='INVERSE_REFUSED', reason=None)
        if k < 0 and rd >= branch_max:
            row['reason'] = 'NEAR_TURN_BOUNDARY' if rd <= turning_max else 'RAW_RADIUS_OUTSIDE_CENTRAL_BRANCH_IMAGE'
        else:
            try:
                inverse = radial.undistort_simple_radial([[x, y]], camera, max_ideal_radius=radius)
                ideal = inverse['ideal_pixels'][0]
                ideal_radius = math.hypot((float(ideal[0])-cx)/f, (float(ideal[1])-cy)/f)
                row.update(status='INVERSE_EVALUATED', ideal_pixels=ideal.tolist(),
                           ideal_radius=ideal_radius,
                           radial_derivative=1. + 3. * (k * ideal_radius) * ideal_radius,
                           forward_residual_pixels=float(inverse['forward_residual_pixels'][0]))
            except (ValueError, ArithmeticError, OverflowError) as exc:
                row['reason'] = type(exc).__name__ + ': ' + str(exc)
        rows.append(row)
    all_ok = all(row['status'] == 'INVERSE_EVALUATED' for row in rows)
    return dict(schema='central_branch_hypothesis-v1',
                status='CONDITIONAL_INVERSES_EVALUATED' if all_ok else 'OBSERVATIONS_PARTIALLY_REFUSED',
                params=camera.tolist(), width=width, height=height,
                branch_selection='ASSUMED_CENTRAL_BRANCH' if k < 0 else 'NONNEGATIVE_K_GLOBAL_MONOTONIC_MODEL',
                derivative_floor=DERIVATIVE_FLOOR, max_ideal_radius=radius,
                turning_ideal_radius=turn, turning_distorted_radius=turning_max,
                domain_radial_derivative=derivative,
                full_image_coverage=dict(covered=bool(covered), max_distorted_radius=maximum_raw,
                    branch_max_distorted_radius=branch_max, corner_distorted_radii=corner_radii,
                    scope='closed raw image rectangle; mathematical coverage conditional on the selected branch'),
                per_point=rows, all_observations_invertible=all_ok,
                physical_inverse_unique=False, identity_verified=False, geometry_acceptance=False,
                native_injected=False,
                limitations=['Negative-k central root is an explicit assumption, not inferred from roundtrip.',
                    'Global image coverage and individual observation invertibility are separate.',
                    'Existing camera parameters are not independently validated by this diagnostic.'])
