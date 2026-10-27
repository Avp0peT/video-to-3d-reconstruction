"""Exact minimum enclosing ellipse ONLY for complete triangular-hull images.

Stdlib rational proof; no numerical solver or ostov integration.
"""
from fractions import Fraction as F
from .independent_cover import normalize, hull, orient, Budget, BudgetExceeded
import hashlib
import json
import math


def _digest(raw):
    rows = [[i, [[str(v.numerator), str(v.denominator)] for v in raw[i]]]
            for i in sorted(raw)]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def _input(points):
    labels = sorted(points) if isinstance(points, dict) and all(type(i) is int for i in points) else []
    return normalize(points, [labels])[0]


def _q(p, c, matrix):
    x, y = p[0]-c[0], p[1]-c[1]
    return matrix[0][0]*x*x+2*matrix[0][1]*x*y+matrix[1][1]*y*y


def exact_triangle_enclosing_ellipse(points, *, max_work=2000000, seconds=30):
    """All supplied distinct labelled points; Fraction/int/finite binary64.

    Cooperative work/time bounds, not a hard wall-clock supervisor.
    """
    result = dict(schema='exact-triangle-ellipse-v1', status='UNSUPPORTED_INPUT',
                  exact_optimum=False, complete=False, work_used=0)
    if (type(max_work) is not int or not 0 <= max_work <= 2000000 or
            type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 30):
        return dict(result, reason='invalid bounded budget')
    budget = Budget(max_work, seconds)
    try:
        budget.spend()
        raw = _input(points)
        budget.spend(len(raw))
        result.update(declared_labels=sorted(raw), source_coordinate_sha256=_digest(raw))
        vertices = hull(sorted(raw), raw, budget)
        result['hull_labels'] = vertices
        if len(vertices) != 3:
            return dict(result, status='UNSUPPORTED_DOMAIN', reason='complete input hull must have exactly three extreme points', work_used=budget.used)
        c = tuple(sum(raw[i][k] for i in vertices)/3 for k in (0, 1))
        d = [tuple(raw[i][k]-c[k] for k in (0, 1)) for i in vertices]
        shape = tuple(tuple(F(2, 3)*sum(p[j]*p[k] for p in d) for k in (0, 1)) for j in (0, 1))
        a, b, e = shape[0][0], shape[0][1], shape[1][1]
        determinant = a*e-b*b
        inverse = ((e/determinant, -b/determinant), (-b/determinant, a/determinant))
        barycentric = {}
        u, v, w = [raw[i] for i in vertices]
        area2 = orient(u, v, w)
        for ident, p in raw.items():
            budget.spend(10)
            barycentric[ident] = (orient(v,w,p)/area2, orient(w,u,p)/area2, orient(u,v,p)/area2)
        trace, disc = a+e, (a-e)**2+4*b*b
        certificate = dict(source_coordinate_sha256=_digest(raw), declared_labels=sorted(raw),
            hull_labels=vertices, barycentric=barycentric, center=c, shape_matrix=shape,
            inverse_shape_matrix=inverse, contact_weights=tuple(F(1,3) for _ in vertices),
            determinant_shape=determinant, covariance_determinant=determinant/4,
            area_squared_over_pi_squared=determinant,
            semiaxis_squared=dict(expression='(trace +/- sqrt(discriminant))/2', trace=trace, discriminant=disc),
            major_axis_length=dict(expression='sqrt(2*(trace+sqrt(discriminant)))', trace=trace, discriminant=disc),
            proof='free_center_covariance_trace_determinant_equality', exact_optimum=True)
        budget.spend(len(raw)*10+30)
        if not verify_exact_triangle_ellipse(points, certificate):
            raise ArithmeticError('independent exact certificate verification failed')
        budget.spend()
        result.update(status='EXACT_OPTIMUM', complete=True, exact_optimum=True,
                      center=c, certificate=certificate, work_used=budget.used)
        return result
    except BudgetExceeded as exc:
        return dict(result, status='PARTIAL_UNKNOWN', reason=str(exc), work_used=budget.used)
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError, IndexError) as exc:
        return dict(result, reason=str(exc), work_used=budget.used)


def verify_exact_triangle_ellipse(points, certificate):
    """No hull, solver, floating arithmetic or trust in result status flags.

    Proves convex hull equality from barycentric identities, containment, and
    equality of the positive-weight covariance lower bound and candidate area.
    """
    try:
        raw = _input(points)
        z = certificate
        if (any(type(i) is not int for i in z['declared_labels']) or
                any(type(i) is not int for i in z['hull_labels']) or
                any(type(i) is not int for i in z['barycentric'])):
            return False
        if z['declared_labels'] != sorted(raw) or z['source_coordinate_sha256'] != _digest(raw):
            return False
        ids = z['hull_labels']
        if len(ids) != 3 or len(set(ids)) != 3 or not set(ids) <= raw.keys():
            return False
        if orient(*(raw[i] for i in ids)) == 0 or set(z['barycentric']) != set(raw):
            return False
        c, s, inv, weights = z['center'], z['shape_matrix'], z['inverse_shape_matrix'], z['contact_weights']
        # Exact rational certificate values only; bool/float must never masquerade as proof.
        values = list(c)+[v for row in s for v in row]+[v for row in inv for v in row]+list(weights)
        values += [v for row in z['barycentric'].values() for v in row]
        values += [z['determinant_shape'], z['covariance_determinant'], z['area_squared_over_pi_squared']]
        values += [z[key][field] for key in ('semiaxis_squared','major_axis_length') for field in ('trace','discriminant')]
        if any(type(v) not in (int, F) for v in values):
            return False
        if len(c) != 2 or len(weights) != 3 or weights != (F(1,3),)*3:
            return False
        if (any(len(row) != 2 for row in s) or len(s) != 2 or
                len(inv) != 2 or any(len(row) != 2 for row in inv) or
                s[0][1] != s[1][0] or s[0][0] <= 0):
            return False
        det = s[0][0]*s[1][1]-s[0][1]**2
        if det <= 0 or z['determinant_shape'] != det or z['covariance_determinant'] != det/4 or z['area_squared_over_pi_squared'] != det:
            return False
        for j in (0,1):
            for k in (0,1):
                if sum(s[j][l]*inv[l][k] for l in (0,1)) != int(j == k):
                    return False
                cov = sum(weights[t]*(raw[i][j]-c[j])*(raw[i][k]-c[k]) for t,i in enumerate(ids))
                if 2*cov != s[j][k]:
                    return False
            if sum(weights[t]*raw[i][j] for t,i in enumerate(ids)) != c[j]:
                return False
        for ident,p in raw.items():
            ws = z['barycentric'][ident]
            if len(ws) != 3 or sum(ws) != 1 or min(ws) < 0:
                return False
            if any(sum(ws[t]*raw[i][k] for t,i in enumerate(ids)) != p[k] for k in (0,1)) or _q(p,c,inv) > 1:
                return False
        if any(_q(raw[i],c,inv) != 1 for i in ids):
            return False
        trace, disc = s[0][0]+s[1][1], (s[0][0]-s[1][1])**2+4*s[0][1]**2
        return (z['semiaxis_squared'] == dict(expression='(trace +/- sqrt(discriminant))/2',trace=trace,discriminant=disc)
                and z['major_axis_length'] == dict(expression='sqrt(2*(trace+sqrt(discriminant)))',trace=trace,discriminant=disc)
                and z['proof'] == 'free_center_covariance_trace_determinant_equality' and z['exact_optimum'] is True)
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError, IndexError):
        return False
