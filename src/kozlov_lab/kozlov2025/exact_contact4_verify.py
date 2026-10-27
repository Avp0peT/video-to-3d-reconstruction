"""Independent positive-moment proof for a complete four-contact ellipse.

Only certificate identities are checked. No conic pencil, RREF, root candidate
enumeration, convex hull, or minimum-ellipse constructor is called here.
"""
from fractions import Fraction as F
import math

from .independent_cover import normalize, Budget, BudgetExceeded
from .exact_triangle_ellipse import _digest
from .real_algebraic import Root, AN

PROOF = 'positive_contact_covariance_equality_and_complete_input_containment'


def _input(points):
    if not isinstance(points, dict) or not 1 <= len(points) <= 7:
        raise ValueError('complete input must contain 1..7 labelled points')
    for label, pair in points.items():
        if (type(label) is not int or not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(type(x) not in (int, F) for x in pair)):
            raise ValueError('integer labels and exact int/Fraction coordinates required')
    return normalize(points, [sorted(points)])[0]


def assess_contact4_certificate(points, certificate, *, max_work=2000000, seconds=30):
    """Return VERIFIED/INVALID/UNKNOWN; resource exhaustion never means false geometry."""
    result = {'schema': 'contact4-certificate-assessment-v1', 'status': 'INVALID',
              'verified': False, 'work_used': 0}
    if (type(max_work) is not int or not 0 <= max_work <= 2000000 or
            type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 30):
        return dict(result, reason='invalid verifier budget')
    budget = Budget(max_work, seconds)
    def require(ok, reason):
        budget.spend()
        if not ok:
            raise ValueError(reason)
    try:
        budget.spend()
        raw = _input(points)
        z = certificate
        require(type(z) is dict and z.get('schema') == 'complete-input-contact4-certificate-v1',
                'certificate schema')
        declared = z['declared_labels']; ids = z['support_labels']
        require(isinstance(declared, (tuple, list)) and all(type(i) is int for i in declared)
                and list(declared) == sorted(raw), 'complete label binding')
        require(isinstance(ids, (tuple, list)) and len(ids) == 4
                and all(type(i) is int for i in ids) and len(set(ids)) == 4
                and set(ids) <= raw.keys(), 'four distinct supplied contact labels')
        require(z['source_coordinate_sha256'] == _digest(raw), 'coordinate digest')
        root = Root.from_data(z['algebraic_root'], budget)
        def number(value):
            return AN.from_data(value, root)
        def vector(value, n):
            require(isinstance(value, (tuple, list)) and len(value) == n, 'vector dimension')
            return tuple(number(v) for v in value)
        def matrix(value):
            require(isinstance(value, (tuple, list)) and len(value) == 2, 'matrix dimension')
            return tuple(vector(row, 2) for row in value)
        center = vector(z['center'], 2)
        shape = matrix(z['shape_matrix']); inverse = matrix(z['inverse_shape_matrix'])
        for value in (shape, inverse):
            require(value[0][1].equals(value[1][0]), 'matrix symmetry')
            require(value[0][0].sign() > 0, 'positive leading principal minor')
            require((value[0][0]*value[1][1]-value[0][1]*value[1][0]).sign() > 0,
                    'positive determinant')
        for j in (0, 1):
            for k in (0, 1):
                require(sum(shape[j][l]*inverse[l][k] for l in (0, 1)).equals(int(j == k)),
                        'shape/inverse identity')
        weights = vector(z['contact_weights'], 4)
        require(all(w.sign() > 0 for w in weights), 'strictly positive contact weights')
        require(sum(weights).equals(1), 'normalized weights')
        offsets = [tuple(raw[i][j]-center[j] for j in (0, 1)) for i in ids]
        for j in (0, 1):
            require(sum(weights[t]*offsets[t][j] for t in range(4)).equals(0), 'weighted mean')
            for k in (0, 1):
                require(sum(weights[t]*offsets[t][j]*offsets[t][k] for t in range(4)).equals(shape[j][k]/2),
                        'positive covariance equality')
        values = z['all_input_quadratic_values']
        require(type(values) is dict and all(type(i) is int for i in values)
                and set(values) == set(raw), 'complete quadratic values')
        for i, p in raw.items():
            x, y = p[0]-center[0], p[1]-center[1]
            q = inverse[0][0]*x*x+2*inverse[0][1]*x*y+inverse[1][1]*y*y
            require(number(values[i]).equals(q), 'quadratic identity')
            require((q-1).sign() <= 0, 'complete input containment')
            if i in ids:
                require(q.equals(1), 'support contact')
        determinant = shape[0][0]*shape[1][1]-shape[0][1]*shape[1][0]
        for key, value in [('determinant_shape', determinant),
                           ('area_squared_over_pi_squared', determinant),
                           ('covariance_determinant', determinant/4)]:
            require(number(z[key]).equals(value), key)
        trace = shape[0][0]+shape[1][1]
        discriminant = (shape[0][0]-shape[1][1])**2+4*shape[0][1]**2
        for key, expression in [('semiaxis_squared', '(trace +/- sqrt(discriminant))/2'),
                                ('major_axis_length', 'sqrt(2*(trace+sqrt(discriminant)))')]:
            item = z[key]
            require(type(item) is dict and set(item) == {'expression', 'trace', 'discriminant'}
                    and item['expression'] == expression, 'size formula')
            require(number(item['trace']).equals(trace) and
                    number(item['discriminant']).equals(discriminant), 'size identity')
        require(z['proof'] == PROOF and z['exact_optimum'] is True, 'optimum claim')
        budget.spend(0)
        return dict(result, status='VERIFIED', verified=True, work_used=budget.used,
                    reason='independent positive covariance lower bound attained on complete input')
    except BudgetExceeded as exc:
        return dict(result, status='UNKNOWN', reason=str(exc), work_used=budget.used)
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError, IndexError) as exc:
        return dict(result, reason=str(exc), work_used=budget.used)


def verify_complete_input_contact4_ellipse(points, certificate, *, max_work=2000000, seconds=30):
    """Convenient boolean gate; callers needing uncertainty use assess_* above."""
    return assess_contact4_certificate(points, certificate, max_work=max_work, seconds=seconds)['verified']
