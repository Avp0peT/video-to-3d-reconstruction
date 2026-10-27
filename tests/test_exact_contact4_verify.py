"""Hand-derived optimum checks independent of the four-contact constructor."""
import copy
from fractions import Fraction as F
import pytest

from kozlov_lab.kozlov2025 import exact_contact4_verify as v
from kozlov_lab.kozlov2025.exact_triangle_ellipse import _digest

SQUARE = {0: (F(0), F(0)), 1: (F(1), F(0)), 2: (F(1), F(1)), 3: (F(0), F(1))}


def number(x):
    x = F(x)
    return {'numerator': [[x.numerator, x.denominator]], 'denominator': [[1, 1]]}


def certificate(points=SQUARE):
    """The minimum ellipse is the circumscribed circle, from positive moments."""
    center = [number(F(1, 2)), number(F(1, 2))]
    shape = [[number(F(1, 2)), number(0)], [number(0), number(F(1, 2))]]
    inverse = [[number(2), number(0)], [number(0), number(2)]]
    return {'schema': 'complete-input-contact4-certificate-v1',
            'algebraic_root': {'poly': [[0, 1], [1, 1]], 'lo': [-1, 1], 'hi': [1, 1]},
            'declared_labels': sorted(points), 'source_coordinate_sha256': _digest(points),
            'support_labels': [0, 1, 2, 3], 'center': center, 'shape_matrix': shape,
            'inverse_shape_matrix': inverse, 'contact_weights': [number(F(1, 4)) for _ in range(4)],
            'all_input_quadratic_values': {i: number(2*((p[0]-F(1, 2))**2+(p[1]-F(1, 2))**2))
                                           for i, p in points.items()},
            'determinant_shape': number(F(1, 4)), 'covariance_determinant': number(F(1, 16)),
            'area_squared_over_pi_squared': number(F(1, 4)),
            'semiaxis_squared': {'expression': '(trace +/- sqrt(discriminant))/2',
                                 'trace': number(1), 'discriminant': number(0)},
            'major_axis_length': {'expression': 'sqrt(2*(trace+sqrt(discriminant)))',
                                  'trace': number(1), 'discriminant': number(0)},
            'proof': v.PROOF, 'exact_optimum': True}


def test_hand_derived_certificate_without_constructor(monkeypatch):
    import kozlov_lab.kozlov2025.real_algebraic as algebra
    import kozlov_lab.kozlov2025.exact_contact5_ellipse as contact5
    def forbidden(*a, **kw):
        raise AssertionError('candidate constructor called from certificate verifier')
    monkeypatch.setattr(algebra, 'isolate_real_roots', forbidden)
    monkeypatch.setattr(contact5, '_rref', forbidden)
    assert v.assess_contact4_certificate(SQUARE, certificate())['status'] == 'VERIFIED'
    points = {**SQUARE, 8: (F(1, 2), F(1, 2))}
    assert v.verify_complete_input_contact4_ellipse(points, certificate(points))


@pytest.mark.parametrize('mutation', ['mean', 'weight', 'zero_weight', 'inverse', 'outside',
    'digest', 'missing_label', 'bool_label', 'approx', 'shape', 'size', 'area', 'root_count',
    'zero_denominator', 'bool_rational', 'proof', 'false_claim'])
def test_independent_counterexamples(mutation):
    points = copy.deepcopy(SQUARE); z = certificate()
    if mutation == 'mean': z['center'][0] = number(0)
    elif mutation == 'weight': z['contact_weights'] = [number(F(1, 3))]*4
    elif mutation == 'zero_weight': z['contact_weights'][0] = number(0)
    elif mutation == 'inverse': z['inverse_shape_matrix'][1][1] = number(1)
    elif mutation == 'outside':
        points[8] = (F(4), F(4)); z = certificate(points)
    elif mutation == 'digest': z['source_coordinate_sha256'] = 'changed'
    elif mutation == 'missing_label': z['declared_labels'].pop()
    elif mutation == 'bool_label': z['support_labels'][0] = False
    elif mutation == 'approx': z['center'][0] = .5
    elif mutation == 'shape': z['shape_matrix'][0][1] = number(1)
    elif mutation == 'size': z['major_axis_length']['trace'] = number(4)
    elif mutation == 'area': z['area_squared_over_pi_squared'] = number(1)
    elif mutation == 'root_count': z['algebraic_root']['poly'] = [[-1, 1], [0, 1], [1, 1]]
    elif mutation == 'zero_denominator': z['center'][0]['denominator'] = [[0, 1], [1, 1]]
    elif mutation == 'bool_rational': z['center'][0]['numerator'] = [[True, 2]]
    elif mutation == 'proof': z['proof'] = 'believe_the_constructor'
    elif mutation == 'false_claim': z['exact_optimum'] = 1
    assert v.assess_contact4_certificate(points, z)['status'] == 'INVALID'


@pytest.mark.parametrize('budget', [{'seconds': 0}, {'max_work': 0}, {'max_work': 30}])
def test_budget_unknown_is_not_invalid_geometry(budget):
    result = v.assess_contact4_certificate(SQUARE, certificate(), **budget)
    assert result['status'] == 'UNKNOWN' and not result['verified']


def test_huge_untrusted_certificate_is_bounded():
    z = certificate(); z['center'][0]['numerator'][0] = [1 << 17000, 1]
    assert v.assess_contact4_certificate(SQUARE, z)['status'] == 'UNKNOWN'
