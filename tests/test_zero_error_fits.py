import copy
import unittest
from fractions import Fraction as F
from unittest.mock import patch

from kozlov_lab.kozlov2025.zero_error_fits import (
    given_zero_error_fits, assess_zero_error_fits_certificate,
    verify_zero_error_fits_certificate,
)
from kozlov_lab.kozlov2025.zero_error_fits import _center, NonRational
from kozlov_lab.kozlov2025.independent_cover import Budget
from kozlov_lab.kozlov2025.real_algebraic import Root


def reference():
    return {0: (-1, -1), 1: (1, -1), 2: (0, 2),
            3: (9, -1), 4: (11, -1), 5: (10, 2)}


PARTS = ((0, 1, 2), (3, 4, 5))
IDENTITY = ((F(1), F(0)), (F(0), F(1)))


class ZeroErrorFitsTests(unittest.TestCase):
    def test_algebraic_constant_recognition_without_approximation(self):
        budget = Budget(2000000, 30)
        root = Root((F(-2), F(0), F(1)), F(1), F(2), budget)
        context = {'domain': 'real-algebraic', 'algebraic_root': root.to_data()}
        constant = root.num((F(3),)).to_data()
        coordinate = root.num((F(0), F(1))).to_data()
        self.assertEqual(_center({'number_context': context, 'center': [constant, constant]}, budget), (3, 3))
        with self.assertRaises(NonRational):
            _center({'number_context': context, 'center': [coordinate, constant]}, budget)

    def test_triangle_blocks_nontrivial_affine_witness(self):
        a = reference()
        b = {i: (F(p[0]-2, 2), F(p[1]-1, 3)) for i, p in a.items()}
        m = ((F(2), F(0)), (F(0), F(3)))
        r = given_zero_error_fits(a, PARTS, b, PARTS, m, (2, 1), {0: 0, 1: 1})
        self.assertEqual(r['status'], 'TRUE', r)
        c = r['certificate']
        self.assertEqual(len(c['point_checks']), 6)
        self.assertEqual(len(c['center_checks']), 2)
        self.assertEqual((c['epsilon1'], c['epsilon2']), (1, 1))
        self.assertTrue(verify_zero_error_fits_certificate(a, b, c))

    def test_given_center_error_wrong_bijection_and_outpoints(self):
        a = reference()
        for m, t, mapping, b in [
            (IDENTITY, (1, 0), {0: 0, 1: 1}, a),
            (IDENTITY, (0, 0), {0: 1, 1: 0}, a),
            (IDENTITY, (0, 0), {0: 0, 1: 1},
             {i: ((p[0]-(0 if i < 3 else 10))*20+(0 if i < 3 else 10), p[1]*20) for i,p in a.items()}),
        ]:
            r = given_zero_error_fits(a, PARTS, b, PARTS, m, t, mapping)
            self.assertEqual(r['status'], 'FALSE_GIVEN_WITNESS', r)

    def test_independent_verifier_does_not_call_producers(self):
        a = reference()
        c = given_zero_error_fits(a, PARTS, a, PARTS, IDENTITY, (0, 0), {0: 0, 1: 1})['certificate']
        with patch('kozlov_lab.kozlov2025.zero_error_fits.given_zero_error_fits', side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.zero_error_fits.solve_exact_center', side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.primary_futlyar.assess_primary_futlyar', side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.exact_center.solve_exact_center', side_effect=AssertionError):
            self.assertTrue(verify_zero_error_fits_certificate(a, a, c))

    def test_full_certificate_tamper(self):
        a = reference()
        original = given_zero_error_fits(a, PARTS, a, PARTS, IDENTITY, (0, 0), {0: 0, 1: 1})['certificate']
        for key, value in [('proof', 'bad'), ('epsilon1', F(0)), ('point_checks', ()),
                           ('center_checks', ()), ('source_points', ()), ('part_bijection', ((0, 0),))]:
            c = copy.deepcopy(original)
            c[key] = value
            self.assertEqual(assess_zero_error_fits_certificate(a, a, c)['status'], 'INVALID')
        c = copy.deepcopy(original)
        c['source_blocks'][0]['center'] = (F(99), F(0))
        self.assertEqual(assess_zero_error_fits_certificate(a, a, c)['status'], 'INVALID')
        c = copy.deepcopy(original)
        c['primary_certificate']['radius_squared'] += 1
        self.assertEqual(assess_zero_error_fits_certificate(a, a, c)['status'], 'INVALID')

    def test_unknown_budget_and_invalid_inputs(self):
        a = reference()
        for options in ({'max_work': 0}, {'seconds': 0}, {'max_work': 1000}):
            self.assertEqual(given_zero_error_fits(a, PARTS, a, PARTS, IDENTITY, (0, 0), {0: 0, 1: 1}, **options)['status'], 'UNKNOWN')
        for parts in (((0, 1, 2), (3, 4)), ((0, 1, 2), (2, 3, 4, 5))):
            self.assertEqual(given_zero_error_fits(a, PARTS, a, parts, IDENTITY, (0, 0), {0: 0, 1: 1})['status'], 'UNSUPPORTED')
        c = given_zero_error_fits(a, PARTS, a, PARTS, IDENTITY, (0, 0), {0: 0, 1: 1})['certificate']
        self.assertEqual(assess_zero_error_fits_certificate(a, a, c, max_work=0)['status'], 'UNKNOWN')

    def test_source_blocks_need_not_be_independent_input_precondition(self):
        a = reference()
        # Source hulls intersect, but this is a valid labelled complete partition.
        b = {0: (-20, -1), 1: (20, -1), 2: (0, 2),
             3: (-10, -1), 4: (30, -1), 5: (10, 2)}
        r = given_zero_error_fits(a, PARTS, b, PARTS, IDENTITY, (0, 0), {0: 0, 1: 1})
        self.assertEqual(r['status'], 'FALSE_GIVEN_WITNESS')
        self.assertIn('strict truncated', r['reason'])


if __name__ == '__main__':
    unittest.main()
