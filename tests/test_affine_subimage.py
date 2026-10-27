import itertools
import unittest
from dataclasses import replace
from fractions import Fraction as F
from unittest.mock import patch

from kozlov_lab.kozlov2025.affine_subimage import (
    find_affine_subimage_2d, verify_affine_subimage_certificate,
)
from kozlov_lab.kozlov2025.independent_cover import Budget


def image(points):
    return dict(enumerate(points))


def oracle(source, target):
    """Slow independent full injection oracle via affine-coordinate identities."""
    s = list(source.values())
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if len(s) == 1:
        return bool(target)
    basis = next((k for k in range(2, len(s)) if cross(s[0], s[1], s[k])), None)
    for t in itertools.permutations(target.values(), len(s)):
        if basis is None:
            axis = 0 if s[0][0] != s[1][0] else 1
            ratios = [F(p[axis]-s[0][axis], s[1][axis]-s[0][axis]) for p in s]
            if all(t[i][j]-t[0][j] == ratios[i]*(t[1][j]-t[0][j])
                   for i in range(len(s)) for j in (0, 1)):
                return True
        else:
            d = cross(s[0], s[1], s[basis])
            if not cross(t[0], t[1], t[basis]):
                continue
            if all(t[i][j]-t[0][j] == F(cross(s[0], s[i], s[basis]), d)*(t[1][j]-t[0][j])
                   + F(cross(s[0], s[1], s[i]), d)*(t[basis][j]-t[0][j])
                   for i in range(len(s)) for j in (0, 1)):
                return True
    return False


class AffineSubimageTests(unittest.TestCase):
    def test_positive_ranks_and_extra_target(self):
        cases = [([(0, 0)], [(9, 3), (0, 0)]),
                 ([(0, 0), (1, 0), (2, 0)], [(3, 1), (3, 4), (3, 7), (99, 99)]),
                 ([(0, 0), (1, 0), (0, 1), (1, 1)],
                  [(2, 3), (4, 3), (2, 6), (4, 6), (100, 100)])]
        for s, t in cases:
            with self.subTest(s=s):
                r = find_affine_subimage_2d(image(s), image(t))
                self.assertEqual(r.status, 'TRUE')
                self.assertTrue(r.complete)
                self.assertEqual(verify_affine_subimage_certificate(image(s), image(t), r.certificate), 'TRUE')

    def test_negative_ranks_and_cardinality(self):
        for s, t in [([(0, 0), (1, 0), (2, 0)], [(0, 0), (1, 0), (3, 0)]),
                     ([(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (0, 1), (2, 2)]),
                     ([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (2, 0)]),
                     ([(0, 0), (1, 0)], [(0, 0)])]:
            r = find_affine_subimage_2d(image(s), image(t))
            self.assertEqual(r.status, 'FALSE')
            self.assertTrue(r.complete)

    def test_all_small_grid_sets_against_independent_injection_oracle(self):
        grid = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
        sources = [image(s) for n in range(1, 5) for s in itertools.combinations(grid, n)]
        targets = [image(t) for n in (3, 4) for t in itertools.combinations(grid, n)]
        for s in sources:
            for t in targets:
                r = find_affine_subimage_2d(s, t)
                self.assertEqual(r.status == 'TRUE', oracle(s, t), (s, t))
                self.assertTrue(r.complete)

    def test_fraction_labels_and_order(self):
        s = {7: (F(1, 3), 0), -2: (F(2, 3), 0), 99: (1, 0)}
        t = {13: (0, 0), 9: (0, F(1, 7)), -4: (0, F(2, 7)), 1: (5, 5)}
        self.assertEqual(find_affine_subimage_2d(s, t).status, 'TRUE')
        self.assertEqual(find_affine_subimage_2d(dict(reversed(list(s.items()))), t).status, 'TRUE')

    def test_unknown_shared_work_and_deadline(self):
        s, t = image([(0, 0), (1, 0)]), image([(0, 0), (2, 0)])
        for b in (Budget(0, 30), Budget(2000000, 0), Budget(20, 30)):
            r = find_affine_subimage_2d(s, t, budget=b)
            self.assertEqual(r.status, 'UNKNOWN')
            self.assertFalse(r.complete)
        b = Budget(2000000, 30)
        r = find_affine_subimage_2d(s, t, budget=b)
        self.assertEqual(r.work_used, b.used)
        self.assertEqual(verify_affine_subimage_certificate(s, t, r.certificate, budget=Budget(0, 30)), 'UNKNOWN')

    def test_invalid_domain(self):
        bad = [{}, {0: (0.0, 0)}, {0: (True, 0)}, {True: (0, 0)},
               {0: (0, 0), 1: (0, 0)}, {0: (0, 0, 0)},
               {0: (2**256, 0)}, image([(i, 0) for i in range(13)])]
        for s in bad:
            with self.subTest(s=s), self.assertRaises(ValueError):
                find_affine_subimage_2d(s, {0: (0, 0)})
        with self.assertRaises(ValueError):
            find_affine_subimage_2d({0: (0, 0)}, image([(i, 0) for i in range(33)]))

    def test_certificate_tampering_and_search_independence(self):
        s, t = image([(0, 0), (1, 0), (0, 1)]), image([(0, 0), (2, 0), (0, 3), (9, 9)])
        c = find_affine_subimage_2d(s, t).certificate
        with patch('kozlov_lab.kozlov2025.affine_subimage.find_affine_subimage_2d', side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.affine_subimage._solve', side_effect=AssertionError), \
             patch('kozlov_lab.kozlov2025.affine_subimage._map', side_effect=AssertionError):
            self.assertEqual(verify_affine_subimage_certificate(s, t, c), 'TRUE')
        bad = [replace(c, version='bad'), replace(c, mapping=c.mapping[:-1]),
               replace(c, mapping=((0, 0), (1, 0), (2, 2))),
               replace(c, translation=(F(1), F(0))),
               replace(c, matrix=((F(0), F(0)), (F(0), F(0)))),
               replace(c, mapping=((False, 0), *c.mapping[1:])),
               replace(c, source=((False, c.source[0][1]), *c.source[1:])),
               replace(c, matrix=((True, F(0)), (F(0), F(1))))]
        for forged in bad:
            self.assertEqual(verify_affine_subimage_certificate(s, t, forged), 'FALSE')
        self.assertEqual(verify_affine_subimage_certificate(s, {**t, 8: (100, 100)}, c), 'FALSE')


if __name__ == '__main__':
    unittest.main()
