import unittest
from dataclasses import replace
from unittest.mock import patch

from kozlov_lab.kozlov2025.finite_environment import (
    enumerate_environment_parts, verify_environment_parts, verify_separate_group,
    enumerate_separate_groups, verify_separate_groups_certificate,
)
from kozlov_lab.kozlov2025.independent_cover import Budget


def points(n):
    return {10+i: (i, i*i) for i in range(n)}


class FiniteEnvironmentTests(unittest.TestCase):
    def test_all_parts_and_star_cardinality(self):
        for n in range(1, 8):
            p = points(n)
            c = enumerate_environment_parts(p)
            self.assertEqual(len(c.parts), 2**n-1)
            self.assertEqual(c.star_cardinality, 2**(2**n-1))
            self.assertTrue(verify_environment_parts(p, c))
            self.assertFalse(verify_environment_parts(p, replace(c, parts=c.parts[:-1])))
            self.assertFalse(verify_environment_parts(p, replace(c, star_cardinality=0)))
            if n > 1:
                self.assertFalse(verify_environment_parts(p, replace(c, parts=tuple(reversed(c.parts)))))

    def test_complete_separate_groups_counts(self):
        # Bell(n+1), independently tabulated mathematical small-n values.
        bell = [2, 5, 15, 52, 203, 877, 4140]
        for n, count in enumerate(bell, 1):
            for empty in (False, True):
                r = enumerate_separate_groups(points(n), include_empty=empty)
                self.assertEqual(r.status, 'COMPLETE')
                self.assertTrue(r.complete)
                self.assertEqual(len(r.groups), count-(not empty))
                self.assertEqual(() in r.groups, empty)
                self.assertTrue(all(verify_separate_group(points(n), g, include_empty=empty).status == 'TRUE'
                                    for g in r.groups))

    def test_label_separation_does_not_mean_hull_independence(self):
        p = {0: (0, 0), 1: (2, 2), 2: (0, 2), 3: (2, 0)}
        # Crossing hulls: label-disjoint group remains S+.
        self.assertEqual(verify_separate_group(p, [[0, 1], [2, 3]]).status, 'TRUE')
        r = verify_separate_group(p, [[0, 1], [1, 2]])
        self.assertEqual(r.status, 'FALSE')
        self.assertEqual(r.overlap, (1,))
        self.assertEqual(verify_separate_group(p, [[0]]).status, 'TRUE')
        self.assertEqual(verify_separate_group(p, []).status, 'EXCLUDED_BY_POLICY')
        self.assertEqual(verify_separate_group(p, [], include_empty=True).status, 'TRUE')

    def test_independent_completeness_and_tamper(self):
        p = points(4)
        c = enumerate_separate_groups(p, include_empty=True).certificate
        with patch('kozlov_lab.kozlov2025.finite_environment.enumerate_separate_groups', side_effect=AssertionError):
            self.assertEqual(verify_separate_groups_certificate(p, c), 'TRUE')
        for bad in (replace(c, groups=c.groups[:-1]), replace(c, groups=c.groups+(c.groups[0],)),
                    replace(c, groups=tuple(reversed(c.groups))), replace(c, include_empty=False),
                    replace(c, version='bad'), replace(c, points=())):
            self.assertEqual(verify_separate_groups_certificate(p, bad), 'FALSE')
        self.assertEqual(verify_separate_groups_certificate(points(5), c), 'FALSE')

    def test_budget_partial_and_shared_verification(self):
        for b in (Budget(0, 30), Budget(50, 30), Budget(2000000, 0)):
            r = enumerate_separate_groups(points(7), budget=b)
            self.assertFalse(r.complete)
            self.assertIsNone(r.certificate)
            self.assertIn(r.status, ('PARTIAL_UNKNOWN', 'UNKNOWN'))
        b = Budget(2000000, 30)
        r = enumerate_separate_groups(points(7), budget=b)
        self.assertEqual(r.work_used, b.used)
        self.assertEqual(verify_separate_groups_certificate(points(7), r.certificate, budget=Budget(0, 30)), 'UNKNOWN')

    def test_invalid_input_and_group(self):
        for p in ({}, points(8), {0: (0.0, 0)}, {True: (0, 0)}, {0: (0, 0), 1: (0, 0)}):
            with self.assertRaises(ValueError):
                enumerate_environment_parts(p)
        for g in ([[10, 10]], [[999]], [[]], [[True]], 'bad'):
            with self.assertRaises(ValueError):
                verify_separate_group(points(2), g)
        with self.assertRaises(ValueError):
            enumerate_separate_groups(points(1), include_empty=1)


if __name__ == '__main__':
    unittest.main()
