import json
import unittest

import numpy as np

from integrations.video_masks.hole_fusion import hole_components, repair_holes, union_masks


class HoleFusionTests(unittest.TestCase):
    def fixture(self):
        base = np.zeros((100, 100), dtype=np.uint8)
        base[5:95, 5:95] = 255
        base[40:44, 40:44] = 0
        support = base.copy()
        support[40:44, 40:44] = 255
        return base, support

    def repair(self, base, support, root=2, **kwargs):
        return repair_holes(base, [{"mask": support, "root_seed_decode_index": root}], 1, **kwargs)

    def test_enclosed_and_no_mutation(self):
        base, support = self.fixture()
        old_base, old_support = base.copy(), support.copy()
        out, report = self.repair(base, support)
        self.assertEqual(report["added_pixels"], 16)
        self.assertEqual(report["holes_after"], 0)
        np.testing.assert_array_equal(out, support)
        np.testing.assert_array_equal(base, old_base)
        np.testing.assert_array_equal(support, old_support)
        self.assertFalse(np.shares_memory(out, base))
        json.dumps(report, allow_nan=False)

    def test_same_root_and_unsupported(self):
        base, support = self.fixture()
        for evidence, root in [(support, 1), (base.copy(), 2), (np.zeros_like(base), 3)]:
            out, report = self.repair(base, evidence, root)
            np.testing.assert_array_equal(out, base)
            self.assertEqual(report["added_pixels"], 0)

    def test_diagonal_and_open_channels_excluded(self):
        for diagonal in (True, False):
            base = np.full((20, 20), 255, np.uint8)
            base[10:13, 10:13] = 0
            if diagonal:
                for i in range(11):
                    base[i, i] = 0
            else:
                base[:11, 10] = 0
            self.assertEqual(hole_components(base), [])
            out, _ = self.repair(base, np.full_like(base, 255), max_hole_foreground_ratio=1)
            np.testing.assert_array_equal(out, base)

    def test_large_hole(self):
        base, support = self.fixture()
        out, report = self.repair(base, support, max_hole_pixels=15)
        np.testing.assert_array_equal(out, base)
        self.assertEqual(report["components"][0]["reason"], "hole_too_large")

    def test_real_background_preserved(self):
        base, support = self.fixture()
        support[40, 40] = 0
        out, report = self.repair(base, support)
        self.assertEqual(report["added_pixels"], 15)
        self.assertEqual(out[40, 40], 0)
        self.assertEqual(report["holes_after"], 1)
        self.assertTrue(np.all(out[base == 255] == 255))
        self.assertTrue(np.all(out[:5] == 0))

    def test_total_cap_rolls_back_every_component(self):
        base, support = self.fixture()
        out, report = self.repair(base, support, max_total_added_fraction=0.001)
        np.testing.assert_array_equal(out, base)
        self.assertTrue(report["total_cap_rejected"])
        self.assertEqual(report["added_pixels"], 0)
        self.assertEqual(report["filled_components"], 0)

    def test_validation(self):
        base, support = self.fixture()
        for invalid in (base.astype(bool), base.astype(float), np.ones_like(base), base[None], np.empty((0, 4), np.uint8)):
            with self.assertRaises(ValueError):
                self.repair(invalid, support)
        with self.assertRaises(ValueError):
            self.repair(base, support[:-1])
        for key, value in [("min_global_iou", float("nan")), ("min_support_fraction", 1.01),
                           ("max_hole_pixels", True), ("max_hole_pixels", 2.5),
                           ("max_total_added_fraction", -1)]:
            with self.assertRaises(ValueError):
                self.repair(base, support, **{key: value})
        with self.assertRaises(ValueError):
            self.repair(base, support, root=True)

    def test_deterministic_repeated_roots_and_ties(self):
        base, support = self.fixture()
        a, b = support.copy(), support.copy()
        a[40, 40] = 0
        b[40, 41] = 0
        supports = [{"mask": a, "root_seed_decode_index": 3},
                    {"mask": b, "root_seed_decode_index": 2},
                    {"mask": a, "root_seed_decode_index": 2}]
        out, report = repair_holes(base, supports, 1)
        other, other_report = repair_holes(base, list(reversed(supports)), 1)
        np.testing.assert_array_equal(out, other)
        self.assertEqual(report, other_report)
        self.assertEqual(report["components"][0]["support_root"], 2)
        self.assertEqual(report["added_pixels"], 15)

    def test_union_includes_holes_and_external_foreground(self):
        base, support = self.fixture()
        support[0, 0] = 255
        saved_base, saved_support = base.copy(), support.copy()
        out, report = union_masks(base, [{"mask": support, "root_seed_decode_index": 1}])
        np.testing.assert_array_equal(out, base | support)
        self.assertEqual(out[0, 0], 255)
        self.assertEqual(report["added_pixels"], 17)
        self.assertEqual(report["holes_before"], 1)
        self.assertEqual(report["holes_after"], 0)
        self.assertEqual(report["removed_pixels"], 0)
        self.assertFalse(report["accuracy_verified"])
        self.assertTrue(report["can_merge_false_positives"])
        np.testing.assert_array_equal(base, saved_base)
        np.testing.assert_array_equal(support, saved_support)
        json.dumps(report, allow_nan=False)

    def test_union_empty_identity_permutation_and_idempotence(self):
        base, support = self.fixture()
        identity, report = union_masks(base, [])
        np.testing.assert_array_equal(identity, base)
        self.assertFalse(np.shares_memory(identity, base))
        self.assertEqual(report["support_roots"], [])
        supports = [{"mask": support, "root_seed_decode_index": 7},
                    {"mask": base, "root_seed_decode_index": 2}]
        out, first = union_masks(base, supports)
        other, second = union_masks(base, list(reversed(supports)))
        np.testing.assert_array_equal(out, other)
        self.assertEqual(first, second)
        duplicated, repeated = union_masks(base, supports + supports)
        np.testing.assert_array_equal(out, duplicated)
        self.assertEqual(repeated["support_roots"], [2, 7])
        self.assertEqual(repeated["support_count"], 4)
        again, _ = union_masks(out, supports)
        np.testing.assert_array_equal(out, again)

    def test_union_validation(self):
        base, support = self.fixture()
        for invalid in (support[:-1], np.ones_like(base), support.astype(bool)):
            with self.assertRaises(ValueError):
                union_masks(base, [{"mask": invalid, "root_seed_decode_index": 1}])
        for invalid_root in (True, -1, 1.5, float("nan"), "1"):
            with self.assertRaises(ValueError):
                union_masks(base, [{"mask": support, "root_seed_decode_index": invalid_root}])
        with self.assertRaises(ValueError):
            union_masks(base.astype(bool), [])


if __name__ == "__main__":
    unittest.main()
