import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from integrations.video_masks.policy import PropagationPolicy, choose_propagation, compare_masks


class VideoMaskPolicyTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.array([[False, True], [True, True]])

    def test_candidate_and_deterministic_selection(self):
        result = choose_propagation(self.mask, self.mask)
        self.assertEqual(result["status"], "PROPAGATED_CANDIDATE")
        self.assertEqual(result["recommended_source"], "forward")
        self.assertFalse(result["direct_sam_ok"])
        result = choose_propagation(self.mask, self.mask, forward_confidence=.2, reverse_confidence=.4)
        self.assertEqual(result["recommended_source"], "reverse")

    def test_original_is_retained_with_review_only(self):
        result = choose_propagation(self.mask, self.mask, self.mask)
        self.assertEqual(result["status"], "ORIGINAL_RETAINED")
        self.assertEqual(result["retained_source"], "original")
        self.assertTrue(result["review_candidate_only"])

    def test_missing_empty_and_disagreement(self):
        empty = np.zeros_like(self.mask)
        for a, b in ((None, self.mask), (self.mask, None), (empty, self.mask), (self.mask, empty)):
            self.assertEqual(choose_propagation(a, b)["status"], "UNRESOLVED")
        self.assertEqual(choose_propagation(empty, empty)["status"], "EMPTY")
        self.assertEqual(choose_propagation(self.mask, ~self.mask)["status"], "UNRESOLVED")

    def test_invalid_arrays(self):
        for bad in (np.array([True]), np.zeros((0, 2)), np.array([[float("nan")]]),
                    np.array([[0., 1.]]), np.array([[2]]), np.array([[0, 1, 255]])):
            with self.assertRaises(ValueError):
                choose_propagation(bad, bad)
        with self.assertRaises(ValueError):
            choose_propagation(self.mask, np.ones((3, 3), bool))

    def test_integer_binary_rules_and_no_mutation(self):
        before = self.mask.copy()
        for scale in (1, 255):
            result = choose_propagation(self.mask.astype(np.uint8) * scale, self.mask)
            self.assertEqual(result["status"], "PROPAGATED_CANDIDATE")
        np.testing.assert_array_equal(before, self.mask)

    def test_confidence_and_original_validation(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                choose_propagation(self.mask, self.mask, forward_confidence=value)
        result = choose_propagation(self.mask, self.mask, reverse_confidence=.9)
        self.assertEqual(result["recommended_source"], "forward")
        result = choose_propagation(None, None, self.mask)
        self.assertEqual(result["status"], "ORIGINAL_RETAINED")
        with self.assertRaises(ValueError):
            choose_propagation(self.mask, self.mask, np.ones((3, 3), bool))

    def test_threshold_configuration(self):
        a = np.ones((2, 5), bool)
        b = a.copy()
        b[:, -1] = False
        self.assertEqual(choose_propagation(a, b)["status"], "PROPAGATED_CANDIDATE")
        self.assertEqual(choose_propagation(a, b, policy=PropagationPolicy(.81, 1.5))["status"], "UNRESOLVED")
        self.assertEqual(choose_propagation(a, b, policy=PropagationPolicy(.8, 1.2))["status"], "UNRESOLVED")
        for kwargs in ({"min_iou": float("nan")}, {"max_area_ratio": .9}):
            with self.assertRaises(ValueError):
                PropagationPolicy(**kwargs)

    def test_file_metrics_and_empty_iou(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.png", Path(directory) / "b.png"
            Image.fromarray(self.mask.astype(np.uint8) * 255).save(a)
            Image.fromarray(self.mask.astype(np.uint8)).save(b)
            result = compare_masks(a, b)
            self.assertEqual(result["intersection_pixels"], 3)
            self.assertEqual(result["union_pixels"], 3)
            self.assertEqual(result["boundary_iou"], 1)
            Image.fromarray(np.zeros((2, 2), np.uint8)).save(a)
            Image.fromarray(np.zeros((2, 2), np.uint8)).save(b)
            self.assertIsNone(compare_masks(a, b)["iou"])
            Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(b)
            with self.assertRaises(ValueError):
                compare_masks(a, b)


if __name__ == "__main__":
    unittest.main()
