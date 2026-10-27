"""Synthetic cycle evidence, independent of torch/model/GPU."""
import unittest

from integrations.demo_pipeline.learned_cycle_check import assess_data


def row(raw, xy, status='VISIBLE_CANDIDATE', track_id=0):
    return dict(decode_index=raw, points=[dict(track_id=track_id, xy=xy, status=status)])


class CycleTests(unittest.TestCase):
    def assess(self, forward, reverse, ids=(0,)):
        return assess_data(forward, reverse, [dict(track_id=ids[0], xy=[0, 0])], ids, 555, 557)

    def test_known_translation_return_uses_query_and_forward_separately(self):
        forward = [row(555+i, [i+1, 0]) for i in range(3)]
        reverse = [row(557-i, [3-i, 0]) for i in range(3)]
        result = self.assess(forward, reverse)
        self.assertEqual(result['all_queries']['maximum'], 0)
        self.assertEqual(result['track_id_0']['return_error_vs_original_query_pixels'], 1)
        self.assertEqual(result['track_id_0']['return_error_vs_forward_first_pixels'], 0)
        self.assertEqual(result['track_id_0']['status'], 'PASS_HEURISTIC')

    def test_drift_and_inclusive_four_pixel_heuristic(self):
        forward = [row(555+i, [i, 0]) for i in range(3)]
        reverse = [row(557, [2, 0]), row(556, [5, 0]), row(555, [5, 0])]
        result = self.assess(forward, reverse)
        self.assertEqual(result['track_id_0']['status'], 'FAIL_HEURISTIC')
        self.assertEqual(result['track_id_0']['rows'][1]['status'], 'PASS_HEURISTIC')
        self.assertEqual(result['all_queries']['maximum'], 5)

    def test_mask_gap_and_missing_are_unknown_without_interpolation(self):
        forward = [row(555, [0, 0]), row(556, [1, 0], 'MASK_UNAVAILABLE'), row(557, [2, 0])]
        reverse = [row(557, [2, 0]), row(556, [1, 0])]
        result = self.assess(forward, reverse)
        self.assertEqual(result['all_queries']['count'], 1)
        self.assertEqual(result['track_id_0']['status'], 'UNKNOWN')
        self.assertIsNone(result['track_id_0']['return_error_vs_original_query_pixels'])
        self.assertIsNone(result['track_id_0']['rows'][1]['discrepancy_pixels'])

    def test_nonzero_id_and_reverse_frame_order(self):
        forward = [row(555+i, [i, 0], track_id=19) for i in range(3)]
        result = self.assess(forward, list(reversed(forward)), (19,))
        self.assertEqual(result['tracks'][0]['track_id'], 19)
        self.assertEqual(result['track_id_0']['status'], 'NOT_SELECTED_AT_ENDPOINT')
        self.assertEqual(result['all_queries']['count'], 3)


if __name__ == '__main__':
    unittest.main()
