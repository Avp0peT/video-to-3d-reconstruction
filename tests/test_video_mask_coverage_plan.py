import unittest
from integrations.video_masks.planning import make_coverage_plan, make_plan
from test_video_mask_planning import VideoMaskPlanningTests


class VideoMaskCoverageTests(VideoMaskPlanningTests):
    def test_coverage_counts_and_no_success(self):
        p = make_coverage_plan(self.root)
        self.assertEqual(p['coverage_counts'], dict(source_OK=2, selected_interval=1, unbounded_gap=0, budget_deferred=0))
        self.assertEqual(p['sum_frame_count'], 21)
        self.assertEqual(p['unique_decode_frame_count'], 21)
        self.assertEqual(p['intervals'][0]['interval_id'], 'clip_000')
        self.assertEqual({k:v for k,v in p['intervals'][0].items() if k != 'interval_id'}, make_plan(self.root, 96))
        self.assertTrue(all(r['propagation_status'] == 'NOT_RUN' for r in p['coverage_ledger']))
        self.assertEqual(sum(p['coverage_counts'].values()), p['sampled_frame_count'])

    def test_budgets_and_bool(self):
        for field in ('max_frames', 'max_intervals', 'max_total_frames'):
            for value in (True, False, 0, -1, 1.5, '6'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    make_coverage_plan(self.root, **{field:value})
        for field, value in (('max_frames', 97), ('max_intervals', 7), ('max_total_frames', 385)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                make_coverage_plan(self.root, **{field:value})
        p = make_coverage_plan(self.root, max_total_frames=20)
        self.assertEqual(p['status'], 'NOT_APPLICABLE')
        self.assertEqual(p['coverage_counts']['budget_deferred'], 1)
        self.assertEqual(p['unresolved_outside_count'], 1)
        self.assertEqual(p['sum_frame_count'], 0)
        self.assertEqual(make_coverage_plan(self.root, max_frames=20)['coverage_counts']['unbounded_gap'], 1)

    def test_selection_is_eligible_but_never_resolved(self):
        self.rows[1]['status'] = 'NEEDS_SELECTION'
        self.save()
        p = make_coverage_plan(self.root)
        self.assertEqual(p['candidate_interval_count'], 1)
        self.assertEqual(p['coverage_ledger'][1]['original_status'], 'NEEDS_SELECTION')
        self.assertEqual(p['coverage_ledger'][1]['candidate_quality'], 'NOT_EVALUATED')
        self.assertEqual(make_plan(self.root)['status'], 'NOT_APPLICABLE')

    def test_multiple_intervals_priority_shared_endpoints_and_deferred(self):
        from unittest.mock import patch
        rows = [dict(input_index=i, decode_index=d, original_status=status, width=4, height=5)
                for i, (d, status) in enumerate(((0, 'NO_MATCH'), (10, 'OK'),
                    (11, 'NO_MATCH'), (12, 'OK'), (13, 'NO_MATCH'),
                    (14, 'NEEDS_SELECTION'), (15, 'OK'), (16, 'NEEDS_SELECTION')))]
        with patch('integrations.video_masks.planning._validated_evidence', return_value=(rows, {})):
            p = make_coverage_plan(self.root, max_intervals=1)
            self.assertEqual(p['intervals'][0]['start_decode_index'], 12)
            self.assertEqual(p['coverage_counts'], dict(source_OK=3, selected_interval=2, unbounded_gap=2, budget_deferred=1))
            self.assertEqual(p['coverage_ledger'][0]['reason'], 'no_bracket_start')
            self.assertEqual(p['coverage_ledger'][-1]['reason'], 'no_bracket_end')
            p = make_coverage_plan(self.root)
            self.assertEqual([i['interval_id'] for i in p['intervals']], ['clip_000', 'clip_001'])
            self.assertEqual(p['sum_frame_count'], 7)
            self.assertEqual(p['unique_decode_frame_count'], 6)
            self.assertEqual(p['repeated_decode_frame_count'], 1)
            self.assertEqual(p['coverage_ledger'][3]['interval_ids'], ['clip_000', 'clip_001'])
            p = make_coverage_plan(self.root, max_total_frames=3)
            self.assertEqual(p['intervals'][0]['start_decode_index'], 10)
            self.assertEqual(p['coverage_counts']['budget_deferred'], 2)

    def test_complete_validation_precedes_empty_budget_selection(self):
        self.rows[0]['selected_mask']['sha256'] = 'bad'
        self.save()
        with self.assertRaisesRegex(ValueError, 'mask hash mismatch'):
            make_coverage_plan(self.root, max_total_frames=1)


if __name__ == '__main__':
    unittest.main()
