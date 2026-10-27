import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integrations.video_masks.full_planning import make_full_plan, _reuse
from integrations.video_masks.planning import _hash


class FullPlanningTests(unittest.TestCase):
    def plan(self, count, anchors, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'timeline.json'
            path.write_text(json.dumps({'frames': [{'decode_index': i} for i in range(count)]}), encoding='utf-8')
            rows = [dict(decode_index=i, input_index=n, frame_id=f'source_{i}', original_status='OK',
                         width=4, height=5, mask_path=f'mask_{i}', mask_sha256='frozen')
                    for n, i in enumerate(anchors)]
            with patch('integrations.video_masks.full_planning._validated_evidence',
                       return_value=(rows, {'timeline_path': str(path)})):
                return make_full_plan(directory, **kwargs)

    def test_long_gap_complete_bidirectional_and_chained_local_ids(self):
        plan = self.plan(301, [0, 300])
        for direction in ('forward', 'reverse'):
            tasks = [t for t in plan['tasks'] if t['direction'] == direction]
            covered = set()
            previous = None
            for task in tasks:
                self.assertLessEqual(task['frame_count'], 96)
                covered.update(range(task['start_decode_index'], task['end_decode_index']+1))
                if previous is None:
                    self.assertIn('seed', task)
                else:
                    source = task['seed_from']
                    self.assertEqual(source['task_id'], previous['task_id'])
                    self.assertEqual(previous['start_decode_index']+source['frame_id'], source['decode_index'])
                    self.assertEqual(task['start_decode_index']+source['frame_index'], source['decode_index'])
                    self.assertLess(source['frame_id'], previous['frame_count'])
                    self.assertNotIn('seed', task)
                previous = task
            self.assertEqual(covered, set(range(301)))

    def test_head_tail_only_have_one_direction(self):
        plan = self.plan(200, [30, 150])
        self.assertEqual([i['role'] for i in plan['intervals']], ['head', 'between_anchors', 'tail'])
        self.assertEqual(plan['intervals'][0]['directions'], ['reverse'])
        self.assertEqual(plan['intervals'][-1]['directions'], ['forward'])
        self.assertTrue(plan['intervals'][0]['no_independent_opposite_anchor'])
        self.assertFalse(plan['intervals'][1]['no_independent_opposite_anchor'])

    def test_single_anchor_and_single_frame(self):
        plan = self.plan(150, [70])
        self.assertEqual([i['role'] for i in plan['intervals']], ['head', 'tail'])
        self.assertEqual(self.plan(1, [0])['tasks'], [])

    def test_no_anchor_and_budget_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'original OK'):
            self.plan(200, [])
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.plan(2000, [0, 1999])
        with self.assertRaises(ValueError):
            self.plan(4097, [0])
        with self.assertRaises(ValueError):
            self.plan(100, [0, 99], overlap=96)

    def test_reuse_input_hash_drift_is_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.write_bytes(b'original')
            frozen = _hash(source)
            (root / 'PLAN.json').write_text('{}', encoding='utf-8')
            (root / 'RESULT.json').write_text(json.dumps({'status': 'REVIEW_READY', 'input_drift': [], 'code_drift': []}), encoding='utf-8')
            (root / 'START_BINDINGS.json').write_text(json.dumps({'input_files': {str(source): frozen}}), encoding='utf-8')
            timeline = root / 'timeline.json'
            timeline.write_text('{}', encoding='utf-8')
            # Bind timeline in PLAN; mutate frozen source after the historical successful result.
            (root / 'PLAN.json').write_text(json.dumps({'timeline_path': str(timeline), 'coverage_ledger': []}), encoding='utf-8')
            source.write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError, 'reuse input hash mismatch'):
                _reuse(root, {'timeline_path': str(timeline)})


if __name__ == '__main__':
    unittest.main()
