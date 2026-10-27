"""Sampling-boundary integration: measurements, masks and pending are distinct."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np
from PIL import Image

from integrations.demo_pipeline.temporal_review import load_cv
from integrations.demo_pipeline.balanced_object_tracks import Profile
from integrations.demo_pipeline.gap_tolerant_tracks import GapTracker
from integrations.demo_pipeline.gap_point_run import sample_population

cv = load_cv([])
if cv is None:
    raise RuntimeError('verified OpenCV unavailable')
cv.setNumThreads(2)


class SampleBoundaryTests(unittest.TestCase):
    def test_mask_first_recovered_outside_never_saved_pending_not_observation(self):
        rng = np.random.default_rng(24)
        gray = rng.integers(0,256,(160,160),dtype=np.uint8)
        tracker = GapTracker(cv)
        tracker.update(gray,0)
        ids = tracker.seed([[40,40],[130,130]],0)
        tracker.pending[999] = dict(frame=0,xy=np.array([65,65],np.float32),neighbors={})
        mask = np.zeros_like(gray)
        mask[20:100,20:100] = 255
        with TemporaryDirectory() as d:
            p = Path(d)/'mask.png'
            Image.fromarray(mask).save(p)
            f = dict(input_index=3,decode_index=0,original_mask_status='OK',mask_path=str(p))
            with patch('integrations.demo_pipeline.balanced_object_tracks.seed_balanced',
                       return_value=(np.empty((0,2),np.float32),np.empty(0,np.int32))):
                row,stats = sample_population(tracker,f,gray,cv,Profile(),set(ids),set(ids))
        self.assertEqual([o['track_id'] for o in row['observations']],[int(ids[0])])
        self.assertTrue(row['observations'][0]['measured'])
        self.assertTrue(row['observations'][0]['recovered'])
        self.assertEqual(row['observations'][0]['xy'],[40.,40.])
        self.assertEqual(stats['filter']['mask_rejected'],1)
        self.assertIn(999,tracker.pending)
        self.assertNotIn(999,[o['track_id'] for o in row['observations']])

    def test_no_match_ends_active_and_pending_without_new_observation(self):
        gray = np.zeros((96,96),np.uint8)
        tracker = GapTracker(cv)
        tracker.update(gray,0)
        ids = tracker.seed([[40,40]],0)
        tracker.pending[100] = dict(frame=0,xy=np.array([60,60],np.float32),neighbors={})
        row,stats = sample_population(tracker,dict(original_mask_status='NO_MATCH'),
                                      gray,cv,Profile(),set(ids),set(ids))
        self.assertIsNone(row)
        self.assertEqual(len(tracker.ids),0)
        self.assertEqual(len(tracker.pending),0)
        self.assertEqual(set(stats['ended_ids']),{int(ids[0]),100})

    def test_recovery_history_not_equal_to_current_step(self):
        gray = np.zeros((96,96),np.uint8)
        tracker = GapTracker(cv)
        tracker.update(gray,0)
        ids = tracker.seed([[40,40]],0)
        with TemporaryDirectory() as d:
            p = Path(d)/'mask.png'
            Image.fromarray(np.full_like(gray,255)).save(p)
            row,stats = sample_population(tracker,dict(input_index=0,decode_index=0,
                original_mask_status='OK',mask_path=str(p)),gray,cv,Profile(),set(ids),set())
        self.assertTrue(row['observations'][0]['recovered'])
        self.assertFalse(row['observations'][0]['recovered_this_decode'])
        self.assertEqual(stats['born'],0)


if __name__ == '__main__':
    unittest.main()
