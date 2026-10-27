"""Integration of new tracking with unchanged measured-only sampling rules."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np
from PIL import Image

from integrations.demo_pipeline.temporal_review import load_cv
from integrations.demo_pipeline.balanced_object_tracks import Profile
from integrations.demo_pipeline.photometric_tracks import PhotometricGapTracker
from integrations.demo_pipeline.photometric_point_run import sample_population

cv = load_cv([])
cv.setNumThreads(2)


class PhotometricSamplingTests(unittest.TestCase):
    def test_original_mask_filters_new_path_before_observation_pending_stays_absent(self):
        gray = np.random.default_rng(13).integers(40,180,(160,160),dtype=np.uint8)
        tracker = PhotometricGapTracker(cv)
        tracker.update(gray,0)
        ids = tracker.seed([[40,40],[130,130]],0)
        tracker.pending[999] = dict(frame=0,xy=np.float32([65,65]),neighbors={})
        mask = np.zeros_like(gray)
        mask[20:100,20:100] = 255
        with TemporaryDirectory() as directory:
            path = Path(directory)/'mask.png'
            Image.fromarray(mask).save(path)
            frame = dict(input_index=3,decode_index=0,original_mask_status='OK',mask_path=str(path))
            with patch('integrations.demo_pipeline.balanced_object_tracks.seed_balanced',
                       return_value=(np.empty((0,2),np.float32),np.empty(0,np.int32))):
                row,stats = sample_population(tracker,frame,gray,cv,Profile(),set(),set())
        self.assertEqual([o['track_id'] for o in row['observations']],[int(ids[0])])
        self.assertEqual(row['observations'][0]['xy'],[40.,40.])
        self.assertTrue(row['observations'][0]['measured'])
        self.assertEqual(stats['filter']['mask_rejected'],1)
        self.assertIn(999,tracker.pending)

    def test_no_match_still_ends_new_active_and_pending_tracks(self):
        gray = np.full((96,96),128,np.uint8)
        tracker = PhotometricGapTracker(cv)
        tracker.update(gray,0)
        ids = tracker.seed([[40,40]],0)
        tracker.pending[999] = dict(frame=0,xy=np.float32([65,65]),neighbors={})
        row,stats = sample_population(tracker,dict(original_mask_status='NO_MATCH'),
            gray,cv,Profile(),set(),set())
        self.assertIsNone(row)
        self.assertEqual(len(tracker.ids),0)
        self.assertEqual(tracker.pending,{})
        self.assertEqual(set(stats['ended_ids']),{int(ids[0]),999})


if __name__ == '__main__':
    unittest.main()
