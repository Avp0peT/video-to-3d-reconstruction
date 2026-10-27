"""Synthetic CPU image evidence only, not real-video acceptance."""
import unittest
from unittest.mock import patch
import numpy as np
from integrations.demo_pipeline.temporal_review import load_cv
from integrations.demo_pipeline.balanced_object_tracks import Profile
from integrations.demo_pipeline.gap_tolerant_tracks import GapTracker,RecoveryProfile
cv=load_cv([])
cv.setNumThreads(2)

class GapTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123)
        self.image=cv.GaussianBlur(rng.integers(20,230,(240,240),dtype=np.uint8),(3,3),.5)
        self.points=np.array([[120,120]]+[[x,y] for y in [60,90,120,150,180] for x in [60,90,120,150,180] if (x,y)!=(120,120)],np.float32)
        self.profile=Profile(pyramid=0)
    def frame(self,n,hide=False):
        out=cv.warpAffine(self.image,np.float32([[1,0,n],[0,1,n]]),(240,240))
        out=np.clip(out.astype(np.int16)+n*4,0,255).astype(np.uint8)
        if hide: out[110+n:131+n,110+n:131+n]=128
        return out
    def tracker(self):
        tracker=GapTracker(cv,self.profile)
        tracker.update(self.frame(0),0); ids=tracker.seed(self.points,0)
        return tracker,ids
    def test_real_translation_brightness_short_occlusion_recovers_id(self):
        tracker,ids=self.tracker(); first=tracker.update(self.frame(1,True),1)
        self.assertNotIn(ids[0],first['active_ids'])
        self.assertGreater(first['pending_count'],0)
        result=tracker.update(self.frame(2),2)
        self.assertIn(ids[0],result['recovered_ids'])
        row=np.flatnonzero(result['active_ids']==ids[0])[0]
        np.testing.assert_allclose(result['active_xy'][row],[122,122],atol=.3)
        event=next(e for e in result['events'] if e['kind']=='recovered' and e['id']==ids[0])
        self.assertEqual(event['missing_decode_indices'],[1])
        self.assertEqual(event['decode_index'],2)
    def test_expiry_without_neighbors_and_no_guessed_observation(self):
        tracker=GapTracker(cv,self.profile); tracker.update(self.frame(0),0); ids=tracker.seed(self.points[:1],0)
        for index in range(1,5):
            result=tracker.update(self.frame(index,True),index)
            self.assertNotIn(ids[0],result['active_ids'])
        self.assertEqual(result['pending_count'],0)
        self.assertTrue(any(e['kind']=='expired' for e in result['events']))
    def test_unrelated_image_does_not_recover(self):
        tracker,ids=self.tracker(); tracker.update(self.frame(1,True),1)
        unrelated=np.random.default_rng(44).integers(0,256,(240,240),dtype=np.uint8)
        result=tracker.update(unrelated,2)
        self.assertNotIn(ids[0],result['recovered_ids'])
    def test_duplicate_texture_ambiguity_rejected(self):
        tracker,ids=self.tracker(); tracker.update(self.frame(1,True),1)
        # An exact duplicate template four pixels away creates an ambiguous peak.
        image=self.frame(2); patch=self.image[113:128,113:128]
        # Directly test the recovery matcher with measured consistent neighbors.
        pending=tracker.pending[int(ids[0])]
        current={i:p+[2,2] for i,p in pending['neighbors'].items() if i!=ids[0]}
        repeated=np.tile(np.array([0,70,180,240],np.uint8),(240,60))
        tracker.images[0]=repeated
        query=np.roll(np.roll(repeated,2,axis=0),2,axis=1)
        q,reason=tracker._recover(pending,query,current)
        self.assertIsNone(q)
        self.assertIn(reason,['ambiguous_peak','zero_texture'])
    def test_one_to_one_active_collision_no_id_merging(self):
        tracker,ids=self.tracker(); tracker.update(self.frame(1,True),1)
        # Insufficient support defers the old ID while a measured new seed is born.
        tracker.recovery=RecoveryProfile(min_neighbors=32)
        tracker.update(self.frame(2),2)
        new=tracker.seed([[122,122]],2)[0]
        tracker.recovery=RecoveryProfile()
        result=tracker.update(self.frame(3),3)
        self.assertNotIn(ids[0],result['recovered_ids'])
        self.assertIn(new,result['active_ids'])
        self.assertEqual(len(result['active_ids']),len(set(result['active_ids'].tolist())))
    def test_mask_and_clear_terminate_without_recovery(self):
        tracker,ids=self.tracker(); tracker.update(self.frame(1,True),1)
        result=tracker.filter_active(np.zeros_like(self.image))
        self.assertGreater(len(result['mask_retired_ids']),0)
        self.assertEqual(len(tracker.ids),0); self.assertGreater(len(tracker.pending),0)
        tracker.clear('NO_MATCH')
        self.assertEqual(len(tracker.pending),0)
        result=tracker.update(self.frame(2),2)
        self.assertEqual(len(result['active_ids']),0)
        tracker.seed([[122,122]],2); event=tracker.clear('unknown_mask')
        self.assertEqual(event['reason'],'unknown_mask'); self.assertEqual(len(tracker.ids),0)
    def test_pending_crosses_committed_mask_without_missing_observation(self):
        tracker,ids=self.tracker(); lost=tracker.update(self.frame(1,True),1)
        self.assertNotIn(ids[0],lost['active_ids'])
        tracker.filter_active(np.full_like(self.image,255),quotas=False)
        self.assertIn(int(ids[0]),tracker.pending)
        self.assertNotIn(ids[0],tracker.ids)
        recovered=tracker.update(self.frame(2),2)
        self.assertIn(ids[0],recovered['recovered_ids'])
        event=next(e for e in recovered['events'] if e['kind']=='recovered' and e['id']==ids[0])
        self.assertEqual(event['missing_decode_indices'],[1])
    def test_final_motion_inliers_must_span_and_surround_target(self):
        tracker=GapTracker(cv,self.profile); tracker.update(self.image,0)
        target=np.float32([120,120])
        for support,expected in [
            (np.float32([[70,70],[72,70],[74,70],[70,72],[72,72],[74,72]]),'inlier_geometry'),
            (np.float32([[60,70],[80,70],[100,70],[120,70],[140,70],[160,70]]),'inlier_geometry'),
            (np.float32([[65,65],[95,65],[115,65],[65,95],[95,95],[115,95]]),'outside_neighbor_support')]:
            # Peripheral outliers make the initial cloud look spatially broad.
            outliers=np.float32([[160,160],[170,90],[90,170]])
            a=np.vstack((support,outliers))
            snapshot={i:q for i,q in enumerate(a)}
            current={i:q+np.float32([1,1]) for i,q in enumerate(a)}
            for i in range(6,9): current[i]=current[i]+[30*(i-5),-25*(i-5)]
            item={'frame':0,'xy':target,'neighbors':snapshot}
            q,reason=tracker._recover(item,self.frame(1),current)
            self.assertIsNone(q)
            self.assertIn(reason,['neighbor_motion','insufficient_motion_inliers','inlier_geometry',
                                  'outside_neighbor_support','neighbor_geometry'])
            # Isolate the final admission gate after an ideal robust motion fit:
            # the wide outlier cloud must never substitute for inlier support.
            with patch('integrations.demo_pipeline.gap_tolerant_tracks._similarity',
                       return_value=np.float64([[1,0,1],[0,1,1]])):
                q,reason=tracker._recover(item,self.frame(1),current)
            self.assertIsNone(q)
            self.assertEqual(reason,expected)
    def test_two_recoveries_cannot_claim_same_measurement(self):
        tracker=GapTracker(cv,self.profile); tracker.update(self.frame(0),0)
        ids=tracker.seed(np.vstack((self.points[:1],self.points)),0)
        tracker.update(self.frame(1,True),1)
        result=tracker.update(self.frame(2),2)
        self.assertNotIn(ids[0],result['recovered_ids'])
        self.assertNotIn(ids[1],result['recovered_ids'])
        self.assertGreaterEqual(result['rejections'].get('recovery_collision',0),2)
    def test_reverse_similar_patch_ambiguity_rejects_false_return(self):
        tracker,ids=self.tracker(); tracker.update(self.frame(1,True),1)
        pending=tracker.pending[int(ids[0])]
        current={i:p+[2,2] for i,p in pending['neighbors'].items() if i!=ids[0]}
        forward=np.zeros((17,17),np.float32); forward[8,8]=.99
        reverse=forward.copy(); reverse[8,13]=.98
        class AmbiguousReverseCV:
            def __init__(self): self.calls=0
            def __getattr__(self,name): return getattr(cv,name)
            def matchTemplate(self,*args):
                self.calls+=1
                return forward.copy() if self.calls==1 else reverse.copy()
        # Controlled correlation responses isolate reciprocal ambiguity admission;
        # actual image matching/occlusion/periodic-texture paths are tested above.
        tracker.cv=AmbiguousReverseCV()
        q,reason=tracker._recover(pending,self.frame(2),current)
        self.assertIsNone(q); self.assertEqual(reason,'reverse_ambiguous_peak')
    def test_native_coordinate_seed_and_bounds(self):
        tracker=GapTracker(cv,self.profile); tracker.update(self.image,0)
        points=np.array([[40.25,50.75]],np.float32); tracker.seed(points,0)
        np.testing.assert_array_equal(tracker.xy,points)
        with self.assertRaises(ValueError): tracker.seed([[np.nan,2]],0)
        with self.assertRaises(ValueError): tracker.update(self.image,2)
        with self.assertRaises(ValueError): RecoveryProfile(max_gap=4)

if __name__=='__main__': unittest.main()
