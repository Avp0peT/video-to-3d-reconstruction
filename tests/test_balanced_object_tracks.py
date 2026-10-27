"""Synthetic image correctness tests; these are not video quality measurements."""
import unittest
import numpy as np
from integrations.demo_pipeline.temporal_review import load_cv
cv = load_cv([])
if cv is None: raise RuntimeError('bound CPU OpenCV unavailable')
from integrations.demo_pipeline.balanced_object_tracks import (
    Profile, seed_balanced, step_tracks, coverage, eroded_mask, points_in_mask, rebalance)

cv.setNumThreads(2)


class BalancedTracksTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(20261019)
        self.image=self.rng.integers(0,256,(256,256),dtype=np.uint8)
        self.image=cv.GaussianBlur(self.image,(3,3),0.6)
        self.mask=np.full(self.image.shape,255,np.uint8)

    def test_balanced_competition_and_replenishment(self):
        image=self.image.copy()
        image[:128,:128]=self.rng.integers(0,256,(128,128),dtype=np.uint8)
        xy,cells=seed_balanced(image,self.mask,[],cv)
        self.assertGreaterEqual(len(xy),500)
        self.assertEqual(len(np.unique(cells)),64)
        self.assertLessEqual(max(np.bincount(cells)),12)
        more,_=seed_balanced(image,self.mask,xy,cv)
        self.assertLessEqual(len(xy)+len(more),768)
        self.assertTrue(points_in_mask(eroded_mask(self.mask,cv),xy).all())
        self.assertEqual(coverage(self.mask,xy)['fraction'],1.0)

    def test_no_texture_no_mask_no_fabrication(self):
        for image,mask in [(np.zeros_like(self.image),self.mask),(self.image,np.zeros_like(self.mask))]:
            xy,cells=seed_balanced(image,mask,[],cv)
            self.assertEqual(len(xy),0)
            self.assertEqual(len(cells),0)
        mask=np.zeros_like(self.mask); mask[40:160,40:160]=255; mask[70:130,70:130]=0
        xy,_=seed_balanced(self.image,mask,[],cv)
        self.assertTrue(points_in_mask(eroded_mask(mask,cv),xy).all())
        self.assertLess(coverage(mask,xy)['supported_cells'],64)

    def test_translation_brightness_and_row_identity(self):
        xy,_=seed_balanced(self.image,self.mask,[],cv)
        xy=xy[(xy[:,0]>25)&(xy[:,0]<225)&(xy[:,1]>25)&(xy[:,1]<225)]
        moved=cv.warpAffine(self.image,np.float32([[1,0,2],[0,1,1]]),(256,256))
        bright=np.clip(moved.astype(np.int16)+15,0,255).astype(np.uint8)
        bright_result=step_tracks(self.image,bright,xy,cv)
        self.assertGreater(bright_result['accepted'].mean(),0.6)
        result=step_tracks(self.image,moved,xy,cv)
        keep=result['accepted']
        self.assertGreater(keep.mean(),0.6)
        self.assertLess(np.median(np.linalg.norm(result['xy'][keep]-xy[keep]-[2,1],axis=1)),0.15)
        duplicate=np.repeat(xy[keep][:1],2,axis=0)
        result=step_tracks(self.image,moved,duplicate,cv)
        self.assertEqual(result['xy'].shape,(2,2))
        self.assertTrue(result['accepted'].all())

    def test_unrelated_and_occluded_reject(self):
        xy,_=seed_balanced(self.image,self.mask,[],cv)
        unrelated=self.rng.integers(0,256,self.image.shape,dtype=np.uint8)
        result=step_tracks(self.image,unrelated,xy,cv)
        self.assertLess(result['accepted'].mean(),0.05)
        occluded=self.image.copy(); occluded[60:190,60:190]=128
        result=step_tracks(self.image,occluded,xy,cv)
        inside=(xy[:,0]>75)&(xy[:,0]<175)&(xy[:,1]>75)&(xy[:,1]<175)
        self.assertFalse(result['accepted'][inside].any())

    def test_invalid_coordinates_fail_closed(self):
        xy=np.array([[np.nan,20],[20,np.inf],[-1,40],[999,20],[1e30,1e30]],np.float32)
        result=step_tracks(self.image,self.image,xy,cv)
        self.assertFalse(result['accepted'].any())
        self.assertTrue(np.isinf(result['fb_error']).all())
        with self.assertRaises(ValueError): seed_balanced(self.image,self.mask,xy,cv)
        with self.assertRaises(ValueError): seed_balanced(self.image, np.ones_like(self.mask), [], cv)
        for kwargs in [{'max_live':769},{'per_cell':13},{'zncc_min':0.7},{'patch':14},{'fb_max':float('nan')}]:
            with self.assertRaises(ValueError): Profile(**kwargs)

    def test_roi_offsets_match_full_image_detector_candidates(self):
        mask=np.zeros_like(self.mask); mask[50:210,60:200]=255
        profile=Profile(grid=1,max_live=12)
        calls=[]
        class RecordingCV:
            def __getattr__(self,name): return getattr(cv,name)
            def goodFeaturesToTrack(self,image,**kwargs):
                calls.append(image.shape)
                return cv.goodFeaturesToTrack(image,**kwargs)
        xy,cells=seed_balanced(self.image,mask,[],RecordingCV(),profile)
        self.assertEqual(calls,[(152,132)])
        self.assertTrue((cells==0).all())
        full=cv.goodFeaturesToTrack(self.image,maxCorners=48,qualityLevel=.001,
            minDistance=5,mask=eroded_mask(mask,cv,profile),blockSize=3,useHarrisDetector=False).reshape(-1,2)
        self.assertEqual(len(xy),12)
        for point in xy:
            self.assertTrue(np.any(np.all(full==point,axis=1)))
        self.assertTrue((xy[:,0]>=67).all())
        self.assertTrue((xy[:,1]>=57).all())

    def test_patch_support_excludes_mask_boundaries_and_holes(self):
        mask=np.zeros_like(self.mask); mask[30:220,40:210]=255
        mask[100:140,100:140]=0
        safe=eroded_mask(mask,cv)
        points=np.float32([[46,60],[47,60],[60,36],[60,37],[93,120],[92,120]])
        self.assertEqual(points_in_mask(safe,points).tolist(),[False,True,False,True,False,True])
        xy,_=seed_balanced(self.image,mask,[],cv)
        self.assertGreater(len(xy),0)
        for x,y in xy.astype(int):
            self.assertTrue((mask[y-7:y+8,x-7:x+8]==255).all())
        with self.assertRaises(ValueError): Profile(erosion=3)

    def test_migration_quota_retires_newer_ids_and_replenishes_gaps(self):
        profile=Profile(grid=2,per_cell=2,max_live=8)
        # Eight persistent tracks moved into one cell: six retire only forward.
        migrated=np.float32([[30+i*7,30] for i in range(8)])
        identities=np.array([80,10,70,20,60,30,50,40])
        original=migrated.copy()
        keep=rebalance(self.mask,migrated,identities,profile)
        self.assertEqual(identities[keep].tolist(),[10,20])
        np.testing.assert_array_equal(migrated,original)
        births,cells=seed_balanced(self.image,self.mask,migrated[keep],cv,profile)
        self.assertEqual(len(births),6)
        self.assertEqual(set(cells.tolist()),{1,2,3})
        combined=np.concatenate([migrated[keep],births])
        self.assertTrue(rebalance(self.mask,combined,np.arange(8),profile).all())

    def test_half_pixel_mask_lookup_and_quota_agree(self):
        mask=np.zeros_like(self.mask); mask[20:100,20:100]=255
        points=np.float32([[19.49,40],[19.5,40],[99.49,40],[99.5,40],
                          [40,19.5],[40,99.5],[np.nan,40]])
        expected=points_in_mask(mask,points)
        kept=rebalance(mask,points,np.arange(len(points)),Profile(grid=1))
        self.assertEqual(expected.tolist(),[False,True,True,False,True,False,False])
        np.testing.assert_array_equal(kept,expected)
        with self.assertRaises(ValueError): rebalance(mask,points,np.zeros(len(points),int))


if __name__=='__main__': unittest.main()
