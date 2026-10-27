"""Synthetic image measurements; not real-video quality acceptance."""
import inspect
import json
import unittest
import numpy as np
from integrations.demo_pipeline.temporal_review import load_cv
from integrations.demo_pipeline.balanced_object_tracks import Profile,seed_balanced,step_tracks
from integrations.demo_pipeline.gap_tolerant_tracks import GapTracker
from integrations.demo_pipeline.photometric_tracks import PhotometricProfile,photometric_step,PhotometricGapTracker
cv=load_cv([])
cv.setNumThreads(2)
MEASUREMENTS=[]

class PhotometricTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(826)
        self.image=cv.GaussianBlur(rng.integers(30,171,(320,320),dtype=np.uint8),(3,3),.6)
        self.points,_=seed_balanced(self.image,np.full_like(self.image,255),[],cv)
        self.points=self.points[(self.points[:,0]>35)&(self.points[:,0]<285)&(self.points[:,1]>35)&(self.points[:,1]<285)]
        self.shift=np.float32([1.5,1.0])
        self.moved=cv.warpAffine(self.image,np.float32([[1,0,1.5],[0,1,1]]),(320,320))
    def measurement(self,name,image):
        metrics={}
        results={}
        for label,method in [('raw',step_tracks),('normalized',photometric_step)]:
            result=method(self.image,image,self.points,cv)
            error=np.linalg.norm(result['xy']-self.points-self.shift,axis=1)
            accepted=result['accepted']
            metrics[label]={'accepted':int(accepted.sum()),'total':len(accepted),
                'median_error_px':float(np.median(error[accepted])) if accepted.any() else None,
                'p95_error_px':float(np.quantile(error[accepted],.95)) if accepted.any() else None,
                'wrong_accepted_gt1px':int(np.sum(accepted&(error>1)))}
            results[label]=result
        row={'synthetic_case':name,'metrics':metrics}; MEASUREMENTS.append(row)
        print(json.dumps(row,allow_nan=False),flush=True)
        accepted=results['normalized']['accepted']
        error=np.linalg.norm(results['normalized']['xy']-self.points-self.shift,axis=1)
        self.assertGreater(accepted.mean(),.5)
        self.assertLess(float(np.median(error[accepted])),.3)
        self.assertLess(float(np.quantile(error[accepted],.95)),.6)
        self.assertEqual(int(np.sum(accepted&(error>1))),0)
        return results
    def test_known_motion_uniform_gain_and_offset(self):
        self.measurement('uniform_offset_30',np.clip(self.moved.astype(float)+30,0,255).astype(np.uint8))
        self.measurement('uniform_gain_1.35_bias_10',np.clip(self.moved.astype(float)*1.35+10,0,255).astype(np.uint8))
    def test_known_motion_smooth_local_shadow(self):
        gain=np.linspace(.55,1.35,320)[None,:]
        self.measurement('smooth_spatial_gain_.55_to_1.35',np.clip(self.moved*gain+10,0,255).astype(np.uint8))
    def test_same_motion_without_photometric_change(self):
        self.measurement('same_motion_no_light_change',self.moved)
    def test_repeated_texture_negative_control_reports_aliasing(self):
        tile=np.random.default_rng(5).integers(30,170,(8,8),dtype=np.uint8)
        original=cv.GaussianBlur(np.tile(tile,(40,40)),(3,3),.5)
        moved=np.roll(original,9,axis=1)
        points,_=seed_balanced(original,np.full_like(original,255),[],cv)
        points=points[(points[:,0]>35)&(points[:,0]<285)&(points[:,1]>35)&(points[:,1]<285)]
        metrics={}
        for label,method in [('raw',step_tracks),('normalized',photometric_step)]:
            result=method(original,moved,points,cv)
            error=np.linalg.norm(result['xy']-points-[9,0],axis=1); keep=result['accepted']
            metrics[label]={'accepted':int(keep.sum()),'total':len(points),
                'median_error_px':float(np.median(error[keep])) if keep.any() else None,
                'p95_error_px':float(np.quantile(error[keep],.95)) if keep.any() else None,
                'wrong_accepted_gt1px':int(np.sum(keep&(error>1)))}
        row={'synthetic_case':'period8_texture_true_shift9_alias_control',
             'limitation':'Local LK and photometric verification cannot resolve whole-period aliases.', 'metrics':metrics}
        MEASUREMENTS.append(row); print(json.dumps(row),flush=True)
    def test_unrelated_texture_saturation_and_nonfinite(self):
        unrelated=np.random.default_rng(456).integers(0,256,self.image.shape,dtype=np.uint8)
        result=photometric_step(self.image,unrelated,self.points,cv)
        metrics={}
        for label,method in [('raw',step_tracks),('normalized',photometric_step)]:
            rejected=method(self.image,unrelated,self.points,cv)
            metrics[label]={'accepted':int(rejected['accepted'].sum()),'total':len(self.points),
                'median_error_px':None,'p95_error_px':None,'wrong_accepted_gt1px':None,
                'false_accepts_unrelated':int(rejected['accepted'].sum())}
        row={'synthetic_case':'unrelated_no_true_correspondence','metrics':metrics}
        MEASUREMENTS.append(row); print(json.dumps(row),flush=True)
        self.assertLessEqual(int(result['accepted'].sum()),2)
        flat=np.full_like(self.image,128)
        self.assertFalse(photometric_step(flat,flat,self.points,cv)['accepted'].any())
        checker=(np.indices(self.image.shape).sum(axis=0)%2*255).astype(np.uint8)
        self.assertFalse(photometric_step(checker,checker,self.points,cv)['accepted'].any())
        saturated=(np.random.default_rng(77).integers(0,2,self.image.shape)*255).astype(np.uint8)
        saturation_result=photometric_step(saturated,saturated,self.points,cv)
        self.assertFalse(saturation_result['accepted'].any())
        self.assertTrue(np.any(saturation_result['reasons']=='photometric_saturation'))
        invalid=np.float32([[np.nan,30],[30,np.inf],[-4,10]])
        self.assertFalse(photometric_step(self.image,self.image,invalid,cv)['accepted'].any())
        with self.assertRaises(ValueError): photometric_step(self.image.astype(float),self.image,self.points,cv)
        with self.assertRaises(ValueError): PhotometricProfile(gain_max=float('inf'))
    def test_raw_gain_bias_admission_gates_are_active(self):
        changed=np.clip(self.moved.astype(float)*1.35+10,0,255).astype(np.uint8)
        gain_result=photometric_step(self.image,changed,self.points,cv,
                                    photometric=PhotometricProfile(gain_max=1.05))
        self.assertTrue(np.any(gain_result['reasons']=='photometric_gain'))
        offset=np.clip(self.moved.astype(float)+30,0,255).astype(np.uint8)
        bias_result=photometric_step(self.image,offset,self.points,cv,
                                    photometric=PhotometricProfile(bias_abs_max=0))
        self.assertTrue(np.any(bias_result['reasons']=='photometric_bias'))
    def test_gap_uses_raw_image_ring_and_preserves_missing_indices(self):
        tracker=PhotometricGapTracker(cv,Profile(pyramid=0))
        points=np.float32([[160,160]]+[[x,y] for y in [100,130,160,190,220] for x in [100,130,160,190,220] if (x,y)!=(160,160)])
        tracker.update(self.image,0); ids=tracker.seed(points,0)
        frame1=cv.warpAffine(self.image,np.float32([[1,0,1],[0,1,1]]),(320,320))
        frame1=np.clip(frame1.astype(float)*1.15+10,0,255).astype(np.uint8)
        frame1[151:172,151:172]=128
        first=tracker.update(frame1,1)
        self.assertNotIn(ids[0],first['active_ids'])
        np.testing.assert_array_equal(tracker.images[1],frame1)
        frame2=cv.warpAffine(self.image,np.float32([[1,0,2],[0,1,2]]),(320,320))
        frame2=np.clip(frame2.astype(float)*1.2+12,0,255).astype(np.uint8)
        second=tracker.update(frame2,2)
        self.assertIn(ids[0],second['recovered_ids'])
        event=next(e for e in second['events'] if e['kind']=='recovered' and e['id']==ids[0])
        self.assertEqual(event['missing_decode_indices'],[1])
        row=np.flatnonzero(second['active_ids']==ids[0])[0]
        np.testing.assert_allclose(second['active_xy'][row],[162,162],atol=.3)
    def test_override_preserves_base_update_policy_exactly(self):
        old=inspect.getsource(GapTracker.update)
        new=inspect.getsource(PhotometricGapTracker.update)
        new=new.replace('move=photometric_step(self.images[previous],gray,oldxy,self.cv,self.profile,self.photometric)',
                        'move=step_tracks(self.images[previous],gray,oldxy,self.cv,self.profile)')
        self.assertEqual(old,new)

if __name__=='__main__': unittest.main()
