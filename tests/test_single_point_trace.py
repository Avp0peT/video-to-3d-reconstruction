"""Pure state tests plus explicitly supervised real OpenCV image tests."""
import json
import os
import unittest
from unittest import mock
import numpy as np
from integrations.demo_pipeline import single_point_trace as module


class PatchOnlyCV:
    def getRectSubPix(self,image,size,center):
        x,y=np.floor(np.asarray(center)+.5).astype(int); rx,ry=size[0]//2,size[1]//2
        return image[y-ry:y+ry+1,x-rx:x+rx+1].copy()


def proposal(xy,accepted,reason='accepted',fb=.1):
    return dict(xy=np.array([xy],np.float32),accepted=np.array([accepted]),reasons=np.array([reason]),
        fb_error=np.array([fb]),zncc=np.array([.9]),gain=np.array([1.1]),bias=np.array([4.]),
        photometric_residual=np.array([2.]))


class TraceStateTests(unittest.TestCase):
    def images(self,n=5):
        return [np.indices((48,48)).sum(axis=0).astype(np.uint8) for _ in range(n)]
    def test_failure_is_permanent_but_finite_shadow_can_continue(self):
        steps=[proposal([17,16],False,'forward_backward',2),proposal([18,17],True),
               proposal([18,17],False,'lk_forward',float('inf'))]
        with mock.patch.object(module,'photometric_step',side_effect=steps) as call:
            result=module.trace(self.images(),list(range(30,35)),[16,16],PatchOnlyCV())
        rows=result['frames'];self.assertEqual(call.call_count,3)
        self.assertEqual(result['first_failure']['decode_index'],31)
        self.assertEqual(rows[1]['candidate_status'],'UNVERIFIED_SHADOW')
        self.assertTrue(rows[2]['step_accepted']);self.assertFalse(rows[2]['continuous_accepted'])
        self.assertTrue(all(row['accepted_xy'] is None for row in rows[1:]))
        self.assertIsNone(rows[3]['candidate_xy']);self.assertEqual(rows[4]['candidate_status'],'NOT_TRACKED')
        self.assertIsNone(rows[3]['fb_error'])
        json.dumps(result,allow_nan=False)
    def test_forward_fallback_is_never_a_new_candidate(self):
        for reason in ('lk_forward','invalid_coordinates','invalid_native'):
            with self.subTest(reason=reason),mock.patch.object(module,'photometric_step',return_value=proposal([16,16],False,reason)) as call:
                result=module.trace(self.images(3),[0,1,2],[16,16],PatchOnlyCV())
                self.assertEqual(call.call_count,1)
                self.assertIsNone(result['frames'][1]['candidate_xy'])
                self.assertIsNone(result['frames'][2]['candidate_xy'])
                self.assertFalse(result['first_failure']['shadow_available'])
    def test_nonfinite_outside_and_inconsistent_native_acceptance_fail_closed(self):
        for point in ([float('nan'),16],[-1,16],[100,16]):
            with self.subTest(point=point),mock.patch.object(module,'photometric_step',return_value=proposal(point,True)):
                result=module.trace(self.images(2),[0,1],[16,16],PatchOnlyCV())
                row=result['frames'][1]
                self.assertFalse(row['step_accepted']);self.assertFalse(row['continuous_accepted'])
                self.assertEqual(row['reason'],'invalid_native_proposal');self.assertIsNone(row['candidate_xy'])
                json.dumps(result,allow_nan=False)
    def test_selected_seed_coordinates_and_fixed_profile_remain_explicit(self):
        point=[16.25,17.75]
        result=module.trace(self.images(1),[50],point,PatchOnlyCV())
        np.testing.assert_array_equal(result['seed_xy'],point)
        self.assertEqual(result['seed_origin'],'selected_saved_observation')
        self.assertFalse(result['physical_identity_verified']);self.assertFalse(result['geometry_accepted'])
        self.assertEqual(result['sfm_calls'],0)
        profile=result['profile'];self.assertEqual((profile['lk_window'],profile['pyramid'],profile['fb_max'],profile['patch'],profile['zncc_min']),(21,3,1.,15,.75))
    def test_invalid_inputs_reject_before_native_step(self):
        with mock.patch.object(module,'photometric_step') as call:
            for images,indices,seed in [(self.images(2),[0,2],[16,16]),(self.images(2),[0,True],[16,16]),
                (self.images(2),[0],[16,16]),(self.images(1),[0],[np.nan,16]),
                (self.images(1),[0],[1,16]),([np.zeros((48,48),float)],[0],[16,16])]:
                with self.assertRaises(ValueError):module.trace(images,indices,seed,PatchOnlyCV())
            call.assert_not_called()


@unittest.skipUnless(os.environ.get('SINGLE_POINT_TRACE_SUPERVISED')=='1','real OpenCV image checks require parent supervisor')
class SupervisedImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from integrations.demo_pipeline.temporal_review import load_cv
        cls.cv=load_cv([])
        if cls.cv is None:raise RuntimeError('bound CPU OpenCV required')
        cls.cv.setNumThreads(2);cls.cv.ocl.setUseOpenCL(False)
    def images(self):
        rng=np.random.default_rng(826)
        raw=self.cv.GaussianBlur(rng.integers(30,171,(320,320),dtype=np.uint8),(3,3),.6)
        frames=[]
        for index in range(5):
            moved=self.cv.warpAffine(raw,np.float32([[1,0,index],[0,1,index]]),(320,320))
            frames.append(np.clip(moved.astype(np.int16)+index*4,0,255).astype(np.uint8))
        return frames
    def test_real_translation_brightness_reports_known_coordinate_errors(self):
        result=module.trace(self.images(),list(range(5)),[160,160],self.cv)
        self.assertIsNone(result['first_failure'])
        errors=[]
        for index,row in enumerate(result['frames']):
            self.assertTrue(row['continuous_accepted'])
            error=float(np.linalg.norm(np.array(row['accepted_xy'])-[160+index,160+index]));errors.append(error)
            self.assertLess(error,.4)
        print(json.dumps(dict(kind='SYNTHETIC_IMAGE_KNOWN_TRANSLATION_BRIGHTNESS',errors_px=errors,
                             maximum_error_px=max(errors),wrong_accepted_gt1px=sum(e>1 for e in errors))),flush=True)
    def test_blank_occlusion_breaks_chain_and_later_texture_never_recertifies(self):
        frames=self.images();frames[2]=np.full_like(frames[2],128)
        result=module.trace(frames,list(range(5)),[160,160],self.cv)
        self.assertIsNotNone(result['first_failure'])
        self.assertEqual(result['first_failure']['decode_index'],2)
        for row in result['frames'][2:]:
            self.assertFalse(row['continuous_accepted']);self.assertIsNone(row['accepted_xy'])
            self.assertIn(row['candidate_status'],('UNVERIFIED_SHADOW','NOT_TRACKED'))
        json.dumps(result,allow_nan=False)
        print(json.dumps(dict(kind='SYNTHETIC_BLANK_OCCLUSION',first_failure=result['first_failure'],
                             statuses=[r['candidate_status'] for r in result['frames']])),flush=True)

if __name__=='__main__':unittest.main()
