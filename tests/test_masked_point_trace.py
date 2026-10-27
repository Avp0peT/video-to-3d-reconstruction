"""Synthetic policy tests; real OpenCV cases require explicit supervision."""
import os
import unittest
from unittest.mock import patch
import numpy as np
from integrations.demo_pipeline.masked_point_trace import trace, _InitialFlow, _mask_metrics


def result(xy, reason='accepted'):
    out=dict(xy=np.array([xy],np.float32),accepted=np.array([reason=='accepted']),reasons=np.array([reason]))
    out.update({k:np.array([0.9]) for k in ('fb_error','zncc','gain','bias','photometric_residual')})
    return out


class MaskedTraceTests(unittest.TestCase):
    def setUp(self):
        self.images=[np.full((80,80),100,np.uint8) for _ in range(5)]
        self.masks=[np.full((80,80),255,np.uint8) for _ in range(5)]

    def run_trace(self, values, masks=None, predict=False, times=None):
        n=len(values)+1
        with patch('integrations.demo_pipeline.masked_point_trace.photometric_step',side_effect=values) as step:
            out=trace(self.images[:n],(masks or self.masks)[:n],list(range(n)),[20,20],object(),predict_motion=predict,times=times)
        return out,step

    def test_current_moving_mask_authorizes_current_point(self):
        self.masks[0][:]=0; self.masks[0][20,20]=255
        self.masks[1][:]=0; self.masks[1][20,30]=255
        out,_=self.run_trace([result([30,20])])
        self.assertEqual(out['frames'][1]['accepted_xy'],[30.,20.])
        self.assertAlmostEqual(out['frames'][1]['mask']['patch15_foreground_fraction'],1/225)

    def test_outside_is_not_snapped_to_nearby_foreground(self):
        self.masks[1][:]=0; self.masks[1][20,31]=255
        out,_=self.run_trace([result([30,20])])
        self.assertIsNone(out['frames'][1]['accepted_xy'])
        self.assertIsNone(out['frames'][1]['candidate_xy'])
        self.assertEqual(out['frames'][1]['raw_proposal_xy'],[30.,20.])

    def test_missing_mask_stops_flow_and_no_bridge(self):
        masks=self.masks.copy(); masks[1]=None
        out,step=self.run_trace([result([21,20]),result([22,20])],masks)
        self.assertEqual(step.call_count,0)
        self.assertTrue(all(r['accepted_xy'] is None for r in out['frames'][1:]))

    def test_failure_never_revives_or_predicts_from_shadow(self):
        out,_=self.run_trace([result([21,20],'photometric_zncc'),result([22,20]),result([23,20])],predict=True)
        self.assertTrue(all(r['accepted_xy'] is None for r in out['frames'][1:]))
        self.assertTrue(all(not r['motion']['attempted'] for r in out['frames'][1:]))
        self.assertEqual(out['frames'][2]['candidate_status'],'UNVERIFIED_SHADOW')

    def test_native_fallback_not_a_shadow(self):
        out,_=self.run_trace([result([20,20],'lk_forward'),result([21,20])])
        self.assertIsNone(out['frames'][1]['candidate_xy'])
        self.assertIsNone(out['frames'][2]['candidate_xy'])

    def test_two_qualified_candidates_ambiguous(self):
        values=[result([21,20]),result([22,20]),result([30,20])]
        with patch('integrations.demo_pipeline.masked_point_trace.photometric_step',side_effect=values):
            out=trace(self.images[:3],self.masks[:3],[0,1,2],[20,20],object(),predict_motion=True)
        self.assertEqual(out['frames'][2]['reason'],'AMBIGUOUS')
        self.assertIsNone(out['frames'][2]['accepted_xy'])

    def test_time_velocity_and_merged_candidates(self):
        with patch('integrations.demo_pipeline.masked_point_trace.photometric_step',side_effect=[result([22,20]),result([26,20]),result([27,20])]):
            out=trace(self.images[:3],self.masks[:3],[0,1,2],[20,20],object(),predict_motion=True,times=[0,1,3])
        self.assertEqual(out['frames'][2]['motion']['predicted_xy'],[26.,20.])
        self.assertEqual(out['frames'][2]['accepted_xy'],[26.,20.])

    def test_rounding_boundary(self):
        self.assertFalse(_mask_metrics(self.masks[0],np.array([79.5,20]))['center_inside'])
        self.assertTrue(_mask_metrics(self.masks[0],np.array([79.49,20]))['center_inside'])

    def test_nonbinary_labels_rejected(self):
        self.masks[0][0,0]=2
        with self.assertRaisesRegex(ValueError,'0/255'):
            self.run_trace([result([21,20])])

    def test_prediction_over_limit_uses_baseline_only(self):
        with patch('integrations.demo_pipeline.masked_point_trace.photometric_step',side_effect=[result([22,20]),result([24,20])]) as step:
            out=trace(self.images[:3],self.masks[:3],[0,1,2],[20,20],object(),predict_motion=True,times=[0,1,101])
        self.assertEqual(step.call_count,2)
        self.assertEqual(out['frames'][2]['motion']['reason'],'over_96px')
        self.assertFalse(out['frames'][2]['motion']['attempted'])

    def test_proxy_forwards_initial_guesses_and_options(self):
        class CV:
            OPTFLOW_USE_INITIAL_FLOW=4
            def __init__(self): self.calls=[]
            def calcOpticalFlowPyrLK(self,*args,**kwargs): self.calls.append((args,kwargs)); return None,None,None
        cv=CV(); proxy=_InitialFlow(cv,np.array([20,20],np.float32),np.array([25,20],np.float32))
        for _ in range(2): proxy.calcOpticalFlowPyrLK(None,None,np.zeros((1,1,2)),None,winSize=(21,21),maxLevel=3)
        self.assertEqual(cv.calls[0][0][3].reshape(2).tolist(),[25.,20.])
        self.assertEqual(cv.calls[1][0][3].reshape(2).tolist(),[20.,20.])
        self.assertEqual(cv.calls[1][1],dict(winSize=(21,21),maxLevel=3,flags=4))


@unittest.skipUnless(os.environ.get('MASKED_TRACE_SUPERVISED')=='1','requires supervised real CPU execution')
class RealCPUTraceTests(unittest.TestCase):
    def test_translation_and_brightness(self):
        import cv2
        rng=np.random.default_rng(28)
        first=rng.integers(35,180,(128,128),dtype=np.uint8)
        images=[cv2.warpAffine(first,np.float32([[1,0,4*i],[0,1,0]]),(128,128)) for i in range(3)]
        images=[np.clip(im.astype(float)*1.1+5,0,255).astype(np.uint8) if i else im for i,im in enumerate(images)]
        out=trace(images,[np.full_like(first,255)]*3,[0,1,2],[50,60],cv2,predict_motion=True)
        self.assertIsNotNone(out['frames'][-1]['accepted_xy'])
        self.assertLess(np.linalg.norm(np.array(out['frames'][-1]['accepted_xy'])-[58,60]),1)

    def test_repeated_texture_retains_identity_disclaimer(self):
        import cv2
        yy,xx=np.indices((128,128)); im=((np.sin(xx*.8)+np.cos(yy*.8))*35+120).astype(np.uint8)
        images=[cv2.warpAffine(im,np.float32([[1,0,12*i],[0,1,0]]),(128,128)) for i in range(3)]
        out=trace(images,[np.full_like(im,255)]*3,[0,1,2],[50,60],cv2,predict_motion=True)
        self.assertFalse(out['physical_identity_verified'])
        for r in out['frames']:
            if r['accepted_xy'] is not None: self.assertTrue(r['mask']['center_inside'])


if __name__=='__main__': unittest.main()
