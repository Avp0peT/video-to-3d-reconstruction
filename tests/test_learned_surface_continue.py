import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from integrations.demo_pipeline import learned_surface_continue as continuation
from integrations.demo_pipeline.balanced_point_run import binding,read,save

PREVIOUS=continuation.ROOT/'outputs/delegation/NEW_VIDEO/20261019_new01/learned_tracking/surface_identity01'


class ContinuationTests(unittest.TestCase):
    @unittest.skipUnless((PREVIOUS/'STAGE_BINDINGS.json').exists(),'read-only real bridge fixture unavailable')
    def test_actual_make_plan_bridge_relative_names_and_frozen_sources(self):
        refs,scene=continuation.audit(PREVIOUS)
        before=[binding(r['path']) for r in refs]
        plan=continuation.make_surface_plan(PREVIOUS)
        self.assertEqual(len(plan['frames']),54)
        self.assertEqual(plan['arguments']['image_root'],str(PREVIOUS/'bridge'))
        self.assertEqual(plan['mask_source_chain'],'NOT_VERIFIED')
        self.assertEqual(plan['input_origin'],'caller_declared_unverified')
        self.assertIsNone(plan['photo_inputs'])
        for frame in plan['frames']:
            self.assertEqual(Path(frame['image_file']['path']),PREVIOUS/'bridge'/frame['image']['name'])
        self.assertEqual(before,[binding(r['path']) for r in refs])
        from integrations.mvs_surface.runner import make_plan
        with self.assertRaises(FileNotFoundError):
            make_plan(PREVIOUS/'bridge/scene.mvs',PREVIOUS/'bridge/undistorted/images',
                PREVIOUS/'masks/masks',PREVIOUS/'masks/validity',continuation.TOOL_ROOT,
                total_seconds=600,stage_seconds=480,memory_bytes=8<<30,output_bytes=1<<30,
                input_origin='unverified',ignore_label=0,photo_source=None)

    def fake_run(self,root,clean=True):
        previous=root/'previous';previous.mkdir()
        (previous/'bridge').mkdir()
        source=previous/'bridge/scene.mvs';source.write_bytes(b'frozen')
        plan=dict(scene=binding(source),tools=[],implementation=[],frames=[],input_origin='caller_declared_unverified',
                  mask_source_chain='NOT_VERIFIED',photo_inputs=None)
        calls=[]
        def invoke(out,label,argv,seconds,memory):
            calls.append((label,argv,seconds,memory))
            (out/'surface').mkdir()
            save(out/'surface/result.json',dict(status='SURFACE_GENERATED_NOT_ACCEPTED',exit_code=0))
            return dict(status='completed' if clean else 'completed_with_errors',root_exit_code=0,
                        errors=[],log_errors=[],reclaim_confirmed=True)
        with patch.object(continuation,'audit',return_value=([binding(source)],dict(images=[{},{}]))), \
             patch.object(continuation,'code_bindings',return_value=[]), \
             patch.object(continuation,'make_surface_plan',return_value=plan), \
             patch.object(continuation,'invoke',invoke):
            result=continuation.run(previous,root/'fresh')
        return result,calls,plan

    def test_continue_launches_only_surface_once_with_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result,calls,plan=self.fake_run(root)
            self.assertEqual(result['status'],'SURFACE_GENERATED_NOT_ACCEPTED')
            self.assertEqual(len(calls),1)
            self.assertEqual((calls[0][0],calls[0][2],calls[0][3]),('surface',660,10<<30))
            self.assertEqual(read(root/'fresh/SURFACE_PLAN.json'),plan)
            self.assertEqual(read(root/'fresh/INPUTS.json')['reused_stages'],['identity_export','bridge','masks'])
            self.assertFalse(result['geometry_accepted'])

    def test_unclean_surface_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls,_=self.fake_run(Path(tmp),clean=False)
            self.assertEqual(result['status'],'FAILED')
            self.assertEqual(result['failed_stage'],'surface')
            self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
