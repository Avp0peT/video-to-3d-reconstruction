import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from integrations.demo_pipeline import learned_surface_run as adapter
from integrations.demo_pipeline.balanced_point_run import save, read


class SurfaceAdapterTests(unittest.TestCase):
    def execute_fake(self, root, bad_supervisor=None, bad_business=None):
        native=root/'native';native.mkdir();(native/'images').mkdir()
        for name in ('PLAN.json','RESULT.json'):save(native/name,{})
        diagnostic=root/'diagnostic.json';save(diagnostic,dict(source_bindings=[]))
        masks=root/'MASKS.json';save(masks,{})
        tools=root/'tools';tools.mkdir()
        for name in ('InterfaceCOLMAP.exe','DensifyPointCloud.exe','ReconstructMesh.exe'):
            (tools/name).write_bytes(b'fake-tool')
        calls=[]
        def invoke(out,label,argv,seconds,memory):
            calls.append((label,argv,seconds,memory))
            target=out/label;target.mkdir()
            if label=='identity_export':
                (target/'binary').mkdir()
                save(target/'RESULT.json',dict(status='IDENTITY_CONSTRAINED_EXPERIMENTAL',source_unchanged=True,
                    native_root=str(native),binary_path=str(target/'binary'),image_root=str(native/'images'),
                    source_bindings=[],final_bindings=[]))
            elif label=='bridge':
                save(target/'RESULT.json',dict(status='PASS',input_bindings_unchanged=True))
                for name in ('UNDISTORT_COMPARISON.json','MVS_COMPARISON.json'):save(target/name,dict(status='PASS'))
            elif label=='masks':
                save(target/'RESULT.json',dict(status='AVAILABLE',available_views=54,registered_views=54,input_drift=[],
                    identity_result=str(out/'identity_export/RESULT.json'),bridge=str(out/'bridge'),source_bindings=[]))
            else:
                save(target/'result.json',dict(status='SURFACE_GENERATED_NOT_ACCEPTED',exit_code=0))
            if label==bad_business:save(target/'RESULT.json',dict(status='FAIL'))
            return dict(status='completed_with_errors' if label==bad_supervisor else 'completed',
                root_exit_code=0,reclaim_confirmed=True,errors=[],log_errors=[])
        plan=dict(input_origin='unverified',photo_inputs=None,marker='preserve-verbatim')
        with patch.object(adapter,'TOOL_ROOT',tools),patch.object(adapter,'code_bindings',return_value=[]), \
             patch.object(adapter,'invoke',invoke), \
             patch('integrations.mvs_surface.runner.make_plan',return_value=plan) as make_plan:
            result=adapter.run(diagnostic,native,masks,root/'out')
        return result,calls,make_plan,plan

    def test_full_sequence_budgets_and_unmodified_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result,calls,make_plan,plan=self.execute_fake(root)
            self.assertEqual(result['status'],'SURFACE_GENERATED_NOT_ACCEPTED')
            self.assertFalse(result['geometry_accepted'])
            self.assertEqual([(c[0],c[2],c[3]) for c in calls],[
                ('identity_export',60,4<<30),('bridge',240,8<<30),('masks',120,4<<30),('surface',660,10<<30)])
            self.assertEqual(make_plan.call_args.kwargs,dict(total_seconds=600,stage_seconds=480,
                memory_bytes=8<<30,output_bytes=1<<30,ignore_label=0,input_origin='unverified',photo_source=None))
            self.assertEqual(read(root/'out/SURFACE_PLAN.json'),plan)
            self.assertEqual(calls[-1][1][-4],'--plan-sha256')
            self.assertEqual(read(root/'out/LEARNED_SOURCE_CHAIN.json')['legacy_experimental_inputs_claimed'],False)
            self.assertTrue(all(s['status']=='BUSINESS_GATE_PASSED' for s in result['stages']))

    def test_unclean_supervisor_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls,make_plan,_=self.execute_fake(Path(tmp),bad_supervisor='identity_export')
            self.assertEqual(result['status'],'FAILED')
            self.assertEqual(len(calls),1)
            make_plan.assert_not_called()

    def test_failed_business_stops_before_next_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls,make_plan,_=self.execute_fake(Path(tmp),bad_business='bridge')
            self.assertEqual(result['status'],'FAILED')
            self.assertEqual([c[0] for c in calls],['identity_export','bridge'])
            make_plan.assert_not_called()

if __name__=='__main__':unittest.main()
