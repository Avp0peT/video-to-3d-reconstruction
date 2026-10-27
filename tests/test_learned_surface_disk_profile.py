import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from integrations.demo_pipeline import learned_surface_disk_profile as profile
from integrations.demo_pipeline.balanced_point_run import binding,read,save

BASE=profile.ROOT/'outputs/delegation/NEW_VIDEO/20261019_new01/learned_tracking'


class DiskProfileTests(unittest.TestCase):
    @unittest.skipUnless((BASE/'surface_identity02/OUTPUT_BINDINGS.json').exists(),'actual disk-stop evidence unavailable')
    def test_actual_disk_stop_and_disk_only_plan_diff_read_only(self):
        previous=BASE/'surface_identity01';failed=BASE/'surface_identity02'
        refs=profile.validate_disk_failure(failed,previous)
        self.assertFalse(profile.check_bindings(refs))
        from integrations.mvs_surface.runner import make_plan
        plan=make_plan(previous/'bridge/scene.mvs',previous/'bridge',previous/'masks/masks',previous/'masks/validity',
            profile.TOOL_ROOT,total_seconds=600,stage_seconds=480,memory_bytes=8<<30,output_bytes=4<<30,
            input_origin='unverified',ignore_label=0,photo_source=None)
        old=read(failed/'SURFACE_PLAN.json')
        profile.compare_disk_only(old,plan)
        altered=copy.deepcopy(plan);altered['options']['dense']['iters']=4
        with self.assertRaises(ValueError):profile.compare_disk_only(old,altered)

    def fake_prepared(self,root):
        previous=root/'previous';previous.mkdir();failed=root/'failed';failed.mkdir()
        source=root/'source';source.write_bytes(b'input')
        old=dict(arguments=dict(output_bytes=1<<30),budgets=dict(output_bytes=1<<30),
                 scene=binding(source),tools=[],implementation=[],frames=[])
        save(failed/'SURFACE_PLAN.json',old)
        plan=copy.deepcopy(old);plan['arguments']['output_bytes']=plan['budgets']['output_bytes']=4<<30
        out=root/'new'
        with patch.object(profile,'audit',return_value=([binding(source)],dict(images=[{},{}]))), \
             patch.object(profile,'validate_disk_failure',return_value=[]), \
             patch.object(profile,'code_bindings',return_value=[]), \
             patch('integrations.mvs_surface.runner.make_plan',return_value=plan):
            profile.prepare(previous,failed,out)
        return out

    def test_fresh_surface_once_monitors_entire_task_and_explicit_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=self.fake_prepared(Path(tmp))
            bootstrap=(out/'worker_bootstrap.py').read_text()
            self.assertIn("os.environ['PYTHONPATH']",bootstrap)
            self.assertIn("os.environ[key]='2'",bootstrap)
            self.assertIn('runpy.run_path',bootstrap)
            def supervised(argv,**kw):
                self.assertEqual(kw['cwd'],out)
                self.assertEqual(kw['output'],out/'outer')
                self.assertEqual((kw['wall_seconds'],kw['memory_bytes'],kw['output_bytes']),(660,10<<30,6<<30))
                self.assertFalse((out/'surface').exists())
                (out/'surface').mkdir()
                save(out/'surface/result.json',dict(status='SURFACE_GENERATED_NOT_ACCEPTED',exit_code=0))
                return dict(status='completed',root_exit_code=0,reclaim_confirmed=True,errors=[],log_errors=[])
            with patch.object(profile,'run_supervised',supervised):
                result=profile.run(out)
            self.assertEqual(result['status'],'SURFACE_GENERATED_NOT_ACCEPTED')
            self.assertFalse(result['geometry_accepted'])
            with self.assertRaises(ValueError):profile.run(out)

    def test_prepared_hash_drift_rejects_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=self.fake_prepared(Path(tmp))
            (out/'worker_bootstrap.py').write_text('changed')
            with patch.object(profile,'run_supervised') as native:
                with self.assertRaises(ValueError):profile.run(out)
                native.assert_not_called()

if __name__=='__main__':unittest.main()
