"""Fail-closed orchestration tests; no native child is executed."""
import tempfile
from pathlib import Path
from unittest import TestCase, mock
from integrations.demo_pipeline import evidence_point_run as e


class RunTests(TestCase):
    def test_existing_or_nested_output_refused_before_source_read(self):
        with tempfile.TemporaryDirectory() as root:
            control=Path(root)/'control'; control.mkdir()
            with mock.patch.object(e,'prepare') as prepare:
                for output in (control,control/'child',Path(root)):
                    with self.assertRaises(ValueError): e.execute(control,output)
                prepare.assert_not_called()

    def test_binding_conflict_rejected_not_silently_replaced(self):
        a={'path':str(Path('same').resolve()),'bytes':1,'sha256':'a'}
        self.assertEqual(e.unique_bindings([a,dict(a)]),[a])
        with self.assertRaises(ValueError): e.unique_bindings([a,{**a,'sha256':'b'}])

    def test_failed_or_unreclaimed_stage_stops_without_retry(self):
        for entry in ({'root_exit_code':1,'reclaim_confirmed':True},
                      {'root_exit_code':0,'reclaim_confirmed':False}):
            with tempfile.TemporaryDirectory() as root:
                control=Path(root)/'control'; control.mkdir(); output=Path(root)/'new'
                e.save(control/'INPUTS.json',{'frames':[]})
                with mock.patch.object(e,'prepare',return_value=([],[],control/'TRACKS.json',{})), \
                     mock.patch.object(e,'binding',side_effect=lambda p:{'path':str(p.resolve()),'bytes':0,'sha256':'test'}), \
                     mock.patch.object(e,'check_bindings',return_value=[]), \
                     mock.patch.object(e,'launch',return_value=entry) as launch:
                    self.assertEqual(e.execute(control,output),1)
                    self.assertEqual(launch.call_count,1)
                    result=e.read(output/'RESULT.json')
                    self.assertEqual(result['status'],'FAILED')
                    self.assertFalse(result['geometry_accepted'])
