import tempfile
import os
import unittest
from pathlib import Path
from unittest import TestCase,mock
from integrations.demo_pipeline import connectivity_point_run as run
from integrations.demo_pipeline.connectivity_graph import components,census


class ConnectivityRunTests(TestCase):
    @unittest.skipUnless(os.environ.get('CONNECTIVITY_NATIVE_SUPERVISED')=='1','native legacy readback requires supervisor')
    def test_legacy_geometry_control_audit_from_bound_fixture(self):
        from integrations.demo_pipeline.connectivity_native import audit_existing
        source=Path(os.environ['CONNECTIVITY_LEGACY_RESULT']).resolve(strict=True)
        target=Path(os.environ['CONNECTIVITY_NATIVE_OUTPUT']).parent/'legacy_audit.json'
        result=audit_existing(source,target)
        self.assertTrue(result['source_unchanged'])
        self.assertFalse(result['quality_accepted'])
        self.assertTrue(result['components'])
        self.assertTrue(all('radial_observation_domain' in c for c in result['components']))

    def test_isolated_nodes_retained_and_no_fake_bridge(self):
        self.assertEqual(components([1,2,3,4],[(1,2),(2,3)]),[[1,2,3],[4]])
        self.assertEqual(components([1,2,3,4],[(1,2),(2,3),(1,3)]),[[1,2,3],[4]])
        with self.assertRaises(ValueError): components([1,2],[(1,3)])

    def test_all_shared_pair_addition_is_not_cross_component_connection(self):
        frames=[dict(input_index=i,observations=[dict(track_id=t,measured=True) for t in range(15)] if i<5 else []) for i in range(6)]
        plan=dict(frames=[dict(database_image_id=i+1,input_index=i) for i in range(6)])
        native=dict(pairs=[dict(image_a=i+1,image_b=i+2,inliers=15) for i in range(4)],models=[],unique_registered_input_indices=[])
        result=census(dict(frames=frames),native,plan)
        self.assertEqual(result['outside_candidate_pairs'],1)
        self.assertEqual(result['raw_allpair_components'],[[0,1,2,3,4],[5]])
        self.assertEqual(result['arms']['geometry_evidence']['outside_cross_verified_component_pairs'],0)
        frames[0]['observations'][0]['measured']=False
        with self.assertRaises(ValueError): census(dict(frames=frames),native,plan)

    def test_existing_and_parent_output_refused_before_prepare(self):
        with tempfile.TemporaryDirectory() as root:
            control=Path(root)/'control'; control.mkdir()
            with mock.patch.object(run,'prepare') as prepare:
                for output in (control,control/'inside',Path(root)):
                    with self.assertRaises(ValueError): run.execute(control,output)
                prepare.assert_not_called()

    def test_failed_or_unknown_reclaim_stops_without_retry(self):
        for entry in ({'root_exit_code':1,'reclaim_confirmed':True},{'root_exit_code':0,'reclaim_confirmed':False}):
            with tempfile.TemporaryDirectory() as root:
                control=Path(root)/'control'; control.mkdir(); output=Path(root)/'new'
                with mock.patch.object(run,'prepare',return_value=([],[],control/'TRACKS.json',[])), \
                     mock.patch.object(run,'binding',side_effect=lambda p:{'path':str(p.resolve()),'bytes':0,'sha256':'test'}), \
                     mock.patch.object(run,'check_bindings',return_value=[]), \
                     mock.patch('integrations.demo_pipeline.connectivity_graph.audit') as audit, \
                     mock.patch.object(run,'launch',return_value=entry) as launch:
                    self.assertEqual(run.execute(control,output),1)
                    self.assertEqual(launch.call_count,1)
                    self.assertEqual(run.read(output/'RESULT.json')['status'],'FAILED')
