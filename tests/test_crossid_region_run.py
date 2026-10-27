import tempfile
from pathlib import Path
from unittest import TestCase,mock
from integrations.demo_pipeline import crossid_region_run as run


class CrossIDRunTests(TestCase):
    def test_graph_counts_isolated_and_fifteen_inlier_threshold(self):
        plan=dict(frames=[dict(input_index=i,database_image_id=i+1) for i in range(4)],
                  candidate_pairs=[dict(image_a=1,image_b=2),dict(image_a=2,image_b=3)])
        native=dict(pairs=[dict(image_a=1,image_b=2,inliers=15),dict(image_a=2,image_b=3,inliers=14)],
                    models=[],unique_registered_count=0)
        census=run.graph_census(native,plan)
        self.assertEqual(census['raw']['component_count'],2)
        self.assertEqual(census['verified']['components'],[[0,1],[2],[3]])
        self.assertEqual(census['verified']['isolated'],2)
        self.assertEqual(census['largest_native_model'],0)

    def test_output_parent_or_existing_stops_before_prepare(self):
        with tempfile.TemporaryDirectory() as root:
            control=Path(root)/'control'; control.mkdir()
            with mock.patch.object(run,'prepare') as prepare:
                for output in (control,control/'nested',Path(root)):
                    with self.assertRaises(ValueError): run.execute(control,output)
                prepare.assert_not_called()

    def test_failure_or_unknown_reclaim_does_not_start_second_stage(self):
        for entry in (dict(root_exit_code=1,reclaim_confirmed=True),dict(root_exit_code=0,reclaim_confirmed=False)):
            with tempfile.TemporaryDirectory() as root:
                control=Path(root)/'control'; control.mkdir(); output=Path(root)/'new'
                with mock.patch.object(run,'prepare',return_value=([],[],control/'TRACKS.json',control/'native')), \
                     mock.patch.object(run,'binding',side_effect=lambda p:dict(path=str(p.resolve()),bytes=0,sha256='fake')), \
                     mock.patch.object(run,'check_bindings',return_value=[]), \
                     mock.patch.object(run,'launch',return_value=entry) as launch:
                    self.assertEqual(run.execute(control,output),1)
                    self.assertEqual(launch.call_count,1)
                    self.assertEqual(run.read(output/'RESULT.json')['status'],'FAILED')
