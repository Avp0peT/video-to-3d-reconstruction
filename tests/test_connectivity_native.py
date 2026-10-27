"""Source graph and policy tests. Native option/mapper tests are supervised only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from integrations.demo_pipeline import connectivity_native as c
from tests.test_evidence_native import native_fixture


def graph_fixture(root,thin=False):
    frames=[]
    for i in range(6):
        image=root/f'image{i}.png'; mask=root/f'mask{i}.png'
        Image.new('RGB',(64,64),(70,80,90)).save(image); Image.new('L',(64,64),255).save(mask)
        ids=list(range(10 if thin else 20))
        if not thin and i in (0,5): ids+=list(range(20,35))
        frames.append(dict(input_index=i*2,decode_index=i*7,frame_id=f'TEST_DATA_GRAPH_{i}',
            image_path=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            mask_path=str(mask),mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),
            width=64,height=64,original_mask_status='OK',
            observations=[dict(track_id=t,xy=[10.+t*.7,20.+i*.2],measured=True) for t in ids]))
    path=root/'TRACKS.json'; path.write_text(json.dumps(dict(schema=c.SCHEMA,arm='balanced',frames=frames,predicted_observations=0)))
    return path


class ConnectivityTests(unittest.TestCase):
    def test_all_pairs_remote_edges_and_rows_are_actual_shared_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path=graph_fixture(Path(folder)); before=c._hash(path)
            for policy in ('all_shared_tracks','all_shared_tracks_zero_radial'):
                plan=c.prepare(path,policy)
                self.assertEqual(len(plan['pairs']),15)
                remote=next(p for p in plan['pairs'] if (p['sequence_a'],p['sequence_b'])==(0,5))
                self.assertEqual(remote['sequence_distance'],5)
                self.assertEqual(remote['input_index_distance'],10)
                self.assertEqual(remote['decode_index_distance'],35)
                self.assertEqual(remote['track_ids'],list(range(35)))
                np.testing.assert_array_equal(remote['matches'],np.column_stack((np.arange(35),np.arange(35))))
                self.assertEqual(plan['shared_track_edges_outside_window'],3)
                self.assertEqual(plan['frames'][0]['track_ids'],list(range(35)))
                self.assertNotIn(20,plan['frames'][1]['track_ids'])
                np.testing.assert_array_equal(plan['frames'][0]['keypoints'][0],[10.5,20.5])
            self.assertEqual(c._hash(path),before)
    def test_insufficient_matches_excluded_for_every_pair_not_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan=c.prepare(graph_fixture(Path(folder),thin=True))
            self.assertEqual(len(plan['pairs']),0)
            self.assertEqual(len(plan['excluded_pairs']),15)
            self.assertTrue(all(p['common_tracks']==10 for p in plan['excluded_pairs']))
            self.assertEqual(max(p['sequence_distance'] for p in plan['excluded_pairs']),5)
    def test_prediction_type_rejected_and_policy_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path=graph_fixture(Path(folder)); doc=json.loads(path.read_text())
            doc['frames'][0]['observations'][0]['observed_type']='prediction'
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError): c.prepare(path)
        self.assertFalse(c._policy('all_shared_tracks'))
        self.assertTrue(c._policy('all_shared_tracks_zero_radial'))
        with self.assertRaises(ValueError): c._policy('geometry_evidence')
    def test_refinement_validation_covers_every_effective_path(self):
        mapping=dict(triangulation=dict(ignore_two_view_tracks=False),ba_refine_extra_params=False,
            ba_refine_focal_length=True,ba_refine_principal_point=False,
            mapper=dict(abs_pose_refine_extra_params=False,abs_pose_refine_focal_length=True))
        effective=copy.deepcopy(mapping['mapper'])
        ba=dict(refine_extra_params=False,refine_focal_length=True,refine_principal_point=False)
        c._validate_refinement_options('all_shared_tracks_zero_radial',mapping,effective,ba,ba)
        for stage in ('pipeline','mapper','effective_mapper','local_ba','global_ba'):
            m=copy.deepcopy(mapping); e=copy.deepcopy(effective); local=copy.deepcopy(ba); glob=copy.deepcopy(ba)
            if stage=='pipeline': m['ba_refine_extra_params']=True
            elif stage=='mapper': m['mapper']['abs_pose_refine_extra_params']=True
            elif stage=='effective_mapper': e['abs_pose_refine_extra_params']=True
            elif stage=='local_ba': local['refine_extra_params']=True
            else: glob['refine_extra_params']=True
            with self.assertRaises(ValueError): c._validate_refinement_options('all_shared_tracks_zero_radial',m,e,local,glob)
    def test_geometry_gates_remain_frozen(self):
        world=np.float64([0,0,5]); obs=[]
        for i,x in enumerate([-.5,.5]):
            local=world-[x,0,0]; xy=100*local[:2]/local[2]+20
            obs.append(dict(native_image_id=i+1,feature_row=0,candidate_track_id=4,native_xy=xy,
                projected_xy=xy,camera_xyz=local,camera_center=[x,0,0]))
        point=c.audit_point_geometry(world,obs)
        self.assertEqual(point['classification'],'two_view_provisional')
        obs[1]['native_xy']=obs[1]['native_xy']+[5,0]
        self.assertFalse(c.audit_point_geometry(world,obs)['pass_frozen_geometry'])
    def test_actual_domain_radial_diagnostic_is_not_absolute_k_or_new_gate(self):
        folding=c._radial_observation([1,0,1],-.5)
        self.assertEqual(folding['normalized_radius_squared'],1.)
        self.assertEqual(folding['radial_scale'],.5)
        self.assertEqual(folding['radial_derivative'],-.5)
        large_but_local=c._radial_observation([.01,0,1],1000.)
        self.assertGreater(large_but_local['radial_scale'],0)
        self.assertGreater(large_but_local['radial_derivative'],0)
        summary=c._radial_summary([folding,large_but_local,c._radial_observation([1,0,0],1)])
        self.assertEqual(summary['finite_observations'],2)
        self.assertEqual(summary['invalid_observations'],1)
        self.assertEqual(summary['nonpositive_scale_count'],0)
        self.assertEqual(summary['nonpositive_derivative_count'],1)
        self.assertEqual(summary['radial_derivative']['min'],-.5)
        # The diagnostic does not supply an additional geometry rejection gate.
        world=np.float64([0,0,5]); obs=[]
        for i,x in enumerate([-.5,.5]):
            local=world-[x,0,0]; xy=100*local[:2]/local[2]+20
            obs.append(dict(native_image_id=i+1,feature_row=0,candidate_track_id=4,native_xy=xy,
                projected_xy=xy,camera_xyz=local,camera_center=[x,0,0],**folding))
        self.assertTrue(c.audit_point_geometry(world,obs)['pass_frozen_geometry'])

    @unittest.skipUnless(os.environ.get('CONNECTIVITY_NATIVE_SUPERVISED')=='1','native option API requires supervisor')
    def test_supervised_actual_option_factory(self):
        import pycolmap as pc
        snapshots={}
        for policy in ('all_shared_tracks','all_shared_tracks_zero_radial'):
            verify,mapper=c.make_options(pc,policy,6)
            self.assertEqual(verify.ransac.num_threads,1)
            self.assertEqual(mapper.num_threads,2)
            self.assertFalse(mapper.triangulation.ignore_two_view_tracks)
            self.assertTrue(mapper.ba_refine_focal_length)
            self.assertFalse(mapper.ba_refine_principal_point)
            extra=policy=='all_shared_tracks'
            self.assertEqual(mapper.get_mapper().abs_pose_refine_extra_params,extra)
            for ba in (mapper.get_local_bundle_adjustment(),mapper.get_global_bundle_adjustment()):
                self.assertEqual(ba.refine_extra_params,extra); self.assertTrue(ba.refine_focal_length)
                self.assertFalse(ba.refine_principal_point)
            snapshots[policy]=c._plain(mapper.todict())
        base=snapshots['all_shared_tracks']; zero=snapshots['all_shared_tracks_zero_radial']
        expected=copy.deepcopy(base); expected['ba_refine_extra_params']=False
        expected['mapper']['abs_pose_refine_extra_params']=False
        self.assertEqual(expected,zero)

    @unittest.skipUnless(os.environ.get('CONNECTIVITY_NATIVE_SUPERVISED')=='1','native mapping requires supervisor')
    def test_supervised_native_fixture_two_arms(self):
        root=Path(os.environ['CONNECTIVITY_NATIVE_OUTPUT']).resolve(); root.mkdir(parents=True,exist_ok=False)
        path=native_fixture(root/'input'); plans=[c.prepare(path,p) for p in ('all_shared_tracks','all_shared_tracks_zero_radial')]
        for a,b in zip(plans[0]['frames'],plans[1]['frames']):
            self.assertEqual(a['track_ids'],b['track_ids']); np.testing.assert_array_equal(a['keypoints'],b['keypoints'])
        for policy in ('all_shared_tracks','all_shared_tracks_zero_radial'):
            result=c.run(path,root/policy,policy)
            self.assertEqual(result['status'],'MODELS_GENERATED_NOT_ACCEPTED')
            self.assertEqual(result['verification_calls'],1); self.assertEqual(result['mapping_calls'],1)
            self.assertIsNone(result['actual_options']['max_pair_sequence_distance'])
            audit=json.loads((root/policy/'GEOMETRY_EVIDENCE.json').read_text())
            points=[p for comp in audit['components'] for p in comp['points']]
            self.assertTrue(any(p['classification']=='two_view_provisional' and all(o['candidate_track_id']>=160 for o in p['observations']) for p in points))
            if policy=='all_shared_tracks_zero_radial':
                self.assertTrue(all(camera['model']=='SIMPLE_RADIAL' and camera['params'][-1]==0 for model in result['models'] for camera in model['cameras']))
            self.assertFalse(audit['quality_accepted'])

if __name__=='__main__': unittest.main()
