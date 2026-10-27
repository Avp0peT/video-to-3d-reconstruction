"""Pure source validation and frozen geometry math. No native mapping calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import os
import numpy as np
from PIL import Image
from integrations.demo_pipeline import evidence_native as e


def source_fixture(root):
    frames=[]
    for index in range(3):
        image=root/f'image{index}.png'; mask=root/f'mask{index}.png'
        Image.new('RGB',(40,40),(70,80,90)).save(image); Image.new('L',(40,40),255).save(mask)
        frames.append(dict(input_index=index,decode_index=index*7,frame_id=f'TEST_DATA_{index}',
            image_path=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            mask_path=str(mask),mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),
            width=40,height=40,original_mask_status='OK',
            observations=[dict(track_id=9,xy=[10.25,12.75],measured=True)] if index<2 else []))
    doc=dict(schema=e.SCHEMA,arm='balanced',frames=frames,predicted_observations=0)
    path=root/'TRACKS.json'; path.write_text(json.dumps(doc),encoding='utf-8')
    return path,doc


def projected_observations(world,centers):
    observations=[]
    for i,center in enumerate(centers):
        local=np.asarray(world)-center; projected=100*local[:2]/local[2]+20
        observations.append(dict(native_image_id=i+1,feature_row=7,candidate_track_id=9,
            native_xy=projected.tolist(),projected_xy=projected.tolist(),camera_xyz=local.tolist(),camera_center=list(center)))
    return observations


def native_fixture(root):
    """TEST_DATA: 160 six-view points plus 50 actual two-view-only points."""
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(739)
    world=np.column_stack((rng.uniform(-.4,.4,210),rng.uniform(-.35,.35,210),rng.uniform(6,6.8,210)))
    frames=[]; w,h=1024,768; focal=1.2*w
    for i in range(6):
        image=root/f'image{i}.png'; mask=root/f'mask{i}.png'
        Image.new('RGB',(w,h),(70,80,90)).save(image); Image.new('L',(w,h),255).save(mask)
        center=np.array([(i-2.5)*.65,.025*np.sin(i),0.])
        local=world-center; xy=focal*local[:,:2]/local[:,2,None]+[w/2,h/2]-.5
        visible=210 if i<2 else 160
        observations=[dict(track_id=t,xy=point.tolist(),measured=True) for t,point in enumerate(xy[:visible])]
        frames.append(dict(input_index=i,decode_index=i*7,frame_id=f'TEST_DATA_NATIVE_{i}',
            image_path=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            mask_path=str(mask),mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),width=w,height=h,
            original_mask_status='OK',observations=observations))
    doc=dict(schema=e.SCHEMA,arm='balanced',frames=frames,predicted_observations=0)
    path=root/'TRACKS.json'; path.write_text(json.dumps(doc),encoding='utf-8')
    return path


class EvidenceTests(unittest.TestCase):
    def test_two_actual_samples_admitted_no_fake_third_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path,doc=source_fixture(root)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            for policy in ('two_sample_candidates','geometry_evidence'):
                plan=e.prepare(path,policy)
                self.assertEqual(plan['eligible_track_count'],1)
                self.assertEqual([len(f['keypoints']) for f in plan['frames']],[1,1,0])
                np.testing.assert_array_equal(plan['frames'][0]['keypoints'],[[10.75,13.25]])
                self.assertEqual(plan['track_view_counts'],{9:2})
                self.assertEqual(plan['policy_id'],policy)
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})
    def test_unmeasured_predicted_and_source_faults_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path,original=source_fixture(root)
            for field,value in [('measured',False),('measured',None),('predicted',True),('observed_type','prediction')]:
                doc=json.loads(json.dumps(original)); doc['frames'][0]['observations'][0][field]=value
                path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError): e.prepare(path)
            doc=json.loads(json.dumps(original)); doc['predicted_observations']=1
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError): e.prepare(path)
            doc=json.loads(json.dumps(original)); doc['frames'][0]['observations'][0]['xy']=[float('nan'),10]
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError): e.prepare(path)
            path.write_text(json.dumps(original)); (root/'image0.png').write_bytes(b'drift')
            with self.assertRaises(ValueError): e.prepare(path)
    def test_policy_switch_is_explicit(self):
        self.assertTrue(e._policy('two_sample_candidates'))
        self.assertFalse(e._policy('geometry_evidence'))
        with self.assertRaises(ValueError): e._policy('relaxed')
    def test_legacy_source_reconstruction_has_three_view_gate_without_old_prepare(self):
        with tempfile.TemporaryDirectory() as folder:
            path,doc=source_fixture(Path(folder))
            for frame in doc['frames']:
                for observation in frame['observations']: del observation['measured']
            path.write_text(json.dumps(doc))
            rebuilt=e._prepare_sources(path,'two_sample_candidates',3,True)
            self.assertEqual(rebuilt['eligible_track_count'],0)
            self.assertTrue(all(len(f['keypoints'])==0 for f in rebuilt['frames']))
            with self.assertRaises(ValueError): e.prepare(path)
    def test_native_coverage_records_observations_without_geometry_side_effect(self):
        result=e._grid(np.float64([[10.75,13.25]]),np.ones((40,40),bool))
        self.assertEqual(set(result),{'4','8','12'})
        self.assertTrue(all(row['occupied']==1 for row in result.values()))
    def test_known_valid_two_view_and_real_three_view_classification(self):
        world=np.array([0.,0.,5.]); centers=[[-.5,0,0],[.5,0,0]]
        result=e.audit_point_geometry(world,projected_observations(world,centers))
        self.assertTrue(result['pass_frozen_geometry'])
        self.assertEqual(result['classification'],'two_view_provisional')
        self.assertEqual(len(result['observations']),2)
        self.assertEqual(result['measured_views'],2)
        self.assertFalse(result['quality_accepted'])
        self.assertGreater(result['max_effective_ray_angle_deg'],10)
        multi=e.audit_point_geometry(world,projected_observations(world,centers+[[0,.5,0]]))
        self.assertEqual(multi['classification'],'multiview_consistent_not_accepted')
    def test_small_baseline_pure_rotation_and_antiparallel_rays_reject(self):
        world=np.array([0.,0.,5.])
        for centers in [[[0,0,0],[.001,0,0]],[[0,0,0],[0,0,0]]]:
            result=e.audit_point_geometry(world,projected_observations(world,centers))
            self.assertFalse(result['pass_frozen_geometry'])
            self.assertIn('insufficient_triangulation_angle',result['rejection_reasons'])
        observations=projected_observations(world,[[0,0,0],[0,0,10]])
        # Opposite facing camera rotates its negative world-relative Z to positive.
        observations[1]['camera_xyz']=[0,0,5]
        result=e.audit_point_geometry(world,observations)
        self.assertEqual(result['max_effective_ray_angle_deg'],0.)
        self.assertTrue(all(o['positive_depth'] for o in result['observations']))
        self.assertFalse(result['pass_frozen_geometry'])
    def test_negative_depth_residual_and_nonfinite_fail_separately(self):
        world=np.array([0.,0.,5.]); centers=[[-.5,0,0],[.5,0,0]]
        observations=projected_observations(world,centers); observations[1]['camera_xyz'][2]=-5
        result=e.audit_point_geometry(world,observations)
        self.assertIn('nonpositive_depth',result['rejection_reasons'])
        observations=projected_observations(world,centers); observations[1]['native_xy'][0]+=5
        result=e.audit_point_geometry(world,observations)
        self.assertIn('reprojection_error',result['rejection_reasons'])
        self.assertEqual(result['max_reprojection_error_px'],5)
        result=e.audit_point_geometry([float('nan'),0,5],projected_observations(world,centers))
        self.assertFalse(result['finite_xyz']); self.assertFalse(result['pass_frozen_geometry'])
    def test_native_feature_identity_duplicate_rejects(self):
        world=np.array([0.,0.,5.]); observations=projected_observations(world,[[-.5,0,0],[.5,0,0]])
        observations[1]['native_image_id']=observations[0]['native_image_id']
        with self.assertRaises(ValueError): e.audit_point_geometry(world,observations)

    @unittest.skipUnless(os.environ.get('EVIDENCE_NATIVE_SUPERVISED')=='1','native execution requires parent supervisor')
    def test_supervised_native_two_policy_fixture(self):
        # The supervisor must bind this output root and process/thread limits.
        root=Path(os.environ['EVIDENCE_NATIVE_OUTPUT']).resolve()
        root.mkdir(parents=True,exist_ok=False)
        path=native_fixture(root/'input')
        plans=[e.prepare(path,p) for p in ('two_sample_candidates','geometry_evidence')]
        self.assertEqual(plans[0]['track_view_counts'],plans[1]['track_view_counts'])
        for a,b in zip(plans[0]['frames'],plans[1]['frames']):
            self.assertEqual(a['track_ids'],b['track_ids']); np.testing.assert_array_equal(a['keypoints'],b['keypoints'])
        native_results={}
        for policy in ('two_sample_candidates','geometry_evidence'):
            result=e.run(path,root/policy,policy)
            self.assertEqual(result['verification_calls'],1); self.assertEqual(result['mapping_calls'],1)
            self.assertEqual(result['status'],'MODELS_GENERATED_NOT_ACCEPTED')
            self.assertTrue(result['source_unchanged'])
            self.assertEqual(result['actual_options']['mapping']['triangulation']['ignore_two_view_tracks'],policy=='two_sample_candidates')
            audit=json.loads((root/policy/'GEOMETRY_EVIDENCE.json').read_text())
            self.assertFalse(audit['quality_accepted'])
            provisional=[p for c in audit['components'] for p in c['points'] if p['classification']=='two_view_provisional']
            two_only=[p for p in provisional if all(o['candidate_track_id']>=160 for o in p['observations'])]
            if policy=='two_sample_candidates': self.assertEqual(len(two_only),0)
            else:
                self.assertGreater(len(two_only),0)
                for point in provisional:
                    self.assertEqual(point['measured_views'],2); self.assertEqual(len(point['observations']),2)
            native_results[policy]=result
            # New fixture sources only: preserve exact bytes around each fault.
            result_path=root/policy/'RESULT.json'; plan_path=root/policy/'PLAN.json'
            plan_bytes=plan_path.read_bytes(); source_image=root/'input'/'image0.png'
            image_bytes=source_image.read_bytes()
            try:
                source_image.write_bytes(b'TEST_DATA_DRIFT')
                with self.assertRaisesRegex(ValueError,'source binding drift'):
                    e.audit_existing(result_path,root/policy/'REJECTED_SOURCE_DRIFT.json')
            finally: source_image.write_bytes(image_bytes)
            try:
                changed=json.loads(plan_bytes); changed['frames'][0]['track_ids'][0]+=10000
                plan_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'track rows differ'):
                    e.audit_existing(result_path,root/policy/'REJECTED_RELABEL.json')
            finally: plan_path.write_bytes(plan_bytes)
            tracks_bytes=path.read_bytes()
            try:
                changed=json.loads(tracks_bytes)
                changed['frames'][0]['observations'][0]['observed_type']='prediction'
                path.write_text(json.dumps(changed))
                changed_plan=json.loads(plan_bytes)
                changed_plan['source_bindings'][0]=e._hash(path)
                plan_path.write_text(json.dumps(changed_plan))
                with self.assertRaisesRegex(ValueError,'observation type contradicts'):
                    e.audit_existing(result_path,root/policy/'REJECTED_PREDICTION_TYPE.json')
            finally:
                path.write_bytes(tracks_bytes); plan_path.write_bytes(plan_bytes)
            readback=e.audit_existing(result_path,root/policy/'REAUDIT_SOURCE_RESTORED.json')
            self.assertTrue(readback['source_unchanged'])

if __name__=='__main__': unittest.main()
