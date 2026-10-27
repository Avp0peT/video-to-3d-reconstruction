"""Explicit cross-ID row supplement tests; native fixture requires supervision."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from integrations.demo_pipeline import crossid_native as c
from integrations.demo_pipeline import connectivity_native as baseline
from tests.test_connectivity_native import graph_fixture
from tests.test_evidence_native import native_fixture


def fixture(root):
    path=graph_fixture(root); doc=json.loads(path.read_text())
    # Two eligible local track fragments, with separate IDs and actual rows.
    for index,frame in enumerate(doc['frames']):
        if index<3: frame['observations'].append(dict(track_id=80,xy=[40,40],measured=True))
        else: frame['observations'].append(dict(track_id=90,xy=[41,40],measured=True))
    # A single observation never enters the eligible row set.
    doc['frames'][2]['observations'].append(dict(track_id=999,xy=[42,40],measured=True))
    path.write_text(json.dumps(doc)); supplement=root/'SUPPLEMENT.json'
    source=root/'evidence.txt'; source.write_text('TEST_DATA_REGION_EVIDENCE')
    spec=dict(schema='crossid-region-matches-v1',tracks_binding=c._hash(path),source_bindings=[c._hash(source)],
        pairs=[dict(sequence_a=2,sequence_b=3,matches=[dict(track_id_a=80,track_id_b=90,score=.95,method='TEST_DATA')])])
    supplement.write_text(json.dumps(spec)); return path,supplement,spec


class CrossIDTests(unittest.TestCase):
    def test_eligible_rows_and_same_id_matches_unchanged_cross_rows_appended(self):
        with tempfile.TemporaryDirectory() as folder:
            path,supplement,_=fixture(Path(folder)); before=c._hash(path)
            old=baseline.prepare(path,'all_shared_tracks_zero_radial'); new=c.prepare(path,supplement)
            self.assertEqual(len(old['frames']),len(new['frames']))
            for a,b in zip(old['frames'],new['frames']):
                self.assertEqual(a['track_ids'],b['track_ids']); np.testing.assert_array_equal(a['keypoints'],b['keypoints'])
            self.assertEqual(len(old['pairs']),len(new['pairs']))
            for a,b in zip(old['pairs'],new['pairs']):
                np.testing.assert_array_equal(a['matches'],b['matches'][:len(a['matches'])])
            pair=next(p for p in new['pairs'] if (p['sequence_a'],p['sequence_b'])==(2,3))
            self.assertEqual(pair['cross_id_matches'],1)
            self.assertEqual(pair['match_provenance'][-1]['track_id_a'],80)
            self.assertEqual(pair['match_provenance'][-1]['track_id_b'],90)
            self.assertEqual(new['supplement_statistics']['used_pairs'],1)
            self.assertEqual(new['supplement_statistics']['used_matches'],1)
            self.assertEqual(c._hash(path),before)
    def test_original_same_id_endpoints_win_with_explicit_conflict_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            path,supplement,spec=fixture(Path(folder))
            spec['pairs'][0]['matches']=[dict(track_id_a=0,track_id_b=90)]
            supplement.write_text(json.dumps(spec)); plan=c.prepare(path,supplement)
            self.assertEqual(plan['supplement_statistics']['used_matches'],0)
            self.assertEqual(plan['supplement_statistics']['excluded_same_id_endpoint_conflicts'],1)
            self.assertEqual(plan['supplement_conflicts'][0]['reason'],'same_id_endpoint_priority')
    def test_malformed_ids_order_duplicates_and_binding_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path,supplement,original=fixture(Path(folder))
            variants=[]
            for ta,tb in [(80,80),(999,90),(12345,90),(True,90)]:
                spec=copy.deepcopy(original);spec['pairs'][0]['matches']=[dict(track_id_a=ta,track_id_b=tb)];variants.append(spec)
            spec=copy.deepcopy(original);spec['pairs'][0]['sequence_a']=3;spec['pairs'][0]['sequence_b']=2;variants.append(spec)
            spec=copy.deepcopy(original);spec['pairs']*=2;variants.append(spec)
            spec=copy.deepcopy(original);spec['pairs'][0]['matches']*=2;variants.append(spec)
            spec=copy.deepcopy(original);spec['pairs'][0]['matches']+=[dict(track_id_a=1,track_id_b=90)];variants.append(spec)
            spec=copy.deepcopy(original);spec['tracks_binding']['sha256']='0'*64;variants.append(spec)
            for spec in variants:
                supplement.write_text(json.dumps(spec))
                with self.assertRaises(ValueError): c.prepare(path,supplement)
            supplement.write_text(json.dumps(original));(Path(folder)/'evidence.txt').write_text('DRIFT')
            with self.assertRaises(ValueError): c.prepare(path,supplement)
    def test_below_threshold_supplement_is_retained_but_not_native_used(self):
        with tempfile.TemporaryDirectory() as folder:
            path,supplement,spec=fixture(Path(folder)); doc=json.loads(path.read_text())
            # Ten same-ID plus one explicit cross-ID still do not reach 15.
            for frame in doc['frames']: frame['observations']=[o for o in frame['observations'] if o['track_id']>=10]
            path.write_text(json.dumps(doc)); spec['tracks_binding']=c._hash(path); supplement.write_text(json.dumps(spec))
            plan=c.prepare(path,supplement)
            self.assertEqual(plan['supplement_statistics']['admitted_matches'],1)
            self.assertEqual(plan['supplement_statistics']['used_matches'],0)
            pair=next(p for p in plan['excluded_pairs'] if (p['sequence_a'],p['sequence_b'])==(2,3))
            self.assertEqual(pair['total_matches'],11)
            self.assertEqual(pair['cross_id_matches'],1)
    def test_cross_id_supplement_connects_pairs_with_zero_original_shared_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path,supplement,spec=fixture(Path(folder)); doc=json.loads(path.read_text())
            for index,frame in enumerate(doc['frames']):
                # Distinct eligible identities across halves; physical row sets
                # are preserved and no same-ID edge exists between halves.
                frame['observations']=[dict(track_id=t+(0 if index<3 else 100),xy=[10.+t*.7,20.],measured=True) for t in range(20)]
            path.write_text(json.dumps(doc));spec['tracks_binding']=c._hash(path)
            spec['pairs']=[dict(sequence_a=2,sequence_b=3,matches=[dict(track_id_a=t,track_id_b=t+100) for t in range(15)])]
            supplement.write_text(json.dumps(spec))
            old=baseline.prepare(path,'all_shared_tracks_zero_radial'); plan=c.prepare(path,supplement)
            self.assertEqual(len(old['pairs']),6)
            self.assertEqual(len(plan['pairs']),7)
            self.assertEqual(plan['supplement_statistics']['newly_submitted_pairs'],1)
            bridge=next(p for p in plan['pairs'] if (p['sequence_a'],p['sequence_b'])==(2,3))
            self.assertEqual(bridge['same_id_matches'],0);self.assertEqual(bridge['cross_id_matches'],15)
            np.testing.assert_array_equal(bridge['matches'],np.column_stack((np.arange(15),np.arange(15))))
            for a,b in zip(old['frames'],plan['frames']):
                self.assertEqual(a['track_ids'],b['track_ids']);np.testing.assert_array_equal(a['keypoints'],b['keypoints'])

    @unittest.skipUnless(os.environ.get('CROSSID_NATIVE_SUPERVISED')=='1','native fixture requires parent supervisor')
    def test_supervised_native_cross_id_fragments(self):
        root=Path(os.environ['CROSSID_NATIVE_OUTPUT']).resolve();root.mkdir(parents=True,exist_ok=False)
        path=native_fixture(root/'input'); doc=json.loads(path.read_text())
        # Reuse the physical two-only fixtures as six-view points with separate
        # measured IDs across the two temporal halves; no third observation is
        # invented. Known source projections are generated for each actual frame.
        rng=np.random.default_rng(739)
        world=np.column_stack((rng.uniform(-.4,.4,210),rng.uniform(-.35,.35,210),rng.uniform(6,6.8,210)))
        for index,frame in enumerate(doc['frames']):
            frame['observations']=[o for o in frame['observations'] if o['track_id']<160]
            center=np.array([(index-2.5)*.65,.025*np.sin(index),0.]); local=world-center
            xy=(1.2*1024)*local[:,:2]/local[:,2,None]+[512,384]-.5
            frame['observations'] += [dict(track_id=t if index<3 else t+100,xy=xy[t].tolist(),measured=True) for t in range(160,210)]
        path.write_text(json.dumps(doc));supplement=root/'SUPPLEMENT.json'
        supplement.write_text(json.dumps(dict(schema='crossid-region-matches-v1',tracks_binding=c._hash(path),source_bindings=[],
            pairs=[dict(sequence_a=2,sequence_b=3,matches=[dict(track_id_a=t,track_id_b=t+100,method='TEST_DATA_KNOWN_PROJECTION') for t in range(160,210)])])))
        result=c.run(path,supplement,root/'native')
        self.assertEqual(result['status'],'MODELS_GENERATED_NOT_ACCEPTED')
        self.assertEqual(result['verification_calls'],1);self.assertEqual(result['mapping_calls'],1)
        self.assertEqual(result['supplement_statistics']['used_matches'],50)
        self.assertGreater(result['verified_cross_id_matches'],0)
        self.assertTrue(any(len({o['candidate_track_id'] for o in p['track']})>1 for model in result['models'] for p in model['point_records']))
        audit=json.loads((root/'native'/'GEOMETRY_EVIDENCE.json').read_text())
        self.assertFalse(audit['quality_accepted']);self.assertTrue(audit['source_unchanged'])

if __name__=='__main__': unittest.main()
