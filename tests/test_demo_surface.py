import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from integrations.demo_pipeline.surface import choose_model, mask_indices, complete_surface, read, selection_ledger


class SurfacePolicyTests(unittest.TestCase):
    def test_json_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
                path.write_text(text)
                with self.assertRaises(ValueError):read(path)

    def test_component_choice_is_deterministic_and_explicit(self):
        models = [dict(model_id='2', usable=True, roundtrip_verified=True, num_registered_images=9, num_points3D=200),
                  dict(model_id='1', usable=True, roundtrip_verified=True, num_registered_images=9, num_points3D=200),
                  dict(model_id='0', usable=True, roundtrip_verified=True, num_registered_images=8, num_points3D=300)]
        selected, record = choose_model(models)
        self.assertEqual(selected['model_id'], '1')
        self.assertFalse(record['components_merged'])
        self.assertEqual(choose_model(models, '0')[0]['model_id'], '0')
        with self.assertRaises(ValueError):
            choose_model(models, 'missing')

    def test_no_match_is_excluded_and_duplicate_success_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            io = Path(tmp) / 'io'
            payload = {'contract_version': 'object-mask-v1', 'source_io': {'path': str(io)},
                       'frames': [{'input_index': 3, 'status': 'OK', 'selected_mask': {'path': 'x'}},
                                  {'input_index': 59, 'status': 'NO_MATCH'}]}
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertEqual(mask_indices([path], io), ([3], {59: 'NO_MATCH'}))
            with self.assertRaises(ValueError):
                mask_indices([path, path], io)
            with self.assertRaises(ValueError):
                mask_indices([path], Path(tmp) / 'other_io')

    def test_native_launch_requires_supervisor(self):
        with self.assertRaisesRegex(ValueError, 'supervisor'):
            complete_surface('.', '.', '.', [], '.', '.', 'python', '.', None)

    def test_multimanifest_selects_unique_ok_and_rejects_identity_or_success_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); primary=root/'primary.json'; supplement=root/'supplement.json'
            bad={'input_index':7,'frame_id':'actual-frame','status':'NO_MATCH',
                 'selected_mask':None,'selected_candidate_id':None}
            good=dict(bad,status='OK',selected_candidate_id='c00',selected_mask={'sha256':'a'*64})
            primary.write_text(json.dumps({'frames':[bad]}));supplement.write_text(json.dumps({'frames':[good]}))
            before=primary.read_bytes()
            ledger=selection_ledger([primary,supplement],[7],{'path':'cert','sha256':'b'*64})
            self.assertEqual(len(ledger['manifests']),2)
            self.assertEqual(len(ledger['selections']),1)
            self.assertEqual(ledger['selections'][0]['manifest_path'],str(supplement.resolve()))
            self.assertEqual(primary.read_bytes(),before)
            self.assertEqual(ledger['excluded_source_rows'],[])
            supplement.write_text(json.dumps({'frames':[dict(good,frame_id='conflicting-frame')]}))
            with self.assertRaisesRegex(ValueError,'conflicting frame identity'):
                selection_ledger([primary,supplement],[7],{})
            primary.write_text(json.dumps({'frames':[good]}));supplement.write_text(json.dumps({'frames':[good]}))
            with self.assertRaisesRegex(ValueError,'duplicate successful mask'):
                selection_ledger([primary,supplement],[7],{})

    def test_exception_marks_surface_failed_and_preserves_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sfm=root/'sfm';sfm.mkdir();(sfm/'run.json').write_text('{"models":[]}')
            run=root/'outputs'/'run';run.mkdir(parents=True)
            def unexpected_stage(*args,**kwargs):raise AssertionError('must not launch')
            result=complete_surface(sfm,root/'io',root/'dataset',[],run,root,'python',root/'tools',
                                    None,stage=unexpected_stage)
            self.assertEqual(result['surface']['status'],'FAILED')
            self.assertEqual(result['error'],'ValueError: requested usable SfM model is unavailable')
            self.assertEqual(result['surface']['error'],result['error'])
            self.assertEqual(result['stages'],{})

    def test_complete_orchestration_and_partial_failure_preserves_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); sfm = root / 'sfm'; sfm.mkdir()
            io = root / 'io'; io.mkdir()
            (sfm / 'run.json').write_text(json.dumps({'models': [dict(model_id='arbitrary',
                usable=True, roundtrip_verified=True, num_registered_images=2, num_points3D=5)]}))
            (sfm / 'frames_map.jsonl').write_text('\n'.join(json.dumps(dict(input_index=i,
                staged_image_name=f'images/{i}.png', registrations=[{'model_id':'arbitrary'}])) for i in [1, 8]))
            manifest = root / 'mask.json'
            manifest.write_text(json.dumps({'contract_version':'object-mask-v1',
                'source_io': {'path':str(io)}, 'frames':[dict(input_index=i,status='OK',frame_id=f'f{i}',
                    selected_candidate_id='c00',selected_mask={'path':'x','sha256':'a'*64}) for i in [1,8]]}))
            supplement=root/'supplement.json';supplement.write_bytes(manifest.read_bytes())
            supplement_doc=json.loads(supplement.read_text());supplement_doc['frames']=supplement_doc['frames'][:1]
            supplement.write_text(json.dumps(supplement_doc))
            primary_doc=json.loads(manifest.read_text());primary_doc['frames'][0].update(
                status='NO_MATCH',selected_mask=None,selected_candidate_id=None)
            manifest.write_text(json.dumps(primary_doc))
            def derive(*args, **kwargs):
                out = args[4]; out.mkdir(); cert = out / 'certificate.json'; cert.write_text('{}')
                return dict(selected_sparse=str(out / 'sparse'), subset_certificate={'path':str(cert)},
                            selected_indices=[1,8], missing_indices=[])
            def stage(name, argv, seconds, memory, extra=None):
                work = extra[0]
                if name == 'native_subset':
                    returned=derive(None,None,None,None,work/'subset')
                    returned['subset_certificate']['sha256']='a'*64
                    (work/'subset'/'RESULT.json').write_text(json.dumps(returned))
                elif name == 'mvs_bridge':
                    target=work/'bridge'; target.mkdir(); (target/'RESULT.json').write_text('{"status":"PASS"}')
                elif name == 'mvs_masks':
                    ledger=read(work/'MASK_SELECTION.json')
                    self.assertEqual(len(ledger['selections']),2)
                    self.assertEqual(next(x for x in ledger['selections'] if x['input_index']==1)['manifest_path'],str(supplement))
                    self.assertIn('--selection-ledger',argv)
                    self.assertEqual(argv.count('--mask-manifest'),2)
                    target=work/'masks'; target.mkdir(); (target/'manifest.json').write_text(
                        '{"coverage_status":"COMPLETE","publishable_arguments":true}')
                elif name == 'surface_plan':
                    (work/'SURFACE_PLAN.json').write_text('{}')
                elif name == 'surface_reconstruction':
                    target=work/'surface'; (target/'work').mkdir(parents=True)
                    (target/'result.json').write_text('{"status":"SURFACE_GENERATED_NOT_ACCEPTED","exit_code":0}')
                    (target/'work'/'__mvs_surface_raw.ply').write_bytes(b'offline fixture')
                elif name == 'blender_export':
                    return 5
                return 0
            run=root/'outputs'/'run'; run.mkdir(parents=True)
            result=complete_surface(sfm,io,root/'dataset.json',[manifest,supplement],run,root,
                                    'python',root/'tools',None,stage=stage)
            self.assertEqual(result['status'],'FAILED')
            self.assertEqual(result['exit_code'],5)
            self.assertEqual(result['surface']['status'],'SUCCESS')
            self.assertEqual(result['surface']['mask_consumption'],'NOT_VERIFIED')
            self.assertEqual(len(result['surface']['artifacts']),1)
            self.assertEqual(result['stages']['blender_export']['status'],'FAILED')


if __name__ == '__main__':
    unittest.main()
