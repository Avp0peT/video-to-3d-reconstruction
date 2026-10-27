import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from integrations.demo_pipeline import learned_point_guarded as guard
from integrations.demo_pipeline.balanced_point_run import save


class GuardTests(unittest.TestCase):
    def test_supervisor_requires_every_clean_field(self):
        clean = dict(status='completed', errors=[], log_errors=[], root_exit_code=0, reclaim_confirmed=True)
        self.assertTrue(guard.completed(clean))
        for field, value in [('status', 'completed_with_errors'), ('errors', ['error']),
                             ('log_errors', ['log']), ('root_exit_code', 1), ('reclaim_confirmed', False)]:
            self.assertFalse(guard.completed(dict(clean, **{field: value})))
        for field in clean:
            self.assertFalse(guard.completed({k: v for k, v in clean.items() if k != field}))

    def test_track_frame_and_query_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save(out/'INPUTS.json', {'frames': [{'decode_index': 7}, {'decode_index': 9}]})
            save(out/'QUERIES.json', {'points': [{'track_id': 0}]})
            tracks = dict(status='TRACKS_GENERATED_NOT_IDENTITY_VERIFIED', frames=[
                dict(decode_index=i, points=[dict(track_id=0, status='VISIBLE_CANDIDATE')]) for i in [7, 9]])
            plan = dict(frames=2, query_count=1)
            save(out/'TRACKS.json', tracks)
            guard.validate_tracks(out, plan)
            for change in ('missing', 'id', 'failed'):
                import copy
                wrong = copy.deepcopy(tracks)
                if change == 'missing': wrong['frames'].pop()
                elif change == 'id': wrong['frames'][0]['points'][0]['track_id'] = 2
                else: wrong['frames'][0]['points'][0]['status'] = 'FAILED'
                save(out/'TRACKS.json', wrong)
                with self.assertRaises(ValueError): guard.validate_tracks(out, plan)

    def test_report_rejection_has_no_output_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'index.html').write_text('preserve')
            before = sorted(out.iterdir())
            with patch.object(guard, 'check', side_effect=ValueError('source drift')):
                with self.assertRaises(ValueError): guard.report(out)
            self.assertEqual(before, sorted(out.iterdir()))
            self.assertEqual((out/'index.html').read_text(), 'preserve')

    def test_missing_receipt_rejected_before_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(guard, 'check', return_value={}):
                with self.assertRaises(FileNotFoundError): guard.report(out)
            self.assertEqual(list(out.iterdir()), [])

    def test_failed_supervision_never_calls_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plan = dict(python='unused', wall_seconds=1, job_commit_bytes=1)
            with patch.object(guard, 'check', return_value=plan), \
                    patch('integrations.demo_pipeline.masked_trace_run.invoke', return_value={
                        'status': 'completed_with_errors', 'root_exit_code': 0,
                        'reclaim_confirmed': True, 'errors': [], 'log_errors': []}), \
                    patch.object(guard, 'report') as render:
                self.assertEqual(guard.run(out), 1)
                render.assert_not_called()
            self.assertFalse((out/guard.RECEIPT).exists())
            self.assertEqual(guard.read(out/'RESULT.json')['status'], 'FAILED')

    def report_fixture(self, out):
        from PIL import Image
        Image.new('RGB', (64, 64), '#335577').save(out/'source.png')
        source = guard.binding(out/'source.png')
        save(out/'INPUTS.json', dict(frames=[dict(path=source['path'], sha256=source['sha256'],
             decode_index=7, time_seconds=0.0)], seed=dict(xy=[32, 32])))
        save(out/'QUERIES.json', dict(points=[dict(track_id=0, xy=[32, 32])]))
        save(out/'TRACKS.json', dict(status='TRACKS_GENERATED_NOT_IDENTITY_VERIFIED',
             frames=[dict(decode_index=7, points=[dict(track_id=0, xy=[32, 32],
             status='VISIBLE_CANDIDATE', supported=True, visibility_probability=1.0, certainty=1.0)])]))
        (out/'checkpoint.fake').write_bytes(b'not a model')
        save(out/'PLAN.json', dict(schema='learned-point-plan-v2',
             publication_guard='guarded-controller-v2', frames=1, query_count=1,
             implementation=guard.controllers(), upstream_bindings=[],
             checkpoint=guard.binding(out/'checkpoint.fake')))
        save(out/'SOURCE_BINDINGS.json', [guard.binding(out/n) for n in
             ('source.png', 'INPUTS.json', 'QUERIES.json', 'PLAN.json')])
        entry = dict(status='completed', errors=[], log_errors=[], root_exit_code=0, reclaim_confirmed=True)
        (out/'processes/inference').mkdir(parents=True)
        save(out/'processes/inference/ENTRY.json', entry)
        save(out/guard.RECEIPT, dict(schema='guarded-run-receipt-v2', inference=entry,
             bindings=guard.snapshot(out, guard.check(out))))

    def test_real_pil_report_and_preserved_input_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.report_fixture(out)
            before = guard.snapshot(out, guard.check(out))
            temporary_directory = tempfile.TemporaryDirectory
            with patch.object(guard.tempfile, 'TemporaryDirectory', wraps=temporary_directory) as staging:
                summary = guard.report(out)
                self.assertEqual(Path(staging.call_args.kwargs['dir']), out)
            self.assertEqual(summary['frames'], 1)
            self.assertEqual(before, guard.snapshot(out, guard.check(out)))
            self.assertTrue(all((out/name).exists() for name in guard.OUTPUTS))
            self.assertFalse(list(out.glob('learned-report-*')))
            published = guard.binding(out/'index.html')
            with self.assertRaises(FileExistsError): guard.report(out)
            self.assertEqual(published, guard.binding(out/'index.html'))

    def test_source_mismatch_rejected_before_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.report_fixture(out)
            (out/'source.png').write_bytes(b'changed')
            before = sorted(str(p.relative_to(out)) for p in out.rglob('*'))
            with patch.object(guard.tempfile, 'TemporaryDirectory') as staging:
                with self.assertRaises(ValueError): guard.report(out)
                staging.assert_not_called()
            self.assertEqual(before, sorted(str(p.relative_to(out)) for p in out.rglob('*')))

    def test_source_mutation_during_render_rejected_before_publication(self):
        from integrations.demo_pipeline import learned_point_report
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.report_fixture(out)
            original_render = learned_point_report.render
            def mutate_after_render(stage):
                result = original_render(stage)
                (out/'source.png').write_bytes(b'changed after rendering')
                return result
            with patch.object(learned_point_report, 'render', mutate_after_render):
                with self.assertRaises(ValueError): guard.report(out)
            self.assertFalse(any((out/n).exists() for n in guard.OUTPUTS))
            self.assertFalse(list(out.glob('learned-report-*')))

    def test_declared_source_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.report_fixture(out)
            inputs = guard.read(out/'INPUTS.json')
            inputs['frames'][0]['sha256'] = '0' * 64
            save(out/'INPUTS.json', inputs)
            refs = guard.read(out/'SOURCE_BINDINGS.json')
            save(out/'SOURCE_BINDINGS.json', [guard.binding(r['path']) for r in refs])
            with patch.object(guard.tempfile, 'TemporaryDirectory') as staging:
                with self.assertRaisesRegex(ValueError, 'source frame SHA'):
                    guard.report(out)
                staging.assert_not_called()

    def test_tracks_mutation_during_render_rejected(self):
        from integrations.demo_pipeline import learned_point_report
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.report_fixture(out)
            original_render = learned_point_report.render
            def mutate_after_render(stage):
                result = original_render(stage)
                tracks = guard.read(out/'TRACKS.json')
                tracks['frames'][0]['points'][0]['xy'] = [5, 5]
                save(out/'TRACKS.json', tracks)
                return result
            with patch.object(learned_point_report, 'render', mutate_after_render):
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    guard.report(out)
            self.assertFalse(any((out/n).exists() for n in guard.OUTPUTS))
            self.assertFalse(list(out.glob('learned-report-*')))

    def test_prepare_freezes_extended_plan_before_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            from argparse import Namespace
            def fake_prepare(args):
                guard.legacy.save(out/'PLAN.json', dict(implementation=[]))
                guard.legacy.save(out/'SOURCE_BINDINGS.json', [])
            with patch.object(guard.legacy, 'prepare', fake_prepare):
                guard.prepare(Namespace(output=out))
            plan = guard.read(out/'PLAN.json')
            self.assertEqual(plan['publication_guard'], 'guarded-controller-v2')
            self.assertEqual(len(plan['implementation']), 2)
            self.assertEqual(guard.read(out/'SOURCE_BINDINGS.json'), [guard.binding(out/'PLAN.json')])
            with self.assertRaises(FileExistsError): guard.prepare(Namespace(output=out))

if __name__ == '__main__':
    unittest.main()
