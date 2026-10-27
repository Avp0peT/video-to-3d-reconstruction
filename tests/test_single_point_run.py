"""Runner/report boundary regressions; no video decode, OpenCV or native SfM.

Temporary artifacts inherit the supervising task's TMP directory. The trace used
for report integration exercises the real state machine with controlled step
responses, rather than claiming synthetic coordinates are image measurements.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from scripts.prepare_video import inspect_image
from integrations.demo_pipeline.single_point_run import verify_endpoint
from integrations.demo_pipeline.single_point_report import render
from integrations.demo_pipeline import single_point_trace
from tests.test_single_point_trace import PatchOnlyCV,proposal


def task_temp():
    return tempfile.TemporaryDirectory(prefix='single_point_run_',dir=os.environ.get('TMP') or os.environ.get('TEMP'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SinglePointRunTests(unittest.TestCase):
    def test_png_decoded_identity_matches_existing_inspector_across_encodings(self):
        with task_temp() as folder:
            root=Path(folder)
            pixels=np.arange(13*9*3,dtype=np.uint16).reshape(9,13,3).astype(np.uint8)
            first=root/'raw.png';second=root/'alternate_encoding.png'
            Image.fromarray(pixels).save(first,compress_level=0)
            metadata=PngInfo();metadata.add_text('TEST_DATA','different PNG encoding, same RGB pixels')
            Image.fromarray(pixels).save(second,compress_level=9,pnginfo=metadata)
            a=inspect_image(first);b=inspect_image(second)
            self.assertNotEqual(a['sha256'],b['sha256'])
            self.assertEqual(a['decoded_rgb_sha256'],b['decoded_rgb_sha256'])
            expected=hashlib.sha256(b'13x9:RGB:'+pixels.tobytes()).hexdigest()
            self.assertEqual(a['decoded_rgb_sha256'],expected)
            for canonical in (first,second):
                saved=dict(image_path=str(canonical),decode_index=7,decoded_rgb_sha256=a['decoded_rgb_sha256'])
                checked=verify_endpoint(pixels,saved)
                self.assertEqual(checked,dict(decode_index=7,decoded_rgb_sha256=expected,equal=True))

    def test_pixel_and_declared_digest_mismatches_reject(self):
        with task_temp() as folder:
            root=Path(folder);path=root/'canonical.png';pixels=np.full((12,14,3),77,np.uint8)
            Image.fromarray(pixels).save(path)
            saved=dict(image_path=str(path),decode_index=1,decoded_rgb_sha256=inspect_image(path)['decoded_rgb_sha256'])
            changed=pixels.copy();changed[4,5,1]+=1
            with self.assertRaisesRegex(ValueError,'pixel identity failed'):verify_endpoint(changed,saved)
            incorrect={**saved,'decoded_rgb_sha256':'0'*64}
            with self.assertRaisesRegex(ValueError,'pixel identity failed'):verify_endpoint(pixels,incorrect)
            # A matching declared raw digest cannot hide changed canonical pixels.
            Image.fromarray(changed).save(path)
            with self.assertRaisesRegex(ValueError,'pixel identity failed'):verify_endpoint(pixels,saved)

    def test_report_consumes_actual_trace_seed_failure_and_no_not_tracked_point(self):
        with task_temp() as folder:
            root=Path(folder);output=root/'report';output.mkdir()
            pixels=np.full((64,64,3),[40,50,60],np.uint8)
            sources=[]
            for index in range(4):
                path=root/f'frame_{index:06d}.png';Image.fromarray(pixels).save(path)
                sources.append(dict(decode_index=10+index,path=str(path),sha256=sha(path),time_seconds=index*.04))
            steps=[proposal([21,20],True),proposal([44,44],False,'forward_backward',2.),
                   proposal([44,44],False,'lk_forward',float('inf'))]
            with mock.patch.object(single_point_trace,'photometric_step',side_effect=steps):
                trace=single_point_trace.trace([np.full((64,64),50,np.uint8) for _ in range(4)],
                    [10,11,12,13],[20,20],PatchOnlyCV())
            self.assertEqual(trace['frames'][0]['candidate_status'],'SELECTED_SAVED_OBSERVATION')
            self.assertEqual(trace['first_failure']['decode_index'],12)
            self.assertEqual(trace['frames'][-1]['candidate_status'],'NOT_TRACKED')
            inputs=dict(seed=dict(decode_index=10,xy=[20,20],track_id=7),
                comparison=dict(decode_index=11,xy=[21,20],track_id=999,used_as_algorithm_input=False),
                frames=sources,bindings=[])
            (output/'INPUTS.json').write_text(json.dumps(inputs),encoding='utf-8')
            (output/'TRACE.json').write_text(json.dumps(trace,allow_nan=False),encoding='utf-8')
            original_hashes={Path(frame['path']):frame['sha256'] for frame in sources}
            summary=render(output)
            self.assertEqual(summary['status'],'REPORT_GENERATED_NOT_ACCEPTED')
            self.assertFalse(summary['geometry_accepted']);self.assertFalse(summary['track_identity_truth'])
            self.assertEqual(summary['first_alarm_decode'],12)
            self.assertEqual(summary['source_drift'],[]);self.assertEqual(summary['link_check']['missing'],[])
            self.assertEqual(summary['frames'][-1]['candidate_status'],'NOT_TRACKED')
            self.assertIsNone(summary['frames'][-1]['display_xy'])
            self.assertIsNone(summary['frames'][-1]['candidate_xy'])
            self.assertIsNone(summary['frames'][-1]['accepted_xy'])
            self.assertTrue((output/'index.html').is_file());self.assertTrue((output/'motion.gif').is_file())
            self.assertTrue((output/'contact_sheet.png').is_file());self.assertTrue((output/'first_alarm_local.png').is_file())
            page=(output/'index.html').read_text(encoding='utf-8')
            payload=json.loads(page.split('const D=',1)[1].split(';const s=',1)[0])
            self.assertIsNone(payload['frames'][-1]['display_xy'])
            self.assertEqual(payload['frames'][0]['candidate_status'],'SELECTED_SAVED_OBSERVATION')
            # The NOT_TRACKED GIF frame may retain historical line segments,
            # but must not draw a new marker at the old shadow position (44,44).
            with Image.open(output/'motion.gif') as gif:
                self.assertEqual(gif.n_frames,4);gif.seek(3);last=np.asarray(gif.convert('RGB'))
            image_left=(576-64)//2;image_top=58
            for x,y in ((37,44),(51,44),(44,37),(44,51)):
                np.testing.assert_array_equal(last[image_top+y,image_left+x],[40,50,60])
            for path,digest in original_hashes.items():self.assertEqual(sha(path),digest)
            for source,display in zip(sources,summary['frames']):
                self.assertEqual(sha(output/display['image']),source['sha256'])


if __name__=='__main__':unittest.main()
