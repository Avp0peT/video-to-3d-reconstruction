"""Small identity rejection cases; no prepare, native process or model launch."""
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from integrations.demo_pipeline import masked_trace_inputs as subject


def task_temp():
    parent = os.environ.get('TEMP') or os.environ.get('TMP')
    if not parent or not Path(parent).is_dir():
        raise RuntimeError('test supervisor must supply an existing TEMP/TMP directory')
    return tempfile.TemporaryDirectory(prefix='masked_trace_inputs_', dir=parent)


def fixture(root):
    """TEST_DATA with deliberately different canonical PNG byte encoding."""
    io = root / 'io'
    io.mkdir()
    (io / 'images').mkdir()
    raw_path = root / 'raw.png'
    canonical = io / 'images' / '000000.png'
    Image.new('RGB', (8, 6), (30, 70, 110)).save(raw_path, compress_level=0)
    metadata = PngInfo()
    metadata.add_text('TEST_DATA', 'same pixels; different PNG encoding')
    with Image.open(raw_path) as image:
        image.save(canonical, compress_level=9, pnginfo=metadata)
    subject._save(io / 'run.json', dict(run_id='test-io', status='SUCCESS'))
    record = dict(input_index=0, status='OK', frame_id='test-frame', input_image='images/raw.png',
                  image_output='images/000000.png', image_sha256=subject._binding(raw_path)['sha256'])
    (io / 'frames.jsonl').write_text(__import__('json').dumps(record) + '\n', encoding='utf-8')
    sam_root = root / 'sam'
    sam_root.mkdir()
    mask_path = sam_root / 'mask.png'
    Image.new('L', (8, 6), 255).save(mask_path)
    mask_sha = subject._binding(mask_path)['sha256']
    manifest_path = sam_root / 'manifest.json'
    source = dict(path=str(io), run_id='test-io',
                  run_json_sha256=subject._binding(io / 'run.json')['sha256'],
                  frames_jsonl_sha256=subject._binding(io / 'frames.jsonl')['sha256'])
    frame = dict(input_index=0, status='OK', frame_id='test-frame', source_image=str(canonical),
                 source_image_sha256=subject._binding(canonical)['sha256'],
                 photo_io_image_sha256=subject._binding(raw_path)['sha256'], width=8, height=6,
                 selected_candidate_id='c00', selected_mask=dict(path='mask.png', sha256=mask_sha),
                 candidates=[dict(candidate_id='c00', mask_path='mask.png', mask_sha256=mask_sha)])
    manifest = dict(source_io=source, frames=[frame])
    subject._save(manifest_path, manifest)
    raw = dict(path=str(raw_path), sha256=subject._binding(raw_path)['sha256'],
               decoded_rgb_sha256=subject._rgb(raw_path)[2])
    return manifest_path, manifest, frame, raw, io


class MaskedTraceInputIdentityTests(unittest.TestCase):
    def test_actual_selected_mask_accepts_identical_rgb_with_different_png_bytes(self):
        with task_temp() as folder:
            args = fixture(Path(folder))
            self.assertNotEqual(args[2]['source_image_sha256'], args[3]['sha256'])
            result = subject._sam_frame(*args)
            self.assertTrue(result['current_observed'])
            self.assertEqual(result['mask_semantics'], 'ACTUAL_SAM_SELECTED_NOT_HUMAN_GROUND_TRUTH')

    def test_current_canonical_pixel_drift_rejected_even_if_declared_byte_hash_updated(self):
        with task_temp() as folder:
            args = fixture(Path(folder))
            image = Path(args[2]['source_image'])
            Image.new('RGB', (8, 6), (30, 70, 111)).save(image)
            args[2]['source_image_sha256'] = subject._binding(image)['sha256']
            with self.assertRaisesRegex(ValueError, 'RGB identity mismatch'):
                subject._sam_frame(*args)

    def test_manifest_source_io_digest_drift_rejected(self):
        with task_temp() as folder:
            args = fixture(Path(folder))
            args[1]['source_io']['run_json_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'source IO digest mismatch'):
                subject._sam_frame(*args)

    def test_unbound_selected_candidate_rejected(self):
        with task_temp() as folder:
            args = fixture(Path(folder))
            args[2]['selected_candidate_id'] = 'c01'
            with self.assertRaisesRegex(ValueError, 'selected candidate binding mismatch'):
                subject._sam_frame(*args)

    def test_multichannel_mask_rejected_with_consistent_declared_hashes(self):
        with task_temp() as folder:
            args = fixture(Path(folder))
            mask = args[0].parent / 'mask.png'
            Image.new('RGB', (8, 6), (255, 255, 255)).save(mask)
            digest = subject._binding(mask)['sha256']
            args[2]['selected_mask']['sha256'] = digest
            args[2]['candidates'][0]['mask_sha256'] = digest
            with self.assertRaisesRegex(ValueError, 'single-channel'):
                subject._sam_frame(*args)

    def test_no_match_stays_unknown_and_cannot_contain_selected_mask(self):
        with task_temp() as folder:
            args = fixture(Path(folder))
            args[2].update(status='NO_MATCH', selected_mask=None, selected_candidate_id=None, candidates=[])
            result = subject._sam_frame(*args)
            self.assertFalse(result['current_observed'])
            self.assertIsNone(result['mask_path'])
            args[2]['selected_mask'] = dict(path='mask.png', sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'non-OK SAM frame'):
                subject._sam_frame(*args)

    def test_relative_path_escape_rejected(self):
        with task_temp() as folder:
            root = Path(folder)
            inside = root / 'inside'
            inside.mkdir()
            (root / 'outside.txt').write_text('TEST_DATA', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'escapes bound root'):
                subject._inside(inside, '../outside.txt')


if __name__ == '__main__':
    unittest.main()
