"""Full export contracts and an optional tiny real CPU codec round trip."""
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from integrations.video_masks import video_export as export


def fixture(tmp_path, values=(0, 255, None)):
    run = tmp_path / 'run'
    (run / 'frames').mkdir(parents=True)
    rows = []
    for i, value in enumerate(values):
        Image.new('RGB', (16, 16), (100, 20, 40)).save(run / 'frames' / f'{i:06d}.jpg')
        source = tmp_path / f'mask{i}.png'
        if value is not None:
            Image.new('L', (16, 16), value).save(source)
        rows.append({'decode_index': i, 'mask_path': str(source) if value is not None else None,
                     'mask_sha256': export._hash(source) if value is not None else None,
                     'status': 'OK', 'independent_bidirectional': i == 0})
    timeline = tmp_path / 'timeline.json'
    timeline.write_text(json.dumps({'time_base': [1, 90000], 'frames': [
        {'decode_index': i, 'pts': round(i * 90000 / 28)} for i in range(len(values))]}))
    plan = {'timeline_path': str(timeline), 'width': 16, 'height': 16, 'total_frames': len(values)}
    return run, plan, rows


def test_timeline_detects_rounded_cfr_and_rejects_duplicates():
    timeline = {'time_base': [1, 90000], 'frames': [
        {'decode_index': i, 'pts': 100 + round(i * 90000 / 28)} for i in range(5)]}
    times, origin, cfr = export._times(timeline, 5)
    assert cfr and float(origin) == pytest.approx(100 / 90000)
    timeline['frames'][2]['pts'] = timeline['frames'][1]['pts']
    with pytest.raises(ValueError, match='strictly increasing'):
        export._times(timeline, 5)


def test_original_label_never_becomes_single_direction():
    label,color=export.overlay_annotation(933,'ORIGINAL_RETAINED',False,False)
    assert label=='000933 ORIGINAL_RETAINED'
    assert color==(40,210,255)
    assert export.overlay_annotation(934,'SINGLE_DIRECTION_CANDIDATE',False,False)[1]==(255,160,0)
    assert export.overlay_annotation(1,'BIDIRECTIONAL_CANDIDATE',False,True)[1]==(0,255,0)
    assert export.overlay_annotation(2,'UNKNOWN',True,False)[1]==(255,0,255)


def test_nonbinary_mask_fails_before_native_command(tmp_path, monkeypatch):
    run, plan, rows = fixture(tmp_path, (127,))
    monkeypatch.setattr(export.pipeline, '_invoke', lambda *a, **k: pytest.fail('native command launched'))
    with pytest.raises(ValueError, match='not binary'):
        export.export_video(tmp_path, run, plan, rows, 'ffmpeg', 'ffprobe')


@pytest.mark.parametrize('vfr', [False, True])
def test_full_cpu_roundtrip_preserves_all_pixels_and_unknown(tmp_path, vfr):
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('local ffmpeg and ffprobe unavailable')
    run, plan, rows = fixture(tmp_path)
    if vfr:
        path = Path(plan['timeline_path'])
        timeline = json.loads(path.read_text())
        timeline['frames'][2]['pts'] = 9000
        path.write_text(json.dumps(timeline))
    result = export.export_video(Path(__file__).resolve().parents[1], run, plan, rows, ffmpeg, ffprobe)
    assert result['pixel_readback_verified'] is True
    assert result['placeholder_count'] == 1
    assert result['counts'] == {'OK': 2, 'UNKNOWN': 1}
    assert all(row['decoded_frames'] == 3 for row in result['outputs'].values())
    manifest = json.loads((run / 'MANIFEST.json').read_text())['rows']
    assert manifest[2]['placeholder'] and manifest[2]['status'] == 'UNKNOWN'
    assert manifest[2]['pixel_sha256'] == hashlib.sha256(bytes(256)).hexdigest()
    assert not (run / 'export' / 'mask_readback.gray').exists()
    assert json.loads((run / 'EXPORT.json').read_text()) == result
