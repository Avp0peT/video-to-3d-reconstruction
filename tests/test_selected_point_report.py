"""Synthetic native SQLite receipts; no model/GPU dependency."""
import hashlib
import json
import sqlite3

import pytest
from PIL import Image

from integrations.demo_pipeline.selected_point_report import build_report
from kozlov_lab.reconstruction.point_selection import apply_point_selection
from test_reconstruction_point_selection import fixture


def make_run(tmp_path):
    root = tmp_path / 'sfm'
    root.mkdir()
    db, masks, _ = fixture(root)
    # A second image exercises native image coverage and ordered overlay output.
    with sqlite3.connect(db) as c:
        c.execute('INSERT INTO images VALUES(2,"two.png",1)')
        c.execute('INSERT INTO keypoints SELECT 2,rows,cols,data FROM keypoints WHERE image_id=1')
        c.execute('INSERT INTO descriptors SELECT 2,rows,cols,data,type FROM descriptors WHERE image_id=1')
    masks.append(dict(masks[0], image_name='two.png'))
    audit = apply_point_selection(db, masks, root / 'point_selection.json')
    maps = []
    for i, name in enumerate(('one.png', 'two.png')):
        image = root / name
        Image.new('RGB', (100, 100), (35 + i, 45, 55)).save(image)
        maps.append(dict(input_index=i, frame_id=str(i), staged_image_name=name,
                         canonical_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    (root / 'frames_map.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in maps), encoding='utf-8')
    run = dict(status='NO_MODEL', point_selection=dict(audit, evidence_sha256=hashlib.sha256((root / 'point_selection.json').read_bytes()).hexdigest()),
               effective_options=dict(foreground=dict(mask_bindings=masks, manifest_sha256='a' * 64)),
               counts=dict(registered_unique_images=0, model_count=0), models=[],
               upstream_io_run_id='same', upstream_file_snapshot_sha256='b' * 64)
    (root / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    return root


def test_native_readback_no_model_and_pixels(tmp_path):
    root = make_run(tmp_path)
    result = build_report(root, tmp_path / 'report')
    assert result['native_status'] == 'NO_MODEL'
    assert result['selected_features'] == 4 and result['original_features'] == 6
    assert result['native_database_slices_verified'] and len(result['overlays']) == 2
    assert result['model_count'] == result['native_points'] == 0
    for row in result['overlays']:
        with Image.open(tmp_path / 'report' / row['file']) as image:
            assert hashlib.sha256(image.convert('RGB').tobytes()).hexdigest() == row['pixel_sha256']
            assert image.getpixel((10, 10)) == (255, 225, 0)
    page = (tmp_path / 'report' / 'index.html').read_text(encoding='utf-8')
    assert '不是完整模型验收' in page and '可能包含手' in page and 'NO_MODEL' in page
    with pytest.raises(ValueError, match='fresh output'):
        build_report(root, tmp_path / 'report')


@pytest.mark.parametrize('tamper', ['descriptor', 'coordinate', 'receipt', 'image', 'backup', 'mask'])
def test_rejects_truth_negative_receipts(tmp_path, tamper):
    root = make_run(tmp_path)
    if tamper == 'descriptor':
        with sqlite3.connect(root / 'database.db') as c:
            c.execute('UPDATE descriptors SET data=zeroblob(rows*128) WHERE image_id=1')
    elif tamper == 'image':
        Image.new('RGB', (100, 100), 'red').save(root / 'one.png')
    elif tamper == 'backup':
        with (root / 'point_selection.json.before.db').open('ab') as f:
            f.write(b'changed')
    elif tamper == 'mask':
        Image.new('L', (100, 100), 0).save(root / 'mask.png')
    else:
        path = root / 'run.json'
        run = json.loads(path.read_text())
        if tamper == 'receipt':
            run['point_selection']['evidence_sha256'] = '0' * 64
        else:
            # Even a mutually consistent forged JSON receipt cannot alter SQLite truth.
            ledger = root / 'point_selection.json'
            audit = json.loads(ledger.read_text())
            audit['per_image'][0]['original_coordinates'][0][0] += 1
            audit['images'] = audit['per_image']
            ledger.write_text(json.dumps(audit))
            run['point_selection'] = dict(audit, evidence_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest())
        path.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        build_report(root, tmp_path / 'report')
    assert not (tmp_path / 'report').exists()


def test_baseline_requires_matching_provenance(tmp_path):
    root = make_run(tmp_path)
    # The same verified run is valid as historical identity reference only.
    result = build_report(root, tmp_path / 'report', root)
    assert '非严格因果' in result['baseline']['label']
    run = json.loads((root / 'run.json').read_text())
    run['upstream_file_snapshot_sha256'] = None
    (root / 'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError, match='baseline'):
        build_report(root, tmp_path / 'refused', root)
