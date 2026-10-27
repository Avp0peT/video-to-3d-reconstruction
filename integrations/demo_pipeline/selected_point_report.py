"""Read-only, identity-bound visual receipt for native selected SIFT rows."""
from pathlib import Path
import hashlib
import html
import json
import math
import sqlite3
import struct

from PIL import Image, ImageDraw, ImageFilter, ImageChops


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def build_report(sfm_run: Path, output: Path, baseline_run: Path | None = None) -> dict:
    """Verify receipts and native DB slices, then publish a fresh diagnostic report.

    This is feature provenance verification, never geometry or mask acceptance.
    No native model, inference, matching, or geometry operation is executed.
    """
    root = Path(sfm_run).resolve(strict=True)
    out = Path(output).resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('fresh output outside the input run required')
    bound = {}

    def read(path, expected=None):
        path = Path(path).resolve(strict=True)
        data = path.read_bytes()
        receipt = dict(path=str(path), bytes=len(data), sha256=_sha(data))
        if expected and any(receipt[k] != expected[k] for k in ('bytes', 'sha256') if k in expected):
            raise ValueError('input binding drift: ' + str(path))
        if str(path) in bound and bound[str(path)] != receipt:
            raise ValueError('input changed during report: ' + str(path))
        bound[str(path)] = receipt
        return data

    def load(path):
        return json.loads(read(path).decode('utf-8-sig'))

    run = load(root / 'run.json')
    audit_raw = read(root / 'point_selection.json')
    audit = json.loads(audit_raw)
    declared = dict(run.get('point_selection', {}))
    if declared.pop('evidence_sha256', None) != _sha(audit_raw) or declared != audit:
        raise ValueError('run/point selection ledger differs')
    if audit.get('status') != 'VERIFIED' or audit.get('policy_id') != 'rcover-v1':
        raise ValueError('verified native point selection required')
    rows = [json.loads(x) for x in read(root / 'frames_map.jsonl').decode('utf-8-sig').splitlines() if x.strip()]
    masks = run['effective_options']['foreground']['mask_bindings']
    by_mask = {m['image_name']: m for m in masks}
    selected = audit['per_image']
    if (not 1 <= len(selected) <= 96 or len({r['image_name'] for r in selected}) != len(selected)
            or audit.get('images') != selected or len(by_mask) != len(masks)
            or {r['image_name'] for r in selected} != set(by_mask)):
        raise ValueError('mask/selection coverage differs')
    backup = Path(audit['before_database_path']).resolve(strict=True)
    if str(backup) != str(Path(audit['source_database_backup']).resolve(strict=True)):
        raise ValueError('backup aliases differ')
    read(backup, {'sha256': audit['before_database_sha256']})
    database = root / 'database.db'
    read(database)
    pictures = {}
    indices = {}
    for item in selected:
        name = item['image_name']
        image_path = (root / name).resolve(strict=True)
        if not image_path.is_relative_to(root):
            raise ValueError('image escapes run')
        corresponding = [r for r in rows if r.get('staged_image_name') == name]
        if not corresponding:
            raise ValueError('image missing from frames map')
        digest = _sha(read(image_path))
        if any(r['canonical_image_sha256'] != digest for r in corresponding):
            raise ValueError('canonical image identity differs')
        indices[name] = min(r['input_index'] for r in corresponding)
        mask = by_mask[name]
        if mask['sha256'] != item['mask_sha256']:
            raise ValueError('selected mask differs')
        for field in ('source_path', 'staged_path'):
            read(mask[field], mask)
        with Image.open(image_path) as im:
            im.load()
            pictures[name] = im.convert('RGB')
        with Image.open(mask['staged_path']) as im:
            im.load()
            if im.mode != 'L' or not set(im.tobytes()).issubset({0, 255}) or im.size != pictures[name].size:
                raise ValueError('binary mask/image shape differs')

    with sqlite3.connect(backup.as_uri() + '?mode=ro', uri=True) as before, sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as after:
        for connection in (before, after):
            actual = dict(connection.execute('SELECT image_id,name FROM images'))
            if actual != {r['image_id']: r['image_name'] for r in selected}:
                raise ValueError('native database image coverage differs')
        for item in selected:
            ident = item['image_id']
            chosen = item['selected_original_rows']
            n = item['original_rows']
            if chosen != sorted(set(chosen)) or any(type(i) is not int or not 0 <= i < n for i in chosen):
                raise ValueError('invalid selected row mapping')
            mapping = [None] * n
            for new, old in enumerate(chosen):
                mapping[old] = new
            if mapping != item['original_to_new'] or len(chosen) != item['retained_rows']:
                raise ValueError('selected row mapping differs')
            for table, stride in (('keypoints', None), ('descriptors', 128)):
                fields = 'rows,cols,data' + (',type' if table == 'descriptors' else '')
                old = before.execute('SELECT ' + fields + ' FROM ' + table + ' WHERE image_id=?', (ident,)).fetchone()
                new = after.execute('SELECT ' + fields + ' FROM ' + table + ' WHERE image_id=?', (ident,)).fetchone()
                if old is None or new is None or old[0] != n or new[0] != len(chosen) or old[1] != new[1]:
                    raise ValueError('native feature dimensions differ')
                if table == 'keypoints':
                    if old[1] not in (2, 4, 6):
                        raise ValueError('keypoint columns differ')
                    stride = old[1] * 4
                    coords = [list(struct.unpack_from('<ff', old[2], i * stride)) for i in range(n)]
                    if coords != item['original_coordinates'] or any(not all(math.isfinite(v) for v in p) for p in coords):
                        raise ValueError('original coordinates differ')
                elif old[1] != 128 or old[3] != 0 or new[3] != 0:
                    raise ValueError('descriptor format differs')
                old_data, new_data = old[2] or b'', new[2] or b''
                sliced = b''.join(old_data[i * stride:(i + 1) * stride] for i in chosen)
                if len(old_data) != n * stride or len(new_data) != len(chosen) * stride or new_data != sliced:
                    raise ValueError('native feature slice differs: ' + table)
                if _sha(old_data) != item['before_' + table + '_sha256'] or _sha(new_data) != item['after_' + table + '_sha256']:
                    raise ValueError('native feature digest differs: ' + table)

    if (audit['before_total'] != sum(r['original_rows'] for r in selected)
            or audit['after_total'] != sum(r['retained_rows'] for r in selected)
            or audit['original_rows'] != audit['before_total']
            or audit['retained_rows'] != audit['after_total']):
        raise ValueError('selection totals differ')
    baseline = None
    if baseline_run is not None:
        base = Path(baseline_run).resolve(strict=True)
        br = load(base / 'run.json')
        bm = [json.loads(x) for x in read(base / 'frames_map.jsonl').decode('utf-8-sig').splitlines() if x.strip()]
        identity = lambda rs: sorted((r['input_index'], r['frame_id'], r['canonical_image_sha256']) for r in rs)
        foreground = lambda r: r.get('effective_options', {}).get('foreground', {}).get('manifest_sha256')
        if not run.get('upstream_file_snapshot_sha256') or run['upstream_file_snapshot_sha256'] != br.get('upstream_file_snapshot_sha256') or run.get('upstream_io_run_id') != br.get('upstream_io_run_id') or not foreground(run) or foreground(run) != foreground(br) or identity(rows) != identity(bm):
            raise ValueError('baseline upstream/manifest/image identity differs')
        for row in bm:
            if row.get('staged_image_name'):
                path = (base / row['staged_image_name']).resolve(strict=True)
                if not path.is_relative_to(base) or _sha(read(path)) != row['canonical_image_sha256']:
                    raise ValueError('baseline canonical image differs')
        baseline = dict(status=br['status'], counts=br['counts'], label='历史同输入参考；非严格因果对照')

    ordered = sorted(selected, key=lambda r: indices[r['image_name']])
    positions = sorted(set(round(i * (len(ordered) - 1) / min(5, len(ordered) - 1)) for i in range(min(6, len(ordered))))) if len(ordered) > 1 else [0]
    out.mkdir(parents=True, exist_ok=False)
    visuals = []
    cards = []
    for number, position in enumerate(positions):
        item = ordered[position]
        name = item['image_name']
        canvas = pictures[name].copy()
        with Image.open(by_mask[name]['staged_path']) as m:
            mask = m.copy()
        # Inner and outer one-pixel contours only; this is not a contour score.
        inner = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(3)))
        outer = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(3)), mask)
        canvas.paste((255, 55, 65), mask=inner)
        canvas.paste((0, 235, 240), mask=outer)
        draw = ImageDraw.Draw(canvas)
        width, height = canvas.size
        for x, y in item['original_coordinates']:
            if 0 <= x < width and 0 <= y < height:
                draw.ellipse((x-0.65, y-0.65, x+0.65, y+0.65), fill=(135, 135, 135))
        for index in item['selected_original_rows']:
            x, y = item['original_coordinates'][index]
            if 0 <= x < width and 0 <= y < height:
                draw.ellipse((x-1.6, y-1.6, x+1.6, y+1.6), fill=(255, 225, 0))
        path = out / ('overlay_%02d.png' % number)
        canvas.save(path)
        with Image.open(path) as check:
            check.load()
            if check.convert('RGB').tobytes() != canvas.tobytes():
                raise ValueError('overlay pixel readback differs')
        visuals.append(dict(file=path.name, bytes=path.stat().st_size, sha256=_sha(path.read_bytes()), pixel_sha256=_sha(canvas.tobytes()), input_index=indices[name]))
        cards.append('<figure><img src="%s"><figcaption>输入 %s · %s<br>原特征 %s → 选点 %s</figcaption></figure>' % (path.name, indices[name], html.escape(name), item['original_rows'], item['retained_rows']))
    counts = run['counts']
    native_points = sum(m.get('num_points3D', m.get('num_points', m.get('counts', {}).get('points3d', 0))) for m in run.get('models', []))
    result = dict(schema='selected-point-report-v1', status='DIAGNOSTIC_NOT_GEOMETRY_ACCEPTANCE', native_status=run['status'], original_features=audit['before_total'], selected_features=audit['after_total'], registered_images=counts.get('registered_unique_images', 0), model_count=counts.get('model_count', 0), native_points=native_points, native_database_slices_verified=True, baseline=baseline, overlays=visuals, quality_status='NOT_EVALUATED', source_unchanged=True)
    intro = '<h1>SAM 区域 → 算法选点 → 原生建模诊断</h1><p>状态：%s；原特征 %s → 选点 %s；注册照片 %s；模型 %s；原生稀疏点 %s。</p>' % (html.escape(run['status']), result['original_features'], result['selected_features'], result['registered_images'], result['model_count'], native_points)
    warning = '<p>红/青细线：SAM 掩膜内/外轮廓；灰小点：原始特征；黄点：保留并交给原生匹配的特征。数据库原始备份及当前关键点、描述子逐行切片已回读核验。图外特征省略绘制；描边和图外区域不评分。</p><p>SAM 是预测区域，可能包含手。不同照片的轮廓点或选点序号不能视为同一物理点。本页不是完整模型验收，不宣称几何精度；NO_MODEL 表示未形成原生模型。</p>'
    if baseline:
        warning += '<p>历史同输入参考；已核上游、清单、照片身份；非严格因果对照：%s</p>' % html.escape(json.dumps(baseline, ensure_ascii=False))
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>选点建模诊断</title><style>body{font:16px system-ui;background:#15191f;color:#eee;margin:32px;line-height:1.7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}figure{margin:0}img{width:100%;image-rendering:auto}figcaption{padding:8px}p{max-width:1100px}</style>' + intro + warning + '<main>' + ''.join(cards) + '</main></html>'
    (out / 'index.html').write_text(document, encoding='utf-8')
    for receipt in list(bound.values()):
        read(receipt['path'], receipt)
    result['input_bindings'] = [dict(r, before_sha256=r['sha256'], after_sha256=r['sha256'], unchanged=True) for r in bound.values()]
    (out / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result
