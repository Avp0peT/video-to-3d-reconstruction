"""Read-only frame overlays for conditional projective camera diagnostics."""
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from .intermediate_population_report import _link, _xy


def _escaped(value):
    return html.escape(str(value), quote=True)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def render_camera_review(output, projection, epipolar, decoded, *, budget=lambda: None):
    """Create new artifacts only; exceptions and budget cancellation propagate."""
    output = Path(output)
    budget()
    records = projection.get('records', [])
    names = ['index.html']
    cohorts = ('STRICT', 'STRICT_PLUS_RELAXED')
    selected = {cohort: [r for r in records[:12] if r.get('cohort') == cohort]
                for cohort in cohorts}
    for cohort in cohorts:
        if any(r.get('query_decode_index') in decoded for r in selected[cohort]):
            names.append(cohort.lower() + '.png')
    for name in names:
        if (output / name).exists():
            raise FileExistsError(output / name)
    files, sections = [], []
    for cohort in cohorts:
        budget()
        available = [r for r in selected[cohort] if r.get('query_decode_index') in decoded]
        name = cohort.lower() + '.png'
        if available:
            columns = min(3, len(available))
            sheet = Image.new('RGB', (404 * columns, 570 * math.ceil(len(available) / columns)), '#151a22')
            draw = ImageDraw.Draw(sheet)
            for slot, record in enumerate(available):
                budget()
                index = record['query_decode_index']
                frame = decoded[index]
                with Image.open(frame['path']) as source:
                    pic = source.convert('RGB')
                width, height = pic.size
                if (frame.get('width'), frame.get('height')) != (width, height):
                    raise ValueError('DECODED_IMAGE_SIZE_MISMATCH')
                pic.thumbnail((384, 490))
                left, top = slot % columns * 404, slot // columns * 570
                x, y = left + (404 - pic.width) // 2, top + 65
                sheet.paste(pic, (x, y))
                # Labels deliberately contain ASCII only (Pillow default font).
                draw.text((left + 8, top + 8), f'{cohort} record={slot + 1} decode={index}', fill='white')
                draw.text((left + 8, top + 26), 'orange: observed; cyan: holdout prediction', fill='white')
                mask = str(record.get('original_mask_status') or 'UNKNOWN')
                mask = mask.encode('ascii', 'backslashreplace').decode('ascii')
                draw.text((left + 8, top + 44), 'mask=' + mask, fill='#ff9f43')
                heldout = (record.get('assessment') or {}).get('heldout_fit') or {}
                predictions = {p['point_id']: p.get('predicted_raw_xy')
                               for p in (heldout.get('holdout_reprojection') or {}).get('rows', [])}
                def position(raw):
                    return x + (raw[0] - .5) * pic.width / width, y + (raw[1] - .5) * pic.height / height
                def inside(raw):
                    return _xy(raw) and .5 <= raw[0] <= width - .5 and .5 <= raw[1] <= height - .5
                outside_predictions = 0
                for point in record.get('points', []):
                    budget()
                    raw = point.get('raw_xy')
                    predicted = predictions.get(point.get('point_id'))
                    if _xy(predicted) and not inside(predicted):
                        outside_predictions += 1
                    if not inside(raw):
                        continue
                    ox, oy = position(raw)
                    if inside(predicted):
                        px, py = position(predicted)
                        draw.line((ox, oy, px, py), fill='#ff9f43', width=2)
                        draw.line((px - 4, py, px + 4, py), fill='#00e5ff', width=2)
                        draw.line((px, py - 4, px, py + 4), fill='#00e5ff', width=2)
                    draw.ellipse((ox - 4, oy - 4, ox + 4, oy + 4), outline='#ff9f43', width=2)
                draw.text((left + 8, top + 556), f'out-of-frame predictions omitted: {outside_predictions}', fill='white')
            budget()
            with (output / name).open('xb') as stream:
                sheet.save(stream, format='PNG')
            files.append(output / name)
            sections.append(f'<h2>{cohort}</h2><p>图像总计最多展示前 12 条记录；表格保留全部记录。</p><img src="{name}" alt="实际观测与留出预测">')
        else:
            sections.append(f'<h2>{cohort}</h2><p>没有可展示的实际帧；未生成占位图。</p>')
    rows = []
    for record in records:
        budget()
        assessment = record.get('assessment') or {}
        all_fit = assessment.get('all_fit') or {}
        heldout = assessment.get('heldout_fit') or {}
        errors = heldout.get('holdout_reprojection') or {}
        all_errors = all_fit.get('reprojection') or {}
        control = assessment.get('negative_control') or {}
        control_errors = control.get('holdout_reprojection') or {}
        frame = decoded.get(record.get('query_decode_index')) or {}
        source = ('<a href="' + _escaped(_link(frame['path'], output)) + '">实际原图</a>') if frame.get('path') else 'MISSING'
        values = [record.get('model_key', 'UNKNOWN'), record.get('query_decode_index'),
                  frame.get('time_seconds', 'UNKNOWN'), record.get('cohort', 'UNKNOWN'),
                  assessment.get('count', len(record.get('points', []))),
                  len(assessment.get('train_ids', [])), len(assessment.get('holdout_ids', [])),
                  all_errors.get('median_px'), all_errors.get('max_px'),
                  errors.get('median_px'), errors.get('max_px'),
                  control_errors.get('median_px'), control_errors.get('max_px'),
                  control.get('status', 'NOT_EVALUATED'), assessment.get('status', 'UNKNOWN'),
                  record.get('original_mask_status') or 'UNKNOWN', _json(assessment.get('reasons', []))]
        rows.append('<tr><td>' + source + '</td>' + ''.join('<td>' + _escaped(v) + '</td>' for v in values) + '</tr>')
    headings = ['原图', 'model', 'decode', '原图时间 (s)', 'cohort', '点数', 'train', 'holdout',
                'all median px', 'all max px', 'holdout median px', 'holdout max px',
                'control holdout median px', 'control holdout max px', 'control status', 'status', 'mask', 'reasons']
    epipolar_rows = []
    for pair in epipolar.get('pairs', []):
        budget()
        counts = pair.get('counts') or {}
        fundamental = pair.get('F') or {}
        homography = pair.get('H') or {}
        fhold, hhold = fundamental.get('holdout') or {}, homography.get('holdout') or {}
        control = pair.get('negative_control') or {}
        values = [f"{pair.get('source_decode_index')} → {pair.get('target_decode_index')}",
                  counts.get('points'), counts.get('train'), counts.get('holdout'),
                  fhold.get('median'), fhold.get('max'), hhold.get('median'), hhold.get('max'),
                  (control.get('F') or {}).get('median'), (control.get('H') or {}).get('median'),
                  fundamental.get('status', 'NOT_EVALUATED'), homography.get('status', 'NOT_EVALUATED'),
                  pair.get('status', 'UNKNOWN'), _json(pair.get('reasons', []))]
        epipolar_rows.append('<tr>' + ''.join('<td>' + _escaped(v) + '</td>' for v in values) + '</tr>')
    epi_headings = ['frame → frame', '点数', 'train', 'holdout', 'F holdout median px', 'F holdout max px',
                    'H holdout median px', 'H holdout max px', 'F control median px', 'H control median px',
                    'F status', 'H status', 'status', 'reasons']
    page = ('<!doctype html><meta charset="utf-8"><title>条件投影诊断</title>'
            '<style>body{font:16px sans-serif;margin:24px;background:#131b26;color:#e7edf5}a{color:#8ac9ff}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #586577;padding:6px;white-space:pre-wrap;overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>'
            '<h1>条件 XYZ 投影诊断</h1><p>在给定条件 XYZ 下比较实际二维观测与投影。未知畸变未校正；留出点共享原始数据，不等于独立真值。'
            '这些诊断不证明物理相机已恢复，不构成新的三维点或 CAD；geometry_accepted=false。</p>'
            '<p>橙圆为实际观测，青十字为留出预测，橙线连接对应点。无留出预测时只画观测。原图外的预测省略，每图底部列出数量。未知掩膜保持 UNKNOWN；None 指指标不可用。</p>'
            '<p><a href="PROJECTION.json">完整投影证据</a> · <a href="EPIPOLAR.json">EPIPOLAR.json</a></p>'
            + ''.join(sections) + '<h2>全部记录</h2><table><thead><tr>'
            + ''.join('<th>' + _escaped(h) + '</th>' for h in headings)
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
            + ('<p>没有投影记录。</p>' if not records else '')
            + '<h2>极线证据摘要</h2><p>F 为平方根 Sampson 像素残差；H 为正向重投影像素误差，二者不使用同一种残差定义。完整矩阵与逐点数据见 EPIPOLAR.json。</p>'
            + '<table><thead><tr>' + ''.join('<th>' + _escaped(h) + '</th>' for h in epi_headings)
            + '</tr></thead><tbody>' + ''.join(epipolar_rows) + '</tbody></table>'
            + ('<p>没有极线配对记录。</p>' if not epipolar_rows else ''))
    for name, text in [('index.html', page)]:
        budget()
        with (output / name).open('x', encoding='utf-8') as stream:
            stream.write(text)
        files.append(output / name)
    return files
