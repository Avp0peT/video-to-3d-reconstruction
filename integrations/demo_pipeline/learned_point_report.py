"""Render frozen learned-point predictions without importing or running a model."""
from pathlib import Path
import html
import json
import math

from PIL import Image, ImageDraw


def render(out: Path):
    """Read INPUTS/TRACKS/QUERIES and publish candidate-only visual evidence.

    Only VISIBLE_CANDIDATE records with supported=True and in-image finite xy
    receive a marker. Hidden/uncertain records never reuse an earlier position.
    """
    out = Path(out)
    names = ('motion.gif', 'contact_sheet.png', 'overview.png', 'index.html',
             'learned_frame_panels', 'LEARNED_POINT_REPORT.json')
    if any((out / name).exists() for name in names):
        raise FileExistsError('Report outputs already exist; use a fresh output directory')
    def read(name):
        return json.loads((out / name).read_text(encoding='utf-8-sig'))
    inputs, tracks, queries = read('INPUTS.json'), read('TRACKS.json'), read('QUERIES.json')
    sources = inputs['frames']
    if not sources:
        raise ValueError('No source frames')
    def key(value):
        return str(value)
    rows = {}
    for row in tracks['frames']:
        ident = key(row['decode_index'])
        if ident in rows:
            raise ValueError('Duplicate TRACKS decode_index: ' + ident)
        rows[ident] = row
    def xy(value):
        try:
            if len(value) == 2:
                point = tuple(float(v) for v in value)
                if all(math.isfinite(v) for v in point):
                    return point
        except (TypeError, ValueError, OverflowError):
            pass
        return None
    def supported(point):
        return point.get('supported') is True
    def drawable(point, image):
        position = xy(point.get('xy'))
        return (point.get('status') == 'VISIBLE_CANDIDATE' and supported(point)
                and position is not None and 0 <= position[0] < image.width
                and 0 <= position[1] < image.height)
    def resize(im, width):
        return im.resize((width, max(1, round(im.height * width / im.width))), Image.Resampling.LANCZOS)
    def mark(im, position, scale, label):
        x, y = (v * scale for v in position)
        d = ImageDraw.Draw(im)
        d.ellipse((x-5, y-5, x+5, y+5), outline='#20ff80', width=2)
        d.line((x-8, y, x+8, y), fill='#20ff80', width=1)
        d.line((x, y-8, x, y+8), fill='#20ff80', width=1)
        d.text((x+7, y+3), label, fill='#20ff80', stroke_width=1, stroke_fill='black')
    def frame_points(row):
        result = {}
        for point in row.get('points', []):
            ident = key(point['track_id'])
            if ident in result:
                raise ValueError('Duplicate track_id in a frame: ' + ident)
            result[ident] = point
        return result
    # Validate source bindings before creating outputs; no synthetic fallback data.
    prepared = []
    for source in sources:
        path = Path(source['path'])
        if not path.is_absolute():
            path = out / path
        with Image.open(path) as opened:
            size = opened.size
        row = rows.get(key(source['decode_index']), {})
        prepared.append((source, path, size, frame_points(row)))
    directory = out / 'learned_frame_panels'
    directory.mkdir()
    animations, panel_paths, ledger = [], [], []
    chosen = sorted(set(round(i * (len(sources)-1) / 7) for i in range(8)))
    selected, overviews = [], []
    seed = next((p.get('xy') for p in queries.get('points', []) if key(p.get('track_id')) == '0'), inputs.get('seed', {}).get('xy'))
    for n, (source, path, _, points) in enumerate(prepared):
        with Image.open(path) as opened:
            original = opened.convert('RGB')
        primary = points.get('0', {})
        visible = drawable(primary, original)
        status = str(primary.get('status', 'TRACK_MISSING'))
        base = resize(original, 360)
        if visible:
            mark(base, xy(primary['xy']), 360 / original.width, 'ID 0 candidate')
        animation = Image.new('RGB', (360, base.height+64), '#111820')
        animation.paste(base, (0, 64))
        d = ImageDraw.Draw(animation)
        d.text((7, 5), 'ID 0 | raw %s | t=%s' % (source['decode_index'], source.get('time_seconds', '?')), fill='white')
        d.text((7, 23), status, fill='#20ff80' if visible else '#ffbf75')
        d.text((7, 41), 'CANDIDATE ONLY' if visible else 'NO POSITION DRAWN', fill='white')
        animations.append(animation)
        panel = Image.new('RGB', (640, max(animation.height, 390)), '#111820')
        panel.paste(animation, (0, 0))
        pd = ImageDraw.Draw(panel)
        pd.text((372, 10), 'ID 0 local raw texture', fill='white')
        if visible:
            x, y = xy(primary['xy'])
            left, top = round(x)-64, round(y)-64
            patch = original.crop((left, top, left+129, top+129))
            panel.paste(patch.resize((258, 258), Image.Resampling.NEAREST), (372, 38))
            pd.text((372, 305), '129px crop; no marker overlay', fill='white')
        else:
            pd.text((372, 50), 'No crop: position unverified', fill='#ffbf75')
            pd.text((372, 70), 'No previous-position reuse', fill='white')
        pd.text((372, 334), 'QUALITY: NOT_EVALUATED', fill='white')
        filename = '%04d.png' % n
        panel.save(directory / filename)
        panel_paths.append('learned_frame_panels/' + filename)
        ledger.append({'decode_index': source['decode_index'], 'time_seconds': source.get('time_seconds'),
                       'status': status, 'position_drawn': visible, 'points': list(points.values())})
        if n in chosen:
            selected.append(panel)
            overview = resize(original, 360)
            for ident, point in points.items():
                if drawable(point, original):
                    mark(overview, xy(point['xy']), 360/original.width, 'ID '+ident)
            tile = Image.new('RGB', (360, overview.height+42), '#111820')
            tile.paste(overview, (0, 42))
            od = ImageDraw.Draw(tile)
            od.text((5, 5), 'raw %s | all visible candidates' % source['decode_index'], fill='white')
            od.text((5, 22), 'Hidden/unsupported points omitted', fill='white')
            overviews.append(tile)
    def sheet(tiles, columns, name):
        width, height = max(p.width for p in tiles), max(p.height for p in tiles)
        canvas = Image.new('RGB', (columns*width, math.ceil(len(tiles)/columns)*height), '#111820')
        for i, tile in enumerate(tiles):
            canvas.paste(tile, ((i % columns)*width, (i // columns)*height))
        canvas.save(out / name)
    sheet(selected, min(4, len(selected)), 'contact_sheet.png')
    sheet(overviews, min(4, len(overviews)), 'overview.png')
    # One fixed canvas keeps GIF frame dimensions stable for heterogeneous inputs.
    gif_height = max(p.height for p in animations)
    gif = []
    for panel in animations:
        canvas = Image.new('RGB', (360, gif_height), '#111820')
        canvas.paste(panel, (0, 0))
        gif.append(canvas.convert('P', palette=Image.Palette.ADAPTIVE))
    gif[0].save(out/'motion.gif', save_all=True, append_images=gif[1:], duration=600, loop=0, disposal=2)
    esc = lambda value: html.escape(str(value))
    table = ''.join('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
        esc(row['decode_index']), esc(point.get('track_id')), esc(point.get('status')),
        esc(point.get('visibility_probability')), esc(point.get('certainty')), esc(point.get('supported')))
        for row in ledger for point in row['points'])
    data = json.dumps({'paths': panel_paths, 'labels': [str(s['decode_index']) for s in sources]}, ensure_ascii=True).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>学习点跟踪候选证据</title>
<style>body{max-width:1400px;margin:auto;padding:24px;background:#111820;color:#eee;font:16px sans-serif}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #778;padding:6px}input{width:100%}pre{white-space:pre-wrap}a{color:#9cf}</style>
<h1>同一点 ID 0：逐帧候选证据</h1><p>质量状态：<b>NOT_EVALUATED</b>。本报告只读取冻结记录，不运行模型，不验收物理同点身份或遮挡原因。</p>
<p>优先检查初始 raw 555 的点 (410,830) 对应 ID 0；实际绑定以输入与查询原文为准。绿色仅表示 supported=true、位于图像内的 VISIBLE_CANDIDATE，仍是未验收候选。</p>
<p>OCCLUDED_OR_UNCERTAIN、OUTSIDE_MASK、MASK_UNAVAILABLE、OUTSIDE_IMAGE 或缺失记录不绘制位置，也不沿用上一帧位置。局部纹理仅为当前可见候选的原图 129×129 裁剪，边缘不足处由 Pillow 补黑。</p>
<h2>慢速过程</h2><img src="motion.gif"><h2>抽取最多八帧与原始局部纹理</h2><img src="contact_sheet.png">
<h2>全部点概览</h2><p>同一组抽样帧；仅标记满足绘制条件的候选 ID。</p><img src="overview.png">
<h2>全部帧浏览</h2><input id="slider" type="range" min="0" value="0"><p id="label"></p><a id="full"><img id="frame"></a>
<h2>各点原始状态</h2><table><tr><th>解码帧</th><th>ID</th><th>状态</th><th>可见概率</th><th>确定度</th><th>supported</th></tr>''' + table + '''</table>
<h2>输入与模型记录</h2><pre>''' + esc(json.dumps({'seed_id_0': seed, 'queries': queries, 'status': tracks.get('status'), 'model': tracks.get('model'), 'metrics': tracks.get('metrics')}, ensure_ascii=False, indent=2)) + '''</pre>
<script>const data=''' + data + ''';const slider=document.getElementById('slider');slider.max=data.paths.length-1;function show(){const i=+slider.value;document.getElementById('frame').src=data.paths[i];document.getElementById('full').href=data.paths[i];document.getElementById('label').textContent=(i+1)+' / '+data.paths.length+' · raw '+data.labels[i]}slider.oninput=show;show();</script></html>'''
    (out/'index.html').write_text(page, encoding='utf-8')
    summary = {'schema': 'learned-point-report-v1', 'quality': 'NOT_EVALUATED',
               'physical_identity': 'UNVERIFIED', 'primary_track_id': 0,
               'frames': len(sources), 'sample_positions': chosen, 'frames_rendered': ledger}
    (out/'LEARNED_POINT_REPORT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
