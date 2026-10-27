"""Actual-frame regional 2D tracking evidence, without inferred point positions."""
import html
import math
import os
from pathlib import Path
from statistics import median
from urllib.parse import quote

from PIL import Image, ImageDraw

GREEN = '#35d879'
RED = '#ff5b62'
GRAY = '#9ba4af'
ORANGE = '#ffad45'


def _escape(value):
    return html.escape(str(value), quote=True)


def _ascii(value):
    return str(value).encode('ascii', 'backslashreplace').decode('ascii')


def _xy(value):
    return (type(value) in (tuple, list) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


def _excluded(point):
    return str(point.get('status', '')).startswith('EXCLUDED_SOURCE_')


def _continuous(point):
    return not _excluded(point) and point.get('verified_legs', 0) >= 2


def _color(point):
    return GRAY if _excluded(point) else GREEN if _continuous(point) else RED


def _observations(point):
    """Only measured coordinates; never fill from center displacement or steps."""
    return {o['decode_index']: o['raw_xy'] for o in point.get('observations', [])
            if type(o.get('decode_index')) is int and _xy(o.get('raw_xy'))}


def _link(path, output):
    return quote(os.path.relpath(Path(path).resolve(), output.resolve()).replace(os.sep, '/'), safe='/.:')


def _mark(draw, xy, point, radius=3):
    x, y = xy
    color = _color(point)
    if point.get('is_center') is True:
        draw.polygon(((x, y-radius-2), (x+radius+2, y), (x, y+radius+2), (x-radius-2, y)), fill=color, outline='white')
    elif _excluded(point):
        draw.rectangle((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
    else:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)


def _open_frame(frame):
    with Image.open(frame['path']) as source:
        image = source.convert('RGB')
    if (frame.get('width'), frame.get('height')) != image.size:
        raise ValueError('REGION_REPORT_DECODED_IMAGE_SIZE_MISMATCH')
    return image


def _summary(points):
    excluded = sum(_excluded(p) for p in points)
    continuous = sum(_continuous(p) for p in points)
    return dict(total=len(points), excluded=excluded, evaluated=len(points)-excluded,
                tracklets_2d=continuous, tracking_rejected=len(points)-excluded-continuous)


def _point_errors(point):
    fb, lk, ncc = [], [], []
    for step in point.get('steps', []):
        value = step.get('fb_error_px')
        if type(value) in (int, float) and math.isfinite(value):
            fb.append(value)
        for direction in ('forward', 'backward'):
            value = (step.get(direction) or {}).get('error')
            if type(value) in (int, float) and math.isfinite(value):
                lk.append(value)
        value = (step.get('ncc') or {}).get('value')
        if type(value) in (int, float) and math.isfinite(value):
            ncc.append(value)
    return (max(fb) if fb else None, max(lk) if lk else None, min(ncc) if ncc else None)


def _panel(region, indices, decoded, budget):
    points = region.get('points', [])
    cellw, cellh = 340, 400
    sheet = Image.new('RGB', (max(600, cellw*len(indices)), 720), '#151d29')
    draw = ImageDraw.Draw(sheet)
    source_index = region.get('source_decode_index')
    source_frame = decoded.get(source_index)
    bbox = region.get('source_bbox_raw_xy')
    if source_frame and source_frame.get('path') and Path(source_frame['path']).is_file():
        budget()
        source = _open_frame(source_frame)
        original_width, original_height = source.size
        source.thumbnail((440, 270))
        sheet.paste(source, (8, 30))
        sx, sy = source.width/original_width, source.height/original_height
        if type(bbox) in (list, tuple) and len(bbox) == 4:
            draw.rectangle((8+(bbox[0]-.5)*sx, 30+(bbox[1]-.5)*sy,
                            8+(bbox[2]-.5)*sx, 30+(bbox[3]-.5)*sy), outline=ORANGE, width=3)
        draw.text((8, 8), f'source decode={source_index}; orange=initial 32px region', fill='white')
    else:
        draw.text((8, 8), 'Source frame unavailable; no replacement image', fill='white')
    draw.text((460, 36), _ascii(region.get('region_id', 'UNKNOWN')), fill='white')
    draw.text((460, 57), 'diamond=center; circles=grid', fill='white')
    draw.text((460, 78), 'green: >=2 verified legs; red: short/rejected', fill='white')
    draw.text((460, 99), 'gray square: excluded by source admission', fill='white')
    draw.text((460, 120), 'Lines use adjacent ACTUAL raw observations only.', fill='white')
    draw.text((460, 141), 'Crops are display windows, not propagated surfaces.', fill='white')
    draw.text((460, 162), _ascii('depth='+str(region.get('depth', 0))+' parent='+str(region.get('parent_region_id'))), fill='white')
    all_observations = {p['point_id']: _observations(p) for p in points}
    decode_path = region.get('decode_path', [])
    for slot, index in enumerate(indices):
        budget()
        frame = decoded[index]
        picture = _open_frame(frame)
        coordinates = [all_observations[p['point_id']][index] for p in points if index in all_observations[p['point_id']]]
        if index == source_index:
            coordinates += [p['source_raw_xy'] for p in points if _excluded(p) and _xy(p.get('source_raw_xy'))]
        center = region.get('center_raw_xy')
        if coordinates:
            cx, cy = median(p[0] for p in coordinates), median(p[1] for p in coordinates)
            side = max(64., max(p[0] for p in coordinates)-min(p[0] for p in coordinates)+16,
                       max(p[1] for p in coordinates)-min(p[1] for p in coordinates)+16)
        else:
            cx, cy = center if _xy(center) else (picture.width/2, picture.height/2)
            side = 64.
        left, top = max(0, math.floor(cx-.5-side/2)), max(0, math.floor(cy-.5-side/2))
        right, bottom = min(picture.width, math.ceil(cx-.5+side/2)), min(picture.height, math.ceil(cy-.5+side/2))
        if right <= left or bottom <= top:
            raise ValueError('REGION_REPORT_CROP_OUTSIDE_IMAGE')
        crop = picture.crop((left, top, right, bottom))
        crop.thumbnail((320, 300))
        # Enlarge actual small crops; no generated/interpolated coordinate data.
        scale = min(320/(right-left), 300/(bottom-top))
        crop = picture.crop((left, top, right, bottom)).resize(
            (max(1, round((right-left)*scale)), max(1, round((bottom-top)*scale))), Image.Resampling.NEAREST)
        overlay = ImageDraw.Draw(crop)
        sx, sy = crop.width/(right-left), crop.height/(bottom-top)
        def pos(raw):
            return (raw[0]-.5-left)*sx, (raw[1]-.5-top)*sy
        if index == source_index and type(bbox) in (list, tuple) and len(bbox) == 4:
            overlay.rectangle((*pos(bbox[:2]), *pos(bbox[2:])), outline=ORANGE, width=2)
        previous = None
        if index in decode_path:
            ordinal = decode_path.index(index)
            if ordinal:
                previous = decode_path[ordinal-1]
        drawn = 0
        for point in points:
            budget()
            obs = all_observations[point['point_id']]
            raw = obs.get(index)
            if raw is None:
                # Gray source samples are explicitly not target observations.
                if index == source_index and _excluded(point) and _xy(point.get('source_raw_xy')):
                    _mark(overlay, pos(point['source_raw_xy']), point)
                continue
            if previous in obs and not _excluded(point):
                overlay.line((*pos(obs[previous]), *pos(raw)), fill=_color(point), width=1)
            _mark(overlay, pos(raw), point)
            drawn += 1
        x, y = slot*cellw+8, 375
        sheet.paste(crop, (x, y))
        draw.text((x, 322), f'decode={index} t={_ascii(frame.get("time_seconds", "UNKNOWN"))}', fill='white')
        draw.text((x, 340), 'SAM=' + _ascii(frame.get('original_mask_status') or 'UNKNOWN'), fill=ORANGE)
        draw.text((x, 358), f'measured points={drawn}; crop=({left},{top},{right},{bottom})', fill='white')
        draw.text((x, 686), 'No missing-point interpolation / no 3D claim', fill='white')
    return sheet


def render_regions(output, result, decoded, *, budget=lambda: None):
    """Create new panel PNGs and Chinese HTML from measured region records."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    budget()
    regions = result.get('regions', [])
    planned = []
    for ordinal, region in enumerate(regions):
        budget()
        indices = list(dict.fromkeys(region.get('decode_path', [])))[:5]
        available = [i for i in indices if i in decoded and decoded[i].get('path') and Path(decoded[i]['path']).is_file()]
        planned.append((region, available, f'region_{ordinal:03d}.png' if available else None))
    for name in ['index.html'] + [name for _, _, name in planned if name]:
        if (output/name).exists():
            raise FileExistsError(output/name)
    files, sections = [], []
    for region, indices, name in planned:
        budget()
        if name:
            panel = _panel(region, indices, decoded, budget)
            budget()
            with (output/name).open('xb') as stream:
                panel.save(stream, format='PNG')
            files.append(output/name)
        points = region.get('points', [])
        center = [p for p in points if p.get('is_center') is True]
        grid = [p for p in points if p.get('is_center') is not True]
        center_summary, grid_summary = _summary(center), _summary(grid)
        occupied = len({p.get('grid_ordinal') for p in grid if _continuous(p)})
        rows = []
        for label, summary in [('中央点基线', center_summary), ('新增网格点（不含中心）', grid_summary)]:
            rows.append('<tr><th>'+label+'</th>'+''.join('<td>'+_escape(summary[k])+'</td>' for k in
                ('total','excluded','evaluated','tracklets_2d','tracking_rejected'))+'</tr>')
        frame_links = []
        for index in region.get('decode_path', []):
            frame = decoded.get(index) or {}
            source_link = ('<a href="'+_escape(_link(frame['path'], output))+'">实际帧</a>') if frame.get('path') and Path(frame['path']).is_file() else '实际帧缺失'
            frame_links.append(f'<li>decode {_escape(index)}；原图时间 {_escape(frame.get("time_seconds", "UNKNOWN"))} s；SAM {_escape(frame.get("original_mask_status") or "UNKNOWN")}；{source_link}</li>')
        details = []
        for point in points:
            budget()
            fb, lk, ncc = _point_errors(point)
            observations = '; '.join(f'{o.get("decode_index")}: {o.get("raw_xy")}' for o in point.get('observations', []))
            values = [point.get('point_id'), '中心菱形' if point.get('is_center') is True else '网格圆点',
                      point.get('status'), point.get('verified_legs', 0), fb, lk, ncc, point.get('reason'), observations]
            details.append('<tr>'+''.join('<td>'+_escape(v)+'</td>' for v in values)+'</tr>')
        affine = []
        for leg in region.get('affine_diagnostics', []):
            held = leg.get('holdout') or {}
            values = [f'{leg.get("from_decode_index")} → {leg.get("to_decode_index")}', leg.get('status'),
                      len(leg.get('train_ids', [])), len(leg.get('holdout_ids', [])),
                      held.get('median_px'), held.get('max_px'), leg.get('reason')]
            affine.append('<tr>'+''.join('<td>'+_escape(v)+'</td>' for v in values)+'</tr>')
        image = f'<img src="{name}" alt="原区域位置和逐帧实测局部轨迹">' if name else '<p>实际帧不可用，未生成占位图。</p>'
        sections.append('<section><h2>'+_escape(region.get('region_id', 'UNKNOWN'))+'</h2><p>层 depth '+_escape(region.get('depth', 0))
            +'；父区域 '+_escape(region.get('parent_region_id'))+'；根短轨迹 '+_escape(region.get('root_tracklet_id', region.get('tracklet_id')))
            +'；source_input_index '+_escape(region.get('source_input_index'))+'</p><p>原短轨迹 '+_escape(region.get('tracklet_id'))
            +'；源区域 raw bbox '+_escape(region.get('source_bbox_raw_xy'))+'</p>'+image
            +'<table><tr><th>2D方案</th><th>候选点</th><th>源域排除</th><th>实际评估</th><th>连续≥2腿</th><th>未形成连续轨迹</th></tr>'+''.join(rows)+'</table>'
            +f'<p>新增网格2D采样格点占据：{occupied}/{len(grid)}。源域排除不计作跟踪失败；占据增加不等于面积/精度已提升。</p>'
            +'<ul>'+''.join(frame_links)+'</ul><details><summary>逐点实测坐标、停止原因与误差</summary><table><tr>'
            +''.join('<th>'+v+'</th>' for v in ('点ID','形状','状态','已验证腿数','最大FB像素误差','最大LK误差','最小NCC','原因','实际观测 decode: raw_xy'))
            +'</tr>'+''.join(details)+'</table></details><details><summary>区域2D仿射留出诊断（不等于真实形变）</summary><table><tr>'
            +''.join('<th>'+v+'</th>' for v in ('帧对','状态','train','holdout','留出median px','留出max px','原因'))
            +'</tr>'+''.join(affine)+'</table></details></section>')
    baseline = _summary([p for r in regions for p in r.get('points', []) if p.get('is_center') is True])
    expanded = _summary([p for r in regions for p in r.get('points', []) if p.get('is_center') is not True])
    layers = []
    for depth in sorted({r.get('depth', 0) for r in regions}):
        layer = [r for r in regions if r.get('depth', 0) == depth]
        centers = _summary([p for r in layer for p in r.get('points', []) if p.get('is_center') is True])
        grids = _summary([p for r in layer for p in r.get('points', []) if p.get('is_center') is not True])
        values = [depth, len(layer), len({r.get('source_input_index') for r in layer}),
                  centers['evaluated'], centers['tracklets_2d'], grids['evaluated'], grids['tracklets_2d'], grids['excluded']]
        layers.append('<tr>'+''.join('<td>'+_escape(v)+'</td>' for v in values)+'</tr>')
    layer_table = '<table><tr>'+''.join('<th>'+v+'</th>' for v in ('depth层','区域数','源帧数','中心评估','中心2D持续','网格评估','网格2D持续','源域排除'))+'</tr>'+''.join(layers)+'</table>'
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>点扩成区域：实际局部跟踪</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#131b26;color:#e7edf5;line-height:1.6}a{color:#8ac9ff}img{max-width:100%;display:block}table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #586577;padding:5px;overflow-wrap:anywhere}section{border-top:1px solid #586577;margin-top:24px}details{margin:12px 0}</style>'
        '<h1>点扩成32px区域：实际局部跟踪对照</h1><p>中央点是基线；新增网格不含中央点，每个点分别使用实际测量，未复制中心位移、未补出缺失点。'
        '每区域最多展示前5帧，全部实际观测保留在逐点表。橙框仅标源帧初始32px区域；后续窗口按实际观测取景，不是传播的物体表面。'
        '连线是相邻帧实际 raw 坐标的2D位移，在同一像素坐标轴上显示。</p>'
        '<p>中心为菱形，网格点为圆；绿色为已连续验证至少2腿的前缀，红色为短前缀/未通过，灰方框为源域排除样点。后续停止不抹去绿色已测前缀。'
        '未知SAM不冒充前景。FB/LK/NCC和区域仿射误差仅为共享图像上的2D一致性，不是独立真值、3D精度或CAD验收。'
        '同一区域及重叠子区域共享图像patch，样点与NCC/仿射诊断存在相关性。递归计数是区域×样点记录，可能重复采样，不是独立样本数或去重覆盖面积。'
        '不同source_input_index处于不同图像坐标系，不能合并为同一2D面积，更不能称3D覆盖。</p>'
        +f'<p>中心2D连续点 {baseline["tracklets_2d"]}/{baseline["evaluated"]} 实际评估（另排除 {baseline["excluded"]}）；新增网格2D连续点 {expanded["tracklets_2d"]}/{expanded["evaluated"]} 实际评估（另排除 {expanded["excluded"]}）。'
        '数量不同，不能只用成功点总数判断局部跟踪准确率提高；还需比较成功率、空间占据与误差。</p>'
        +'<h2>逐层2D摘要</h2>'+layer_table
        +(''.join(sections) if sections else '<p>没有区域记录；未生成占位图。</p>')+'</html>')
    budget()
    target = output/'index.html'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(page)
    files.append(target)
    return files
