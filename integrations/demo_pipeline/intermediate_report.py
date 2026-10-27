"""Inspectable raw-frame evidence for the intermediate LK diagnosis only."""
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw


def _esc(value):
    return html.escape(str(value), quote=True)


def _link(path, output):
    return quote(os.path.relpath(Path(path).resolve(), output.resolve()).replace(os.sep, '/'), safe='/.:')


def _canvas(path, decoded, edge):
    """Four columns, full aspect-ratio frames; never crop the sequence."""
    cw, ch = 404, 570
    sheet = Image.new('RGB', (cw * 4, ch * math.ceil(len(path) / 4)), '#151a22')
    draw = ImageDraw.Draw(sheet)
    positions = {}
    for slot, index in enumerate(path):
        frame = decoded[index]
        with Image.open(frame['path']) as source:
            picture = source.convert('RGB')
        width, height = picture.size
        picture.thumbnail((384, 512))
        x = (slot % 4) * cw + (cw - picture.width) // 2
        y = (slot // 4) * ch + 48
        sheet.paste(picture, (x, y))
        positions[index] = (x, y, picture.width / width, picture.height / height)
        endpoint = (' source input=' + str(edge['source_input_index'])) if index == path[0] else (' target input=' + str(edge['target_input_index'])) if index == path[-1] else ''
        draw.text(((slot % 4) * cw + 10, (slot // 4) * ch + 6), f'decode={index}{endpoint}', fill='white')
        draw.text(((slot % 4) * cw + 10, (slot // 4) * ch + 20), f"PTS={frame.get('pts')} time_base={frame.get('time_base')}", fill='white')
        unknown = frame.get('mask') is None
        draw.text(((slot % 4) * cw + 10, (slot // 4) * ch + 34), 'UNKNOWN - no intermediate SAM mask' if unknown else str(frame.get('original_mask_status')), fill='orange' if unknown else '#a9c9ec')
    return sheet, positions


def _mark(draw, positions, index, raw, number, color, failed=False, direct=False):
    if index not in positions or not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return
    x, y, sx, sy = positions[index]
    # Raw coordinates are pixel centers, so image-display origin is raw - .5.
    x += (raw[0] - .5) * sx
    y += (raw[1] - .5) * sy
    if failed:
        for a, b in [(-7, -3), (2, 6)]:
            draw.line((x+a, y-7, x+b, y-7), fill='red', width=2)
            draw.line((x+a, y+7, x+b, y+7), fill='red', width=2)
        draw.text((x+8, y+5), f'{number} candidate failed', fill='red')
    elif direct:
        draw.rectangle((x-6, y-6, x+6, y+6), outline=color, width=2)
        draw.text((x+7, y-12), f'D{number}', fill=color)
    else:
        draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
        draw.text((x+5, y-5), str(number), fill=color)


def _point_crop(path, decoded, row, number):
    """Two arms at identical 1:1 zoom; missing measurements stay blank."""
    sheet = Image.new('RGB', (112 * len(path), 292), '#555555')
    draw = ImageDraw.Draw(sheet)
    for arm_number, name in enumerate(('dense', 'direct')):
        arm = row.get(name, {})
        observations = {r['decode_index']: (r['raw_xy'], False) for r in arm.get('observed_raw_coordinates', [])}
        for step in arm.get('steps', []):
            record = step.get('forward', {})
            if record.get('observed_raw_xy') is not None:
                observations.setdefault(record['to_decode_index'], (record['observed_raw_xy'], True))
        color = '#44ed79' if arm.get('status') == 'PASSED' else '#ffd45c'
        for slot, index in enumerate(path):
            x, y = slot * 112, arm_number * 146
            draw.text((x+3,y+2), f'{name} #{number}', fill=color)
            draw.text((x+3,y+15), f'decode={index}', fill='white')
            draw.text((x+3,y+28), f"gid={ascii(str(row.get('group_id')))[1:-1]}", fill='white')
            if index not in observations:
                draw.text((x+3,y+83), 'NOT_REACHED', fill='white')
                continue
            raw, candidate = observations[index]
            # A fixed 96x96 source-pixel window, with gray outside frame bounds.
            cx, cy = raw[0]-.5, raw[1]-.5
            left, top = math.floor(cx)-48, math.floor(cy)-48
            patch = Image.new('RGB',(96,96),'#555555')
            with Image.open(decoded[index]['path']) as original:
                original = original.convert('L').convert('RGB')
                box = (max(0,left),max(0,top),min(original.width,left+96),min(original.height,top+96))
                if box[2]>box[0] and box[3]>box[1]:
                    patch.paste(original.crop(box),(box[0]-left,box[1]-top))
            sheet.paste(patch,(x+8,y+45))
            px, py = x+8+cx-left, y+45+cy-top
            draw.line((px-5,py,px+5,py),fill='red' if candidate else color,width=1)
            draw.line((px,py-5,px,py+5),fill='red' if candidate else color,width=1)
            if candidate:
                draw.text((x+9,y+46),'candidate failed',fill='red')
    return sheet


def render_report(output, result, decoded, budget=lambda: None):
    """Write an HTML report and raw sequence/tracking contact sheets.

    Existing source images are linked in full resolution, never re-described as
    original-resolution video when the extraction receipt says otherwise.
    """
    output = Path(output)
    artifacts = []
    sections = []
    plans = result.get('plan', {}).get('edges', [])
    comparisons = result.get('edges', [])
    for edge_number, edge in enumerate(plans):
        budget()
        comparison = comparisons[edge_number] if edge_number < len(comparisons) else {}
        path = edge['decode_path']
        sequence, positions = _canvas(path, decoded, edge)
        sequence_path = output / f'sequence_{edge_number:03d}.png'
        sequence.save(sequence_path)
        artifacts.append(sequence_path)
        tracking = sequence.copy()
        draw = ImageDraw.Draw(tracking)
        rows = comparison.get('rows', [])
        table = []
        crop_links = []
        crop_counts = {'RECOVERED': 0, 'BOTH_FAIL': 0}
        for number, row in enumerate(rows, 1):
            budget()
            outcome = row.get('outcome')
            crop_link = ''
            if outcome in crop_counts and crop_counts[outcome] < 2:
                crop_counts[outcome] += 1
                crop_path = output / f'crop_{edge_number:03d}_{number:03d}.png'
                _point_crop(path, decoded, row, number).save(crop_path)
                artifacts.append(crop_path)
                crop_link = f'<a href="{crop_path.name}">点 {number} / {_esc(row.get("group_id"))} 局部序列</a>'
                crop_links.append(f'<p>{crop_link} ({_esc(outcome)})</p><img src="{crop_path.name}">')
            for name in ('dense', 'direct'):
                arm = row.get(name, {})
                color = '#44ed79' if arm.get('status') == 'PASSED' else '#ffd45c'
                for observation in arm.get('observed_raw_coordinates', []):
                    if name == 'direct' and observation['decode_index'] != path[-1]:
                        continue
                    _mark(draw, positions, observation['decode_index'], observation['raw_xy'], number, color, direct=name == 'direct')
                if arm.get('status') != 'PASSED':
                    steps = arm.get('steps', [])
                    cycle_steps = arm.get('fullcycle', {}).get('steps', [])
                    # The failing measurement can be a backward or cycle leg;
                    # do not silently substitute the final target observation.
                    candidates = cycle_steps[-1:] if cycle_steps else ([steps[-1].get('backward', {}), steps[-1].get('forward', {})] if steps else [])
                    for candidate in candidates:
                        if candidate.get('observed_raw_xy') is not None:
                            _mark(draw, positions, candidate.get('to_decode_index'), candidate['observed_raw_xy'], number, 'red', failed=True)
                            break
            details = _esc(json.dumps({name: row.get(name, {}) for name in ('direct', 'dense')}, ensure_ascii=False, indent=2))
            table.append(f'<tr id="edge-{edge_number}-point-{number}"><td>{number}</td><td>{_esc(row.get("group_id"))}</td><td>{_esc(row.get("outcome", "INCOMPLETE"))}</td>' + ''.join(f'<td>{_esc(row.get(name, {}).get("status", "NOT_RUN"))}<br>{_esc(row.get(name, {}).get("reasons", []))}<br>fullcycle: {_esc({k: row.get(name, {}).get("fullcycle", {}).get(k) for k in ("status", "error_px")})}</td>' for name in ('direct','dense')) + f'<td>{crop_link}<details><summary>精确坐标与逐步证据</summary><pre>{details}</pre></details></td></tr>')
        tracking_path = output / f'tracking_{edge_number:03d}.png'
        tracking.save(tracking_path)
        artifacts.append(tracking_path)
        counts = {key: sum(row.get('outcome') == key for row in rows) for key in ('RECOVERED','LOST','BOTH_PASS','BOTH_FAIL')}
        counts.update(direct_passed=sum(row.get('direct', {}).get('status') == 'PASSED' for row in rows), dense_passed=sum(row.get('dense', {}).get('status') == 'PASSED' for row in rows), incomplete=sum('outcome' not in row for row in rows))
        links = ' | '.join(f'<a href="{_esc(_link(decoded[i]["path"], output))}">decode {i}: full stored image ({decoded[i]["width"]}x{decoded[i]["height"]})</a>' for i in path)
        legend = ', '.join(f'<a href="#edge-{edge_number}-point-{n}">{n} = {_esc(row.get("group_id"))}</a>' for n,row in enumerate(rows,1))
        sections.append(f'<section><h2>帧对 {edge_number}: {_esc(edge["model_key"])}, input {edge["source_input_index"]} → {edge["target_input_index"]}</h2><p>{_esc(counts)}; flow calls: {_esc(comparison.get("flow_calls", "NOT_RUN"))}</p><p>{links}</p><h3>原始完整帧序列</h3><a href="{sequence_path.name}"><img src="{sequence_path.name}"></a><h3>点跟踪观测</h3><p>绿色：该分支 PASSED；黄色：REJECTED 分支已有观测；红色虚线：失败候选。圆圈为 dense，方框 D 为 direct 终点。编号对应下表 group ID；重叠时以精确坐标为准。</p><p>{legend}</p><a href="{tracking_path.name}"><img src="{tracking_path.name}"></a><h3>逐点局部序列（固定 96 像素窗口，1:1）</h3><p>每类按原顺序展示前两点。灰色 NOT_REACHED 表示没有观测坐标，不推测位置；红色候选未通过检查。图中十字为实际测量中心。</p>{"".join(crop_links)}<table><tr><th>编号</th><th>Group ID</th><th>结果</th><th>Direct</th><th>Dense</th><th>证据</th></tr>{"".join(table)}</table></section>')
    budget()
    page = output / 'index.html'
    page.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>中间帧跟踪诊断</title><style>body{font:15px sans-serif;background:#111820;color:#eee;margin:24px}a{color:#8ecaff}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #667;padding:8px;vertical-align:top}pre{white-space:pre-wrap;max-width:700px}section{margin:40px 0}</style><h1>中间帧跟踪诊断</h1><p>Status: ' + _esc(result.get('status')) + '</p><p>DIAGNOSIS ONLY：仅用于诊断。中间帧掩码为 UNKNOWN / NOT SEGMENTED，未运行中间帧 SAM。LK 数值通过不能验证物体身份，也不构成三维/CAD 验收。完整帧序列保留存储图像全部画幅；链接打开完整存储分辨率图像，提取缩放情况见回执。</p><details><summary>参数与提取回执</summary><pre>' + _esc(json.dumps({'profile':result.get('profile'), 'extraction':result.get('extraction')},ensure_ascii=False,indent=2)) + '</pre></details>' + ''.join(sections) + '</html>', encoding='utf-8')
    artifacts.append(page)
    return artifacts
