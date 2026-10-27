"""Read-only presentation of saved strict and neighborhood evidence."""
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


def _escape(value):
    return html.escape(str(value), quote=True)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def _points(value):
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(not isinstance(p, (list, tuple)) or len(p) != 2
                   or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)
                   for p in value)):
        raise ValueError('NEIGHBOR_REPORT_INVALID_POLYGON')
    return value


def _predict(matrix, polygon):
    if matrix is None or polygon is None:
        return None
    if (not isinstance(matrix, (list, tuple)) or len(matrix) != 3
            or any(not isinstance(row, (list, tuple)) or len(row) != 3
                   or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
                   for row in matrix)):
        raise ValueError('NEIGHBOR_REPORT_INVALID_MATRIX')
    result = []
    for x, y in polygon:
        q = [row[0]*x+row[1]*y+row[2] for row in matrix]
        if not all(math.isfinite(v) for v in q) or abs(q[2]) <= 1e-12:
            raise ValueError('NEIGHBOR_REPORT_INVALID_PROJECTION')
        result.append([q[0]/q[2], q[1]/q[2]])
    return result


def _polygon(points, color, *, prediction=False):
    if points is None:
        return ''
    coords = ' '.join(f'{x},{y}' for x, y in points)
    dash = ' stroke-dasharray="8 5"' if prediction else ''
    kind = 'neighborhood-prediction' if prediction else 'saved-measurement'
    return (f'<polygon data-kind="{kind}" points="{coords}" fill="none" '
            f'stroke="{color}" stroke-width="2.5" vector-effect="non-scaling-stroke"{dash}/>')


def _side(frame, output, polygon, *, target=False, prediction=None):
    path = frame.get('path')
    available = bool(path and Path(path).is_file())
    width, height = frame.get('width'), frame.get('height')
    valid_size = all(type(v) is int and v > 0 for v in (width, height))
    if not available or not valid_size:
        return '<text x="20" y="100" fill="white">实际图像或尺寸缺失：不画图像和区域</text>'
    href = quote(os.path.relpath(Path(path).resolve(), output.resolve()).replace(os.sep, '/'), safe='/.:')
    # SVG image edges are 0..width/height; raw pixel centers already include +0.5.
    return (f'<svg x="0" y="45" width="590" height="380" viewBox="0 0 {width} {height}">'
            f'<image href="{_escape(href)}" x="0" y="0" width="{width}" height="{height}"/>'
            + _polygon(polygon, '#ffad45' if target else '#ffffff')
            + (_polygon(prediction, '#c594ff', prediction=True) if target else '') + '</svg>')


def _evidence(record, decoded, output):
    measurement = record.get('measurement') or {}
    neighborhood = record.get('neighborhood') or {}
    source = _points(measurement.get('source_polygon_raw4x2'))
    target = _points(measurement.get('target_polygon_raw4x2'))
    predicted = _predict(neighborhood.get('matrix_global_raw3x3'), source)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500" viewBox="0 0 1200 500">',
             '<rect width="1200" height="500" fill="#16202d"/>',
             '<g font-family="sans-serif" font-size="16" fill="white">',
             '<text x="12" y="24">源图：白色为保存的源区域</text>',
             '<text x="612" y="24">目标图：橙色为实际 forward 拟合（不代表通过）</text>',
             _side(decoded.get(record.get('source_decode_index')) or {}, output, source),
             '<g transform="translate(600 0)">',
             _side(decoded.get(record.get('target_decode_index')) or {}, output, target,
                   target=True, prediction=predicted), '</g>',
             '<text x="12" y="450">紫色虚线：邻域预测，仅用于比较，不是新测量；无矩阵则不画预测。</text>',
             '<text x="12" y="480">'+_escape(record.get('region_id', 'UNKNOWN'))+'</text>', '</g></svg>']
    return ''.join(parts)


def render_neighborhood(output, result, decoded, *, budget=lambda: None):
    """Render saved dictionaries without registration, extrapolation or mutation."""
    budget()
    output = Path(output)
    records = result.get('records') or []
    eligible = []
    for ordinal, record in enumerate(records):
        budget()
        if record.get('measurement') is not None:
            status = (record.get('neighborhood') or {}).get('status')
            eligible.append((status != 'NEIGHBOR_CONSISTENT_CANDIDATE', ordinal, record))
    selected = sorted(eligible, key=lambda item: (item[0], item[1]))[:8]
    evidence = {}
    artifacts = []
    for number, (_, ordinal, record) in enumerate(selected):
        budget()
        name = f'evidence_{number:03d}.svg'
        evidence[ordinal] = name
        artifacts.append((output/name, _evidence(record, decoded, output)))
    rows = []
    for ordinal, record in enumerate(records):
        budget()
        neighborhood = record.get('neighborhood') or {}
        continuation = record.get('continuation') or {}
        values = [record.get(k, 'UNKNOWN') for k in ('region_id', 'source_input_index', 'source_decode_index', 'target_decode_index')]
        values += [record.get('strict_status', 'UNKNOWN'), _json(record.get('strict_reasons')),
                   record.get('cycle_error_px'), record.get('eligibility_reason'),
                   neighborhood.get('status', 'UNKNOWN'), _json(neighborhood.get('reasons')),
                   continuation.get('status', 'UNKNOWN')]
        link = ('<a href="'+evidence[ordinal]+'">实际图证据</a>') if ordinal in evidence else '未生成证据图'
        rows.append('<tr>'+''.join('<td>'+_escape(v)+'</td>' for v in values)
                    + '<td>'+link+'<details><summary>完整原始记录</summary><pre>'
                    + _escape(_json(record))+'</pre></details></td></tr>')
    headings = ['区域', '源输入', '源decode', '目标decode', '原strict状态', '原strict失败理由',
                '原cycle误差px', '资格理由', '邻域状态', '邻域理由', '续跟踪状态', '证据/原始记录']
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>邻域一致性只读报告</title>'
            '<style>body{font:16px/1.6 system-ui;background:#16202d;color:#e7edf5;margin:24px}a{color:#90d0ff}table{border-collapse:collapse}td,th{border:1px solid #607080;padding:6px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            '<h1>邻域比较与实际测量</h1><p>原strict结果未改。候选不等于三维验收或物体身份确认；重叠纹理不构成独立证据。geometry_accepted=false。</p>'
            '<p>仅显示保存的实际 forward 拟合；邻域预测仅用于比较，不是新测量。未测、缺失与待定记录全部保留，不补轨迹。缺图时明确不画。</p>'
            '<p>最多8张证据图，优先邻域一致候选，各组保持原记录次序；表格保留全部记录。</p>'
            '<details><summary>配置与计数</summary><pre>'+_escape(_json(dict(profile=result.get('profile'), counts=result.get('counts'))))
            + '</pre></details><table><tr>'+''.join('<th>'+_escape(h)+'</th>' for h in headings)
            + '</tr>'+''.join(rows)+'</table></html>')
    artifacts.append((output/'index.html', page))
    for path, _ in artifacts:
        budget()
        if path.exists():
            raise FileExistsError(path)
    output.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts:
        budget()
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
    return [path for path, _ in artifacts]
