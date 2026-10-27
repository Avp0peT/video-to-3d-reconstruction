"""Layered source-image SAM footprints and measured overlap-edge diagnostics."""
import base64
import html
from io import BytesIO
import math
import os
from pathlib import Path
from urllib.parse import quote


def _escape(value):
    return html.escape(str(value), quote=True)


def _link(path, output):
    return quote(os.path.relpath(Path(path).resolve(), output.resolve()).replace(os.sep, '/'), safe='/.:')


def _number(value):
    return '未知' if value is None else str(value)


def _roi(value, width, height):
    if type(value) is not dict or set(value) != {'left', 'top', 'width', 'height'} or any(type(v) is not int for v in value.values()):
        raise ValueError('OVERLAP_REPORT_INTEGER_ROI_REQUIRED')
    x, y, w, h = (value[k] for k in ('left', 'top', 'width', 'height'))
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
        raise ValueError('OVERLAP_REPORT_ROI_OUTSIDE_SOURCE')
    return x, y, x+w, y+h


def _source_svg(source, candidates, plan, result, decoded, budget):
    from PIL import Image, ImageDraw, ImageFilter, ImageChops
    index = source['source_decode_index']
    frame = decoded.get(index)
    if not frame or not frame.get('path') or not Path(frame['path']).is_file():
        raise ValueError('OVERLAP_REPORT_SOURCE_FRAME_UNAVAILABLE')
    budget()
    raw_bytes = Path(frame['path']).read_bytes()
    with Image.open(BytesIO(raw_bytes)) as original:
        width, height = original.size
        mime = Image.MIME.get(original.format)
    if type(frame.get('width')) is not int or type(frame.get('height')) is not int or (width, height) != (frame['width'], frame['height']):
        raise ValueError('OVERLAP_REPORT_SOURCE_DIMENSIONS_MISMATCH')
    if not mime:
        raise ValueError('OVERLAP_REPORT_UNSUPPORTED_IMAGE_FORMAT')
    domain = plan['frame_domains'][str(index)]
    if (domain.get('width'), domain.get('height')) != (width, height):
        raise ValueError('OVERLAP_REPORT_MASK_DIMENSIONS_MISMATCH')
    rle = domain.get('mask_rle')
    if domain.get('original_mask_status') != 'OK' or type(rle) is not list or len(rle) != height:
        raise ValueError('OVERLAP_REPORT_ORIGINAL_SAM_MASK_REQUIRED')
    mask = Image.new('L', (width, height))
    mask_draw = ImageDraw.Draw(mask)
    for y, runs in enumerate(rle):
        budget()
        previous = 0
        for interval in runs:
            if type(interval) is not list or len(interval) != 2 or any(type(v) is not int for v in interval):
                raise ValueError('OVERLAP_REPORT_MASK_RLE_INTERVAL')
            a, b = interval
            if not previous <= a < b <= width:
                raise ValueError('OVERLAP_REPORT_MASK_RLE_ORDER')
            mask_draw.line((a, y, b-1, y), fill=255)
            previous = b
    outline = Image.new('RGBA', (width, height), (175, 181, 191, 0))
    outline.putalpha(mask.filter(ImageFilter.FIND_EDGES))
    buffer = BytesIO()
    outline.save(buffer, format='PNG')
    states = {row['region_id']: row for row in result.get('overlap', {}).get('regions', [])}
    selected = {row['region_id']: row['roi'] for row in candidates if row.get('selected') is True and row.get('region_id')}
    actual_regions = {r['region_id']: r for r in result.get('regions', [])}
    for rid in list(selected):
        measured = actual_regions.get(rid)
        if measured and measured.get('template_roi') is not None:
            selected[rid] = measured['template_roi']
    local = {rid for rid in selected if states.get(rid, {}).get('local_tracklet') is True}
    network = {rid for rid in selected if states.get(rid, {}).get('network_tracklet') is True}
    conflicts = {rid for rid in selected if states.get(rid, {}).get('conflict_edge_ids')}
    for edge in result.get('overlap', {}).get('edges', []):
        if any(t.get('status') == 'CONFLICT' for t in edge.get('targets', [])):
            conflicts.update({edge['region_a'], edge['region_b']} & set(selected))
    def rectangle(roi, rid=None, candidate=False):
        if candidate:
            # Frozen candidate ledgers intentionally retain image-boundary
            # exclusions. Show their visible part, never treat them as admitted.
            if type(roi) is not dict or set(roi) != {'left','top','width','height'} or any(type(v) is not int for v in roi.values()):
                raise ValueError('OVERLAP_REPORT_INTEGER_ROI_REQUIRED')
            x,y,w,h=(roi[k] for k in ('left','top','width','height'))
            if min(x,y)<0 or min(w,h)<=0:
                raise ValueError('OVERLAP_REPORT_INVALID_CANDIDATE_ROI')
            right,bottom=x+w,y+h
        else:
            x, y, right, bottom = _roi(roi, width, height)
        title = '<title>'+_escape(rid)+'</title>' if rid else ''
        return f'<rect x="{x}" y="{y}" width="{right-x}" height="{bottom-y}">{title}</rect>'
    groups = ['<g class="layer_candidates" fill="none" stroke="#cbd3df" stroke-width="0.7" opacity="0.25">']
    for item in candidates:
        budget()
        groups.append(rectangle(item['roi'], item.get('candidate_id'), candidate=True))
    groups.append('</g>')
    groups.append('<g class="layer_connections" fill="none" stroke="#acbaff" stroke-width="1" stroke-dasharray="3 2">')
    for edge in plan.get('edges', []):
        if edge.get('source_input_index') != source['source_input_index'] or edge.get('source_decode_index') != index:
            continue
        a, b = edge.get('region_a'), edge.get('region_b')
        if a in selected and b in selected:
            budget()
            centers = []
            for rid in (a, b):
                left, top, right, bottom = _roi(selected[rid], width, height)
                centers.append(((left+right)/2, (top+bottom)/2))
            groups.append(f'<line x1="{centers[0][0]}" y1="{centers[0][1]}" x2="{centers[1][0]}" y2="{centers[1][1]}"><title>'+_escape(edge.get('edge_id'))+'</title></line>')
    groups.append('</g>')
    for name, identities, color, opacity in [('selected', set(selected), '#449cff', '.10'),
                                           ('local', local, '#ffd35a', '.20'),
                                           ('network', network, '#35d879', '.25'),
                                           ('conflicts', conflicts, '#ff5b62', '.30')]:
        groups.append(f'<g class="layer_{name}" fill="{color}" fill-opacity="{opacity}" stroke="{color}" stroke-width="1.5">')
        for rid in sorted(identities):
            budget()
            groups.append(rectangle(selected[rid], rid))
        groups.append('</g>')
    def footprint(identities):
        union = Image.new('L', (width, height))
        draw = ImageDraw.Draw(union)
        for rid in identities:
            budget()
            x, y, right, bottom = _roi(selected[rid], width, height)
            draw.rectangle((x, y, right-1, bottom-1), fill=255)
        return ImageChops.multiply(union, mask).histogram()[255]
    counts = dict(foreground_pixels=mask.histogram()[255], planned_union_pixels=footprint(set(selected)),
                  local_two_target_union_pixels=footprint(local), overlap_two_target_union_pixels=footprint(network))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" overflow="hidden">'
           '<title>实际源图上的SAM前景采样联合区域，不是三维表面</title>'
           f'<image x="0" y="0" width="{width}" height="{height}" href="data:{mime};base64,{base64.b64encode(raw_bytes).decode("ascii")}"/>'
           f'<g class="layer_mask"><image x="0" y="0" width="{width}" height="{height}" href="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode("ascii")}"/></g>'
           +''.join(groups)+'</svg>')
    return svg, counts, local, network, conflicts


def render_overlap(output, plan, result, decoded, *, budget=lambda: None):
    """Create self-contained actual-image SVGs and a layered Chinese report."""
    budget()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = plan.get('sources', [])
    names = [f'overlap_source_{i:03d}.svg' for i in range(len(sources))]
    for name in names+['index.html']:
        if (output/name).exists():
            raise FileExistsError(output/name)
    overlap = result.get('overlap') or {}
    measured = {row['region_id']: row for row in result.get('regions', [])}
    region_links, batch_links = {}, []
    for batch in result.get('batches', []):
        href = batch.get('report')
        if not href:
            continue
        # Links are local paths only; do not turn evidence text into script URLs.
        candidate = (output/href).resolve()
        if not candidate.is_relative_to(output.resolve()):
            raise ValueError('OVERLAP_REPORT_BATCH_LINK_OUTSIDE_OUTPUT')
        url = quote(Path(href).as_posix(), safe='/')
        batch_links.append('<a href="'+_escape(url)+'">批次 '+_escape(batch.get('ordinal'))+' 整体配准详情</a>')
        identities = batch.get('region_ids', batch.get('regions'))
        if type(identities) is list:
            for region in identities:
                rid = region.get('region_id') if type(region) is dict else region
                if type(rid) is str:
                    region_links[rid] = url
    def region_label(rid):
        label = _escape(rid)
        return '<a href="'+_escape(region_links[rid])+'">'+label+'</a>' if rid in region_links else label
    groups = {(c['source_input_index'], c['source_decode_index']): c.get('items', []) for c in plan.get('candidates', [])}
    declared_coverage = {(row['source_input_index'], row['source_decode_index']): row for row in overlap.get('coverage', [])}
    maps, files = [], []
    for source, name in zip(sources, names):
        budget()
        key = source['source_input_index'], source['source_decode_index']
        candidates = groups.get(key, [])
        svg, counts, local, network, conflicts = _source_svg(source, candidates, plan, result, decoded, budget)
        claimed = declared_coverage.get(key, {})
        coverage_rows = []
        for field, label in [('foreground_pixels','SAM原前景'),('planned_union_pixels','蓝色已选区域'),
                             ('local_two_target_union_pixels','黄色局部连续≥2目标'),('overlap_two_target_union_pixels','绿色重叠支持连续≥2目标')]:
            actual = counts[field]
            fraction = actual/counts['foreground_pixels'] if counts['foreground_pixels'] else None
            declared = claimed.get(field)
            if declared is not None and (type(declared) is not int or actual != declared):
                raise ValueError('OVERLAP_REPORT_DECLARED_COVERAGE_MISMATCH: '+field)
            agreement = '未记录' if declared is None else '一致' if type(declared) is int and actual == declared else '不一致，请复核'
            coverage_rows.append('<tr><td>'+label+'</td><td>'+str(actual)+'</td><td>'+_escape(fraction)+'</td><td>'+_escape(_number(declared))+'</td><td>'+agreement+'</td></tr>')
        budget()
        with (output/name).open('x', encoding='utf-8') as stream:
            stream.write(svg)
        files.append(output/name)
        omitted = [item for item in candidates if item.get('eligible') is True and item.get('selected') is not True]
        excluded = [item for item in candidates if item.get('eligible') is not True]
        unmeasured = [item for item in candidates if item.get('selected') is True and item.get('region_id') not in measured]
        inactive = [r for r in result.get('regions', []) if r.get('source_input_index') == key[0] and not r.get('region_legs')]
        def omitted_table(items, identity):
            return '<table><tr><th>ID</th><th>ROI</th><th>原因</th></tr>'+''.join('<tr><td>'+_escape(item.get(identity))+'</td><td>'+_escape(item.get('roi', item.get('template_roi')))+'</td><td>'+_escape(item.get('reason', item.get('stop_reason', '未记录测量')))+'</td></tr>' for item in items)+'</table>'
        maps.append('<section><h2>源 input '+_escape(key[0])+' / decode '+_escape(key[1])+'</h2><p><a href="'+name+'">独立SVG</a> · <a href="'+_escape(_link(decoded[key[1]]['path'], output))+'">实际源图</a>'
            +f'；局部持续区域 {len(local)}；重叠支持持续区域 {len(network)}；涉及冲突区域 {len(conflicts)}</p>'+svg
            +'<table><tr><th>2D源图联合区域</th><th>实际RLE交集像素</th><th>占原前景比例</th><th>核声明像素</th><th>核对</th></tr>'+''.join(coverage_rows)+'</table>'
            +f'<details><summary>符合源域但预算未选候选：{len(omitted)}</summary>'+omitted_table(omitted,'candidate_id')+'</details>'
            +f'<details><summary>源域或图像边界排除候选：{len(excluded)}</summary>'+omitted_table(excluded,'candidate_id')+'</details>'
            +f'<details><summary>已选但没有测量记录：{len(unmeasured)}</summary>'+omitted_table(unmeasured,'region_id')+'</details>'
            +f'<details><summary>已初始化但未测任何目标：{len(inactive)}</summary>'+omitted_table(inactive,'region_id')+'</details></section>')
    edge_rows = []
    assessed_edges = {e['edge_id']: e for e in overlap.get('edges', [])}
    source_paths = {s['source_input_index']: s.get('decode_path', []) for s in sources}
    for frozen in plan.get('edges', []):
        budget()
        edge = assessed_edges.get(frozen['edge_id'], frozen)
        targets = list(edge.get('targets') or [])
        known = {t.get('target_decode_index') for t in targets}
        for index in source_paths.get(edge['source_input_index'], [])[1:]:
            if index not in known:
                targets.append(dict(target_decode_index=index,status='UNTESTED',reasons=['没有逐目标验证记录']))
        if not targets:
            targets = [dict(target_decode_index=None,status='UNTESTED',reasons=['没有逐目标验证记录'])]
        for target in targets:
            budget()
            status = target.get('status', 'UNTESTED')
            values = [edge.get('edge_id'), edge.get('source_input_index'), target.get('target_decode_index'), status,
                      _number(target.get('max_error_px')), _number(target.get('p50_error_px')), _number(target.get('p95_error_px')),
                      edge.get('shared_foreground_pixels'), '; '.join(str(r) for r in target.get('reasons', []))]
            edge_rows.append('<tr class="edge_'+_escape(status)+'"><td>'+region_label(edge.get('region_a'))+'</td><td>'+region_label(edge.get('region_b'))+'</td>'
                             +''.join('<td>'+_escape(v)+'</td>' for v in values)+'</tr>')
    toggles = [('mask','灰：原SAM轮廓'),('candidates','淡框：所有候选'),('connections','虚线：冻结源邻接'),('selected','蓝：已选'),
               ('local','黄：局部连续≥2目标'),('network','绿：重叠支持连续≥2目标'),('conflicts','红：冲突')]
    controls = ''.join(f'<input type="checkbox" id="show_{key}" checked><label for="show_{key}">{label}</label> ' for key,label in toggles)
    css = ''.join(f'#show_{key}:not(:checked) ~ .maps .layer_{key}{{display:none}}' for key,_ in toggles)
    baseline = result.get('baseline') or {}
    baseline_rows = ''.join('<tr>'+''.join('<td>'+_escape(row.get(k))+'</td>' for k in ('source_input_index','source_decode_index','foreground_pixels','local_union_pixels','local_fraction'))+'</tr>' for row in baseline.get('source_coverage', []))
    baseline_section = ('<h2>上一局部区域方法的同口径只读对照</h2><p>same_estimator='+_escape(baseline.get('same_estimator'))+'；'+_escape(baseline.get('comparison_scope'))
        +'；测量秒 '+_escape(baseline.get('measurement_wall_seconds'))+'</p><table><tr><th>source</th><th>decode</th><th>SAM前景像素</th><th>局部联合像素</th><th>比例</th></tr>'+baseline_rows+'</table>') if baseline else ''
    def records_table(records, keys):
        return '<table><tr>'+''.join('<th>'+_escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+_escape(_number(row.get(k)))+'</td>' for k in keys)+'</tr>' for row in records)+'</table>'
    def component_section(records):
        if not records:
            return '<p>计划连通性：未知（没有冻结分量记录）。</p>'
        rows = []
        by_source = {(row.get('source_input_index'),row.get('source_decode_index')):row for row in records}
        ordered = [by_source.get((source['source_input_index'],source['source_decode_index']),
                   dict(source_input_index=source['source_input_index'],source_decode_index=source['source_decode_index'])) for source in sources]
        for row in ordered or records:
            count = row.get('component_count')
            rows.append(dict(row, connectivity='未知' if type(count) is not int else '未连通' if count > 1 else '单分量' if count == 1 else '没有已选区域'))
        return records_table(rows, ('source_input_index','source_decode_index','component_count','connectivity','isolated_region_ids','components'))
    planning_section = '<h2>冻结源图计划：连通分量与空间选择</h2><p>虚线仅表示冻结候选邻接，不表示整物体表面已连接，也不表示目标帧已经验证连通。</p>'+component_section(plan.get('planned_connected_components'))
    planning_section += '<p>选择统计：'+_escape(_number(plan.get('selection_counts')))+'</p>'
    planning_section += '<p>空间格按候选中心归入3×3格；各格分母为该格候选覆盖的原SAM前景联合像素。不同格的ROI可重叠，格面积不能加总充当整张源图覆盖。</p>'
    for source in sources:
        budget()
        planning_section += '<h3>源 input '+_escape(source['source_input_index'])+' / decode '+_escape(source['source_decode_index'])+' 空间选择</h3>'
        planning_section += records_table([source],('selected_component_count','candidate_component_count','unreachable_eligible_candidates','budget_unselected_candidates'))
        strata = source.get('spatial_strata')
        planning_section += records_table(strata,('stratum','row','column','eligible_candidates','selected_count','eligible_union_pixels','planned_union_pixels','planned_fraction')) if strata else '<p>空间格覆盖：未知（旧计划未记录）。</p>'
    prefix = result.get('same_budget64')
    prefix_section = ''
    if prefix:
        keys = ('source_input_index','source_decode_index','foreground_pixels','planned_union_pixels','local_two_target_union_pixels','overlap_two_target_union_pixels','planned_fraction','local_two_target_fraction','overlap_two_target_fraction')
        prefix_section = '<h2>64前缀与全量区域：同口径2D覆盖对照</h2><p>64前缀复用同批独立区域测量，不是第二次运行，也不是速度收益证据。'+_escape(_number(prefix.get('note')))+'</p><h3>64前缀</h3>'+records_table(prefix.get('coverage', []),keys)+component_section(prefix.get('planned_connected_components'))+'<p>计数：'+_escape(prefix.get('counts'))+'；选择统计：'+_escape(prefix.get('selection_counts'))+'</p><h3>全量</h3>'+records_table(overlap.get('coverage', []),keys)
    profile = plan.get('profile') or {}
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>重叠区域：源图覆盖与一致性</title>'
        '<style>body{font:16px sans-serif;background:#131b26;color:#e7edf5;margin:24px;line-height:1.6}a{color:#8ac9ff}svg{max-width:100%;height:auto}td,th{border:1px solid #586577;padding:5px;overflow-wrap:anywhere}table{border-collapse:collapse;margin:12px 0}section{border-top:1px solid #586577;margin-top:24px}label{margin-right:12px}.edge_CONFLICT{color:#ff8990}.edge_UNTESTED{color:#bbc2cb}'+css+'</style>'
        '<h1>重叠区域联合检查</h1><p>仍用原有整体6参数光度区域配准；像素亮度容差原样保留：留出NCC≥0.80、归一化RMSE≤0.7、有效覆盖≥0.85，并用训练集亮度校正。'
        '它们与2px几何回环阈值不同。邻边重叠检查比较两个独立区域变换对同一源位置的预测，不复制相邻区域位移，不把变换采样当成独立匹配点，不产生新3D点。</p>'
        '<p>覆盖仅是每张SAM源图的前景与区域ROI联合交集，重叠像素只计一次；不同源图分别列出，不是3D覆盖或整个物体表面。'
        '绿色表示固定预算内的重叠支持，红色冲突优先显示；并非物理对应验收。共享图像纹理、插值和重叠像素相关，不能当独立证据。'
        '未知目标SAM仍未知；UNTESTED不等于一致。</p>'
        +f'<p>区域预算最多{_escape(profile.get("max_regions",64))}、源预算最多{_escape(profile.get("max_sources",2))}；本次源数 {len(sources)}、计划区域 {len(plan.get("regions", []))}、实际区域记录 {len(result.get("regions", []))}。未选/未测名单逐源明确列出。'
        +'测量耗时 '+_escape(_number(result.get('measurement_wall_seconds')))+' s；'
        +('总耗时于RESULT.json收尾写入' if result.get('wall_seconds') is None else '总耗时 '+_escape(result['wall_seconds'])+' s')
        +'（按调用方记录，不自行推断是否包含批次报告）。</p>'
        +'<p>实际计数：'+_escape(result.get('counts', {}))+'；原生调用：'+_escape(result.get('native_calls', {}))+'。淡框包含边界排除候选，超出源图部分裁切显示，不计为已选或已跟踪。</p>'
        +('<p><a href="neighborhood/index.html">邻域独立判断与有限候选续跟踪</a>：严格结果保留，候选不计为上表通过或3D点。</p>' if result.get('neighborhood') else '')
        +planning_section+prefix_section+'<p>'+' · '.join(batch_links)+'</p>'+controls+'<div class="maps">'+''.join(maps)+'</div>'
        +'<h2>冻结邻边的逐目标验证</h2><table><tr>'+''.join('<th>'+h+'</th>' for h in ('区域A','区域B','边ID','source','target decode','状态','max px','p50 px','p95 px','共享源前景像素','原因'))+'</tr>'+''.join(edge_rows)+'</table>'
        +('<p>没有逐边验证记录，不能声明已检验一致。</p>' if not edge_rows else '')+baseline_section+'</html>')
    budget()
    with (output/'index.html').open('x',encoding='utf-8') as stream:
        stream.write(page)
    files.append(output/'index.html')
    return files
