"""Actual images and saved whole-area ECC evidence; no point correspondences."""
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


def _escape(value):
    return html.escape(str(value), quote=True)


def _ascii(value):
    return str(value).encode('ascii', 'backslashreplace').decode('ascii')


def _link(path, output):
    return quote(os.path.relpath(Path(path).resolve(), output.resolve()).replace(os.sep, '/'), safe='/.:')


def _image(frame):
    from PIL import Image
    with Image.open(frame['path']) as source:
        image = source.convert('RGB')
    if (type(frame.get('width')) is not int or type(frame.get('height')) is not int
            or image.size != (frame['width'], frame['height'])):
        raise ValueError('AREA_REPORT_SOURCE_DIMENSIONS_MISMATCH')
    return image


def _roi(value, image):
    if type(value) is not dict or set(value) != {'left', 'top', 'width', 'height'}:
        raise ValueError('AREA_REPORT_ROI_FIELDS')
    if any(type(value[k]) is not int for k in value):
        raise ValueError('AREA_REPORT_INTEGER_ROI_REQUIRED')
    x, y, w, h = (value[k] for k in ('left', 'top', 'width', 'height'))
    if min(x, y) < 0 or min(w, h) < 1 or x+w > image.width or y+h > image.height:
        raise ValueError('AREA_REPORT_ROI_OUTSIDE_IMAGE')
    return x, y, x+w, y+h


def _matrix(value, shape=None):
    if type(value) is not list or not value or type(value[0]) is not list or not value[0]:
        raise ValueError('AREA_REPORT_PIXEL_MATRIX_REQUIRED')
    h, w = len(value), len(value[0])
    if max(h, w) > 1024 or any(type(row) is not list or len(row) != w for row in value):
        raise ValueError('AREA_REPORT_PIXEL_MATRIX_BOUND')
    if shape is not None and (w, h) != shape:
        raise ValueError('AREA_REPORT_PIXEL_MATRIX_SHAPE')
    return w, h


def _mask(value, shape):
    _matrix(value, shape)
    if any(type(v) is not int or v not in (0, 255) for row in value for v in row):
        raise ValueError('AREA_REPORT_BINARY_MASK_REQUIRED')
    return value


def _gray(value, valid=None):
    from PIL import Image
    shape = _matrix(value)
    if valid is not None:
        _mask(valid, shape)
    pixels = []
    for y, row in enumerate(value):
        for x, v in enumerate(row):
            if valid is not None and valid[y][x] == 0:
                pixels.append((110, 116, 126))
            else:
                if type(v) not in (int, float) or not math.isfinite(v):
                    raise ValueError('AREA_REPORT_NONFINITE_PIXEL')
                g = max(0, min(255, round(v)))
                pixels.append((g, g, g))
    image = Image.new('RGB', shape)
    image.putdata(pixels)
    return image


def _heatmap(residual, valid):
    from PIL import Image
    shape = _matrix(residual)
    _mask(valid, shape)
    pixels = []
    for y, row in enumerate(residual):
        for x, value in enumerate(row):
            if valid[y][x] == 0:
                pixels.append((110, 116, 126))
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('AREA_REPORT_NONFINITE_RESIDUAL')
            t = min(1., abs(value)/32.)
            pixels.append((round(255*t), round(90*(1-t)), round(255*(1-t))))
    image = Image.new('RGB', shape)
    image.putdata(pixels)
    return image


def _split(evidence):
    from PIL import Image
    shape = _matrix(evidence['template_uint8'])
    train = _mask(evidence['train_mask_uint8'], shape)
    held = _mask(evidence['holdout_mask_uint8'], shape)
    template = _mask(evidence['template_mask_uint8'], shape)
    pixels = []
    for y in range(shape[1]):
        for x in range(shape[0]):
            if train[y][x] and held[y][x]:
                pixels.append((255, 91, 98))
            elif train[y][x]:
                pixels.append((75, 213, 255))
            elif held[y][x]:
                pixels.append((255, 211, 90))
            elif template[y][x]:
                pixels.append((45, 57, 74))
            else:
                pixels.append((110, 116, 126))
    image = Image.new('RGB', shape)
    image.putdata(pixels)
    return image


def _polygon(draw, polygon, origin=(0, 0), scales=(1, 1), color='#ffad45'):
    if polygon is None:
        return
    if (type(polygon) is not list or len(polygon) != 4
            or any(type(p) is not list or len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in p) for p in polygon)):
        raise ValueError('AREA_REPORT_QUADRILATERAL_REQUIRED')
    points = [((p[0]-.5-origin[0])*scales[0], (p[1]-.5-origin[1])*scales[1]) for p in polygon]
    draw.line(points+[points[0]], fill=color, width=3)


def _paste(sheet, image, box, nearest=False):
    from PIL import Image
    left, top, width, height = box
    scale = min(width/image.width, height/image.height)
    pic = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))),
                       Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
    sheet.paste(pic, (left, top))


def motion_records(result, decoded, output, *, budget=lambda: None):
    """Read saved positions only. A failed fit is never an accepted position."""
    output = Path(output)
    records = []
    for region in result.get('regions', []):
        budget()
        polygon = region.get('source_polygon_raw4x2')
        if polygon is not None:
            if (len(polygon) != 4 or any(len(p) != 2 or any(type(v) not in (int, float)
                    or not math.isfinite(v) for v in p) for p in polygon)):
                raise ValueError('MOTION_REPORT_INVALID_SOURCE_POLYGON')
            center = [sum(p[k] for p in polygon)/4 for k in (0, 1)]
        else:
            center = None
        previous = center
        previous_index = region['source_decode_index']
        legs = region.get('region_legs', [])
        by_index = {r['target_decode_index']: r for r in legs}
        if len(by_index) != len(legs):
            raise ValueError('MOTION_REPORT_DUPLICATE_TARGET')
        path = region.get('decode_path') or [region['source_decode_index']]+list(by_index)
        rows = []
        for index in path:
            budget()
            source = index == region['source_decode_index']
            leg = by_index.get(index) or {}
            forward = leg.get('forward') or {}
            quad = polygon if source else forward.get('polygon_raw4x2')
            frame = decoded.get(index) or {}
            available = bool(frame.get('path') and Path(frame['path']).is_file())
            position = None
            if quad is not None:
                if (len(quad) != 4 or any(len(p) != 2 or any(type(v) not in (int, float)
                        or not math.isfinite(v) for v in p) for p in quad)):
                    raise ValueError('MOTION_REPORT_INVALID_TARGET_POLYGON')
                position = [sum(p[k] for p in quad)/4 for k in (0, 1)]
            state = 'SOURCE_REFERENCE' if source else leg.get('status', 'NOT_RUN')
            accepted = not source and state == 'ACCEPTED'
            if accepted and position is None:
                raise ValueError('MOTION_REPORT_ACCEPTED_WITHOUT_POSITION')
            roi = forward.get('search_roi')
            matches = None
            if roi is not None and previous is not None:
                # Reconstruct the existing last-accepted-center policy, including
                # pixel-center convention, rounding and image-edge clipping.
                x, y = [round(v-.5) for v in previous]
                left, top = max(0, x-48), max(0, y-48)
                expected = dict(left=left, top=top, width=min(frame['width'], x+49)-left,
                                height=min(frame['height'], y+49)-top)
                matches = roi == expected
            rows.append(dict(decode_index=index, time_seconds=frame.get('time_seconds'),
                image=_link(frame['path'], output) if available else None,
                width=frame.get('width'), height=frame.get('height'),
                mask_status=frame.get('original_mask_status') or 'UNKNOWN', status=state,
                search_roi=roi, search_requested_raw_xy=list(previous) if roi is not None and previous is not None else None,
                search_anchor_decode_index=previous_index if roi is not None else None,
                search_roi_matches_last_accepted_center=matches,
                source_reference_polygon=polygon, fitted_polygon=quad,
                fitted_center_raw_xy=position, accepted=accepted,
                accepted_center_raw_xy=position if accepted else None,
                displacement_from_source_px=math.dist(position, center) if position is not None and center is not None else None,
                reasons=leg.get('reasons', []), cycle_error_px=leg.get('cycle_error_px')))
            if accepted:
                previous, previous_index = position, index
        records.append(dict(region_id=region['region_id'], source_decode_index=region['source_decode_index'],
            source_input_index=region.get('source_input_index'), verified_targets=region.get('verified_targets'),
            stop_reason=region.get('stop_reason'), frames=rows))
    return dict(schema='area-motion-view-v1', evidence='saved measurements; no new registration',
        policy='fixed source appearance; search follows last accepted center; no velocity extrapolation',
        coordinate_system='raw pixel centers in full image; SVG image edges at 0,width and 0,height',
        geometry_accepted=False, new_model_points=0, regions=records)


def render_area_motion(output, result, decoded, *, budget=lambda: None):
    """Interactive full-frame position view, automatically used by both area paths."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output/'motion.json', output/'motion.html']
    for path in paths:
        if path.exists():
            raise FileExistsError(path)
    data = motion_records(result, decoded, output, budget=budget)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False)
    safe = encoded.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>区域随帧移动：实际位置</title><style>body{font:16px/1.65 system-ui;max-width:1250px;margin:24px auto;padding:0 16px;background:#16202d;color:#e7edf5}a{color:#90d0ff}select{max-width:100%;padding:8px}button{padding:8px}svg{width:100%;max-height:70vh;background:#0c121b}input[type=range]{width:50%}pre{white-space:pre-wrap;overflow-wrap:anywhere}.legend{padding:12px;background:#233349}</style>
<h1>区域的位置与形状随帧变化</h1><p>选择区域并拖动帧滑块。这里只显示实际保存的图像和测量，不补间、不预测遮挡后的轨迹。源图覆盖总览中的框只是起始位置。</p>
<p class="legend">蓝框＝本帧实际搜索窗；蓝十字＝搜索中心依据（上一验收位置，非速度预测）；绿四边形＝本帧记录通过；橙四边形＝已拟合但未通过全部检查。没有拟合结果就不画当前位置。原SAM未知仍为UNKNOWN。</p>
<p>固定的是初始33×33纹理参考，目标位置可平移、旋转、缩放与剪切。搜索窗目前跟随上一验收中心，尚无速度外推或自适应扩大。</p>
<select id="region" aria-label="区域"></select><p><button id="prev">上一帧</button> <button id="play">播放实际帧</button> <button id="next">下一帧</button> <input id="frame" type="range" min="0" value="0" aria-label="实际帧"><label><input id="ghost" type="checkbox">显示固定源坐标对照（不是当前结果）</label></p>
<p id="label"></p><svg id="view" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="实际帧与区域位置"></svg><pre id="details"></pre><p><a href="motion.json">坐标与状态记录</a>。二维拟合和往返一致性不等于物理对应或三维验收。</p>
<script type="application/json" id="data">DATA_TOKEN</script><script>
const data=JSON.parse(document.getElementById('data').textContent),sel=document.getElementById('region'),slider=document.getElementById('frame'),view=document.getElementById('view');let timer=null;
const ns='http://www.w3.org/2000/svg';function shape(tag,attrs){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);view.appendChild(e);return e;}
function polygon(q,color,dash){if(q)shape('polygon',{points:q.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2.5,'stroke-dasharray':dash||'','vector-effect':'non-scaling-stroke'});}
function stop(){if(timer)clearInterval(timer);timer=null;document.getElementById('play').textContent='播放实际帧';}
function draw(){const r=data.regions[+sel.value];view.replaceChildren();if(!r)return;slider.max=Math.max(0,r.frames.length-1);const f=r.frames[+slider.value];view.setAttribute('viewBox',`0 0 ${f.width||1024} ${f.height||768}`);
if(f.image)shape('image',{href:f.image,x:0,y:0,width:f.width,height:f.height});
if(document.getElementById('ghost').checked)polygon(f.source_reference_polygon,'#bdc4ce','6 6');
if(f.search_roi){const b=f.search_roi;shape('rect',{x:b.left,y:b.top,width:b.width,height:b.height,fill:'none',stroke:'#58baff','stroke-width':2,'vector-effect':'non-scaling-stroke'});}
if(f.search_requested_raw_xy){const [x,y]=f.search_requested_raw_xy;shape('path',{d:`M${x-5},${y}H${x+5}M${x},${y-5}V${y+5}`,stroke:'#58baff','stroke-width':2});}
polygon(f.fitted_polygon,f.accepted?'#54edab':f.status==='SOURCE_REFERENCE'?'#ffffff':'#ffad45');
document.getElementById('label').textContent=`decode ${f.decode_index} | ${f.status} | 原SAM ${f.mask_status} | 距源中心 ${f.displacement_from_source_px==null?'未知':f.displacement_from_source_px.toFixed(3)+' px'}`;
document.getElementById('details').textContent=JSON.stringify({region:r.region_id,search_anchor_decode:f.search_anchor_decode_index,search_requested_raw_xy:f.search_requested_raw_xy,search_roi_matches_last_accepted_center:f.search_roi_matches_last_accepted_center,fitted_center_raw_xy:f.fitted_center_raw_xy,accepted_center_raw_xy:f.accepted_center_raw_xy,cycle_error_px:f.cycle_error_px,reasons:f.reasons,image_available:!!f.image},null,2);}
data.regions.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${r.region_id} | 通过${r.verified_targets} | 源${r.source_decode_index}`;sel.appendChild(o);});
if(data.regions.length)sel.value=data.regions.reduce((best,r,i)=>r.verified_targets>data.regions[best].verified_targets?i:best,0);
sel.onchange=()=>{stop();slider.value=0;draw();};slider.oninput=()=>{stop();draw();};document.getElementById('ghost').onchange=draw;
document.getElementById('prev').onclick=()=>{stop();slider.value=Math.max(0,+slider.value-1);draw();};document.getElementById('next').onclick=()=>{stop();slider.value=Math.min(+slider.max,+slider.value+1);draw();};
document.getElementById('play').onclick=()=>{if(timer){stop();return;}if(+slider.value===+slider.max)slider.value=0;draw();document.getElementById('play').textContent='暂停';timer=setInterval(()=>{if(+slider.value>=+slider.max){stop();return;}slider.value=+slider.value+1;draw();},900);};draw();
</script></html>'''.replace('DATA_TOKEN', safe)
    budget()
    with paths[0].open('x', encoding='utf-8') as stream:
        stream.write(encoded+'\n')
    with paths[1].open('x', encoding='utf-8') as stream:
        stream.write(page)
    return paths


def _panel(region, decoded, budget):
    from PIL import Image, ImageDraw
    source = _image(decoded[region['source_decode_index']])
    template_box = _roi(region['template_roi'], source) if region.get('template_roi') is not None else None
    if template_box is not None and (template_box[2]-template_box[0] != 33 or template_box[3]-template_box[1] != 33):
        raise ValueError('AREA_REPORT_TEMPLATE_MUST_BE_33X33')
    legs = region.get('region_legs', [])
    sheet = Image.new('RGB', (1440, 340+330*len(legs)), '#151d29')
    draw = ImageDraw.Draw(sheet)
    source_preview = source.copy()
    _polygon(ImageDraw.Draw(source_preview), region.get('source_polygon_raw4x2'))
    _paste(sheet, source_preview, (8, 40, 400, 270))
    if template_box is not None:
        _paste(sheet, source.crop(template_box), (440, 40, 250, 250), nearest=True)
    else:
        draw.text((440, 85), 'No valid actual template ROI', fill='white')
    draw.text((8, 8), f'source decode={region["source_decode_index"]}: actual frame / initial region', fill='white')
    draw.text((440, 8), 'actual source template: 33x33', fill='white')
    evidence = None
    for leg in legs:
        candidate = ((leg.get('forward') or {}).get('alignment') or {}).get('pixel_evidence')
        if candidate:
            evidence = candidate
            break
    if evidence:
        _paste(sheet, _split(evidence), (730, 40, 250, 250), nearest=True)
    draw.text((730, 8), 'cyan=train; yellow=heldout; gray=excluded', fill='white')
    draw.text((1010, 45), 'One joint 6-parameter affine warp', fill='white')
    draw.text((1010, 67), 'Photometric fit; no point matching', fill='white')
    draw.text((1010, 89), 'Quad = transformed area boundary', fill='white')
    draw.text((1010, 111), 'Not a recovered physical surface', fill='white')
    for slot, leg in enumerate(legs):
        budget()
        y = 340+slot*330
        index = leg['target_decode_index']
        frame = decoded.get(index) or {}
        forward = leg.get('forward') or {}
        alignment = forward.get('alignment') or {}
        pixels = alignment.get('pixel_evidence')
        draw.text((8, y+4), _ascii(f'decode={index} leg={leg.get("status")} align={alignment.get("status", "NOT_AVAILABLE")} cycle_px={leg.get("cycle_error_px")}'), fill='white')
        draw.text((8, y+23), 'SAM='+_ascii(frame.get('original_mask_status') or 'UNKNOWN'), fill='#ffad45')
        if frame.get('path') and Path(frame['path']).is_file() and forward.get('search_roi'):
            actual = _image(frame)
            box = _roi(forward['search_roi'], actual)
            crop = actual.crop(box)
            _polygon(ImageDraw.Draw(crop), forward.get('polygon_raw4x2'), origin=box[:2])
            _paste(sheet, crop, (8, y+65, 400, 245))
        else:
            draw.text((8, y+85), 'Actual target crop unavailable / no saved search ROI', fill='white')
        draw.text((8, y+46), 'actual target search; orange=one model-derived quad', fill='white')
        if not pixels or any(pixels.get(k) is None for k in ('warped_search_gray','valid_mask_uint8','residual_after_train_photometric')):
            draw.text((440, y+85), 'NO PIXEL EVIDENCE: no fabricated warp or residual', fill='white')
            continue
        shape = _matrix(pixels['template_uint8'])
        if shape != (33, 33):
            raise ValueError('AREA_REPORT_TEMPLATE_EVIDENCE_SHAPE')
        _matrix(pixels['warped_search_gray'], shape)
        _matrix(pixels['residual_after_train_photometric'], shape)
        valid = _mask(pixels['valid_mask_uint8'], shape)
        template_valid = _mask(pixels['template_mask_uint8'], shape)
        foreground = pixels.get('search_foreground_valid_mask_uint8')
        if foreground is not None:
            _matrix(foreground, shape)
            if any(type(v) is not int or not 0 <= v <= 255 for row in foreground for v in row):
                raise ValueError('AREA_REPORT_SUPPORT_MASK_UINT8_REQUIRED')
        residual_valid = [[255 if valid[y][x] and template_valid[y][x]
                           and (foreground is None or foreground[y][x] > 0) else 0
                           for x in range(shape[0])] for y in range(shape[1])]
        _paste(sheet, _gray(pixels['warped_search_gray'], valid), (440, y+65, 245, 245), nearest=True)
        _paste(sheet, _heatmap(pixels['residual_after_train_photometric'], residual_valid), (730, y+65, 245, 245), nearest=True)
        _paste(sheet, _split(pixels), (1010, y+65, 245, 245), nearest=True)
        draw.text((440, y+46), 'saved target pulled to template', fill='white')
        draw.text((730, y+46), 'abs photometric residual; 32 gray-level cap', fill='white')
        draw.text((1010, y+46), 'fixed train / heldout support', fill='white')
    return sheet


def render_area_tracking(output, result, decoded, *, budget=lambda: None):
    """Write exclusive new PNG/HTML artifacts; never mutate input evidence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    budget()
    regions = result.get('regions', [])
    planned = []
    for i, region in enumerate(regions):
        frame = decoded.get(region.get('source_decode_index')) or {}
        available = frame.get('path') and Path(frame['path']).is_file()
        planned.append((region, f'area_{i:03d}.png' if available else None))
    motion_names = ['motion.html', 'motion.json'] if any(name for _, name in planned) else []
    for name in ['index.html']+motion_names+[name for _, name in planned if name]:
        if (output/name).exists():
            raise FileExistsError(output/name)
    baseline = {row['region_id']: row for row in result.get('point_baseline', [])}
    files, sections = [], []
    for region, name in planned:
        budget()
        if name:
            panel = _panel(region, decoded, budget)
            budget()
            with (output/name).open('xb') as stream:
                panel.save(stream, format='PNG')
            files.append(output/name)
        rows = []
        for leg in region.get('region_legs', []):
            budget()
            forward = leg.get('forward') or {}
            alignment = forward.get('alignment') or {}
            frame = decoded.get(leg['target_decode_index']) or {}
            link = ('<a href="'+_escape(_link(frame['path'], output))+'">实际目标图</a>') if frame.get('path') and Path(frame['path']).is_file() else '实际目标图缺失'
            metrics = []
            for subset in ('train', 'holdout'):
                values = alignment.get(subset) or {}
                metrics.extend(values.get(k) for k in ('count','valid_count','coverage','ncc','rmse','normalized_rmse'))
            values = [leg['target_decode_index'], frame.get('time_seconds', 'UNKNOWN'), frame.get('original_mask_status') or 'UNKNOWN',
                      leg.get('status'), alignment.get('status', 'NOT_AVAILABLE'), leg.get('cycle_error_px'),
                      json.dumps(leg.get('reasons', []), ensure_ascii=False), *metrics,
                      json.dumps(alignment.get('photometric'), ensure_ascii=False), json.dumps(alignment.get('native_calls'), ensure_ascii=False)]
            rows.append('<tr><td>'+link+'</td>'+''.join('<td>'+_escape(v)+'</td>' for v in values)+'</tr>')
        headings = ['目标图','decode','原图时间s','SAM','区域状态','光度状态','循环误差px','失败原因']
        headings += [subset+' '+metric for subset in ('train','holdout') for metric in ('count','valid','coverage','NCC','RMSE','normalized RMSE')]
        headings += ['光度拟合','native calls']
        picture = f'<img src="{name}" alt="整体区域配准：原模板、目标四边形、拉回图、残差和训练留出区">' if name else '<p>实际源图不可用，未生成配准图。</p>'
        point_comparison = baseline.get(region.get('region_id'))
        comparison = ('<p>旧点方法只读对照（点与整体区域口径不同，不能直接比较成功总数）：</p><pre>'+_escape(json.dumps(point_comparison, ensure_ascii=False, indent=2))+'</pre>') if point_comparison else ''
        source_frame = decoded.get(region.get('source_decode_index')) or {}
        source_link = ('<a href="'+_escape(_link(source_frame['path'], output))+'">实际源图</a>') if source_frame.get('path') and Path(source_frame['path']).is_file() else '实际源图缺失'
        sections.append('<section><h2>'+_escape(region.get('region_id'))+'</h2><p>整体状态 '+_escape(region.get('status'))
            +'；通过目标数 '+_escape(region.get('verified_targets'))+'；停止原因 '+_escape(region.get('stop_reason'))
            +'；源decode '+_escape(region.get('source_decode_index'))+'；源SAM '+_escape(source_frame.get('original_mask_status') or 'UNKNOWN')
            +'；源模板ROI '+_escape(region.get('template_roi'))+'；'+source_link+'</p>'+picture
            +'<table><tr>'+''.join('<th>'+_escape(h)+'</th>' for h in headings)+'</tr>'+''.join(rows)+'</table>'+comparison+'</section>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>区域整体匹配与形变</title>'
        '<style>body{font:16px sans-serif;background:#131b26;color:#e7edf5;margin:24px;line-height:1.6}a{color:#8ac9ff}img{max-width:100%;display:block}table{border-collapse:collapse}td,th{border:1px solid #586577;padding:5px}section{margin-top:24px;border-top:1px solid #586577}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>区域整体匹配：联合光度仿射配准</h1>'
        +('<p><a href="motion.html">逐帧查看实际区域如何移动、旋转及失败</a>：固定源模板不等于固定目标位置，搜索窗跟随上一次验收位置。</p>' if motion_names else '')
        +'<p>每个33×33固定源模板对各目标独立估计一个整体6参数仿射变换；这是整块像素的光度联合配准。'
        '橙色四边形表示同一个形变模型产生的区域边界，不是独立匹配点；不绘制独立点箭头，不把变换网格算作新对应。'
        '双向拟合与循环误差、training/heldout光度误差分别报告；train拟合光度参数不等于独立真值。</p>'
        '<p>每图含源图位置、实际33×33模板、目标实际搜索裁剪、保存的目标拉回灰图、训练光度拟合后的绝对残差热图及固定训练/留出区。'
        '热图以32个8bit亮度级饱和显示，灰色表示图像外、模板排除或无目标采样支持；无拉回像素证据不生成拉回图或残差图。'
        '青色训练、黄色留出、灰色排除；红色训练/留出重叠只表示记录冲突，不是有效留出。</p>'
        '<p>局部仿射和光度模型是假设，不能表达一般非刚性形变或保证遮挡恢复；未知SAM保持未知，不冒充目标前景。'
        '共享纹理/插值/图像patch可能相关，高相关或循环一致不证明唯一物体对应；不称完整物体恢复、物理表面或3D精度提高。</p>'
        +(''.join(sections) if sections else '<p>没有整体区域记录；未生成占位配准图。</p>')+'</html>')
    budget()
    target = output/'index.html'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(page)
    files.append(target)
    if motion_names:
        files.extend(render_area_motion(output, result, decoded, budget=budget))
    return files
