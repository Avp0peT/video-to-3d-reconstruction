"""Population evidence renderer; observations are never promoted to 3D evidence."""
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw
from .intermediate_population import _verified_leg

NEW = '#ffd45c'
CONTINUING = '#50a8ff'
PASS = '#44ed79'


def _link(path, output):
    return quote(os.path.relpath(Path(path).resolve(), output.resolve()).replace(os.sep, '/'), safe='/.:')


def _xy(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in value)


def _accepted(arm):
    """Only completed per-leg forward/backward checks produce lifecycle dots."""
    observations = []
    for step in arm.get('steps', []):
        forward, backward = step.get('forward', {}), step.get('backward', {})
        if not _verified_leg(step):
            break
        if _xy(forward.get('observed_raw_xy')):
            observations.append(dict(decode_index=forward.get('to_decode_index'), raw_xy=forward['observed_raw_xy']))
    return observations


def _sheet(indices, decoded, columns=4):
    cellw, cellh = 404, 570
    sheet = Image.new('RGB', (cellw * min(columns, max(1, len(indices))), cellh * max(1, math.ceil(len(indices)/columns))), '#151a22')
    positions = {}
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(indices):
        frame = decoded[index]
        with Image.open(frame['path']) as original:
            pic = original.convert('RGB')
        w, h = pic.size
        pic.thumbnail((384, 512))
        x, y = slot % columns * cellw + (cellw-pic.width)//2, slot//columns * cellh + 54
        sheet.paste(pic, (x, y)); positions[slot] = (x, y, pic.width/w, pic.height/h)
        draw.text((slot%columns*cellw+10,slot//columns*cellh+8), f'decode={index}  t={frame.get("time_seconds", "?")}', fill='white')
        unknown = frame.get('mask') is None
        draw.text((slot%columns*cellw+10,slot//columns*cellh+27), 'UNKNOWN / NOT SEGMENTED' if unknown else str(frame.get('original_mask_status', 'UNKNOWN')), fill='#ffbd59' if unknown else '#b5cbe4')
    return sheet, positions


def _dot(draw, position, raw, color):
    if not _xy(raw):
        return
    x,y,sx,sy = position; x += (raw[0]-.5)*sx; y += (raw[1]-.5)*sy
    draw.ellipse((x-2,y-2,x+2,y+2), fill=color)


def render_population(output, result, decoded, budget=lambda: None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    files, evidence = [], []
    plans = result.get('plan', {}).get('edges', [])
    results = result.get('edges', [])
    for number, edge in enumerate(plans):
        budget()
        measured = results[number] if number < len(results) else {}
        rows = measured.get('rows', [])
        points = edge.get('points', [])
        byid = {r['group_id']: r for r in rows}
        path = edge.get('decode_path', [])
        if not path:
            continue
        source, target = path[0], path[-1]
        counts = {'planned':len(points), 'measured':len(rows), 'new_seed':sum(p.get('birth_kind')=='NEW_SEED' for p in points), 'continuing':sum(p.get('birth_kind')=='CONTINUING_OBSERVATION' for p in points)}
        counts.update({name+'_passed':sum(r.get(name,{}).get('status')=='PASSED' for r in rows) for name in ('direct','dense')})
        sheet, pos = _sheet([source], decoded); draw = ImageDraw.Draw(sheet)
        for point in points:
            _dot(draw,pos[0],point.get('source_raw_xy'), NEW if point.get('birth_kind')=='NEW_SEED' else CONTINUING if point.get('birth_kind')=='CONTINUING_OBSERVATION' else '#aaaaaa')
        name = f'source_population_{number:03d}.png'; sheet.save(output/name); files.append(output/name)
        sheet,pos = _sheet([target,target],decoded,2); draw = ImageDraw.Draw(sheet)
        for slot,armname in enumerate(('direct','dense')):
            draw.text((slot*404+10,42), f'{armname}: PASSED={counts[armname+"_passed"]}',fill=PASS)
            for row in rows:
                arm=row.get(armname,{})
                if arm.get('status')=='PASSED':
                    for obs in arm.get('observed_raw_coordinates',[]):
                        if obs.get('decode_index')==target:_dot(draw,pos[slot],obs.get('raw_xy'),PASS)
        name=f'endpoint_comparison_{number:03d}.png';sheet.save(output/name);files.append(output/name)
        sheet,pos = _sheet(path,decoded); draw=ImageDraw.Draw(sheet)
        for slot in range(len(path)):
            draw.text((slot*404+10,42), 'SOURCE COHORT' if slot==0 else 'PER-LEG CHECKS ONLY; SEE ENDPOINT GATE', fill=PASS)
        for point in points:_dot(draw,pos[0],point.get('source_raw_xy'),NEW if point.get('birth_kind')=='NEW_SEED' else CONTINUING)
        for row in rows:
            for obs in _accepted(row.get('dense',{})):
                if obs['decode_index'] in path:_dot(draw,pos[path.index(obs['decode_index'])],obs['raw_xy'],PASS)
        name=f'lifecycle_{number:03d}.png';sheet.save(output/name);files.append(output/name)
        frames=[dict(decode_index=i,url=_link(decoded[i]['path'],output),width=decoded[i]['width'],height=decoded[i]['height'],mask_status=decoded[i].get('original_mask_status','UNKNOWN'),time_seconds=decoded[i].get('time_seconds')) for i in path]
        records=[]
        for point in points:
            row=byid.get(point['group_id'],{})
            records.append(dict(**point,measurement=row,accepted_dense=_accepted(row.get('dense',{})),outcome=row.get('outcome','NOT_MEASURED')))
        evidence.append(dict(number=number,counts=counts,summary=measured.get('summary', {}),source_domain=edge.get('source_domain'),target_domain=edge.get('target_domain'),frames=frames,points=records))
    evidence_path=output/'population_evidence.json'
    evidence_path.write_text(json.dumps(dict(status=result.get('status','UNKNOWN'),edges=evidence),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');files.append(evidence_path)
    payload=json.dumps(evidence,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    def table(title, records):
        if not records:
            return '<h3>'+html.escape(title)+'</h3><p>尚无测量汇总</p>'
        columns=list(dict.fromkeys(k for row in records for k in row))
        return '<h3>'+html.escape(title)+'</h3><table border="1" cellpadding="6"><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(k, '—')))+'</td>' for k in columns)+'</tr>' for row in records)+'</table>'
    sections=[]
    for edge in evidence:
        n=edge['number']; c=html.escape(json.dumps(edge['counts'],ensure_ascii=False))
        links=' '.join(f'<a href="{html.escape(f["url"],quote=True)}">原始全分辨率 decode={f["decode_index"]}</a>' for f in edge['frames'])
        summary=edge['summary']
        coverage=[]
        for label, record in summary.get('coverage',{}).items():
            coverage.append({'对象掩膜网格':label,'已占网格':record.get('occupied'),'有效对象网格':record.get('total'),'二维比例':record.get('fraction')})
        tables=table('原始对象掩膜内二维覆盖（不是表面覆盖）',coverage)
        tables+=table('按出生类型统计',[dict(出生类型=k,**v) for k,v in summary.get('by_birth_kind',{}).items()])
        tables+=table('逐段通过前向与反向检查的存活点',summary.get('dense_per_leg_survival',[]))
        tables+=table('生命周期长度分布（已验证段数）',summary.get('survival_histogram',[]))
        sections.append(f'<section><h2>边 {n}</h2><p>{c}</p>{tables}<p>{links}</p><img src="source_population_{n:03d}.png"><img src="endpoint_comparison_{n:03d}.png"><img src="lifecycle_{n:03d}.png"><div class="interactive" data-edge="{n}"></div></section>')
    page='''<!doctype html><meta charset="utf-8"><title>候选点群与生命周期</title><style>body{background:#111923;color:#e7edf5;font:16px sans-serif;margin:24px}img{max-width:100%;display:block;margin:12px 0}a{color:#8ac9ff;margin-right:16px}canvas{max-width:100%;border:1px solid #576678}pre{white-space:pre-wrap;overflow-wrap:anywhere}select{margin:8px}</style><h1>全物体二维候选点群与生命周期</h1><p>黄色：新播种；蓝色：前序延续观测；绿色：通过相应检查的观测。生命周期图只表示逐段双向通过，整条路线闭环仍可能失败；端点对照图只画整条路线通过的点。图中仅画点，点击交互源图选择点 ID，查看坐标、失败原因及逐帧路径。</p><p>这是二维网格覆盖与光流证据，不代表物体表面覆盖、三维点或几何验收。新增中间帧未分割，掩膜为 UNKNOWN（琥珀色）；中间帧没有追加角点播种。缺失测量保持 NOT_MEASURED。</p><a href="population_evidence.json">完整证据 JSON</a>'''+''.join(sections)+'''<script id="evidence" type="application/json">'''+payload+'''</script><script>
const edges=JSON.parse(document.getElementById('evidence').textContent);
for(const e of edges){const host=document.querySelector('[data-edge="'+e.number+'"]');
const filter=document.createElement('select');for(const s of ['ALL','NEW_SEED','CONTINUING_OBSERVATION','NOT_MEASURED','RECOVERED','LOST','BOTH_PASS','BOTH_FAIL'])filter.add(new Option(s,s));host.append(filter);
const select=document.createElement('select');host.append(select);const panel=document.createElement('div');host.append(panel);const detail=document.createElement('pre');host.append(detail);let selected=null;const imgs=[];
const visible=()=>e.points.filter(p=>filter.value==='ALL'||p.birth_kind===filter.value||p.outcome===filter.value);
function refresh(){select.replaceChildren(new Option('点击源图或选择点 ID',''));for(const p of visible())select.add(new Option(p.group_id,p.group_id));draw();}
function draw(){panel.replaceChildren();detail.textContent=selected?JSON.stringify(selected,null,2):'未选点；仅显示源帧候选点。';e.frames.forEach((f,i)=>{const canvas=document.createElement('canvas');canvas.width=404;canvas.height=570;panel.append(canvas);const ctx=canvas.getContext('2d');const im=imgs[i];if(!im||!im.complete||!im.naturalWidth)return;const scale=Math.min(384/f.width,512/f.height),x=(404-f.width*scale)/2,y=40;ctx.fillStyle='#ffbd59';ctx.fillText('decode='+f.decode_index+' '+f.mask_status,10,20);ctx.drawImage(im,x,y,f.width*scale,f.height*scale);
function dot(raw,color){if(!raw)return;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x+(raw[0]-.5)*scale,y+(raw[1]-.5)*scale,selected?4:2,0,7);ctx.fill();}
if(i===0){for(const p of selected?[selected]:visible())dot(p.source_raw_xy,p.birth_kind==='NEW_SEED'?'#ffd45c':'#50a8ff');canvas.onclick=event=>{const rect=canvas.getBoundingClientRect(),px=(event.clientX-rect.left)*404/rect.width,py=(event.clientY-rect.top)*570/rect.height;let best=null,d=100;for(const p of visible()){const raw=p.source_raw_xy;if(!raw)continue;const dd=Math.hypot(px-x-(raw[0]-.5)*scale,py-y-(raw[1]-.5)*scale);if(dd<d){d=dd;best=p;}}if(best){selected=best;select.value=best.group_id;draw();}};}else if(selected){for(const o of selected.accepted_dense)if(o.decode_index===f.decode_index)dot(o.raw_xy,'#44ed79');}});}
filter.onchange=()=>{selected=null;refresh();};select.onchange=()=>{selected=e.points.find(p=>p.group_id===select.value)||null;draw();};e.frames.forEach((f,i)=>{const im=new Image();imgs[i]=im;im.onload=draw;im.src=f.url;});refresh();}
</script>'''
    page_path=output/'index.html';page_path.write_text(page,encoding='utf-8');files.append(page_path)
    return files
