"""Inspectable verified short prefixes, without geometry promotion."""
import html
import json
from collections import defaultdict
from pathlib import Path

from PIL import ImageDraw
from .intermediate_population_report import _dot, _link, _sheet


def render_tracklets(output, ledger, geometry, decoded, budget=lambda: None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records = ledger.get('records', [])
    geometry_by_id = {r['tracklet_id']: r for r in geometry.get('records', [])}
    groups = defaultdict(list)
    for record in records:
        groups[record['edge_ordinal']].append(record)
    files, sections, evidence = [], [], []
    for ordinal, cohort in sorted(groups.items()):
        budget()
        # The frame union contains observed frames only. No endpoint or missing
        # intermediate coordinate is interpolated from the frozen source.
        indices = list(dict.fromkeys(o['decode_index'] for r in cohort for o in r.get('observations', [])))
        if indices:
            direction = -1 if any(len(r.get('observations', []))>1 and r['observations'][1]['decode_index']<r['observations'][0]['decode_index'] for r in cohort) else 1
            indices.sort(reverse=direction<0)
            sheet, positions = _sheet(indices, decoded)
            draw = ImageDraw.Draw(sheet)
            for record in cohort:
                for i, obs in enumerate(record.get('observations', [])):
                    _dot(draw, positions[indices.index(obs['decode_index'])], obs['raw_xy'], '#50a8ff' if i==0 else '#44ed79')
            name = f'prefix_{ordinal:03d}.png'; sheet.save(output/name); files.append(output/name)
        else:
            name = None
        sources = list(dict.fromkeys(r.get('source_point', {}).get('source_decode_index') for r in cohort))
        sources = [i for i in sources if i in decoded]
        source_name = None
        if sources:
            sheet, positions = _sheet(sources, decoded); draw = ImageDraw.Draw(sheet)
            for record in cohort:
                point = record.get('source_point', {})
                index = point.get('source_decode_index')
                if index in sources:
                    _dot(draw,positions[sources.index(index)],point.get('source_raw_xy'),'#50a8ff')
            source_name=f'source_{ordinal:03d}.png';sheet.save(output/source_name);files.append(output/source_name)
        enriched=[]
        for record in cohort:
            item=dict(record)
            item['geometry']=geometry_by_id.get(record['tracklet_id'],dict(status='NOT_EVALUATED',model_assessments=[]))
            item['observations']=[dict(o,url=_link(o['path'],output)) for o in record.get('observations',[]) if o.get('path')]
            # Missing file references remain explicit in the ledger, never
            # replaced by a different image or inferred coordinate.
            item['observation_evidence']=record.get('observations',[])
            point=record.get('source_point',{});frame=decoded.get(point.get('source_decode_index'))
            item['source_image']=dict(url=_link(frame['path'],output),width=frame['width'],height=frame['height']) if frame else None
            enriched.append(item)
        evidence.extend(enriched)
        picture=(' <img src="'+source_name+'">') if source_name else '<p>源帧图像缺失</p>'
        picture+=(' <img src="'+name+'">') if name else '<p>NOT_OBSERVED：没有已验证观测图</p>'
        sections.append(f'<section><h2>区间 {ordinal}</h2><p>源点 {len(cohort)}；连续三帧候选 {sum(r.get("status")=="CANDIDATE_2D_TRACKLET" for r in cohort)}</p>{picture}<div data-edge="{ordinal}"></div></section>')
    document=dict(status=ledger.get('status','UNKNOWN'),counts=ledger.get('counts',{}),geometry_counts=geometry.get('counts',{}),records=evidence)
    target=output/'evidence.json';target.write_text(json.dumps(document,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8');files.append(target)
    payload=json.dumps(document,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page='''<!doctype html><meta charset="utf-8"><title>已验证短轨迹前缀</title><style>body{background:#131b26;color:#e7edf5;font:16px sans-serif;margin:24px}a{color:#8ac9ff}img{max-width:100%;display:block;margin:12px 0}canvas{max-width:100%;border:1px solid #586577}pre{white-space:pre-wrap;overflow-wrap:anywhere}select{max-width:100%;margin:12px}article{display:inline-block;vertical-align:top;width:404px}</style><h1>已验证短轨迹前缀</h1><p>连续至少三帧（两段已验证光流）才列为二维短轨迹候选。仅保留已验证的连续前缀，后续失败不抹去已有前缀。蓝色为源点，绿色为已验证观测；中间帧未知掩膜用琥珀色标识。短轨迹不是新增三维点，也不代表几何验收。</p><p>点击源图的点或选择 ID，查看该点的原图、坐标、终止原因、整段回环状态以及缺少位姿或掩膜的几何阻塞。后续 NOT_OBSERVED 不预测、不补点。</p><a href="evidence.json">完整证据 JSON</a>'''+''.join(sections)+'''<script type="application/json" id="evidence">'''+payload+'''</script><script>
const data=JSON.parse(document.getElementById('evidence').textContent);
for(const host of document.querySelectorAll('[data-edge]')){const records=data.records.filter(r=>r.edge_ordinal===Number(host.dataset.edge));const select=document.createElement('select');select.add(new Option('选择短轨迹 ID',''));for(const r of records)select.add(new Option(r.tracklet_id+' / '+r.status,r.tracklet_id));host.append(select);const source=document.createElement('canvas');source.width=404;source.height=570;host.append(source);const frames=document.createElement('div');host.append(frames);const detail=document.createElement('pre');host.append(detail);let selected=null;const reference=records.find(r=>r.source_image);let sourceScale=1,sourceX=0,sourceY=40;
if(reference){const im=new Image();im.onload=()=>{sourceScale=Math.min(384/im.naturalWidth,512/im.naturalHeight);sourceX=(404-im.naturalWidth*sourceScale)/2;const ctx=source.getContext('2d');ctx.drawImage(im,sourceX,sourceY,im.naturalWidth*sourceScale,im.naturalHeight*sourceScale);for(const r of records){const xy=r.source_point.source_raw_xy;if(!xy)continue;ctx.fillStyle='#50a8ff';ctx.beginPath();ctx.arc(sourceX+(xy[0]-.5)*sourceScale,sourceY+(xy[1]-.5)*sourceScale,2,0,7);ctx.fill();}};im.src=reference.source_image.url;}
function show(r){selected=r;select.value=r.tracklet_id;detail.textContent=JSON.stringify(r,null,2);frames.replaceChildren();for(const o of r.observations){const article=document.createElement('article'),text=document.createElement('p');text.textContent='decode='+o.decode_index+' raw='+JSON.stringify(o.raw_xy)+' mask='+o.original_mask_status;text.style.color=o.original_mask_status==='OK'?'#e7edf5':'#ffbd59';article.append(text);const link=document.createElement('a');link.href=o.url;link.textContent='原始全分辨率图像';article.append(link);const canvas=document.createElement('canvas');canvas.width=404;canvas.height=530;article.append(canvas);frames.append(article);const im=new Image();im.onload=()=>{const ctx=canvas.getContext('2d'),scale=Math.min(384/im.naturalWidth,512/im.naturalHeight),x=(404-im.naturalWidth*scale)/2;ctx.drawImage(im,x,0,im.naturalWidth*scale,im.naturalHeight*scale);ctx.fillStyle=o===r.observations[0]?'#50a8ff':'#44ed79';ctx.beginPath();ctx.arc(x+(o.raw_xy[0]-.5)*scale,(o.raw_xy[1]-.5)*scale,4,0,7);ctx.fill();};im.src=o.url;}const stop=document.createElement('p');stop.textContent='前缀之后：NOT_OBSERVED；终止原因：'+r.termination_reason;frames.append(stop);}
select.onchange=()=>{const r=records.find(r=>r.tracklet_id===select.value);if(r)show(r);};source.onclick=event=>{const rect=source.getBoundingClientRect(),x=(event.clientX-rect.left)*404/rect.width,y=(event.clientY-rect.top)*570/rect.height;let best=null,d=40;for(const r of records){const xy=r.source_point.source_raw_xy;if(!xy)continue;const distance=Math.hypot(x-sourceX-(xy[0]-.5)*sourceScale,y-sourceY-(xy[1]-.5)*sourceScale);if(distance<d){best=r;d=distance;}}if(best)show(best);};}
</script>'''
    target=output/'index.html';target.write_text(page,encoding='utf-8');files.append(target)
    return files
