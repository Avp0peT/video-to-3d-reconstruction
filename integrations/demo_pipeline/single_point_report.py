"""Offline visual report of saved single-point measurements, never tracking."""
from pathlib import Path
import json
import html
import math
from .balanced_point_report import _read, _sha


def render(output):
    """Read frozen INPUTS/TRACE; render actual frames, GIF and local crops."""
    from PIL import Image, ImageDraw
    out=Path(output).resolve(strict=True)
    if any((out/p).exists() for p in ('index.html','SUMMARY.json','report_assets','motion.gif','contact_sheet.png','first_alarm_local.png')):
        raise ValueError('fresh single-point report required')
    bound={}
    def bind(path,digest=None):
        p=Path(path).resolve(strict=True);h=_sha(p)
        if digest is not None and h!=digest:raise ValueError('source hash mismatch: '+str(p))
        if str(p) in bound and bound[str(p)]!=h:raise ValueError('source changed')
        bound[str(p)]=h;return p
    def read(path):return _read(bind(path))
    inputs=read(out/'INPUTS.json');trace=read(out/'TRACE.json')
    for doc in (inputs,trace):
        for r in doc.get('source_bindings',doc.get('bindings',[])):
            p=bind(r['path'],r['sha256'])
            if 'bytes' in r and p.stat().st_size!=r['bytes']:raise ValueError('source byte mismatch')
    source={f['decode_index']:f for f in inputs['frames']}
    if len(source)!=len(inputs['frames']):raise ValueError('duplicate decoded frame identity')
    for frame in inputs['frames']:
        bind(frame['path'],frame['sha256'])
    seed=inputs['seed'];comparison=inputs['comparison'];frames=trace['frames']
    if not frames:raise ValueError('empty trace')
    seen=set();previous=-1;display=[];assets=out/'report_assets';assets.mkdir()
    def xy(value):
        if value is None:return None
        if not isinstance(value,(list,tuple)) or len(value)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in value):
            raise ValueError('invalid finite measured coordinate')
        return [float(v) for v in value]
    allowed={'SELECTED_SAVED_OBSERVATION','SEED','CONTINUOUS_CHECKS_PASSED','UNVERIFIED_SHADOW','NOT_TRACKED'}
    for number,f in enumerate(frames):
        i=f['decode_index'];s=source[i]
        if i in seen or i<=previous:raise ValueError('unordered/duplicate trace identity')
        seen.add(i);previous=i;status=f['candidate_status']
        if status not in allowed:raise ValueError('unknown candidate status')
        candidate=xy(f.get('candidate_xy'));accepted=xy(f.get('accepted_xy'))
        if status=='NOT_TRACKED':point=None
        elif status in ('SELECTED_SAVED_OBSERVATION','SEED'):
            if i!=seed['decode_index']:raise ValueError('seed on wrong frame')
            point=accepted if accepted is not None else candidate
        elif status=='CONTINUOUS_CHECKS_PASSED':
            if f.get('continuous_accepted') is not True or f.get('step_accepted') is not True or accepted is None:
                raise ValueError('continuous point has inconsistent acceptance labels')
            point=accepted
        else:point=candidate
        p=bind(s['path'],s['sha256'])
        with Image.open(p) as im:w,h=im.size
        for q in (candidate,accepted):
            if q is not None and not(0<=q[0]<w and 0<=q[1]<h):
                # A diagnostic proposal may leave the image. Preserve numbers;
                # the canvas clips it rather than inventing a new position.
                if q==accepted:raise ValueError('accepted point outside image')
        target=assets/f'frame_{i:06d}{p.suffix.lower()}'
        with target.open('xb') as stream:stream.write(p.read_bytes())
        display.append(dict(**{k:v for k,v in f.items() if k not in ('image','display_xy')},
            image='report_assets/'+target.name,width=w,height=h,time_seconds=s.get('time_seconds'),display_xy=point,
            comparison_xy=xy(comparison['xy']) if i==comparison['decode_index'] else None))
    history=[];last_kind=None;last_size=None
    for f in display:
        kind='diagnostic' if f['candidate_status']=='UNVERIFIED_SHADOW' else 'continuous'
        q=f['display_xy'];size=(f['width'],f['height'])
        if q is None:
            last_kind=None
        else:
            if kind!=last_kind or size!=last_size:
                history.append(dict(kind=kind,points=[]))
            history[-1]['points'].append(q)
            last_kind=kind;last_size=size
        # Coordinates are historical pixel positions, not surface reprojections.
        f['history_segments']=[dict(kind=s['kind'],points=list(s['points'])) for s in history]
    def painted(frame):
        with Image.open(out/frame['image']) as im:im=im.convert('RGB')
        d=ImageDraw.Draw(im);q=frame['display_xy'];r=7
        for segment in frame['history_segments']:
            if len(segment['points'])>=2:
                d.line([tuple(p) for p in segment['points']],fill=(255,155,25) if segment['kind']=='diagnostic' else (40,255,70),width=2)
        if q is not None:
            color=(255,155,25) if frame['candidate_status']=='UNVERIFIED_SHADOW' else (40,255,70)
            x,y=q;d.ellipse((x-r,y-r,x+r,y+r),outline=color,width=3);d.line((x-11,y,x+11,y),fill=color,width=2);d.line((x,y-11,x,y+11),fill=color,width=2)
        if frame['comparison_xy'] is not None:
            x,y=frame['comparison_xy'];d.line((x-10,y-10,x+10,y+10),fill='red',width=4);d.line((x-10,y+10,x+10,y-10),fill='red',width=4)
        return im
    selected={round(j*(len(display)-1)/min(7,len(display)-1)) for j in range(min(8,len(display)))} if len(display)>1 else {0}
    alarm=next((i for i,f in enumerate(display) if f['candidate_status'] in ('UNVERIFIED_SHADOW','NOT_TRACKED')),None)
    if alarm is not None:selected.update(i for i in (alarm-1,alarm,alarm+1) if 0<=i<len(display))
    selected.add(0)
    selected.update(i for i,f in enumerate(display) if f['decode_index']==comparison['decode_index'])
    tiles=[];animation=[]
    for i,f in enumerate(display):
        im=painted(f);im.thumbnail((576,720))
        tile=Image.new('RGB',(576,im.height+58),'white');tile.paste(im,((576-im.width)//2,58));d=ImageDraw.Draw(tile)
        d.text((5,4),f'decode {f["decode_index"]} / time {f["time_seconds"]} s / {f["candidate_status"]}',fill='black')
        d.text((5,23),f'FB {f.get("fb_error")} NCC {f.get("zncc")} anchor {f.get("anchor_zncc")}',fill='black')
        d.text((5,41),'green=continuous gate / orange=diagnostic / red X=old comparison only',fill='black')
        animation.append(tile)
        if i in selected:tiles.append((i,tile))
    # Fixed GIF canvas and fixed playback delay; original timestamps stay in JSON.
    gifh=max(im.height for im in animation);gif=[]
    for im in animation:
        canvas=Image.new('RGB',(576,gifh),'white');canvas.paste(im,(0,0));gif.append(canvas)
    gif[0].save(out/'motion.gif',save_all=True,append_images=gif[1:],duration=120,loop=0,disposal=2)
    th=max(im.height for _,im in tiles);sheet=Image.new('RGB',(576*3,th*((len(tiles)+2)//3)),'white')
    for k,(_,im) in enumerate(tiles):sheet.paste(im,((k%3)*576,(k//3)*th))
    sheet.save(out/'contact_sheet.png')
    alarm_crop=None
    if alarm is not None:
        q=display[alarm]['display_xy']
        if q is None and alarm>0:q=display[alarm-1]['display_xy']
        if q is not None:
            indices=[i for i in (alarm-1,alarm,alarm+1) if 0<=i<len(display)]
            crop=Image.new('RGB',(384*len(indices),430),'white')
            for col,i in enumerate(indices):
                im=painted(display[i]);local=im.crop((q[0]-48,q[1]-48,q[0]+48,q[1]+48)).resize((384,384),Image.Resampling.NEAREST)
                crop.paste(local,(384*col,46));d=ImageDraw.Draw(crop)
                d.text((384*col+5,5),f'decode {display[i]["decode_index"]}',fill='black')
                d.text((384*col+5,24),display[i]['candidate_status'],fill='black')
            crop.save(out/'first_alarm_local.png');alarm_crop='first_alarm_local.png'
    limitations=['绿点表示保存的当前步及连续门通过，seed是指定起点；这些都不是物理身份或遮挡真值。',
        '橙点为报警后的UNVERIFIED_SHADOW诊断候选；不是已接受轨迹，也不将中断后预测、插值或虚线作为观测。',
        'NOT_TRACKED不画新点；红叉是旧cross-ID目标比较，仅在指定decode显示，从不作为算法seed。',
        '细线是历次图像像素坐标的历史，叠加于当前图仅为运动示意，不是向当前物体表面重投影。绿/橙段分开，NOT_TRACKED断线；不生成中断处观测。',
        '单点图像运动不是3D重建、相机标定、全局对应或准确性验收。GIF固定120ms仅便于查看；真实采集时间保留于逐帧JSON。']
    links=[out/'motion.gif',out/'contact_sheet.png']+[out/f['image'] for f in display]
    if alarm_crop:links.append(out/alarm_crop)
    summary=dict(schema='single-point-report-v1',status='REPORT_GENERATED_NOT_ACCEPTED',seed=seed,comparison=comparison,
        frame_count=len(display),candidate_status_counts={s:sum(f['candidate_status']==s for f in display) for s in sorted(allowed)},
        first_alarm_decode=None if alarm is None else display[alarm]['decode_index'],contact_sheet_decodes=[display[i]['decode_index'] for i in sorted(selected)],
        frames=display,first_alarm_local=alarm_crop,gif_duration_ms=120,geometry_accepted=False,track_identity_truth=False,limitations=limitations,
        source_bindings=[dict(path=p,sha256=h) for p,h in sorted(bound.items())],source_drift=[p for p,h in bound.items() if _sha(p)!=h],
        link_check=dict(checked=len(links)+2,missing=[str(p) for p in links+[out/'TRACE.json'] if not p.is_file()]))
    if summary['source_drift'] or summary['link_check']['missing']:raise ValueError('source/link drift')
    js=json.dumps(dict(frames=display),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>单点连续运动检查</title><style>body{font:17px/1.6 system-ui;margin:24px}aside{background:#fff0d5;padding:16px}.grid{display:grid;grid-template-columns:minmax(400px,2fr) minmax(240px,1fr);gap:18px}canvas{max-width:100%;border:1px solid #aaa}pre{white-space:pre-wrap}input{width:60%}</style><h1>单点连续运动：逐帧检查</h1><aside>绿色=指定seed或当前步与连续门通过；橙色=已报警后的诊断候选。它们不证明物理身份或遮挡真值。红叉仅在旧比较目标对应帧显示，不输入算法。未跟踪帧不补点、不插值。</aside><p><a href="SUMMARY.json">逐帧数据与hash</a> · <a href="TRACE.json">原TRACE</a> · <a href="motion.gif">运动GIF（固定120ms）</a> · <a href="contact_sheet.png">关键帧联系表</a></p><button id="play">播放</button><input id="slider" type="range" min="0" value="0" step="1"><span id="where"></span><div class="grid"><section><a id="original">打开真实原图</a><canvas id="main"></canvas></section><section><h3>实际图像局部放大（不修改坐标）</h3><canvas id="zoom" width="384" height="384"></canvas><pre id="stats"></pre></section></div><script>const D=PAYLOAD;const s=document.getElementById('slider');s.max=D.frames.length-1;let token=0,timer=null;function draw(){const t=++token,f=D.frames[+s.value];document.getElementById('where').textContent=`decode ${f.decode_index} / ${f.time_seconds}s / ${f.candidate_status}`;document.getElementById('original').href=f.image;document.getElementById('stats').textContent=JSON.stringify({candidate:f.candidate_xy,accepted:f.accepted_xy,continuous:f.continuous_accepted,step:f.step_accepted,reason:f.reason,FB:f.fb_error,NCC:f.zncc,anchor_NCC:f.anchor_zncc,displacement_px:f.displacement_px},null,2);const im=new Image();im.onload=()=>{if(t!==token)return;const c=document.getElementById('main');c.width=f.width;c.height=f.height;const x=c.getContext('2d');x.drawImage(im,0,0);function mark(ctx,q,color,cross){if(!q)return;ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();if(cross){ctx.moveTo(q[0]-10,q[1]-10);ctx.lineTo(q[0]+10,q[1]+10);ctx.moveTo(q[0]-10,q[1]+10);ctx.lineTo(q[0]+10,q[1]-10)}else{ctx.arc(q[0],q[1],7,0,2*Math.PI)}ctx.stroke()}mark(x,f.display_xy,f.candidate_status==='UNVERIFIED_SHADOW'?'#ff9b19':'#28ff46',false);mark(x,f.comparison_xy,'red',true);const z=document.getElementById('zoom').getContext('2d');z.clearRect(0,0,384,384);if(f.display_xy){const q=f.display_xy;z.imageSmoothingEnabled=false;z.drawImage(c,q[0]-48,q[1]-48,96,96,0,0,384,384)}else{z.fillStyle='#555';z.fillText('NOT_TRACKED: no new point/crop',10,30)}};im.src=f.image}s.oninput=draw;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放'}else{document.getElementById('play').textContent='暂停';timer=setInterval(()=>{s.value=(+s.value+1)%D.frames.length;draw()},120)}};draw();</script><h2>联系表</h2><a href="contact_sheet.png"><img width="100%" src="contact_sheet.png"></a><h2>边界</h2>LIMITATIONS'''
    page=page.replace("x.drawImage(im,0,0);function mark", "x.drawImage(im,0,0);for(const segment of f.history_segments){if(segment.points.length<2)continue;x.strokeStyle=segment.kind==='diagnostic'?'#ff9b19':'#28ff46';x.lineWidth=2;x.beginPath();segment.points.forEach((q,i)=>i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1]));x.stroke()}function mark")
    page=page.replace('<h2>联系表</h2>','<p>细线仅是图像坐标历史，非当前物体表面重投影；绿/橙分段，NOT_TRACKED断线。</p>'+('<h2>首报警前后：同一像素局部窗口</h2><a href="first_alarm_local.png"><img width="100%" src="first_alarm_local.png"></a>' if alarm_crop else '')+'<h2>联系表（含首报警前后帧）</h2>')
    page=page.replace('PAYLOAD',js).replace('LIMITATIONS','<ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in limitations)+'</ul>')
    (out/'index.html').write_text(page,encoding='utf-8')
    with (out/'SUMMARY.json').open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
    return summary


build=render
