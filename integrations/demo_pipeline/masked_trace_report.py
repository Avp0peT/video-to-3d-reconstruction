"""Offline, hash-bound three-arm report. No tracking or segmentation occurs here."""
from pathlib import Path
import json
import math
from .balanced_point_report import _read, _sha


def render(output):
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    out=Path(output).resolve(strict=True)
    products=('index.html','SUMMARY.json','report_assets','comparison.gif','first_failure_contact.png')
    if any((out/name).exists() for name in products):
        raise ValueError('fresh masked trace report required')
    bound={}
    def bind(path,digest=None):
        p=Path(path)
        if not p.is_absolute(): p=out/p
        p=p.resolve(strict=True); h=_sha(p)
        if digest is not None and h!=digest: raise ValueError('source hash mismatch: '+str(p))
        if str(p) in bound and bound[str(p)]!=h: raise ValueError('source changed: '+str(p))
        bound[str(p)]=h
        return p
    def read(name): return _read(bind(out/name))
    inputs=read('INPUTS.json'); masks=read('MASKS.json')
    traces=[read(name) for name in ('BASELINE_TRACE.json','TRACE_MASKED.json','TRACE_MOTION.json')]
    maskrows=masks.get('frames',masks.get('masks',[]))
    maskmap={r['decode_index']:r for r in maskrows}
    if len(maskmap)!=len(maskrows): raise ValueError('duplicate mask frame')
    frames=inputs['frames']; indices=[r['decode_index'] for r in frames]
    if not frames or len(set(indices))!=len(indices): raise ValueError('empty/duplicate input frames')
    if any(b!=a+1 for a,b in zip(indices,indices[1:])): raise ValueError('consecutive frames required')
    maps=[]
    for trace in traces:
        rows=trace['frames']
        if [r['decode_index'] for r in rows]!=indices: raise ValueError('trace/input frame identity mismatch')
        maps.append({r['decode_index']:r for r in rows})
    assets=out/'report_assets'; assets.mkdir()
    def xy(q):
        if q is None: return None
        if len(q)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in q): raise ValueError('invalid coordinate')
        return list(map(float,q))
    def inside(a,q):
        if a is None or q is None: return None
        x,y=[math.floor(v+.5) for v in q]
        return bool(0<=y<a.shape[0] and 0<=x<a.shape[1] and a[y,x])
    display=[]; animation=[]; stats=[dict(accepted=0,accepted_outside=0,candidate_outside=0,mask_unknown=0) for _ in traces]
    labels=['A old baseline / red diagnostic shadow','B current mask only','C mask + bounded velocity init']
    for source in frames:
        i=source['decode_index']; p=bind(source['path'],source['sha256'])
        with Image.open(p) as im: raw=im.convert('RGB')
        w,h=raw.size; target=assets/f'frame_{i:06d}.png'; raw.save(target)
        m=maskmap.get(i,{}); status=m.get('status',m.get('mask_status','UNKNOWN'))
        if m.get('raw_sha256') is not None and m['raw_sha256']!=source['sha256']:
            raise ValueError('mask bound to a different decoded raw image')
        if m.get('raw_path') is not None and bind(m['raw_path'],m.get('raw_sha256'))!=p:
            raise ValueError('mask raw path identity mismatch')
        mp=m.get('path',m.get('mask_path')); digest=m.get('sha256',m.get('mask_sha256'))
        a=None; overlay=None; maskimage=None
        if mp is not None:
            if digest is None: raise ValueError('mask hash required')
            path=bind(mp,digest)
            with Image.open(path) as im: arr=np.asarray(im)
            if arr.ndim!=2 or arr.shape!=(h,w) or not np.isin(arr,[0,1,255]).all(): raise ValueError('binary mask shape/value mismatch')
            a=arr>0
            binary=Image.fromarray(a.astype('uint8')*255)
            edge=np.asarray(binary)!=np.asarray(binary.filter(ImageFilter.MinFilter(3)))
            rgba=np.zeros((h,w,4),np.uint8); rgba[edge]=[30,255,80,170]
            overlay=Image.fromarray(rgba); maskimage=f'report_assets/mask_{i:06d}.png'; overlay.save(out/maskimage)
        observed=(a is not None and m.get('current_observed') is True)
        arms=[]; tiles=[]
        for arm,rows in enumerate(maps):
            row=rows[i]; candidate=xy(row.get('candidate_xy')); accepted=xy(row.get('accepted_xy'))
            ai=inside(a,accepted); ci=inside(a,candidate)
            if accepted is not None:
                stats[arm]['accepted']+=1
                if ai is False: stats[arm]['accepted_outside']+=1
                if arm and (not observed or ai is not True): raise ValueError('new accepted point without observed current-mask support')
            if ci is False: stats[arm]['candidate_outside']+=1
            if not observed: stats[arm]['mask_unknown']+=1
            point=candidate if arm==0 else accepted
            # An unaccepted new proposal is disclosed in metadata, never drawn as a valid point.
            im=raw.convert('RGBA')
            if overlay is not None: im=Image.alpha_composite(im,overlay)
            im=im.convert('RGB'); d=ImageDraw.Draw(im)
            if point is not None:
                x,y=point; color=(255,35,35) if arm==0 else ((50,190,255) if arm==1 else (255,210,25))
                r=3 if arm==0 and row.get('candidate_status')=='UNVERIFIED_SHADOW' else 6
                d.ellipse((x-r,y-r,x+r,y+r),outline=color,width=2)
            im.thumbnail((320,480)); tile=Image.new('RGB',(320,im.height+70),'white'); tile.paste(im,((320-im.width)//2,70))
            d=ImageDraw.Draw(tile); d.text((4,3),labels[arm],fill='black'); d.text((4,20),f'raw {i} mask {status}' + ('' if observed else ' / UNKNOWN'),fill='black' if observed else 'red')
            d.text((4,37),str(row.get('reason',''))[:46],fill='black')
            d.text((4,54),f'FB {row.get("fb_error")} init {row.get("motion",{}).get("attempted",False)}',fill='black')
            tiles.append(tile)
            arms.append(dict(row=row,draw_xy=point,candidate_inside_current_mask=ci,accepted_inside_current_mask=ai))
        canvas=Image.new('RGB',(960,max(t.height for t in tiles)),'white')
        for k,tile in enumerate(tiles): canvas.paste(tile,(k*320,0))
        animation.append(canvas)
        display.append(dict(decode_index=i,time_seconds=source.get('time_seconds'),image='report_assets/'+target.name,width=w,height=h,mask_image=maskimage,mask_status=status,mask_source=m,mask_observed=observed,arms=arms))
    height=max(t.height for t in animation); gif=[]
    for tile in animation:
        fixed=Image.new('RGB',(960,height),'white'); fixed.paste(tile,(0,0)); gif.append(fixed)
    gif[0].save(out/'comparison.gif',save_all=True,append_images=gif[1:],duration=120,loop=0,disposal=2)
    failures=[t.get('first_failure') for t in traces]
    alarm=[f.get('decode_index') for f in failures if isinstance(f,dict) and f.get('decode_index') in indices]
    chosen={0}
    for i in alarm:
        j=indices.index(i); chosen.update(k for k in (j-1,j,j+1) if 0<=k<len(indices))
    sheet=Image.new('RGB',(960,height*len(chosen)),'white')
    for row,j in enumerate(sorted(chosen)): sheet.paste(gif[j],(0,row*height))
    sheet.save(out/'first_failure_contact.png')
    drift=[p for p,h in bound.items() if _sha(p)!=h]
    if drift: raise ValueError('source changed during rendering: '+repr(drift))
    summary=dict(schema='masked-trace-report-v1',status='REPORT_GENERATED_NOT_IDENTITY_ACCEPTED',frame_count=len(frames),arm_statistics=stats,first_failures=failures,physical_identity_verified=False,geometry_accepted=False,source_bindings=[dict(path=p,sha256=h) for p,h in sorted(bound.items())],source_drift=drift,meaning='Mask constraints may reduce outside candidates; containment and image checks do not establish physical identity. No history is projected onto the current image. Missing/NO_MATCH masks remain unknown.')
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    payload=json.dumps(dict(frames=display,labels=labels),ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>Current-mask tracking comparison</title><style>body{font:16px system-ui;margin:20px;background:#fafafa}aside{padding:14px;background:#fff0ca}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}canvas{width:100%;border:1px solid #888}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}input{width:60%}.unknown{color:#b00;font-weight:bold}</style><h1>当前掩膜与有界运动初始化：同帧三路比较</h1><aside>红色旧 shadow 仅为错误示范，不能当新有效点。B 蓝点、C 黄点仅显示 accepted_xy，逐帧核查当前真实 mask 内；绿色半透明为 mask 轮廓。UNKNOWN / NO_MATCH 不代表目标内。包含关系及图像检查不证明物理身份。当前图只画当前点，不叠加历史轨迹。</aside><p><a href="SUMMARY.json">结论与源hash</a> · <a href="comparison.gif">三路GIF</a> · <a href="first_failure_contact.png">首失败前后</a></p><button id="play">播放</button> <input type="range" id="slider" min="0" value="0"><span id="where"></span><div class="grid" id="panels"></div><script>const D=PAYLOAD,s=document.getElementById('slider');s.max=D.frames.length-1;let token=0,timer=null;const panels=document.getElementById('panels');for(let k=0;k<3;k++){const e=document.createElement('section');e.innerHTML='<h3>'+D.labels[k]+'</h3><canvas id="c'+k+'"></canvas><p>局部放大（当前点；无点则无局部）</p><canvas id="z'+k+'" width="320" height="320"></canvas><pre id="t'+k+'"></pre>';panels.append(e)}function load(src){return new Promise((ok,no)=>{let im=new Image();im.onload=()=>ok(im);im.onerror=()=>no(Error(src));im.src=src})}async function draw(){const ticket=++token,f=D.frames[+s.value];document.getElementById('where').textContent='raw '+f.decode_index+' / '+f.time_seconds+'s / mask '+f.mask_status;const im=await load(f.image),mask=f.mask_image?await load(f.mask_image):null;if(ticket!==token)return;for(let k=0;k<3;k++){let a=f.arms[k],c=document.getElementById('c'+k);c.width=f.width;c.height=f.height;let x=c.getContext('2d');x.drawImage(im,0,0);if(mask)x.drawImage(mask,0,0);let q=a.draw_xy;if(q){x.strokeStyle=k===0?'#ff2323':k===1?'#32beff':'#ffd219';x.lineWidth=3;x.beginPath();x.arc(q[0],q[1],k===0?3:6,0,Math.PI*2);x.stroke()}let z=document.getElementById('z'+k).getContext('2d');z.clearRect(0,0,320,320);if(q){z.imageSmoothingEnabled=false;z.drawImage(c,q[0]-40,q[1]-40,80,80,0,0,320,320)}document.getElementById('t'+k).textContent=JSON.stringify({mask_status:f.mask_status,mask_source:f.mask_source,inside:a.accepted_inside_current_mask,candidate_inside:a.candidate_inside_current_mask,...a.row},null,2)}}s.oninput=()=>draw().catch(alert);document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{s.value=(+s.value+1)%D.frames.length;draw().catch(alert)},120)};draw().catch(alert);</script>'''
    (out/'index.html').write_text(page.replace('PAYLOAD',payload),encoding='utf-8')
    return summary


build=render
