"""Saved-byte comparison of balanced control and gap-tolerant observations."""
from pathlib import Path
from collections import Counter
import html
import json

from .balanced_point_report import _read, _sha, _stats, _coverage

ARMS=('control','gap_tolerant')
LABELS={'control':'旧分区连续跟踪','gap_tolerant':'允许短暂中断后恢复'}
STAGES=('candidate','eligible','verified','model')


def build(output:Path)->dict:
    """Render only saved measurements; no matching, reconstruction or torch."""
    import numpy as np
    from PIL import Image,ImageDraw
    out=Path(output).resolve(strict=True)
    if (out/'SUMMARY.json').exists() or (out/'index.html').exists():raise ValueError('fresh report required')
    expected={}
    def bind(path,digest=None):
        path=Path(path).resolve(strict=True);h=_sha(path)
        if digest is not None and h!=digest:raise ValueError('source binding differs: '+str(path))
        if str(path) in expected and expected[str(path)]!=h:raise ValueError('source changed during read')
        expected[str(path)]=h;return path
    def read(path):return _read(bind(path))
    inputs=read(out/'INPUTS.json');plan=read(out/'PLAN.json');run=read(out/'RESULT.json');tracking=read(out/'TRACKING_RESULT.json')
    control=Path(plan['control_run']).resolve(strict=True)
    for ref in inputs.get('bindings',[]):
        p=bind(ref['path'],ref['sha256'])
        if p.stat().st_size!=ref['bytes']:raise ValueError('source byte count differs')
    frames=sorted(inputs['frames'],key=lambda f:f['input_index']);byindex={f['input_index']:f for f in frames}
    if len(byindex)!=len(frames):raise ValueError('duplicate source input indices')
    payload={arm:{i:dict(candidate=[],eligible=[],verified=[],model=[],recovered_candidate=[],recovered_eligible=[],
        recovered_verified=[],recovered_model=[],component=None,component_options=[]) for i in byindex} for arm in ARMS}
    models={};missing={};actual_options={};unique_registered={}
    for arm,folder in [('control',control/'balanced'),('gap_tolerant',out/'gap_tolerant')]:
        tp=folder/'TRACKS.json';rp=folder/'native/RESULT.json';pp=folder/'native/PLAN.json'
        if not tp.is_file():missing[arm]='TRACKS_MISSING';models[arm]=[];unique_registered[arm]=[];continue
        tracks=read(tp)
        if tracks.get('schema')!='balanced-object-observations-v1':raise ValueError('unsupported observation contract')
        obs={};counts=Counter()
        for frame in tracks['frames']:
            i=frame['input_index'];source=byindex[i]
            if i in obs:raise ValueError('repeated sampled frame cannot increase track support')
            if source['original_mask_status']!='OK' or frame['frame_id']!=source['frame_id'] or frame['decode_index']!=source['decode_index']:
                raise ValueError('sample/mask identity mismatch')
            if frame['image_sha256']!=source['image_sha256'] or frame['mask_sha256']!=source['mask_sha256']:
                raise ValueError('control and new frame bytes differ')
            current={}
            for o in frame['observations']:
                if o.get('predicted') is True or o.get('measured') is False or o.get('observation_type')=='prediction':
                    raise ValueError('predictions cannot enter measured observation report')
                tid=o['track_id']
                if type(tid) is not int or tid in current:raise ValueError('duplicate/invalid sample track')
                xy=np.asarray(o['xy'],float)
                if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('invalid observed xy')
                current[tid]=dict(xy=(xy+.5).tolist(),recovered=bool(o.get('recovered',False)))
                counts[tid]+=1
            obs[i]=current
        for i,current in obs.items():
            item=payload[arm][i]
            for tid,o in current.items():
                item['candidate'].append(o['xy'])
                if o['recovered']:item['recovered_candidate'].append(o['xy'])
                if counts[tid]>=3:
                    item['eligible'].append(o['xy'])
                    if o['recovered']:item['recovered_eligible'].append(o['xy'])
        if not rp.is_file() or not pp.is_file():missing[arm]='NATIVE_RESULT_MISSING';models[arm]=[];unique_registered[arm]=[];continue
        native=read(rp);nativeplan=read(pp);actual_options[arm]=native.get('actual_options')
        for ref in nativeplan.get('source_bindings',[]):bind(ref['path'],ref['sha256'])
        ledger={f['database_image_id']:f for f in nativeplan['frames']};verified={did:set() for did in ledger}
        for pair in native.get('pairs',[]):
            for a,b in pair.get('inlier_feature_rows',[]):verified[pair['image_a']].add(a);verified[pair['image_b']].add(b)
        for did,rows in verified.items():
            frame=ledger[did];i=frame['input_index']
            if frame['frame_id']!=byindex[i]['frame_id']:raise ValueError('native ledger identity mismatch')
            for row in sorted(rows):
                tid=frame['track_ids'][row];o=obs[i][tid]
                # Reproduce the adapter's exact float64 +0.5 -> float32 storage conversion.
                stored=np.asarray(o['xy'],np.float32).astype(float).tolist()
                payload[arm][i]['verified'].append(stored)
                if o['recovered']:payload[arm][i]['recovered_verified'].append(stored)
        models[arm]=native.get('models',[])
        unique_registered[arm]=sorted({i for m in models[arm] for i in m['registered_input_indices']})
        if len(unique_registered[arm])!=native['unique_registered_count']:raise ValueError('unique registered count mismatch')
        for model in models[arm]:
            for view in model.get('views',[]):
                i=view['input_index'];item=payload[arm][i];pts=view['native_xy'];mid=str(model['model_id'])
                item['component_options'].append(dict(model_id=mid,observations=len(pts)))
                if len(pts)>len(item['model']) or (len(pts)==len(item['model']) and (item['component'] is None or mid<item['component'])):
                    item['component']=mid;item['model']=pts;item['recovered_model']=[]
                    for o in view.get('observations',[]):
                        tid=o['candidate_track_id']
                        if tid in obs.get(i,{}) and obs[i][tid]['recovered']:item['recovered_model'].append(o['native_xy'])
    eventpath=out/'recovery_events.jsonl';event_counts=Counter();event_sample=[];event_count=0
    if eventpath.is_file():
        bind(eventpath)
        with eventpath.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():continue
                event=json.loads(line);event_count+=1
                kind=str(event.get('event',event.get('kind',event.get('status','UNSPECIFIED'))));event_counts[kind]+=1
                if len(event_sample)<30:event_sample.append(event)
    assets=out/'report_assets';assets.mkdir(exist_ok=False);links=[];viewframes=[];metrics={arm:[] for arm in ARMS}
    for frame in frames:
        i=frame['input_index'];ip=bind(frame['image_path'],frame['image_sha256'])
        with Image.open(ip) as im:base=im.convert('RGB')
        mask=None;bbox=None
        if frame['original_mask_status']=='OK':
            mp=bind(frame['mask_path'],frame['mask_sha256'])
            with Image.open(mp) as im:mask=np.asarray(im)>0
            y,x=np.where(mask);bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
            draw=ImageDraw.Draw(base);x0,y0,x1,y1=bbox
            for j in range(9):
                xx=x0+j*(x1-x0)/8;yy=y0+j*(y1-y0)/8
                draw.line((xx,y0,xx,y1),fill=(80,160,80));draw.line((x0,yy,x1,yy),fill=(80,160,80))
            edge=mask.copy();edge[1:-1,1:-1]&=~(mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:])
            image=np.asarray(base).copy();ey,ex=np.where(edge);image[ey,ex]=[0,255,255];base=Image.fromarray(image)
        name=f'frame_{i:06d}.jpg';base.save(assets/name,quality=88);links.append(assets/name)
        vf=dict(input_index=i,decode_index=frame['decode_index'],frame_id=frame['frame_id'],time_seconds=frame.get('time_seconds'),
            mask_status=frame['original_mask_status'],width=frame['width'],height=frame['height'],image='report_assets/'+name,arms={})
        for arm in ARMS:
            item=payload[arm][i];m={stage:None if mask is None else dict(points=len(item[stage]),recovered_points=len(item['recovered_'+stage]),grid=_coverage(item[stage],mask)) for stage in STAGES}
            vf['arms'][arm]=dict(**item,metrics=m)
            if mask is not None:metrics[arm].append(dict(input_index=i,**m,selected_component=item['component']))
        viewframes.append(vf)
    # Static 2 arms x 4 stages, two example frames from the control's largest component.
    ranked=sorted(models.get('control',[]),key=lambda m:(-m['registered_views'],-m['points'],str(m['model_id'])))
    examples=sorted(ranked[0]['registered_input_indices'])[:2] if ranked else []
    static=None
    if examples:
        canvases=[]
        for i in examples:
            vf=next(f for f in viewframes if f['input_index']==i)
            for arm in ARMS:
                row=[]
                for stage in STAGES:
                    with Image.open(assets/f'frame_{i:06d}.jpg') as im:base=im.convert('RGB')
                    c=Image.new('RGB',(base.width,base.height+45),'white');c.paste(base,(0,45));d=ImageDraw.Draw(c)
                    item=payload[arm][i];m=vf['arms'][arm]['metrics'][stage];g=m['grid']['8']
                    d.text((5,4),f'input{i} {arm}/{stage}: {len(item[stage])}; grid8 {g["occupied"]}/{g["nonempty_mask_cells"]}',fill='black')
                    for x,y in item[stage]:d.ellipse((x-2,y+43,x+1,y+46),fill=(255,220,0))
                    for x,y in item['recovered_'+stage]:d.ellipse((x-3,y+42,x+2,y+47),fill=(255,80,100))
                    row.append(c)
                canvases.append(row)
        w,h=canvases[0][0].size;picture=Image.new('RGB',(4*w,len(canvases)*h),'white')
        for y,row in enumerate(canvases):
            for x,c in enumerate(row):picture.paste(c,(x*w,y*h))
        path=assets/'stage_comparison.png';picture.save(path);links.append(path);static='report_assets/'+path.name
    modelsummary={arm:[{k:m.get(k) for k in ('model_id','registered_views','points','registered_input_indices','cameras')} for m in models.get(arm,[])] for arm in ARMS}
    summary=dict(schema='gap-point-report-v1',status='REPORT_GENERATED_NOT_ACCEPTED',run_status=run.get('status'),control_run=str(control),
        frames=len(frames),mask_status_counts=dict(Counter(f['original_mask_status'] for f in frames)),models=modelsummary,
        unique_registered_inputs=unique_registered,unique_registered_counts={arm:len(v) for arm,v in unique_registered.items()},
        per_frame_metrics=metrics,coverage_distributions={arm:{stage:{str(n):_stats([m[stage]['grid'][str(n)]['fraction'] for m in metrics[arm]]) for n in (4,8,12)} for stage in STAGES} for arm in ARMS},
        recovery_events=dict(path='recovery_events.jsonl' if eventpath.is_file() else None,count=event_count,kinds=dict(event_counts),first_30=event_sample),
        native_missing=missing,actual_options=actual_options,static_comparison=static,geometry_accepted=False,coverage_accepted=False,scale_status='unknown',
        strongest_component_policy='Per frame maximum observation count; lexical model ID tie-break. Never combine components; different frames may display different components.',
        limitations=['Only actual TRACKS observations are counted; explicit predictions or nonmeasurements are refused. Recovery labels do not prove physical identity.',
            'Eligibility counts at least three different saved sampled frames, not intermediate predictions or recovery attempts.',
            'Verified means union of unique endpoint feature rows across native verified pairs: 2D support, not 3D.',
            'Model support is per-frame strongest single component. Components and camera parameters remain unaccepted; no coordinate/point fusion.',
            '8x8 image-mask grid coverage is not physical surface coverage. 4/12 sensitivities are saved; no invented acceptance threshold.',
            'NO_MATCH frames remain visible and empty; no masks, observations or holes are fabricated.'],
        source_bindings=[dict(path=p,sha256=h) for p,h in sorted(expected.items())],source_drift=[p for p,h in expected.items() if _sha(p)!=h],
        link_check=dict(checked=len(links)+1+int(eventpath.is_file()),missing=[str(p) for p in links if not p.is_file()]))
    js=json.dumps(dict(frames=viewframes,labels=LABELS),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    modelhtml=''.join('<h3>'+html.escape(LABELS[arm])+'</h3><pre>'+html.escape(json.dumps(modelsummary[arm],ensure_ascii=False,indent=2))+'</pre>' for arm in ARMS)
    static_html='' if static is None else '<h2>旧最大分量样本的四阶段对照</h2><a href="'+static+'"><img style="width:100%" src="'+static+'"></a>'
    page='''<!doctype html><meta charset="utf-8"><title>短暂中断恢复对照</title><style>body{font:16px/1.6 system-ui;margin:24px}aside{background:#fff0ce;padding:16px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}canvas{width:100%}input{width:60%}pre{white-space:pre-wrap}</style>
<h1>短暂跟踪中断恢复：逐帧证据</h1><aside>黄点为保存的真实候选/观测，粉红点为带recovered标记的已测量位置；未画预测。青色mask轮廓、绿色mask bbox 8×8网格。恢复事件不等于成功观测，候选不等于三维；所有外形、相机参数与物理尺度均未验收。</aside>
<p><a href="SUMMARY.json">统计、来源hash与分量</a> · EVENT_LINK</p><input id="frame" type="range" min="0" value="0" step="1"><span id="which"></span><p><select id="stage"><option value="candidate">二维候选</option><option value="eligible">至少3真采样候选</option><option value="verified">原生验证内点</option><option value="model">单一分量3D观测</option></select></p><div id="cols" class="cols"></div>
<script>const D=PAYLOAD;let generation=0;const slider=document.getElementById('frame');slider.max=D.frames.length-1;function draw(){const token=++generation,f=D.frames[+slider.value],stage=document.getElementById('stage').value;document.getElementById('which').textContent=`input ${f.input_index}, decode ${f.decode_index}, ${f.time_seconds}s, mask ${f.mask_status}`;document.getElementById('cols').replaceChildren();for(const arm of ['control','gap_tolerant']){const a=f.arms[arm],m=a.metrics[stage],div=document.createElement('section'),h=document.createElement('h3'),p=document.createElement('p'),c=document.createElement('canvas');h.textContent=D.labels[arm];p.textContent=`${a[stage].length} 点，恢复标记 ${a['recovered_'+stage].length}，8×8 ${m?m.grid['8'].occupied+'/'+m.grid['8'].nonempty_mask_cells:'N/A'}，单一model ${a.component??'无'}，分量 ${JSON.stringify(a.component_options)}`;c.width=f.width;c.height=f.height;div.append(h,p,c);document.getElementById('cols').append(div);const im=new Image();im.onload=()=>{if(token!==generation)return;const x=c.getContext('2d');x.drawImage(im,0,0);for(const [points,color,r] of [[a[stage],'#ffe000',2],[a['recovered_'+stage],'#ff5064',3]]){x.fillStyle=color;for(const q of points){x.beginPath();x.arc(q[0]-.5,q[1]-.5,r,0,Math.PI*2);x.fill()}}};im.src=f.image}}slider.oninput=draw;document.getElementById('stage').onchange=draw;draw();</script>STATIC<h2>各原生分量（不可合并）</h2>MODELS<h2>统计解释</h2><p>格子分母只含非空mask像素的mask bbox网格；模型覆盖中位数0可能只是多数帧未注册，不表示整个任务没有3D点。支持更广不等于物理对应正确，低误差不等于外形准确。</p>'''
    eventlink='<a href="recovery_events.jsonl">逐步恢复事件原账本</a>' if eventpath.is_file() else '恢复事件账本不可用'
    page=page.replace('PAYLOAD',js).replace('EVENT_LINK',eventlink).replace('STATIC',static_html).replace('MODELS',modelhtml)
    if summary['source_drift'] or summary['link_check']['missing']:raise ValueError('report source/link validation failed')
    (out/'index.html').write_text(page,encoding='utf-8')
    with (out/'SUMMARY.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return dict(status=summary['status'],report=str(out/'index.html'),summary=str(out/'SUMMARY.json'),geometry_accepted=False)
