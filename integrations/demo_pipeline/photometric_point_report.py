"""Read-only three-way report for saved photometric-gap tracking results."""
from pathlib import Path
from collections import Counter
import html
import json

from .balanced_point_report import _read, _sha, _stats, _coverage

ARMS=('seed_control','gap_control','photometric')
LABELS={'seed_control':'辅助对照：分区连续跟踪','gap_control':'主要对照：短中断恢复','photometric':'新实验：光照稳健支持'}
STAGES=('candidate','eligible','model')


def _photometric_fields(value,path='tracking'):
    """Keep names and source paths; never confuse event counters with observations."""
    result=[]
    if isinstance(value,dict):
        for k,v in value.items():
            key=path+'.'+str(k)
            if k in ('photometric_accepted','photometric_changed') and isinstance(v,(int,float,dict,list)):
                result.append(dict(source_field=key,value=v))
            elif isinstance(v,dict):result.extend(_photometric_fields(v,key))
    return result


def build(output:Path)->dict:
    """Consume completed results and render actual saved coordinates only."""
    import numpy as np
    from PIL import Image,ImageDraw
    out=Path(output).resolve(strict=True)
    if (out/'SUMMARY.json').exists() or (out/'index.html').exists():raise ValueError('fresh report required')
    expected={}
    def bind(path,digest=None):
        path=Path(path).resolve(strict=True);value=_sha(path)
        if digest is not None and value!=digest:raise ValueError('source hash differs: '+str(path))
        if str(path) in expected and expected[str(path)]!=value:raise ValueError('source changed during report')
        expected[str(path)]=value;return path
    def read(path):return _read(bind(path))
    inputs=read(out/'INPUTS.json');plan=read(out/'PLAN.json');run=read(out/'RESULT.json');tracking=read(out/'TRACKING_RESULT.json')
    control=Path(plan['control_run']).resolve(strict=True);seed=Path(plan['seed_control_run']).resolve(strict=True)
    for ref in inputs.get('bindings',[]):
        path=bind(ref['path'],ref['sha256'])
        if path.stat().st_size!=ref['bytes']:raise ValueError('source byte count differs')
    frames=sorted(inputs['frames'],key=lambda f:f['input_index']);byindex={f['input_index']:f for f in frames}
    if len(byindex)!=len(frames):raise ValueError('duplicate frame identities')
    folders={'seed_control':seed/'balanced','gap_control':control/'gap_tolerant','photometric':out/'photometric'}
    data={arm:{i:dict(candidate=[],eligible=[],model=[],supported_candidate=[],supported_eligible=[],supported_model=[],
        changed_candidate=[],changed_eligible=[],changed_model=[],component=None,component_options=[]) for i in byindex} for arm in ARMS}
    model_lists={};native_status={};unique={};trackstats={};native_policies={}
    for arm in ARMS:
        folder=folders[arm];tracks_path=folder/'TRACKS.json';native_path=folder/'native/RESULT.json'
        if not tracks_path.is_file():
            native_status[arm]='TRACKS_MISSING';model_lists[arm]=[];unique[arm]=[];continue
        tracks=read(tracks_path)
        if tracks.get('schema')!='balanced-object-observations-v1':raise ValueError('unknown observations contract')
        observations={};frequency=Counter();history_support=history_changed=0;history_ids=set();current_only_ids=set()
        for frame in tracks['frames']:
            i=frame['input_index'];original=byindex[i]
            if i in observations or frame['frame_id']!=original['frame_id'] or frame['decode_index']!=original['decode_index']:
                raise ValueError('sample identity mismatch or duplicate')
            if original['original_mask_status']!='OK' or frame['image_sha256']!=original['image_sha256'] or frame['mask_sha256']!=original['mask_sha256']:
                raise ValueError('sample image/mask differs from controls')
            observed={}
            for o in frame['observations']:
                if o.get('predicted') is True or o.get('measured') is False or o.get('observation_type')=='prediction':
                    raise ValueError('prediction cannot count as observation')
                tid=o['track_id'];xy=np.asarray(o['xy'],float)
                if type(tid) is not int or tid in observed or xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('invalid observation')
                # Labels are histories if producers define them that way; never infer a new event from each label.
                support=bool(o.get('photometric_history',False));changed=bool(o.get('photometric_only_this_decode',False))
                observed[tid]=dict(xy=(xy+.5).tolist(),supported=support,changed=changed)
                frequency[tid]+=1;history_support+=int(support);history_changed+=int(changed)
                if support:history_ids.add(tid)
                if changed:current_only_ids.add(tid)
            observations[i]=observed
        for i,observed in observations.items():
            for tid,o in observed.items():
                item=data[arm][i];item['candidate'].append(o['xy'])
                if o['supported']:item['supported_candidate'].append(o['xy'])
                if o['changed']:item['changed_candidate'].append(o['xy'])
                if frequency[tid]>=3:
                    item['eligible'].append(o['xy'])
                    if o['supported']:item['supported_eligible'].append(o['xy'])
                    if o['changed']:item['changed_eligible'].append(o['xy'])
        trackstats[arm]=dict(sample_frames=len(observations),observations=sum(frequency.values()),tracks_with_three_samples=sum(n>=3 for n in frequency.values()),
            photometric_history_sample_observations=history_support,photometric_only_this_decode_sample_observations=history_changed,
            sampled_photometric_history_unique_ids=len(history_ids),sampled_photometric_only_current_decode_unique_ids=len(current_only_ids),
            observation_fields=dict(support='photometric_history: ID had photometric-only support at/before this observation',
                changed='photometric_only_this_decode: current sample decode accepted only by photometric step, not coordinate-change count'),
            observation_label_semantics=tracks.get('photometric_label',tracks.get('photometric_support_label','Producer label semantics not supplied; do not interpret labels as current-frame events.')),
            tracking_variant=tracks.get('tracking_variant'))
        native_plan=folder/'native/PLAN.json'
        if not native_path.is_file():native_status[arm]='NATIVE_RESULT_MISSING';model_lists[arm]=[];unique[arm]=[];continue
        native=read(native_path);native_status[arm]=native.get('status');native_policies[arm]=native.get('actual_options')
        if native_plan.is_file():
            for ref in read(native_plan).get('source_bindings',[]):bind(ref['path'],ref['sha256'])
        model_lists[arm]=native.get('models',[])
        unique[arm]=sorted({i for m in model_lists[arm] for i in m['registered_input_indices']})
        if len(unique[arm])!=native['unique_registered_count']:raise ValueError('native unique registration count differs')
        for m in model_lists[arm]:
            for view in m.get('views',[]):
                i=view['input_index'];item=data[arm][i];points=view['native_xy'];mid=str(m['model_id'])
                if 'frame_id' in view and view['frame_id']!=byindex[i]['frame_id']:raise ValueError('native view identity differs')
                item['component_options'].append(dict(model_id=mid,observations=len(points)))
                if len(points)>len(item['model']) or (len(points)==len(item['model']) and (item['component'] is None or mid<item['component'])):
                    item['model']=points;item['component']=mid;item['supported_model']=[];item['changed_model']=[]
                    for o in view.get('observations',[]):
                        source=observations.get(i,{}).get(o['candidate_track_id'])
                        if source is not None:
                            if source['supported']:item['supported_model'].append(o['native_xy'])
                            if source['changed']:item['changed_model'].append(o['native_xy'])
    eventinfo={}
    for name in ('photometric_events.jsonl','recovery_events.jsonl'):
        path=out/name
        if not path.is_file():continue
        bind(path);kinds=Counter();first=[];n=0
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():continue
                event=json.loads(line);n+=1;kinds[str(event.get('kind',event.get('event',event.get('status','UNSPECIFIED'))))]+=1
                if len(first)<20:first.append(event)
        eventinfo[name]=dict(count=n,kinds=dict(kinds),first_20=first)
    differential=tracking.get('photometric_differential')
    differential_fields=('tested','raw_accepted','photometric_accepted','photometric_only','raw_only','common')
    differential_check=None
    if differential is not None:
        if any(type(differential.get(k)) is not int or differential[k]<0 for k in differential_fields):raise ValueError('invalid photometric differential counters')
        differential_check=(differential['raw_accepted']==differential['raw_only']+differential['common']
            and differential['photometric_accepted']==differential['photometric_only']+differential['common']
            and max(differential['raw_accepted'],differential['photometric_accepted'])<=differential['tested'])
        if not differential_check:raise ValueError('inconsistent photometric differential counts')
    assets=out/'report_assets';assets.mkdir(exist_ok=False);links=[];visual=[];metrics={arm:[] for arm in ARMS};offline=[]
    for number,ref in enumerate(plan.get('offline_evidence',[])):
        path=bind(ref['path'],ref['sha256'])
        if 'bytes' in ref and path.stat().st_size!=ref['bytes']:raise ValueError('offline evidence size differs')
        if path.stat().st_size>16<<20:raise ValueError('offline evidence copy exceeds 16MiB')
        target=assets/f'offline_{number:02d}_{path.name}';target.write_bytes(path.read_bytes());links.append(target)
        evidence=dict(source=str(path),path='report_assets/'+target.name,sha256=ref['sha256'],bytes=path.stat().st_size)
        if path.suffix.lower()=='.json':
            measured=read(path)
            if isinstance(measured.get('cases'),list):evidence.update(kind=measured.get('kind'),measurement_cases=measured['cases'])
        elif path.suffix.lower()=='.xml':
            import xml.etree.ElementTree as ET
            root=ET.parse(path).getroot();suites=[root] if root.tag=='testsuite' else list(root.findall('testsuite'))
            evidence['junit']=dict(tests=sum(int(s.get('tests','0')) for s in suites),failures=sum(int(s.get('failures','0')) for s in suites),errors=sum(int(s.get('errors','0')) for s in suites),skipped=sum(int(s.get('skipped','0')) for s in suites))
        offline.append(evidence)
    for f in frames:
        i=f['input_index'];ip=bind(f['image_path'],f['image_sha256'])
        with Image.open(ip) as im:image=im.convert('RGB')
        mask=None
        if f['original_mask_status']=='OK':
            mp=bind(f['mask_path'],f['mask_sha256'])
            with Image.open(mp) as im:mask=np.asarray(im)>0
            y,x=np.where(mask);x0,y0,x1,y1=int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1);d=ImageDraw.Draw(image)
            for j in range(9):
                xx=x0+j*(x1-x0)/8;yy=y0+j*(y1-y0)/8;d.line((xx,y0,xx,y1),fill=(80,160,80));d.line((x0,yy,x1,yy),fill=(80,160,80))
            edge=mask.copy();edge[1:-1,1:-1]&=~(mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:])
            a=np.asarray(image).copy();ey,ex=np.where(edge);a[ey,ex]=[0,255,255];image=Image.fromarray(a)
        path=assets/f'frame_{i:06d}.jpg';image.save(path,quality=88);links.append(path)
        vf=dict(input_index=i,frame_id=f['frame_id'],decode_index=f['decode_index'],time_seconds=f.get('time_seconds'),mask_status=f['original_mask_status'],
            width=f['width'],height=f['height'],image='report_assets/'+path.name,arms={})
        for arm in ARMS:
            item=data[arm][i];m={stage:None if mask is None else dict(points=len(item[stage]),grid=_coverage(item[stage],mask),
                photometric_history_observations=len(item['supported_'+stage]),photometric_only_this_decode_observations=len(item['changed_'+stage])) for stage in STAGES}
            vf['arms'][arm]=dict(**item,metrics=m)
            if mask is not None:metrics[arm].append(dict(input_index=i,**m,selected_component=item['component']))
        visual.append(vf)
    # Same chosen frames for every arm and stage. Select from the primary control, without task-specific IDs.
    ranked=sorted(model_lists.get('gap_control',[]),key=lambda m:(-m['registered_views'],-m['points'],str(m['model_id'])))
    example_indices=sorted(ranked[0]['registered_input_indices'])[:2] if ranked else [f['input_index'] for f in frames if f['original_mask_status']=='OK'][:2]
    static=[]
    for stage in STAGES:
        rows=[]
        for i in example_indices:
            vf=next(f for f in visual if f['input_index']==i);row=[]
            for arm in ARMS:
                with Image.open(assets/f'frame_{i:06d}.jpg') as im:base=im.convert('RGB')
                c=Image.new('RGB',(base.width,base.height+48),'white');c.paste(base,(0,48));d=ImageDraw.Draw(c);a=data[arm][i];m=vf['arms'][arm]['metrics'][stage];g=m['grid']['8']
                d.text((5,4),f'input{i} {arm}/{stage}: {len(a[stage])}; grid8 {g["occupied"]}/{g["nonempty_mask_cells"]}',fill='black')
                d.text((5,22),'yellow=measured; pink=photometric label; no accepted shape',fill='black')
                for points,color,r in [(a[stage],(255,220,0),2),(a['supported_'+stage],(255,80,110),3)]:
                    for x,y in points:d.ellipse((x-.5-r,y+47.5-r,x-.5+r,y+47.5+r),fill=color)
                row.append(c)
            rows.append(row)
        if rows:
            w,h=rows[0][0].size;montage=Image.new('RGB',(3*w,len(rows)*h),'white')
            for y,row in enumerate(rows):
                for x,c in enumerate(row):montage.paste(c,(x*w,y*h))
            path=assets/f'comparison_{stage}.png';montage.save(path);links.append(path);static.append(dict(stage=stage,path='report_assets/'+path.name,input_indices=example_indices))
    modelsummary={arm:[{k:m.get(k) for k in ('model_id','registered_views','points','registered_input_indices','cameras')} for m in model_lists.get(arm,[])] for arm in ARMS}
    literature=[dict(source='Сборка макета',printed_pages='61–62',pdf_pages='55–56',boundary='点集仿射对齐与ε整组对应检验；不是像素光照增益/偏置模型。'),
        dict(source='Козлов, Математическое моделирование зрительного восприятия, 1996',printed_page=331,pdf_page=12,
             boundary='平行射线投影和点标签一一对应是定义前提，不能由光照支持次数替代。')]
    summary=dict(schema='photometric-point-report-v1',status='REPORT_GENERATED_NOT_ACCEPTED',run_status=run.get('status'),
        control_run=str(control),seed_control_run=str(seed),mask_status_counts=dict(Counter(f['original_mask_status'] for f in frames)),
        tracking=trackstats,photometric_counters=_photometric_fields(tracking),photometric_differential=differential,photometric_differential_consistent=differential_check,
        photometric_differential_semantics='Per-step old-active coordinates and identical input pixels: diagnostic raw step versus normalized step. Raw diagnostic does not update its own tracker. Repeated steps are not unique identities, independent trajectories, geometry inliers or 3D contributions.',
        photometric_history_producer=dict(unique_ids=tracking.get('photometric_history_unique_ids'),observations=tracking.get('photometric_history_observations')),
        photometric_event_overviews=eventinfo,native_status=native_status,offline_evidence=offline,
        offline_evidence_status='BOUND_COPIES_LINKED' if offline else 'NOT_SUPPLIED_IN_PLAN',
        models=modelsummary,unique_registered_inputs=unique,unique_registered_counts={arm:len(v) for arm,v in unique.items()},
        per_frame_metrics=metrics,grid_fraction_distributions={arm:{stage:{str(n):_stats([m[stage]['grid'][str(n)]['fraction'] for m in metrics[arm]]) for n in (4,8,12)} for stage in STAGES} for arm in ARMS},
        actual_native_options=native_policies,static_comparisons=static,literature_boundaries=literature,
        geometry_accepted=False,coverage_accepted=False,scale_status='unknown',causal_improvement_claim=False,
        strongest_component_policy='Per frame most observed points, lexical model-ID tie; no cross-component union. Different frames may display different components.',
        limitations=['Photometric accepted/changed are producer counters or observation labels, not numbers of unique tracks, geometric inliers or 3D points.',
            'Raw diagnostic and normalized step share the current live seed population; this is not a separately evolving raw tracker or a cross-run track-ID comparison.',
            'Pre-normalization plus raw-image gain/bias residual validation is not joint motion/brightness optimization. Raw gap recovery keeps its existing gates.',
            'Known-transform periodic-texture negative control accepted 450 wrong matches in both raw and normalized variants; photometric support does not resolve whole-period aliasing.',
            'Support histories, if used by the producer, do not mean the current frame was corrected. Missing label semantics stay explicitly unknown.',
            'At least three different saved measured sample frames are required for eligible tracks; predictions are never observations.',
            'Grid occupancy is image support in nonempty mask-bbox cells, not surface coverage, physical identity or accepted shape. No invented pass threshold.',
            'Native models and estimated camera parameters remain unaccepted, independent components; successful completion is not causal improvement.',
            'This photometric tracking and perspective native geometry are engineering modules, not Kozlov original reconstruction algorithms.'],
        source_bindings=[dict(path=p,sha256=h) for p,h in sorted(expected.items())],source_drift=[p for p,h in expected.items() if _sha(p)!=h],
        link_check=dict(checked=len(links)+1+len(eventinfo),missing=[str(p) for p in links if not p.is_file()]))
    js=json.dumps(dict(frames=visual,labels=LABELS),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    modelhtml=''.join('<h3>'+html.escape(LABELS[arm])+'</h3><pre>'+html.escape(json.dumps(modelsummary[arm],ensure_ascii=False,indent=2))+'</pre>' for arm in ARMS)
    pics=''.join('<h3>'+html.escape(x['stage'])+'</h3><a href="'+x['path']+'"><img style="width:100%" src="'+x['path']+'"></a>' for x in static)
    counterhtml=html.escape(json.dumps(dict(per_step_differential=differential,producer_history=summary['photometric_history_producer']),ensure_ascii=False,indent=2))
    offlinehtml=''.join('<p><a href="'+e['path']+'">'+html.escape(Path(e['source']).name)+'</a> · SHA256 '+e['sha256']+'</p>' for e in offline)
    if not offline:offlinehtml='<p>PLAN未提供offline_evidence，未声称本报告已绑定离线测量表。</p>'
    for e in offline:
        if 'junit' in e:offlinehtml+='<p>离线测试记录：'+html.escape(str(e['junit']))+'；不等于真实视频验收。</p>'
        if 'measurement_cases' not in e:continue
        offlinehtml+='<p>已知变换合成测量，不是真实视频接受结论。</p><table border="1"><tr><th>case</th><th>raw接受/总数</th><th>normalized接受/总数</th><th>raw错误接受&gt;1px</th><th>normalized错误接受&gt;1px</th></tr>'
        for case in e['measurement_cases']:
            metrics_case=case.get('metrics',{});raw=metrics_case.get('raw',{});normalized=metrics_case.get('normalized',{})
            values=(case.get('synthetic_case'),f'{raw.get("accepted")}/{raw.get("total")}',f'{normalized.get("accepted")}/{normalized.get("total")}',raw.get('wrong_accepted_gt1px'),normalized.get('wrong_accepted_gt1px'))
            offlinehtml+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
        offlinehtml+='</table>'
    eventlinks=' · '.join('<a href="'+name+'">'+html.escape(name)+'</a>' for name in eventinfo)
    page='''<!doctype html><meta charset="utf-8"><title>光照稳健跟踪对照</title><style>body{font:16px/1.6 system-ui;margin:24px}.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}canvas{width:100%}aside{background:#fff0cd;padding:18px}input{width:60%}pre{white-space:pre-wrap}</style>
<h1>光照稳健支持：主要与辅助对照</h1><aside>光照支持次数≠独立轨迹数≠几何内点数≠三维点数。黄点为实际保存位置；粉色表示该ID在本帧或此前有过photometric-only支持，不代表本帧刚被修正。候选与三采样点仍为二维；model层为每帧最强单一分量的原生3D观测，未拼接、未质量验收。不同运行的出生ID可能不同，不跨run对track ID。</aside><p><a href="SUMMARY.json">精确统计与来源hash</a> EVENTS</p><input id="frame" type="range" min="0" value="0" step="1"><span id="which"></span><p><select id="stage"><option value="candidate">保存采样候选</option><option value="eligible">至少3真采样候选</option><option value="model">单一分量3D观测</option></select></p><div id="cols" class="cols"></div>
<script>const D=PAYLOAD;let version=0;const s=document.getElementById('frame');s.max=D.frames.length-1;function draw(){const token=++version,f=D.frames[+s.value],stage=document.getElementById('stage').value;document.getElementById('which').textContent=`input ${f.input_index}, decode ${f.decode_index}, ${f.time_seconds}s, mask ${f.mask_status}`;document.getElementById('cols').replaceChildren();for(const arm of ['seed_control','gap_control','photometric']){const a=f.arms[arm],m=a.metrics[stage],card=document.createElement('section'),h=document.createElement('h3'),p=document.createElement('p'),c=document.createElement('canvas');h.textContent=D.labels[arm];p.textContent=`${a[stage].length} 点; 8×8 ${m?m.grid['8'].occupied+'/'+m.grid['8'].nonempty_mask_cells:'N/A'}; 历史支持/本帧only ${a['supported_'+stage].length}/${a['changed_'+stage].length}; 单一model ${a.component??'无'}; 分量 ${JSON.stringify(a.component_options)}`;c.width=f.width;c.height=f.height;card.append(h,p,c);document.getElementById('cols').append(card);const im=new Image();im.onload=()=>{if(token!==version)return;const x=c.getContext('2d');x.drawImage(im,0,0);for(const [points,color,r] of [[a[stage],'#ffe000',2],[a['supported_'+stage],'#ff5070',3]]){x.fillStyle=color;for(const q of points){x.beginPath();x.arc(q[0]-.5,q[1]-.5,r,0,Math.PI*2);x.fill()}}};im.src=f.image}}s.oninput=draw;document.getElementById('stage').onchange=draw;draw();</script>
<h2>同输入单步差分（不是独立raw轨迹）</h2><p>每步对同一old-active原坐标和图像运行raw与normalized诊断，raw诊断不更新自己的tracker。tested/raw_accepted/photometric_accepted/photometric_only/raw_only/common是累计单步数量，不是独立轨迹或3D贡献；photometric_history是历史标签，photometric_only_this_decode才指当前decode只由新step接受。</p><pre>COUNTERS</pre><h2>离线测量与反例</h2>OFFLINE<p>方法是预归一化加原图gain/bias残差校验，不是joint optimization；raw gap仍保持旧门。周期纹理负对照旧新均接受450个错误匹配，光照稳健支持不能消除整周期歧义；也不能由本方法名称断定照明是实际视频失败主因。</p><h2>可检查同帧对照</h2>PICS<h2>所有独立分量及未验收相机参数</h2>MODELS<h2>论文与工程边界</h2><p>书稿《Сборка макета》印刷61–62页（PDF55–56页）讨论点集仿射对齐与ε整组核对；1996论文印刷331页（PDF12页）定义平行射线投影和点标签一一对应。这些前提不能由灰度增益、偏置或光照accepted次数代替。本模块是工程跟踪与透视重建证据报告，不是Kozlov原算法。没有给出因果改善或外形接受结论。</p>'''
    page=page.replace('PAYLOAD',js).replace('EVENTS',eventlinks).replace('COUNTERS',counterhtml).replace('OFFLINE',offlinehtml).replace('PICS',pics).replace('MODELS',modelhtml)
    if summary['source_drift'] or summary['link_check']['missing']:raise ValueError('report source or link drift')
    (out/'index.html').write_text(page,encoding='utf-8')
    with (out/'SUMMARY.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return dict(status=summary['status'],report=str(out/'index.html'),summary=str(out/'SUMMARY.json'),geometry_accepted=False)
