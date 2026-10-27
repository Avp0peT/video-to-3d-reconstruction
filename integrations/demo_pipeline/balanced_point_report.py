"""Automatic read-only comparison report for a completed balanced-point run."""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import os
import sqlite3

ARMS=('baseline','sift_control','balanced')
LABELS={'baseline':'旧原生 SIFT','sift_control':'同 SIFT 种子连续跟踪','balanced':'分区种子连续跟踪'}


def _read(path,limit=96<<20):
    path=Path(path)
    if path.stat().st_size>limit:raise ValueError('report metadata exceeds bound: '+str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def _stats(values):
    import numpy as np
    a=np.asarray(values,float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(n=0)


def _coverage(points,mask):
    import numpy as np
    y,x=np.where(mask);x0,y0=int(x.min()),int(y.min());w,h=int(x.max()+1-x0),int(y.max()+1-y0)
    result={}
    for n in (4,8,12):
        support=set(zip(((x+.5-x0)*n/w).astype(int).tolist(),((y+.5-y0)*n/h).astype(int).tolist()))
        occupied=set()
        for px,py in points:
            ix,iy=int(np.floor(px)),int(np.floor(py))
            if 0<=ix<mask.shape[1] and 0<=iy<mask.shape[0] and mask[iy,ix]:
                occupied.add((min(n-1,int((px-x0)*n/w)),min(n-1,int((py-y0)*n/h))))
        result[str(n)]=dict(occupied=len(occupied&support),nonempty_mask_cells=len(support),fraction=len(occupied&support)/len(support))
    return result


def build(output:Path)->dict:
    """Read saved run outputs, render HTML/PNG, and never run a native solver."""
    import numpy as np
    from PIL import Image,ImageDraw
    out=Path(output).resolve(strict=True)
    if (out/'SUMMARY.json').exists() or (out/'index.html').exists():raise ValueError('report output already exists')
    inputs=_read(out/'INPUTS.json');plan=_read(out/'PLAN.json');tracking=_read(out/'TRACKING_RESULT.json');run=_read(out/'RESULT.json')
    source_refs=inputs.get('bindings',[])
    expected={str(Path(ref['path'])):ref['sha256'] for ref in source_refs}
    for ref in source_refs:
        p=Path(ref['path'])
        if not p.is_file() or _sha(p)!=ref['sha256'] or p.stat().st_size!=ref['bytes']:
            raise ValueError('report input binding drift: '+str(p))
    frames=sorted(inputs['frames'],key=lambda f:f['input_index']);ok=[f for f in frames if f['original_mask_status']=='OK']
    byindex={f['input_index']:f for f in frames};indices=[f['input_index'] for f in frames]
    refs=[out/n for n in ('INPUTS.json','PLAN.json','TRACKING_RESULT.json','RESULT.json')]
    data={arm:{i:dict(candidate=[],eligible=None if arm=='baseline' else [],verified=[],model=[],component=None,component_options=[]) for i in indices} for arm in ARMS}
    model_lists={};native_missing={};links=[]
    baseline_path=Path(inputs['baseline_sfm']);baseline=_read(baseline_path);maps_path=baseline_path.parent/'frames_map.jsonl'
    maps=[json.loads(s) for s in maps_path.read_text(encoding='utf-8-sig').splitlines() if s.strip()];mapsby={r['input_index']:r for r in maps}
    refs += [baseline_path,maps_path,Path(inputs['database'])]
    database_to_input={f['database_image_id']:f['input_index'] for f in ok}
    db=sqlite3.connect(Path(inputs['database']).as_uri()+'?mode=ro',uri=True)
    keypoints={}
    for iid,n,c,blob in db.execute('SELECT image_id,rows,cols,data FROM keypoints'):
        if iid not in database_to_input:continue
        if n>20000 or c not in (2,4,6):raise ValueError('keypoint shape bound')
        a=np.frombuffer(blob or b'',np.float32).reshape(n,c)[:,:2].copy();keypoints[iid]=a
        data['baseline'][database_to_input[iid]]['candidate']=a.tolist()
    inliers={i:set() for i in database_to_input}
    for pid,n,c,blob in db.execute('SELECT pair_id,rows,cols,data FROM two_view_geometries WHERE rows>0'):
        a,b=divmod(pid,2147483647);m=np.frombuffer(blob,np.uint32).reshape(n,c)
        if a in inliers:inliers[a].update(map(int,m[:,0]))
        if b in inliers:inliers[b].update(map(int,m[:,1]))
    for iid,rs in inliers.items():data['baseline'][database_to_input[iid]]['verified']=keypoints[iid][sorted(rs)].tolist()
    db.close()
    import pycolmap
    baseline_models=[]
    for m in baseline.get('models',[]):
        binary=baseline_path.parent/m['binary_path'];refs += sorted(binary.glob('*.bin'))
        for binary_file in binary.glob('*.bin'):expected[str(binary_file)]=_sha(binary_file)
        rec=pycolmap.Reconstruction(str(binary));views=[]
        for mapped in maps:
            for registration in mapped.get('registrations',[]):
                if str(registration['model_id'])!=str(m['model_id']):continue
                iid=registration['image_id'];im=rec.images[iid]
                if im.name!=mapped['staged_image_name']:raise ValueError('baseline native frame mapping differs')
                pts=[q.xy.tolist() for q in im.points2D if q.has_point3D()]
                views.append(dict(input_index=mapped['input_index'],native_xy=pts))
        baseline_models.append(dict(model_id=str(m['model_id']),registered_views=int(rec.num_reg_images()),points=int(rec.num_points3D()),views=views,
            binary_path=str(binary),cameras=[dict(camera_id=int(cid),model=c.model_name,params=c.params.tolist()) for cid,c in rec.cameras.items()]))
    model_lists['baseline']=baseline_models
    for arm in ('sift_control','balanced'):
        tp=out/arm/'TRACKS.json';npth=out/arm/'native/RESULT.json';nplan=out/arm/'native/PLAN.json'
        if not tp.is_file():native_missing[arm]='TRACKS_MISSING';model_lists[arm]=[];continue
        refs.append(tp);expected[str(tp)]=_sha(tp);tracks=_read(tp)
        frequency=Counter(o['track_id'] for f in tracks['frames'] for o in f['observations'])
        for f in tracks['frames']:
            i=f['input_index'];original=byindex[i]
            if f['frame_id']!=original['frame_id'] or f['decode_index']!=original['decode_index']:raise ValueError('track report identity differs')
            data[arm][i]['candidate']=[[float(x)+.5,float(y)+.5] for o in f['observations'] for x,y in [o['xy']]]
            data[arm][i]['eligible']=[[float(x)+.5,float(y)+.5] for o in f['observations'] if frequency[o['track_id']]>=3 for x,y in [o['xy']]]
        if not npth.is_file() or not nplan.is_file():native_missing[arm]='NATIVE_RESULT_MISSING';model_lists[arm]=[];continue
        refs += [npth,nplan];expected[str(npth)]=_sha(npth);expected[str(nplan)]=_sha(nplan)
        native=_read(npth);nativeplan=_read(nplan)
        for ref in nativeplan.get('source_bindings',[]):
            if _sha(ref['path'])!=ref['sha256']:raise ValueError('native report source drift')
        rows_by_id={f['database_image_id']:f for f in nativeplan['frames']};track_by_input={f['input_index']:f for f in tracks['frames']}
        verified_rows={did:set() for did in rows_by_id}
        for pair in native.get('pairs',[]):
            for a,b in pair.get('inlier_feature_rows',[]):verified_rows[pair['image_a']].add(a);verified_rows[pair['image_b']].add(b)
        for did,rs in verified_rows.items():
            ledger=rows_by_id[did];i=ledger['input_index'];xy={o['track_id']:o['xy'] for o in track_by_input[i]['observations']}
            data[arm][i]['verified']=[[float(v)+.5 for v in xy[ledger['track_ids'][r]]] for r in sorted(rs)]
        model_lists[arm]=native.get('models',[])
    # Each frame shows its strongest SINGLE component, never union of components.
    for arm,models in model_lists.items():
        for model in models:
            for view in model.get('views',[]):
                item=data[arm][view['input_index']];pts=view['native_xy'];mid=str(model['model_id'])
                item['component_options'].append(dict(model_id=mid,observations=len(pts)))
                if len(pts)>len(item['model']) or (len(pts)==len(item['model']) and (item['component'] is None or mid<item['component'])):
                    item['model']=pts;item['component']=mid
    assets=out/'report_assets';assets.mkdir(exist_ok=False);metrics={arm:[] for arm in ARMS};frame_payload=[]
    for f in frames:
        i=f['input_index'];ip=Path(f['image_path']);refs.append(ip)
        if _sha(ip)!=f['image_sha256']:raise ValueError('image hash differs')
        with Image.open(ip) as image:image=image.convert('RGB')
        mask=None;bbox=None
        if f['original_mask_status']=='OK':
            mp=Path(f['mask_path']);refs.append(mp)
            if _sha(mp)!=f['mask_sha256']:raise ValueError('mask hash differs')
            with Image.open(mp) as mi:mask=np.asarray(mi)>0
            yy,xx=np.where(mask);bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
            draw=ImageDraw.Draw(image);x0,y0,x1,y1=bbox
            for j in range(9):
                x=x0+(x1-x0)*j/8;y=y0+(y1-y0)*j/8
                draw.line((x,y0,x,y1),fill=(70,150,70));draw.line((x0,y,x1,y),fill=(70,150,70))
            edge=mask.copy();edge[1:-1,1:-1]&=~(mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:])
            a=np.asarray(image).copy();ey,ex=np.where(edge);a[ey,ex]=[0,255,255];image=Image.fromarray(a)
        name=f'frame_{i:06d}.jpg';image.save(assets/name,quality=88);links.append(assets/name)
        payload=dict(input_index=i,decode_index=f['decode_index'],frame_id=f['frame_id'],time_seconds=f.get('time_seconds'),mask_status=f['original_mask_status'],
            image='report_assets/'+name,width=f['width'],height=f['height'],bbox=bbox,arms={})
        for arm in ARMS:
            item=data[arm][i];mm={}
            for stage in ('candidate','eligible','verified','model'):
                pts=item[stage]
                mm[stage]=None if mask is None or pts is None else dict(points=len(pts),grid=_coverage(pts,mask))
            payload['arms'][arm]=dict(**item,metrics=mm)
            if mask is not None:metrics[arm].append(dict(input_index=i,**mm,selected_component=item['component']))
        frame_payload.append(payload)
    # Four saved stages for the first two inputs of the baseline's largest component.
    biggest=sorted(baseline_models,key=lambda m:(-m['registered_views'],-m['points'],m['model_id']))
    examples=sorted(v['input_index'] for v in biggest[0]['views'])[:2] if biggest else []
    four_stage=None
    if len(examples)==2:
        fa,fb=[byindex[i] for i in examples];a,b=sorted([fa['database_image_id'],fb['database_image_id']]);pid=2147483647*a+b
        db=sqlite3.connect(Path(inputs['database']).as_uri()+'?mode=ro',uri=True);matches={}
        for table in ('matches','two_view_geometries'):
            row=db.execute('SELECT rows,cols,data FROM '+table+' WHERE pair_id=?',(pid,)).fetchone()
            matches[table]=np.empty((0,2),np.uint32) if row is None else np.frombuffer(row[2] or b'',np.uint32).reshape(row[0],row[1])
        db.close();panels=[];comparison=[]
        for i in examples:
            f=byindex[i];did=f['database_image_id'];col=0 if did==a else 1
            base=Image.open(assets/f'frame_{i:06d}.jpg').convert('RGB');row=[];names=[]
            for title,points in [('原SIFT',data['baseline'][i]['candidate']),('保存raw匹配',keypoints[did][matches['matches'][:,col]].tolist()),
                    ('几何验证',keypoints[did][matches['two_view_geometries'][:,col]].tolist()),('原生3D观测',data['baseline'][i]['model'])]:
                canvas=Image.new('RGB',(base.width,base.height+42),'white');canvas.paste(base,(0,42));draw=ImageDraw.Draw(canvas)
                # ASCII captions work with the default Pillow font on all hosts.
                stage_name=('SIFT','saved raw','verified','3D')[len(row)]
                draw.text((5,4),f'input{i} {stage_name}: {len(points)} points',fill='black')
                for x,y in points:draw.ellipse((x-2.5,y+39.5,x+1.5,y+43.5),fill=(255,220,0))
                row.append(canvas);names.append(dict(stage=title,points=len(points)))
            panels.append(row);comparison.append(dict(input_index=i,stages=names))
        w,h=panels[0][0].size;montage=Image.new('RGB',(4*w,2*h),'white')
        for y,row in enumerate(panels):
            for x,im in enumerate(row):montage.paste(im,(x*w,y*h))
        png=assets/'baseline_four_stage.png';montage.save(png);links.append(png);four_stage=dict(path='report_assets/'+png.name,frames=comparison)
    before={str(p):_sha(p) for p in set(refs)}
    for path,digest in expected.items():
        if path in before and before[path]!=digest:raise ValueError('source changed during report read: '+path)
        before[path]=digest
    aggregates={}
    for arm in ARMS:
        aggregates[arm]={stage:{str(n):_stats([r[stage]['grid'][str(n)]['fraction'] for r in metrics[arm] if r[stage] is not None]) for n in (4,8,12)} for stage in ('candidate','eligible','verified','model')}
    model_summaries={arm:[{k:m.get(k) for k in ('model_id','registered_views','points','registered_input_indices','cameras')} for m in models] for arm,models in model_lists.items()}
    summary=dict(schema='balanced-point-report-v1',status='REPORT_GENERATED_NOT_ACCEPTED',run_status=run.get('status'),input_frames=len(frames),mask_ok_frames=len(ok),
        arms=model_summaries,per_frame_metrics=metrics,grid_fraction_distributions=aggregates,native_missing=native_missing,
        baseline_eligible_status='NOT_AVAILABLE: old SIFT has no pre-native temporal candidate-ID ledger; do not invent a three-sample candidate set.',
        strongest_component_policy='Per frame: highest observation count, tie lexical model ID. Never union coordinates or observations from different models. Different frames may select different components.',
        four_stage=four_stage,geometry_accepted=False,scale_status='unknown',coverage_scope='2D image support, not surface coverage or correspondence truth',
        limits=['Verified stage deduplicates actual native inlier feature-row endpoints across saved pairs; it is still 2D correspondence support, not 3D.',
            'Candidate temporal track IDs are not independently established physical identity.',
            '4/8/12 grids use mask bbox; denominator is cells containing nonzero mask pixels. No coverage acceptance threshold.',
            'Saved raw matches are not every internal descriptor candidate; missing surfaces or masks are not filled.',
            'Old SIFT pre-native three-sample candidates are unavailable and shown as not applicable.'],
        source_bindings=[dict(path=p,sha256=h) for p,h in sorted(before.items())],source_drift=[p for p,h in before.items() if _sha(p)!=h],
        link_check=dict(checked=len(links)+1,missing=[str(p) for p in links if not p.is_file()]))
    payload=json.dumps(dict(frames=frame_payload,labels=LABELS),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    models_html=''.join('<h3>'+html.escape(LABELS[arm])+'</h3><pre>'+html.escape(json.dumps(model_summaries.get(arm,[]),ensure_ascii=False,indent=2))+'</pre>' for arm in ARMS)
    four_html='' if four_stage is None else '<h2>旧基线两注册帧四阶段</h2><a href="'+four_stage['path']+'"><img style="width:100%" src="'+four_stage['path']+'"></a>'
    page='''<!doctype html><meta charset="utf-8"><title>选点与跟踪逐帧对照</title><style>body{font:16px/1.6 system-ui;margin:24px}aside{background:#fff0ce;padding:15px}.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}canvas{width:100%;height:auto}pre{white-space:pre-wrap}input{width:65%}</style>
<h1>选点、连续跟踪与原生几何分阶段对照</h1><aside>黄色为实际保存坐标。候选、三样本候选、验证内点都是二维支持；只有model观测关联已有原生3D点，仍未质量验收。每帧只显示最强单一分量，不拼接分量。青线为mask，绿网格为mask bbox 8×8。</aside>
<p><a href="SUMMARY.json">逐帧指标、4/8/12敏感性、来源证据</a></p><label>帧 <input id="frame" type="range" min="0" step="1" value="0"></label><span id="which"></span>
<p>阶段 <select id="stage"><option value="candidate">二维候选</option><option value="eligible">至少3样本候选</option><option value="verified">原生验证内点</option><option value="model">单一分量3D观测</option></select></p><div class="cols" id="columns"></div>
<script>const DATA=PAYLOAD;const arms=['baseline','sift_control','balanced'];let token=0;const slider=document.getElementById('frame');slider.max=DATA.frames.length-1;
function draw(){const current=++token;const f=DATA.frames[Number(slider.value)],stage=document.getElementById('stage').value;document.getElementById('which').textContent=`input ${f.input_index}, decode ${f.decode_index}, ${f.time_seconds}s, mask ${f.mask_status}`;
document.getElementById('columns').replaceChildren();for(const arm of arms){const card=document.createElement('section'),h=document.createElement('h3'),p=document.createElement('p'),c=document.createElement('canvas');h.textContent=DATA.labels[arm];const a=f.arms[arm],pts=a[stage],m=a.metrics[stage];p.textContent=pts===null?'旧SIFT无原生前时序候选账本：不可用':`${pts.length} 点; 8×8占格 ${m?m.grid['8'].occupied+'/'+m.grid['8'].nonempty_mask_cells:'N/A'}; model ${a.component??'无'}; 可用分量 ${JSON.stringify(a.component_options)}`;c.width=f.width;c.height=f.height;card.append(h,p,c);document.getElementById('columns').append(card);const im=new Image();im.onload=()=>{if(current!==token)return;const x=c.getContext('2d');x.drawImage(im,0,0,c.width,c.height);x.fillStyle='#ffe000';for(const q of pts??[]){x.beginPath();x.arc(q[0]-.5,q[1]-.5,2,0,Math.PI*2);x.fill()}};im.src=f.image}}
slider.oninput=draw;document.getElementById('stage').onchange=draw;draw();</script>FOUR<h2>所有原生分量（坐标不可合并）</h2>MODELS<h2>限制</h2><p>原始匹配表不是全部内部候选。低重投影误差、广泛图像占格、成功导出均不是物理外形准确性。未知/失败状态保留；没有补点、补mask或补洞。</p>'''.replace('PAYLOAD',payload).replace('FOUR',four_html).replace('MODELS',models_html)
    (out/'index.html').write_text(page,encoding='utf-8')
    with (out/'SUMMARY.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
    if summary['link_check']['missing'] or not (out/'SUMMARY.json').is_file():raise ValueError('report links missing')
    if summary['source_drift']:raise ValueError('source changed during report generation')
    return dict(status=summary['status'],report=str(out/'index.html'),summary=str(out/'SUMMARY.json'),mask_ok_frames=len(ok),geometry_accepted=False)
