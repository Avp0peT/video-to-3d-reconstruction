"""Image-backed region tracking atlas; no native solver or database writes."""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import math
import os
import sqlite3
import struct


def _read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def _path(value, base):
    p = Path(value)
    return p if p.is_absolute() else base / p


def _sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda: s.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def _color(t):
    b = hashlib.sha256(str(t).encode()).digest()
    return tuple(70 + x % 186 for x in b[:3])


def _sample(rows, n):
    return rows if len(rows) <= n else [rows[i * (len(rows)-1)//(n-1)] for i in range(n)]


def _baseline(plan, run):
    value = plan.get('baseline_sfm')
    if not value: return {}, 'NOT_AVAILABLE'
    value = value.get('path', value.get('run', value.get('directory'))) if isinstance(value, dict) else value
    if not value: return {}, 'NOT_AVAILABLE'
    root = _path(value, run)
    maps = root / 'frames_map.jsonl'
    dbpath = root / 'database.db'
    if not maps.exists() or not dbpath.exists(): return {}, 'EXPLICIT_FRAME_MAP_OR_DATABASE_MISSING'
    refs = [_sha(maps), _sha(dbpath)]
    rows = [json.loads(s) for s in maps.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    out = {}
    db = sqlite3.connect(dbpath.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        for r in rows:
            iid = r.get('database_image_id')
            if iid is None: continue
            q = db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (iid,)).fetchone()
            if q is None: continue
            n, cols, data = q
            if cols not in (2, 4, 6) or len(data or b'') != n*cols*4: raise ValueError('invalid baseline keypoint blob')
            vals = struct.unpack('<' + 'f'*(n*cols), data or b'')
            # COLMAP stores corner-origin centres; report uses OpenCV pixel centres.
            xy = [[vals[i*cols]-.5, vals[i*cols+1]-.5] for i in range(n)]
            out[r['input_index']] = dict(xy=xy, database_image_id=iid, status=r.get('status'), registrations=r.get('registrations', []))
    finally: db.close()
    if refs != [_sha(maps), _sha(dbpath)]: raise ValueError('baseline source drift during report')
    return out, 'READ_ONLY_EXPLICIT_IMAGE_MAP'


def render(run, source_geometry=None):
    """Render from tracking evidence; optional COMPARISON supplies native observations.

    COMPARISON.frames rows may contain input_index/native_xy/registered.
    Missing native evidence is explicitly NOT_EVALUATED, never inferred from tracks.
    """
    from PIL import Image, ImageDraw, ImageFont
    from .region_experiment import resolve_source_geometry
    run = Path(run).resolve()
    plan = _read(run/'PLAN.json')
    source = run/'tracking/REGION_TRACKS.json'
    tracks = _read(source)
    if tracks.get('schema') != 'glove-region-tracks-v1': raise ValueError('unexpected tracking schema')
    frames = tracks['frames']
    if len({f['input_index'] for f in frames}) != len(frames): raise ValueError('duplicate endpoint')
    frames = sorted(frames, key=lambda f: f['input_index'])
    comp_path = run/'COMPARISON.json'
    comp = _read(comp_path) if comp_path.exists() else {}
    # Historical comparison JSON remains untouched. Its unbound part labels must
    # not become object-part evidence merely because a new report reads them.
    part_scope = dict(status='NOT_EVALUATED',
        reason='No bound object-part annotation is evaluated; legacy frame-number groups are not reused.')
    comp['visual_back_group'] = None
    for metric in comp.get('globalmetrics', {}).values():
        if isinstance(metric, dict) and 'labelled_back_registered' in metric:
            metric.update(labelled_back_registered=None, region_annotation_status='NOT_EVALUATED')
    for model in comp.get('models', []):
        if 'previously_labelled_back_views' in model:
            model.update(previously_labelled_back_views=None, region_annotation_status='NOT_EVALUATED')
    geometry_name = resolve_source_geometry(run,
        source_geometry if source_geometry is not None else comp.get('source_geometry'), required=False)
    geometry_path = run / geometry_name / 'RESULT.json' if geometry_name is not None else None
    if source_geometry is None and comp.get('source_geometry') is not None:
        expected = comp.get('source_bindings', {}).get(str(geometry_path))
        if expected is None or _sha(geometry_path) != expected:
            raise ValueError('comparison source geometry binding missing or drifted')
    geometry = _read(geometry_path) if geometry_path else {}
    if not comp.get('models') and geometry.get('models'): comp['models'] = geometry['models']
    native = {f['input_index']: f for f in comp.get('frames', [])}
    baseline, baseline_status = _baseline(plan, run)
    counts = Counter(o['track_id'] for f in frames for o in f['observations'])
    supported_ids = {t for t,n in counts.items() if n >= 2}
    out = run/'report'; out.mkdir(exist_ok=True)
    fontpath = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(fontpath), 17) if fontpath.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(fontpath), 14) if fontpath.exists() else font
    width = 384
    loaded = {}
    summaries = []

    def load(f):
        p = _path(f['image_path'], run)
        if f.get('image_sha256') and _sha(p) != f['image_sha256']: raise ValueError('frame hash drift')
        im = Image.open(p).convert('RGB')
        if im.size != (f['width'], f['height']): raise ValueError('frame dimensions drift')
        mp = _path(f['mask_path'], run)
        if f.get('mask_sha256') and _sha(mp) != f['mask_sha256']: raise ValueError('mask hash drift')
        mask = Image.open(mp).convert('L')
        if mask.size != im.size: raise ValueError('mask domain drift')
        return im, mask

    def overlay(f, points, color=None):
        im, mask = loaded[f['input_index']]
        im = im.copy(); shade = Image.new('RGB', im.size, (12, 15, 22))
        im = Image.composite(im, Image.blend(im, shade, .65), mask)
        scale = width/im.width
        im = im.resize((width, round(im.height*scale)))
        d = ImageDraw.Draw(im)
        for xy, t in points:
            x,y = xy; x*=scale; y*=scale
            c = color or _color(t)
            d.ellipse((x-2,y-2,x+2,y+2), fill=c)
        return im

    for f in frames:
        idx = f['input_index']; loaded[idx] = load(f)
        obs = f['observations']
        ids = [o['track_id'] for o in obs]
        if len(set(ids)) != len(ids): raise ValueError('duplicate track per frame')
        for o in obs:
            if len(o['xy']) != 2 or not all(math.isfinite(v) for v in o['xy']): raise ValueError('nonfinite observation')
        shared = [o for o in obs if o['track_id'] in supported_ids]
        seedonly = [o for o in obs if o['track_id'] not in supported_ids]
        old = baseline.get(idx, {})
        a = overlay(f, [(p,0) for p in old.get('xy', [])], (255,190,45))
        b = overlay(f, [(o['xy'],o['track_id']) for o in shared])
        d = ImageDraw.Draw(b); scale=width/f['width']
        for o in seedonly:
            x,y=[v*scale for v in o['xy']]; d.rectangle((x-2,y-2,x+2,y+2), outline=(130,130,130))
        nr = native.get(idx, {}); nxy = nr.get('native_xy')
        if nxy is not None:
            for x,y in nxy:
                x*=scale;y*=scale;d.ellipse((x-4,y-4,x+4,y+4),outline=(255,70,90),width=2)
        canvas=Image.new('RGB',(width*2,a.height+76),(18,23,33)); canvas.paste(a,(0,76));canvas.paste(b,(width,76))
        d=ImageDraw.Draw(canvas)
        d.text((8,4),f'端点 {idx} / 原帧 {f["decode_index"]}',font=font,fill='white')
        d.text((8,29),f'SIFT中心 {len(old.get("xy",[]))} / {old.get("status","NOT_AVAILABLE")}',font=small,fill=(255,190,45))
        d.text((width+8,4),f'跨端点候选 {len(shared)} / 单次seed {len(seedonly)}',font=small,fill='white')
        d.text((width+8,29),'红圈: 实际3D关联 '+(str(len(nxy)) if nxy is not None else 'NOT_EVALUATED'),font=small,fill=(255,90,110))
        name=f'frame_{idx:03d}.jpg';canvas.save(out/name,quality=90)
        # Occupancy is only projected 2D grid coverage inside the current mask.
        _,mask=loaded[idx];mw,mh=mask.size; occupied=set();valid=set()
        for gy in range(8):
            for gx in range(8):
                box=(gx*mw//8,gy*mh//8,(gx+1)*mw//8,(gy+1)*mh//8)
                if mask.crop(box).getbbox(): valid.add((gx,gy))
        for o in shared:
            x,y=o['xy']
            if 0<=x<mw and 0<=y<mh and mask.getpixel((int(x),int(y)))>0: occupied.add((min(7,int(x*8/mw)),min(7,int(y*8/mh))))
        summaries.append(dict(input_index=idx,decode_index=f['decode_index'],sift_centres=len(old.get('xy',[])),shared_endpoint_candidates=len(shared),singleton_seeds=len(seedonly),native_associated_observations=len(nxy) if nxy is not None else None,grid_occupied=len(occupied),grid_mask_cells=len(valid),grid_fraction=len(occupied)/len(valid) if valid else None,tracked_over_sift=len(shared)/len(old['xy']) if old.get('xy') else None,image=name))

    adjacent=list(zip(frames,frames[1:]))
    chosen=_sample(adjacent, min(24,len(adjacent)))
    pairs=[]
    for a,b in chosen:
        aa={o['track_id']:o['xy'] for o in a['observations']};bb={o['track_id']:o['xy'] for o in b['observations']}
        common=sorted(set(aa)&set(bb),key=str);show=_sample(common,60)
        ia=overlay(a,[]);ib=overlay(b,[]);h=max(ia.height,ib.height)
        canvas=Image.new('RGB',(width*2,h+50),(18,23,33));canvas.paste(ia,(0,50));canvas.paste(ib,(width,50));d=ImageDraw.Draw(canvas)
        d.text((8,5),f'端点 {a["input_index"]} → {b["input_index"]}: 共同候选 {len(common)}，均匀画 {len(show)}',font=font,fill='white')
        for t in show:
            x,y=aa[t];u,v=bb[t];s=width/a['width'];q=width/b['width']
            d.line((x*s,y*s+50,u*q+width,v*q+50),fill=_color(t),width=1)
        name=f'pair_{a["input_index"]:03d}_{b["input_index"]:03d}.jpg';canvas.save(out/name,quality=90)
        patches=[]
        for j,t in enumerate(_sample(common, min(3,len(common)))):
            strip=Image.new('RGB',(264,164),(18,23,33));pd=ImageDraw.Draw(strip)
            pd.text((4,2),f'ID {t} / endpoint crops',font=small,fill='white')
            for k,(f,xy) in enumerate(((a,aa[t]),(b,bb[t]))):
                im=loaded[f['input_index']][0];x,y=xy
                crop=im.crop((round(x)-16,round(y)-16,round(x)+17,round(y)+17)).resize((132,132))
                cd=ImageDraw.Draw(crop);cd.line((60,64,72,64),fill=_color(t),width=2);cd.line((66,58,66,70),fill=_color(t),width=2)
                strip.paste(crop,(k*132,32))
            pn=f'patch_{a["input_index"]:03d}_{b["input_index"]:03d}_{j}.png';strip.save(out/pn)
            patches.append(dict(track_id=t,image=pn,ncc_status='NOT_EVALUATED',scope='endpoint crops; not consecutive decoded-frame evidence'))
        pairs.append(dict(input_indices=[a['input_index'],b['input_index']],common_candidates=len(common),displayed=len(show),image=name,patches=patches))

    esc=lambda x:html.escape(str(x))
    links=[]
    for label,value in [('固定SIFT逐阶段报告',plan.get('baseline_report')),('跟踪原始记录',source),('几何原始结果',geometry_path),('比较账本',comp_path)]:
        if value:
            p=_path(value,run)
            if p.is_dir():p=p/'diagnostics/index.html' if (p/'diagnostics/index.html').exists() else p/'index.html'
            if p.exists():links.append(f'<a href="{esc(os.path.relpath(p,run).replace(chr(92),chr(47)))}">{esc(label)}</a>')
    report=dict(schema='glove-region-report-v1',status='REPORT_GENERATED_NOT_QUALITY_ACCEPTED',tracking_status=tracks.get('status'),frames=summaries,pairs=pairs,endpoint_count=len(frames),shared_track_ids=len(supported_ids),singleton_track_ids=sum(n==1 for n in counts.values()),baseline_read_status=baseline_status,globalmetrics=comp.get('globalmetrics',{}),models=comp.get('models',[]),notes=comp.get('notes'),timeline_stats=tracks.get('timeline_stats'),tracking_metrics=tracks.get('metrics'),physical_identity_verified=False,geometry_accepted=False,coverage_scope='8x8 full-image grid intersected with mask; projected observation occupancy, not physical surface coverage',source_bindings=[dict(path=str(source),sha256=_sha(source)),dict(path=str(run/'PLAN.json'),sha256=_sha(run/'PLAN.json'))])
    report.update(object_scope='object-agnostic mask-bound region candidates; legacy glove schema retained for compatibility',
                  region_annotation_scope=part_scope, visual_back_group=None,
                  source_geometry=geometry_name,
                  pair_selection_scope='uniform sample of up to 24 adjacent endpoint pairs; no inferred object-part priority')
    report['source_bindings'].extend(dict(path=str(path), sha256=_sha(path))
                                    for path in (comp_path, geometry_path) if path is not None and path.is_file())
    (run/'OUTPUT_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    body=['<!doctype html><meta charset="utf-8"><title>目标区域对应逐阶段证据</title><style>body{background:#101722;color:#e8edf5;font:16px system-ui;max-width:1280px;margin:30px auto;padding:15px}a{color:#8bcaff}img{width:100%;height:auto}details{border:1px solid #456;padding:12px;margin:10px 0}pre{white-space:pre-wrap}nav{display:flex;gap:20px;flex-wrap:wrap}</style>', '<h1>目标：区域匹配与时序跟踪证据</h1>','<p>状态：REPORT_GENERATED_NOT_QUALITY_ACCEPTED。相同颜色只表示同一跟踪候选ID；不证明同一物理表面。灰色方框是只出现一次的seed，不计有效跨端点对应。红圈仅来自实际native关联3D观测，缺数据明确保留NOT_EVALUATED。</p>','<nav>'+ ' · '.join(links)+'</nav>','<p>对象部位统计：NOT_EVALUATED。此报告没有使用已绑定的部位标注，不从端点编号猜测物体部位。请结合原图逐端点审核纹理与轮廓。</p>', '<h2>1. 固定SIFT与新增候选对照</h2>','<p>左图为旧数据库实际SIFT中心，右图为区域跟踪候选。特征中心数、匹配行数、三维点数是不同指标。8×8格占用只描述当前图像中的投影分布；tracked/SIFT只是数量比。</p>','<pre>'+esc(json.dumps(dict(endpoint_count=len(frames),shared_track_ids=len(supported_ids),baseline_read_status=baseline_status,globalmetrics=comp.get('globalmetrics',{})),ensure_ascii=False,indent=2))+'</pre>']
    for f in summaries:
        body.append(f'<details id="frame{f["input_index"]}"><summary>端点 {f["input_index"]} / 原帧 {f["decode_index"]}：跨端点候选 {f["shared_endpoint_candidates"]}；SIFT {f["sift_centres"]}；格覆盖 {f["grid_occupied"]}/{f["grid_mask_cells"]}</summary><img loading="lazy" src="report/{f["image"]}"><pre>{esc(json.dumps(f,ensure_ascii=False,indent=2))}</pre></details>')
    body+=['<h2>2. 相邻端点的共享轨迹</h2>','<p>最多显示24组；每组共同候选最多均匀显示60条，同时保留总数。0共同候选仍显示原图。连线是候选对应，不是几何验证通过的匹配。</p>']
    for p in pairs:
        crops=''.join(f'<img style="width:264px" loading="lazy" src="report/{q["image"]}">' for q in p['patches'])
        body.append(f'<details><summary>{esc(p["input_indices"])}：共同 {p["common_candidates"]} / 画 {p["displayed"]}</summary><img loading="lazy" src="report/{p["image"]}"><p>实际33×33端点局部放大（不是逐原帧连续证据）；NCC: NOT_EVALUATED。</p>{crops}</details>')
    timeline=tracks.get('timeline_stats') or []
    rows=timeline if isinstance(timeline,list) else timeline.get('frames',timeline.get('rows',[])) if isinstance(timeline,dict) else []
    active=[];reasons=Counter()
    for r in rows:
        if not isinstance(r,dict):continue
        for key in ('active_tracks','active','active_count','observations'):
            v=r.get(key)
            if isinstance(v,(int,float)):active.append(v);break
            if isinstance(v,list):active.append(len(v));break
        for key in ('reject_reasons','failure_reasons','rejection_reasons','reasons','stop_reasons'):
            v=r.get(key)
            if isinstance(v,dict):
                for k,n in v.items():
                    if isinstance(n,(int,float)):reasons[str(k)]+=n
            elif isinstance(v,list):reasons.update(str(x) for x in v)
            elif isinstance(v,str):reasons[v]+=1
    active.sort();median=(active[(len(active)-1)//2]+active[len(active)//2])/2 if active else None
    aggregate=dict(timeline_rows=len(rows),active_median=median,failure_reasons=dict(reasons))
    if isinstance(timeline,dict):
        for k,v in timeline.items():
            if isinstance(v,(str,int,float,bool)) or v is None:aggregate[k]=v
    body+=['<h2>3. 连续时间与原生几何</h2>','<p>此图册展示端点对应；连续帧统计来自跟踪账本。局部NCC、正反检查和几何验证若未提供，均不从图像外观猜测。格覆盖为与掩膜相交的全图8×8格投影占用。红圈可以汇集多个独立component的实际关联观测；这不表示将分量重构或拼成同一个物体。</p>','<pre>'+esc(json.dumps(aggregate,ensure_ascii=False,indent=2))+'</pre>', '<p>以下各模型/分量独立展示，不拼接；候选数量不能当作新增三维点。</p>', '<table border="1" cellpadding="8"><tr><th>模型</th><th>注册视图</th><th>三维点</th><th>一个区域ID对应多个3D点</th><th>注册端点</th><th>查看</th></tr>']
    for m in comp.get('models',[]):
        viewer=m.get('viewer');link='NOT_AVAILABLE'
        if viewer:
            vp=_path(viewer,run)
            link=f'<a href="{esc(os.path.relpath(vp,run).replace(chr(92),chr(47)))}">独立交互查看</a>'
        body.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in (str(m.get('label','UNKNOWN'))+' / '+str(m.get('model_id','?'))+' / '+str(m.get('camera_domain_status','NOT_EVALUATED')),m.get('registered_views','NOT_EVALUATED'),m.get('points','NOT_EVALUATED'),m.get('region_ids_split_into_multiple_3d_points','NOT_EVALUATED'),m.get('registered_indices','NOT_EVALUATED')))+f'<td>{link}</td></tr>')
    body+=['</table>','<pre>'+esc(json.dumps(comp.get('notes'),ensure_ascii=False,indent=2))+'</pre>','<p><a href="OUTPUT_REPORT.json">完整报告统计JSON</a> · <a href="tracking/REGION_TRACKS.json">连续跟踪来源JSON</a></p>']
    (run/'index.html').write_text('\n'.join(body),encoding='utf-8')
    return report
