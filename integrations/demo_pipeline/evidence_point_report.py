"""Saved-coordinate report for sample admission and frozen geometry evidence.

This module never imports a solver or changes a native reconstruction.
"""
from pathlib import Path
from collections import Counter
import html
import json

from .balanced_point_report import _read, _sha, _stats, _coverage

ARMS = ('control', 'two_sample_candidates', 'geometry_evidence')
LABELS = {'control': '旧光照跟踪：≥3采样准入',
          'two_sample_candidates': '≥2采样准入；保留旧两视开关',
          'geometry_evidence': '≥2采样准入；允许两视三角化'}
STAGES = ('eligible', 'verified', 'model', 'consistent')


def build(output: Path) -> dict:
    """Render completed saved results; preserve every independent component."""
    import numpy as np
    from PIL import Image, ImageDraw
    out = Path(output).resolve(strict=True)
    if any((out / p).exists() for p in ('SUMMARY.json', 'index.html', 'report_assets')):
        raise ValueError('fresh evidence report output required')
    expected = {}

    def bind(path, digest=None, size=None):
        p = Path(path).resolve(strict=True)
        h = _sha(p)
        if digest is not None and h != digest:
            raise ValueError('source digest differs: ' + str(p))
        if size is not None and p.stat().st_size != size:
            raise ValueError('source size differs: ' + str(p))
        if str(p) in expected and expected[str(p)] != h:
            raise ValueError('source changed: ' + str(p))
        expected[str(p)] = h
        return p

    def read(p):
        return _read(bind(p))

    def refs(doc, field='source_bindings'):
        values = doc.get(field, [])
        if isinstance(values, dict):
            values = list(values.values())
        for r in values:
            if isinstance(r, dict) and 'path' in r and 'sha256' in r:
                bind(r['path'], r['sha256'], r.get('bytes'))

    plan = read(out / 'PLAN.json')
    run = read(out / 'RESULT.json')
    control = Path(plan['control_run']).resolve(strict=True)
    inputs_path = out / 'INPUTS.json'
    inputs = read(inputs_path if inputs_path.is_file() else control / 'INPUTS.json')
    refs(inputs, 'bindings'); refs(plan); refs(plan, 'implementation')
    tracks_path = bind(plan['tracks_path'])
    tracks = read(tracks_path)
    if tracks.get('schema') != 'balanced-object-observations-v1':
        raise ValueError('unknown shared observations schema')
    frames = sorted(inputs['frames'], key=lambda f: f['input_index'])
    originals = {f['input_index']: f for f in frames}
    if len(originals) != len(frames):
        raise ValueError('duplicate source frame index')
    obs = {}; frequency = Counter()
    for f in tracks['frames']:
        i = f['input_index']; original = originals[i]
        if i in obs or any(f[k] != original[k] for k in ('frame_id', 'decode_index', 'image_sha256', 'mask_sha256')):
            raise ValueError('shared observations/source identity mismatch')
        if original['original_mask_status'] != 'OK':
            raise ValueError('observations on non-OK mask')
        row = {}
        for o in f['observations']:
            tid = o['track_id']; xy = np.asarray(o['xy'], dtype=float)
            if type(tid) is not int or tid in row or xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError('invalid shared observation')
            if o.get('predicted') is True or o.get('measured') is False:
                raise ValueError('predictions are not measured observations')
            row[tid] = (xy + .5).astype(np.float32).astype(float).tolist()
            frequency[tid] += 1
        obs[i] = row
    data = {a: {i: dict(eligible=[], verified=[], model=[], consistent=[], provisional=[],
                      component=None, component_options=[]) for i in originals} for a in ARMS}
    directories = {'control': control / 'photometric/native',
                   **{a: out / a / 'native' for a in ARMS[1:]}}
    summaries = {}; models_by_arm = {}; audits = {}
    for arm in ARMS:
        folder = directories[arm]
        native = read(folder / 'RESULT.json'); native_plan = read(folder / 'PLAN.json')
        refs(native_plan)
        audit_path = out / 'control_geometry_evidence.json' if arm == 'control' else folder / 'GEOMETRY_EVIDENCE.json'
        audit = read(audit_path); refs(audit)
        if audit.get('schema') != 'frozen-native-geometry-evidence-v1':
            raise ValueError('unknown frozen geometry evidence schema')
        if Path(audit['native_result_path']).resolve(strict=True) != (folder / 'RESULT.json').resolve(strict=True):
            raise ValueError('audit is bound to another native result')
        audits[arm] = audit
        ledgers = {}; by_input = {}
        for f in native_plan['frames']:
            i = f['input_index']; tids = f['track_ids']
            if i in by_input or f['database_image_id'] in ledgers or len(set(tids)) != len(tids):
                raise ValueError('duplicate native feature ledger')
            if f['frame_id'] != originals[i]['frame_id'] or f['decode_index'] != originals[i]['decode_index']:
                raise ValueError('native ledger frame mismatch')
            points = [obs[i][tid] for tid in tids]
            data[arm][i]['eligible'] = points
            ledgers[f['database_image_id']] = (i, points, tids)
            by_input[i] = f
        verified = {i: set() for i in originals}
        for p in native.get('pairs', []):
            a, b = ledgers[p['image_a']], ledgers[p['image_b']]
            pairs = p['inlier_feature_rows']
            if len(pairs) != p['inliers']:
                raise ValueError('verified feature count mismatch')
            for ra, rb in pairs:
                if not (0 <= ra < len(a[1]) and 0 <= rb < len(b[1])):
                    raise ValueError('verified feature row outside ledger')
                verified[a[0]].add(ra); verified[b[0]].add(rb)
        for iid, (i, points, tids) in ledgers.items():
            data[arm][i]['verified'] = [points[r] for r in sorted(verified[i])]
        audit_components = {str(c['model_id']): c for c in audit['components']}
        models = native.get('models', []); models_by_arm[arm] = models
        if set(audit_components) != {str(m['model_id']) for m in models}:
            raise ValueError('audit/native component identities differ')
        component_summary = []
        for m in models:
            mid = str(m['model_id']); ac = audit_components[mid]
            records = {p['point3D_id']: p for p in m['point_records']}
            ap = {p['point3D_id']: p for p in ac['points']}
            if set(records) != set(ap):
                raise ValueError('audit/native point IDs differ')
            lengths = Counter(len({t['native_image_id'] for t in p['track']}) for p in records.values())
            failures = Counter(r for p in ap.values() for r in p['rejection_reasons'])
            component_summary.append(dict(model_id=m['model_id'], registered_views=m['registered_views'],
                registered_input_indices=m['registered_input_indices'], raw_points=m['points'],
                raw_two_view_points=lengths[2], raw_multi_view_points=sum(v for k,v in lengths.items() if k>=3),
                consistent_points=sum(p['pass_frozen_geometry'] is True for p in ap.values()),
                classifications=dict(Counter(p['classification'] for p in ap.values())),
                consistency_failed_points=sum(p['pass_frozen_geometry'] is not True for p in ap.values()),
                rejection_reasons=dict(failures), cameras=m.get('cameras'), components_merged=False))
            for view in m['views']:
                i = view['input_index']; item = data[arm][i]
                if view['frame_id'] != originals[i]['frame_id']:
                    raise ValueError('native view identity mismatch')
                raw = []; good = []; provisional = []
                for o in view['observations']:
                    row = o['feature_row']; ledger = ledgers[view['native_image_id']]
                    if ledger[0] != i or ledger[2][row] != o['candidate_track_id'] or ledger[1][row] != o['native_xy']:
                        raise ValueError('native observation/feature identity mismatch')
                    p = ap[o['point3D_id']]; raw.append(o['native_xy'])
                    if p['pass_frozen_geometry'] is True:
                        good.append(o['native_xy'])
                        if p['classification'] == 'two_view_provisional': provisional.append(o['native_xy'])
                item['component_options'].append(dict(model_id=mid, raw_observations=len(raw), consistent_observations=len(good)))
                if item['component'] is None or (-len(raw), mid) < (-len(item['model']), item['component']):
                    item.update(model=raw, consistent=good, provisional=provisional, component=mid)
        unique = sorted({i for m in models for i in m['registered_input_indices']})
        if len(unique) != native['unique_registered_count']:
            raise ValueError('unique registered count mismatch')
        summaries[arm] = dict(native_status=native.get('status'), unique_registered_count=len(unique),
            unique_registered_inputs=unique, components=len(models), largest_component_views=max((m['registered_views'] for m in models), default=0),
            candidate_pair_count=len(native_plan.get('candidate_pairs', [])),
            raw_pair_match_rows=sum(p['raw_matches'] for p in native.get('pairs', [])),
            verified_pair_match_rows=sum(p['inliers'] for p in native.get('pairs', [])),
            verified_nonzero_pairs=sum(p['inliers']>0 for p in native.get('pairs', [])),
            component_statistics=component_summary, actual_options=native.get('actual_options'),
            audit_thresholds=audit['thresholds'], geometry_accepted=False)
    assets = out / 'report_assets'; assets.mkdir()
    visual = []; metrics = {a: [] for a in ARMS}; links = []
    for f in frames:
        i = f['input_index']
        with Image.open(bind(f['image_path'], f['image_sha256'])) as im: image = im.convert('RGB')
        mask = None
        if f['original_mask_status'] == 'OK':
            with Image.open(bind(f['mask_path'], f['mask_sha256'])) as im: mask = np.asarray(im)>0
            if mask.ndim != 2 or not mask.any(): raise ValueError('invalid report mask')
            y,x = np.where(mask); x0,y0,x1,y1 = int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)
            draw=ImageDraw.Draw(image)
            for j in range(9):
                xx=x0+j*(x1-x0)/8; yy=y0+j*(y1-y0)/8
                draw.line((xx,y0,xx,y1),fill=(70,150,70)); draw.line((x0,yy,x1,yy),fill=(70,150,70))
            edge=mask.copy(); edge[1:-1,1:-1]&=~(mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:])
            pixels=np.asarray(image).copy(); pixels[edge]=[0,255,255]; image=Image.fromarray(pixels)
        path=assets/f'frame_{i:06d}.jpg'; image.save(path,quality=90); links.append(path)
        vf=dict(input_index=i,frame_id=f['frame_id'],decode_index=f['decode_index'],mask_status=f['original_mask_status'],
            width=f['width'],height=f['height'],image='report_assets/'+path.name,arms={})
        for arm in ARMS:
            item=data[arm][i]
            ms={s:None if mask is None else dict(points=len(item[s]),grid=_coverage(item[s],mask)) for s in STAGES}
            vf['arms'][arm]=dict(**item,metrics=ms)
            if mask is not None: metrics[arm].append(dict(input_index=i,**ms,component=item['component']))
        visual.append(vf)
    for arm in ARMS:
        summaries[arm]['stage_distributions']={s:dict(points=_stats([m[s]['points'] for m in metrics[arm]]),
            grid8_fraction=_stats([m[s]['grid']['8']['fraction'] for m in metrics[arm]])) for s in STAGES}
    ranked=sorted(models_by_arm['control'],key=lambda m:(-m['registered_views'],-m['points'],str(m['model_id'])))
    examples=sorted(ranked[0]['registered_input_indices'])[:2] if ranked else sorted(obs)[:2]
    static=[]
    for stage in STAGES:
        tiles=[]
        for i in examples:
            row=[]
            for arm in ARMS:
                with Image.open(assets/f'frame_{i:06d}.jpg') as im: base=im.convert('RGB')
                tile=Image.new('RGB',(base.width,base.height+44),'white'); tile.paste(base,(0,44)); d=ImageDraw.Draw(tile)
                item=data[arm][i]; d.text((5,4),f'input{i} {arm}/{stage}: {len(item[stage])} model={item["component"]}',fill='black')
                d.text((5,22),'yellow=raw/consistent multi; pink=consistent two-view provisional',fill='black')
                for points,color,r in [(item[stage],(255,220,0),2)]+([(item['provisional'],(255,70,130),3)] if stage in ('model','consistent') else []):
                    for x,y in points:d.ellipse((x-.5-r,y+43.5-r,x-.5+r,y+43.5+r),fill=color)
                row.append(tile)
            tiles.append(row)
        if tiles:
            w,h=tiles[0][0].size; montage=Image.new('RGB',(3*w,h*len(tiles)),'white')
            for y,row in enumerate(tiles):
                for x,t in enumerate(row):montage.paste(t,(x*w,y*h))
            p=assets/f'comparison_{stage}.png'; montage.save(p); links.append(p)
            static.append(dict(stage=stage,path='report_assets/'+p.name,input_indices=examples))
    limitations=['同一份已保存TRACKS，未重新跟踪、分割或更改坐标；旧门是至少3个不同采样观测，不要求连续3帧。',
        'eligible来自每个native PLAN的实际track_ids；verified是至少一个候选对的验证内点行去重，不是三维点。',
        '原ignore_two_view_tracks=true并不保证最终每个三维点至少3视；以保存point track实际视图数为准。',
        '一致性仅复核拟合使用的冻结模型观测、正深度、重投影与光线夹角；不是held-out检验、真实身份或几何准确性验收。',
        '两视通过点仍为provisional。多视通过点也未质量验收；低重投影误差不证明正确形状。',
        '每帧选择原始3D观测最多的单一分量，同数按model ID排序；一致性层保持同一分量。各分量独立，不合并或累加为完整点云。',
        '网格占用分母是mask bbox内具有mask像素的格子；4/8/12粒度均记录，仅表示图像支持范围，不是表面覆盖率，无通过阈值。',
        '相机自标定、遮挡、纹理重复、尺度unknown与实际外形仍未接受；不能由本报告作因果改善声明。']
    drift=[p for p,h in expected.items() if _sha(p)!=h]
    summary=dict(schema='evidence-point-report-v1',status='REPORT_GENERATED_NOT_ACCEPTED',run_status=run.get('status'),
        shared_tracks_path=str(tracks_path),control_run=str(control),mask_status_counts=dict(Counter(f['original_mask_status'] for f in frames)),
        shared_tracks=dict(observations=sum(frequency.values()),unique_ids=len(frequency),exactly_two_sample_tracks=sum(n==2 for n in frequency.values()),
            at_least_two_sample_tracks=sum(n>=2 for n in frequency.values()),at_least_three_sample_tracks=sum(n>=3 for n in frequency.values())),
        arms=summaries,per_frame_metrics=metrics,static_comparisons=static,limitations=limitations,components_merged=False,
        geometry_accepted=False,track_identity_truth=False,scale_status='unknown',heldout_validation=False,
        source_bindings=[dict(path=p,sha256=h) for p,h in sorted(expected.items())],source_drift=drift,
        link_check=dict(checked=len(links)+1,missing=[str(p) for p in links if not p.is_file()]))
    if drift or summary['link_check']['missing']: raise ValueError('report source/link drift')
    payload=json.dumps(dict(frames=visual,labels=LABELS),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    rows=''
    for arm in ARMS:
        a=summaries[arm]; ds=a['stage_distributions']; cell=[LABELS[arm],a['unique_registered_count'],a['components'],a['largest_component_views']]
        cell += [f'{ds[s]["points"].get("median")} / {ds[s]["grid8_fraction"].get("median")}' for s in STAGES]
        rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cell)+'</tr>'
    pics=''.join('<h3>'+p['stage']+'</h3><a href="'+p['path']+'"><img width="100%" src="'+p['path']+'"></a>' for p in static)
    component_rows=''
    for arm in ARMS:
        for m in summaries[arm]['component_statistics']:
            values=(LABELS[arm],m['model_id'],m['registered_views'],m['raw_points'],m['raw_two_view_points'],
                    m['raw_multi_view_points'],m['consistent_points'],m['consistency_failed_points'])
            component_rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    trackhtml=html.escape(json.dumps(summary['shared_tracks'],ensure_ascii=False))
    details=''.join('<h3>'+html.escape(LABELS[a])+'</h3><pre>'+html.escape(json.dumps(summaries[a],ensure_ascii=False,indent=2))+'</pre>' for a in ARMS)
    page='''<!doctype html><meta charset="utf-8"><title>采样准入与几何证据对照</title><style>body{font:16px/1.6 system-ui;margin:24px}.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}canvas{width:100%}aside{background:#fff0cd;padding:16px}pre{white-space:pre-wrap}td,th{padding:7px;border:1px solid #aaa}table{border-collapse:collapse}</style>
<h1>同源观测：准入门与两视几何证据</h1><aside>二维候选、验证匹配、原始三维观测与冻结模型一致性是不同层。粉色为通过一致性检查的两视临时点，仍非准确性验收。原始模型层黄点可能未通过一致性；一致性层黄点为通过的多视点。各分量独立，未拼接为完整物体。</aside><p><a href="SUMMARY.json">全部统计、独立分量、相机参数与来源hash</a></p>
<p>共享观测数量（exactly_two_sample_tracks为恰好2次真采样的候选ID，不是3D点）：TRACKCOUNTS</p><table><tr><th>方案</th><th>独立注册帧</th><th>分量数</th><th>最大视图数</th><th>eligible中位 点/8格比例</th><th>verified中位 点/8格比例</th><th>model中位 点/8格比例</th><th>一致性中位 点/8格比例</th></tr>ROWS</table><h2>各独立分量：不可相加为完整点云</h2><table><tr><th>方案</th><th>model</th><th>视图</th><th>原始点</th><th>原始两视点</th><th>原始多视点</th><th>一致性通过</th><th>一致性失败</th></tr>COMPONENTROWS</table>
<p><input id="frame" type="range" min="0" value="0" step="1" style="width:65%"><span id="identity"></span></p><select id="stage"><option value="eligible">实际入库二维候选</option><option value="verified">验证内点（按行去重）</option><option value="model">原始模型观测</option><option value="consistent">同一分量一致性支持</option></select><div id="cols" class="cols"></div>
<script>const D=PAYLOAD;let v=0;const slider=document.getElementById('frame');slider.max=D.frames.length-1;function draw(){const token=++v,f=D.frames[+slider.value],s=document.getElementById('stage').value;document.getElementById('identity').textContent=` input${f.input_index} decode${f.decode_index} ${f.mask_status}`;const cols=document.getElementById('cols');cols.replaceChildren();for(const arm of ['control','two_sample_candidates','geometry_evidence']){const a=f.arms[arm],m=a.metrics[s],card=document.createElement('section'),h=document.createElement('h3'),p=document.createElement('p'),c=document.createElement('canvas');h.textContent=D.labels[arm];p.textContent=`${a[s].length}点，grid8 ${m?m.grid['8'].occupied+'/'+m.grid['8'].nonempty_mask_cells:'N/A'}；单一model ${a.component??'无'}；可用分量 ${JSON.stringify(a.component_options)}`;c.width=f.width;c.height=f.height;card.append(h,p,c);cols.append(card);const im=new Image();im.onload=()=>{if(v!==token)return;const ctx=c.getContext('2d');ctx.drawImage(im,0,0);let layers=[[a[s],'#ffdc00',2]];if(s==='model'||s==='consistent')layers.push([a.provisional,'#ff4682',3]);for(const [points,color,r] of layers){ctx.fillStyle=color;for(const q of points){ctx.beginPath();ctx.arc(q[0]-.5,q[1]-.5,r,0,2*Math.PI);ctx.fill()}}};im.src=f.image}}slider.oninput=draw;document.getElementById('stage').onchange=draw;draw();</script><h2>同帧静态对照</h2>PICS<h2>解释限制</h2>LIMITS<h2>独立分量与原生实际参数</h2>DETAILS'''
    page=page.replace('PAYLOAD',payload).replace('COMPONENTROWS',component_rows).replace('TRACKCOUNTS',trackhtml).replace('ROWS',rows).replace('PICS',pics).replace('DETAILS',details).replace('LIMITS','<ul>'+''.join('<li>'+html.escape(t)+'</li>' for t in limitations)+'</ul>')
    (out/'index.html').write_text(page,encoding='utf-8')
    with (out/'SUMMARY.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return summary
