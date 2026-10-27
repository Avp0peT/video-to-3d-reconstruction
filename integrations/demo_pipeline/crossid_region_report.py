"""Read saved explicit cross-ID matches and native results; no solver calls."""
from pathlib import Path
from collections import Counter
import json
import html
from .balanced_point_report import _read, _sha


def render(run_dir):
    """Build an exclusive report after root RESULT and GRAPH_AUDIT are frozen."""
    import numpy as np
    from PIL import Image, ImageDraw
    out = Path(run_dir).resolve(strict=True)
    if any((out/p).exists() for p in ('SUMMARY.json','index.html','report_assets')):
        raise ValueError('fresh report required')
    expected = {}
    def bind(path, digest=None, size=None):
        p=Path(path).resolve(strict=True); h=_sha(p)
        if digest is not None and h!=digest: raise ValueError('source digest mismatch: '+str(p))
        if size is not None and p.stat().st_size!=size: raise ValueError('source byte mismatch')
        if str(p) in expected and expected[str(p)]!=h: raise ValueError('source changed')
        expected[str(p)]=h; return p
    def read(path):return _read(bind(path))
    def refs(doc,key='source_bindings'):
        for r in doc.get(key,[]):
            bind(r['path'],r['sha256'],r.get('bytes'))
    # The supervising root appends report ENTRY/final status after this child.
    # It is an execution envelope, not an immutable algorithm input.
    result=_read(out/'RESULT.json')
    if (out/'PLAN.json').is_file():
        plan=read(out/'PLAN.json'); refs(plan); refs(plan,'implementation')
    tracks=read(result['tracks_path']); refs(tracks)
    supplement=read(result['supplement_matches_path']);refs(supplement)
    if supplement.get('schema')!='crossid-region-matches-v1':raise ValueError('unknown matches schema')
    if 'tracks_binding' in supplement:
        r=supplement['tracks_binding'];bind(r['path'],r['sha256'],r.get('bytes'))
        if Path(r['path']).resolve()!=Path(result['tracks_path']).resolve():raise ValueError('supplement uses other tracks')
    graph=read(result['graph_audit_path']);refs(graph)
    native={};plans={};stats={}
    for arm,prefix in [('control','control_'),('crossid','')]:
        native[arm]=read(result[prefix+'native_result_path']);plans[arm]=read(result[prefix+'native_plan_path']);refs(plans[arm])
        n=native[arm]; models=n['models']; cameras=[]
        for m in models:
            for c in m['cameras']:
                p=c['params']
                if c['model']!='SIMPLE_RADIAL' or len(p)!=4 or not np.isfinite(p).all():raise ValueError('unexpected camera contract')
                cameras.append(dict(model_id=m['model_id'],registered_views=m['registered_views'],camera_id=c['camera_id'],
                    focal=p[0],radial_k=p[3],focal_over_initial=p[0]/n['actual_options']['camera_initial_params'][0],
                    focal_over_max_image_side=p[0]/max(tracks['frames'][0]['width'],tracks['frames'][0]['height']),
                    interpretation='Estimated parameter only; no invented accuracy/acceptable-focal threshold.'))
        unique=sorted({i for m in models for i in m['registered_input_indices']})
        if len(unique)!=n['unique_registered_count']:raise ValueError('unique registered count mismatch')
        stats[arm]=dict(status=n['status'],unique_registered_count=len(unique),unique_registered_indices=unique,
            native_components=len(models),largest_component_views=max((m['registered_views'] for m in models),default=0),
            raw_pair_match_rows=sum(p['raw_matches'] for p in n['pairs']),verified_pair_match_rows=sum(p['inliers'] for p in n['pairs']),
            verified_nonzero_pairs=sum(p['inliers']>0 for p in n['pairs']),
            cameras=cameras,focal_range=None if not cameras else [min(c['focal'] for c in cameras),max(c['focal'] for c in cameras)],
            radial_k_range=None if not cameras else [min(c['radial_k'] for c in cameras),max(c['radial_k'] for c in cameras)],
            nonzero_k_abs_gt_1e_10=sum(abs(c['radial_k'])>1e-10 for c in cameras),actual_options=n['actual_options'],
            models=[{k:m[k] for k in ('model_id','registered_views','registered_input_indices','points','cameras')} for m in models])
    ledgers=plans['crossid']['frames']; source=tracks['frames']
    if len(source)!=len(ledgers):raise ValueError('source/native frame count differs')
    oldframes=plans['control']['frames']
    if len(oldframes)!=len(ledgers):raise ValueError('control/new eligible frame count differs')
    for old,new in zip(oldframes,ledgers):
        if any(old[k]!=new[k] for k in ('input_index','frame_id','decode_index','track_ids','keypoint_rows')):
            raise ValueError('control/new original feature row ledger changed')
    pointrows={}
    for seq,(f,l) in enumerate(zip(source,ledgers)):
        if any(f[k]!=l[k] for k in ('input_index','frame_id','decode_index')):raise ValueError('native/source identity mismatch')
        obs={o['track_id']:o for o in f['observations']}
        if len(obs)!=len(f['observations']):raise ValueError('duplicate track observation')
        pointrows[seq]=[(np.asarray(obs[t]['xy'],float)+.5).astype(np.float32).astype(float).tolist() for t in l['track_ids']]
    # Preserve rows exactly. Do not infer image IDs from source input_index.
    idseq={l['database_image_id']:s for s,l in enumerate(ledgers)}
    verified={}
    for p in native['crossid']['pairs']:
        a,b=idseq[p['image_a']],idseq[p['image_b']]
        verified[a,b]={tuple(r) for r in p['inlier_feature_rows']}
    submitted={}
    for p in plans['crossid']['candidate_pairs']:
        key=(p['sequence_a'],p['sequence_b'])
        submitted[key]={tuple(r) for r in p['raw_feature_rows']}
    gp={(p['sequence_a'],p['sequence_b']):p for p in graph['supplement_pairs']}
    pairs=[]
    for position,p in enumerate(supplement['pairs']):
        a,b=p['sequence_a'],p['sequence_b'];la,lb=ledgers[a],ledgers[b]
        rows_a={t:r for r,t in enumerate(la['track_ids'])};rows_b={t:r for r,t in enumerate(lb['track_ids'])}
        proposed=[]
        for m in p['matches']:
            ta,tb=m['track_id_a'],m['track_id_b']
            if ta==tb:raise ValueError('supplement contains same-ID row')
            proposed.append((rows_a[ta],rows_b[tb]))
        if len(set(proposed))!=len(proposed):raise ValueError('duplicate cross-ID row pair')
        used=[r for r in proposed if r in submitted.get((a,b),set())]
        good=[r for r in used if r in verified.get((a,b),set())]
        info=gp.get((a,b),{})
        pairs.append(dict(sequence_a=a,sequence_b=b,input_index_a=la['input_index'],input_index_b=lb['input_index'],
            candidate_order=position,proposed_crossid=len(proposed),submitted_crossid=len(used),verified_crossid=len(good),
            cross_control_components=info.get('cross_control_components'),raw_feature_rows=[list(r) for r in used],
            verified_feature_rows=[list(r) for r in good]))
    # First four crossing candidates retain declared candidate order; then strongest
    # verified crossing pairs, then strongest remaining pairs. Deduplicate, <=12.
    crossing=[p for p in pairs if p['cross_control_components'] is True]
    rank=lambda p:(-p['verified_crossid'],p['candidate_order'])
    selected=[];seen=set()
    for p in crossing[:4]+sorted(crossing,key=rank)+sorted(pairs,key=rank):
        key=p['sequence_a'],p['sequence_b']
        if key not in seen:selected.append(p);seen.add(key)
        if len(selected)==12:break
    assets=out/'report_assets';assets.mkdir();links=[]
    def image(seq):
        f=source[seq]
        with Image.open(bind(f['image_path'],f['image_sha256'])) as im:base=im.convert('RGB')
        with Image.open(bind(f['mask_path'],f['mask_sha256'])) as im:mask=np.asarray(im)>0
        edge=mask.copy();edge[1:-1,1:-1]&=~(mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:])
        px=np.asarray(base).copy();px[edge]=[0,255,255];return Image.fromarray(px)
    figures=[]
    for p in selected:
        a,b=p['sequence_a'],p['sequence_b'];left,right=image(a),image(b)
        for stage,rows in [('submitted_raw',p['raw_feature_rows']),('native_verified',p['verified_feature_rows'])]:
            canvas=Image.new('RGB',(left.width+right.width,max(left.height,right.height)+44),'white')
            canvas.paste(left,(0,44));canvas.paste(right,(left.width,44));d=ImageDraw.Draw(canvas)
            d.text((5,5),f'input{p["input_index_a"]} - input{p["input_index_b"]} / {stage}: {len(rows)} explicit cross-ID rows',fill='black')
            d.text((5,23),'cyan=mask; orange=raw candidates; green=native verified; neither proves identity/shape',fill='black')
            color=(255,160,20) if stage=='submitted_raw' else (30,255,80)
            for ra,rb in rows:
                x,y=pointrows[a][ra];u,v=pointrows[b][rb]
                x,y,u,v=x-.5,y+43.5,u-.5+left.width,v+43.5
                d.line((x,y,u,v),fill=color,width=1)
                for xx,yy in [(x,y),(u,v)]:d.ellipse((xx-2,yy-2,xx+2,yy+2),fill=color)
            path=assets/f'pair_{a:03d}_{b:03d}_{stage}.png';canvas.save(path);links.append(path)
            figures.append(dict(sequence_a=a,sequence_b=b,stage=stage,path='report_assets/'+path.name,count=len(rows)))
    limitations=['跨track ID为显式候选对应，保留原ID与feature row，不融合出生ID，也不把raw当作几何验证。',
        '几何内点仅为所拟合模型的内部一致性，不是held-out验证或物理身份真值；新增连接不代表正确外形。',
        '独立模型分量不可拼接或累加为完整点云；注册帧按input_index去重。',
        '焦距、径向参数列实际估计和相对初始化比，不捏造合理区间；k=0不是完成标定，低重投影误差不证明准确。',
        '图中只显示实际提交raw及其原生验证子集，排除冲突候选不冒充进入native；最多12对，未显示不等于无数据。',
        '工程局部匹配与透视几何验证不是Kozlov原算法；局部候选的一致不能替代整组全局对应与投影前提。']
    summary=dict(schema='crossid-region-report-v1',status='REPORT_GENERATED_NOT_ACCEPTED',run_status=result.get('status'),
        execution_envelope=dict(path=str(out/'RESULT.json'),state='FINALIZING',immutable_source=False,
            note='Root may append report ENTRY and final status after this report. Consult final RESULT for execution acceptance.'),
        arms=stats,graph_audit=graph,pair_statistics=pairs,figures=figures,
        selection_policy='First four declared crossing candidates, then crossing pairs by verified inliers descending/candidate-order tie, then remaining strongest; unique <=12.',
        proposed_crossid_rows=sum(p['proposed_crossid'] for p in pairs),submitted_crossid_rows=sum(p['submitted_crossid'] for p in pairs),
        verified_crossid_rows=sum(p['verified_crossid'] for p in pairs),limitations=limitations,
        eligible_feature_rows_control_equal=True,geometry_accepted=False,heldout_validation=False,components_merged=False,scale_status='unknown',
        source_bindings=[dict(path=p,sha256=h) for p,h in sorted(expected.items())],source_drift=[p for p,h in expected.items() if _sha(p)!=h],
        link_check=dict(checked=len(links)+2,missing=[str(p) for p in links+[out/'RESULT.json'] if not p.is_file()]))
    if summary['source_drift'] or summary['link_check']['missing']:raise ValueError('report sources/links changed')
    esc=lambda x:html.escape(json.dumps(x,ensure_ascii=False,indent=2))
    pics=''.join('<h3>'+html.escape(f'input {ledgers[f["sequence_a"]]["input_index"]} → input {ledgers[f["sequence_b"]]["input_index"]} · {f["stage"]} ({f["count"]})')+'</h3><a href="'+f['path']+'">打开原尺寸PNG<img style="max-width:100%" src="'+f['path']+'"></a>' for f in figures)
    def count(value):
        return len(value) if isinstance(value,(list,dict,set)) else value
    rows=''
    for arm,label in [('control','原：全共享ID、固定k0'),('crossid','新增跨ID匹配')]:
        g=graph.get(arm,{})
        raw=g.get('raw',{});verified_graph=g.get('verified',{})
        values=[label,count(raw.get('edges')),count(verified_graph.get('edges')),
                count(raw.get('components')),count(verified_graph.get('components')),
                count(raw.get('isolated')),count(verified_graph.get('isolated')),
                stats[arm]['unique_registered_count'],stats[arm]['native_components'],stats[arm]['largest_component_views']]
        rows+='<tr>'+''.join('<td>'+html.escape(str(v) if v is not None else 'unknown')+'</td>' for v in values)+'</tr>'
    camera_rows=''
    for arm,label in [('control','原'),('crossid','新')]:
        for c in stats[arm]['cameras']:
            values=(label,c['model_id'],c['registered_views'],round(c['focal'],4),round(c['radial_k'],8),round(c['focal_over_initial'],4),round(c['focal_over_max_image_side'],4))
            camera_rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    headline=f'跨ID匹配行数：提议 {summary["proposed_crossid_rows"]} → 实际提交 {summary["submitted_crossid_rows"]} → 原生验证 {summary["verified_crossid_rows"]}。跨旧组件验证pair数：{count(graph.get("new_cross_component_verified_pairs","unknown"))}。'
    page='<!doctype html><meta charset="utf-8"><title>跨ID局部对应与原生连接证据</title><style>body{font:16px/1.6 system-ui;margin:24px}pre{white-space:pre-wrap}aside{background:#fff0cd;padding:18px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}details{margin:16px 0}</style><h1>跨ID区域匹配：候选与验证分开</h1><aside>新增raw线不等于可靠连接；绿色原生内点也不等于正确物体形状。保留所有原ID/row，独立分量不合并，参数未标定，几何未质量验收。报告执行时外层RESULT仍在收尾，请以最终RESULT及ENTRY判断执行状态。</aside><p><a href="SUMMARY.json">精确统计、路径和hash</a> · <a href="RESULT.json">最终执行记录</a></p><h2>原/新连接与注册</h2><p>'+html.escape(headline)+'</p><table><tr><th>方案</th><th>raw边</th><th>验证边</th><th>raw图组件</th><th>验证图组件</th><th>raw孤点</th><th>验证孤点</th><th>去重注册帧</th><th>模型分量</th><th>最大分量视图</th></tr>'+rows+'</table><details><summary>各分量实际焦距与径向参数（估计值，非准确性）</summary><table><tr><th>方案</th><th>model</th><th>视图</th><th>f</th><th>k</th><th>f/初值</th><th>f/maxside</th></tr>'+camera_rows+'</table></details><h2>真实显式cross-ID线（最多12对）</h2><p>'+html.escape(summary['selection_policy'])+'</p>'+pics+'<h2>边界</h2><ul>'+''.join('<li>'+html.escape(t)+'</li>' for t in limitations)+'</ul><details><summary>完整图审计、来源路径与绑定</summary><pre>'+esc(graph)+'</pre></details><details><summary>全部模型、相机与实际native参数</summary><pre>'+esc(stats)+'</pre></details>'
    (out/'index.html').write_text(page,encoding='utf-8')
    with (out/'SUMMARY.json').open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
    return summary


build = render
