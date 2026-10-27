"""Bounded camera-assumption sensitivity on frozen matches; no reconstruction."""
from collections import Counter
from pathlib import Path
import html
import json
import sqlite3

import numpy as np

from . import parallel_photo_audit as native
from .intrinsics_stability import Inputs, _matrix, _unique
from .intrinsics_sensitivity_math import analyse_pair


def _bound(inputs, declared, path):
    ref=inputs.bind(path)
    if declared.get(ref['path']) != ref:
        raise ValueError('Required source is missing or differs from prior binding: '+ref['path'])
    return ref


def audit_sensitivity(intrinsics, output):
    source=Path(intrinsics).resolve(strict=True);out=Path(output).resolve()
    inputs=Inputs();prior=inputs.read(source)
    if (prior.get('schema')!='intrinsics-stability-diagnostic-v1'
            or prior.get('status')!='INTRINSICS_DIAGNOSTIC_COMPLETE'
            or prior.get('source_unchanged') is not True):
        raise ValueError('Complete bound intrinsics diagnostic required')
    sfm=Path(prior['sfm_run']).resolve(strict=True);root=sfm.parent
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in (root,source.parent)):
        raise ValueError('fresh output outside input report and SfM required')
    receipts=prior['source_bindings']+prior['implementation']
    if len(receipts)>256:
        raise ValueError('source binding bound exceeded')
    declared={}
    for ref in receipts:
        actual=inputs.bind(ref['path'])
        if actual!=ref:raise ValueError('prior source drift: '+ref['path'])
        if actual['path'] in declared:raise ValueError('duplicate prior source binding')
        declared[actual['path']]=actual
    _bound(inputs,declared,sfm)
    fmap=root/'frames_map.jsonl';_bound(inputs,declared,fmap)
    frames=_unique(inputs.rows(fmap),'input_index')
    mapped=_unique([r for r in frames.values() if r.get('database_image_id') is not None],'database_image_id')
    run=inputs.read(sfm)
    if not 2<=len(mapped)<=96:raise ValueError('2..96 images required')
    database=root/'database.db';_bound(inputs,declared,database)
    def frozen_db():
        for suffix in ('-wal','-journal'):
            p=Path(str(database)+suffix)
            if p.exists() and p.stat().st_size:raise ValueError('active database journal')
    frozen_db()
    previous=_unique(prior['pairs'],'pair_id')
    pairs=[];cameras={};xy={};image_cameras={}
    db=sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        image_rows=db.execute('SELECT image_id,name,camera_id FROM images ORDER BY image_id').fetchall()
        if {r[0] for r in image_rows}!=set(mapped):raise ValueError('image identity set differs')
        for cid,model,w,h,blob in db.execute('SELECT camera_id,model,width,height,params FROM cameras'):
            if len(cameras)>=96 or len(blob)>128 or len(blob)%8:raise ValueError('camera bound')
            params=np.frombuffer(blob,dtype='<f8')
            if not np.isfinite(params).all():raise ValueError('nonfinite camera')
            cameras[cid]=dict(model=model,width=w,height=h,params=params.tolist())
        for iid,name,cid in image_rows:
            row=mapped[iid]
            if row['staged_image_name']!=name or row['upstream_io_run_id']!=run['upstream_io_run_id']:
                raise ValueError('frame map identity differs')
            image=_bound(inputs,declared,native.under(root,name))
            if image['sha256']!=row['canonical_image_sha256']:raise ValueError('canonical identity differs')
            image_cameras[iid]=cid
            r=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(iid,)).fetchone()
            if not r or not 0<=r[0]<=20000 or r[1] not in (2,4,6) or len(r[2])!=r[0]*r[1]*4:
                raise ValueError('keypoint representation/bound differs')
            xy[iid]=np.frombuffer(r[2],dtype='<f4').reshape(r[0],r[1])[:,:2]
            if not np.isfinite(xy[iid]).all():raise ValueError('nonfinite keypoint')
        rows=db.execute('SELECT pair_id,rows,cols,data,config,F FROM two_view_geometries ORDER BY pair_id').fetchmany(4609)
        if len(rows)>4608 or {r[0] for r in rows}!=set(previous):raise ValueError('stored pair set differs')
        for pid,n,cols,blob,config,fb in rows:
            a,b=divmod(pid,2147483647)
            if a not in mapped or b not in mapped or not a<b:raise ValueError('invalid pair image identity')
            p=previous[pid]
            indices=[mapped[a]['input_index'],mapped[b]['input_index']]
            if p['image_ids']!=[a,b] or p['input_indices']!=indices or p['config']!=config or p['inlier_count']!=n:
                raise ValueError('prior pair identity/count/config differs')
            if n==0:continue
            if not 0<n<=4096 or cols!=2 or len(blob)!=n*8:raise ValueError('match representation/bound differs')
            ids=np.frombuffer(blob,dtype='<u4').reshape(n,2)
            if (ids[:,0]>=len(xy[a])).any() or (ids[:,1]>=len(xy[b])).any():raise ValueError('feature identity out of range')
            F=_matrix(fb)
            if p.get('fundamental') is not None and not np.array_equal(F,np.array(p['fundamental'])):
                raise ValueError('stored fundamental differs from prior report')
            ca,cb=cameras[image_cameras[a]],cameras[image_cameras[b]]
            pair=dict(pair_id=pid,image_ids=[a,b],input_indices=indices,source_configuration=config,inlier_count=n,
                      source_fundamental=F.tolist() if F is not None else None,
                      source_selection='Previously verified inliers, not independent of original F/H selection')
            if image_cameras[a]!=image_cameras[b] or ca!=cb or ca['model']!=2 or len(ca['params'])!=4 or ca['params'][3]!=0:
                pair['analysis']=dict(status='SENSITIVITY_REFUSED',reason='shared SIMPLE_RADIAL initial k=0 required')
            else:
                pair['analysis']=analyse_pair(xy[a][ids[:,0]],xy[b][ids[:,1]],ids,
                    ca['params'][1:3],[ca['width'],ca['height']],ca['params'][0],stored_f=F,config=config)
            pairs.append(pair)
    finally:
        db.close()
    frozen_db();inputs.verify()
    summary=summarize(pairs)
    result=dict(schema='intrinsics-sensitivity-v1',status='INTRINSICS_SENSITIVITY_COMPLETE',source_unchanged=True,
        sfm_run=str(sfm),prior_diagnostic=str(source),pairs=pairs,summary=summary,
        policy=dict(principal_offsets='center, +/-2% width on cx, +/-2% height on cy; fixed probes, not uncertainty bounds',
                    k_scenarios=[-0.1,0.0,0.1],reference_focal='database initial focal, fixed during point undistortion',
                    radial_model='SIMPLE_RADIAL; invert each pixel on monotonic branch; refit F/H, no linear F distortion transform',
                    split='lexicographic feature-pair identities, even/odd two complementary folds; train>=12, holdout>=12',
                    fits='normalized eight-point F, normalized DLT H, proper rotation on reference-K unit rays',
                    homography_compatibility='both folds >=80% at <=2 corrected pixels and no invalid predictions; diagnostic only'),
        decision=dict(calibration_accepted=False,new_ba_authorized=False,geometry_accepted=False,
                      recommended_camera_params=None,adequate_triangulation_parallax='NOT_ESTABLISHED',
                      reason='Sensitivity and homography compatibility do not establish physical intrinsics or 3D depth'),
        limitations=['Existing inlier selection precedes splitting; these are conditional internal holdouts, not independent ground truth',
                     'No best k is selected; error units after nonlinear correction belong to each scenario',
                     'A homography can explain planar, pure-rotation, distant, shallow-depth or other weak-baseline data',
                     'Failure of H or rotation fit does not prove sufficient parallax; matches or camera assumptions can also be wrong',
                     'No fundamental matrix refit is written into the native database; original configurations are retained',
                     'Principal point and k are probed separately, not jointly estimated; no temporal zoom conclusion'],
        native_model_executions=0,geometry_changed=False,scale_status='unknown',
        source_bindings=list(inputs.refs.values()),implementation=[native.binding(Path(__file__)),
            native.binding(Path(__file__).with_name('intrinsics_sensitivity_math.py'))])
    out.mkdir(parents=True,exist_ok=False)
    native.write_json(out/'SENSITIVITY.json',result)
    render(out,result)
    return result


def summarize(pairs):
    states=Counter();by_k={str(k):Counter() for k in (-.1,0.,.1)};principal=[]
    for p in pairs:
        a=p['analysis'];states[a['status']]+=1
        profiles=a.get('principal_profiles',[])
        focal=[r['summary'].get('minimum',{}).get('focal') for r in profiles]
        finite=[f for f in focal if f is not None]
        principal.append(dict(input_indices=p['input_indices'],
            complete_five=len(profiles)==5 and len(finite)==5,
            grid_minimum_focal_min=min(finite) if finite else None,
            grid_minimum_focal_max=max(finite) if finite else None,
            ratio=max(finite)/min(finite) if finite else None,
            states=[r['summary'].get('focal_observability_status',r['summary']['status']) for r in profiles]))
        for s in a.get('distortion_scenarios',[]):
            state=s.get('homography_compatible')
            by_k[str(s['k'])]['compatible' if state is True else 'not_compatible' if state is False else 'not_evaluated']+=1
    return dict(nonempty_pairs=len(pairs),analysis_statuses=dict(states),
                homography_compatibility={k:dict(v) for k,v in by_k.items()},principal_point_sensitivity=principal)


def render(out,result):
    esc=lambda x:html.escape(str(x))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>内参假设与退化风险</title>
<style>body{font:16px/1.65 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#26394c}aside{background:#fff0cc;padding:16px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #dce3e8;padding:8px;text-align:left}details{margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#137fa4}</style>
<h1>主点、畸变假设与退化风险</h1><aside>这是同源匹配的条件留出诊断。未标定相机，未改变模型；可用于三角化的视差尚未建立。</aside>
<p>书中要求不同投影方向和同名点；当前透视图像还需检查相机假设。主点按宽/高±2%分别试探，畸变按k=−0.1、0、+0.1分别去畸变再拟合；这些值不是标定范围或最佳参数。</p>
<h2>主点扰动：同一原始F的焦距网格最低值</h2><p>区间跨越或边界最低说明这项必要条件有风险；区间窄也不证明校准正确。未统计置信区间。</p><table><tr><th>原始帧对</th><th>5场景最低f范围(px)</th><th>最大/最小</th><th>各场景状态</th></tr>'''
    for p in result['summary']['principal_point_sensitivity']:
        vals=[p['input_indices'],f"{p['grid_minimum_focal_min']:.1f}–{p['grid_minimum_focal_max']:.1f}" if p['ratio'] else '未计算',f"{p['ratio']:.2f}" if p['ratio'] else '—',', '.join(p['states'])]
        page+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in vals)+'</tr>'
    page+='</table><h2>单应模型留出检查</h2><p>每个场景固定原始特征ID分成两个互补折。两折都至少80%的留出点在2个校正后像素内、无无效预测，才标“相容”。这不等于物体是平面或相机只旋转；不相容也不能证明深度可靠。</p><table><tr><th>原始帧对 / 内点</th><th>k=−0.1</th><th>k=0</th><th>k=+0.1</th></tr>'
    for p in result['pairs']:
        cells=[str(p['input_indices'])+' / '+str(p['inlier_count'])]
        for s in p['analysis'].get('distortion_scenarios',[]):
            v=s.get('homography_compatible');cells.append('相容' if v is True else '不相容' if v is False else '未评估')
        while len(cells)<4:cells.append('未评估')
        page+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in cells)+'</tr>'
    page+='</table><h2>逐图对证据</h2><p>F的Sampson距离和H的双向传递距离定义不同，不直接按数值大小选模型。旋转角残差依赖启发式内参，不能当作三角化视差。</p>'
    for p in result['pairs']:
        compact=dict(source_configuration=p['source_configuration'],status=p['analysis']['status'],scenarios=[])
        for s in p['analysis'].get('distortion_scenarios',[]):
            compact['scenarios'].append(dict(k=s['k'],status=s['status'],homography_compatible=s.get('homography_compatible'),
                folds=[dict(status=f['status'],train_count=len(f['training_feature_pairs']),heldout_count=len(f['heldout_feature_pairs']),
                    models={name:{k:v for k,v in m.items() if k!='matrix'} for name,m in f['models'].items()}) for f in s['folds']]))
        page+='<details><summary>'+esc(p['input_indices'])+'：训练/留出数与F/H/R结果</summary><pre>'+esc(json.dumps(compact,ensure_ascii=False,indent=2))+'</pre></details>'
    page+='<p><a href="SENSITIVITY.json">完整数据、所有主点曲线与来源哈希</a> · <a href="https://users.cecs.anu.edu.au/~hartley/Papers/fundamental/ICCV-final/fundamental.pdf">Hartley归一化八点法</a> · <a href="https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/sensor/models.h">COLMAP 4.2.1镜头模型</a></p></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
