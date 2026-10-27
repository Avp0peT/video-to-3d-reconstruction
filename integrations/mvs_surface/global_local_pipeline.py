"""Preserve a whole reconstruction and test one bound reference-domain repair.

This is an experimental CPU depth search, not an OpenMVS prior interface.
Only proposals checked against multiple existing views are added to holes.
"""
from pathlib import Path
import hashlib
import html
import json
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from integrations.openmvs_depth.dmap import read_dmap
from integrations.viewer.ply_data import read_ply
from integrations.viewer.export import export_html
from .mask_consumption_probe import authenticate_depth
from .local_depth_search import propose_local_depth


def binding(path):
    p=Path(path).resolve(strict=True); stat=p.stat(); h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if p.stat().st_size!=stat.st_size or p.stat().st_mtime_ns!=stat.st_mtime_ns:
        raise ValueError('file drift while binding')
    return dict(path=str(p),bytes=stat.st_size,sha256=h.hexdigest())


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load(path):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    value=json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,
        parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    def finite(v):
        if isinstance(v,float) and not np.isfinite(v):raise ValueError('nonfinite JSON')
        if isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,list):
            for x in v:finite(x)
    finite(value);return value


def historical_plan_binding(prepared):
    path=Path(prepared['source_plan']).resolve(strict=True)
    found=[b for b in prepared['refs'] if Path(b['path']).resolve()==path]
    if len(found)!=1:raise ValueError('exact historical source-plan binding required')
    actual=binding(path)
    if any(actual[k]!=found[0][k] for k in ('bytes','sha256')):raise ValueError('historical source plan changed')
    return actual


def add_missing_only(whole,proposal,accepted,roi,allowed):
    """Never delete or move a previously present global measurement."""
    if whole.shape!=proposal.shape or any(x.shape!=whole.shape or x.dtype!=bool for x in (accepted,roi,allowed)):
        raise ValueError('inconsistent combination arrays')
    if whole.dtype.kind!='f' or proposal.dtype.kind!='f' or not np.isfinite(whole).all() or not np.isfinite(proposal).all() or (whole<0).any() or (proposal<0).any():
        raise ValueError('finite nonnegative axial depths required')
    if (accepted & ((proposal<=0)|~roi|~allowed)).any():raise ValueError('accepted proposal outside allowed ROI')
    added=accepted&(whole==0); out=whole.copy();out[added]=proposal[added]
    assert np.array_equal(out[whole>0],whole[whole>0])
    return out,added


def _binary(ref,shape,refs):
    actual=binding(ref['path'])
    if actual!=ref:raise ValueError('mask binding changed')
    refs[actual['path']]=actual
    with Image.open(actual['path']) as im:a=np.asarray(im).copy()
    if a.shape!=shape or a.dtype!=np.uint8 or not np.isin(a,[0,255]).all():raise ValueError('binary image domain differs')
    return a!=0


def _roi(spec,decode,allowed):
    if type(spec.get('decode_index')) is not int or spec['decode_index']!=decode:raise ValueError('ROI reference identity differs')
    out=np.zeros(allowed.shape,bool);h,w=allowed.shape
    cells=spec.get('cells')
    if not isinstance(cells,list) or not 1<=len(cells)<=1024:raise ValueError('bounded explicit cells required')
    for row in cells:
        b=row.get('bounds')
        if not isinstance(b,list) or len(b)!=4 or any(type(v) is not int for v in b):raise ValueError('invalid ROI cell')
        x0,y0,x1,y1=b
        if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('ROI cell outside image')
        out[y0:y1,x0:x1]=True
    return out&allowed


def _write_cloud(path,xyz,colors):
    xyz=np.asarray(xyz,dtype=np.float64).reshape(-1,3);colors=np.asarray(colors,dtype=np.uint8).reshape(-1,3)
    if len(xyz)!=len(colors) or not np.isfinite(xyz).all():raise ValueError('invalid point cloud')
    dtype=np.dtype([('xyz','<f8',(3,)),('rgb','u1',(3,))]);rows=np.empty(len(xyz),dtype=dtype)
    rows['xyz']=xyz;rows['rgb']=colors
    header=('ply\nformat binary_little_endian 1.0\ncomment experimental geometry; not accepted\n'
        f'element vertex {len(xyz)}\nproperty double x\nproperty double y\nproperty double z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('xb') as f:f.write(header.encode('ascii'));f.write(rows.tobytes())
    read=read_ply(path,max_bytes=16<<20,max_vertices=200000,max_faces=1)
    if read['faces']:raise ValueError('unexpected faces in point cloud')
    if not np.array_equal(np.asarray(read['vertices']).reshape(-1,3),xyz) or not np.array_equal(np.asarray(read['colors']).reshape(-1,3),colors):
        raise ValueError('cloud exact readback differs')
    return dict(**binding(path),vertices=len(xyz),exact_readback=True)


def _overlay(path,rgb,roi,panels):
    h,w=rgb.shape[:2];canvas=Image.new('RGB',(w*len(panels),h+38),(20,30,42));draw=ImageDraw.Draw(canvas)
    edge=cv2.morphologyEx(roi.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))!=0
    for i,(title,layers) in enumerate(panels):
        arr=rgb.copy()
        for mask,color in layers:arr[mask]=(.35*arr[mask]+.65*np.array(color)).astype(np.uint8)
        arr[edge]=[245,220,50];canvas.paste(Image.fromarray(arr),(i*w,38));draw.text((i*w+8,10),title,fill='white')
    canvas.save(path)


def run(source_experiment,reference_id,roi_spec,output,*,seconds=300,method='small-patch-v1'):
    if method not in ('small-patch-v1','regional-plane-v2'):raise ValueError('unsupported local method')
    start=time.monotonic();out=Path(output).resolve();exp=Path(source_experiment).resolve(strict=True)
    roi_path=Path(roi_spec).resolve(strict=True)
    if type(reference_id) is not int or reference_id<0 or type(seconds) is not int or not 1<=seconds<=900:
        raise ValueError('invalid reference or CPU budget')
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in (exp,roi_path.parent)):
        raise ValueError('fresh disjoint output required')
    refs={}
    def bound(path,expected=None):
        now=binding(path)
        if expected is not None and any(now[k]!=expected[k] for k in ('bytes','sha256')):raise ValueError('input binding drift: '+str(path))
        if now['path'] in refs and refs[now['path']]!=now:raise ValueError('input changed between reads')
        refs[now['path']]=now;return now
    def read(path):
        before=bound(path);v=load(path)
        if binding(path)!=before:raise ValueError('JSON drift')
        return v
    def check():
        if time.monotonic()-start>seconds:raise TimeoutError('global/local CPU time limit')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>512<<20:raise RuntimeError('512MiB output limit')
    previous=read(exp/'RESULT.json');prepared=read(exp/'PREPARED.json')
    if previous['status']!='REGIONAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED' or not previous['arms']['WHOLE']['execution_ok']:
        raise ValueError('existing completed WHOLE experiment required')
    historic=historical_plan_binding(prepared);bound(historic['path'],historic)
    plan=read(historic['path']);spec=read(roi_path)
    cameras={f['image']['global_id']:f for f in plan['frames']}
    if len(cameras)!=len(plan['frames']) or reference_id not in cameras:raise ValueError('camera identities differ')
    selected={v['global_id']:v for v in prepared['views']}
    if set(selected)!=set(cameras):raise ValueError('selection identities differ')
    maps={};all_depth_bindings=[]
    for b in previous['arms']['WHOLE']['depth_bindings']:
        check();bound(b['path'],b);d=read_dmap(b['path']);iid=d.view_ids[0]
        if iid in maps or iid not in cameras:raise ValueError('unknown/duplicate depth camera')
        authenticate_depth(d,cameras[iid]['image']);maps[iid]=d;all_depth_bindings.append(b)
    if reference_id not in maps:raise ValueError('reference depth missing')
    refd=maps[reference_id];neighbors=list(refd.view_ids[1:])[:5]
    used=[i for i in neighbors if i in maps]
    if len(used)<2:raise ValueError('fewer than two available bound neighbor depths')
    loaded={}
    for iid in [reference_id,*used]:
        check();f=cameras[iid];d=maps[iid];image_ref=bound(f['image_file']['path'],f['image_file'])
        with Image.open(image_ref['path']) as im:rgb=np.array(im.convert('RGB'))
        if rgb.shape[:2]!=d.depth.shape:raise ValueError('v1 requires native image/depth grid; no implicit resize')
        mask=_binary(f['mask'],d.depth.shape,refs);valid=_binary(f['validity'],d.depth.shape,refs)
        loaded[iid]=dict(view_id=iid,K=d.K,R=d.R,C=d.C,image=rgb,mask=mask&valid,validity=valid,depth=d.depth)
    ref=loaded[reference_id];sources=[loaded[i] for i in used];roi=_roi(spec,selected[reference_id]['decode_index'],ref['mask'])
    # Retain all global measurements, even those whose quality is unverified.
    baseline_cloud=exp/'WHOLE/work/__mvs_dense.ply';cloud_binding=bound(baseline_cloud)
    base=read_ply(baseline_cloud,max_bytes=16<<20,max_vertices=100000,max_faces=1)
    if base['faces'] or base['colors'] is None:raise ValueError('colored point cloud baseline required')
    root=Path(__file__).resolve().parents[2]
    for rel in ['integrations/mvs_surface/global_local_pipeline.py','integrations/mvs_surface/local_depth_search.py',
                'integrations/mvs_surface/mask_consumption_probe.py','integrations/openmvs_depth/dmap.py',
                'integrations/viewer/export.py','integrations/viewer/ply_data.py',
                'scripts/refine_local_mvs_depth.py','scripts/run_region_mask_review.py']:
        bound(root/rel)
    if method=='regional-plane-v2':
        if len(sources)<4:raise ValueError('regional-plane mode requires four source depths including a diagnostic source')
        bound(root/'integrations/mvs_surface/regional_plane_search.py')
    out.mkdir(parents=True);cv2.setNumThreads(2)
    intent=dict(schema='global-local-depth-experiment-v1',method=method,reference_id=reference_id,
        decode_index=selected[reference_id]['decode_index'],source_ids=used,
        missing_source_ids=[i for i in neighbors if i not in maps],missing_global_depth_ids=sorted(set(cameras)-set(maps)),
        roi_pixels=int(roi.sum()),roi_origin='Previously fixed posthoc evaluation cells, not anatomical truth',
        depth_bounds=[refd.d_min,refd.d_max],depth_bounds_semantics='Original native range; outside-range shape untested',
        source_mask_quality='NOT_EVALUATED',combine_policy='KEEP_ALL_WHOLE_ADD_VERIFIED_HOLES_ONLY',
        physical_scale='unknown',geometry_accepted=False,threads=2,seconds=seconds,
        whole_cloud_binding_status='CURRENT_CAPTURE_NOT_HISTORIC_BINDING',
        source_plan_historical_binding=historic,
        diagnostic_sentinels={'best_photo_cost':'positive infinity = unavailable',
                              'uniqueness_gap':'negative infinity = unavailable'},
        input_bindings=list(refs.values()),argv=sys.argv)
    save(out/'PLAN.json',intent)
    status=dict(status='RUNNING',geometry_accepted=False,default_replaced=False,new_openmvs_entries=0,new_sam_inferences=0)
    save(out/'RESULT.json',status)
    try:
        Image.fromarray(roi.astype(np.uint8)*255).save(out/'01_roi.png')
        _overlay(out/'01_input.png',ref['image'],roi,[('REFERENCE / ROI',[]),('WHOLE DEPTH',[(ref['depth']>0,(30,210,80))])])
        if method=='regional-plane-v2':
            from .regional_plane_search import propose_regional_depth
            result=propose_regional_depth(ref,sources,roi,[refd.d_min,refd.d_max],check=check)
        else:
            result=propose_local_depth(ref,sources,roi,[refd.d_min,refd.d_max],check=check)
        combined,added=add_missing_only(ref['depth'],result['proposal_depth'],result['accepted'],roi,ref['mask'])
        present=roi&(ref['depth']>0); overlap=result['accepted']&present
        conflicts=overlap&(abs(result['proposal_depth']-ref['depth'])>.02*ref['depth'])
        np.save(out/'whole_depth.npy',ref['depth'],allow_pickle=False)
        np.save(out/'combined_depth.npy',combined,allow_pickle=False)
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(out/'LOCAL_PROPOSAL.npz',**arrays,added=added,roi=roi)
        for name,arr in [('02_photo_supported',roi&(result['raw_best_depth']>0)),('03_validated',result['accepted']),('04_added',added)]:
            Image.fromarray(arr.astype(np.uint8)*255).save(out/(name+'.png'))
        for path,arr in [(out/'whole_depth.npy',ref['depth']),(out/'combined_depth.npy',combined)]:
            if not np.array_equal(np.load(path,allow_pickle=False),arr):raise ValueError('saved depth readback differs')
        with np.load(out/'LOCAL_PROPOSAL.npz',allow_pickle=False) as stored:
            for k,arr in arrays.items():
                if not np.array_equal(stored[k],arr):raise ValueError('proposal readback differs')
        _overlay(out/'02_search.png',ref['image'],roi,[(('INFERRED PLANE CANDIDATES' if method=='regional-plane-v2' else 'PHOTOMETRIC BEST'),[(roi&(result['raw_best_depth']>0),(255,165,40))]),('VERIFIED PROPOSALS',[(result['accepted'],(40,210,90))]),('REJECTED PROPOSALS',[(roi&(result['raw_best_depth']>0)&~result['accepted'],(240,80,55))])])
        _overlay(out/'03_combination.png',ref['image'],roi,[('WHOLE BEFORE',[(ref['depth']>0,(30,160,230))]),('ADDED ONLY',[(added,(40,230,80))]),('WHOLE + LOCAL',[(ref['depth']>0,(30,160,230)),(added,(40,230,80))])])
        y,x=np.nonzero(added);z=combined[y,x]
        rays=np.column_stack([x,y,np.ones(len(x))])@np.linalg.inv(ref['K']).T
        xyz=(rays*z[:,None])@ref['R']+ref['C'];colors=ref['image'][y,x]
        local=_write_cloud(out/'local_additions.ply',xyz,colors)
        shutil.copyfile(baseline_cloud,out/'whole_cloud.ply')
        if binding(out/'whole_cloud.ply')['sha256']!=cloud_binding['sha256']:raise ValueError('whole cloud copy differs')
        original_xyz=np.asarray(base['vertices'],float).reshape(-1,3)
        original_rgb=np.asarray(base['colors'],np.uint8).reshape(-1,3)
        combined_cloud=_write_cloud(out/'whole_plus_local.ply',np.vstack([original_xyz,xyz]),np.vstack([original_rgb,colors]))
        viewer_spec=dict(schema='ply-viewer-scenes-v1',scenes=[])
        plus_title='Whole + inferred plane estimates' if method=='regional-plane-v2' else 'Whole + local depth observations'
        plus_note=('Regional plane estimates checked against original neighbor depths; not new texture observations.' if method=='regional-plane-v2' else 'Appended verified reference-depth holes; no deduplication, mesh or physical completeness claim.')
        for file,title,note in [('whole_cloud.ply','Whole baseline','Existing globally fused point cloud; quality not accepted.'),('whole_plus_local.ply',plus_title,plus_note)]:
            viewer_spec['scenes'].append(dict(ply=file,sha256=binding(out/file)['sha256'],title=title,
                data_kind='real_photos',declared_unit=None,note=note))
        save(out/'VIEWER_SPEC.json',viewer_spec)
        viewer=export_html(out/'VIEWER_SPEC.json',out/'viewer.html',profile='surface400k')
        for b in refs.values():
            check()
            if binding(b['path'])!=b:raise ValueError('frozen input drift')
        measures=dict(roi_pixels=int(roi.sum()),whole_before=int(present.sum()),whole_retained=int(present.sum()),
            candidate_pixels=int((roi&(result['raw_best_depth']>0)).sum()),validated_proposals=int(result['accepted'].sum()),
            added_depth_pixels=int(added.sum()),whole_after=int(((combined>0)&roi).sum()),lost_whole_pixels=0,
            proposed_existing_conflicts=int(conflicts.sum()),replaced_whole_pixels=0,
            foreground_added_outside_mask=int((added&~ref['mask']).sum()),
            remaining_unknown=int((roi&(combined==0)).sum()))
        status.update(status=('REGIONAL_PLANE_ESTIMATES_COMBINED_NOT_GEOMETRY_ACCEPTED' if method=='regional-plane-v2' else 'LOCAL_PROPOSALS_COMBINED_NOT_GEOMETRY_ACCEPTED'),method=method,metrics=measures,
            policy=result['policy'],policy_version=result['policy_version'],counters=result['counters'],
            candidate_validation=result['validation']['per_source'],prior_validation=result['prior_validation']['per_source'],
            reference_prior_used=int(result['prior_used'].sum()),
            missing_reference_depth_uses_prior=method=='regional-plane-v2',whole_constraints_on_holes=('Indirect whole prior via planes fitted to old supported anchors; no per-pixel prior in missing reference depth' if method=='regional-plane-v2' else 'Shared cameras and neighbor whole depth checks; no reference soft prior where missing'),
            local_cloud=local,combined_cloud=combined_cloud,viewer=viewer,
            new_cpu_searches=1,source_ids=used,input_drift=[],exact_array_readback=True,
            limitations=['Only one reference ROI estimated; not a complete part reconstruction.',
                'Source depths share a reconstruction and are not independent ground truth.',
                'Frontoparallel planes, fixed native range, textureless areas and deformation remain limitations.',
                'Existing whole errors are retained. The combination is additive, not corrective.',
                'Cloud additions are reference observations and can duplicate global surface measurements.',
                'No mesh fusion or boundary triangulation performed.'],
            wall_seconds=time.monotonic()-start)
        if method=='regional-plane-v2':
            # This is a proposal-stage diagnostic view, not an independently acquired ground truth.
            held=result['heldout_validation']
            arrays_held={k:v for k,v in held.items() if isinstance(v,np.ndarray)}
            np.savez_compressed(out/'HELDOUT_DIAGNOSTICS.npz',**arrays_held)
            save(out/'PLANE_FITS.json',result['per_scale_fits'])
            new_check={}
            for key in ('support_count','occluded_count','free_space_conflict_count','unknown_count','out_of_view_count','background_count'):
                new_check[key]=int(held[key][added].sum())
            assert sum(new_check.values())==int(added.sum())
            new_check['status']='EVALUATED' if sum(new_check[k] for k in ('support_count','occluded_count','free_space_conflict_count')) else 'NOT_EVALUABLE_OR_ONLY_UNKNOWN'
            status.update(proposal_kind=result['proposal_kind'],admission_source_ids=result['source_ids'],
                heldout_source_ids=result['heldout_source_ids'],heldout_original_depths_may_share_historical_inputs=True,
                heldout_validation=held['per_source'],heldout_new_pixels=new_check,
                heldout_outcomes=held['outcomes_per_source'],admission_outcomes=result['admission_outcomes']['outcomes_per_source'],
                heldout_scope=result['heldout_scope'],scale_semantics=result['scale_semantics'],
                new_regional_plane_estimations=1,new_photometric_searches=0,
                plane_fit_count=sum(len(v) for v in result['per_scale_fits'].values()),
                support_window_sizes=list(result['per_scale_fits']),
                limitations=['Multiple support-window sizes are not the full ACMM image pyramid.',
                    'Piecewise-planar estimates are not fresh texture correspondences or measured truth.',
                    'Only the current proposal/admission excludes the final source; historical depths may have used it.',
                    'Only one reference ROI, no complete physical-part segmentation or mesh fusion.',
                    'Original whole errors remain; all source depths share reconstruction assumptions.'])
            _overlay(out/'04_plane_evidence.png',ref['image'],roi,[('OLD QUALIFIED ANCHORS',[(result['anchor_mask'],(70,190,240))]),('ESTIMATED ADDITIONS',[(added,(40,230,80))]),('DIAGNOSTIC SOURCE SUPPORT',[(added&(held['support_count']>0),(50,220,90)),(added&(held['conflict_count']>0),(245,60,55))])])
        save(out/'RESULT.json',status);save(out/'INPUT_BINDINGS_POST.json',list(refs.values()))
        stats=''.join('<tr><td>'+html.escape(k)+'</td><td>'+str(v)+'</td></tr>' for k,v in measures.items())
        page='<!doctype html><meta charset="utf-8"><title>主体约束下的局部深度搜索</title><style>body{max-width:1300px;margin:35px auto;font:18px system-ui;line-height:1.7;background:#132130;color:#ecf3fc}a{color:#8bd3ff}img{max-width:100%}td{border:1px solid #59718a;padding:7px}table{border-collapse:collapse}</style>'
        if method=='regional-plane-v2':
            page+='<h1>主体约束下的局部平面估计</h1><p>用多个尺寸的支持区域，在旧主体深度中拟合受约束的局部面。绿色增补属于平面推定，不是新纹理测量。最后一个来源只作本轮事后检查；历史模型可能已经使用过它。没有声称复现完整ACMM或TAPA-MVS。</p>'
        else:
            page+=f'<h1>主体保留＋局部深度搜索</h1><p>固定全局相机、{len(used)}个既有邻视和同一侧支ROI，执行新CPU多视匹配。主体缺失处允许在原深度范围内搜索，禁止用插值或填洞冒充新观测。没有调用OpenMVS新功能或声称复现Cascade网络。</p>'
        page+='<p>状态：'+status['status']+'。原主体不删不移；本轮只是单参考深度增补，不是完整物理部件或CAD。</p><table>'+stats+'</table>'
        for title,file,text in [('1 输入与原主体','01_input.png','黄线为此前冻结的评估ROI，不是解剖真值。'),('2 局部候选与验证','02_search.png',('局部面由旧深度支持；每个候选仍需在至少两个原始邻视深度中得到一致支持。' if method=='regional-plane-v2' else '需要同一来源视图同时支持光度与几何；不足或冲突拒绝。')),('3 增补对照','03_combination.png','蓝色为原主体深度，绿色为新增。有效深度增加不等于几何正确。')]:
            page+='<h2>'+title+'</h2><p>'+text+'</p><img src="'+file+'">'
        page+='<p><a href="viewer.html">主体点云／主体＋局部点云对照</a> · <a href="whole_plus_local.ply">Blender可导入PLY</a> · <a href="local_additions.ply">仅新增局部点</a> · <a href="RESULT.json">逐项结果</a></p>'
        if method=='regional-plane-v2':page+='<h2>4 主体锚点与事后检查</h2><img src="04_plane_evidence.png"><p>右图只显示增补点在事后检查来源的支持（绿）及冲突（红）；未知不算通过。<a href="PLANE_FITS.json">拟合窗口记录</a></p>'
        page+='<p>仍未解决：无支持或遮挡部分、原主体错误、非刚性变形、完整多参考局部表面和网格融合。没有按结果调松规则。</p>'
        (out/'index.html').write_text(page,encoding='utf8');check();return status
    except Exception as exc:
        status.update(status='FAILED_NOT_ADOPTED',error=type(exc).__name__+': '+str(exc),wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',status);raise
