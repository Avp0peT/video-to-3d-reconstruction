"""Inspectable sampled silhouette estimates, without adopting or combining depth."""
from pathlib import Path
import html
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .global_local_pipeline import (binding, save, load, historical_plan_binding,
    _binary, _roi, _write_cloud, _overlay)
from .mask_consumption_probe import authenticate_depth
from integrations.openmvs_depth.dmap import read_dmap
from integrations.viewer.ply_data import read_ply
from integrations.viewer.export import export_html
from .silhouette_intervals import _classify
from .silhouette_multiview import tighten_silhouette_intervals
from .local_depth_search import _rays, _project, _view

LABEL = 'INFERRED_OUTER_SHAPE_NOT_OBSERVED_GEOMETRY'


def run(source_experiment, reference_id, roi_spec, output, *, seconds=180):
    start = time.monotonic()
    exp = Path(source_experiment).resolve(strict=True)
    spec_path = Path(roi_spec).resolve(strict=True)
    out = Path(output).resolve()
    if type(reference_id) is not int or reference_id < 0 or type(seconds) is not int or not 1 <= seconds <= 900:
        raise ValueError('invalid reference or CPU budget')
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in (exp,spec_path.parent)):
        raise ValueError('fresh disjoint output required')
    refs = {}
    def bound(path, expected=None):
        b = binding(path)
        if expected is not None and any(b[k] != expected[k] for k in ('bytes','sha256')):
            raise ValueError('input binding drift: '+str(path))
        if b['path'] in refs and refs[b['path']] != b:
            raise ValueError('input changed between reads')
        refs[b['path']] = b
        return b
    def read(path):
        b = bound(path)
        value = load(path)
        if binding(path) != b:
            raise ValueError('JSON drift')
        return value
    def check():
        if time.monotonic()-start > seconds:
            raise TimeoutError('silhouette CPU time limit')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 512 << 20:
            raise RuntimeError('512MiB output limit')
    previous = read(exp/'RESULT.json')
    prepared = read(exp/'PREPARED.json')
    if previous['status'] != 'REGIONAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED' or not previous['arms']['WHOLE']['execution_ok']:
        raise ValueError('completed WHOLE experiment required')
    historic = historical_plan_binding(prepared)
    bound(historic['path'], historic)
    plan, spec = read(historic['path']), read(spec_path)
    cameras = {f['image']['global_id']: f for f in plan['frames']}
    selected = {v['global_id']: v for v in prepared['views']}
    if len(cameras) != len(plan['frames']) or len(selected) != len(prepared['views']) or set(selected) != set(cameras) or reference_id not in cameras:
        raise ValueError('camera/selection identities differ')
    maps = {}
    for b in previous['arms']['WHOLE']['depth_bindings']:
        check()
        bound(b['path'],b)
        d = read_dmap(b['path'])
        iid = d.view_ids[0]
        if iid in maps or iid not in cameras:
            raise ValueError('unknown/duplicate depth camera')
        authenticate_depth(d,cameras[iid]['image'])
        maps[iid] = d
    if reference_id not in maps:
        raise ValueError('reference depth missing')
    refd = maps[reference_id]
    baseline_ids = list(refd.view_ids[1:4])
    if len(baseline_ids) != 3 or len(set(baseline_ids)) != 3 or any(i not in cameras or i == reference_id for i in baseline_ids):
        raise ValueError('three distinct frozen native neighbor IDs required')
    remaining = sorted(set(cameras)-{reference_id}-set(baseline_ids))
    if not remaining:
        raise ValueError('additional calibrated view required for final diagnostic')
    used = baseline_ids + remaining
    loaded = {}
    for iid in [reference_id,*used]:
        check()
        f = cameras[iid]
        camera = f['image']
        b = bound(f['image_file']['path'],f['image_file'])
        with Image.open(b['path']) as im:
            rgb = np.array(im.convert('RGB'))
        shape = (camera['height'],camera['width'])
        if rgb.shape[:2] != shape:
            raise ValueError('native calibrated image domain required')
        mask = _binary(f['mask'],shape,refs)
        valid = _binary(f['validity'],shape,refs)
        available = iid in maps
        if available:
            d = maps[iid]
            if d.depth.shape != shape:
                raise ValueError('native image/depth domain required')
            k,r,c,depth = d.K,d.R,d.C,d.depth
        else:
            k,r,c = camera['K'],camera['R'],camera['C']
            depth = np.zeros(shape,np.float32)
        loaded[iid] = _view(dict(view_id=iid,K=k,R=r,C=c,image=rgb,mask=mask,validity=valid,depth=depth,depth_available=available))
    ref = loaded[reference_id]
    roi = _roi(spec,selected[reference_id]['decode_index'],ref['mask'] & ref['validity'])
    whole = exp/'WHOLE/work/__mvs_dense.ply'
    whole_binding = bound(whole)
    base = read_ply(whole,max_bytes=16<<20,max_vertices=100000,max_faces=1)
    if base['faces'] or base['colors'] is None:
        raise ValueError('colored WHOLE cloud required')
    root = Path(__file__).resolve().parents[2]
    for rel in ('integrations/mvs_surface/silhouette_multiview_pipeline.py','integrations/mvs_surface/silhouette_multiview.py','integrations/mvs_surface/silhouette_intervals.py','integrations/mvs_surface/local_depth_search.py','integrations/mvs_surface/global_local_pipeline.py','integrations/mvs_surface/mask_consumption_probe.py','integrations/openmvs_depth/dmap.py','integrations/viewer/export.py','integrations/viewer/ply_data.py'):
        bound(root/rel)
    out.mkdir(parents=True)
    cv2.setNumThreads(2)
    intent = dict(schema='multiview-sampled-silhouette-experiment-v1',label=LABEL,reference_id=reference_id,source_ids=used,
        baseline_admission_source_ids=baseline_ids,admission_source_ids=used[:-1],withheld_source_id=used[-1],
        source_order_policy='Frozen native first3 neighbors, sorted remaining global IDs, final largest remaining ID diagnostic only',
        source_depth_available={str(i):loaded[i]['depth_available'] for i in [reference_id,*used]},source_plan_historical_binding=historic,
        depth_bounds=[refd.d_min,refd.d_max],roi_pixels=int(roi.sum()),threads=2,seconds=seconds,
        output_limit_bytes=512<<20,geometry_accepted=False,complete=False,input_bindings=list(refs.values()),
        policy='No depth adoption or combination; endpoint midpoint clouds only',
        roi_origin='Previously fixed evaluation cells; not anatomical truth',physical_scale='unknown')
    save(out/'PLAN.json',intent)
    status = dict(status='RUNNING',geometry_accepted=False,complete=False,label=LABEL,existing_whole_unchanged=True)
    save(out/'RESULT.json',status)
    try:
        Image.fromarray(roi.astype(np.uint8)*255).save(out/'roi.png')
        constraint_images = []
        for iid in [reference_id,*used]:
            v = loaded[iid]
            file = ('reference' if iid == reference_id else 'source_'+str(iid))+'.png'
            role = 'REFERENCE' if iid == reference_id else ('HELDOUT DIAGNOSTIC' if iid == used[-1] else 'ADMISSION')
            _overlay(out/file,v['image'],np.zeros(v['mask'].shape,bool),[(str(iid)+' '+role+' / FULL IMAGE',[]),(str(iid)+' FOREGROUND + VALIDITY',[(v['mask'] & v['validity'],(60,190,90)),(~v['validity'],(240,70,70))])])
            constraint_images.append(file)
        result = tighten_silhouette_intervals(ref,[loaded[i] for i in used],roi,[refd.d_min,refd.d_max],check=check)
        arrays = {k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(out/'SILHOUETTE_INTERVALS.npz',**arrays,roi=roi)
        with np.load(out/'SILHOUETTE_INTERVALS.npz',allow_pickle=False) as stored:
            for key,arr in dict(arrays,roi=roi).items():
                if not np.array_equal(stored[key],arr):
                    raise ValueError('saved silhouette array readback differs')
        bounded = result['bounded_mask']
        if bounded.dtype != bool or bounded.shape != roi.shape or (bounded & ~roi).any():
            raise ValueError('invalid bounded reference domain')
        reason = result['reason']
        _overlay(out/'overview.png',ref['image'],roi,[('OLD3 BOUNDED',[(result['baseline_reason']==5,(255,150,30))]),('ALL STAGE BOUNDED',[(bounded,(255,150,30))]),('WHOLE DEPTH',[(ref['depth']>0,(40,150,230))]),('UNBOUNDED / NO SURVIVOR',[(reason==3,(170,90,240)),(reason==1,(240,70,70))])])
        shutil.copyfile(whole,out/'whole_cloud.ply')
        if binding(out/'whole_cloud.ply')['sha256'] != whole_binding['sha256']:
            raise ValueError('WHOLE byte copy differs')
        scenes = [dict(ply='whole_cloud.ply',sha256=whole_binding['sha256'],title='WHOLE baseline',data_kind='real_photos',declared_unit=None,note='Existing cloud; geometry not accepted.')]
        cloud_artifacts = {}
        projection_images = []
        projection_counts = {}
        if bounded.any():
            y,x = np.nonzero(bounded)
            rays = np.column_stack((x,y,np.ones(len(x))))@np.linalg.inv(ref['K']).T
            points,colors = [],[]
            for side,color in [('front',(255,150,30)),('back',(175,90,240))]:
                z = result[side+'_depth'][y,x]
                if not np.isfinite(z).all() or (z<=0).any():
                    raise ValueError('invalid midpoint depths')
                xyz = (rays*z[:,None])@ref['R']+ref['C']
                rgb = np.tile(np.array(color,np.uint8),(len(x),1))
                cloud_artifacts[side] = _write_cloud(out/(side+'_midpoints.ply'),xyz,rgb)
                points.append(xyz);colors.append(rgb)
            for iid in used:
                check()
                source = loaded[iid]
                canvas = source['image'].copy()
                edge = cv2.morphologyEx(source['mask'].astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)) != 0
                canvas[edge] = (70,230,110)
                panel = Image.fromarray(canvas)
                draw = ImageDraw.Draw(panel)
                counts = {}
                for side,color in [('front',(255,150,30)),('back',(175,90,240))]:
                    z = result[side+'_depth'][y,x]
                    fg,bg,unknown = _classify(ref,source,rays,z)
                    uv,_,positive = _project(ref,source,rays,z)
                    h,w = source['mask'].shape
                    drawable = positive & (uv[:,0]>=0) & (uv[:,0]<=w-1) & (uv[:,1]>=0) & (uv[:,1]<=h-1)
                    counts[side] = dict(foreground=int(fg.sum()),background=int(bg.sum()),unknown=int(unknown.sum()),evaluated=len(z),drawn=int(drawable.sum()),not_drawn_outside_image_or_behind=int((~drawable).sum()))
                    for j in np.flatnonzero(drawable):
                        px,py = map(float,uv[j])
                        ink = (240,60,60) if bg[j] else ((150,150,150) if unknown[j] else color)
                        # Different marker shapes retain endpoint identity on overlap.
                        if side == 'front':
                            draw.ellipse((px-2,py-2,px+2,py+2),outline=ink,width=1)
                        else:
                            draw.line((px-3,py,px+3,py),fill=ink,width=1)
                            draw.line((px,py-3,px,py+3),fill=ink,width=1)
                title = Image.new('RGB',(panel.width,panel.height+54),(20,30,42))
                title.paste(panel,(0,54))
                caption = ImageDraw.Draw(title)
                role = 'HELDOUT DIAGNOSTIC' if iid == used[-1] else 'ADMISSION VIEW / ENDPOINT DIAGNOSTIC'
                caption.text((5,3),str(iid)+' '+role,fill='white')
                caption.text((5,19),'Front orange circles / back purple crosses; background red; unknown gray',fill='white')
                caption.text((5,35),'Midpoints may fall outside mask within transition bracket; no snapping',fill='white')
                file = 'projection_'+str(iid)+'.png'
                title.save(out/file)
                projection_images.append(file)
                projection_counts[str(iid)] = counts
            xyz,rgb = np.vstack(points),np.vstack(colors)
            cloud_artifacts['estimates'] = _write_cloud(out/'silhouette_midpoints.ply',xyz,rgb)
            cloud_artifacts['whole_plus_estimates'] = _write_cloud(out/'whole_plus_silhouette.ply',np.vstack((np.asarray(base['vertices']).reshape(-1,3),xyz)),np.vstack((np.asarray(base['colors'],np.uint8).reshape(-1,3),rgb)))
            for name,title in [('whole_plus_silhouette.ply','WHOLE + silhouette estimate layer'),('silhouette_midpoints.ply','Silhouette midpoint estimates only')]:
                scenes.append(dict(ply=name,sha256=binding(out/name)['sha256'],title=title,data_kind='real_photos',declared_unit=None,note=LABEL+'; orange front / purple back; no observed surface or hidden concavity claim.'))
        save(out/'VIEWER_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=scenes))
        viewer = export_html(out/'VIEWER_SPEC.json',out/'viewer.html',profile='surface400k')
        for b in refs.values():
            check()
            if binding(b['path']) != b:
                raise ValueError('frozen input drift')
        metadata = {k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        status.update(status='MULTIVIEW_SILHOUETTE_ESTIMATES_COMPLETE_NOT_GEOMETRY_ACCEPTED',metrics=result['counts'],
            kernel_metadata=metadata,source_depth_available={str(i):loaded[i]['depth_available'] for i in [reference_id,*used]},baseline_admission_source_ids=baseline_ids,source_ids=used,admission_source_ids=used[:-1],withheld_source_id=used[-1],
            midpoint_projection_counts=projection_counts,midpoint_projection_semantics='Diagnostic only, not admission. Midpoints may lie outside masks within transition brackets; no snapping or endpoint deletion.',
            clouds=cloud_artifacts,viewer=viewer,exact_array_readback=True,input_drift=[],
            new_depth_adoptions=0,new_meshes=0,assumptions=['rigid scene','calibrated cameras','correct masks'],
            limitations=['Sampled rays and inverse depths can miss intervals.','Outer silhouette constraints cannot recover hidden concavities.','Historical inputs may share heldout reconstruction evidence.'],wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',status)
        save(out/'INPUT_BINDINGS_POST.json',list(refs.values()))
        page = '<!doctype html><meta charset="utf-8"><title>多视角轮廓约束估计</title><style>body{max-width:1400px;margin:30px auto;font:18px system-ui;background:#142130;color:#eee}img{max-width:100%}a{color:#8bd3ff}td,th{padding:6px;border:1px solid #587080}table{border-collapse:collapse}</style>'
        page += '<h1>多视角轮廓外形范围估计</h1><p>'+LABEL+'</p><p>橙色前端、紫色后端为离散采样区间的过渡中点估计，并非测量表面。主体点云逐字节保留，没有深度采纳、网格或完整部件结论。前三个原邻视构成同轮基线，随后按冻结顺序收紧；最后一视图仅供本轮诊断。历史深度和相机可能共享该视图证据，因此留出并非独立物理真值。</p>'
        page += '<p>未知投影不会充当前景。没有存活采样或没有双端界定时，不输出伪造端点；空结果仅展示原主体。采样可能漏掉薄区间，轮廓不能恢复隐藏凹腔。端点中点可能在过渡括号内投影到背景，图上红色仅为诊断，不吸附或删除点。</p>'
        page += '<p><a href="viewer.html">交互对照</a> · <a href="RESULT.json">结果与逐来源诊断</a> · <a href="PLAN.json">冻结计划与绑定</a> · <a href="SILHOUETTE_INTERVALS.npz">完整数组</a></p><img src="overview.png">'
        page += '<h2>逐阶段收紧记录</h2><table><tr><th>阶段</th><th>准入来源</th><th>存活采样</th><th>有存活射线</th><th>双端有界</th><th>无界</th><th>无存活</th><th>不连续</th><th>括号未知</th></tr>'
        for index,stage in enumerate(result.get('stage_counts',[])):
            page += '<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in [index,stage.get('admission_source_ids',[]),*[stage.get(key,0) for key in ('surviving_samples','any_surviving_rays','bounded','unbounded','no_survivor','disjoint','unknown_bracket')]])+'</tr>'
        page += '</table><h2>旧深度与轮廓估计诊断</h2><pre>'+html.escape(str(result['counts']))+'</pre><h2>完整图像、掩膜和投影约束</h2>'
        for file in [*constraint_images,*projection_images]:
            page += '<p><a href="'+html.escape(file)+'">'+html.escape(file)+'</a></p><img src="'+html.escape(file)+'">'
        (out/'index.html').write_text(page,encoding='utf-8')
        check()
        return status
    except Exception as exc:
        status.update(status='FAILED_NOT_ADOPTED',error=type(exc).__name__+': '+str(exc),wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',status)
        raise
