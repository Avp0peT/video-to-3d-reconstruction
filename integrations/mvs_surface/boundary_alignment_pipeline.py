"""Frozen-input mask/image alignment review; no corrections are applied."""
from pathlib import Path
import hashlib
import html
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .global_local_pipeline import binding,load,save,_binary
from .local_depth_search import _view,_rays,_project
from .mask_consumption_probe import authenticate_depth
from integrations.openmvs_depth.dmap import read_dmap
from .alignment_controls import select_reference_controls,audit_control_alignment
from .boundary_image_evidence import inspect_mask_boundary


def _array_binding(a):
    return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def _npz(path,result):
    arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False) as stored:
        if set(stored.files)!=set(arrays): raise ValueError('NPZ keys differ')
        for key,value in arrays.items():
            if not np.array_equal(stored[key],value,equal_nan=True): raise ValueError('NPZ readback differs')
    return {k:v for k,v in result.items() if not isinstance(v,np.ndarray)}


def _projection(ref,source,roi):
    y,x=np.nonzero(roi & (ref['depth']>0))
    uv,_,positive=_project(ref,source,_rays(ref,x,y),ref['depth'][y,x])
    return uv,positive


def _projection_stats(source,uv,positive):
    h,w=source['mask'].shape
    inside=positive & np.isfinite(uv).all(axis=1) & (uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
    ix=np.rint(np.clip(uv[:,0],0,w-1)).astype(int); iy=np.rint(np.clip(uv[:,1],0,h-1)).astype(int)
    known=inside & source['validity'][iy,ix]
    outside=known & ~source['mask'][iy,ix]
    distance=cv2.distanceTransform((~source['mask']).astype(np.uint8),cv2.DIST_L2,5)
    distances=distance[iy[outside],ix[outside]]
    return dict(projected=len(uv),in_frame=int(inside.sum()),outside_mask=int(outside.sum()),unknown=int((~known).sum()),
                outside_distance_median_px=float(np.median(distances)) if len(distances) else None,
                outside_distance_max_px=float(distances.max()) if len(distances) else None),inside,outside


def _figure(path,source,uv,positive,hint,edge,track):
    rgb=source['image']; h,w=rgb.shape[:2]
    canvas=Image.new('RGB',(w*4,h+50),(20,30,42)); draw=ImageDraw.Draw(canvas)
    titles=['OLD ROI: blue / background red','HYPOTHETICAL SHIFT: ORANGE / NOT APPLIED','MASK green / distinct IMAGE EDGE yellow','BLUE -> CYAN: accepted / GRAY: rejected']
    panels=[Image.fromarray(rgb.copy()) for _ in range(4)]
    original,inside,outside=_projection_stats(source,uv,positive)
    d=ImageDraw.Draw(panels[0])
    for j in np.flatnonzero(inside):
        x,y=uv[j]; d.point((x,y),fill=(240,55,55) if outside[j] else (40,140,255))
    proposal=track['proposal_translation_xy']
    if proposal is not None:
        shifted=uv+np.asarray(proposal); _,show,_=_projection_stats(source,shifted,positive)
        d=ImageDraw.Draw(panels[1])
        for j in np.flatnonzero(show): d.point(tuple(shifted[j]),fill=(255,155,30))
    else: ImageDraw.Draw(panels[1]).text((4,4),'NOT EVALUABLE - NO HYPOTHESIS',fill='white')
    arr=rgb.copy(); boundary=source['mask'] & ~cv2.erode(source['mask'].astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    arr[boundary & hint]=(60,235,100); panels[2]=Image.fromarray(arr); d=ImageDraw.Draw(panels[2])
    for point in edge['peak_xy'][edge['accepted_distinct'] & edge['region_hint_selected']]: d.point(tuple(point),fill=(255,235,30))
    d=ImageDraw.Draw(panels[3])
    for i,(pred,lk) in enumerate(zip(track['predicted_xy'],track['tracked_xy'])):
        if not np.isfinite(pred).all(): continue
        px,py=pred
        if not (0<=px<w and 0<=py<h): continue
        if track['accepted'][i] and np.isfinite(lk).all():
            d.ellipse((px-1,py-1,px+1,py+1),fill=(40,140,255)); d.line((px,py,float(lk[0]),float(lk[1])),fill=(60,230,240),width=1)
            lx,ly=map(float,lk); d.ellipse((lx-1,ly-1,lx+1,ly+1),fill=(60,230,240))
            delta=np.array([lx-px,ly-py]); length=float(np.linalg.norm(delta))
            if length>1:
                direction=delta/length; normal=np.array([-direction[1],direction[0]])
                for sign in (-1,1):
                    tip=np.array([lx,ly])-direction*3+normal*(sign*2)
                    d.line((lx,ly,float(tip[0]),float(tip[1])),fill=(60,230,240),width=1)
        else: d.point((px,py),fill=(150,150,150))
    for i,panel in enumerate(panels): canvas.paste(panel,(i*w,50)); draw.text((i*w+4,4),titles[i],fill='white')
    canvas.save(path)


def run(multiview_run,output,seconds=180):
    start=time.monotonic(); exp=Path(multiview_run).resolve(strict=True); out=Path(output).resolve()
    if type(seconds) is not int or not 1<=seconds<=900: raise ValueError('invalid CPU budget')
    if out.exists() or out.is_relative_to(exp) or exp.is_relative_to(out): raise ValueError('fresh disjoint output required')
    out.mkdir(parents=True); refs={}; status=dict(status='RUNNING',geometry_accepted=False,masks_changed=False,geometry_changed=False,new_meshes=0)
    save(out/'RESULT.json',status)
    def check():
        if time.monotonic()-start>seconds: raise TimeoutError('alignment CPU time limit')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>512<<20: raise RuntimeError('512MiB output limit')
    def bound(path,expected=None):
        check(); now=binding(path)
        if expected is not None and now!=expected: raise ValueError('exact input binding drift: '+str(path))
        if now['path'] in refs and refs[now['path']]!=now: raise ValueError('input changed between reads')
        if out.is_relative_to(Path(now['path']).parent): raise ValueError('output overlaps input directory')
        refs[now['path']]=now; return now
    def read(path):
        before=bound(path); result=load(path)
        if binding(path)!=before: raise ValueError('JSON drift')
        return result
    try:
        prior=read(exp/'PLAN.json'); previous=read(exp/'RESULT.json'); npz_binding=bound(exp/'SILHOUETTE_INTERVALS.npz')
        if previous.get('status')!='MULTIVIEW_SILHOUETTE_ESTIMATES_COMPLETE_NOT_GEOMETRY_ACCEPTED': raise ValueError('completed multiview run required')
        for b in prior['input_bindings']: bound(b['path'],b)
        historic=prior['source_plan_historical_binding']; bound(historic['path'],historic); plan=read(historic['path'])
        cameras={f['image']['global_id']:f for f in plan['frames']}
        if len(cameras)!=len(plan['frames']): raise ValueError('duplicate historical camera')
        refid=prior['reference_id']; ids=prior['source_ids']
        if type(refid) is not int or not isinstance(ids,list) or any(type(i) is not int for i in ids) or len(set([refid,*ids]))!=len(ids)+1 or any(i not in cameras for i in [refid,*ids]): raise ValueError('invalid frozen view identities')
        maps={}
        for b in prior['input_bindings']:
            if Path(b['path']).suffix.lower()!='.dmap': continue
            check(); d=read_dmap(b['path']); iid=d.view_ids[0]
            if iid not in cameras or iid in maps: raise ValueError('unknown/duplicate native depth')
            authenticate_depth(d,cameras[iid]['image']); maps[iid]=d
        loaded={}
        for iid in [refid,*ids]:
            check(); f=cameras[iid]; cam=f['image']; shape=(cam['height'],cam['width'])
            image_ref=bound(f['image_file']['path'],f['image_file'])
            with Image.open(image_ref['path']) as im: rgb=np.array(im.convert('RGB'))
            if rgb.shape[:2]!=shape: raise ValueError('native image dimensions differ')
            mask=_binary(f['mask'],shape,refs); valid=_binary(f['validity'],shape,refs)
            d=maps.get(iid)
            if d is not None and d.depth.shape!=shape: raise ValueError('native depth dimensions differ')
            loaded[iid]=_view(dict(view_id=iid,image=rgb,mask=mask,validity=valid,K=d.K if d else cam['K'],R=d.R if d else cam['R'],C=d.C if d else cam['C'],depth=d.depth if d else np.zeros(shape,np.float32),depth_available=d is not None))
        ref=loaded[refid]
        if not ref['depth_available']: raise ValueError('reference native depth required')
        with np.load(exp/'SILHOUETTE_INTERVALS.npz',allow_pickle=False) as stored: roi=stored['roi'].copy()
        if binding(exp/'SILHOUETTE_INTERVALS.npz')!=npz_binding: raise ValueError('ROI archive drift')
        if roi.dtype!=bool or roi.shape!=ref['mask'].shape or (roi & ~(ref['mask'] & ref['validity'])).any(): raise ValueError('invalid saved ROI native domain')
        root=Path(__file__).resolve().parents[2]
        for rel in ('scripts/run_region_mask_review.py','integrations/mvs_surface/boundary_alignment_pipeline.py','integrations/mvs_surface/alignment_controls.py','integrations/mvs_surface/boundary_image_evidence.py','integrations/mvs_surface/global_local_pipeline.py','integrations/mvs_surface/local_depth_search.py','integrations/mvs_surface/mask_consumption_probe.py','integrations/openmvs_depth/dmap.py'): bound(root/rel)
        cv2.setNumThreads(2)
        intent=dict(schema='boundary-alignment-review-v1',reference_id=refid,source_ids=ids,source_order='prior PLAN order; all assessment only',
                    source_plan_historical_binding=historic,prior_plan_binding=refs[str((exp/'PLAN.json').resolve())],roi_array_binding=_array_binding(roi),
                    threads=2,seconds=seconds,output_limit_bytes=512<<20,source_depth_available={str(i):loaded[i]['depth_available'] for i in [refid,*ids]},
                    parameters=dict(region_hint_radius_px=20,reference_controls=dict(body_tiles=8,body_max_per_tile=6,roi_tiles=4,roi_max_per_tile=4,erosion_pixels=8,quality=.01,min_distance=5,block_size=5,tile_margin_pixels=4,body_roi_exclusion_radius=6),
                                    tracking=dict(lk_window=31,lk_max_level=4,lk_iterations=30,lk_epsilon=.01,lk_flags=0,geometry_initialized=False,max_fb_error=1.,patch_size=7,min_patch_std=3.,min_ncc=.8,max_translation=20.,min_train=12,min_test=8,min_tiles_each=3),
                                    edge_profiles=dict(max_boundary_samples=2048,offset_min=-12,offset_max=12,derivative_step=1,gray_gaussian_sigma=1.,minimum_strength=6.,distinct_ratio=1.25,alternative_exclusion_pixels=2)),
                    input_bindings=list(refs.values()),geometry_accepted=False,masks_changed=False,geometry_changed=False)
        save(out/'PLAN.json',intent)
        controls=select_reference_controls(ref,roi,check=check)
        controls_meta=_npz(out/'CONTROLS.npz',controls)
        rows={}; images=[]
        for iid in [refid,*ids]:
            check(); view=loaded[iid]
            if iid==refid: hint=roi.copy(); uv=positive=None; hint_origin='frozen reference ROI'
            else:
                uv,positive=_projection(ref,view,roi); _,inside,_=_projection_stats(view,uv,positive)
                raster=np.zeros(view['mask'].shape,np.uint8)
                if inside.any():
                    coords=np.rint(uv[inside]).astype(int); raster[coords[:,1],coords[:,0]]=1
                    hint=cv2.distanceTransform(1-raster,cv2.DIST_L2,5)<=20
                else: hint=np.zeros(view['mask'].shape,bool)
                hint_origin='radius20 around in-frame positive old-depth ROI projections; empty projections give empty cohort'
            edge=inspect_mask_boundary(view['image'],view['mask'],view['validity'],hint,check=check)
            row=dict(edge=_npz(out/('edge_'+str(iid)+'.npz'),dict(edge,region_hint=hint)),image_shape=list(view['image'].shape),depth_available=view['depth_available'],region_hint_pixels=int(hint.sum()),region_hint_origin=hint_origin)
            if iid!=refid:
                track=audit_control_alignment(ref,view,controls,check=check); row['tracking']=_npz(out/('track_'+str(iid)+'.npz'),track)
                row['original_roi_projection']=_projection_stats(view,uv,positive)[0]
                proposal=track['proposal_translation_xy']
                row['hypothetical_roi_projection']=_projection_stats(view,uv+np.asarray(proposal),positive)[0] if proposal is not None else None
                file='view_'+str(iid)+'.png'; _figure(out/file,view,uv,positive,hint,edge,track); images.append(file); row['figure']=file
            else:
                from .global_local_pipeline import _overlay
                file='reference.png'; _overlay(out/file,view['image'],roi,[('REFERENCE ROI / MASK',[(view['mask'],(60,190,90))])]); images.append(file); row['figure']=file
            rows[str(iid)]=row
        for b in refs.values():
            check()
            if binding(b['path'])!=b: raise ValueError('frozen input drift')
        status.update(status='BOUNDARY_ALIGNMENT_REVIEW_COMPLETE_NOT_ADOPTED',views=rows,reference_id=refid,source_ids=ids,
                      controls=dict(metadata=controls_meta,total=len(controls['control_xy']),body=int((controls['group']==0).sum()),roi=int((controls['group']==1).sum())),
                      exact_array_readback=True,input_drift=[],source_depth_available=intent['source_depth_available'],
                      cause_verdict='NOT_ESTABLISHED',semantics='Image-edge agreement and image-only tracking do not prove mask truth or a 3D cause. Hypothetical translations are diagnostic only and never applied.',wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',status); save(out/'INPUT_BINDINGS_POST.json',list(refs.values()))
        page='<!doctype html><meta charset="utf-8"><title>边缘与投影错位核查</title><style>body{font:17px system-ui;max-width:1600px;margin:auto}img{max-width:100%}td,th{border:1px solid #aaa;padding:6px}table{border-collapse:collapse}</style><h1>边缘与投影错位核查</h1><p>原掩膜、相机与几何均未修改；橙色假设平移仅作诊断，未应用。图像边缘一致或追踪残差不能证明掩膜真值，也不能确定三维原因。灰色控制点为尝试后未通过，青色箭头仅显示通过控制点。</p><p><a href="PLAN.json">冻结计划</a> · <a href="RESULT.json">完整统计</a> · <a href="CONTROLS.npz">参考控制点</a></p><table><tr><th>视图</th><th>原生深度</th><th>边缘统计（全图与区域）</th><th>控制点与门槛</th><th>原投影 / 假设投影</th></tr>'
        for iid,row in rows.items(): page+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [iid,row['depth_available'],row['edge']['stats'],row.get('tracking',{}).get('per_group','参考视图'),[row.get('original_roi_projection'),row.get('hypothetical_roi_projection')]])+'</tr>'
        page+='</table>'
        for file in images: page+='<h2>'+html.escape(file)+'</h2><img src="'+html.escape(file)+'">'
        (out/'index.html').write_text(page,encoding='utf-8'); check(); return status
    except Exception as exc:
        status.update(status='FAILED_NOT_ADOPTED',error=type(exc).__name__+': '+str(exc),wall_seconds=time.monotonic()-start); save(out/'RESULT.json',status); raise
