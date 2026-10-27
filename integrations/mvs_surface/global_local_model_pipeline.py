"""Frozen calibrated views to separate depth surfaces and silhouette proposals."""
from pathlib import Path
import html, shutil, struct, time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .global_local_pipeline import binding,load,save,_binary,_write_cloud
from .boundary_alignment_pipeline import _npz
from .mask_consumption_probe import authenticate_depth
from .local_depth_search import _view
from .temporal_local_geometry import recover_local_geometry
from .evidence_surface import build_depth_surface
from .local_visual_hull import build_local_visual_hull
from integrations.openmvs_depth.dmap import read_dmap
from integrations.viewer.ply_data import read_ply
from integrations.viewer.export import export_html


def write_mesh(path,vertices,faces,colors):
    v=np.asarray(vertices,np.float64);f=np.asarray(faces);c=np.asarray(colors)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v) or c.dtype!=np.uint8 or c.shape!=v.shape:
        raise ValueError('finite nonempty indexed triangle mesh required')
    header=('ply\nformat binary_little_endian 1.0\ncomment experimental separate geometry layer; scale unknown\n'
        f'element vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n'
        f'element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n')
    with Path(path).open('xb') as stream:
        stream.write(header.encode('ascii'))
        for p,color in zip(v,c):stream.write(struct.pack('<3d3B',*p,*color))
        for face in f:stream.write(struct.pack('<B3i',3,*face))
    actual=read_ply(path,max_bytes=16<<20,max_vertices=200000,max_faces=400000)
    if not np.array_equal(np.asarray(actual['vertices']),v) or not np.array_equal(np.asarray(actual['faces']),f) or not np.array_equal(np.asarray(actual['colors']),c):raise ValueError('mesh readback mismatch')
    obj=Path(path).with_suffix('.obj')
    with obj.open('x',encoding='ascii',newline='\n') as stream:
        stream.write('# Experimental surface; unknown units. No smoothing or hole filling.\n')
        for p in v:stream.write('v '+' '.join(format(float(x),'.17g') for x in p)+'\n')
        for face in f:stream.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
    edges=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
    _,n=np.unique(edges,axis=0,return_counts=True)
    cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    return dict(**binding(path),obj=binding(obj),vertices=len(v),faces=len(f),exact_readback=True,
        topology=dict(boundary_edges=int((n==1).sum()),nonmanifold_edges=int((n>2).sum()),zero_area_faces=int((np.linalg.norm(cross,axis=1)==0).sum()),self_intersections='NOT_EVALUATED'))


def _scene(path,title,note):
    return dict(ply=path.name,sha256=binding(path)['sha256'],title=title,data_kind='real_photos',declared_unit=None,note=note)


def combined_layer_budget(artifacts):
    layers=[artifacts[k] for k in ('observed_surface','inferred_shell') if k in artifacts]
    vertices=sum(x['vertices'] for x in layers);faces=sum(x['faces'] for x in layers)
    return dict(vertices=vertices,faces=faces,combined_allowed=vertices<=200000 and faces<=400000,policy='No truncation: export independent layers when aggregate viewer budget is exceeded')


def hull_view_diagnostic(vertices,faces,view):
    """Withheld-view point samples only, not full-face or visibility proof."""
    output={}
    for label,points in [('vertices',vertices),('triangle_centers',vertices[faces].mean(axis=1) if len(faces) else np.empty((0,3)))]:
        camera=(points-view['C'])@view['R'].T;positive=camera[:,2]>1e-9;pixels=camera@view['K'].T
        xy=pixels[:,:2]/np.where(positive,camera[:,2],1)[:,None];h,w=view['mask'].shape
        inside=positive & np.isfinite(xy).all(axis=1) & (xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
        safe=np.where(inside[:,None],xy,0);ix=np.rint(safe[:,0]).astype(int);iy=np.rint(safe[:,1]).astype(int)
        known=inside & view['validity'][iy,ix];fg=known & view['mask'][iy,ix];bg=known & ~view['mask'][iy,ix]
        output[label]=dict(total=len(points),foreground=int(fg.sum()),background=int(bg.sum()),unknown=int((~known).sum()))
    return dict(view_id=view['view_id'],samples=output,semantics='Vertices and face centers only, not full-triangle or depth visibility. This view was excluded from hull classification, but historical cameras and WHOLE bounds may share it; not independent truth.',adopted=False)


def run(multiview_run,output,*,blender=False,seconds=900):
    start=time.monotonic(); exp=Path(multiview_run).resolve(strict=True);out=Path(output).resolve()
    if type(seconds) is not int or not 1<=seconds<=1200 or type(blender) is not bool:raise ValueError('invalid execution budget/options')
    if out.exists() or out.is_relative_to(exp) or exp.is_relative_to(out):raise ValueError('fresh disjoint output required')
    out.mkdir(parents=True); refs={};last_scan=[start];status=dict(status='RUNNING',geometry_accepted=False,complete=False,scale_status='unknown',historical_inputs_changed=False)
    save(out/'RESULT.json',status)
    def check():
        now=time.monotonic()
        if now-start>seconds:raise TimeoutError('model CPU deadline exceeded')
        if now-last_scan[0]>=1:
            last_scan[0]=now
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>700<<20:raise RuntimeError('model output threshold exceeded')
    def bound(path,expected=None):
        check(); b=binding(path)
        if expected is not None and b!=expected:raise ValueError('input binding drift: '+str(path))
        if b['path'] in refs and refs[b['path']]!=b:raise ValueError('input changed between reads')
        refs[b['path']]=b;return b
    def read(path):
        before=bound(path);value=load(path)
        if binding(path)!=before:raise ValueError('JSON drift')
        return value
    try:
        prior=read(exp/'PLAN.json');old=read(exp/'RESULT.json');archive=bound(exp/'SILHOUETTE_INTERVALS.npz')
        if old.get('status')!='MULTIVIEW_SILHOUETTE_ESTIMATES_COMPLETE_NOT_GEOMETRY_ACCEPTED':raise ValueError('completed frozen multiview run required')
        for b in prior['input_bindings']:bound(b['path'],b)
        historic=prior['source_plan_historical_binding'];bound(historic['path'],historic);plan=read(historic['path'])
        cameras={f['image']['global_id']:f for f in plan['frames']}
        prepared_refs=[b for b in prior['input_bindings'] if Path(b['path']).name=='PREPARED.json']
        if len(prepared_refs)!=1:raise ValueError('unique frozen temporal identity map required')
        prepared=read(prepared_refs[0]['path']);decodes={x['global_id']:x['decode_index'] for x in prepared['views']}
        refid=prior['reference_id'];ids=[refid,*prior['source_ids']]
        if len(cameras)!=len(plan['frames']) or len(set(ids))!=len(ids) or any(type(i) is not int or i not in cameras or i not in decodes for i in ids):raise ValueError('invalid frozen image identities')
        if any(type(decodes[i]) is not int for i in ids) or len({decodes[i] for i in ids})!=len(ids):raise ValueError('invalid temporal decode identities')
        maps={}
        for b in prior['input_bindings']:
            if Path(b['path']).suffix.lower()!='.dmap':continue
            check();d=read_dmap(b['path']);iid=d.view_ids[0]
            if iid not in cameras or iid in maps:raise ValueError('unexpected native depth identity')
            authenticate_depth(d,cameras[iid]['image']);maps[iid]=d
        views=[]
        for iid in ids:
            check();f=cameras[iid];cam=f['image'];shape=(cam['height'],cam['width']);bound(f['image_file']['path'],f['image_file'])
            with Image.open(f['image_file']['path']) as image:rgb=np.array(image.convert('RGB'))
            if rgb.shape[:2]!=shape:raise ValueError('native image domain changed')
            mask=_binary(f['mask'],shape,refs);valid=_binary(f['validity'],shape,refs);d=maps.get(iid)
            if d is not None and d.depth.shape!=shape:raise ValueError('native depth domain changed')
            views.append(_view(dict(view_id=iid,decode_index=decodes[iid],image=rgb,mask=mask,validity=valid,
                K=d.K if d else cam['K'],R=d.R if d else cam['R'],C=d.C if d else cam['C'],depth=d.depth if d else np.zeros(shape,np.float32),depth_available=d is not None)))
        ref=views[0]
        with np.load(exp/'SILHOUETTE_INTERVALS.npz',allow_pickle=False) as z:roi=z['roi'].copy()
        if binding(exp/'SILHOUETTE_INTERVALS.npz')!=archive:raise ValueError('ROI archive drift')
        if roi.dtype!=bool or roi.shape!=ref['mask'].shape or (roi&~(ref['mask']&ref['validity'])).any():raise ValueError('invalid frozen reference ROI')
        whole_refs=[b for b in prior['input_bindings'] if Path(b['path']).name=='__mvs_dense.ply']
        if len(whole_refs)!=1:raise ValueError('unique whole cloud binding required')
        whole=read_ply(whole_refs[0]['path'],max_bytes=16<<20,max_vertices=100000,max_faces=1)
        vertices=np.asarray(whole['vertices']);lo=vertices.min(axis=0);hi=vertices.max(axis=0);span=hi-lo
        if (span<=0).any():raise ValueError('3D whole bounds required')
        bounds=np.stack((lo-.1*span,hi+.1*span))
        heldout=prior['withheld_source_id'];hull_ids=[i for i in ids if i!=heldout]
        root=Path(__file__).resolve().parents[2]
        for rel in ('global_local_model_pipeline.py','temporal_local_geometry.py','evidence_surface.py','local_visual_hull.py','alignment_controls.py','boundary_alignment_pipeline.py','global_local_pipeline.py','local_depth_search.py','mask_consumption_probe.py'):
            bound(root/'integrations/mvs_surface'/rel)
        for rel in ('scripts/run_region_mask_review.py','integrations/openmvs_depth/dmap.py','integrations/viewer/ply_data.py','integrations/viewer/export.py','integrations/blender_export/export.py','integrations/blender_export/worker.py'):bound(root/rel)
        cv2.setNumThreads(2)
        intent=dict(schema='global-local-model-v1',reference_id=refid,view_ids=ids,temporal_order=[v['view_id'] for v in sorted(views,key=lambda v:v['decode_index'])],decode_indices=decodes,
            input_bindings=list(refs.values()),source_plan_historical_binding=historic,whole_bounds=bounds.tolist(),bounds_policy='All unchanged WHOLE point extrema plus 10 percent per-axis span; domain bound, not shape truth',
            roi_origin='Previously selected evaluation cells, not automatic semantic segmentation',source_depth_available={str(v['view_id']):v['depth_available'] for v in views},
            hull_admission_ids=hull_ids,hull_diagnostic_id=heldout,threads=2,seconds=seconds,blender_requested=blender,
            steps=['image-only neighboring tracks with reverse chain closure and fixed-camera triangulation','native depth-grid faces with full-mask-footprint gates','independent voxel silhouette outer boundary','separate PLY OBJ HTML layers','optional supervised Blender create and readback'],
            geometry_fusion=False,scale_status='unknown',fixed_kernel_parameters='Bound implementation and per-stage metadata; no runtime tuning')
        save(out/'PLAN.json',intent)
        (out/'tracks').mkdir();stages={};artifacts={};scenes=[]
        print('MODEL_STAGE temporal_tracks',flush=True)
        t=recover_local_geometry(views,refid,roi,check=check);stages['tracks']=_npz(out/'TRACKS.npz',t);save(out/'TRACKS.json',stages['tracks'])
        if t['accepted'].any():
            p=out/'local_triangulated_points.ply';artifacts['local_points']=_write_cloud(p,t['points'][t['accepted']],t['rgb'][t['accepted']])
        lookup={v['view_id']:v for v in views}
        for col,iid in enumerate(t['view_ids']):
            image=Image.fromarray(lookup[iid]['image']);draw=ImageDraw.Draw(image)
            for i in np.flatnonzero(t['observation_mask'][:,col]):
                x,y=t['observations'][i,col];color=(40,230,100) if t['accepted'][i] else (255,195,30)
                draw.ellipse((float(x-2),float(y-2),float(x+2),float(y+2)),fill=color)
            image.save(out/'tracks'/f'view_{iid}.png')
        save(out/'PROGRESS.json',dict(stage='tracks_complete',elapsed_seconds=time.monotonic()-start,counts=t['counters']))
        print('MODEL_STAGE observed_surface',flush=True)
        s=build_depth_surface(views,check=check);stages['surface']=_npz(out/'SURFACE_PROVENANCE.npz',s);save(out/'SURFACE.json',stages['surface'])
        if len(s['faces']):
            p=out/'observed_depth_surface.ply';artifacts['observed_surface']=write_mesh(p,s['worldvertices'],s['faces'],s['colors'])
            scenes.append(_scene(p,'深度支持的开放表面','Interpolated triangles from original native depth samples; full mask-footprint checks; open layered geometry, not accepted or watertight.'))
        save(out/'PROGRESS.json',dict(stage='surface_complete',elapsed_seconds=time.monotonic()-start,counts=s['counts']))
        print('MODEL_STAGE silhouette_shell',flush=True)
        h=build_local_visual_hull([v for v in views if v['view_id']!=heldout],refid,roi,bounds,check=check)
        stages['hull']=_npz(out/'INFERRED_HULL.npz',h);save(out/'INFERRED_HULL.json',stages['hull'])
        stages['hull_diagnostic']=hull_view_diagnostic(h['vertices'],h['faces'],lookup[heldout]);save(out/'HULL_DIAGNOSTIC.json',stages['hull_diagnostic'])
        if len(h['faces']):
            p=out/'inferred_local_shell.ply';artifacts['inferred_shell']=write_mesh(p,h['vertices'],h['faces'],h['colors'])
            scenes.append(_scene(p,'橙色：轮廓推定的局部外壳','INFERRED outer boundary; center-sampled voxels, not observed surface. Unknown and ROI/volume cuts remain open; mixed-corner flags retained. No hidden-concavity or thin-gap guarantee.'))
        if len(s['faces']) and len(h['faces']):
            p=out/'layered_preview.ply';cv=np.vstack((s['worldvertices'],h['vertices']));cf=np.vstack((s['faces'],h['faces']+len(s['worldvertices'])));cc=np.vstack((s['colors'],h['colors']))
            if len(cv)<=200000 and len(cf)<=400000:
                artifacts['layered_preview']=write_mesh(p,cv,cf,cc)
                save(out/'layered_preview.SOURCES.json',dict(observed_vertex_count=len(s['worldvertices']),observed_face_count=len(s['faces']),inferred_vertex_offset=len(s['worldvertices']),inferred_face_offset=len(s['faces']),welded=False,geometry_accepted=False))
                save(out/'PREVIEW_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=[_scene(p,'主体表面 + 橙色推定外壳','Two independent unmerged layers; orange is inferred geometry, not observations. No new hole-filled surface or CAD acceptance.')]))
                export_html(out/'PREVIEW_SPEC.json',out/'model_viewer.html',profile='surface400k')
        shutil.copyfile(whole_refs[0]['path'],out/'baseline_points.ply')
        if binding(out/'baseline_points.ply')['sha256']!=whole_refs[0]['sha256']:raise ValueError('baseline copy drift')
        save(out/'BASELINE_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=[_scene(out/'baseline_points.ply','原主体点云','Unchanged old dense points; no new reconstruction or acceptance.')]))
        export_html(out/'BASELINE_SPEC.json',out/'baseline_viewer.html',profile='surface400k')
        if 'local_points' in artifacts:
            save(out/'POINTS_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=[_scene(out/'local_triangulated_points.ply','近邻帧新增三角化候选点','Fixed-camera fit-excluded diagnostic; not independent calibrated ground truth or dense surface.')]))
            export_html(out/'POINTS_SPEC.json',out/'points_viewer.html',profile='surface400k')
        layer_budget=combined_layer_budget(artifacts);layer_specs=[]
        if scenes:
            for scene in scenes:
                stem=Path(scene['ply']).stem;spec=out/(stem+'_SPEC.json')
                save(spec,dict(schema='ply-viewer-scenes-v1',scenes=[scene]));export_html(spec,out/(stem+'_viewer.html'),profile='surface400k')
                layer_specs.append((stem,spec))
            if layer_budget['combined_allowed']:
                save(out/'MODEL_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=scenes))
                export_html(out/'MODEL_SPEC.json',out/'layers_viewer.html',profile='surface400k')
        for b in refs.values():
            if binding(b['path'])!=b:raise ValueError('input changed during modeling')
        status.update(status='MODEL_GENERATED_NOT_GEOMETRY_ACCEPTED' if scenes else 'NO_EVIDENCE_SUPPORTED_MODEL',stages=stages,artifacts=artifacts,source_bindings=list(refs.values()),input_drift=[],new_meshes=len(scenes),new_triangulated_points=int(t['accepted'].sum()),
            observed_surface_accepted=False,inferred_shell_accepted=False,geometry_fused=False,layer_export_budget=layer_budget,model_seconds=time.monotonic()-start,blender={'status':'NOT_RUN'},limitations=['Old depths, calibrated poses and masks can contain errors.','Triangles are depth interpolation, not independent new measurements.','Silhouette outer shell is an independent approximate proposal, not recovered concavity or verified topology.','Triangulated points have a separate PLY/viewer and are not mesh vertices or part of the Blender mesh scenes.','No CAD, absolute scale or completeness acceptance.'])
        save(out/'RESULT.json',status);save(out/'INPUT_BINDINGS_POST.json',list(refs.values()))
        if blender and scenes:
            print('MODEL_STAGE blender_export',flush=True)
            from integrations.blender_export.export import export_blend
            if layer_budget['combined_allowed']:
                status['blender']=export_blend(out/'MODEL_SPEC.json',out/'blender',profile='surface400k')
            else:
                exports={stem:export_blend(spec,out/('blender_'+stem),profile='surface400k') for stem,spec in layer_specs}
                status['blender']=dict(status='SUCCESS' if all(x['status']=='SUCCESS' for x in exports.values()) else 'PARTIAL',separate_files=True,exports=exports)
        for b in refs.values():
            if binding(b['path'])!=b:raise ValueError('input changed during export')
        status['wall_seconds']=time.monotonic()-start;save(out/'RESULT.json',status)
        links=['<a href="baseline_viewer.html">原主体点云</a>']
        for file,label in [('model_viewer.html','合并显示（橙色为推定）'),('layers_viewer.html','分别查看表面与外壳'),('observed_depth_surface_viewer.html','深度表面'),('inferred_local_shell_viewer.html','独立推定外壳'),('points_viewer.html','新增三角化候选点'),('blender/model.blend','Blender 模型'),('blender_observed_depth_surface/model.blend','深度表面 Blender'),('blender_inferred_local_shell/model.blend','推定外壳 Blender')]:
            if (out/file).exists():links.append(f'<a href="{file}">{label}</a>')
        page='<!doctype html><meta charset="utf-8"><title>整体约束与局部模型</title><style>body{max-width:1400px;margin:30px auto;background:#142130;color:#eee;font:18px system-ui}a{color:#8cd4ff}img{max-width:32%}pre{white-space:pre-wrap}</style><h1>整体约束与局部模型</h1><p>'+status['status']+'</p><p>深度支持面与橙色轮廓推定外壳分开保存，未焊接、未填洞、未声称完整模型或 CAD。未知尺度。原模型不改。</p><p>'+' · '.join(links)+'</p>'
        page+='<h2>阶段结果</h2><pre>'+html.escape(str(dict(tracks=t['counters'],surface=s['counts'],inferred_shell=h['counts'])))+'</pre><p><a href="PLAN.json">冻结计划</a> · <a href="RESULT.json">全部结果与限制</a> · <a href="TRACKS.json">轨迹门槛</a> · <a href="SURFACE.json">表面来源</a> · <a href="INFERRED_HULL.json">外壳推定限制</a></p><h2>近邻帧轨迹：绿=通过几何检查，黄=仅有影像观测</h2>'
        for iid in t['view_ids']:page+=f'<a href="tracks/view_{iid}.png"><img src="tracks/view_{iid}.png" alt="view {iid}"></a>'
        (out/'index.html').write_text(page,encoding='utf8')
        return status
    except Exception as exc:
        status.update(status='FAILED_MODEL_PIPELINE',error=type(exc).__name__+': '+str(exc),wall_seconds=time.monotonic()-start);save(out/'RESULT.json',status);raise
