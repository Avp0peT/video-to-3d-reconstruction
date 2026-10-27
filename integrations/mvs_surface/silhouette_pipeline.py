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
from .silhouette_intervals import estimate_silhouette_intervals, _classify
from .local_depth_search import _rays, _project

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
    neighbors = list(refd.view_ids[1:])[:5]
    used = [i for i in neighbors if i in maps]
    if len(used) < 4:
        raise ValueError('four available bound neighbor depths required')
    loaded = {}
    for iid in [reference_id,*used]:
        check()
        f,d = cameras[iid],maps[iid]
        b = bound(f['image_file']['path'],f['image_file'])
        with Image.open(b['path']) as im:
            rgb = np.array(im.convert('RGB'))
        if rgb.shape[:2] != d.depth.shape:
            raise ValueError('native image/depth domain required')
        mask = _binary(f['mask'],d.depth.shape,refs)
        valid = _binary(f['validity'],d.depth.shape,refs)
        loaded[iid] = dict(view_id=iid,K=d.K,R=d.R,C=d.C,image=rgb,mask=mask,validity=valid,depth=d.depth)
    ref = loaded[reference_id]
    roi = _roi(spec,selected[reference_id]['decode_index'],ref['mask'] & ref['validity'])
    whole = exp/'WHOLE/work/__mvs_dense.ply'
    whole_binding = bound(whole)
    base = read_ply(whole,max_bytes=16<<20,max_vertices=100000,max_faces=1)
    if base['faces'] or base['colors'] is None:
        raise ValueError('colored WHOLE cloud required')
    root = Path(__file__).resolve().parents[2]
    for rel in ('integrations/mvs_surface/silhouette_pipeline.py','integrations/mvs_surface/silhouette_intervals.py','integrations/mvs_surface/local_depth_search.py','integrations/mvs_surface/global_local_pipeline.py','integrations/mvs_surface/mask_consumption_probe.py','integrations/openmvs_depth/dmap.py','integrations/viewer/export.py','integrations/viewer/ply_data.py'):
        bound(root/rel)
    out.mkdir(parents=True)
    cv2.setNumThreads(2)
    intent = dict(schema='sampled-silhouette-experiment-v1',label=LABEL,reference_id=reference_id,source_ids=used,
        admission_source_ids=used[:-1],withheld_source_id=used[-1],source_plan_historical_binding=historic,
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
        result = estimate_silhouette_intervals(ref,[loaded[i] for i in used],roi,[refd.d_min,refd.d_max],check=check)
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
        _overlay(out/'overview.png',ref['image'],roi,[('ROI',[]),('WHOLE DEPTH',[(ref['depth']>0,(40,150,230))]),('BOUNDED MIDPOINTS',[(bounded,(255,150,30))]),('UNBOUNDED / NO SURVIVOR',[(reason==3,(170,90,240)),(reason==1,(240,70,70))])])
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
        status.update(status='SILHOUETTE_ESTIMATES_COMPLETE_NOT_GEOMETRY_ACCEPTED',metrics=result['counts'],
            kernel_metadata=metadata,source_ids=used,admission_source_ids=used[:-1],withheld_source_id=used[-1],
            midpoint_projection_counts=projection_counts,midpoint_projection_semantics='Diagnostic only, not admission. Midpoints may lie outside masks within transition brackets; no snapping or endpoint deletion.',
            clouds=cloud_artifacts,viewer=viewer,exact_array_readback=True,input_drift=[],
            new_depth_adoptions=0,new_meshes=0,assumptions=['rigid scene','calibrated cameras','correct masks'],
            limitations=['Sampled rays and inverse depths can miss intervals.','Outer silhouette constraints cannot recover hidden concavities.','Historical inputs may share heldout reconstruction evidence.'],wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',status)
        save(out/'INPUT_BINDINGS_POST.json',list(refs.values()))
        page = '<!doctype html><meta charset="utf-8"><title>Silhouette outer-shape estimates</title><style>body{max-width:1400px;margin:30px auto;font:18px system-ui;background:#142130;color:#eee}img{max-width:100%}a{color:#8bd3ff}</style>'
        page += '<h1>'+LABEL+'</h1><p>Orange front and purple back are sampled transition midpoint estimates. No measured surface, complete part, hidden concavity, mesh or geometry acceptance is claimed. WHOLE remains unchanged.</p>'
        page += '<p><a href="viewer.html">Interactive comparison</a> · <a href="RESULT.json">Result and diagnostics</a> · <a href="PLAN.json">Frozen plan</a> · <a href="SILHOUETTE_INTERVALS.npz">All interval arrays</a></p><img src="overview.png">'
        for file in [*constraint_images,*projection_images]:
            page += '<p><a href="'+html.escape(file)+'">'+html.escape(file)+'</a></p><img src="'+html.escape(file)+'">'
        (out/'index.html').write_text(page,encoding='utf-8')
        check()
        return status
    except Exception as exc:
        status.update(status='FAILED_NOT_ADOPTED',error=type(exc).__name__+': '+str(exc),wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',status)
        raise
