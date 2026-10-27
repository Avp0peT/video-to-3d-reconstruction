"""Read-only, region-wise measured support and raw/candidate silhouette audit.

An engineering warning gate, not a shape estimator or Kozlov construction.
Projected points, stored depths and surface interpolation remain separate.
"""
from pathlib import Path
import html
import time
import numpy as np
from PIL import Image, ImageDraw

from integrations.video_masks.boundary_geometry_binding import _bind, _read, _binary, _camera_check, _need
from integrations.video_masks.face_mask_adapter import verify_depth, _save
from integrations.viewer.ply_data import read_ply
from integrations.mvs_surface.scene import read_scene
from integrations.openmvs_depth.dmap import read_dmap
from .silhouette_review import project, raster, metrics, _edge

ROOT = Path(__file__).resolve().parents[2]
POLICY = dict(grid_columns=8, grid_rows=12, min_region_pixels=64,
              min_valid_fraction=.8, depth_relative_tolerance=.02,
              support_radius_pixels=8, low_support_fraction=.2,
              baseline_missing_fraction=.15, regression_fraction=.05,
              min_regression_pixels=16, max_projected_views=16)
LIMITATIONS = [
    'Masks, cameras and depth share input images; this is not independent accuracy.',
    'A projected point is not necessarily visible. Depth agreement is a separate conditional measurement.',
    'Unknown depths, invalid pixels and excluded camera views are not zero-quality observations or passes.',
    'Silhouette agreement does not prove thickness, correct correspondence, rigidity or hidden shape.',
    'Grid and warning thresholds are generic engineering policy, not physical density or accuracy guarantees.',
    'All artifacts remain available. No points are added, moved or removed; no holes are filled.',
]


def _bool(a, shape=None):
    a = np.asarray(a)
    if a.dtype != np.bool_ or a.ndim != 2 or (shape is not None and a.shape != shape):
        raise ValueError('same-domain boolean raster required')
    return a


def region_metrics(mask, valid, depth_valid=None, point_near=None, raw=None, final=None, policy=None):
    """Local denominators detect thin missing parts despite high global recall."""
    p = dict(POLICY if policy is None else policy)
    mask = _bool(mask); valid = _bool(valid, mask.shape)
    maps = {k: None if a is None else _bool(a, mask.shape) for k, a in
            [('depth', depth_valid), ('points', point_near), ('raw', raw), ('final', final)]}
    if not mask.any():
        return [dict(status='UNKNOWN', reasons=['EMPTY_SOURCE_MASK'], bbox=None)]
    yy, xx = np.where(mask)
    xe = np.linspace(xx.min(), xx.max()+1, p['grid_columns']+1).astype(int)
    ye = np.linspace(yy.min(), yy.max()+1, p['grid_rows']+1).astype(int)
    rows = []
    for gy in range(p['grid_rows']):
        for gx in range(p['grid_columns']):
            x0,x1,y0,y1 = int(xe[gx]),int(xe[gx+1]),int(ye[gy]),int(ye[gy+1])
            s = np.s_[y0:y1,x0:x1]; m = mask[s]; fg = m & valid[s]
            count = int(fg.sum()); total = int(m.sum())
            if total == 0:
                continue
            row = dict(grid=[gx,gy], bbox=[x0,y0,x1,y1], mask_pixels=total,
                       valid_mask_pixels=count, status='MEASURED', reasons=[])
            if count < p['min_region_pixels'] or count/total < p['min_valid_fraction']:
                row.update(status='UNKNOWN', reasons=['INSUFFICIENT_VALID_MASK_AREA'])
                rows.append(row); continue
            for key, arr in maps.items():
                row[key+'_coverage'] = None if arr is None else float((arr[s] & fg).sum()/count)
            row['regression_pixels'] = None if raw is None or final is None else int((maps['raw'][s] & ~maps['final'][s] & fg).sum())
            if depth_valid is None:
                row['reasons'].append('DEPTH_UNAVAILABLE')
            elif row['depth_coverage'] < p['low_support_fraction']:
                row['reasons'].append('LOW_STORED_DEPTH_SUPPORT')
            if point_near is None:
                row['reasons'].append('DEPTH_TESTED_POINTS_UNAVAILABLE')
            elif row['points_coverage'] < p['low_support_fraction']:
                row['reasons'].append('LOW_DEPTH_AGREEING_POINT_SUPPORT')
            if raw is None:
                row['reasons'].append('RAW_PROJECTION_UNAVAILABLE')
            elif row['raw_coverage'] < 1-p['baseline_missing_fraction']:
                row['reasons'].append('BASELINE_SILHOUETTE_GAP')
            if final is None:
                row['reasons'].append('CANDIDATE_PROJECTION_UNAVAILABLE')
            elif row['regression_pixels'] is not None and row['regression_pixels'] >= p['min_regression_pixels'] and row['regression_pixels']/count > p['regression_fraction']:
                row['reasons'].append('POSTPROCESS_FOREGROUND_REGRESSION')
            rows.append(row)
    return rows


def point_support(points, camera, depth, cv):
    """Nearest-pixel z buffer, then 2% native camera-Z comparison. Not ground truth."""
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError('finite 3D points required')
    h,w = camera['height'],camera['width']
    xyz = (pts-np.asarray(camera['C'])) @ np.asarray(camera['R']).T
    q = xyz @ np.asarray(camera['K']).T
    front = xyz[:,2] > 0
    uv = q[front,:2]/q[front,2:]
    inside = (uv[:,0]>=-.5)&(uv[:,0]<w-.5)&(uv[:,1]>=-.5)&(uv[:,1]<h-.5)
    uv = np.floor(uv[inside]+.5).astype(int); z = xyz[front,2][inside]
    buffer = np.full((h,w),np.inf)
    np.minimum.at(buffer,(uv[:,1],uv[:,0]),z)
    hit = np.isfinite(buffer)
    def near(a):
        if not a.any(): return np.zeros_like(a)
        return cv.distanceTransform((~a).astype(np.uint8),cv.DIST_L2,cv.DIST_MASK_PRECISE) <= POLICY['support_radius_pixels']
    agree = None
    if depth is not None:
        d = np.asarray(depth)
        if d.shape != (h,w): raise ValueError('native depth size mismatch')
        good = np.isfinite(d)&(d>0)
        agree = hit & good & (np.abs(buffer-d)<=POLICY['depth_relative_tolerance']*d)
    return dict(projected=hit, projected_near=near(hit), depth_agreeing=agree,
                depth_agreeing_near=None if agree is None else near(agree),
                point_count=len(pts), positive_camera_points=int(front.sum()),
                inside_camera_points=int(inside.sum()))


def verdict(views, has_candidate):
    reasons = set()
    if not views: reasons.add('NO_EVALUATED_VIEWS')
    for view in views:
        if view.get('status') != 'EVALUATED': reasons.add('UNKNOWN_VIEW')
        if view.get('mask_touches_frame') is True: reasons.add('SOURCE_FRAME_CLIPPING_RISK')
        if not view.get('regions'): reasons.add('UNKNOWN_REGIONS')
        for row in view.get('regions',[]):
            if row.get('status') != 'MEASURED': reasons.add('UNKNOWN_REGION')
            reasons.update(row.get('reasons',[]))
    if not has_candidate: reasons.add('NO_CANDIDATE')
    return dict(status='REVIEW_REQUIRED' if reasons else 'NO_FLAGGED_REGRESSION_NOT_QUALITY_ACCEPTANCE',
                candidate_promotable=bool(has_candidate and views and not reasons), reasons=sorted(reasons))


def _geometry(record, refs):
    path = _bind(record['path'],refs,record)
    mesh = read_ply(path,max_bytes=32<<20,max_vertices=500000,max_faces=600000)
    _need(len(mesh['vertices'])==record['vertices'], 'native vertex count drift')
    _need(len(mesh['faces'])==record.get('triangles',record.get('faces')), 'native face count drift')
    return mesh


def context(surface_run, candidate_run, refs):
    source = Path(surface_run).resolve(strict=True)
    nr = _read(source/'surface/result.json',refs)
    _need(nr.get('status')=='SURFACE_GENERATED_NOT_ACCEPTED' and not nr.get('input_drift'), 'surface contract refused')
    roles = {}
    for stage in nr['stages']:
        for rec in stage.get('outputs',{}).get('geometry',[]):
            key = (stage['name'],rec['role'])
            _need(key not in roles,'duplicate geometry role'); roles[key]=rec
    dense = _geometry(roles[('dense','dense')],refs)
    raw = _geometry(roles[('mesh','pre_cleanup_surface')],refs)
    final = None
    if candidate_run is not None:
        c = Path(candidate_run).resolve(strict=True); lr = _read(c/'RESULT.json',refs)
        _need(lr.get('schema')=='local-boundary-surface-v1' and lr.get('status')=='LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED'
              and lr.get('source_unchanged') is True and Path(lr['source_run']).resolve()==source, 'candidate contract refused')
        rec = lr['outputs']['retained']
        _need(Path(rec['path']).resolve()==c/'retained.ply','candidate output path mismatch')
        final = _geometry(rec,refs)
    masks = _read(source/'masks/RESULT.json',refs); export = _read(source/'export/EXPORT.json',refs)
    plan = _read(source/'surface/plan.json',refs)
    _need(plan.get('contract_version')=='mvs-surface-run-v1','native plan schema mismatch')
    _need(masks.get('schema')=='region-surface-masks-v1' and masks.get('coverage_status')=='COMPLETE'
          and masks.get('adapter_source_chain')=='VERIFIED' and not masks.get('input_drift') and not masks.get('output_drift'),'mask contract refused')
    _need(export.get('schema')=='region-surface-export-v1' and export.get('source_unchanged') is True,'export contract refused')
    fmap = _read(source/'export/FRAME_MAP.json',refs,export['frame_map_sha256'])
    scene = read_scene(_bind(source/'bridge/scene.mvs',refs,masks['scene_binding']))
    _need(np.array_equal(scene['transform'],np.eye(4)),'unsupported scene transform')
    images={Path(x['name']).name:x for x in scene['images']}
    fm={x['exported_name']:x for x in fmap['frames']}; mf={x['exported_name']:x for x in masks['frames']}
    pf={Path(x['image']['name']).name:x for x in plan['frames']}
    _need(0<len(images)<=256 and len(images)==len(scene['images'])==len(fm)==len(fmap['frames'])==len(mf)==len(masks['frames'])
          and len(pf)==len(plan['frames'])==len(images) and images.keys()==fm.keys()==mf.keys()==pf.keys(),'view identity sets differ')
    views={}; seen=set()
    for name,cam in images.items():
        f,e=mf[name],fm[name]; i=f['decode_index']
        _need(type(i) is int and i>=0 and i not in seen and i==e['decode_index'] and f['input_index']==e['input_index']
              and f['status']=='OK' and f['readback_verified'] is True,'frame identity mismatch')
        seen.add(i); _camera_check(f,cam,e)
        shape=(cam['height'],cam['width'])
        _need(max(shape)<=2048,'image audit size bound')
        mask=_binary(_bind(f['mask']['path'],refs,f['mask']),shape)!=0
        valid=_binary(_bind(f['validity']['path'],refs,f['validity']),shape)!=0
        pr=pf[name]; pc=pr['image']
        _need(pc['global_id']==cam['global_id'] and (pc['width'],pc['height'])==(cam['width'],cam['height'])
              and all(np.array_equal(pc[k],cam[k]) for k in ('K','R','C')), 'native plan camera differs')
        for k in ('mask','validity'):
            _need(Path(pr[k]['path']).resolve()==Path(f[k]['path']).resolve() and all(pr[k][q]==f[k][q] for q in ('sha256','bytes')),'native plan masks differ')
        ip=(source/'bridge'/cam['name']).resolve()
        _need(ip==Path(pr['image_file']['path']).resolve() and ip.is_relative_to(source/'bridge'),'native RGB path differs')
        rgb=_bind(ip,refs,pr['image_file'])
        border=bool(mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any())
        views[name]=dict(decode_index=i,camera=cam,mask=mask,valid=valid,image=str(rgb),depth_path=None,mask_touches_frame=border)
    for path in sorted((source/'surface/work').glob('depth*.dmap')):
        d=read_dmap(path); name=Path(d.image_name_utf8.replace('\\','/')).name
        _need(name in views and views[name]['depth_path'] is None,'unknown or duplicate depth identity')
        verify_depth(d,images[name]); _bind(path,refs,dict(bytes=d.file_bytes,sha256=d.sha256))
        views[name]['depth_path']=str(path)
        fg=views[name]['mask']&views[name]['valid']; positive=np.isfinite(d.depth)&(d.depth>0)
        views[name]['depth_coverage']=float((positive&fg).sum()/fg.sum()) if fg.any() else None
    return source,raw,final,dense,sorted(views.values(),key=lambda v:v['decode_index'])


def _select(views, center):
    # Include worst-support view per temporal stratum, then spread camera directions.
    selected=[]; remaining=list(range(len(views)))
    for group in np.array_split(remaining,min(4,len(views))):
        selected.append(min(map(int,group),key=lambda i:(views[i].get('depth_coverage',-1) if views[i].get('depth_coverage') is not None else -1,views[i]['decode_index'])))
    directions=[]
    for v in views:
        d=np.asarray(v['camera']['C'])-center; length=np.linalg.norm(d)
        directions.append(d/length if length>1e-12 else np.zeros(3))
    while len(selected)<min(POLICY['max_projected_views'],len(views)):
        remaining=[i for i in remaining if i not in selected]
        selected.append(max(remaining,key=lambda i:(-max(float(directions[i]@directions[j]) for j in selected),-views[i]['decode_index'])))
    return sorted(selected)


def _overlay(path, rgb, mask, valid, support, raw, final, regions):
    h,w=mask.shape; source=np.asarray(rgb).copy(); source[_edge(mask)]=[255,230,30]
    panels=[source]; titles=['SOURCE / YELLOW: INPUT MASK']
    for name,a in [('DEPTH-AGREEING POINTS (8px)',support['depth_agreeing_near']),('RAW MESH SILHOUETTE',raw),('FINAL CANDIDATE SILHOUETTE',final)]:
        ar=np.asarray(rgb).copy()
        if a is not None:
            hit=a&mask&valid; lost=~a&mask&valid
            ar[hit]=(.55*ar[hit]+.45*np.array([35,180,255])).astype(np.uint8)
            ar[lost]=(.4*ar[lost]+.6*np.array([255,130,30])).astype(np.uint8)
        ar[_edge(mask)]=[255,230,30]; panels.append(ar); titles.append(name if a is not None else name+' : UNKNOWN')
    out=Image.new('RGB',(w*4,h+50),(18,25,36)); draw=ImageDraw.Draw(out)
    for col,(ar,title) in enumerate(zip(panels,titles)):
        out.paste(Image.fromarray(ar),(col*w,50));draw.text((col*w+5,10),title,fill='white')
    for row in regions:
        if row.get('reasons') and row.get('bbox'):
            x0,y0,x1,y1=row['bbox'];draw.rectangle((x0+3*w,y0+50,x1+3*w-1,y1+49),outline=(255,80,110))
    out.save(path)


def render(out, result):
    cards=[]
    for v in result['views']:
        counts={}
        for r in v.get('regions',[]):
            for reason in r.get('reasons',[]): counts[reason]=counts.get(reason,0)+1
        cards.append('<article><h3>帧 '+str(v['decode_index'])+'</h3><p>'+html.escape(str(counts))+'</p>'+
                     ('<a href="'+v['overlay']+'"><img loading="lazy" src="'+v['overlay']+'"></a>' if v.get('overlay') else '')+'</article>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>三维点支撑与局部形状检查</title><style>'
        'body{background:#14202c;color:#edf4ff;font:16px system-ui;margin:24px}a{color:#7ccaff}img{width:100%}article{margin:24px 0}pre{white-space:pre-wrap}</style>'
        '<h1>三维点支撑 → 原始网格 → 后处理模型</h1><p>'+html.escape(result['status'])+'</p>'
        '<p>蓝色：条件性覆盖；橙色：掩膜内未覆盖；黄色：输入掩膜边缘。点支撑列只显示与该相机保存深度相符的点在 8 像素内的邻域；缺深度显示 UNKNOWN。后两列仅是网格轮廓，不代表真实三维表面。</p>'
        '<p>自动按物体包围框分成 8×12 个区域；局部缺失不会被整体覆盖率掩盖。红框为需要复核区域，不据此自动删点或补面。</p>'
        '<pre>'+html.escape(str(result.get('summary',{})))+'</pre><p><a href="RESULT.json">完整分区结果</a> · <a href="PLAN.json">冻结检查规则与输入</a></p>'
        '<p>本次只诊断，旧模型保持不变；无完整模型、精度或 CAD 验收。</p>'+''.join(cards),encoding='utf8')


def run(surface_run, output, candidate_run=None):
    started=time.monotonic(); out=Path(output).resolve()
    _need(not out.exists() and out.is_relative_to(ROOT/'outputs'),'new repository output required')
    for p in (surface_run,candidate_run):
        if p is not None:
            source=Path(p).resolve()
            _need(not source.is_relative_to(out),'output encloses source')
            _need(not out.is_relative_to(source) or out==source/'shape_support','output overlaps source subtree')
    refs=[]; out.mkdir(parents=True)
    result=dict(schema='shape-support-audit-v1',status='REVIEW_REQUIRED',source_surface_run=str(Path(surface_run).resolve()),
                candidate_run=None if candidate_run is None else str(Path(candidate_run).resolve()),
                source_unchanged=False,geometry_changed=False,candidate_promotable=False,
                quality_status='NOT_EVALUATED',physical_scale='unknown',views=[],limitations=LIMITATIONS)
    def check():
        if time.monotonic()-started>540: raise TimeoutError('shape support 540s deadline')
    try:
        from .temporal_review import load_cv
        cv=load_cv(refs);cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
        _bind(Path(__file__),refs)
        source,raw,final,dense,views=context(surface_run,candidate_run,refs);check()
        selected=_select(views,np.asarray(raw['vertices']).mean(axis=0))
        _save(out/'PLAN.json',dict(policy=POLICY,source_bindings=refs,selection_rule='FOUR_TEMPORAL_WORST_DEPTH_THEN_CAMERA_DIRECTION_SPREAD',
            selected_indices=[views[i]['decode_index'] for i in selected],registered_views=len(views),
            scope='ALL_REGISTERED_DEPTHS_SCANNED__SELECTED_VIEWS_PROJECTED',budget=dict(inner_seconds=540,outer_seconds=600,memory_bytes=4<<30)))
        (out/'overlays').mkdir();(out/'per_view').mkdir()
        for vi in selected:
            check();v=views[vi];cam=v['camera'];idx=v['decode_index'];mask=v['mask'];valid=v['valid']
            row=dict(decode_index=idx,view_id=cam['global_id'],status='EVALUATED',regions=[],mask_touches_frame=v['mask_touches_frame'])
            try:
                depth=read_dmap(v['depth_path']).depth if v['depth_path'] else None
                supported=point_support(dense['vertices'],cam,depth,cv)
                dv=None if depth is None else np.isfinite(depth)&(depth>0)
                predictions=[]; projection_errors=[]
                for model in (raw,final):
                    if model is None: predictions.append(None); projection_errors.append('NO_CANDIDATE'); continue
                    try:
                        uv=project(model['vertices'],cam['R'],cam['C'],cam['K'])
                        predictions.append(raster(uv,np.asarray(model['faces'],np.int64),cam['width'],cam['height'],check=check));projection_errors.append(None)
                    except ValueError as exc:
                        predictions.append(None); projection_errors.append(str(exc))
                a,b=predictions
                row['regions']=region_metrics(mask,valid,dv,supported['depth_agreeing_near'],a,b)
                row['projection_errors']=projection_errors
                fg=mask&valid;n=int(fg.sum())
                row['global']=dict(mask_pixels=n,depth_coverage=None if dv is None or n==0 else float((dv&fg).sum()/n),
                    projected_point_pixels=int((supported['projected']&fg).sum()),
                    depth_agreeing_point_pixels=None if supported['depth_agreeing'] is None else int((supported['depth_agreeing']&fg).sum()),
                    raw=None if a is None else metrics(a,mask,valid),final=None if b is None else metrics(b,mask,valid))
                with Image.open(v['image']) as im:rgb=im.convert('RGB')
                _need(rgb.size==(cam['width'],cam['height']),'RGB domain differs')
                row['overlay']=f'overlays/{idx:06d}.png';_overlay(out/row['overlay'],rgb,mask,valid,supported,a,b,row['regions'])
                arrays=dict(mask=mask,valid=valid,projected_points=supported['projected'])
                for k,arr in [('stored_depth_valid',dv),('depth_agreeing_points',supported['depth_agreeing']),('raw',a),('final',b)]:
                    if arr is not None:arrays[k]=arr
                np.savez_compressed(out/'per_view'/f'{idx:06d}.npz',**arrays)
            except (ValueError,FileNotFoundError) as exc:
                row.update(status='UNKNOWN',error=str(exc))
            result['views'].append(row)
        result.update(verdict(result['views'],final is not None))
        coverages=[v['depth_coverage'] for v in views if v.get('depth_coverage') is not None]
        reason_counts={}
        for v in result['views']:
            for r in v['regions']:
                for reason in r['reasons']:reason_counts[reason]=reason_counts.get(reason,0)+1
        result['summary']=dict(dense_points=len(dense['vertices']),raw_vertices=len(raw['vertices']),raw_faces=len(raw['faces']),
            candidate_vertices=None if final is None else len(final['vertices']),candidate_faces=None if final is None else len(final['faces']),
            registered_views=len(views),saved_depth_views=len(coverages),projected_views=len(result['views']),
            mask_touches_frame_count=sum(v['mask_touches_frame'] for v in views),
            unprojected_views=len(views)-len(result['views']),stored_depth_coverage_median=float(np.median(coverages)) if coverages else None,
            region_reason_counts=reason_counts)
    except Exception as exc:
        result.update(status='REVIEW_REQUIRED',candidate_promotable=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            for ref in list(refs):_bind(ref['path'],[],ref)
            result['source_unchanged']=True
        except Exception as exc:
            result.update(source_unchanged=False,status='REVIEW_REQUIRED',candidate_promotable=False,drift_error=str(exc))
        result.update(elapsed_seconds=time.monotonic()-started,source_bindings=refs)
        _save(out/'RESULT.json',result);render(out,result)
    return result
