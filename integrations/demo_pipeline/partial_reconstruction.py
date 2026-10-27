"""Bounded local reconstruction from saved foreground matches, with explicit holes.

This is an engineering perspective reconstruction, not Kozlov's parallel theorem.
The fixed zero distortion assumption is never reported as physical calibration.
"""
from collections import Counter, defaultdict
from pathlib import Path
import html
import json
import os
import shutil
import sqlite3
import time

import numpy as np

from . import parallel_photo_audit as io
from .registration_probe import _binding, _plain

PROFILE = dict(id='local-three-view-k0-v1', min_pair_matches=15, min_track_views=3,
    min_group_tracks=15, max_groups=8, max_images=32, max_points=1000,
    mapping_seconds_per_group=45, final_ba_seconds=15, final_ba_iterations=100,
    max_error_px=2., max_loo_error_px=3., min_angle_deg=2., max_condition=1e4,
    max_edge_px=60., max_edge_factor=3., threads=2, seed=0,
    calibration='shared focal refined; principal point and DB k=0 fixed',
    scale_status='unknown', physical_calibration_known=False)


DEFAULT_PROFILE_ID = PROFILE['id']
WHOLE_VIDEO_PROFILE_ID = 'local-three-view-k0-whole-video-v1'
WHOLE_VIDEO_PROFILE = dict(PROFILE, id=WHOLE_VIDEO_PROFILE_ID, max_images=192)
WHOLE_VIDEO_DENSE_PROFILE_ID = 'local-three-view-k0-whole-video-dense-v1'
WHOLE_VIDEO_DENSE_PROFILE = dict(WHOLE_VIDEO_PROFILE, id=WHOLE_VIDEO_DENSE_PROFILE_ID, max_points=4096)


def resolve_profile(profile_id=DEFAULT_PROFILE_ID):
    if profile_id == DEFAULT_PROFILE_ID: return PROFILE.copy()
    if profile_id == WHOLE_VIDEO_PROFILE_ID: return WHOLE_VIDEO_PROFILE.copy()
    if profile_id == WHOLE_VIDEO_DENSE_PROFILE_ID: return WHOLE_VIDEO_DENSE_PROFILE.copy()
    raise ValueError('unknown partial reconstruction profile')


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(_plain(value), f, ensure_ascii=False, allow_nan=False, indent=2)


class Union:
    def __init__(self): self.parent = {}

    def find(self, x):
        p = self.parent
        if x not in p: p[x] = x
        root = x
        while p[root] != root: root = p[root]
        while p[x] != x:
            old = p[x]; p[x] = root; x = old
        return root

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def graph_plan(image_ids, pairs, *, profile_id=DEFAULT_PROFILE_ID):
    """Use stored geometric inliers, rejecting many-to-one connected tracks.

    Connectivity is a scheduling device, not proof of physical point identity.
    All groups/isolates and all conflicting tracks remain in the ledger.
    """
    profile = resolve_profile(profile_id)
    if len(set(image_ids)) != len(image_ids) or not 1 <= len(image_ids) <= profile['max_images']:
        raise ValueError('bounded unique image IDs required')
    if any(type(i) is not int or i < 1 for i in image_ids): raise ValueError('plain image IDs required')
    images, tracks = Union(), Union()
    for ident in image_ids: images.find(ident)
    edges = []
    seen_pairs=set()
    for a, b, matches in pairs:
        if a not in images.parent or b not in images.parent or a >= b:
            raise ValueError('ordered bound pair identities required')
        if (a,b) in seen_pairs:raise ValueError('duplicate image pair')
        seen_pairs.add((a,b))
        if len({tuple(m) for m in matches})!=len(matches):raise ValueError('duplicate match rows')
        if len(matches) < profile['min_pair_matches']: continue
        edges.append(dict(image_ids=[a, b], matches=len(matches)))
        images.join(a, b)
        for x, y in matches:
            if type(x) is not int or type(y) is not int or x < 0 or y < 0:
                raise ValueError('plain feature rows required')
            tracks.join((a, x), (b, y))
    components, members = defaultdict(list), defaultdict(list)
    for i in image_ids: components[images.find(i)].append(i)
    for key in sorted(tracks.parent): members[tracks.find(key)].append(key)
    track_rows = []
    for keys in members.values():
        unique = len({k[0] for k in keys}) == len(keys)
        track_rows.append(dict(elements=[list(k) for k in keys], conflict=not unique,
                               eligible=unique and len(keys) >= profile['min_track_views']))
    groups = []
    for index, ids in enumerate(sorted(components.values())):
        relevant = [t for t in track_rows if t['elements'][0][0] in ids]
        count = sum(t['eligible'] for t in relevant)
        runnable = len(ids) >= 3 and count >= profile['min_group_tracks']
        groups.append(dict(group_id=index, image_ids=ids, three_view_tracks=count,
            conflicting_tracks=sum(t['conflict'] for t in relevant), scheduled=runnable,
            reason=None if runnable else 'INSUFFICIENT_THREE_VIEW_SUPPORT'))
    if sum(g['scheduled'] for g in groups) > profile['max_groups']:
        raise ValueError('group budget exceeded; no silent truncation')
    return dict(groups=groups, tracks=track_rows, edges=edges)


def prepare(sfm_run, *, profile_id=DEFAULT_PROFILE_ID):
    """Pure saved-input inspection. No native importer or model computation."""
    profile = resolve_profile(profile_id)
    source = Path(sfm_run).resolve(strict=True)
    if source.is_dir(): source /= 'run.json'
    root = source.parent
    refs = {}
    def bind(p, expected=None):
        r = _binding(p)
        if expected is not None and r['sha256'] != expected: raise ValueError('source hash differs: '+str(p))
        refs[r['path']] = r
        return r
    bind(source); run = io.read_json(source)
    if (run.get('contract_version') != ('sfm-native-object-video-v1' if profile_id in (WHOLE_VIDEO_PROFILE_ID, WHOLE_VIDEO_DENSE_PROFILE_ID) else 'sfm-native-object-v1') or run.get('backend', {}).get('version') != '4.2.1'
            or run.get('backend', {}).get('name') != 'pycolmap'):
        raise ValueError('saved foreground pycolmap 4.2.1 source required')
    fmap = root/'frames_map.jsonl'; bind(fmap)
    rows = [io.loads(s) for s in fmap.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if profile_id in (WHOLE_VIDEO_PROFILE_ID, WHOLE_VIDEO_DENSE_PROFILE_ID) and len(rows) > profile['max_images']:
        raise ValueError('whole-video original input row limit exceeded')
    if len({r['input_index'] for r in rows}) != len(rows): raise ValueError('duplicate input identity')
    frames = {}
    masks = run['effective_options']['foreground']['mask_bindings']
    masks_by_index = {r['input_index']: r for r in masks}
    if len(masks_by_index) != len(masks): raise ValueError('duplicate mask identity')
    for r in rows:
        iid = r.get('database_image_id')
        if iid is None: continue
        if type(iid) is not int or iid < 1 or iid in frames or r.get('foreground_selected') is not True:
            raise ValueError('unique selected foreground identities required')
        if r['upstream_io_run_id'] != run['upstream_io_run_id']: raise ValueError('upstream identity differs')
        image = io.under(root, r['staged_image_name']); bind(image, r['canonical_image_sha256'])
        mask = masks_by_index[r['input_index']]
        if mask['image_name'] != r['staged_image_name']: raise ValueError('mask image identity differs')
        mask_path = Path(mask['staged_path']).resolve(strict=True)
        if not mask_path.is_relative_to(root): raise ValueError('staged mask outside source')
        bind(mask_path, mask['sha256'])
        frames[iid] = dict(r, image_path=str(image), mask_path=str(mask_path))
    db = root/'database.db'
    for suffix in ('-wal', '-journal'):
        p = Path(str(db)+suffix)
        if p.exists() and p.stat().st_size: raise ValueError('active source database sidecar')
    bind(db)
    points, pairs = {}, []
    with sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1', uri=True) as con:
        database_images = list(con.execute('SELECT image_id,name,camera_id FROM images'))
        names = {i:n for i,n,c in database_images}
        if set(names) != set(frames) or any(names[i] != frames[i]['staged_image_name'] for i in names):
            raise ValueError('database/frame identity differs')
        cameras = list(con.execute('SELECT camera_id,model,width,height,params FROM cameras'))
        if len(cameras)!=1 or {c for i,n,c in database_images}!={cameras[0][0]}:
            raise ValueError('one shared camera required by this profile')
        for cid, model, width, height, blob in cameras:
            params = np.frombuffer(blob, '<f8').tolist()
            if model != 2 or len(params) != 4 or params[3] != 0 or params[0] <= 0 or not np.isfinite(params).all():
                raise ValueError('SIMPLE_RADIAL initial zero-k camera required')
            from PIL import Image
            if type(width) is not int or type(height) is not int or min(width,height)<=0 or max(width,height)>4096:
                raise ValueError('bounded positive camera dimensions required')
            for frame in frames.values():
                for key in ('image_path','mask_path'):
                    with Image.open(frame[key]) as im:
                        if im.size!=(width,height): raise ValueError('DB camera/image/mask size differs')
                frame['camera_id']=int(cid)
        for iid, n, cols, blob in con.execute('SELECT image_id,rows,cols,data FROM keypoints'):
            if not 0 <= n <= 4096 or cols not in (2,4,6): raise ValueError('bounded keypoint shape required')
            points[iid] = np.frombuffer(blob or b'', '<f4').reshape(n,cols)[:,:2].copy()
            if not np.isfinite(points[iid]).all(): raise ValueError('nonfinite keypoints')
        for pair_id, n, cols, blob in con.execute('SELECT pair_id,rows,cols,data FROM two_view_geometries ORDER BY pair_id'):
            if not 0 <= n <= 4096 or (n and cols != 2): raise ValueError('bounded match shape required')
            if n < profile['min_pair_matches']: continue
            a, b = divmod(pair_id, 2147483647)
            matches = np.frombuffer(blob, '<u4').reshape(n,2)
            if a not in points or b not in points or np.any(matches[:,0] >= len(points[a])) or np.any(matches[:,1] >= len(points[b])):
                raise ValueError('match feature row outside source')
            pairs.append((a,b,matches.tolist()))
    graph = graph_plan(sorted(frames), pairs, profile_id=profile_id)
    for g in graph['groups']: g['input_indices'] = [frames[i]['input_index'] for i in g['image_ids']]
    for ref in refs.values():
        if _binding(ref['path']) != ref: raise ValueError('source drift during planning')
    return dict(schema='local-partial-reconstruction-plan-v1', profile=profile.copy(),
        sfm_run=str(source), image_root=str(root), database=str(db), frames=list(frames.values()),
        graph=graph, source_bindings=list(refs.values()),
        excluded_source_rows=[dict(input_index=r['input_index'], status=r['status'], reason=r.get('reason')) for r in rows if r.get('database_image_id') is None],
        calibration_accepted=False, components_merged=False, hole_filling=False,
        fitting_scope='All saved geometric matches in scheduled group may affect camera fitting; only exported points pass the conflict/three-view gate.'), points


def pipeline_options(pc, group, frames):
    o = pc.IncrementalPipelineOptions()
    o.num_threads=2; o.random_seed=0; o.multiple_models=False; o.max_num_models=1
    o.min_model_size=3; o.image_names=[frames[i]['staged_image_name'] for i in group['image_ids']]
    o.max_runtime_seconds=PROFILE['mapping_seconds_per_group']
    o.ba_refine_extra_params=False; o.ba_refine_principal_point=False
    o.ba_refine_focal_length=True; o.ba_use_gpu=False
    o.mapper.abs_pose_refine_extra_params=False; o.mapper.abs_pose_refine_focal_length=True
    o.mapper.num_threads=2; o.mapper.random_seed=0
    o.structure_less_registration_fallback=False
    o.triangulation.ignore_two_view_tracks=True
    o.ba_global_max_num_iterations=100; o.ba_local_max_num_iterations=50
    return o


def write_ply(path, points, faces=()):
    lines=['ply','format ascii 1.0',f'element vertex {len(points)}',
        'property double x','property double y','property double z',
        'property uchar red','property uchar green','property uchar blue',
        f'element face {len(faces)}','property list uchar int vertex_indices','end_header']
    for p in points:
        lines.append(' '.join(format(float(x),'.17g') for x in p['xyz'])+' '+' '.join(map(str,p['color'])))
    lines += ['3 '+' '.join(map(str,f)) for f in faces]
    Path(path).write_text('\n'.join(lines)+'\n',encoding='ascii')


def supported_mask_faces(points, mesh, masks):
    """Reject interpolation across any foreground-mask hole in common views."""
    from PIL import Image, ImageDraw
    kept, supports = [], []
    for face, support in zip(mesh['faces'], mesh['face_support']):
        ids = support if isinstance(support,list) else support['common_image_ids']
        okay = True
        for iid in ids:
            uv = [[o['raw_pixels'] for o in points[i]['observations'] if o['image_id']==iid][0] for i in face]
            mask = masks[iid]
            raster = Image.new('1', (mask.shape[1], mask.shape[0]), 0)
            ImageDraw.Draw(raster).polygon([tuple(v) for v in uv], fill=1)
            area = np.asarray(raster, dtype=bool)
            if np.any(area & ~mask): okay=False; break
        if okay: kept.append(face); supports.append(support)
    return kept, supports


def run(plan_path, output):
    """Run only inside the existing bounded Job supervisor. One call per group."""
    from PIL import Image, ImageDraw
    from .partial_point_gate import assess_point
    from .partial_surface import build_faces
    from kozlov_lab.reconstruction.export import compare_models
    import pycolmap as pc
    if pc.__version__ != '4.2.1': raise ValueError('frozen backend version required')
    plan = io.read_json(plan_path)
    profile_id = plan['profile']['id']
    profile = resolve_profile(profile_id)
    if plan['profile'] != profile: raise ValueError('profile drift')
    fresh, keypoints = prepare(plan['sfm_run'], profile_id=profile_id)
    if fresh != plan: raise ValueError('planned input or groups changed')
    output=Path(output).resolve()
    source=Path(plan['image_root'])
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output): raise ValueError('fresh nonoverlapping output required')
    output.mkdir(parents=True)
    frames={f['database_image_id']:f for f in plan['frames']}
    masks={}
    for iid,f in frames.items():
        with Image.open(f['mask_path']) as im:
            data=np.asarray(im).copy()
            if im.mode!='L' or not np.isin(data,[0,255]).all(): raise ValueError('binary L mask required')
            masks[iid]=data!=0
        with Image.open(f['image_path']) as im:
            if im.size!=(data.shape[1],data.shape[0]): raise ValueError('mask/image dimensions differ')
    track_index={tuple(e):n for n,t in enumerate(plan['graph']['tracks']) for e in t['elements']}
    result=dict(schema='local-partial-reconstruction-result-v1',status='RUNNING',groups=[],
        profile=profile,calibration_accepted=False,geometry_accepted=False,scale_status='unknown',
        quality_status='NOT_EVALUATED',components_merged=False,hole_filling=False,source_unchanged=None,
        mapping_calls=0,final_ba_calls=0,scenes=[],source_bindings=plan['source_bindings'])
    def event(kind,**data):
        with (output/'EVENTS.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(kind=kind,monotonic=time.monotonic(),**data),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    for group in plan['graph']['groups']:
        entry=dict(group,status='NOT_RUN',models=[])
        result['groups'].append(entry)
        if not group['scheduled']: continue
        folder=output/('group_%02d'%group['group_id']);folder.mkdir()
        snapshot=folder/'database.db';shutil.copyfile(plan['database'],snapshot)
        options=pipeline_options(pc,group,frames)
        save(folder/'OPTIONS.json',options.todict())
        native_output=folder/'mapping';native_output.mkdir()
        try:
            event('mapping_begin',group_id=group['group_id']);result['mapping_calls']+=1
            models=pc.incremental_mapping(snapshot,plan['image_root'],native_output,options)
            event('mapping_return',group_id=group['group_id'],models=len(models))
            entry['status']='NO_MODEL' if not models else 'EVALUATED'
            for model_id,rec in sorted(models.items()):
                item=dict(model_id=int(model_id),status='CANDIDATE',points_before=len(rec.points3D),
                          registered_input_indices=[frames[int(i)]['input_index'] for i in sorted(rec.reg_image_ids())])
                entry['models'].append(item)
                if len(rec.reg_image_ids())<3 or len(rec.points3D)>profile['max_points']:
                    item['status']='MODEL_SCOPE_REFUSED';continue
                if not set(rec.reg_image_ids())<=set(group['image_ids']): raise ValueError('model escaped group')
                if any(c.model_name!='SIMPLE_RADIAL' or c.params[3]!=0 for c in rec.cameras.values()): raise ValueError('fixed k changed')
                ba=pc.BundleAdjustmentOptions();ba.refine_extra_params=False;ba.refine_principal_point=False
                ba.ceres.use_gpu=False;ba.ceres.solver_options.num_threads=2
                ba.ceres.solver_options.max_num_iterations=profile['final_ba_iterations']
                ba.ceres.solver_options.max_solver_time_in_seconds=profile['final_ba_seconds']
                config=pc.BundleAdjustmentConfig()
                for iid in rec.reg_image_ids():config.add_image(iid)
                config.fix_gauge(pc.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
                result['final_ba_calls']+=1;event('final_ba_begin',group_id=group['group_id'],model_id=int(model_id))
                summary=pc.create_default_bundle_adjuster(ba,config,rec).solve()
                item['ba_summary']=_plain(summary.todict());item['ba_report']=summary.brief_report()
                termination=str(summary.termination_type).rsplit('.',1)[-1]
                item['ba_converged']=termination=='CONVERGENCE'
                event('final_ba_return',group_id=group['group_id'],termination=termination)
                final=folder/('model_%d'%model_id);final.mkdir();rec.write(final)
                reread=pc.Reconstruction(final);differences=compare_models(rec,reread,tolerance=1e-10)
                item['roundtrip_differences']=differences
                if differences:raise ValueError('native roundtrip mismatch')
                ledger=[];supported=[];reasons=Counter()
                for pid,p in sorted(rec.points3D.items()):
                    observations=[];problems=[];components=set()
                    for e in p.track.elements:
                        iid,row=int(e.image_id),int(e.point2D_idx)
                        image=rec.image(iid);cam=rec.cameras[image.camera_id];pose=image.cam_from_world()
                        xy=np.asarray(image.points2D[row].xy)
                        if row>=len(keypoints[iid]) or not np.allclose(xy,keypoints[iid][row],rtol=0,atol=1e-6):raise ValueError('native observation differs from DB')
                        if int(image.points2D[row].point3D_id)!=int(pid): raise ValueError('nonreciprocal track')
                        ti=track_index.get((iid,row))
                        if ti is None:problems.append('MISSING_SOURCE_TRACK')
                        else:
                            components.add(ti)
                            if not plan['graph']['tracks'][ti]['eligible']:problems.append('SOURCE_TRACK_CONFLICT_OR_TOO_SHORT')
                        x,y=np.floor(xy).astype(int);mask=masks[iid]
                        foreground=bool(0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x])
                        observations.append(dict(image_id=iid,feature_row=row,input_index=frames[iid]['input_index'],
                            raw_pixels=xy.tolist(),params=list(cam.params),R=pose.rotation.matrix().tolist(),t=pose.translation.tolist(),foreground=foreground))
                    if len(components)!=1:problems.append('DISCONNECTED_SOURCE_TRACK')
                    decision=assess_point(p.xyz,observations,max_error_px=profile['max_error_px'],
                        min_angle_deg=profile['min_angle_deg'],max_loo_error_px=profile['max_loo_error_px'])
                    problems+=decision['reasons']
                    if not item['ba_converged']:problems.append('BA_NOT_CONVERGED')
                    if any(c.has_bogus_params(.1,10.,1.) for c in rec.cameras.values()):problems.append('BOGUS_CAMERA')
                    accepted=not problems and decision['status']=='CONDITIONAL_POINT_SUPPORTED'
                    row=dict(point_id=int(pid),xyz=p.xyz.tolist(),color=p.color.tolist(),observations=observations,
                        accepted_conditionally=accepted,reasons=sorted(set(problems)),decision=decision)
                    ledger.append(row);reasons.update(row['reasons'])
                    if accepted:supported.append(row)
                save(final/'POINT_LEDGER.json',ledger)
                item.update(supported_points=len(supported),rejected_points=len(ledger)-len(supported),rejection_reasons=dict(reasons),
                            camera_params={str(cid):list(c.params) for cid,c in rec.cameras.items()})
                if supported:
                    generated_faces,supports,diagnostics=build_faces(supported,max_edge_px=profile['max_edge_px'],max_edge_factor=profile['max_edge_factor'])
                    mesh=dict(faces=generated_faces,face_support=supports,diagnostics=diagnostics)
                    faces,face_support=supported_mask_faces(supported,mesh,masks)
                    save(final/'SURFACE_SUPPORT.json',dict(generator=mesh,faces_after_mask_gate=faces,face_support=face_support,
                        mask_rejected_faces=len(mesh['faces'])-len(faces),surface_kind='local_linear_interpolation_not_measured_surface',hole_filling=False))
                    ply=final/'supported.ply';write_ply(ply,supported,faces)
                    from integrations.viewer.ply_data import read_ply
                    readback=read_ply(ply)
                    if (readback['vertices'] != [p['xyz'] for p in supported] or readback['faces'] != faces
                            or readback['colors'] != [p['color'] for p in supported]):
                        raise ValueError('PLY roundtrip differs')
                    item['ply_readback']=dict(vertices=len(supported),faces=len(faces),binding=_binding(ply))
                    item['faces']=len(faces);item['status']='PARTIAL_CONDITIONAL_MODEL'
                    result['scenes'].append(dict(ply=str(ply),sha256=_binding(ply)['sha256'],
                        title='局部视角 '+str(item['registered_input_indices']),data_kind='real_photos',declared_unit=None,
                        note='未标定相机条件下的局部重建；仅三视图支持点。面为局部插值；不补洞，不代表完整物体或尺寸精度。'))
                    # Original images are not edited. These overlays expose accepted/rejected locations.
                    for iid in sorted(rec.reg_image_ids()):
                        with Image.open(frames[iid]['image_path']) as im:overlay=im.convert('RGB')
                        draw=ImageDraw.Draw(overlay)
                        for point in ledger:
                            for o in point['observations']:
                                if o['image_id']==iid:
                                    x,y=o['raw_pixels'];draw.ellipse((x-2,y-2,x+2,y+2),fill=(40,240,100) if point['accepted_conditionally'] else (240,60,70))
                        overlay.save(final/('view_%06d.png'%frames[iid]['input_index']))
                else:item['status']='NO_SUPPORTED_POINTS'
        except Exception as exc:
            entry.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
            event('group_failure',group_id=group['group_id'],error=entry['error'])
            # No scene from a group with incomplete validation can be published.
            result['scenes']=[s for s in result['scenes'] if not Path(s['ply']).is_relative_to(folder)]
    drift=[r['path'] for r in plan['source_bindings'] if _binding(r['path'])!=r]
    result['source_unchanged']=not drift;result['source_drift']=drift
    failures=[g['group_id'] for g in result['groups'] if g['status']=='FAILED']
    result['failed_groups']=failures
    if drift: result['status']='FAILED_SOURCE_DRIFT';result['scenes']=[]
    elif failures:result['status']='PARTIAL_WITH_FAILURE' if result['scenes'] else 'FAILED'
    else:result['status']='PARTIAL_CONDITIONAL_MODEL' if result['scenes'] else 'NO_SUPPORTED_LOCAL_MODEL'
    if result['scenes']:
        spec=output/'SCENES.json';save(spec,dict(schema='ply-viewer-scenes-v1',scenes=result['scenes']))
        from integrations.viewer.export import export_html
        save(output/'VIEWER_RESULT.json',export_html(spec,output/'viewer.html'))
    save(output/'RESULT.json',result);render(output,result)
    return result


def render(output, result):
    page='<!doctype html><meta charset="utf-8"><title>局部建模</title><style>body{font:16px system-ui;max-width:1100px;margin:35px auto;background:#102029;color:#eef5f7}a{color:#86cfff}td,th{padding:12px;border:1px solid #527080}table{border-collapse:collapse}p{line-height:1.6}</style>'
    page+='<h1>只重建可支持的局部</h1><p>各组独立坐标，未拼接。相机未独立标定，尺度未知；不补洞。三视图、重投影、留一法和视差检查只是条件内部一致性，不是实物真值。</p>'
    page+='<p>'+html.escape(result['status'])+'</p>'
    if result['scenes']:page+='<p><a href="viewer.html">打开局部3D模型</a> · <a href="SCENES.json">Blender导出场景</a></p>'
    page+='<table><tr><th>原始帧</th><th>结果</th><th>模型/点/面</th></tr>'
    for g in result['groups']:
        stats=[dict(views=m.get('registered_input_indices'),points=m.get('supported_points'),faces=m.get('faces'),status=m['status'],ba_converged=m.get('ba_converged')) for m in g['models']]
        page+='<tr><td>'+html.escape(str(g['input_indices']))+'</td><td>'+html.escape(g['status'])+'</td><td>'+html.escape(str(stats))+'</td></tr>'
    page+='</table><p>绿色叠点=通过本轮条件门；红色=拒绝。局部三角面是短边插值，不能证明面内部细节。未采样区域、缺失侧面和分量间空隙保留。</p><p><a href="RESULT.json">全部状态、来源与拒绝原因</a></p>'
    (output/'index.html').write_text(page,encoding='utf-8')
