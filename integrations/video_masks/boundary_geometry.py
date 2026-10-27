"""Automatic camera-bound RGB/depth boundary diagnostics; never edits geometry."""
from __future__ import annotations
import csv
import html
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from .auto_boundary import bind, unchanged, execute_auto_boundary, READY
from .boundary_precision import _read, _save

ROOT = Path(__file__).resolve().parents[2]
READY_GEOMETRY = 'BOUNDARY_GEOMETRY_DIAGNOSTIC_NOT_ACCEPTED'


def select_registered(views, risks, center, max_frames=16):
    """Select only exact registered views with depth, then spread view directions.

    One highest-expansion view per temporal stratum starts the set. Subsequent
    views maximize their minimum angle to the selected directions. Neither
    expansion nor camera separation is a claim of segmentation correctness.
    """
    if type(max_frames) is not int or not 1 <= max_frames <= 32:
        raise ValueError('max_frames must be 1..32')
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('finite model center required')
    by = {}; excluded = []; directions = {}
    for view in views:
        i = view['frame']['decode_index']
        if type(i) is not int or i < 0 or i in by or any(x['decode_index']==i for x in excluded):
            raise ValueError('unique registered integer frame indices required')
        if not view.get('depth_path'):
            excluded.append(dict(decode_index=i, reason='NO_BOUND_DEPTH')); continue
        if view.get('frame_identity',{}).get('status')=='UNVERIFIED_FRAME_IDENTITY':
            excluded.append(dict(decode_index=i, reason='UNVERIFIED_FRAME_IDENTITY')); continue
        vector = np.asarray(view['camera_scene']['C'],float)-center
        length = np.linalg.norm(vector)
        if vector.shape!=(3,) or not np.isfinite(vector).all() or length<=1e-12:
            excluded.append(dict(decode_index=i, reason='INVALID_VIEW_DIRECTION')); continue
        score = risks.get(i,0)
        if type(score) not in (int,float) or not math.isfinite(score) or score<0:
            raise ValueError('finite nonnegative expansion score required')
        by[i] = view; directions[i] = vector/length
    ids = sorted(by); selected=[]
    for group in np.array_split(ids,min(4,max_frames,len(ids))) if ids else []:
        selected.append(max((int(i) for i in group),key=lambda i:(risks.get(i,0),-i)))
    while len(selected)<min(max_frames,len(ids)):
        def score(i):
            # acos is monotonic; use the smallest separation cosine's negative.
            separation = -max(float(np.dot(directions[i],directions[j])) for j in selected)
            return separation, risks.get(i,0), -i
        selected.append(max((i for i in ids if i not in selected),key=score))
    return dict(indices=sorted(selected),eligible_count=len(ids),registered_count=len(views),
        unselected_eligible_count=len(ids)-len(selected),excluded=sorted(excluded,key=lambda r:r['decode_index']),
        rule='REGISTERED_DEPTH_FOUR_TEMPORAL_RISK_STRATA_THEN_CAMERA_DIRECTION_SPREAD',
        full_timeline_reviewed=False,unselected_status='NOT_EVALUATED',max_frames=max_frames)


def combine(vertices,faces,rows,cameras,prior_protected=None):
    from .face_mask_gate import aggregate_views, Policy
    agg=aggregate_views(vertices,faces,rows,cameras,Policy(min_angle_degrees=8))
    protected=np.zeros(len(faces),bool)
    if prior_protected is not None:
        old=np.asarray(prior_protected)
        if old.dtype!=bool or old.shape!=protected.shape:raise ValueError('bound prior protection shape')
        protected |= old
    for row in rows:protected |= row['core_protected']
    candidate=agg['quarantine'] & (agg['conflict_views']>=2) & ~protected
    return dict(candidate=candidate,protected=protected,conflict_views=agg['conflict_views'],
        max_angle=agg['max_conflict_angle_degrees'],witness_view_ids=agg['witness_view_ids'])


def _overlay(path, domain, camera, vertices, faces, conflict):
    with Image.open(domain['image_path']) as im:im=im.convert('RGB')
    h,w=domain['mask'].shape
    if im.size!=(w,h):raise ValueError('overlay domain differs from camera')
    rgb=np.array(im);support=domain['supported_gap']
    tint=rgb.copy();tint[support]=(30,230,120)
    panel=Image.new('RGB',(2*w,h+44),(20,28,42));panel.paste(im,(0,44));panel.paste(Image.fromarray(tint),(w,44))
    draw=ImageDraw.Draw(panel);draw.text((6,8),'UNDISTORTED SOURCE',fill='white')
    draw.text((w+6,8),'GREEN: RGB GAP SUPPORT / RED: CONFLICT',fill='white')
    xyz=(vertices-np.asarray(camera['C']))@np.asarray(camera['R']).T
    q=xyz@np.asarray(camera['K']).T
    with np.errstate(divide='ignore',invalid='ignore'):uv=q[:,:2]/q[:,2:]
    for fid in np.flatnonzero(conflict)[:500]:
        pts=[(float(x+w),float(y+44)) for x,y in uv[faces[fid]]]
        draw.line(pts+[pts[0]],fill=(255,65,55),width=1)
    panel.save(path)


def render(out,result):
    cards=[]
    for row in result.get('views',[]):
        idx=row['decode_index']; link=row.get('overlay')
        image=f'<a href="{link}"><img loading="lazy" src="{link}"></a>' if link else ''
        cards.append('<article><h3>帧 '+str(idx)+'</h3><p>'+html.escape(str({k:v for k,v in row.items() if k not in ('overlay','identity')}))+'</p>'+image+'</article>')
    summary=html.escape(str(result.get('counts',result.get('reason',result.get('error','')))))
    base=html.escape(result.get('artifact_base',''),quote=True)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>边界与三维证据</title>'
        '<style>body{background:#111d2c;color:#e5efff;font:16px system-ui;margin:24px}a{color:#74cbff}article{background:#1b2b40;padding:16px;margin:16px 0}img{max-width:100%;max-height:760px}p{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>'
        '<h1>自动选择已注册视角，检查物体边界</h1><p>'+html.escape(result['status'])+'</p>'
        '<p>先核对视频帧、相机和去畸变域，再检查 RGB 缝隙与保存深度。弱证据、遮挡或缺少深度不作为删面理由。</p>'
        '<p><strong>本阶段仅诊断，未改动任何顶点、面或掩膜。候选冲突也不等于已证实的错误。</strong></p>'
        '<pre>'+summary+'</pre><p><a href="RESULT.json">结果记录</a> · <a href="'+base+'SELECTION.json">自动选帧</a> · '
        '<a href="'+base+'rgb_review/index.html">原分辨率边界检查</a> · <a href="mesh_viewer.html">原模型（未改动）</a> · '
        '<a href="'+base+'FACE_LEDGER.csv">逐面记录</a></p>'+''.join(cards),encoding='utf8')


def publish_mesh_viewer(out,mesh_path,evidence):
    """Persist immutable evidence before the viewer opens any referenced file."""
    from integrations.viewer.export import export_html
    out=Path(out);mesh_path=Path(mesh_path);mesh_ref=bind(mesh_path)
    _save(out/'EVIDENCE.json',evidence)
    _save(out/'scenes.json',dict(schema='ply-viewer-scenes-v1',scenes=[dict(ply=str(mesh_path),sha256=mesh_ref['sha256'],
        title='原局部模型（本轮只诊断，未改动）',data_kind='real_photos',declared_unit=None,
        note='Unchanged local candidate; no geometry acceptance or measured scale.',evidence=[str(out/'EVIDENCE.json')])]))
    export_html(out/'scenes.json',out/'mesh_viewer.html',profile='surface400k')
    return mesh_ref


def run(surface_run,local_run,output,max_frames=16):
    from .boundary_geometry_binding import prepare_context, warp_evidence
    from .boundary_geometry_evidence import evaluate_rgb_view
    from .face_mask_adapter import LIMITS
    from integrations.viewer.ply_data import read_ply
    from integrations.viewer.export import export_html
    started=time.monotonic();out=Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'outputs'):
        raise ValueError('fresh repository output required')
    for source in (surface_run,local_run):
        p=Path(source).resolve()
        if out.is_relative_to(p) and out!=p/'boundary_geometry':
            # The pipeline uses a sibling of face_local, within the surface run.
            raise ValueError('output overlaps protected source subtree')
        if p.is_relative_to(out):raise ValueError('output encloses source')
    ctx=prepare_context(surface_run,local_run)
    out.mkdir(parents=True)
    refs=list(ctx.get('source_bindings',[]));result=dict(status='RUNNING',views=[],source_unchanged=False,
        geometry_accepted=False,geometry_changed=False,mask_changed=False,new_model_executions=0,
        source_run=str(Path(surface_run).resolve()),local_run=str(Path(local_run).resolve()),
        method_origin='ENGINEERING_DIAGNOSTIC_NOT_KOZLOV_THEOREM',report=str(out/'index.html'))
    def check():
        if time.monotonic()-started>540:raise TimeoutError('boundary geometry total deadline')
    try:
        for name in ('boundary_geometry.py','boundary_geometry_binding.py','boundary_geometry_evidence.py'):
            refs.append(bind(Path(__file__).with_name(name)))
        if ctx.get('status')=='NOT_APPLICABLE':
            unchanged({r['path']:r for r in refs}.values())
            result.update(status='NOT_APPLICABLE',reason=ctx['reason'],source_unchanged=True)
            return result
        mesh_path=Path(local_run)/'retained.ply';refs.append(bind(mesh_path))
        mesh=read_ply(mesh_path,**LIMITS);v=np.asarray(mesh['vertices'],float).reshape(-1,3);f=np.asarray(mesh['faces'],np.int64).reshape(-1,3)
        fusion_path=Path(ctx['union_run'])/'FUSION.json';refs.append(bind(fusion_path))
        risks={r['decode_index']:r['added_fraction_of_base'] for r in _read(fusion_path)['rows']}
        selection=select_registered(ctx['views'],risks,v.mean(axis=0),max_frames)
        _save(out/'SELECTION.json',selection)
        _save(out/'PLAN.json',dict(source_bindings=refs,selection=selection,
            budget=dict(wall_seconds=540,outer_job_seconds=600,memory_bytes=4<<30),
            geometry_changes_authorized=False,required_views=2,minimum_angle_degrees=8,
            no_camera_interpolation=True,no_mask_dilation=True))
        if not selection['indices']:
            unchanged({r['path']:r for r in refs}.values())
            result.update(status=READY_GEOMETRY,reason='NO_REGISTERED_DEPTH_VIEWS',source_unchanged=True,
                counts=dict(selected_views=0,multi_view_candidate_faces=0,changed_faces=0))
            return result
        review=execute_auto_boundary(ctx['union_run'],out/'rgb_review',indices=selection['indices'],max_frames=max_frames)
        if review['status']!=READY or not review['source_unchanged']:raise ValueError('RGB evidence review incomplete')
        refs.append(bind(out/'rgb_review/RESULT.json'))
        (out/'per_view').mkdir();(out/'overlays').mkdir()
        by={row['frame']['decode_index']:row for row in ctx['views']};rows=[];cameras={}
        for idx in selection['indices']:
            check();view=by[idx];cam=view['camera_scene']
            try:
                domain=warp_evidence(view,out/'rgb_review',refs)
            except ValueError as exc:
                result['views'].append(dict(decode_index=idx,status='UNKNOWN_IDENTITY_OR_DOMAIN',reason=str(exc)))
                continue
            row=evaluate_rgb_view(v,f,K=cam['K'],R=cam['R'],C=cam['C'],mask=domain['mask'],
                validity=domain['validity'],depth=domain['depth'],supported_gap=domain['supported_gap'],
                view_id=str(cam['global_id']),check=check)
            np.savez_compressed(out/'per_view'/f'{idx:06d}.npz',**{k:a for k,a in row.items() if isinstance(a,np.ndarray)})
            rows.append(row);cameras[row['view_id']]=cam['C']
            link=f'overlays/{idx:06d}.png';_overlay(out/link,domain,cam,v,f,row['negative'])
            values,counts=np.unique(row['reason'],return_counts=True)
            names=row.get('reason_names',{})
            reasons={str(names.get(int(code),names.get(str(int(code)),str(code)))):int(count) for code,count in zip(values,counts)}
            result['views'].append(dict(decode_index=idx,view_id=row['view_id'],status='EVALUATED_NOT_ACCEPTED',
                rgb_supported_pixels=int(domain['supported_gap'].sum()),negative_faces=int(row['negative'].sum()),
                core_protected_faces=int(row['core_protected'].sum()),reason_counts=reasons,
                identity=domain.get('identity'),overlay=link))
            _save(out/'PROGRESS.json',dict(completed_views=len(result['views']),last_decode_index=idx))
            print('BOUNDARY_GEOMETRY_VIEW',idx,'support',int(domain['supported_gap'].sum()),'negative',int(row['negative'].sum()),flush=True)
        prior=None
        decision=Path(local_run)/'DECISIONS.npz';mapping=Path(local_run)/'retained.mapping.npz'
        if decision.is_file() and mapping.is_file():
            refs.extend([bind(decision),bind(mapping)])
            with np.load(mapping,allow_pickle=False) as a:ids=a['source_face_ids'].copy()
            with np.load(decision,allow_pickle=False) as a:prior=a['protected_any'][ids].copy()
        joined=combine(v,f,rows,cameras,prior)
        np.savez_compressed(out/'DECISIONS.npz',**{k:a for k,a in joined.items() if isinstance(a,np.ndarray)})
        with (out/'FACE_LEDGER.csv').open('x',encoding='utf8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['retained_face_id','diagnostic_candidate','protected','negative_views','max_angle_degrees','witness_view_ids','geometry_changed'])
            for i in range(len(f)):writer.writerow([i,bool(joined['candidate'][i]),bool(joined['protected'][i]),int(joined['conflict_views'][i]),float(joined['max_angle'][i]),joined['witness_view_ids'][i],False])
        mesh_ref=publish_mesh_viewer(out,mesh_path,dict(status='COMPUTED_NOT_GEOMETRY_ACCEPTED',
            geometry_changed=False,geometry_accepted=False,source_run=result['source_run'],local_run=result['local_run'],
            views=result['views'],multi_view_candidate_faces=int(joined['candidate'].sum())))
        refs=list({r['path']:r for r in refs}.values())
        unchanged(refs);check()
        result.update(status=READY_GEOMETRY,source_unchanged=True,mesh=mesh_ref,
            counts=dict(registered_views=len(ctx['views']),available_depth_views=sum(bool(x.get('depth_path')) for x in ctx['views']),
                eligible_depth_identity_views=selection['eligible_count'],selected_views=len(selection['indices']),
                evaluated_views=len(rows),unknown_views=len(selection['indices'])-len(rows),
                vertices=len(v),faces=len(f),multi_view_candidate_faces=int(joined['candidate'].sum()),
                protected_faces=int(joined['protected'].sum()),changed_faces=0),
            limitations=['RGB contrast is not physical background truth.', 'Depth and cameras are estimates, not ground truth.',
                'Only selected registered frames evaluated; no interpolation of missing poses.',
                'Pixel-center coverage is not a full continuous triangle visibility proof.',
                'Unchanged model may retain adhesions; this run does not certify their removal.'])
    except Exception as exc:
        result.update(status='FAILED',error=type(exc).__name__+': '+str(exc))
    finally:
        refs=list({r['path']:r for r in refs}.values())
        result.update(elapsed_seconds=time.monotonic()-started,bindings=refs)
        _save(out/'RESULT.json',result);render(out,result)
    return result
