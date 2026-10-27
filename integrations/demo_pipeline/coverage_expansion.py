"""Object-mask candidate coverage and conditional densification of saved local poses.

Background points never enter the object's 3D model. No pose is re-estimated and
no unknown segmentation is replaced by a fabricated mask.
"""
from collections import Counter
from pathlib import Path
import html
import json
import numpy as np
from PIL import Image, ImageDraw

from . import parallel_photo_audit as io
from .partial_reconstruction import save, write_ply, supported_mask_faces
from .partial_surface import build_faces
from .registration_probe import _binding
from .temporal_review import load_inputs, load_cv, read_bound, bind

PROFILE = dict(id='mask-coverage-expansion-v2', max_groups=8,
    pose_policy='reuse_saved_local_cameras_without_refinement',
    strict=dict(reprojection_px=2., loo_px=3., angle_deg=2.),
    relaxed=dict(reprojection_px=4., loo_px=6., angle_deg=1.),
    dedup_common_views=2, dedup_px=2., max_points_per_group=3000,
    max_surface_points=1000, threads=2, scale_status='unknown',
    geometry_accepted=False, hole_filling=False)


def camera_table(ledger):
    """Read actual saved cameras; inconsistent copies cannot define a model."""
    cameras = {}
    for point in ledger:
        for obs in point['observations']:
            fields = {k: obs[k] for k in ('image_id', 'input_index', 'params', 'R', 't')}
            index = fields['input_index']
            if index in cameras and cameras[index] != fields:
                raise ValueError('camera records disagree')
            cameras[index] = fields
    return cameras


def duplicate_observations(obs, points):
    """Explicit local image-space deduplication, not a physical identity proof."""
    query = {o['image_id']: np.asarray(o['raw_pixels']) for o in obs}
    for point in points:
        other = {o['image_id']: o['raw_pixels'] for o in point['observations']}
        common = set(query) & set(other)
        if len(common) >= PROFILE['dedup_common_views'] and all(
                np.linalg.norm(query[i]-other[i]) <= PROFILE['dedup_px'] for i in common):
            return point['point_id']
    return None


def occupancy(points, mask, columns=8, rows=12, bounds=None):
    """Count occupied cells, not surface area or reconstruction completeness."""
    h, w = mask.shape
    from kozlov_lab.stereo.coverage_tracks import grid_edges, grid_cell
    bounds = [0,0,w,h] if bounds is None else bounds
    x0,y0,x1,y1 = bounds
    ys, xs = np.where(mask)
    inside = (xs>=x0)&(xs<x1)&(ys>=y0)&(ys<y1)
    xs,ys = xs[inside],ys[inside]
    xe,ye = grid_edges(bounds,columns,rows)
    available = set(zip(np.searchsorted(xe,xs,side='right')-1,np.searchsorted(ye,ys,side='right')-1))
    covered = set()
    for xy in points:
        x, y = np.floor(xy).astype(int)
        if 0 <= x < w and 0 <= y < h and x0<=x<x1 and y0<=y<y1 and mask[y,x]:
            row,col = grid_cell([x,y],bounds,columns,rows)
            covered.add((col,row))
    return dict(occupied_cells=len(covered), available_cells=len(available),
                occupied_fraction=len(covered)/len(available) if available else None)


def load_local_models(local, frames, refs, sfm_run):
    result = read_bound(local/'RESULT.json', refs)
    if result['status'] != 'PARTIAL_CONDITIONAL_MODEL' or not result['source_unchanged']:
        raise ValueError('committed local candidate source required')
    for ref in result['source_bindings']:
        bind(ref['path'], refs, ref)
    source_paths={str(Path(r['path']).resolve()) for r in result['source_bindings']}
    sfm_run=Path(sfm_run).resolve(strict=True)
    if str(sfm_run) not in source_paths:
        raise ValueError('local models belong to a different SfM input')
    sfm=io.read_json(sfm_run)
    fmap=[io.loads(s) for s in (sfm_run.parent/'frames_map.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    old_frames={r['input_index']:r for r in fmap}
    old_masks={m['input_index']:m for m in sfm['effective_options']['foreground']['mask_bindings']}
    models = []
    by_index = {f['input_index']: f for f in frames}
    import pycolmap as pc
    if pc.__version__!='4.2.1':raise ValueError('pycolmap 4.2.1 required for native readback')
    bind(pc.__file__,refs)
    from .partial_point_gate import assess_point
    from integrations.viewer.ply_data import read_ply
    for group in result['groups']:
        for model in group.get('models', []):
            if model['status'] != 'PARTIAL_CONDITIONAL_MODEL':
                continue
            if not model['ba_converged'] or model['roundtrip_differences']:
                raise ValueError('local model lacks convergence or roundtrip')
            folder = local/('group_%02d'%group['group_id'])/('model_%d'%model['model_id'])
            bind(folder/'POINT_LEDGER.json',refs)
            ledger=io.loads((folder/'POINT_LEDGER.json').read_text(encoding='utf-8-sig'))
            if not isinstance(ledger,list):raise ValueError('point ledger must be a list')
            for name in ('cameras.bin','images.bin','points3D.bin','rigs.bin','frames.bin'):
                if (folder/name).is_file(): bind(folder/name, refs)
            bind(folder/'supported.ply', refs, model['ply_readback']['binding'])
            cameras = camera_table(ledger)
            if set(cameras) != set(model['registered_input_indices']):
                raise ValueError('camera scope differs from registered images')
            for index in cameras:
                if by_index[index]['original_mask_status'] != 'OK':
                    raise ValueError('registered camera without committed object mask')
                frame=by_index[index];old=old_frames[index]
                if cameras[index]['image_id']!=old['database_image_id']:
                    raise ValueError('input index/native image identity differs')
                if (old['frame_id']!=frame['frame_id'] or old['canonical_image_sha256']!=frame['sha256']
                        or old_masks[index]['sha256']!=frame['mask_binding']['sha256']):
                    raise ValueError('local/current canonical image or mask identity differs')
            native=pc.Reconstruction(folder)
            if set(native.reg_image_ids())!={c['image_id'] for c in cameras.values()}:
                raise ValueError('native camera scope differs')
            for c in cameras.values():
                image=native.image(c['image_id']);camera=native.cameras[image.camera_id];pose=image.cam_from_world()
                if image.name!=old_frames[c['input_index']]['staged_image_name']:
                    raise ValueError('native image name differs from input binding')
                if (camera.model_name!='SIMPLE_RADIAL' or not np.allclose(c['params'],camera.params,rtol=0,atol=1e-12)
                    or not np.allclose(c['R'],pose.rotation.matrix(),rtol=0,atol=1e-12)
                    or not np.allclose(c['t'],pose.translation,rtol=0,atol=1e-12)):
                    raise ValueError('ledger/native camera differs')
            accepted=[p for p in ledger if p['accepted_conditionally']]
            saved=read_ply(folder/'supported.ply')
            if saved['vertices']!=[p['xyz'] for p in accepted] or saved['colors']!=[p['color'] for p in accepted]:
                raise ValueError('accepted ledger differs from bound PLY')
            for p in accepted:
                point=native.points3D[p['point_id']]
                if (not np.array_equal(point.xyz,p['xyz']) or list(point.color)!=p['color']
                    or assess_point(p['xyz'],p['observations'])['status']!='CONDITIONAL_POINT_SUPPORTED'):
                    raise ValueError('old point fails native or strict readback')
                if {(int(e.image_id),int(e.point2D_idx)) for e in point.track.elements}!={(o['image_id'],o['feature_row']) for o in p['observations']}:
                    raise ValueError('old native track identity differs')
                for o in p['observations']:
                    if not np.allclose(native.image(o['image_id']).points2D[o['feature_row']].xy,o['raw_pixels'],atol=1e-12,rtol=0):
                        raise ValueError('old native pixel differs')
            models.append(dict(group_id=group['group_id'],model_id=model['model_id'],
                cameras=cameras, ledger=ledger, folder=str(folder)))
    if not 1 <= len(models) <= PROFILE['max_groups']:
        raise ValueError('bounded nonempty local models required')
    return models


def verified_object_observation(observation, masks):
    """Unknown flow locations never provide 3D or face support."""
    if observation.get('label')!='foreground' or observation.get('target_verified') is not True:
        return False
    mask=masks[observation['frame_index']]
    if mask is None:return False
    xy=np.asarray(observation['xy'],dtype=float)
    if xy.shape!=(2,) or not np.isfinite(xy).all():return False
    x,y=np.floor(xy).astype(int)
    return 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]==255


def observation_segments(track, incremental=False):
    """Candidate-flow continuity does not establish identity across unknown gaps."""
    if not incremental:return [(0, track['observations'])]
    groups={};previous=-1
    for observation in track['observations']:
        segment=observation.get('identity_segment')
        if type(segment) is not int or segment<0 or segment<previous:
            raise ValueError('incremental observations require monotone nonnegative identity_segment')
        previous=segment
        groups.setdefault(segment,[]).append(observation)
    return sorted(groups.items())


def independent_segment_points(points, track_id, segment):
    # Coincident image positions cannot resolve unconfirmed cross-gap identity.
    return [p for p in points if not (p.get('candidate_track_id')==track_id
            and p.get('identity_segment')!=segment)]


def run(pipeline, local, output, selection_scope='object', tracking_profile=None, sfm_run=None):
    from kozlov_lab.stereo.coverage_tracks import track_coverage, profile_for
    from .coverage_triangulation import triangulate_candidate
    if tracking_profile not in (None, 'whole-video-incremental-v1'):raise ValueError('unknown tracking profile')
    if tracking_profile and selection_scope!='object':raise ValueError('incremental profile requires object selection')
    if sfm_run and not tracking_profile:raise ValueError('SfM override requires explicit incremental profile')
    track_profile = profile_for(selection_scope, profile_id=tracking_profile) if tracking_profile else profile_for(selection_scope)
    profile = dict(PROFILE,selection_scope=selection_scope,tracking_profile=tracking_profile)
    if tracking_profile:profile.update(id=tracking_profile,max_points_per_group=4096,max_surface_points=4096)
    pipeline, local, output = map(lambda p: Path(p).resolve(), (pipeline, local, output))
    if output.exists(): raise ValueError('fresh output required')
    for source in (pipeline, local):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('output must not overlap sources')
    refs = []
    discovery = read_bound(pipeline/'pipeline.json', refs)
    if tracking_profile:
        frames, grays, masks, meta = load_inputs(pipeline, discovery, refs, max_frame_profile=tracking_profile, sfm_run=sfm_run)
    else:
        frames, grays, masks, meta = load_inputs(pipeline, discovery, refs)
    cv = load_cv(refs)
    if cv is None: raise ValueError('existing OpenCV runtime unavailable')
    cv.setNumThreads(2); cv.setRNGSeed(0)
    models = load_local_models(local, frames, refs, str(Path(sfm_run).resolve(strict=True)) if sfm_run else discovery['stages']['sfm']['run_json'])
    # This receipt lands before tracking, triangulation or output generation.
    output.mkdir(parents=True)
    save(output/'INPUTS.json', dict(profile=profile, track_profile=track_profile,
        source_bindings=refs, video=meta, frames=frames))
    tracking = (track_coverage(grays, masks, cv, selection_scope=selection_scope, profile_id=tracking_profile)
                if tracking_profile else track_coverage(grays, masks, cv, selection_scope=selection_scope))
    save(output/'TRACKS.json', tracking)
    by_index = {f['input_index']: i for i, f in enumerate(frames)}
    object_points = {i: [] for i in range(len(frames))}
    result = dict(schema='coverage-expansion-v2', status='RUNNING', profile=profile,
        selection_scope=selection_scope, fullframe_candidates=selection_scope=='whole-frame',
        tracker_profile=track_profile, source_sfm_override=meta.get('source_sfm_override'), groups=[],
        source_bindings=refs, geometry_accepted=False, quality_status='NOT_EVALUATED',
        calibration_accepted=False, scale_status='unknown', components_merged=False,
        new_pose_estimation=False, new_segmentation=False, background_in_object_model=False)
    scenes = []
    for model in models:
        cameras = model['cameras']
        supported = [dict(p, source='saved_sift', tier='STRICT_SUPPORTED')
                     for p in model['ledger'] if p['accepted_conditionally']]
        original_count = len(supported)
        if original_count>profile['max_points_per_group']:raise ValueError('saved points exceed explicit profile budget; no truncation')
        ledger, provisional = [], []
        native_masks = {c['image_id']: masks[by_index[index]] != 0 for index, c in cameras.items()}
        next_id = max((p['point_id'] for p in model['ledger']), default=0)+1
        prioritized=sorted(tracking['tracks'],key=lambda t:(-sum(
            frames[o['frame_index']]['input_index'] in cameras and o['label']=='foreground'
            for o in t['observations']),t['track_id']))
        for track in prioritized:
            for segment, segment_observations in observation_segments(track, bool(tracking_profile)):
                selected = [o for o in segment_observations
                            if frames[o['frame_index']]['input_index'] in cameras and o['label']=='foreground'
                            and (not tracking_profile or verified_object_observation(o,masks))]
                if len(selected) < 2: continue
                start, end = selected[0]['frame_index'], selected[-1]['frame_index']
                observations = []
                for o in selected:
                    index = frames[o['frame_index']]['input_index']
                    observations.append(dict(cameras[index], feature_row=track['track_id'],
                        raw_pixels=o['xy'], foreground=True,
                        candidate_track_id=track['track_id'],identity_segment=segment,
                        feature_identity_domain='coverage_track_id_not_database_row'))
                if any(o['label']=='background' and start<=o['frame_index']<=end for o in segment_observations):
                    verdict = dict(xyz=None, tier='REJECTED', reasons=['KNOWN_BACKGROUND_WITHIN_OBJECT_TRACK'])
                else:
                    verdict = triangulate_candidate(observations)
                entry = dict(track_id=track['track_id'], candidate_track_id=track['track_id'], identity_segment=segment,
                             cross_segment_identity='UNCONFIRMED' if tracking_profile else 'LEGACY_SINGLE_SEGMENT',
                             observations=observations, decision=verdict,
                             source='temporal_object_mask' if selection_scope=='object' else 'temporal_fullframe', retained=False, duplicate_of=None,
                             unknown_bridge_steps=sum(o['label']=='unknown' and start<=o['frame_index']<=end for o in segment_observations))
                ledger.append(entry)
                if verdict['tier'] not in ('STRICT_SUPPORTED','RELAXED_SUPPORTED','TWO_VIEW_PROVISIONAL'):
                    continue
                is_provisional = verdict['tier']=='TWO_VIEW_PROVISIONAL'
                dedup_points=independent_segment_points(supported+provisional,track['track_id'],segment) if tracking_profile else supported+provisional
                duplicate = duplicate_observations(observations, dedup_points)
                if duplicate is not None:
                    entry['duplicate_of']=duplicate; entry['duplicate_relation']='GEOMETRIC_OVERLAP_NOT_IDENTITY'; continue
                if len(supported)+len(provisional)>=profile['max_points_per_group']:
                    raise ValueError('group point budget exceeded, no silent truncation')
                first = selected[0]; frame = frames[first['frame_index']]
                x,y = np.floor(first['xy']).astype(int)
                with Image.open(frame['path']) as image: color=list(image.convert('RGB').getpixel((int(x),int(y))))
                point = dict(point_id=next_id, xyz=verdict['xyz'], color=color,
                    observations=observations, tier=verdict['tier'],
                    source='temporal_object_mask' if selection_scope=='object' else 'temporal_fullframe',
                    track_id=track['track_id'],candidate_track_id=track['track_id'],identity_segment=segment,
                    cross_segment_identity='UNCONFIRMED' if tracking_profile else 'LEGACY_SINGLE_SEGMENT')
                next_id+=1; entry['retained']=True; entry['point_id']=point['point_id']
                (provisional if is_provisional else supported).append(point)
        folder=output/('group_%02d_model_%d'%(model['group_id'],model['model_id'])); folder.mkdir()
        save(folder/'CANDIDATE_LEDGER.json', ledger)
        save(folder/'SUPPORTED_POINTS.json', supported)
        save(folder/'TWO_VIEW_PROVISIONAL.json', provisional)
        faces=[]
        if len(supported)<=profile['max_surface_points']:
            generated, fs, diagnostic=build_faces(supported)
            faces, kept=supported_mask_faces(supported,dict(faces=generated,face_support=fs),native_masks)
            save(folder/'SURFACE_SUPPORT.json',dict(faces=faces,face_support=kept,diagnostic=diagnostic,
                interpolated=True,hole_filling=False,relaxed_points_allowed=True))
        else:
            if tracking_profile:raise ValueError('incremental surface point budget exceeded; points not silently dropped')
            save(folder/'SURFACE_SUPPORT.json',dict(faces=[],reason='SURFACE_POINT_BUDGET_POINTS_PRESERVED'))
        counts=Counter(p['tier'] for p in supported[original_count:])
        summary=dict(group_id=model['group_id'],input_indices=sorted(cameras),
            original_points=original_count,added_strict=counts['STRICT_SUPPORTED'],
            added_relaxed=counts['RELAXED_SUPPORTED'],supported_points=len(supported),
            two_view_provisional=len(provisional),faces=len(faces),
            considered_tracks=len(ledger),rejected_tracks=sum(e['decision']['tier']=='REJECTED' for e in ledger),
            deduplicated_tracks=sum(e['duplicate_of'] is not None for e in ledger))
        result['groups'].append(summary)
        for name, points, surface in (('supported',supported,faces),('two_view_provisional',provisional,[])):
            if not points: continue
            target=folder/(name+'.ply'); write_ply(target,points,surface)
            from integrations.viewer.ply_data import read_ply
            check=read_ply(target)
            if check['vertices'] != [p['xyz'] for p in points] or check['faces']!=surface or check['colors']!=[p['color'] for p in points]:
                raise ValueError('PLY readback differs')
            scene=dict(ply=str(target),sha256=_binding(target)['sha256'],
                title='组%s %s'%(model['group_id'],'支持点与局部面' if name=='supported' else '双视图暂定点'),
                data_kind='real_photos',declared_unit=None,
                note='复用未标定局部相机。宽松点为4px/6px/1度内部检查；双视图暂定点不建面。非完整外形或尺寸验收。')
            scenes.append(scene)
        for point in supported+provisional:
            for obs in point['observations']:
                object_points[by_index[obs['input_index']]].append(dict(xy=obs['raw_pixels'],tier=point['tier'],source=point['source']))
    # The reusable viewer has a hard limit of eight scenes; refuse rather than drop scenes.
    if len(scenes)>8: raise ValueError('viewer scene budget exceeded')
    save(output/'SCENES.json',dict(schema='ply-viewer-scenes-v1',scenes=scenes))
    from integrations.viewer.export import export_html
    save(output/'VIEWER_RESULT.json',export_html(output/'SCENES.json',output/'viewer.html'))
    coverage=render(output,frames,masks,tracking,object_points,result)
    save(output/'COVERAGE.json',coverage)
    drift=[r['path'] for r in refs if _binding(r['path'])!=r]
    result.update(status='PARTIAL_CONDITIONAL_EXPANSION' if not drift else 'FAILED_SOURCE_DRIFT',
        source_unchanged=not drift,source_drift=drift,counts=dict(tracks=len(tracking['tracks']),
            observations=sum(len(t['observations']) for t in tracking['tracks']),
            frames_with_object_mask=sum(m is not None and bool(m.any()) for m in masks),
            frames_without_object_mask=sum(m is None or not bool(m.any()) for m in masks),
            foreground_observations=sum(o['label']=='foreground' for t in tracking['tracks'] for o in t['observations']),
            background_observations=sum(o['label']=='background' for t in tracking['tracks'] for o in t['observations']),
            unknown_observations=sum(o['label']=='unknown' for t in tracking['tracks'] for o in t['observations']),
            weak_flow_observations=sum(not o['strong'] and o['fb_error_px'] is not None for t in tracking['tracks'] for o in t['observations'])),
        frame_count=len(frames),scenes=scenes)
    save(output/'RESULT.json',result)
    return result


def render(output,frames,masks,tracking,object_points,result):
    from kozlov_lab.stereo.coverage_tracks import grid_edges
    object_only=tracking['profile']['selection_scope']=='object'
    overlays=output/'overlays';overlays.mkdir()
    observations={i:[] for i in range(len(frames))}
    for track in tracking['tracks']:
        for obs in track['observations']:observations[obs['frame_index']].append(obs)
    coverage=[]
    for i,frame in enumerate(frames):
        with Image.open(frame['path']) as source: image=source.convert('RGB')
        w,h=image.size
        stats=tracking['coverage'][i];bounds=stats['grid_bounds']
        if bounds is not None:
            grid_image=image.copy();grid_draw=ImageDraw.Draw(grid_image)
            xe,ye=grid_edges(bounds);x0,y0,x1,y1=bounds
            for x in xe[1:-1]:grid_draw.line((int(x),y0,int(x),y1-1),fill=(80,140,125))
            for y in ye[1:-1]:grid_draw.line((x0,int(y),x1-1,int(y)),fill=(80,140,125))
            image=Image.composite(grid_image,image,Image.fromarray(masks[i])) if object_only else grid_image
        draw=ImageDraw.Draw(image)
        for obs in observations[i]:
            x,y=obs['xy'];color={'foreground':(50,235,155),'background':(65,150,255),'unknown':(255,200,65)}[obs['label']]
            draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        image.save(overlays/('%06d_candidates.png'%frame['input_index']))
        with Image.open(frame['path']) as source: image=source.convert('RGB')
        draw=ImageDraw.Draw(image)
        for point in object_points[i]:
            x,y=point['xy'];color={'STRICT_SUPPORTED':(45,245,130),'RELAXED_SUPPORTED':(255,165,40),'TWO_VIEW_PROVISIONAL':(220,100,255)}[point['tier']]
            draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        image.save(overlays/('%06d_3d.png'%frame['input_index']))
        known=masks[i] is not None
        entry=dict(input_index=frame['input_index'],mask_status=frame['original_mask_status'],
            selection_scope=tracking['profile']['selection_scope'],selection_status=stats['selection_status'],
            seed_grid_bounds=bounds,seed_grid_domain=stats['grid_domain'],
            seed_grid_counts=stats['grid_counts'],
            candidates=len(observations[i]),labels=dict(Counter(o['label'] for o in observations[i])),
            object_candidate_grid=occupancy([o['xy'] for o in observations[i]],masks[i]!=0,bounds=bounds) if known else None,
            old_3d_grid=occupancy([p['xy'] for p in object_points[i] if p['source']=='saved_sift'],masks[i]!=0,bounds=bounds) if known else None,
            supported_3d_grid=occupancy([p['xy'] for p in object_points[i] if p['tier']!='TWO_VIEW_PROVISIONAL'],masks[i]!=0,bounds=bounds) if known else None,
            comparison_fixed_image_grid=dict(domain='whole_image_8x12',
                object_candidates=occupancy([o['xy'] for o in observations[i]],masks[i]!=0),
                supported_3d=occupancy([p['xy'] for p in object_points[i] if p['tier']!='TWO_VIEW_PROVISIONAL'],masks[i]!=0)) if known else None,
            fullframe_grid=occupancy([o['xy'] for o in observations[i]],np.ones((h,w),dtype=bool)))
        entry.update({key:stats.get(key,0) for key in ('tracked','seeded','recovered','suspended')})
        coverage.append(entry)
    options=''.join('<option value="%06d">帧 %s / %s</option>'%(f['input_index'],f['input_index'],html.escape(f['original_mask_status'])) for f in frames)
    table=''.join('<tr>'+''.join('<td>%s</td>'%g[k] for k in ('group_id','original_points','added_strict','added_relaxed','supported_points','two_view_provisional','faces'))+'</tr>' for g in result['groups'])
    title='物体内选点与分级匹配' if object_only else '全图候选诊断（显式选用）'
    description=('先读取已有 SAM 物体掩膜，再在物体外接范围内划分 8×12 格补充真实角点。'
        '候选点中心必须位于掩膜内，孔洞与背景不选点。跟踪点离开物体即结束；未识别帧不选点，之后重新播种使用新 ID。'
        if object_only else '显式全图诊断模式：背景与未知掩膜区域也选点展示，但不接入物体三维。')
    if result['profile'].get('tracking_profile')=='whole-video-incremental-v1':
        description=('每帧在真实已提交掩膜内补充候选点；未知帧不播种、不确认目标。未知区间只保持候选光流连续，'
            '重获前景时开启独立 identity_segment。跨段物理身份未确认，不把未知区间两端合并三角化；'
            '每段分别使用真实前景观测验证三维，未知观测不支撑点或面。')
    local_count=len(result['groups'])
    incremental=result['profile'].get('tracking_profile')=='whole-video-incremental-v1'
    candidate_note=('网格仅在已确认物体内；黄点为未知帧暂存候选光流，不确认物体身份、不参与三维。' if incremental
        else '网格仅绘制在物体内，无法识别时不显示网格与候选点。' if object_only else '蓝：已知背景；黄：掩膜未知。')
    default_frame=next((f['input_index'] for i,f in enumerate(frames) if masks[i] is not None and bool(np.any(masks[i]))),frames[0]['input_index'])
    document='''<!doctype html><meta charset="utf-8"><title>'''+title+'''</title>
<style>body{background:#102029;color:#edf6fa;font:16px system-ui;max-width:1150px;margin:28px auto;padding:15px}a{color:#82caff}p{line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #526779;padding:10px}.views{display:flex;gap:15px}.views img{width:100%;max-height:850px;object-fit:contain}.views div{width:50%}select{font:inherit;padding:8px;background:#243e4a;color:white}</style>
<h1>'''+title+'''</h1>
<p>'''+description+''' 初配往返误差仍允许到 3px；LK 误差上限仍为 40，弱候选继续保留。三维验证门槛没有改变。</p>
<p><a href="viewer.html">查看局部三维</a> · <a href="../blender/model.blend">Blender 文件</a> · <a href="RESULT.json">结果记录</a> · <a href="COVERAGE.json">覆盖计数</a></p>
<table><tr><th>局部组</th><th>原点</th><th>新增严格</th><th>新增宽松</th><th>支持点合计</th><th>双视图暂定</th><th>局部面</th></tr>'''+table+'''</table>
<p>相机固定为已有 '+str(local_count)+' 个局部的估计；没有新增相机配准。新增点仍是条件几何结果，严格/宽松内部一致性都不是人工真值。'''+str(local_count)+''' 个局部坐标系未拼接，缺测处保留。双视图暂定点独立展示、不建面。覆盖格数不是物体表面积或完成百分比。</p>
<label>画面 <select id="frame">'''+options+'''</select></label>
<p id="frame-status"></p>
<div class="views"><div><h3>候选点</h3><p>绿：已识别物体内。'''+candidate_note+'''</p><img id="candidate"></div><div><h3>已有相机支持的三维点</h3><p>绿：严格支持；橙：宽松支持；紫：双视图暂定。</p><img id="three"></div></div>
<p>算法依据：书籍《Сборка макета》41–42页同一物理点对应与不同投影方向；此处仍为工程透视重建，未冒充论文平行投影构造。实现参考 <a href="https://docs.opencv.org/4.13.0/d4/d8c/tutorial_py_shi_tomasi.html">OpenCV角点</a> 与 <a href="https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html">光流</a>。</p>
<script>const data='''+json.dumps(coverage,ensure_ascii=False,allow_nan=False)+''';const f=document.getElementById('frame');function update(){document.getElementById('candidate').src='overlays/'+f.value+'_candidates.png';document.getElementById('three').src='overlays/'+f.value+'_3d.png';const r=data.find(r=>r.input_index===Number(f.value));const g=r.object_candidate_grid;document.getElementById('frame-status').textContent='原分割状态：'+r.mask_status+' · 候选点：'+r.candidates+' · 背景点：'+(r.labels.background||0)+(g?' · 物体格覆盖：'+g.occupied_cells+'/'+g.available_cells:' · 掩膜不可用，本帧不确认物体位置。');}f.value='''+json.dumps('%06d'%default_frame)+''';f.onchange=update;update();</script>'''
    (output/'index.html').write_text(document,encoding='utf-8')
    return coverage
