"""Fresh native continuation, matched SIFT-only revalidation, and readback report."""
from pathlib import Path
from collections import Counter, defaultdict
import copy
import json

from integrations.demo_pipeline.pipeline import ROOT, LIGHT_REL, _invoke, entry_code, read, save, sha, utc


def run_control(run):
    """Reverify the same SIFT raw data once; never replay extraction/matching."""
    run = Path(run).resolve(strict=True)
    plan = read(run / 'PLAN.json')
    target = run / 'sift_control'
    control_path = run / 'SIFT_CONTROL_TRACKS.json'
    if target.exists() or control_path.exists() or (run / 'logs' / 'sift_control').exists():
        raise FileExistsError('control already attempted')
    control = copy.deepcopy(read(run / 'tracking' / 'REGION_TRACKS.json'))
    for row in control['frames']:
        row['observations'] = []
    control.update(tracks=[], timeline_stats=[], metrics={'observations': 0},
                   control_kind='same_native_reverify_without_any_region_correspondences')
    save(control_path, control)
    argv = [str(ROOT / LIGHT_REL), '-B', '-X', 'utf8', '-m', 'integrations.video_masks.region_geometry',
            '--sfm-run', plan['baseline_sfm'], '--tracks', str(control_path), '--output', str(target)]
    entry = _invoke(ROOT, run, 'sift_control', argv, seconds=600, memory=4 << 30)
    if entry_code(entry) != 0 or not (target / 'RESULT.json').is_file():
        raise RuntimeError('SIFT-only control failed; no automatic retry')
    return read(target / 'RESULT.json')


def continue_native(run, geometry_name='geometry02'):
    """One independent native attempt plus one new SIFT-only control, no retuning."""
    run = Path(run).resolve(strict=True)
    plan = read(run / 'PLAN.json')
    receipt = run / ('CONTINUATION_' + geometry_name + '.json')
    if receipt.exists() or (run / geometry_name).exists():
        raise FileExistsError('continuation already attempted')
    state = dict(started_utc=utc(), geometry_name=geometry_name, stages={},
                 old_failure_preserved=True, code_bindings={str(ROOT / p): sha(ROOT / p) for p in (
                     'integrations/video_masks/region_geometry.py', 'integrations/video_masks/region_comparison.py')})
    save(receipt, state)
    tracks = read(run / 'tracking' / 'REGION_TRACKS.json')
    control = copy.deepcopy(tracks)
    for f in control['frames']:
        f['observations'] = []
    control.update(tracks=[], timeline_stats=[], metrics=dict(observations=0, endpoint_count=len(control['frames'])),
                   control_kind='same_native_reverify_without_any_region_correspondences',
                   source_region_tracks=dict(path=str(run / 'tracking' / 'REGION_TRACKS.json'),
                                             sha256=sha(run / 'tracking' / 'REGION_TRACKS.json')))
    control_path = run / ('CONTROL_' + geometry_name + '.json')
    save(control_path, control)
    for label, trackfile in [(geometry_name, run / 'tracking' / 'REGION_TRACKS.json'),
                             ('sift_control', control_path)]:
        argv = [str(ROOT / LIGHT_REL), '-B', '-X', 'utf8', '-m', 'integrations.video_masks.region_geometry',
                '--sfm-run', plan['baseline_sfm'], '--tracks', str(trackfile), '--output', str(run / label)]
        print('STAGE_BEGIN', label, flush=True)
        entry = _invoke(ROOT, run, label, argv, seconds=600, memory=4 << 30)
        code = entry_code(entry)
        state['stages'][label] = dict(exit_code=code, reclaimed=entry.get('reclaim_confirmed'),
                                     result=str(run / label / 'RESULT.json'), finished_utc=utc())
        save(receipt, state)
        print('STAGE_END', label, code, flush=True)
        if code != 0:
            state.update(status='STOPPED', error=label + ' failed; no automatic retry')
            save(receipt, state)
            return state
    state.update(status='NATIVE_EXPERIMENTS_COMPLETE_NOT_ACCEPTED', finished_utc=utc())
    save(receipt, state)
    return state


def analyse(run, geometry_name='geometry02'):
    import numpy as np
    import pycolmap as pc
    from integrations.viewer.export import export_html
    from integrations.demo_pipeline.camera_domain_gate import audit_camera_domain
    run = Path(run).resolve(strict=True)
    plan = read(run / 'PLAN.json')
    baseline = Path(plan['baseline_sfm'])
    original = read(baseline / 'run.json')
    fmap = [json.loads(s) for s in (baseline / 'frames_map.jsonl').read_text(encoding='utf8').splitlines() if s]
    index_by_iid = {r['database_image_id']: r['input_index'] for r in fmap}
    tracks = read(run / 'tracking' / 'REGION_TRACKS.json')
    regional = read(run / geometry_name / 'RESULT.json')
    control = read(run / 'sift_control' / 'RESULT.json')
    native_plan = read(run / geometry_name / 'PLAN.json')
    native_frames = {r['image_id']: r for r in native_plan['frames']}
    row_track = {iid: {int(row): int(tid) for tid, row in f['row_by_track'].items()} for iid, f in native_frames.items()}
    region_annotation_scope = dict(status='NOT_EVALUATED',
        reason='No bound object-part annotation is evaluated; frame indices do not identify object parts.')
    labels = [('historical_sift', [(baseline / m['binary_path'], baseline / m.get('ply_path', 'models/'+str(m['model_id'])+'/points.ply'), m['model_id']) for m in original['models']]),
              ('reverified_sift', [(Path(m['binary_path']), Path(m['ply_path']), m['model_id']) for m in control['models']]),
              ('sift_plus_regions', [(Path(m['binary_path']), Path(m['ply_path']), m['model_id']) for m in regional['models']])]
    all_models = []
    per_frame = defaultdict(list)
    for label, items in labels:
        for binary, ply, mid in items:
            rec = pc.Reconstruction(binary)
            indices = sorted(index_by_iid[i] for i in rec.reg_image_ids())
            lineage = defaultdict(set)
            point_origins = Counter()
            reproj, positive, total = [], 0, 0
            points_array = np.asarray([p.xyz for p in rec.points3D.values()], float)
            for pid, p in rec.points3D.items():
                kinds = set()
                for e in p.track.elements:
                    im = rec.images[e.image_id]
                    q = im.points2D[e.point2D_idx]
                    pose = im.cam_from_world()
                    local = pose.rotation.matrix() @ p.xyz + pose.translation
                    total += 1; positive += bool(local[2] > 0)
                    px = rec.cameras[im.camera_id].img_from_cam(local)
                    if px is not None:
                        reproj.append(float(np.linalg.norm(px-q.xy)))
                    if label == 'sift_plus_regions':
                        tid = row_track.get(e.image_id, {}).get(e.point2D_idx)
                        kinds.add('region' if tid is not None else 'sift')
                        if tid is not None:
                            lineage[tid].add(int(pid))
                if label == 'sift_plus_regions':
                    point_origins['+'.join(sorted(kinds))] += 1
            centered = points_array - np.median(points_array, axis=0) if len(points_array) else points_array
            singular = np.linalg.svd(centered, compute_uv=False) if len(points_array) >= 3 else np.array([])
            item = dict(label=label, model_id=int(mid), registered_views=len(indices), registered_indices=indices,
                        previously_labelled_back_views=None, region_annotation_status='NOT_EVALUATED',
                        points=int(rec.num_points3D()),
                        point_origins=dict(point_origins), region_ids_split_into_multiple_3d_points=sum(len(v)>1 for v in lineage.values()),
                        triangulated_region_ids=len(lineage), positive_depth_observations=positive, total_observations=total,
                        recomputed_reprojection_median=float(np.median(reproj)) if reproj else None,
                        recomputed_reprojection_p95=float(np.quantile(reproj,.95)) if reproj else None,
                        centered_xyz_singular_values=singular.tolist(), binary_path=str(binary),
                        physical_geometry_verified=False)
            domain = audit_camera_domain(binary)
            item['camera_domain_status'] = domain['status']
            item['camera_domain'] = domain
            if label == 'sift_plus_regions' and domain['status'] == 'CAMERA_DOMAIN_VALID':
                for iid in rec.reg_image_ids():
                    im = rec.images[iid]
                    for q in im.points2D:
                        if q.has_point3D():
                            per_frame[index_by_iid[iid]].append((q.xy-.5).tolist())
            # PLY existence is read back, never substituted by another run's file.
            if not ply.is_file():
                choices = [binary.parent/'points.ply', binary.parent/'model.ply']
                ply = next((p for p in choices if p.is_file()), ply)
            if ply.is_file():
                target = run / 'viewers' / f'{label}_{mid}'
                target.mkdir(parents=True, exist_ok=False)
                spec = target/'scenes.json'
                save(spec, dict(schema='ply-viewer-scenes-v1', scenes=[dict(ply=str(ply),sha256=sha(ply),
                    title=f'{label} / component {mid}',data_kind='real_photos',declared_unit=None,
                    note='Independent unaligned component. Sparse candidate reconstruction; not quality accepted.',evidence=[])]))
                export_html(spec,target/'viewer.html')
                item['viewer'] = str(target/'viewer.html')
                item['ply_path'] = str(ply)
            all_models.append(item)
    metrics = {}
    for label, _ in labels:
        group = [m for m in all_models if m['label'] == label]
        registered = set(i for m in group for i in m['registered_indices'])
        largest = max(group, key=lambda m:(m['registered_views'], m['points']), default=None)
        domain_views=set(i for m in group if m['camera_domain_status']=='CAMERA_DOMAIN_VALID' for i in m['registered_indices'])
        metrics[label] = dict(components=len(group),unique_registered_views=len(registered),
                             largest_component_views=largest['registered_views'] if largest else 0,
                             largest_component_points=largest['points'] if largest else 0,
                             labelled_back_registered=None, region_annotation_status='NOT_EVALUATED',
                             camera_domain_valid_unique_views=len(domain_views),
                             camera_domain_rejected_components=[m['model_id'] for m in group if m['camera_domain_status']!='CAMERA_DOMAIN_VALID'])
    metrics['region_correspondences'] = dict(raw=sum(p['region_raw'] for p in regional['pairs']),
        verified=sum(p['region_verified'] for p in regional['pairs']), counts_are_pair_rows_not_unique_points=True)
    comparison = dict(schema='glove-region-comparison-v1', globalmetrics=metrics, models=all_models,
        source_geometry=geometry_name,
        object_scope='object-agnostic mask-bound region candidates; legacy glove schema retained for compatibility',
        region_annotation_scope=region_annotation_scope,
        frames=[dict(input_index=r['input_index'],native_xy=per_frame[r['input_index']],
                     registered=r['input_index'] in regional['unique_registered_input_indices']) for r in tracks['frames']],
        notes=['RANSAC random_seed=-1 inherited in all runs; single runs are not a causal/statistical efficacy proof.',
               'All components use independent coordinates; do not merge point counts into a full object.',
               'Adjacent-patch similarity and geometric inliers are not physical identity truth.',
               'Object deformation and silhouette/mask errors remain possible.',
               'Object-part coverage is NOT_EVALUATED without bound region annotations.'],
        visual_back_group=None, quality_status='NOT_EVALUATED', scale_status='unknown',
        source_bindings={str(p):sha(p) for p in [run/geometry_name/'RESULT.json',run/'sift_control'/'RESULT.json',baseline/'run.json']})
    save(run/'COMPARISON.json', comparison)
    return comparison
