"""Frozen multi-pair engineering ablation; never claims multiview fusion.

All methods reuse existing COLMAP calibration/correspondence. Negative controls
are deliberate identity mismatches and are never exported as model points.
"""
from collections import Counter, defaultdict
from itertools import combinations
import html
from pathlib import Path
import time
import numpy as np

from . import parallel_photo_audit as native
from . import approximate_photo_bridge as bridge
from kozlov_lab.stereo.baseline import radial_diagnostics as radial
from kozlov_lab.stereo.baseline import rectified_parallel as rect
from kozlov_lab.stereo.baseline.rectified_ablation import evaluate_six

MAX_PAIRS=3
MAX_GROUPS=40
WORLD_OK='WORLD_CONVERTED_NOT_ACCEPTED'


def select_pairs(images,points,reference_pair,max_pairs=MAX_PAIRS):
    if type(max_pairs) is not int or not 1 <= max_pairs <= MAX_PAIRS:
        raise ValueError('pair limit must be 1..3')
    reference=tuple(reference_pair)
    if (len(reference)!=2 or any(type(i) is not int for i in reference)
            or tuple(sorted(reference))!=reference or reference[0]==reference[1]
            or not set(reference).issubset(images)):
        raise ValueError('two distinct sorted registered reference images required')
    eligible=defaultdict(list);raw_counts=Counter()
    for pid,p in sorted(points.items()):
        for pair in combinations(sorted(p.get('image_ids',p['track'])),2):
            raw_counts[pair]+=1
        for pair in combinations(sorted(p['track']),2):
            eligible[pair].append(pid)
    all_pairs=list(combinations(sorted(images),2))
    all_pairs.sort(key=lambda pair:(-len(eligible[pair]),pair))
    if len(eligible[reference])<6:
        raise ValueError('reference needs six eligible tracks')
    selected=[dict(image_pair=list(reference),common_ids=eligible[reference],role='reference')]
    used=set(reference);decisions={reference:'SELECTED_REFERENCE'}
    for pair in all_pairs:
        if pair==reference:continue
        if len(eligible[pair])<6:reason='INSUFFICIENT_COMMON_TRACKS'
        elif set(pair)&used:reason='SHARES_IMAGE_WITH_SELECTED_PAIR'
        elif len(selected)>=max_pairs:reason='PAIR_BUDGET_NOT_SELECTED'
        else:
            selected.append(dict(image_pair=list(pair),common_ids=eligible[pair],role='additional'))
            used.update(pair);reason='SELECTED_ADDITIONAL'
        decisions[pair]=reason
    ranking=[dict(image_pair=list(pair),eligible_count=len(eligible[pair]),
        raw_shared_count=raw_counts[pair],ambiguous_excluded_count=raw_counts[pair]-len(eligible[pair]),
        decision=decisions[pair]) for pair in all_pairs]
    return dict(selected=selected,ranking=ranking,
        selection='reference then eligible-count descending / image IDs ascending; image-disjoint greedy',
        independent_datasets=False)


def select_groups(common,max_groups=MAX_GROUPS):
    if (type(common) is not list or any(type(i) is not int for i in common)
            or sorted(common)!=common or len(set(common))!=len(common)):
        raise ValueError('sorted unique integer track IDs required')
    if type(max_groups) is not int or not 1 <= max_groups <= MAX_GROUPS:
        raise ValueError('group limit must be 1..40')
    blocks=[dict(block_index=i,point_ids=common[i*6:(i+1)*6]) for i in range(len(common)//6)]
    if len(blocks)<=max_groups:indices=list(range(len(blocks)))
    elif max_groups==1:indices=[0]
    else:indices=[i*(len(blocks)-1)//(max_groups-1) for i in range(max_groups)]
    return dict(selected=[blocks[i] for i in indices],
        unselected_blocks=[block for block in blocks if block['block_index'] not in indices],
        tail_point_ids=common[(len(common)//6)*6:],
        formula='consecutive sorted six; if over cap floor(k*(block_count-1)/(cap-1)); cap1 takes first',
        unselected_status='NOT_EVALUATED')


def _load(prior_input):
    path=Path(prior_input).resolve(strict=True);frozen=native.read_json(path)
    if frozen.get('schema')!='photo-parallel-selected-observations-v1':
        raise ValueError('prior exact-photo INPUT required')
    refs=frozen['source_bindings'];names={};seen=set()
    for ref in refs:
        bridge.verify_binding(ref)
        p=Path(ref['path']).resolve(strict=True)
        if p in seen:raise ValueError('duplicate source path')
        seen.add(p)
        if p.name in ('run.json','images.bin','points3D.bin','cameras.bin','frames_map.jsonl'):
            if p.name in names:raise ValueError('duplicate native source name')
            names[p.name]=p
    if len(names)!=5:raise ValueError('complete native run bindings required')
    images,points=native.read_native(names['images.bin'],names['points3D.bin'])
    pair,common,_=native.choose_pair(images,points)
    six=frozen['point_ids']
    if (list(pair)!=frozen['image_pair'] or common[:6]!=six or len(common)!=frozen['common_tracks']
            or type(frozen.get('observations')) is not list or len(frozen['observations'])!=6):
        raise ValueError('prior selection/complete six ledger mismatch')
    for item,pid in zip(frozen['observations'],six):
        if item['point3D_id']!=pid or len(item['observations'])!=2:raise ValueError('prior track ledger mismatch')
        for old,image_id in zip(item['observations'],pair):
            index=points[pid]['track'][image_id]
            xy=images[image_id]['observations'][index][:2]
            if (old['image_id']!=image_id or old['point2D_index']!=index or old.get('xy')!=list(xy)
                    or any(type(v) not in (int,float) for v in old.get('xy',[]))
                    or old['binary64_hex']!=[v.hex() for v in xy]):
                raise ValueError('prior original observations changed')
    return path,frozen,names,images,points,pair


def _bind_photos(root,run,images,selected,names):
    rows=[native.loads(line) for line in names['frames_map.jsonl'].read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    models=[m for m in run['models'] if (root/m['binary_path']).resolve()==names['images.bin'].parent]
    if len(models)!=1 or not models[0]['usable'] or not models[0]['roundtrip_verified']:
        raise ValueError('one accepted native model required')
    model_id=str(models[0]['model_id']);frames=[]
    for image_id in sorted({i for pair in selected for i in pair['image_pair']}):
        matches=[row for row in rows if any(str(reg['model_id'])==model_id and reg['image_id']==image_id for reg in row['registrations'])]
        if len(matches)!=1:raise ValueError('one frame binding per image required')
        row=matches[0]
        regs=[reg for reg in row['registrations'] if str(reg['model_id'])==model_id and reg['image_id']==image_id]
        if (len(regs)!=1 or regs[0]['camera_id']!=images[image_id]['camera_id']
                or row['staged_image_name']!=images[image_id]['name'] or row['status']!='REGISTERED'
                or row['upstream_io_run_id']!=run['upstream_io_run_id']):
            raise ValueError('frame/image/camera/upstream mismatch')
        photo=native.under(root,row['staged_image_name']);ref=native.binding(photo)
        if ref['sha256']!=row['canonical_image_sha256']:raise ValueError('image byte identity mismatch')
        frames.append(dict(image_id=image_id,input_index=row['input_index'],frame_id=row['frame_id'],binding=ref))
    return frames


def _projection_metrics(method,raw,cameras,poses,stereo):
    if method.get('world_xyz') is None:return dict(status='NOT_EVALUATED',reason='no world coordinates')
    xyz=np.asarray(method['world_xyz']);views=[]
    try:
        for j,(camera,pose) in enumerate(zip(cameras,poses)):
            cp=xyz@pose['R'].T+pose['t']
            if not np.isfinite(cp).all() or np.any(cp[:,2]<=0):raise ValueError('original-camera nonpositive depth')
            h=cp@camera['K'].T;ideal=h[:,:2]/h[:,2,None]
            rectified=rect.apply_homography(ideal,stereo['homography'+str(j+1)])
            wanted=method['first_pixels' if j==0 else 'second_pixels']
            closed=float(np.max(np.abs(rectified-wanted)))
            if closed>1e-7:raise ValueError('calibrated roundtrip beyond1e-7 pixel')
            projected=radial.distort_simple_radial(ideal,camera['params'],max_ideal_radius=camera['radius'])
            residual=projected-raw[:,2*j:2*j+2]
            views.append(dict(raw_reprojection=projected,residual_vectors=residual,
                residual_norms=np.linalg.norm(residual,axis=1),calibrated_roundtrip_max_abs=closed))
        return dict(status='NUMERIC_ROUNDTRIP_VERIFIED',views=views,
                    acceptance_threshold=None,quality_status='NOT_EVALUATED')
    except (ValueError,ArithmeticError) as exc:
        return dict(status='PROJECTION_REFUSED',reason=str(exc),views=views)


def _case(raw,cameras,poses,stereo,labels,kind,right_ids):
    row=dict(case_kind=kind,left_source_track_ids=labels,right_source_track_ids=right_ids,raw_pixels=raw,
             status='NOT_EVALUATED',ablation=None,preprocessing=None)
    try:
        inverses=[radial.undistort_simple_radial(raw[:,j*2:j*2+2],camera['params'],max_ideal_radius=camera['radius'])
                  for j,camera in enumerate(cameras)]
        ideal=[inv['ideal_pixels'] for inv in inverses]
        views=[rect.apply_homography(ideal[j],stereo['homography'+str(j+1)]) for j in range(2)]
        difference=views[0][:,1]-views[1][:,1]
        row['preprocessing']=dict(ideal_pixels=ideal,rectified_pixels=views,
            vertical_difference_before_correction=difference,
            vertical_difference_stats_pixels=bridge.summary_stats(difference),
            common_y_change_pixels=np.column_stack((-difference/2,difference/2)),
            inverse_forward_max_pixels=[float(np.max(inv['forward_residual_pixels'])) for inv in inverses])
        row['ablation']=evaluate_six(*views,stereo,labels)
        for method in ('A','B','C'):
            item=row['ablation'][method]
            item['projection_metrics']=_projection_metrics(item,raw,cameras,poses,stereo)
        row['status']='EVALUATED_WITH_STAGE_STATUSES'
    except (ValueError,ArithmeticError,TypeError,KeyError) as exc:
        row.update(status='PREPROCESS_OR_KERNEL_REFUSED',reason=str(exc))
    return row


def _counts(rows):
    c=dict(requested=len(rows),evaluated=0,not_evaluated=0,A_world_ok=0,B_world_ok=0,
           C_world_ok=0,C_certificate_verified=0,C_constructed=0,C_original_projection_ok=0,
           B_C_comparable=0,B_C_bitwise_equal=0)
    statuses=Counter()
    for row in rows:
        a=row.get('ablation')
        if a is None:
            c['not_evaluated']+=1;statuses[row['status']]+=1;continue
        c['evaluated']+=1
        for m in ('A','B','C'):
            if a[m].get('status')==WORLD_OK:c[m+'_world_ok']+=1
        if a['C']['independent']['status']=='VERIFIED':c['C_certificate_verified']+=1
        if a['C'].get('exact_status')=='CONSTRUCTED':c['C_constructed']+=1
        if a['C'].get('projection_metrics',{}).get('status')=='NUMERIC_ROUNDTRIP_VERIFIED':c['C_original_projection_ok']+=1
        if a['B'].get('world_xyz') is not None and a['C'].get('world_xyz') is not None:
            c['B_C_comparable']+=1
            c['B_C_bitwise_equal']+=int(np.array_equal(a['B']['world_xyz'],a['C']['world_xyz']))
        statuses[a['C']['status']]+=1
    c['C_status_counts']=dict(statuses)
    return c


def _method_stats(rows,field):
    values=[]
    for row in rows:
        if row.get('ablation') is None:continue
        a=row['ablation']
        if field=='A_vs_B':methods=('A','B')
        else:methods=('B','C')
        if all(a[m].get('world_xyz') is not None for m in methods):
            values.extend(np.linalg.norm(a[methods[0]]['world_xyz']-a[methods[1]]['world_xyz'],axis=1))
    return bridge.summary_stats(values) if values else None


def _reprojection_stats(rows):
    values=[]
    for row in rows:
        if row.get('ablation') is None:continue
        metrics=row['ablation']['C'].get('projection_metrics',{})
        if metrics.get('status')=='NUMERIC_ROUNDTRIP_VERIFIED':
            for view in metrics['views']:values.extend(view['residual_norms'])
    return bridge.summary_stats(values) if values else None


def _ply(path,samples):
    xyz=np.array([row['world_xyz'] for row in samples])
    text=('ply\nformat ascii 1.0\ncomment CALIBRATED_PAIR_SAMPLES_NOT_FUSED_NOT_ACCEPTED\n'
          'comment repeated track IDs retained as separate samples; no faces; scale unknown\n'
          f'element vertex {len(samples)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
    text+=''.join(' '.join(format(float(v),'.17g') for v in p)+'\n' for p in xyz)
    with path.open('x',encoding='ascii') as f:f.write(text)
    loaded=np.array([[float(v) for v in line.split()] for line in path.read_text().split('end_header\n',1)[1].splitlines()])
    if not np.array_equal(loaded,xyz):raise ValueError('sample PLY readback mismatch')
    return native.binding(path)


def experiment(prior_input,output):
    old_path,frozen,names,images,points,reference=_load(prior_input)
    output=Path(output).resolve();root=names['run.json'].parent
    if output.exists():raise ValueError('new output required; no overwrite')
    if output.is_relative_to(root) or output.is_relative_to(old_path.parent):raise ValueError('immutable input run cannot contain output')
    selection=select_pairs(images,points,reference)
    cameras,poses=bridge.read_calibration(names['cameras.bin'],names['images.bin'])
    frames=_bind_photos(root,native.read_json(names['run.json']),images,selection['selected'],names)
    refs=list(frozen['source_bindings']);ref_paths={row['path'] for row in refs}
    for frame in frames:
        if frame['binding']['path'] not in ref_paths:
            refs.append(frame['binding']);ref_paths.add(frame['binding']['path'])
    for selected in selection['selected']:
        selected['group_selection']=select_groups(selected['common_ids'])
        selected['calibration']=[dict(image_id=i,pose=poses[i],camera=cameras[poses[i]['camera_id']]) for i in selected['image_pair']]
    plan=dict(schema='multi-pair-ablation-plan-v1',prior_input=native.binding(old_path),source_bindings=refs,
        pairs=selection['selected'],pair_ranking=selection['ranking'],frames=frames,
        max_pairs=MAX_PAIRS,max_groups_per_pair=MAX_GROUPS,grid_bits=18,normalized_scale=1,
        negative_control='for every frozen group, right original pixel rows shift left one; labels unchanged; full source ID ledger',
        negative_controls_are_model_candidates=False,construction_work=2000000,construction_seconds=10,
        verification_work=2000000,verification_seconds=10,whole_run_soft_seconds=90,outer_hard_seconds=120,
        calibrated_roundtrip_tolerance_pixels=1e-7,quality_acceptance_threshold=None,
        reuse_colmap_calibration_and_correspondence=True,independent_datasets=False,
        fusion_or_global_optimization=False,scale_status='unknown')
    output.mkdir(parents=True,exist_ok=False);bridge.write(output/'PLAN.json',plan)
    (output/'groups').mkdir();deadline=time.monotonic()+90
    all_rows={'positive':[],'negative_control':[]};pair_results=[];samples=[]
    for pair_index,selected in enumerate(selection['selected']):
        pair=selected['image_pair'];pc=[cameras[poses[i]['camera_id']] for i in pair];pp=[poses[i] for i in pair]
        stereo=None;pair_error=None
        try:
            stereo=rect.calibrated_rectification(pc[0]['K'],pp[0]['R'],pp[0]['t'],pc[1]['K'],pp[1]['R'],pp[1]['t'])
        except (ValueError,ArithmeticError) as exc:pair_error=str(exc)
        bridge.write(output/f'pair{pair_index}_RECTIFICATION.json',dict(rectification=stereo,error=pair_error))
        pair_rows={'positive':[],'negative_control':[]}
        for group_index,group in enumerate(selected['group_selection']['selected']):
            labels=group['point_ids'];raw=np.array([[v for i in pair for v in images[i]['observations'][points[pid]['track'][i]][:2]] for pid in labels])
            for kind in ('positive','negative_control'):
                right_ids=labels if kind=='positive' else labels[1:]+labels[:1]
                case_raw=raw.copy()
                if kind=='negative_control':case_raw[:,2:]=np.roll(raw[:,2:],-1,axis=0)
                if time.monotonic()>=deadline or stereo is None:
                    row=dict(status='NOT_EVALUATED_BUDGET' if stereo is not None else 'PAIR_RECTIFICATION_REFUSED',
                        reason='shared run deadline' if stereo is not None else pair_error,ablation=None,
                        raw_pixels=case_raw,left_source_track_ids=labels,right_source_track_ids=right_ids,case_kind=kind)
                else:row=_case(case_raw,pc,pp,stereo,labels,kind,right_ids)
                row.update(pair_index=pair_index,image_pair=pair,block_index=group['block_index'],point_ids=labels)
                name=f'p{pair_index}_g{group_index:04d}_{kind}.json'
                bridge.write(output/'groups'/name,row);row['evidence_file']='groups/'+name
                pair_rows[kind].append(row);all_rows[kind].append(row)
                if kind=='positive' and row.get('ablation') is not None:
                    c=row['ablation']['C']
                    if c.get('projection_metrics',{}).get('status')=='NUMERIC_ROUNDTRIP_VERIFIED':
                        for pid,xyz in zip(labels,c['world_xyz']):
                            samples.append(dict(pair_index=pair_index,image_pair=pair,block_index=group['block_index'],
                                point_id=pid,world_xyz=xyz,evidence_file=row['evidence_file']))
        pair_results.append(dict(image_pair=pair,role=selected['role'],common_count=len(selected['common_ids']),
            selected_group_count=len(selected['group_selection']['selected']),
            unevaluated_block_count=len(selected['group_selection']['unselected_blocks']),tail=selected['group_selection']['tail_point_ids'],
            counts={kind:_counts(rows) for kind,rows in pair_rows.items()},
            positive_A_B_unknown_units=_method_stats(pair_rows['positive'],'A_vs_B'),
            positive_B_C_unknown_units=_method_stats(pair_rows['positive'],'B_vs_C'),
            positive_reprojection_pixels=_reprojection_stats(pair_rows['positive']),
            negative_reprojection_pixels=_reprojection_stats(pair_rows['negative_control'])))
    # Reference XYZ enters numerical comparison only after candidate computation.
    # The earlier native parser consumed the binary records but discarded XYZ.
    reference_stats=None;ply=None
    if samples:
        ids=sorted({row['point_id'] for row in samples})
        ref_xyz=bridge.read_reference_xyz(names['points3D.bin'],ids);ref_by_id=dict(zip(ids,ref_xyz))
        differences=[]
        for row in samples:
            old=ref_by_id[row['point_id']];distance=float(np.linalg.norm(row['world_xyz']-old));differences.append(distance)
            row.update(old_colmap_xyz=old,distance_to_old_unknown_units=distance)
        reference_stats=bridge.summary_stats(differences)
        ply=_ply(output/'candidate_samples.ply',samples)
    tracks=defaultdict(list)
    for row in samples:tracks[row['point_id']].append(row)
    repeats=[]
    for pid,rows in sorted(tracks.items()):
        if len(rows)<2:continue
        for a,b in combinations(rows,2):
            if a['pair_index']==b['pair_index']:raise ValueError('duplicate track within pair')
            repeats.append(dict(point_id=pid,pair_indices=[a['pair_index'],b['pair_index']],
                distance_unknown_units=float(np.linalg.norm(a['world_xyz']-b['world_xyz']))))
    bridge.write(output/'CANDIDATE_SAMPLES.json',dict(samples=samples,ply=ply,unique_track_count=len(tracks),
        fused=False,foreground_certified=False,scale_status='unknown',quality_status='NOT_EVALUATED',
        old_reference_comparison_after_candidates=True,old_reference_is_ground_truth=False))
    bridge.write(output/'CROSS_PAIR.json',dict(repeated_track_count=len({row['point_id'] for row in repeats}),
        pairwise_differences=repeats,stats=bridge.summary_stats([r['distance_unknown_units'] for r in repeats]) if repeats else None,
        shared_calibration=True,registered_or_averaged=False,ground_truth=False))
    for ref in refs+[plan['prior_input']]:bridge.verify_binding(ref)
    counts={kind:_counts(rows) for kind,rows in all_rows.items()}
    result=dict(schema='multi-pair-ablation-result-v1',status='EXPERIMENT_COMPLETE',source_unchanged=True,
        selected_pairs_count=len(selection['selected']),group_count=len(all_rows['positive']),counts=counts,pair_results=pair_results,
        candidate_sample_count=len(samples),unique_candidate_track_count=len(tracks),ply=ply,
        positive_A_B_unknown_units=_method_stats(all_rows['positive'],'A_vs_B'),
        positive_B_C_unknown_units=_method_stats(all_rows['positive'],'B_vs_C'),
        positive_reprojection_pixels=_reprojection_stats(all_rows['positive']),
        negative_reprojection_pixels=_reprojection_stats(all_rows['negative_control']),
        repeated_track_count=len({r['point_id'] for r in repeats}),old_reference_distance_unknown_units=reference_stats,
        no_geometry_acceptance=True,independent_new_reconstruction=False,multiview_fusion=False,
        model_gain=None,physical_accuracy=None,quality_status='NOT_EVALUATED',scale_status='unknown',
        findings=dict(same_surrogate_B_C_equivalence_observed=counts['positive']['B_C_comparable']>0 and counts['positive']['B_C_bitwise_equal']==counts['positive']['B_C_comparable'],
            deliberately_mismatched_surrogates_constructed=counts['negative_control']['C_constructed'],
            certificate_is_not_original_match_validation=True),
        limits=['Same scene and existing calibration; image-disjoint pairs are not independent datasets.',
            'Selection favors high common-track counts; sampled groups and unselected tails are explicit.',
            'AlgT verifies modified surrogate consistency, not original correspondence or noisy-photo validity.',
            'Repeated track samples remain separate; no alignment, averaging, fusion, global optimization or mesh.',
            'Raw reprojection and old COLMAP differences are diagnostics, not independent accuracy.'])
    bridge.write(output/'RESULT.json',result);_report(output,result)
    return result


def _report(output,r):
    h='<!doctype html><meta charset="utf-8"><title>Kozlov 多照片对消融实验</title><style>body{font:16px/1.65 system-ui;max-width:1150px;margin:32px auto;padding:0 24px;color:#263b51}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #dce2e9;text-align:left}aside{background:#fff1d6;padding:16px}a{color:#166590}</style>'
    h+='<h1>多照片对：论文构造增加了什么？</h1><aside>这是同一场景中不同照片对的对照实验，不是已经完成多视图融合。已有 COLMAP 相机和对应仍是前提。</aside><p>A：共同纵坐标、未量化双目恢复。B：固定量化后的双目恢复。C：与B相同的输入经过实际 AlgT′、独立证书核验，再恢复三维。</p>'
    h+='<table><tr><th>照片对</th><th>点组</th><th>正常 C 三维</th><th>B/C 完全相同</th><th>错配控制 C 构造</th><th>未评估块/尾点</th></tr>'
    for p in r['pair_results']:
        a=p['counts']['positive'];n=p['counts']['negative_control']
        cells=[p['image_pair'],p['selected_group_count'],f"{a['C_world_ok']}/{a['requested']}",f"{a['B_C_bitwise_equal']}/{a['B_C_comparable']}",f"{n['C_constructed']}/{n['requested']}",f"{p['unevaluated_block_count']} / {len(p['tail'])}"]
        h+='<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in cells)+'</tr>'
    h+='</table><p>故意将右侧原始观测循环错配，再经过相同的共同y修正。如果C仍可构造，只证明修改后的几何内部自洽，不能用证书判定原照片匹配正确。</p>'
    h+=f"<p>导出 {r['candidate_sample_count']} 个正常组坐标样本，涉及 {r['unique_candidate_track_count']} 个不同track；跨对重复track {r['repeated_track_count']} 个。重复点不平均、不拼接，没有面，未认证前景。</p>"
    h+='<p>多图能增加覆盖与观测约束，但需要跨视图匹配、全局优化、稠密融合，不能把多次双目构造等同于完整多视图算法。</p><p><a href="PLAN.json">预冻结选择及全部未评估块</a> · <a href="RESULT.json">完整分层统计</a> · <a href="CANDIDATE_SAMPLES.json">逐点身份与对照</a> · <a href="CROSS_PAIR.json">重复track跨对差异</a></p>'
    if r['ply']:h+='<p><a href="candidate_samples.ply">三维样本 PLY（未融合，无面）</a></p>'
    (output/'report.html').write_text(h,encoding='utf-8')
