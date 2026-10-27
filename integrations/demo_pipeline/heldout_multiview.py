"""Raw-observation holdout diagnostics, conditional on existing COLMAP calibration.

The added fit is ordinary fixed-camera point optimization, not a Kozlov theorem.
No correspondence, camera, mask, source point, or previous result is modified.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import html
from pathlib import Path
import time
import numpy as np

from . import multi_pair_ablation as previous
from . import parallel_photo_audit as native
from . import approximate_photo_bridge as bridge
from kozlov_lab.stereo.baseline import multiview_fixed as kernel

THRESHOLDS = (1., 2., 4.)
METHODS = ('A', 'C', 'J')


def split_views(track_views, pair):
    views = list(track_views); pair = list(pair)
    if (any(type(i) is not int for i in views + pair)
            or len(set(views)) != len(views) or len(pair) != 2
            or len(set(pair)) != 2 or not set(pair).issubset(views)):
        raise ValueError('unique integer views containing two distinct pair IDs required')
    extra = sorted(set(views) - set(pair))
    enough = len(extra) >= 2
    return dict(status='SPLIT_READY' if enough else 'INSUFFICIENT_EXTRA_VIEWS',
                pair=pair, additional_views=extra,
                train_extra=extra[::2] if enough else [],
                heldout=extra[1::2] if enough else [])


def evaluate_sample(sample):
    """Fit only pair+train_extra; heldout is passed only to evaluate, never fit."""
    row = deepcopy(sample); obs = row['observations']
    row['methods'] = {}
    if row['split']['status'] != 'SPLIT_READY':
        row['status'] = 'INSUFFICIENT_EXTRA_VIEWS'
        for m in METHODS:
            row['methods'][m] = dict(status='NOT_EVALUATED', xyz=None,
                reason='at least two extra singleton views needed', training=None,
                heldout=None, pair=None)
        return row
    train = obs['pair'] + obs['train_extra']
    # The separate input lists make held-out pixel leakage observable in tests.
    fitted = kernel.fit(train, max_iterations=20)
    row['fit_details'] = fitted
    candidates = dict(row['candidates'], J=fitted)
    for m in METHODS:
        candidate = candidates[m]; xyz = candidate.get('xyz')
        if xyz is None:
            row['methods'][m] = dict(status=candidate['status'], xyz=None,
                reason=candidate.get('reason'), training=None, heldout=None, pair=None)
        else:
            row['methods'][m] = dict(status=candidate['status'], xyz=xyz,
                training=kernel.evaluate(xyz, train),
                heldout=kernel.evaluate(xyz, obs['heldout']),
                pair=kernel.evaluate(xyz, obs['pair']))
    row['status'] = 'EVALUATED_WITH_STAGE_STATUSES'
    return row


def _prior_files(prior, names):
    """Check any prior published checksum before using its evidence."""
    manifest = prior.parent / 'SHA256SUMS.txt'
    bindings = []
    if manifest.exists():
        entries = {}
        for line in manifest.read_text(encoding='utf-8').splitlines():
            digest, relative = line.split('  ', 1)
            if relative in entries: raise ValueError('duplicate prior manifest path')
            entries[relative] = digest
        for path in names:
            relative = path.relative_to(prior.parent).as_posix()
            if (relative not in entries or
                    hashlib.sha256(path.read_bytes()).hexdigest() != entries[relative]):
                raise ValueError('prior sealed evidence differs: ' + relative)
        bindings.append(native.binding(manifest))
    bindings.extend(native.binding(path) for path in names)
    return bindings, manifest.exists()


def _observation(image_id, pid, images, points, cameras, poses):
    index = points[pid]['track'][image_id]
    pose = poses[image_id]; camera = cameras[pose['camera_id']]
    xy = images[image_id]['observations'][index]
    if xy[2] != pid: raise ValueError('reciprocal observation identity mismatch')
    return dict(image_id=image_id, source_track_id=pid, point2D_index=index,
        raw_pixels=list(xy[:2]), params=camera['params'], R=pose['R'], t=pose['t'],
        max_ideal_radius=camera['radius'])


def _stats(values):
    return bridge.summary_stats(values) if len(values) else None


def summarize(samples):
    result = {}
    for kind in ('positive', 'negative_control'):
        rows = [r for r in samples if r['case_kind'] == kind]
        summary = dict(requested=len(rows),
            unique_left_tracks=len({r['point_id'] for r in rows}),
            split_ready=sum(r['split']['status'] == 'SPLIT_READY' for r in rows),
            joint_converged=sum((r.get('fit_details') or {}).get('converged') is True for r in rows),
            joint_fit_reasons=dict(Counter(str((r.get('fit_details') or {}).get('reason')) for r in rows)),
            statuses=dict(Counter(r['status'] for r in rows)), methods={})
        for m in METHODS:
            available = [r for r in rows if r['methods'][m].get('xyz') is not None]
            evaluated = [r for r in available if (r['methods'][m].get('heldout') or {}).get('status') == 'EVALUATED']
            norms = [v for r in evaluated for v in r['methods'][m]['heldout']['residual_norms']]
            training = [v for r in available if (r['methods'][m].get('training') or {}).get('status') == 'EVALUATED'
                        for v in r['methods'][m]['training']['residual_norms']]
            summary['methods'][m] = dict(candidate_slots=len(available),
                evaluated_slots=len(evaluated), evaluated_groups=len({r['source_group'] for r in evaluated}),
                projection_refused_slots=len(available)-len(evaluated),
                candidate_statuses=dict(Counter(r['methods'][m]['status'] for r in rows)),
                heldout_point_view_residual_pixels=_stats(norms), training_residual_pixels=_stats(training),
                threshold_diagnostics={str(t):dict(
                    eligible_slots=len(evaluated),
                    heldout_max_exceeds=sum(r['methods'][m]['heldout']['max'] > t for r in evaluated),
                    heldout_max_within=sum(r['methods'][m]['heldout']['max'] <= t for r in evaluated),
                    pair_and_heldout_evaluable=sum(r['methods'][m]['pair']['status'] == 'EVALUATED' for r in evaluated),
                    pair_or_heldout_max_exceeds=sum(
                        r['methods'][m]['heldout']['max'] > t or r['methods'][m]['pair']['max'] > t
                        for r in evaluated if r['methods'][m]['pair']['status'] == 'EVALUATED'),
                    pair_and_heldout_max_within=sum(
                        max(r['methods'][m]['heldout']['max'], r['methods'][m]['pair']['max']) <= t
                        for r in evaluated if r['methods'][m]['pair']['status'] == 'EVALUATED'),
                    pair_projection_refused=sum(r['methods'][m]['pair']['status'] != 'EVALUATED' for r in evaluated))
                    for t in THRESHOLDS})
        summary['paired_comparisons'] = {}
        for m in ('A', 'C'):
            both = [r for r in rows if all((r['methods'][v].get('heldout') or {}).get('status') == 'EVALUATED' for v in (m, 'J'))]
            summary['paired_comparisons'][m + '_vs_J'] = dict(
                slots=len(both), groups=len({r['source_group'] for r in both}),
                unique_left_tracks=len({r['point_id'] for r in both}),
                pair_candidate_heldout=_stats([v for r in both for v in r['methods'][m]['heldout']['residual_norms']]),
                joint_candidate_heldout=_stats([v for r in both for v in r['methods']['J']['heldout']['residual_norms']]),
                joint_lower_slot_rms=sum(r['methods']['J']['heldout']['rms'] < r['methods'][m]['heldout']['rms'] for r in both),
                joint_higher_slot_rms=sum(r['methods']['J']['heldout']['rms'] > r['methods'][m]['heldout']['rms'] for r in both),
                joint_equal_slot_rms=sum(r['methods']['J']['heldout']['rms'] == r['methods'][m]['heldout']['rms'] for r in both))
        result[kind] = summary
    return result


def experiment(prior_ablation_dir, output):
    prior = Path(prior_ablation_dir).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(prior):
        raise ValueError('fresh output outside previous evidence required')
    old_plan = native.read_json(prior / 'PLAN.json')
    if old_plan.get('schema') != 'multi-pair-ablation-plan-v1':
        raise ValueError('frozen multi-pair plan required')
    bridge.verify_binding(old_plan['prior_input'])
    _, frozen, names, images, points, _ = previous._load(old_plan['prior_input']['path'])
    root = names['run.json'].parent
    if output.is_relative_to(root): raise ValueError('source run is immutable')
    cameras, poses = bridge.read_calibration(names['cameras.bin'], names['images.bin'])
    if not 1 <= len(old_plan['pairs']) <= 3: raise ValueError('one to three frozen pairs required')
    samples = []; group_paths = []; seen = set()
    for pindex, p in enumerate(old_plan['pairs']):
        pair = p['image_pair']; groups = p['group_selection']['selected']
        if not 1 <= len(groups) <= 40: raise ValueError('one to forty groups required')
        for gindex, group in enumerate(groups):
            labels = group['point_ids']
            if len(labels) != 6 or len(set(labels)) != 6: raise ValueError('six unique labels required')
            for kind in ('positive', 'negative_control'):
                group_path = prior / 'groups' / f'p{pindex}_g{gindex:04d}_{kind}.json'
                case = native.read_json(group_path); group_paths.append(group_path)
                right = labels if kind == 'positive' else labels[1:] + labels[:1]
                if (case['point_ids'] != labels or case['left_source_track_ids'] != labels
                        or case['right_source_track_ids'] != right or case['image_pair'] != pair
                        or case['pair_index'] != pindex or case['block_index'] != group['block_index']
                        or case['case_kind'] != kind): raise ValueError('prior group identity mismatch')
                expected = [[*images[pair[0]]['observations'][points[pid]['track'][pair[0]]][:2],
                             *images[pair[1]]['observations'][points[rid]['track'][pair[1]]][:2]]
                            for pid, rid in zip(labels, right)]
                if case['raw_pixels'] != expected: raise ValueError('prior raw observations mismatch')
                for slot, (pid, rid) in enumerate(zip(labels, right)):
                    key = (kind, pindex, pid)
                    if key in seen: raise ValueError('duplicate point slot within pair')
                    seen.add(key); split = split_views(points[pid]['track'], pair)
                    obs = dict(pair=[_observation(i, id_, images, points, cameras, poses) for i, id_ in zip(pair, (pid, rid))],
                        train_extra=[_observation(i, pid, images, points, cameras, poses) for i in split['train_extra']],
                        heldout=[_observation(i, pid, images, points, cameras, poses) for i in split['heldout']])
                    candidates = {}
                    for m in ('A', 'C'):
                        previous_method = (case.get('ablation') or {}).get(m, {})
                        xyz = previous_method.get('world_xyz')
                        if xyz is not None:
                            a = np.asarray(xyz, dtype=float)
                            if a.shape != (6, 3) or not np.isfinite(a).all(): raise ValueError('invalid prior coordinates')
                            xyz = a[slot]
                        candidates[m] = dict(status=previous_method.get('status', 'NO_PRIOR_CANDIDATE'),
                            xyz=xyz, reason=previous_method.get('reason'))
                    samples.append(dict(case_kind=kind, pair_index=pindex, image_pair=pair,
                        point_id=pid, right_source_track_id=rid, source_group=group_path.name,
                        slot=slot, split=split, observations=obs, candidates=candidates,
                        ambiguous_views_excluded=points[pid].get('ambiguous_observations', {})))
    refs, sealed = _prior_files(prior, [prior / 'PLAN.json', prior / 'RESULT.json'] + group_paths)
    refs.extend(old_plan['source_bindings'])
    frames = previous._bind_photos(root, native.read_json(names['run.json']), images,
        [{'image_pair': sorted(images)}], names)
    refs.extend(f['binding'] for f in frames)
    # De-duplicate only exact path bindings, never observations or point samples.
    by_path = {}
    for ref in refs:
        if ref['path'] in by_path and by_path[ref['path']] != ref: raise ValueError('conflicting input binding')
        bridge.verify_binding(ref); by_path[ref['path']] = ref
    refs = list(by_path.values())
    plan = dict(schema='heldout-multiview-plan-v1', source_bindings=refs,
        previous_evidence_sealed=sealed, frames=frames,
        sample_count=len(samples), group_count=len(group_paths),
        split_rule='sorted extra singleton image IDs: even positions train, odd positions heldout; require at least two extras',
        splits=[dict(case_kind=s['case_kind'], pair_index=s['pair_index'], point_id=s['point_id'],
            right_source_track_id=s['right_source_track_id'], source_group=s['source_group'], split=s['split']) for s in samples],
        diagnostic_thresholds_pixels=list(THRESHOLDS), threshold_metric='maximum 2D Euclidean point-view residual',
        quality_acceptance_threshold=None, max_fit_iterations=20, whole_run_soft_seconds=180,
        outer_hard_seconds=240, optimization='XYZ only; fixed existing SIMPLE_RADIAL cameras; no robust loss/outlier deletion',
        calibration_and_tracks_previously_used_heldout_observations=True,
        independent_validation=False, negative_correspondences_preserved=True,
        old_colmap_xyz_used=False, original_algorithms_reexecuted=False,
        scope='Reuse bound A/C candidates; J is standard multiview DLT plus fixed-camera LM, not a Kozlov extension')
    output.mkdir(parents=True, exist_ok=False)
    bridge.write(output / 'PLAN.json', plan)
    deadline = time.monotonic() + 180
    evaluated = []
    for sample in samples:
        if time.monotonic() < deadline:
            row = evaluate_sample(sample)
        else:
            row = deepcopy(sample); row['status'] = 'NOT_EVALUATED_BUDGET'
            row['methods'] = {m:dict(status='NOT_EVALUATED_BUDGET', xyz=None,
                training=None, heldout=None, pair=None) for m in METHODS}
        evaluated.append(row)
    bridge.write(output / 'SAMPLES.json', dict(schema='heldout-multiview-samples-v1', samples=evaluated))
    for ref in refs: bridge.verify_binding(ref)
    result = dict(schema='heldout-multiview-result-v1', status='EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED',
        sample_count=len(evaluated), group_count=len(group_paths), counts=summarize(evaluated),
        source_unchanged=True, new_camera_estimation=False, fixed_camera_point_optimization=True,
        independent_validation=False, reused_prior_kozlov_candidates=True,
        kozlov_algorithm_gain=None, physical_accuracy=None, scale_status='unknown',
        quality_status='NOT_EVALUATED', fusion=False, mesh=False, cad=False,
        limits=['Cameras and tracks were previously estimated and filtered using heldout observations.',
                'Deterministic ID split and highly connected pairs are not an independent dataset.',
                'Point slots repeat track IDs across pairs; samples are correlated.',
                '1/2/4 px are diagnostic sensitivity levels, not accepted accuracy or truth labels.',
                'Joint fit retains the wrong right observation; many correct extra views can reduce its influence.',
                'A/C prior coordinates are reused; this run tests raw observations, not another exact construction.',
                'No accepted geometry, fusion, dense reconstruction, foreground certification or CAD.'])
    bridge.write(output / 'RESULT.json', result)
    _report(output, result)
    return result


def _report(output, result):
    page = '<!doctype html><meta charset="utf-8"><title>多视图留出检验</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#25364a}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}aside{background:#fff1d6;padding:16px}</style>'
    page += '<h1>多视图：原始观测留出检验</h1><aside>相机标定和轨迹来自既有 COLMAP，曾使用留出照片。因此这是条件化诊断，不是独立精度验收。</aside>'
    page += '<p>A：原有未量化双目候选；C：原有论文构造证书回投影候选；J：固定相机、多张原始照片联合拟合一个三维点。额外视图按 ID 排序交替分为拟合和留出集合；没有修改原始坐标。</p>'
    page += '<table><tr><th>样本</th><th>方法</th><th>可检验点槽</th><th>留出观测数</th><th>留出 RMS / px</th><th>最大残差超 2 px 点槽</th></tr>'
    for kind, summary in result['counts'].items():
        for m, stat in summary['methods'].items():
            residual = stat['heldout_point_view_residual_pixels']
            cells = [kind, m, f"{stat['evaluated_slots']}/{summary['requested']}",
                residual['count'] if residual else None, round(residual['rms'], 5) if residual else None,
                stat['threshold_diagnostics']['2.0']['heldout_max_exceeds']]
            page += '<tr>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in cells) + '</tr>'
    page += '</table><p>不同方法可评估分母不同。正式数值比较使用 RESULT 的 paired_comparisons，限制为同一批点槽和完全相同的留出观测。故意错配不代表自然错误率；不删除原右观测。</p>'
    page += '<p>这一步增加工程诊断与固定相机优化，没有把多视图优化冒称为 Kozlov 原论文算法，也没有生成新的网格或 CAD。</p><p><a href="PLAN.json">预冻结计划及分割</a> · <a href="RESULT.json">分层统计与配对比较</a> · <a href="SAMPLES.json">逐点原始观测、拟合与留出残差</a></p>'
    (output / 'report.html').write_text(page, encoding='utf-8')
