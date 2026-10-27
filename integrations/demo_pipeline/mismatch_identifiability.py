"""Explain saved correspondence controls using original observations and fixed cameras.

This is a post-selected diagnostic, not a new matching rule or physical accuracy
estimate. No optimization, coordinate edits, track merging, or paper reruns occur.
"""
from collections import Counter
import html
from pathlib import Path
import numpy as np

from . import approximate_photo_bridge as bridge
from . import parallel_photo_audit as native
from . import heldout_multiview as prior_helpers
from kozlov_lab.stereo.baseline import multiview_fixed as numeric
from kozlov_lab.stereo.baseline import observability

THRESHOLD = 2.0
SEGMENT_FRACTIONS = (0., .25, .5, .75, 1.)


def full_observation_diagnostic(sample):
    """Diagnostic only: every supplied view must be checked, never just holdout.

    A subthreshold residual is NOT a certificate of correspondence or geometry.
    Insufficient original split coverage is preserved rather than filled in.
    """
    result=dict(case_kind=sample['case_kind'],pair_index=sample['pair_index'],
        point_id=sample['point_id'],threshold_pixels=THRESHOLD,
        decision_is_production_acceptance=False,observation_count=0)
    if sample['split']['status']!='SPLIT_READY':
        return dict(result,status='NOT_EVALUATED_INSUFFICIENT_VIEWS',metrics=None)
    xyz=sample['candidates']['C'].get('xyz')
    if xyz is None: return dict(result,status='NO_CANDIDATE',metrics=None)
    obs=sample['observations']
    full=obs['pair']+obs['train_extra']+obs['heldout']
    result['observation_count']=len(full)
    metrics=numeric.evaluate(xyz,full)
    if metrics['status']!='EVALUATED': status='PROJECTION_REFUSED'
    elif metrics['max']>THRESHOLD: status='OBSERVATION_INCONSISTENCY_DETECTED'
    else: status='WITHIN_DIAGNOSTIC_LINE_NOT_VERIFIED'
    return dict(result,status=status,metrics=metrics)


def inspect_pair(positive, negative, *, include_segment=False):
    """One saved slot: compare its reference and mismatched candidate, no refit."""
    identity = ('pair_index', 'point_id', 'image_pair')
    if any(positive[k] != negative[k] for k in identity):
        raise ValueError('same original point slot required')
    if positive['case_kind'] != 'positive' or negative['case_kind'] != 'negative_control':
        raise ValueError('positive/negative slot roles required')
    if positive['split'] != negative['split']:
        raise ValueError('identical frozen view split required')
    for bucket in ('train_extra', 'heldout'):
        if positive['observations'][bucket] != negative['observations'][bucket]:
            raise ValueError('extra observations cannot change in a negative control')
    if positive['observations']['pair'][0] != negative['observations']['pair'][0]:
        raise ValueError('left observation cannot change')
    pxyz = positive['candidates']['A'].get('xyz')
    nxyz = negative['candidates']['C'].get('xyz')
    if pxyz is None or nxyz is None:
        return dict(status='CANDIDATE_UNAVAILABLE', reason='reference A and negative C required')
    pxyz = np.asarray(pxyz, dtype=float); nxyz = np.asarray(nxyz, dtype=float)
    if pxyz.shape != (3,) or nxyz.shape != (3,) or not np.isfinite([pxyz,nxyz]).all():
        raise ValueError('finite XYZ required')
    observations = positive['observations']
    support = observations['pair'][:1] + observations['train_extra'] + observations['heldout']
    ids = [o['image_id'] for o in support]
    right = observations['pair'][1]
    if len(ids) != len(set(ids)) or right['image_id'] in ids:
        raise ValueError('support must have unique views and exclude right camera')
    delta = np.array(negative['observations']['pair'][1]['raw_pixels']) - np.array(right['raw_pixels'])
    right_delta = float(np.linalg.norm(delta))
    separation = nxyz-pxyz
    observations_by_id = {o['image_id']:o for o in support+[right]}
    prediction_deltas=[]
    for image_id, o in observations_by_id.items():
        try:
            pp = numeric.project(pxyz, o); nn = numeric.project(nxyz, o)
            p=pp['pixels']; n=nn['pixels']; measured=np.array(o['raw_pixels'])
            exact_delta=n-p; p_res=p-measured; n_res=n-measured
            linear_delta=pp['jacobian']@separation
            prediction_deltas.append(dict(image_id=image_id, status='EVALUATED',
                reference_raw_projection=p, negative_raw_projection=n,
                distance_pixels=float(np.linalg.norm(exact_delta)),
                exact_delta=exact_delta, reference_residual=p_res, negative_residual=n_res,
                residual_identity_error=float(np.linalg.norm(n_res-(p_res+exact_delta))),
                linear_delta_at_reference=linear_delta,
                linearization_error_pixels=float(np.linalg.norm(linear_delta-exact_delta))))
        except (ValueError, ArithmeticError) as exc:
            prediction_deltas.append(dict(image_id=image_id, status='PROJECTION_REFUSED', reason=str(exc)))
    evals = {}
    for name, xyz in (('reference_A', pxyz), ('negative_C', nxyz)):
        evals[name] = dict(
            support=numeric.evaluate(xyz,support),
            # Right reference is diagnostic hindsight, never an input to inference.
            original_correct_right=numeric.evaluate(xyz,[right]),
            original_train_extra=numeric.evaluate(xyz, observations['train_extra']) if observations['train_extra'] else None,
            original_heldout=numeric.evaluate(xyz, observations['heldout']) if observations['heldout'] else None,
            observability=observability.analyze(xyz,support,predict_observation=right),
            left_plus_heldout_observability=observability.analyze(xyz,
                observations['pair'][:1]+observations['heldout'],predict_observation=right))
    segment=[]
    if include_segment:
        for fraction in SEGMENT_FRACTIONS:
            xyz=pxyz+fraction*separation
            segment.append(dict(fraction=fraction, xyz=xyz,
                support=numeric.evaluate(xyz,support),
                original_correct_right=numeric.evaluate(xyz,[right])))
    e=evals['negative_C']['support']
    if right_delta == 0:
        interpretation='RIGHT_LABEL_CHANGE_WITH_IDENTICAL_PIXELS'
    elif e['status'] != 'EVALUATED':
        interpretation='SUPPORT_PROJECTION_REFUSED'
    elif e['max'] > THRESHOLD:
        interpretation='ADDITIONAL_SUPPORT_EXCEEDS_DIAGNOSTIC_LINE'
    else:
        interpretation='ALTERNATIVE_CANDIDATE_FITS_AVAILABLE_SUPPORT_AT_DIAGNOSTIC_LINE'
    return dict(status='DIAGNOSED', interpretation=interpretation,
        pair_index=positive['pair_index'], point_id=positive['point_id'],
        right_source_track_id=negative['right_source_track_id'], image_pair=positive['image_pair'],
        split=positive['split'], right_raw_difference_pixels=right_delta,
        right_raw_exactly_equal=right_delta==0,
        reference_A_xyz=pxyz, negative_C_xyz=nxyz,
        separation_unknown_units=float(np.linalg.norm(separation)),
        support_image_ids=ids, excluded_right_image_id=right['image_id'],
        full_supplied_observation_diagnostic=full_observation_diagnostic(negative),
        predictions=prediction_deltas, diagnostics=evals, segment_samples=segment,
        old_reference_is_ground_truth=False, source_raw_pixels_edited=False)


def _read_inputs(prior):
    expected=['real01/PLAN.json','real01/RESULT.json','real01/SAMPLES.json','design_review/AUDIT.json']
    manifest=prior/'SHA256SUMS.txt'
    if not manifest.is_file(): raise ValueError('published prior checksum manifest required')
    entries={}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if name in entries: raise ValueError('duplicate prior checksum path')
        entries[name]=digest
    refs=[native.binding(manifest)]
    for relative in expected:
        binding=native.binding(prior/relative)
        if entries.get(relative) != binding['sha256']: raise ValueError('prior evidence drift: '+relative)
        refs.append(binding)
    plan=native.read_json(prior/'real01/PLAN.json')
    if plan.get('schema')!='heldout-multiview-plan-v1': raise ValueError('heldout source plan required')
    for binding in plan['source_bindings']:
        bridge.verify_binding(binding)
    refs.extend(plan['source_bindings'])
    rows=native.read_json(prior/'real01/SAMPLES.json')['samples']
    if not 1<=len(rows)<=1440: raise ValueError('bounded prior sample ledger required')
    table={}
    for row in rows:
        key=(row['case_kind'],row['pair_index'],row['point_id'])
        if key in table: raise ValueError('duplicate prior point slot')
        table[key]=row
    return table,refs


def experiment(prior_run, output):
    prior=Path(prior_run).resolve(strict=True); output=Path(output).resolve()
    if output.exists() or output.is_relative_to(prior): raise ValueError('fresh output outside prior required')
    table,refs=_read_inputs(prior)
    negatives=[r for key,r in sorted(table.items()) if key[0]=='negative_control']
    cases=[]; selected=[]
    for row in negatives:
        positive=table.get(('positive',row['pair_index'],row['point_id']))
        if positive is None: raise ValueError('every control needs its original positive slot')
        e=row['methods']['C'].get('heldout')
        if e and e['status']=='EVALUATED' and e['max']<=THRESHOLD:
            delta=np.array(row['observations']['pair'][1]['raw_pixels'])-np.array(positive['observations']['pair'][1]['raw_pixels'])
            selected.append(dict(pair_index=row['pair_index'],point_id=row['point_id'],
                right_source_track_id=row['right_source_track_id'],right_raw_exactly_equal=bool(np.all(delta==0))))
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(schema='mismatch-identifiability-plan-v1',source_bindings=refs,
        selection='all negative C heldout EVALUATED with max<=2 px; include exact-coordinate aliases',
        negative_requested=len(negatives),selected=selected,diagnostic_threshold_pixels=THRESHOLD,
        segment_fractions=list(SEGMENT_FRACTIONS),reference='positive unquantized A, not ground truth',
        support='original left plus all recorded extra singleton views; exclude original right',
        observability='fixed-camera original-pixel Jacobian SVD and unit-noise linear response only',
        post_selected_diagnostic=True,performance_estimate=False,model_optimization=False,
        correspondence_or_pixel_edit=False,quality_acceptance_threshold=None)
    bridge.write(output/'PLAN.json',plan)
    all_diagnostics=[full_observation_diagnostic(row) for _,row in sorted(table.items())]
    bridge.write(output/'FULL_OBSERVATION_DIAGNOSTICS.json',dict(
        schema='full-observation-diagnostic-v1',samples=all_diagnostics,
        no_coordinates_changed=True,no_points_deleted=True,production_acceptance=False))
    selected_keys={(r['pair_index'],r['point_id']) for r in selected}
    for negative in negatives:
        key=(negative['pair_index'],negative['point_id'])
        if key not in selected_keys: continue
        positive=table[('positive',*key)]
        cases.append(inspect_pair(positive,negative,include_segment=True))
    bridge.write(output/'CASES.json',dict(schema='mismatch-identifiability-cases-v1',cases=cases))
    for binding in refs: bridge.verify_binding(binding)
    distinct=[c for c in cases if c.get('right_raw_exactly_equal') is False]
    result=dict(schema='mismatch-identifiability-result-v1',status='DIAGNOSTIC_COMPLETE_NOT_GEOMETRY_ACCEPTED',
        negative_requested=len(negatives),selected_count=len(cases),
        interpretation_counts=dict(Counter(c.get('interpretation',c['status']) for c in cases)),
        distinct_right_pixel_cases=len(distinct),same_right_pixel_cases=sum(c.get('right_raw_exactly_equal') is True for c in cases),
        full_observation_statuses={kind:dict(Counter(r['status'] for r in all_diagnostics if r['case_kind']==kind))
            for kind in ('positive','negative_control')},
        distinct_case_summaries=[dict(pair_index=c['pair_index'],point_id=c['point_id'],
            right_source_track_id=c['right_source_track_id'],image_pair=c['image_pair'],
            interpretation=c['interpretation'],right_raw_difference_pixels=c['right_raw_difference_pixels'],
            support_image_ids=c['support_image_ids'],negative_support=c['diagnostics']['negative_C']['support'],
            reference_support=c['diagnostics']['reference_A']['support'],
            reference_observability=c['diagnostics']['reference_A']['observability'],
            negative_observability=c['diagnostics']['negative_C']['observability']) for c in distinct],
        source_unchanged=True,quality_status='NOT_EVALUATED',scale_status='unknown',
        physical_uncertainty=None,new_geometry=False,algorithm_gain=None,
        limits=['Cases are selected after the prior diagnostic, not a held-out performance benchmark.',
                'Camera calibration and track identity reuse old COLMAP data and are not independent truth.',
                'SVD is local sensitivity under fixed cameras; unit noise is hypothetical, not measured error.',
                'Five samples on a candidate segment do not prove the entire continuous segment is feasible.',
                'Different track IDs can yield identical observations; coordinate tests cannot disambiguate such labels.',
                'No matching, SfM, MVS, SAM3, paper construction, or CAD was rerun.'])
    bridge.write(output/'RESULT.json',result); _report(output,result)
    return result


def _report(output,result):
    page='<!doctype html><meta charset="utf-8"><title>低残差错配诊断</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#20364a}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}aside{padding:15px;background:#fff1d6}</style>'
    page+='<h1>为什么错误右像素仍可能得到低残差？</h1><aside>这是对既有异常的事后诊断，不是识别率、真实精度或新重建成果。相机固定，原始观测没有修改。</aside>'
    page+=f"<p>从 {result['negative_requested']} 个原有负对照点槽中按原定 2 px 留出线选中 {result['selected_count']} 个：{result['same_right_pixel_cases']} 个右坐标完全相同，{result['distinct_right_pixel_cases']} 个右坐标确实不同。训练额外视图也纳入本次诊断，原右相机保留作对照。</p>"
    page+='<table><tr><th>原轨迹→错误右轨迹</th><th>右像素变化/px</th><th>支持视图</th><th>参考A 最大残差/px</th><th>负对照C 最大残差/px</th><th>状态</th></tr>'
    for c in result['distinct_case_summaries']:
        cells=[f"{c['point_id']} → {c['right_source_track_id']}",round(c['right_raw_difference_pixels'],4),c['support_image_ids'],c['reference_support']['max'],c['negative_support']['max'],c['interpretation']]
        page+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in cells)+'</tr>'
    page+='</table><p>局部雅可比奇异值和假设每坐标1像素噪声的线性响应只描述固定相机条件下的敏感度，不能称为毫米误差、真实置信区间或已确认物理歧义。</p><p><a href="PLAN.json">选择与分析边界</a> · <a href="RESULT.json">分层结论</a> · <a href="CASES.json">逐案投影与敏感度</a></p>'
    (output/'report.html').write_text(page,encoding='utf-8')
