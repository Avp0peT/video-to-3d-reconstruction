"""Recheck frozen radial anchor candidates without fitting, I/O or acceptance."""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import math
import numpy as np
from .original_anchor_review import PROFILE as SOURCE_PROFILE, _json
from kozlov_lab.stereo.baseline import multiview_fixed as fixed, observability, radial_diagnostics as radial
from kozlov_lab.stereo.baseline.radial_branch import propose_central_domain

PROFILE=dict(profile_id='radial-anchor-candidate-v3',max_groups=96,batch_size=32,max_batches=3,max_observations=12,
    strict=dict(min_views=3,max_raw_error_px=2.,max_loo_error_px=3.,min_measured_angle_deg=2.),
    relaxed=dict(min_views=3,max_raw_error_px=4.,max_loo_error_px=6.,min_measured_angle_deg=1.),
    two_view=dict(views=2,max_raw_error_px=4.,min_measured_angle_deg=1.),
    max_condition=1e4,rank_relative_threshold=1e-12)
RIGHTS=dict(geometry_accepted=False,native_injected=False,identity_verified=False,scale_status='unknown')


def _require(condition,reason):
    if not condition:raise ValueError(reason)


def _identity(group):
    _require(isinstance(group.get('model_key'),str) and bool(group['model_key']),'INVALID_MODEL_KEY')
    _require(all(type(group.get(k)) is int and group[k]>=0 for k in ('track_id','identity_segment')),'INVALID_TRACK_SEGMENT')
    _require(group.get('group_id')==f"{group['model_key']}:{group['track_id']}:{group['identity_segment']}",'GROUP_ID_MISMATCH')
    _require(all(group.get(k) is False for k in ('geometry_acceptance','native_injected','identity_verified')),'SOURCE_RIGHTS_INVALID')


def _observations(group,budget):
    admitted=group['admitted_observations'];source=group['observations']
    _require(isinstance(admitted,list) and 2<=len(admitted)<=12 and isinstance(source,list),'OBSERVATION_COUNT_OR_STRUCTURE')
    rows={};source_frames=set()
    for row in source:
        _require(type(row.get('input_index')) is int and row['input_index']>=0 and row['input_index'] not in rows,'SOURCE_INPUT_IDENTITY')
        _require(isinstance(row.get('frame_id'),str) and row['frame_id'] and row['frame_id'] not in source_frames,'SOURCE_FRAME_IDENTITY')
        _require(row.get('track_id')==group['track_id'] and type(row.get('track_id')) is int and row.get('identity_segment')==group['identity_segment'] and type(row.get('identity_segment')) is int,'SOURCE_TRACK_IDENTITY')
        _require(row.get('status') in ('ADMITTED_ORIGINAL_ANCHOR','EXCLUDED') and isinstance(row.get('reasons'),list),'SOURCE_ADMISSION_STATUS')
        _require((row['reasons']==[])==(row['status']=='ADMITTED_ORIGINAL_ANCHOR'),'SOURCE_ADMISSION_REASONS')
        rows[row['input_index']]=row;source_frames.add(row['frame_id'])
    output=[];seen=set();images=set();frames=set()
    for observation in admitted:
        budget();index=observation['input_index'];row=rows.get(index)
        _require(type(index) is int and row is not None and index not in seen,'ADMITTED_INPUT_IDENTITY')
        _require(row['status']=='ADMITTED_ORIGINAL_ANCHOR' and observation.get('frame_id')==row['frame_id'],'ADMITTED_SOURCE_IDENTITY')
        raw=radial._array(observation['raw_pixels'],(2,));saved=radial._array(row['source_xy'],(2,))
        _require(np.array_equal(raw,saved),'ADMITTED_SOURCE_XY_MISMATCH')
        iid=observation['image_id'];fid=observation['frame_id']
        _require(type(iid) is int and iid>0 and iid not in images and fid not in frames,'DUPLICATE_OR_INVALID_CAMERA_IDENTITY')
        _require(observation.get('camera_model')=='SIMPLE_RADIAL','UNSUPPORTED_CAMERA_MODEL')
        _require(all(type(observation.get(k)) is int and 1<=observation[k]<=16384 for k in ('width','height')),'INVALID_IMAGE_DIMENSIONS')
        _require(0<=raw[0]<observation['width'] and 0<=raw[1]<observation['height'],'RAW_PIXEL_OUTSIDE_IMAGE')
        params=radial._camera(observation['params'],2);rotation=radial._rotation(observation['R']);translation=radial._array(observation['t'],(3,))
        domain=propose_central_domain(params.tolist(),observation['width'],observation['height'],[raw.tolist()])
        _require(domain['all_observations_invertible'],'CENTRAL_BRANCH_INVERSE_REFUSED')
        output.append(dict(deepcopy(observation),max_ideal_radius=domain['max_ideal_radius']))
        seen.add(index);images.add(iid);frames.add(fid)
    _require(seen=={i for i,row in rows.items() if row['status']=='ADMITTED_ORIGINAL_ANCHOR'},'ADMITTED_SUBSET_INCOMPLETE')
    # A seed is a distinct evidence type. Its original absent flow data is never made strong.
    admitted_by_index={o['input_index']:o for o in admitted}
    for observation in admitted:
        row=rows[observation['input_index']]
        if observation.get('evidence_kind')=='VERIFIED_SEED_PAIR' or row.get('evidence_kind')=='VERIFIED_SEED_PAIR':
            from .seed_pair_validation import validate_certificate
            c=observation.get('seed_certificate')
            _require(row.get('evidence_kind')==observation.get('evidence_kind')=='VERIFIED_SEED_PAIR'
                and isinstance(c,dict) and c==row.get('seed_certificate'),'SEED_EVIDENCE_BINDING')
            _require(validate_certificate(c)['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY','SEED_PAIR_NUMERICS_REFUSED')
            _require((c['group_id'],c['model_key'],c['track_id'],c['identity_segment'])==
                (group['group_id'],group['model_key'],group['track_id'],group['identity_segment']),'SEED_GROUP_BINDING')
            successor=admitted_by_index.get(c['successor_input_index']);source_tracking=row.get('original_tracking',{})
            _require(source_tracking.get('tracking_status')=='seed' and source_tracking.get('strong') is False
                and source_tracking.get('fb_error_px') is None and source_tracking.get('lk_error') is None
                and source_tracking.get('sequence_position')==c['source_position'],'ORIGINAL_SEED_STATE_INVALID')
            _require(successor is not None and successor.get('evidence_kind')!='VERIFIED_SEED_PAIR','SEED_SUCCESSOR_NOT_STRONG')
            next_tracking=rows[successor['input_index']].get('original_tracking',{})
            _require(next_tracking.get('strong') is True and next_tracking.get('sequence_position')==c['successor_position'],'SEED_SUCCESSOR_ORIGINAL_STATE')
            strong=[o for o in admitted if o.get('evidence_kind')!='VERIFIED_SEED_PAIR']
            _require(len(strong)==2 and c['strong_anchor_image_ids']==[o['image_id'] for o in strong],'SEED_STRONG_SET_BINDING')
            for role,o in [('source',observation),('successor',successor)]:
                _require(c[role+'_input_index']==o['input_index'] and c[role+'_frame_id']==o['frame_id']
                    and c[role+'_image_id']==o['image_id'] and c[role+'_raw_xy']==o['raw_pixels'],'SEED_MEASUREMENT_BINDING')
                _require(all(c[role+'_camera'][key]==o[key] for key in
                    ('input_index','frame_id','image_id','camera_model','params','R','t','width','height')),'SEED_CAMERA_BINDING')
                _require(c[role+'_image_binding']==rows[o['input_index']].get('image_binding')
                    and c[role+'_mask_binding']==rows[o['input_index']].get('mask_binding'),'SEED_FILE_BINDING')
    return output


def _fit_point(fit,expected):
    _require(isinstance(fit,dict) and fit.get('status')=='FITTED' and fit.get('converged') is True,'FIT_NOT_CONVERGED')
    _require(fit.get('train_image_ids')==expected and all(type(i) is int for i in fit['train_image_ids']),'FIT_TRAIN_IDENTITY_MISMATCH')
    return radial._array(fit['xyz'],(3,))


def _measure(xyz,observations,budget):
    budget();projections=[fixed.project(xyz,o) for o in observations]
    _require(all(p['depth']>0 and math.isfinite(p['depth']) for p in projections),'NON_POSITIVE_DEPTH')
    evaluated=fixed.evaluate(xyz,observations)
    _require(evaluated.get('status')=='EVALUATED' and len(evaluated['residual_norms'])==len(observations),'PROJECTION_REFUSED')
    diagnostic=observability.analyze(xyz,observations)
    _require(diagnostic.get('status')=='DIAGNOSTIC_EVALUATED' and diagnostic.get('rank')==3,'OBSERVABILITY_REFUSED_OR_RANK_DEFICIENT')
    singular=np.linalg.svd(np.vstack([p['jacobian'] for p in projections]),compute_uv=False)
    rank=int(np.count_nonzero(singular>singular[0]*PROFILE['rank_relative_threshold']))
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else math.inf
    _require(rank==3 and math.isfinite(condition) and condition<=PROFILE['max_condition'],'ILL_CONDITIONED_POINT')
    angle=max(p['ray_separation_degrees'] for p in diagnostic['pairs'])
    return dict(max_raw_error_px=max(evaluated['residual_norms']),measurement_ray_angle_deg=angle,
        jacobian_rank=rank,jacobian_condition=condition,all_depths_positive=True,
        raw_predictions=[dict(input_index=o['input_index'],raw_pixels=o['raw_pixels'],predicted_raw_pixels=p['pixels'],depth=p['depth']) for o,p in zip(observations,projections)])


def _tier(n,error,loo,angle):
    if n==2:return 'TWO_VIEW_PROVISIONAL' if error<=4. and angle>=1. else 'REJECTED'
    if n>=3 and error<=2. and loo<=3. and angle>=2.:return 'STRICT_CONDITIONAL'
    if n>=3 and error<=4. and loo<=6. and angle>=1.:return 'RELAXED_CONDITIONAL'
    return 'REJECTED'


def _classify(group,budget):
    row=dict(group_id=group.get('group_id'),model_key=group.get('model_key'),track_id=group.get('track_id'),
        identity_segment=group.get('identity_segment'),tier='REJECTED',reasons=[],xyz=None,metrics={},sources=[],
        branch_scope='MONOTONIC_RADIAL_CONDITIONAL',branch_scope_explanation='Frozen cameras; central branch where k<0 is assumed, not physically verified',source_indices=[],frame_ids=[],
        original_strict_domain_reference=deepcopy(group.get('strict_radial_domain_check',[])),**RIGHTS)
    try:
        _identity(group);observations=_observations(group,budget)
        row['branch_scope']='CENTRAL_BRANCH_CONDITIONAL' if any(o['params'][3]<0 for o in observations) else 'MONOTONIC_RADIAL_CONDITIONAL'
        row['source_indices']=[o['input_index'] for o in observations];row['frame_ids']=[o['frame_id'] for o in observations]
        row['sources']=[dict(input_index=o['input_index'],frame_id=o['frame_id'],image_id=o['image_id'],raw_pixels=o['raw_pixels'],
            evidence_kind=o.get('evidence_kind','ORIGINAL_STRONG_FLOW')) for o in observations]
        xyz=_fit_point(group['fit'],[o['image_id'] for o in observations]);metrics=_measure(xyz,observations,budget)
        folds=group['leave_one_view_out'];_require(isinstance(folds,list),'INVALID_LOO_STRUCTURE')
        loo=[]
        if len(observations)==2:_require(folds==[],'TWO_VIEW_UNEXPECTED_LOO')
        else:
            _require(len(folds)==len(observations),'LOO_FOLD_COUNT_MISMATCH');by_index={o['input_index']:o for o in observations};seen=set()
            for fold in folds:
                budget();_require(all(fold.get(k) is False for k in ('geometry_acceptance','native_injected','identity_verified')),'LOO_SOURCE_RIGHTS_INVALID')
                index=fold['heldout_input_index']
                _require(type(index) is int and index in by_index and index not in seen,'LOO_HELDOUT_IDENTITY')
                held=by_index[index];train=[o for o in observations if o['input_index']!=index]
                _require(type(fold.get('heldout_image_id')) is int and fold['heldout_image_id']==held['image_id'],'LOO_HELDOUT_CAMERA_IDENTITY')
                _require(fold.get('training_input_indices')==[o['input_index'] for o in train] and all(type(i) is int for i in fold['training_input_indices']),'LOO_TRAIN_IDENTITY')
                point=_fit_point(fold['fit'],[o['image_id'] for o in train])
                _measure(point,train,budget);all_metrics=_measure(point,observations,budget)
                prediction=fixed.evaluate(point,[held]);_require(prediction['status']=='EVALUATED','LOO_PROJECTION_REFUSED')
                loo.append(dict(heldout_input_index=index,error_px=prediction['residual_norms'][0],all_depths_positive=True,
                    metrics=all_metrics));seen.add(index)
        metrics['loo']=loo;metrics['max_loo_error_px']=max((f['error_px'] for f in loo),default=None)
        row['metrics']=metrics;row['tier']=_tier(len(observations),metrics['max_raw_error_px'],metrics['max_loo_error_px'],metrics['measurement_ray_angle_deg'])
        if row['tier']=='REJECTED':row['reasons']=['PIXEL_ERROR_OR_MEASURED_ANGLE_THRESHOLD']
        else:row['xyz']=xyz.tolist()
        _json(row)
    except (ValueError,KeyError,TypeError,ArithmeticError,np.linalg.LinAlgError) as exc:
        row.update(tier='REJECTED',reasons=[str(exc)],xyz=None,metrics={})
    return row


def classify_original_candidates(diagnostics,budget=lambda:None):
    """Caller rebuilds and authenticates original mask/track/camera bindings.

    Saved error, rank, domain, summary and quality labels never grant admission.
    Structural partition errors raise; individual selected failures are retained.
    """
    _require(isinstance(diagnostics,dict) and diagnostics.get('profile')==SOURCE_PROFILE,'SOURCE_PROFILE_MISMATCH')
    _require(diagnostics.get('status')=='ORIGINAL_ANCHOR_DIAGNOSTICS_COMPLETE','SOURCE_STATUS_INVALID')
    _require(all(diagnostics.get(k) is False for k in ('geometry_acceptance','native_injected','identity_verified')),'SOURCE_RIGHTS_INVALID')
    groups=diagnostics['groups'];selected=diagnostics['selected_groups'];omitted=diagnostics['omitted_groups'];counts=diagnostics['counts']
    _require(isinstance(groups,list) and isinstance(selected,list) and len(selected)<=PROFILE['max_groups'] and isinstance(omitted,list),'PARTITION_STRUCTURE')
    ids=[g['group_id'] for g in groups];omitted_ids=[o['group_id'] for o in omitted]
    _require(all(isinstance(i,str) for i in ids+selected+omitted_ids) and len(set(ids))==len(ids) and len(set(selected))==len(selected) and len(set(omitted_ids))==len(omitted_ids),'DUPLICATE_GROUP_ID')
    _require(not set(selected)&set(omitted_ids) and set(selected)|set(omitted_ids)==set(ids),'GROUP_PARTITION_MISMATCH')
    _require(all(type(counts.get(k)) is int for k in ('groups','selected_groups','omitted_groups')),'GROUP_COUNT_TYPE')
    _require(counts.get('groups')==len(groups) and counts.get('selected_groups')==len(selected) and counts.get('omitted_groups')==len(omitted),'GROUP_COUNT_MISMATCH')
    statuses=Counter(g['status'] for g in groups)
    _require(counts.get('statuses')==dict(statuses),'STATUS_COUNT_MISMATCH')
    by_id={g['group_id']:g for g in groups}
    _require(all(o.get('status')==by_id[o['group_id']]['status'] for o in omitted),'OMITTED_STATUS_MISMATCH')
    # Rebuild the source adapter's deterministic numerical budget partition.
    expected=[]
    for multiview in (True,False):
        buckets=defaultdict(deque)
        for group in sorted(groups,key=lambda g:(g['model_key'],g['track_id'],g['identity_segment'])):
            _require(isinstance(group.get('admitted_observations'),list),'ADMITTED_STRUCTURE')
            n=len(group['admitted_observations'])
            if 2<=n<=PROFILE['max_observations'] and (n>=3)==multiview:buckets[group['model_key']].append(group['group_id'])
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:expected.append(buckets[key].popleft())
    _require(selected==expected[:PROFILE['max_groups']],'NUMERIC_SELECTION_PARTITION_MISMATCH')
    selection=dict(policy=SOURCE_PROFILE['selection'],eligible_groups=len(expected),batch_size=PROFILE['batch_size'],
                   max_batches=PROFILE['max_batches'],max_groups=PROFILE['max_groups'])
    supplied_selection=diagnostics.get('selection')
    _require(isinstance(supplied_selection,dict) and all(type(supplied_selection.get(k)) is int for k in
        ('eligible_groups','batch_size','max_batches','max_groups')) and supplied_selection==selection,'BATCH_SELECTION_POLICY_MISMATCH')
    batches=diagnostics.get('batches')
    expected_batches=[selected[i:i+PROFILE['batch_size']] for i in range(0,len(selected),PROFILE['batch_size'])]
    _require(isinstance(batches,list) and len(batches)==len(expected_batches)<=PROFILE['max_batches'],'BATCH_PARTITION_COUNT')
    _require(type(counts.get('batches_completed')) is int and counts['batches_completed']==len(batches),'BATCH_COMPLETION_COUNT')
    complete_statuses={'CONDITIONAL_DIAGNOSTIC_EVALUATED','NUMERIC_REFUSED','RADIAL_DOMAIN_REFUSED'}
    for i,(batch,identities) in enumerate(zip(batches,expected_batches)):
        _require(isinstance(batch,dict) and type(batch.get('batch_id')) is int and batch['batch_id']==i
            and batch.get('group_ids')==identities and batch.get('status')=='COMPLETE','BATCH_PARTITION_OR_STATUS')
        _require(type(batch.get('evaluated_groups')) is int and batch['evaluated_groups']==len(identities),'BATCH_EVALUATION_COUNT')
        source_groups=[by_id[ident] for ident in identities]
        _require(all(g['status'] in complete_statuses for g in source_groups),'BATCH_GROUP_NOT_EVALUATED')
        _require(isinstance(batch.get('group_statuses'),dict) and all(type(v) is int for v in batch['group_statuses'].values())
            and batch['group_statuses']==dict(Counter(g['status'] for g in source_groups)),'BATCH_GROUP_STATUS_COUNTS')
        _require(type(batch.get('loo_checks')) is int and batch['loo_checks']==sum(len(g['leave_one_view_out']) for g in source_groups),'BATCH_LOO_COUNT')
    for omitted_row in omitted:
        group=by_id[omitted_row['group_id']];_identity(group);n=len(group['admitted_observations'])
        expected_status=('NOT_EVALUATED_GROUP_BUDGET' if 2<=n<=12 else
                         'NOT_EVALUATED_OBSERVATION_BUDGET' if n>12 else 'INSUFFICIENT_ORIGINAL_ANCHORS')
        _require(group['status']==expected_status,'OMITTED_BUDGET_REASON_MISMATCH')
    rows=[]
    for group in groups:
        budget()
        if group['group_id'] in selected:rows.append(_classify(group,budget))
        else:rows.append(dict(group_id=group['group_id'],model_key=group.get('model_key'),track_id=group.get('track_id'),identity_segment=group.get('identity_segment'),
            tier='NOT_EVALUATED',reasons=[group['status']],xyz=None,metrics={},sources=[],
            branch_scope='CENTRAL_BRANCH_CONDITIONAL' if any(isinstance(o.get('params'),list) and len(o['params'])==4 and type(o['params'][3]) in (int,float) and o['params'][3]<0 for o in group['admitted_observations']) else 'MONOTONIC_RADIAL_CONDITIONAL',branch_scope_explanation='Not numerically selected; branch label does not imply evaluation',source_indices=[o.get('input_index') for o in group['admitted_observations']],frame_ids=[o.get('frame_id') for o in group['admitted_observations']],original_strict_domain_reference=deepcopy(group.get('strict_radial_domain_check',[])),**RIGHTS))
    return _json(dict(status='ANCHOR_CANDIDATES_CLASSIFIED',profile=PROFILE,rows=rows,
        batches=deepcopy(batches),selection=deepcopy(selection),
        selected_groups=deepcopy(selected),omitted_groups=deepcopy(omitted),
        counts=dict(groups=len(rows),selected_groups=len(selected),omitted_groups=len(omitted),batches_completed=len(batches),exportable=sum(r['tier'] in ('STRICT_CONDITIONAL','RELAXED_CONDITIONAL','TWO_VIEW_PROVISIONAL') for r in rows),tier_counts=dict(Counter(r['tier'] for r in rows))),
        evidence_scope='conditional fixed-camera candidates; no physical proof, native points or acceptance',**RIGHTS))
