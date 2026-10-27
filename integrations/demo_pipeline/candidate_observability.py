"""Conditional standard multiview geometry on existing cameras; no reconstruction."""
from collections import Counter
from itertools import combinations
import hashlib
import html
import json
from pathlib import Path
import numpy as np

from . import parallel_photo_audit as native
from . import approximate_photo_bridge as bridge
from .registration_audit import audit_registration
from kozlov_lab.stereo.baseline import multiview_fixed as fixed
from kozlov_lab.stereo.baseline import observability, radial_diagnostics as radial
from kozlov_lab.stereo.baseline.radial_branch import propose_central_domain


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _geometry(observations, *, domain_policy='existing twice-normalized-corner domain; refusal is domain-specific, not proof that every pixel has no inverse'):
    """Small pure-data diagnostic, including exactly three leave-one-view checks."""
    result=dict(status='CONDITIONAL_DIAGNOSTIC_REFUSED',fit=None,xyz=None,evaluation=None,
        observability=None,leave_one_view_out=[],positive_depths=None,
        radial_domains=[],
        identity_verified=False,geometry_acceptance=False,native_injected=False)
    try:
        for o in observations:
            f,cx,cy,k=map(float,o['params']);radius=o['max_ideal_radius']
            result['radial_domains'].append(dict(image_id=o['image_id'],k=k,
                declared_ideal_radius=radius,
                radial_derivative_at_bound=None if radius is None else float(1+3*k*radius*radius),
                positive_branch_turning_radius=None if k>=0 else float(np.sqrt(-1/(3*k))),
                measured_normalized_radius=float(np.linalg.norm((np.asarray(o['raw_pixels'])-[cx,cy])/f)),
                policy=domain_policy))
            radial._domain(float(o['params'][3]),o['max_ideal_radius'])
            radial.undistort_simple_radial(np.asarray(o['raw_pixels'],float)[None,:],o['params'],max_ideal_radius=o['max_ideal_radius'])
    except (ValueError,ArithmeticError) as exc:
        result.update(status='RADIAL_DOMAIN_REFUSED',reason=str(exc));return result
    fit=fixed.fit(observations,max_iterations=20);result['fit']=fit;result['xyz']=fit.get('xyz')
    if result['xyz'] is not None:
        xyz=result['xyz'];result['evaluation']=fixed.evaluate(xyz,observations)
        result['observability']=observability.analyze(xyz,observations)
        result['positive_depths']=[dict(image_id=o['image_id'],depth=float((np.asarray(o['R'])@xyz+o['t'])[2]),
            positive=bool((np.asarray(o['R'])@xyz+o['t'])[2]>0)) for o in observations]
        result['status']='CONDITIONAL_DIAGNOSTIC_EVALUATED'
    else:result['reason']=fit.get('reason')
    for heldout in range(3):
        train=[o for i,o in enumerate(observations) if i!=heldout];test=observations[heldout]
        fit_two=fixed.fit(train,max_iterations=20)
        result['leave_one_view_out'].append(dict(heldout_image_id=test['image_id'],training_image_ids=[o['image_id'] for o in train],
            fit=fit_two,prediction=None if fit_two.get('xyz') is None else fixed.evaluate(fit_two['xyz'],[test]),
            scope='Held-out measured pixel only; cameras are existing non-independent estimates.'))
    return result


def _central_branch_hypothesis(observations):
    """Additional conditional experiment; never replace the strict verdict.

    The derivative margin is fixed by the versioned radial helper, not selected
    using fit errors. Cameras, observations and original identity stay frozen.
    All fit and held-out projections enforce the same ideal-radius domain.
    """
    result=dict(policy_id='central_branch_hypothesis-v1',status='NOT_EVALUATED',
        branch_selection='ASSUMED_CENTRAL_BRANCH',physical_branch_verified=False,
        identity_verified=False,geometry_acceptance=False,native_injected=False,
        domains=[],observations=[],geometry=None)
    conditional=[]
    for o in observations:
        proposed=propose_central_domain(o['params'],o['image_width'],o['image_height'],[o['raw_pixels']])
        result['domains'].append(dict(image_id=o['image_id'],**proposed))
        conditional.append(dict(o,max_ideal_radius=proposed['max_ideal_radius']))
    result['observations']=conditional
    if not all(r['all_observations_invertible'] for r in result['domains']):
        result.update(status='CENTRAL_BRANCH_INPUT_REFUSED',reason='At least one observed pixel has no verified inverse in the fixed central branch domain.')
        return result
    result['geometry']=_geometry(conditional,domain_policy='central_branch_hypothesis-v1; fixed derivative floor; physical branch is assumed, not verified')
    result['status']=result['geometry']['status']
    return result


def review_candidate_observability(pipeline,features,frames,multiview,output,refs):
    output=Path(output).resolve()
    if output.exists():raise ValueError('fresh observability output required')
    local=[]
    def bind(path,declared=None):
        current=native.binding(path)
        if declared and any(current[k]!=declared[k] for k in ('bytes','sha256') if k in declared):raise ValueError('source binding differs')
        for old in refs+local:
            if Path(old['path']).resolve()==Path(current['path']) and any(old[k]!=current[k] for k in ('bytes','sha256')):
                raise ValueError('previous source binding differs')
        local.append(current);return current
    if not isinstance(pipeline,dict):bind(pipeline);pipeline=native.read_json(pipeline)
    run_path=Path(pipeline['stages']['sfm']['run_json'])
    if not run_path.is_absolute():raise ValueError('absolute recorded SfM run path required')
    run_path=run_path.resolve(strict=True);bind(run_path)
    root=run_path.parent
    if output.is_relative_to(root) or root.is_relative_to(output):raise ValueError('output overlaps immutable native source')
    run=native.read_json(run_path)
    if run.get('contract_version')!='sfm-native-object-v1' or run.get('status') not in ('SUCCESS','PARTIAL'):raise ValueError('usable object SfM run required')
    fmap=root/'frames_map.jsonl';bind(fmap)
    mapped={}
    for line in fmap.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        row=native.loads(line);index=row['input_index']
        if type(index) is not int or index in mapped:raise ValueError('duplicate or invalid frame-map identity')
        if not run.get('upstream_io_run_id') or row.get('upstream_io_run_id')!=run['upstream_io_run_id']:
            raise ValueError('native run/frame-map upstream identity differs')
        mapped[index]=row
    feature_frames=features['frames'] if isinstance(features,dict) else features
    by_frame={};feature_snapshot=[]
    for f in feature_frames:
        index=f['frame_index']
        if type(index) is not int or index in by_frame or not 0<=index<len(frames):raise ValueError('feature/frame identity differs')
        by_frame[index]=f
        feature_snapshot.append(dict(frame_index=index,input_index=f['input_index'],db_image_id=f['db_image_id'],
            image_name=f['image_name'],points=f['points']))
    feature_hash=_digest(feature_snapshot)
    closure_hash=_digest(multiview['closures'])
    eligible=[];excluded=[]
    seed_ids={e for c in multiview['seed_graph']['components'] for e in c['edge_ids']}
    for i,c in enumerate(multiview['closures']):
        if (c.get('descriptor_cycle_closed') is True and c.get('direct_geometry',{}).get('status')=='EXISTING_F_SUPPORT_CANDIDATE'
            and len(c.get('two_edge_ids',[]))==2 and all(e in seed_ids for e in c['two_edge_ids'])):eligible.append(i)
        else:excluded.append(i)
    selected=eligible if len(eligible)<=32 else [eligible[i*(len(eligible)-1)//31] for i in range(32)]
    cache={};cases=[];missing_pose_inputs=set()
    for closure_index in selected:
        c=multiview['closures'][closure_index];nodes=sorted([c['endpoint_a'],c['center'],c['endpoint_b']],key=lambda n:(n['frame'],n['feature_id']))
        if len({n['frame'] for n in nodes})!=3:raise ValueError('three distinct candidate frames required')
        case=dict(closure_index=closure_index,wedge_index=c.get('wedge_index'),nodes=nodes,
            observed_descriptor_competition=c.get('observed_descriptor_competition'),competition=c.get('competition',[]),
            identity_verified=False,geometry_acceptance=False,native_injected=False,uniqueness='NOT_EVALUATED')
        identities=[];common=None
        for node in nodes:
            f=by_frame[node['frame']];source=frames[node['frame']];m=mapped.get(f['input_index'])
            if m is None or source['input_index']!=f['input_index'] or source['frame_id']!=m['frame_id'] or source['sha256']!=m['canonical_image_sha256']:
                raise ValueError('source/map/native frame identity differs')
            if f['db_image_id']!=m['database_image_id'] or f['image_name']!=m['staged_image_name']:raise ValueError('native image identity differs')
            bind(source['path'],{'sha256':source['sha256']});bind(native.under(root,m['staged_image_name']),{'sha256':source['sha256']})
            points=[p for p in f['points'] if p['id']==node['feature_id']]
            if len(points)!=1:raise ValueError('unique native feature identity required')
            point=points[0]
            if type(point['feature_row']) is not int or point['feature_row']<0 or point['id']!=str(f['db_image_id'])+':'+str(point['feature_row']):raise ValueError('native feature row identity differs')
            registrations={str(r['model_id']):r for r in m['registrations']}
            if len(registrations)!=len(m['registrations']):raise ValueError('duplicate model registration')
            common=set(registrations) if common is None else common&set(registrations)
            identities.append((f,m,point,registrations))
        if not common:
            missing_pose_inputs.update(f['input_index'] for f,m,point,registrations in identities if not registrations)
            case.update(status='MISSING_COMMON_MODEL_POSES',registration_model_ids=[sorted(v[3]) for v in identities]);cases.append(case);continue
        if len(common)!=1:
            case.update(status='AMBIGUOUS_COMMON_MODELS',common_model_ids=sorted(common));cases.append(case);continue
        model_id=next(iter(common));models=[m for m in run['models'] if str(m['model_id'])==model_id]
        if len(models)!=1 or models[0].get('usable') is not True or models[0].get('roundtrip_verified') is not True:raise ValueError('unique usable roundtrip model required')
        if model_id not in cache:
            model=models[0];binary=native.under(root,model['binary_path']);paths={name:native.under(root,str((binary/name).relative_to(root))) for name in ('cameras.bin','images.bin','points3D.bin')}
            for path in paths.values():bind(path)
            images,points=native.read_native(paths['images.bin'],paths['points3D.bin'])
            cameras,poses=bridge.read_calibration(paths['cameras.bin'],paths['images.bin'])
            if sorted(images)!=sorted(model['registered_image_ids']) or set(poses)!=set(images) or len(points)!=model['num_points3D']:raise ValueError('native model identity/count differs')
            cache[model_id]=(images,cameras,poses)
        images,cameras,poses=cache[model_id];observations=[]
        for f,m,point,registrations in identities:
            reg=registrations[model_id];ident=f['db_image_id'];pose=poses.get(ident);image=images.get(ident)
            if pose is None or image is None or reg['image_id']!=ident or reg['camera_id']!=pose['camera_id'] or image['camera_id']!=pose['camera_id'] or pose['name']!=f['image_name'] or image['name']!=f['image_name']:raise ValueError('native pose/name/camera binding differs')
            row=point['feature_row'];xy=np.asarray(point['xy'],dtype=float)
            if row>=len(image['observations']) or xy.shape!=(2,) or not np.isfinite(xy).all() or not np.array_equal(xy,np.asarray(image['observations'][row][:2])):raise ValueError('native feature row/xy differs')
            camera=cameras[pose['camera_id']]
            observations.append(dict(image_id=ident,frame_index=f['frame_index'],feature_id=point['id'],point2D_index=row,
                raw_pixels=xy.tolist(),params=list(camera['params']),R=np.asarray(pose['R']).tolist(),t=np.asarray(pose['t']).tolist(),max_ideal_radius=camera['radius'],
                image_width=camera['width'],image_height=camera['height']))
        case.update(model_id=model_id,observations=observations,**_geometry(observations))
        if case['status']=='RADIAL_DOMAIN_REFUSED':
            case['central_branch_hypothesis']=_central_branch_hypothesis(observations)
        cases.append(case)
    # Rebuild feature identities after computation: mutable input changes refuse publication.
    current=[dict(frame_index=f['frame_index'],input_index=f['input_index'],db_image_id=f['db_image_id'],image_name=f['image_name'],points=f['points']) for f in feature_frames]
    if _digest(current)!=feature_hash or _digest(multiview['closures'])!=closure_hash:raise ValueError('candidate/feature identities changed during diagnostic')
    # The caller owns its heterogeneous binding list (including a potentially
    # >64 MiB feature database). Recheck our own inputs with our own 64 MiB
    # reader; the parent checks all inherited bindings with its original policy.
    for ref in local:bridge.verify_binding(ref)
    # Existing evidence only. A missing pose triggers a bounded, separate audit;
    # it never triggers registration, camera updates, or a two-view fallback.
    output.mkdir(parents=True,exist_ok=False)
    registration_audits=[]
    missing=sorted(missing_pose_inputs)
    selected_missing=missing if len(missing)<=8 else [missing[i*(len(missing)-1)//7] for i in range(8)]
    for index in selected_missing:
        try:
            audit=audit_registration(run_path,index,output/('registration_'+str(index)),refs)
            registration_audits.append(dict(input_index=index,status=audit['status'],counts=audit['counts'],
                failure_cause=audit['registration_failure_cause'],report='registration_'+str(index)+'/REGISTRATION.json'))
        except Exception as exc:
            registration_audits.append(dict(input_index=index,status='AUDIT_FAILED',reason=type(exc).__name__+': '+str(exc),failure_cause='UNKNOWN'))
    result=dict(schema='candidate-observability-v2',status='CONDITIONAL_DIAGNOSTIC_COMPLETE',cases=cases,
        registration_audits=registration_audits,registration_audit_profile=dict(max_missing_frames=8,selected=selected_missing,
            omitted=[i for i in missing if i not in selected_missing],native_registration_executions=0),
        counts=dict(eligible=len(eligible),selected=len(selected),omitted=len(eligible)-len(selected),statuses=dict(Counter(c['status'] for c in cases))),
        additional_branch_counts=dict(Counter(c['central_branch_hypothesis']['status'] for c in cases if 'central_branch_hypothesis' in c)),
        selected_closure_indices=selected,omitted_closure_indices=[i for i in eligible if i not in selected],excluded_closure_indices=excluded,
        feature_identity_sha256=feature_hash,feature_identities=feature_snapshot,closure_sha256=closure_hash,source_bindings=local,
        radial_domain_policy='Existing bridge policy: twice maximum normalized image-corner radius; require strict monotonicity on the entire declared domain; otherwise refuse.',
        algorithm='standard fixed-camera multiview geometry; not Kozlov algorithm',camera_independence=False,
        identity_verified=False,geometry_acceptance=False,native_injected=False,uniqueness='NOT_EVALUATED',physical_scale='unknown',
        limitations=['All XYZ and residuals are conditional on the proposed identities and existing cameras.',
            'Camera estimates are not independent truth; held-out pixels do not make cameras independent.',
            'Descriptor competition is retained; no candidate becomes accepted from low residuals.'])
    bridge.write(output/'OBSERVABILITY.json',result)
    rows=''.join('<tr><td>'+str(c['closure_index'])+'</td><td>'+html.escape(c['status'])+'</td><td>'+html.escape(str(c.get('model_id','—')))+'</td><td>'+html.escape(c.get('central_branch_hypothesis',{}).get('status','NOT_EVALUATED'))+'</td></tr>' for c in cases)
    (output/'index.html').write_text('<meta charset="utf-8"><title>Conditional candidate observability</title><h1>条件化三视图可观测性</h1><p>标准多视几何；相机与对应身份均非独立真值，无几何验收或毫米精度结论。中心分支是单列的条件假设，严格原域拒绝不被覆盖。</p><table><tr><th>候选</th><th>原域状态</th><th>原模型</th><th>附加中心分支假设</th></tr>'+rows+'</table><p><a href="OBSERVABILITY.json">完整数值与来源</a></p>',encoding='utf-8')
    for ref in local:bridge.verify_binding(ref)
    refs.extend(local)
    return result
