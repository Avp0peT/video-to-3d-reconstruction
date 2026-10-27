"""Immutable original-observation diagnostics; never correspondence certification."""
from collections import Counter
from pathlib import Path
import math
import hashlib
import time

from . import parallel_photo_audit as native
from . import approximate_photo_bridge as bridge
from .heldout_multiview import _observation
from .multi_pair_ablation import _bind_photos
from kozlov_lab.stereo.baseline import multiview_fixed as fixed

SOFT_SECONDS = 180
METHODS = ('C', 'J', 'J_conditional')


def _output_binding(path):
    """Own output may exceed the immutable-input reader's 64 MiB limit."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''): digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def _finite(value):
    if isinstance(value,float) and not math.isfinite(value):
        raise ValueError('nonfinite diagnostic value')
    if isinstance(value,dict):
        for v in value.values(): _finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value: _finite(v)


def _metric(xyz, observations):
    return fixed.evaluate(xyz, observations) if xyz is not None and observations else dict(
        status='NOT_EVALUATED', max=None, rms=None, residual_norms=[], residual_vectors=[])


def _status(metric):
    if metric['status'] != 'EVALUATED': return 'fit_refused'
    return 'inconsistency' if metric['max'] > 2 else 'consistent_not_verified'


def audit_point(point, *, run_fit=True):
    _finite(point)
    obs = point['observations']; pair = point['image_pair']
    ids = [o['image_id'] for o in obs]
    if len(set(ids)) != len(ids) or len(set(pair)) != 2 or not set(pair).issubset(ids):
        raise ValueError('unique singleton observations and existing C pair required')
    extras = sorted(set(ids)-set(pair)); heldout_ids = extras[::2]
    train_ids = sorted(set(pair) | set(extras[1::2]))
    train = [o for o in obs if o['image_id'] in train_ids]
    heldout = [o for o in obs if o['image_id'] in heldout_ids]
    ambiguous = point['ambiguous_observations']
    fitted = dict(status='NOT_RUN_BUDGET', xyz=None)
    if run_fit and extras:
        fitted = fixed.fit(train, max_iterations=20)
    elif run_fit:
        fitted = dict(status='INSUFFICIENT_EXTRA_VIEWS', xyz=None)
    xyzs = dict(C=point['C_xyz'], J=point['J_xyz'], J_conditional=fitted.get('xyz'))
    methods = {}
    for name, xyz in xyzs.items():
        metrics = dict(full=_metric(xyz,obs), train=_metric(xyz,train), heldout=_metric(xyz,heldout))
        status = _status(metrics['full'])
        if name == 'C' and xyz is None: status = 'NOT_AVAILABLE'
        elif name == 'J_conditional' and not run_fit: status = 'NOT_RUN_BUDGET'
        elif ambiguous: status = 'ambiguous'
        elif not extras: status = 'insufficient'
        methods[name] = dict(status=status, xyz=xyz, **metrics)
    views = []
    for o in obs:
        view = dict(o, measured_xy=o['raw_pixels'], role='heldout' if o['image_id'] in heldout_ids else 'train', projections={})
        for name, xyz in xyzs.items():
            try:
                if xyz is None: raise ValueError('candidate unavailable')
                projection = fixed.project(xyz,o)
                view['projections'][name] = dict(status='EVALUATED', projected_xy=projection['pixels'], reason=None)
            except (ValueError, ArithmeticError, KeyError, TypeError) as exc:
                view['projections'][name] = dict(status='PROJECTION_REFUSED', projected_xy=None, reason=str(exc))
        views.append(view)
    row = bridge.json_value(dict(point_id=point['point_id'], status=methods['J_conditional']['status'],
        C_pair_image_ids=pair, support_count=len(obs), ambiguous_observations=ambiguous,
        train_image_ids=train_ids, heldout_image_ids=heldout_ids, methods=methods,
        conditional_fit=fitted, views=views, original_track_elements=point['original_track_elements']))
    _finite(row)
    return row


def summarize(rows):
    summary = dict(support_histogram=dict(Counter(str(r['support_count']) for r in rows)),
                   statuses=dict(Counter(r['status'] for r in rows)), methods={}, paired={})
    for name in METHODS:
        summary['methods'][name] = dict(statuses=dict(Counter(r['methods'][name]['status'] for r in rows)))
        for bucket in ('full','heldout'):
            metrics = [r['methods'][name][bucket] for r in rows]
            evaluated = [m for m in metrics if m['status']=='EVALUATED']
            summary['methods'][name][bucket] = dict(statuses=dict(Counter(m['status'] for m in metrics)),
                thresholds={str(t):dict(evaluated=len(evaluated), within=sum(m['max']<=t for m in evaluated),
                                       exceeds=sum(m['max']>t for m in evaluated)) for t in (1,2,4)})
    for bucket in ('full','heldout'):
        eligible = [r for r in rows if all(r['methods'][m][bucket]['status']=='EVALUATED' for m in METHODS)]
        summary['paired'][bucket] = dict(count=len(eligible), thresholds={str(t):{
            m:dict(within=sum(r['methods'][m][bucket]['max']<=t for r in eligible),
                   exceeds=sum(r['methods'][m][bucket]['max']>t for r in eligible)) for m in METHODS} for t in (1,2,4)})
        comparison=('J','J_conditional')
        paired=[r for r in rows if all(r['methods'][m][bucket]['status']=='EVALUATED' for m in comparison)]
        summary['paired']['J_vs_J_conditional_'+bucket]=dict(count=len(paired),thresholds={str(t):{
            m:dict(within=sum(r['methods'][m][bucket]['max']<=t for r in paired),
                   exceeds=sum(r['methods'][m][bucket]['max']>t for r in paired)) for m in comparison} for t in (1,2,4)})
    return summary


def audit(previous_comparison_run, new_output):
    previous = Path(previous_comparison_run).resolve(strict=True); output = Path(new_output).resolve()
    if output.exists() or output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError('fresh nonoverlapping output required')
    paths = [previous/'surface_inputs'/m/'POINT_IDENTITY.json' for m in ('C','J')]
    paths += [previous/'foreground/SHARED_FOREGROUND_POINTS.json', previous/'expanded/PLAN.json']
    refs = [native.binding(p) for p in paths]
    manifest = previous/'SHA256SUMS.txt'
    if manifest.exists():
        entries = {}
        for line in manifest.read_text(encoding='utf-8-sig').splitlines():
            digest, relative = line.split('  ',1)
            if relative in entries: raise ValueError('duplicate sealed manifest path')
            entries[relative.replace('\\','/')] = digest
        for path, ref in zip(paths,refs):
            if entries.get(path.relative_to(previous).as_posix()) != ref['sha256']:
                raise ValueError('sealed previous comparison evidence differs')
        refs.append(native.binding(manifest))
    else:
        raise ValueError('sealed previous comparison SHA256SUMS required')
    identities = [native.loads(p.read_text(encoding='utf-8-sig')) for p in paths[:2]]
    if identities[0] != identities[1]: raise ValueError('C/J identity ledgers differ')
    requested = [v['point_id'] for v in identities[0]]
    if len(set(requested)) != len(requested): raise ValueError('duplicate requested point ID')
    plan = native.read_json(paths[3]); refs += plan['source_bindings']
    refs += [f['binding'] for f in plan['frames']]
    unique_refs = {}
    for ref in refs:
        if ref['path'] in unique_refs and unique_refs[ref['path']] != ref:
            raise ValueError('conflicting source bindings')
        unique_refs[ref['path']] = ref
    refs = list(unique_refs.values())
    for ref in refs: bridge.verify_binding(ref)
    native_paths = {Path(r['path']).name:Path(r['path']) for r in refs if Path(r['path']).name in ('images.bin','points3D.bin','cameras.bin')}
    images, points = native.read_native(native_paths['images.bin'],native_paths['points3D.bin'])
    cameras, poses = bridge.read_calibration(native_paths['cameras.bin'],native_paths['images.bin'])
    records = native.read_json(paths[2])['points']; by_id = {p['point_id']:p for p in records}
    if len(by_id)!=len(records) or not set(requested).issubset(by_id): raise ValueError('point request ledger mismatch')
    selected = [by_id[i] for i in requested]
    for p in selected:
        pid = p['point_id']; expected = [bridge.json_value(_observation(i,pid,images,points,cameras,poses)) for i in sorted(points[pid]['track'])]
        if len(expected)!=len(p['observations']): raise ValueError('native observation coverage differs')
        for old, actual in zip(p['observations'],expected):
            if any(old[k]!=actual[k] for k in actual): raise ValueError('native reciprocal pixel/calibration identity differs')
            if old['raw_binary64_hex'] != [float(v).hex() for v in actual['raw_pixels']]: raise ValueError('raw pixel hex differs')
        if p['ambiguous_observations'] != points[pid]['ambiguous_observations']: raise ValueError('ambiguity ledger differs')
        expected_elements = [(i,j) for i,j in sorted(points[pid]['track_elements'])]
        if [(e['image_id'],e['point2D_index']) for e in p['original_track_elements']] != expected_elements: raise ValueError('native element coverage differs')
        for e in p['original_track_elements']:
            xy = list(images[e['image_id']]['observations'][e['point2D_index']][:2])
            if e['raw_pixels']!=xy or e['raw_binary64_hex']!=[float(v).hex() for v in xy]: raise ValueError('native element pixels differ')
    for ref in refs: bridge.verify_binding(ref)
    output.mkdir(parents=True,exist_ok=False)
    bridge.write(output/'PLAN.json',dict(schema='correspondence-audit-plan-v1',source_bindings=refs,
        frame_bindings=plan['frames'],
        requested_point_ids=requested, soft_seconds=SOFT_SECONDS, max_fit_iterations=20,
        split_rule='C pair fixed train; sorted extras[::2] heldout; extras[1::2] train',
        conditional_not_independent=True, old_colmap_used_all_observations=True,
        diagnostic_thresholds_pixels=[1,2,4], correspondence_verified=False))
    deadline = time.monotonic()+SOFT_SECONDS
    rows = [audit_point(p,run_fit=time.monotonic()<deadline) for p in selected]
    if [r['point_id'] for r in rows]!=requested: raise ValueError('exact request ledger differs')
    bridge.write(output/'TRACKS.json',dict(schema='correspondence-audit-tracks-v1',tracks=rows))
    for ref in refs: bridge.verify_binding(ref)
    result = dict(schema='correspondence-audit-result-v1',status='DIAGNOSTICS_NOT_CORRESPONDENCE_VERIFIED',
        requested_count=len(requested), emitted_count=len(rows), summary=summarize(rows),
        SOURCE_immutable=True, exact_request_ledger=True, conditional_not_independent=True,
        correspondence_verified=False, geometry_acceptance=False,
        interpretations=dict(C='old two-view proxy candidate checked against original observations',
            J='old all-view fitted candidate; residual only self consistency',
            J_conditional='train-only XYZ refit; heldout pixels excluded; old calibration used all views'),
        artifacts=[_output_binding(output/n) for n in ('PLAN.json','TRACKS.json')])
    bridge.write(output/'RESULT.json',result)
    return result


def audit_native(sfm_run, model_id, new_output):
    """Explicit native component; standard fixed-camera fits, no C construction."""
    source=Path(sfm_run).resolve(strict=True); output=Path(new_output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('fresh nonoverlapping output required')
    if not isinstance(model_id,str) or not model_id: raise ValueError('explicit model ID required')
    run=native.read_json(source/'run.json')
    models=[m for m in run['models'] if str(m['model_id'])==model_id]
    if len(models)!=1 or not models[0]['usable'] or not models[0]['roundtrip_verified']:
        raise ValueError('explicit usable roundtrip-verified model required')
    binary=native.under(source,models[0]['binary_path'])
    names={n:binary/n for n in ('images.bin','points3D.bin','cameras.bin')}
    names.update({'frames_map.jsonl':source/'frames_map.jsonl','run.json':source/'run.json'})
    refs=[native.binding(p) for p in names.values()]
    images,points=native.read_native(names['images.bin'],names['points3D.bin'])
    cameras,poses=bridge.read_calibration(names['cameras.bin'],names['images.bin'])
    if sorted(models[0]['registered_image_ids'])!=sorted(images) or models[0]['num_points3D']!=len(points):
        raise ValueError('native model identity/count mismatch')
    frames=_bind_photos(source,run,images,[dict(image_pair=sorted(images))],names)
    refs.extend(f['binding'] for f in frames)
    for ref in refs: bridge.verify_binding(ref)
    ids=sorted(points); selected=ids[:10000]
    output.mkdir(parents=True,exist_ok=False)
    bridge.write(output/'PLAN.json',dict(schema='correspondence-audit-plan-v1',mode='native',
        source_model_id=model_id,source_bindings=refs,frame_bindings=frames,
        requested_point_ids=selected,not_sampled_point_ids=ids[10000:],max_tracks=10000,
        soft_seconds=SOFT_SECONDS,max_fit_iterations=20,
        split_rule='ascending native IDs; first two train; remaining extras[::2] heldout, extras[1::2] train',
        old_xyz_used=False,C_algorithm_called=False,conditional_not_independent=True,
        old_colmap_used_all_observations=True,diagnostic_thresholds_pixels=[1,2,4],
        correspondence_verified=False))
    deadline=time.monotonic()+SOFT_SECONDS; rows=[]
    for pid in selected:
        point=points[pid]
        obs=[bridge.json_value(_observation(i,pid,images,points,cameras,poses)) for i in sorted(point['track'])]
        for o in obs: o['raw_binary64_hex']=[float(v).hex() for v in o['raw_pixels']]
        elements=[]
        for i,j in sorted(point['track_elements']):
            xy=list(images[i]['observations'][j][:2])
            elements.append(dict(image_id=i,point2D_index=j,raw_pixels=xy,raw_binary64_hex=[float(v).hex() for v in xy]))
        available=time.monotonic()<deadline
        joint=fixed.fit(obs,max_iterations=20) if available and len(obs)>=2 else dict(status='NOT_RUN_BUDGET' if not available else 'INSUFFICIENT_VIEWS',xyz=None)
        data=dict(point_id=pid,observations=obs,image_pair=[o['image_id'] for o in obs[:2]],
            C_xyz=None,J_xyz=joint.get('xyz'),ambiguous_observations=point['ambiguous_observations'],original_track_elements=elements)
        if len(obs)>=2:
            row=audit_point(data,run_fit=time.monotonic()<deadline)
        else:
            row=dict(point_id=pid,status='insufficient' if available else 'NOT_RUN_BUDGET',support_count=len(obs),
                C_pair_image_ids=[],train_image_ids=[o['image_id'] for o in obs],heldout_image_ids=[],
                ambiguous_observations=point['ambiguous_observations'],original_track_elements=elements,views=[],methods={})
            for m in METHODS: row['methods'][m]=dict(status='NOT_AVAILABLE' if m=='C' else row['status'],xyz=None,
                full=_metric(None,obs),train=_metric(None,obs),heldout=_metric(None,[]))
            for o in obs: row['views'].append(dict(o,measured_xy=o['raw_pixels'],role='train',projections={m:dict(status='PROJECTION_REFUSED',projected_xy=None,reason='insufficient singleton views') for m in METHODS}))
        row['all_view_fit']=bridge.json_value(joint); row['baseline_pair_image_ids']=row.pop('C_pair_image_ids')
        row['mode']='native'; _finite(row); rows.append(row)
    if [r['point_id'] for r in rows]!=selected: raise ValueError('exact native request ledger differs')
    bridge.write(output/'TRACKS.json',dict(schema='correspondence-audit-tracks-v1',tracks=rows))
    for ref in refs: bridge.verify_binding(ref)
    result=dict(schema='correspondence-audit-result-v1',mode='native',source_model_id=model_id,
        status='DIAGNOSTICS_NOT_CORRESPONDENCE_VERIFIED',requested_count=len(selected),emitted_count=len(rows),
        not_sampled_count=len(ids)-len(selected),summary=summarize(rows),SOURCE_immutable=True,
        exact_request_ledger=True,conditional_not_independent=True,old_xyz_used=False,
        C_algorithm_called=False,correspondence_verified=False,geometry_acceptance=False,
        interpretations=dict(C='NOT_AVAILABLE; no paper construction executed',J='standard all-view fixed-camera fit; self consistency',
            J_conditional='train-only XYZ refit; heldout excluded; existing calibration used all photos'),
        artifacts=[_output_binding(output/n) for n in ('PLAN.json','TRACKS.json')])
    bridge.write(output/'RESULT.json',result)
    return result
