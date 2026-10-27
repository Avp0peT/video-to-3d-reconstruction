"""Isolated reverse candidates, admitted adjacent endpoint segments only."""
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from . import region_tracks as rt
from . import region_anchor as anchor

PROFILE = dict(id='glove-region-reverse-v1', tracking=deepcopy(rt.PROFILE),
               anchor=deepcopy(anchor.PROFILE), minimum_segment_endpoints=2,
               unknown_policy='split_without_claiming_mismatch')


def split_segments(observations, admitted_edges):
    """Split one candidate at unknown/missing edges; never merge candidate IDs."""
    segments = []; current = []
    for obs in observations:
        if current and (current[-1]['decode_index'], obs['decode_index']) not in admitted_edges:
            if len(current) >= 2:
                segments.append(current)
            current = []
        current.append(obs)
    if len(current) >= 2:
        segments.append(current)
    return segments


def run(plan_path, forward_tracks_path, output):
    """Keep every original endpoint observation; append independent reverse IDs."""
    refs = []; seen = {}
    def bind(path, expected=None):
        path = Path(path).resolve(strict=True); blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if expected is not None and expected != digest:
            raise ValueError('SHA256 mismatch: '+str(path))
        if str(path) in seen and seen[str(path)] != digest:
            raise ValueError('input changed: '+str(path))
        if str(path) not in seen:
            refs.append(dict(path=str(path), sha256=digest, bytes=len(blob)))
            seen[str(path)] = digest
        return blob
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(bind(plan_path))
    forward = json.loads(bind(forward_tracks_path))
    for path in (Path(__file__), Path(anchor.__file__), Path(rt.__file__)):
        bind(path)
    if plan.get('schema') != 'glove-region-plan-v1' or plan.get('profile', rt.PROFILE) != rt.PROFILE:
        raise ValueError('frozen glove region plan required')
    density_policy = plan.get('density_policy')
    if density_policy is not None:
        from .density_sampling import validate_policy, seed_with_density
        validate_policy(density_policy)
        for name in ('density_sampling.py','point_density.py'):
            bind(Path(__file__).with_name(name))
    if forward.get('schema') != 'glove-region-tracks-v1' or forward.get('coordinate_convention') != 'opencv_integer_centers':
        raise ValueError('compatible original tracks required')
    original_binding = forward['source_bindings'][0]
    if Path(original_binding['path']).resolve() != plan_path or original_binding['sha256'] != refs[0]['sha256']:
        raise ValueError('original tracks plan binding differs')
    timeline = plan['timeline']
    indices = [r['decode_index'] for r in timeline]
    if not indices or any(type(i) is not int or i < 0 for i in indices) or indices != list(range(indices[0], indices[-1]+1)):
        raise ValueError('complete consecutive timeline required')
    endpoints = [r for r in timeline if r.get('input_index') is not None]
    inputs = [r['input_index'] for r in endpoints]
    if any(type(i) is not int or i < 0 for i in inputs) or any(a >= b for a,b in zip(inputs,inputs[1:])):
        raise ValueError('ordered unique input_index required')
    if len(endpoints) != len(forward['frames']):
        raise ValueError('endpoint count differs')
    for row, old in zip(endpoints, forward['frames']):
        for field in ('input_index','decode_index','image_sha256','mask_sha256','width','height'):
            if row[field] != old[field]:
                raise ValueError('endpoint differs: '+field)
        for field in ('image_path','mask_path'):
            if Path(row[field]).resolve() != Path(old[field]).resolve():
                raise ValueError('endpoint path differs')
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    def record(value):
        with (out/'progress.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n')
    record(dict(event='START',profile=PROFILE,timeline_frames=len(timeline)))
    cv = rt.load_cv(refs)
    if cv is None:
        raise RuntimeError('bound CPU OpenCV required')
    cv.ocl.setUseOpenCL(False); cv.setNumThreads(2); cv.setRNGSeed(0)
    shape = (timeline[0]['height'],timeline[0]['width'])
    def load(row):
        arrays=[]
        for kind in ('image','mask'):
            blob=bind(row[kind+'_path'],row[kind+'_sha256'])
            with Image.open(io.BytesIO(blob)) as im:
                if kind=='mask':
                    if im.mode!='L':
                        raise ValueError('mask must be mode L')
                    array=np.asarray(im).copy()
                    if not np.isin(array,[0,255]).all():
                        raise ValueError('mask must contain 0/255 labels')
                else:
                    if im.mode not in ('RGB','L'):
                        raise ValueError('RGB/L image required')
                    array=np.asarray(im.convert('L')).copy()
            if array.shape!=shape or array.shape!=(row['height'],row['width']):
                raise ValueError('same grid required')
            arrays.append(array)
        return arrays
    print('REVERSE_TRACKING_START',flush=True)
    active={}; candidates={}; candidate_id=0; previous=previous_mask=None
    endpoint_observations={}; timeline_stats=[]
    for row in reversed(timeline):
        gray,mask=load(row); lost=[]; births=[]; attempted=len(active)
        if previous is not None and active:
            ids=list(active)
            q,reasons,_=rt.step(previous,gray,previous_mask,mask,list(active.values()),cv)
            for n,ident in enumerate(ids):
                if reasons[n]=='accepted':
                    active[ident]=q[n]
                else:
                    del active[ident]; lost.append(dict(reverse_candidate_id=ident,reason=reasons[n]))
        if density_policy is not None:
            new_points, density = seed_with_density(gray,mask,list(active.values()),cv,density_policy)
        elif previous is None or row.get('input_index') is not None:
            new_points = rt.seed(gray,mask,list(active.values()),cv)
        else:
            new_points = []
        for xy in new_points:
            active[candidate_id]=xy; candidates[candidate_id]=[]; births.append(candidate_id); candidate_id+=1
        if row.get('input_index') is not None:
            obs=[]
            for ident,xy in active.items():
                point=dict(input_index=row['input_index'],decode_index=row['decode_index'],xy=[float(v) for v in xy])
                candidates[ident].append(point)
                obs.append(dict(track_id=ident,xy=point['xy']))
            endpoint_observations[row['decode_index']]=obs
        stat=dict(event='REVERSE_FRAME',decode_index=row['decode_index'],attempted=attempted,
                  accepted=attempted-len(lost),active=len(active),births=births,lost=lost,
                  reject_reasons=dict(Counter(r['reason'] for r in lost)))
        timeline_stats.append(stat); record(stat)
        previous,previous_mask=gray,mask
    previous=previous_mask=None
    print('REVERSE_ANCHOR_AUDIT_START',flush=True)
    audits=[]; edges={ident:set() for ident in candidates}
    for left,right in zip(endpoints,endpoints[1:]):
        a,am=load(left); b,bm=load(right)
        audit=anchor.audit_pair(a,b,am,bm,endpoint_observations[left['decode_index']],endpoint_observations[right['decode_index']],cv)
        audit.update(source_decode_index=left['decode_index'],target_decode_index=right['decode_index'])
        audits.append(audit)
        for row in audit['candidates']:
            if row['status']=='admitted':
                edges[row['track_id']].add((left['decode_index'],right['decode_index']))
        record(dict(event='ANCHOR_PAIR',source_decode_index=left['decode_index'],target_decode_index=right['decode_index'],metrics=audit['metrics']))
    result=deepcopy(forward)
    old_ids=[t['track_id'] for t in forward['tracks']]+[o['track_id'] for f in forward['frames'] for o in f['observations']]
    if any(type(i) is not int or i<0 for i in old_ids):
        raise ValueError('original IDs must be nonnegative integers')
    next_id=max(old_ids,default=-1)+1; first_new_id=next_id
    frame_by_decode={f['decode_index']:f for f in result['frames']}
    segments=[]; candidate_decisions=[]
    for ident,observations in candidates.items():
        ordered=sorted(observations,key=lambda o:o['decode_index'])
        retained=split_segments(ordered,edges[ident])
        used={p['decode_index'] for segment in retained for p in segment}
        candidate_decisions.append(dict(reverse_candidate_id=ident,
                                        endpoint_decode_indices=[p['decode_index'] for p in ordered],
                                        retained_segments=len(retained),
                                        omitted_endpoint_decode_indices=[p['decode_index'] for p in ordered if p['decode_index'] not in used],
                                        omission_reason='No incident admitted adjacent endpoint edge; singleton not exported. UNKNOWN does not prove wrong.'))
        for segment in retained:
            track=dict(track_id=next_id,observations=deepcopy(segment),origin='reverse_anchor_validated_candidate',reverse_candidate_id=ident)
            result['tracks'].append(track)
            for point in segment:
                frame_by_decode[point['decode_index']]['observations'].append(dict(track_id=next_id,xy=point['xy']))
            segments.append(dict(track_id=next_id,reverse_candidate_id=ident,decode_indices=[p['decode_index'] for p in segment]))
            next_id+=1
    result['source_bindings']=refs
    result['reverse_track_ids']=[s['track_id'] for s in segments]
    result['reverse_supplement']=dict(profile=PROFILE,first_new_id=first_new_id,new_tracks=len(segments),original_forward_identity_newly_verified=False,
                                     identity_verified=False,geometry_accepted=False,
                                     direct_validation_semantics='Direct patch evidence is candidate admission, not physical identity truth.')
    result['metrics']=dict(forward.get('metrics',{}),tracks=len(result['tracks']),observations=sum(len(f['observations']) for f in result['frames']),reverse_new_tracks=len(segments))
    audit_result=dict(schema='glove-region-reverse-audit-v1',profile=PROFILE,source_bindings=refs,
                      timeline_stats=timeline_stats,endpoint_pair_audits=audits,added_segments=segments,
                      candidate_decisions=candidate_decisions,
                      candidate_count=len(candidates),original_forward_identity_newly_verified=False,
                      unknown_semantics='Unobservable or inconsistent direct evidence; not proved wrong correspondence.')
    for name,doc in (('REGION_TRACKS.json',result),('REVERSE_AUDIT.json',audit_result)):
        (out/name).write_text(json.dumps(doc,indent=2,allow_nan=False),encoding='utf8')
    record(dict(event='FINAL',status='MEASURED_NOT_GEOMETRY_ACCEPTED',new_tracks=len(segments)))
    print('REVERSE_COMPLETE new_tracks='+str(len(segments)),flush=True)
    return result
