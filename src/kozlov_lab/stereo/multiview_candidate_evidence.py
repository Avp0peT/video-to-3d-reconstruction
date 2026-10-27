"""Frozen raw-pixel candidate evidence, never matching truth or correction."""
from copy import deepcopy
from itertools import combinations
import math
import numpy as np
from .viewpoint_consistency import fit_fundamental, _residual


def _tracks(tracks):
    if type(tracks) is not dict or not 1 <= len(tracks) <= 4096:
        raise ValueError('1..4096 explicitly identified tracks required')
    images=set()
    for pid,views in tracks.items():
        if type(pid) is not int or pid<=0 or type(views) is not dict or not views:
            raise ValueError('positive integer track IDs and nonempty view maps required')
        for iid,xy in views.items():
            if type(iid) is not int or iid<=0 or not isinstance(xy,(list,tuple)) or len(xy)!=2:
                raise ValueError('positive integer image IDs and two real coordinates required')
            if any(type(v) not in (int,float) or abs(v)>1e9 or not math.isfinite(v) for v in xy):
                raise ValueError('finite bounded real pixels required; bool rejected')
            images.add(iid)
    if len(images)>12:raise ValueError('at most twelve views required')
    return deepcopy(tracks)


def _groups(tracks):
    parent={pid:pid for pid in tracks};seen={}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for pid in sorted(tracks):
        for iid,xy in sorted(tracks[pid].items()):
            key=(iid,tuple(xy))
            if key in seen:
                a,b=find(pid),find(seen[key]);parent[max(a,b)]=min(a,b)
            else:seen[key]=pid
    groups={}
    for pid in sorted(tracks):groups.setdefault(find(pid),[]).append(pid)
    return sorted(groups.values(),key=lambda g:g[0])


def freeze_cases(tracks,pair_swaps):
    """Freeze all source pixels, global aliases, per-exclusion models and 2 slots/swap.

    models: key,image_ids,excluded_ids,train_ids,train_pixels (Nx4).
    cases: case_id,pair_ids,original_id,alternative_id,excluded_ids,pair_model_key,
    original_pair_pixels,alternative_pair_pixels,extra: image_id,model_key,
    reference_pixels,original_pixels,alternative_pixels. Every extra model is
    oriented [right image, extra image]; no image-order inference is required.
    """
    source=_tracks(tracks)
    if type(pair_swaps) is not list or not 1<=len(pair_swaps)<=64:
        raise ValueError('1..64 frozen pair swaps required')
    groups=_groups(source);by_pid={pid:g for g in groups for pid in g}
    models=[];keys={};cases=[]
    def model(pair,excluded):
        identity=(tuple(pair),tuple(excluded))
        if identity in keys:return keys[identity]
        ids=[]
        for group in groups:
            available=[pid for pid in group if pid not in excluded and all(i in source[pid] for i in pair)]
            if available:ids.append(min(available))
        ids.sort();key='model_'+str(len(models));keys[identity]=key
        models.append(dict(key=key,image_ids=list(pair),excluded_ids=list(excluded),train_ids=ids,
            train_pixels=[list(source[pid][pair[0]])+list(source[pid][pair[1]]) for pid in ids]))
        return key
    for index,swap in enumerate(pair_swaps):
        if type(swap) is not dict or set(swap)!={'image_ids','source_point_ids'}:
            raise ValueError('explicit image_ids and source_point_ids required')
        pair=swap['image_ids'];pids=swap['source_point_ids']
        if (type(pair) not in (list,tuple) or len(pair)!=2 or len(set(pair))!=2 or any(type(i) is not int or i<=0 for i in pair)
            or type(pids) not in (list,tuple) or len(pids)!=2 or len(set(pids))!=2 or any(type(i) is not int or i not in source for i in pids)):
            raise ValueError('two distinct image IDs and existing distinct track IDs required')
        if any(not all(i in source[pid] for i in pair) for pid in pids):
            raise ValueError('both candidates require both original pair observations')
        excluded=sorted(set(by_pid[pids[0]])|set(by_pid[pids[1]]));pairkey=model(pair,excluded)
        for slot,(original,alternative) in enumerate((pids,pids[::-1])):
            i,j=pair;extra=[]
            for k in sorted(set(source[original])-set(pair)):
                ref=list(source[original][k])
                extra.append(dict(image_id=k,model_key=model([j,k],excluded),reference_pixels=ref,
                    original_pixels=list(source[original][j])+ref,
                    alternative_pixels=list(source[alternative][j])+ref))
            cases.append(dict(case_id=f'swap_{index}_slot_{slot}',pair_ids=list(pair),original_id=original,
                alternative_id=alternative,excluded_ids=excluded,pair_model_key=pairkey,
                original_pair_pixels=list(source[original][i])+list(source[original][j]),
                alternative_pair_pixels=list(source[original][i])+list(source[alternative][j]),extra=extra))
    return dict(schema='frozen-multiview-candidate-plan-v1',tracks=source,pair_swaps=deepcopy(pair_swaps),
        global_alias_groups=groups,models=models,cases=cases,coordinates_modified=False,
        quality_threshold=None,scope='conditional raw distorted-pixel fixed-F comparisons; no matching truth')


def _support(original,alternative):
    tolerance=1e-10*max(1.,abs(original),abs(alternative))
    if abs(original-alternative)<=tolerance:return 'tie'
    return 'original' if original<alternative else 'alternative'


def _aggregate(evaluations):
    good=[e for e in evaluations if e['status']=='EVALUATED']
    if not good:return dict(status='UNAVAILABLE',support='unavailable',paired_count=0,original_rms=None,alternative_rms=None,
        reason='no paired evaluable observations')
    a=math.hypot(*(e['original'] for e in good))/math.sqrt(len(good))
    b=math.hypot(*(e['alternative'] for e in good))/math.sqrt(len(good))
    return dict(status='EVALUATED',support=_support(a,b),paired_count=len(good),original_rms=a,alternative_rms=b,
        included_model_keys=[e['model_key'] for e in good])


def analyse_plan(plan):
    """Validate freeze, fit each keyed model once; score candidates on same views."""
    if type(plan) is not dict or plan.get('schema')!='frozen-multiview-candidate-plan-v1':
        raise ValueError('frozen multiview candidate plan required')
    expected=freeze_cases(plan['tracks'],plan['pair_swaps'])
    if plan!=expected:raise ValueError('frozen plan differs from source-derived identities/pixels/splits')
    fitted={};models=[]
    for item in plan['models']:
        row=deepcopy(item)
        try:
            if len(item['train_ids'])<8:raise ValueError('fewer than eight independent alias-group representatives')
            parameters=fit_fundamental(np.asarray(item['train_pixels'],dtype=float))
            row.update(status='FITTED',parameters=parameters,reason=None)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
            row.update(status='NOT_EVALUATED',parameters=None,reason=str(exc))
        fitted[item['key']]=row;models.append(row)
    def evaluate(key,original,alternative):
        model=fitted[key];row=dict(model_key=key,status='UNAVAILABLE',original=None,alternative=None,support='unavailable',reason=None)
        if model['status']!='FITTED':row['reason']=model['reason'];return row
        try:
            a=float(_residual('fundamental',model['parameters'],np.asarray([original],dtype=float))[0])
            b=float(_residual('fundamental',model['parameters'],np.asarray([alternative],dtype=float))[0])
            row.update(status='EVALUATED',original=a,alternative=b,support=_support(a,b))
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:row['reason']=str(exc)
        return row
    cases=[]
    for source in plan['cases']:
        row=deepcopy(source);pair=evaluate(source['pair_model_key'],source['original_pair_pixels'],source['alternative_pair_pixels'])
        extras=[dict(e,evaluation=evaluate(e['model_key'],e['original_pixels'],e['alternative_pixels'])) for e in source['extra']]
        values=[e['evaluation'] for e in extras]
        row.update(pair=pair,extra=extras,extra_summary=_aggregate(values),pair_plus_extra_summary=_aggregate([pair]+values),
            extra_requested_count=len(values),extra_unavailable_count=sum(e['status']!='EVALUATED' for e in values))
        cases.append(row)
    return dict(schema='multiview-candidate-evidence-v1',models=models,cases=cases,coordinates_modified=False,
        equality_tolerance='1e-10*max(1,original,alternative); numerical comparison, not quality threshold',
        residual_definition='sqrt Sampson squared distance in raw distorted pixels',
        correspondence_verified=False,geometry_acceptance=False,independent_validation=False,
        limitations=['Existing tracks and views are not independent truth; shared fits make evidence correlated.',
            'Raw distorted pixels need not obey an ideal fundamental matrix.',
            'RMS compares identical paired evaluable views only; unavailable views remain explicit.',
            'No triangulation, inverse distortion, proxy construction, source-coordinate edits or automatic correction.'])
