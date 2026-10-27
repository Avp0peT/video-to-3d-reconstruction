"""Bounded foreground region coverage, independent registrations and overlap checks."""
import json
from pathlib import Path
import time
import numpy as np
from .temporal_review import fingerprint
from .area_tracking import track_areas, _mask
from .area_tracking_report import render_area_tracking
from .overlap_regions import assess_overlap
from .connected_regions import build_connected_plan, prefix_plan
from .overlap_report import render_overlap
from .neighborhood_review import review_neighborhood


def _write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')


def _baseline(plan,path,budget):
    if path is None:return None
    budget();path=Path(path);binding=fingerprint(path)
    value=json.loads(path.read_text(encoding='utf-8'))
    coverage=[]
    for source in plan['sources']:
        budget();i=source['source_input_index'];d=source['source_decode_index']
        domain=plan['frame_domains'][str(d)]
        mask=_mask(dict(left=0,top=0,width=domain['width'],height=domain['height']),domain)>0
        union=np.zeros_like(mask)
        for r in value['regions']:
            if r['source_input_index']!=i or r['source_decode_index']!=d or r['verified_targets']<2:continue
            roi=r['template_roi'];x,y,w,h=[roi[k] for k in ('left','top','width','height')]
            union[y:y+h,x:x+w]=True
        total=int(mask.sum());count=int((union&mask).sum())
        coverage.append(dict(source_input_index=i,source_decode_index=d,foreground_pixels=total,local_union_pixels=count,local_fraction=count/total if total else None))
    if fingerprint(path)!=binding:raise ValueError('area baseline drift')
    return dict(binding=binding,counts=value['counts'],measurement_wall_seconds=value['measurement_wall_seconds'],source_coverage=coverage,
        same_estimator=True,comparison_scope='Same source views and estimator; routes use the longest declared compatible prefix. Region selection and workload differ. Not a controlled speedup or independent 3D quality comparison.')


def _counts(regions, assessment):
    legs=[l for r in regions for l in r['region_legs']]
    edges=[t for e in assessment['edges'] for t in e['targets']]
    return dict(regions=len(regions),local_tracklets_2d=sum(r['verified_targets']>=2 for r in regions),
        network_tracklets_2d=sum(r['network_tracklet'] for r in assessment['regions']),
        targets_attempted=len(legs),targets_accepted=sum(l['status']=='ACCEPTED' for l in legs),
        edge_targets_consistent=sum(e['status']=='CONSISTENT' for e in edges),edge_targets_conflict=sum(e['status']=='CONFLICT' for e in edges),edge_targets_untested=sum(e['status']=='UNTESTED' for e in edges))


def review_overlap_sequence(source_plan,decoded,output,refs,cv,*,baseline_path=None,budget=lambda:None):
    budget();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();artifacts=[]
    result=dict(status='RUNNING',regions=[],batches=[],native_calls=dict(matchTemplate=0,findTransformECCWithMask=0),
        counts={},geometry_accepted=False,identity_verified=False,new_cameras=0,new_model_points=0,
        segmentation_calls=0,lk_calls=0,coverage_scope='Union of original SAM foreground pixels in each source image; not 3D surface coverage')
    def verify():
        budget()
        for row in refs:
            if fingerprint(row['path'])!=row:raise ValueError('overlap source drift: '+row['path'])
    try:
        verify();plan=build_connected_plan(source_plan,budget=budget)
        _write(output/'PLAN.json',plan);artifacts.append(output/'PLAN.json')
        small=prefix_plan(plan,64,budget=budget)
        _write(output/'PREFIX64_PLAN.json',small);artifacts.append(output/'PREFIX64_PLAN.json')
        # No estimator sees neighboring fitted motion; every ROI is independently registered.
        for start in range(0,len(plan['regions']),8):
            budget();ordinal=start//8;batch=output/f'batch_{ordinal:03d}';batch.mkdir(exist_ok=False)
            batch_plan=dict(regions=plan['regions'][start:start+8],frame_domains=plan['frame_domains'])
            _write(batch/'PLAN.json',batch_plan);artifacts.append(batch/'PLAN.json')
            tick=time.monotonic()
            try:measured=track_areas(batch_plan,decoded,cv,budget=budget)
            except Exception as exc:
                partial=getattr(exc,'partial_result',None)
                if partial is not None:
                    _write(batch/'FAILURE.json',partial)
                    result['interrupted_batch']=dict(ordinal=ordinal,evidence=f'batch_{ordinal:03d}/FAILURE.json',measurement=partial)
                    for name,count in partial['native_calls'].items():result['native_calls'][name]+=count
                raise
            seconds=time.monotonic()-tick
            _write(batch/'RESULT.json',measured);artifacts.append(batch/'RESULT.json')
            result['regions'].extend(measured['regions'])
            for name,count in measured['native_calls'].items():result['native_calls'][name]+=count
            result['batches'].append(dict(ordinal=ordinal,regions=len(measured['regions']),region_ids=[r['region_id'] for r in measured['regions']],result=f'batch_{ordinal:03d}/RESULT.json',report=f'batch_{ordinal:03d}/index.html',wall_seconds=seconds))
        result['overlap']=assess_overlap(plan,result['regions'],budget=budget)
        result['counts']=_counts(result['regions'],result['overlap'])
        prefix_ids={r['region_id'] for r in small['regions']}
        prefix_regions=[r for r in result['regions'] if r['region_id'] in prefix_ids]
        prefix_assessment=assess_overlap(small,prefix_regions,budget=budget)
        _write(output/'PREFIX64_ASSESSMENT.json',prefix_assessment);artifacts.append(output/'PREFIX64_ASSESSMENT.json')
        result['same_budget64']=dict(counts=_counts(prefix_regions,prefix_assessment),coverage=prefix_assessment['coverage'],
            planned_connected_components=small['planned_connected_components'],selection_counts=small['selection_counts'],
            note='First64 of the frozen sequence, independently fitted within this one run; assessment excludes all later regions and edges. No second native run or controlled speedup claim.')
        result['measurement_wall_seconds']=time.monotonic()-started
        result['baseline']=_baseline(plan,baseline_path,budget)
        # Strict fits/coverage remain frozen. Provisional continuation is separate.
        result['strict_native_calls']=dict(result['native_calls'])
        neighborhood,extra_artifacts=review_neighborhood(plan,result['regions'],decoded,output/'neighborhood',cv,budget=budget)
        artifacts+=extra_artifacts
        result['neighborhood']=dict(status=neighborhood['status'],counts=neighborhood['counts'],
            native_calls=neighborhood['native_calls'],report='neighborhood/index.html',geometry_accepted=False,new_model_points=0)
        for name,count in neighborhood['native_calls'].items():result['native_calls'][name]+=count
        verify()
        # Reports reuse saved evidence, never call registration again.
        for batch in result['batches']:
            budget();directory=output/f'batch_{batch["ordinal"]:03d}'
            saved=json.loads((directory/'RESULT.json').read_text(encoding='utf-8'))
            artifacts+=render_area_tracking(directory,saved,decoded,budget=budget)
        result['status']='OVERLAP_REVIEW_COMPLETE';result['source_unchanged']=True
        artifacts+=render_overlap(output,plan,result,decoded,budget=budget)
        result['wall_seconds']=time.monotonic()-started
        _write(output/'RESULT.json',result);artifacts.append(output/'RESULT.json')
        verify()
        return dict(status=result['status'],counts=result['counts'],native_calls=result['native_calls'],
            neighborhood=result['neighborhood'],strict_native_calls=result['strict_native_calls'],
            wall_seconds=result['wall_seconds'],geometry_accepted=False,new_model_points=0,
            report='intermediate_frames/population/regions/overlap/index.html',source_bindings=[fingerprint(p) for p in artifacts])
    except Exception as exc:
        partial=getattr(exc,'neighborhood_result',None)
        if partial is not None:
            result['interrupted_neighborhood']=dict(evidence='neighborhood/FAILURE.json',status=partial['status'],native_calls=partial['native_calls'])
            for name,count in partial['native_calls'].items():result['native_calls'][name]+=count
        result.update(status='RESOURCE_LIMIT' if isinstance(exc,(TimeoutError,MemoryError)) else 'FAILED',
            reason=f'{type(exc).__name__}: {exc}',source_unchanged=None,wall_seconds=time.monotonic()-started)
        try:_write(output/'FAILURE.json',result)
        except Exception as write_error:
            if hasattr(exc,'add_note'):exc.add_note('overlap failure evidence: '+str(write_error))
        raise
