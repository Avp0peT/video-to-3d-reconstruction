"""Read-only local motion review of a sealed association; no candidate selection."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT)]
import numpy as np
from review_mask_association import (read,save,bind,verify,integer,index_frames,validate_sources,
    verified_tracks,load_mask,trajectories,reference_pair,CLIPS)
PRIOR_SUMS_SHA256='da23090d4545a58b9a970c3c2be41369680c9245a1fdb2d7ee24250a34ef8196'


def fresh_output(output,sources):
    output=Path(output).resolve()
    if output.exists() or any(output.is_relative_to(Path(s).resolve()) or Path(s).resolve().is_relative_to(output) for s in sources):
        raise ValueError('fresh nonoverlapping output required')
    return output


def sealed_association(association):
    association=Path(association).resolve(strict=True);task=association.parent.parent
    sums=verify(dict(path=str(task/'SHA256SUMS.txt'),sha256=PRIOR_SUMS_SHA256));entries={}
    for line in Path(sums['path']).read_text(encoding='utf-8').splitlines():
        digest,relative=line.split('  ',1);relative=relative.replace('\\','/')
        if relative in entries:raise ValueError('duplicate checksum path')
        entries[relative]=digest
    paths=[task/'DELIVERY_CHECK.json']+[association/n for n in ('PLAN.json','RESULT.json','DECISIONS.json','SELECTION_LEDGER.json')]
    refs=[sums]
    for p in paths:
        relative=p.relative_to(task).as_posix()
        if relative not in entries:raise ValueError('missing sealed checksum')
        refs.append(verify(dict(path=str(p),sha256=entries[relative])))
    check=read(paths[0]);plan=read(association/'PLAN.json');result=read(association/'RESULT.json');decisions=read(association/'DECISIONS.json')
    if check.get('status')!='PASS' or check.get('failures')!=[]:raise ValueError('delivery check not PASS')
    if read(association/'SELECTION_LEDGER.json')!=[] or result.get('selected_candidates')!=0:raise ValueError('prior selection ledger must be empty')
    if result.get('status')!='ASSOCIATION_CANDIDATES_ONLY' or result.get('source_unchanged') is not True or result.get('frames')!=decisions or len(decisions)!=28 or plan.get('status')!='FROZEN_BEFORE_ASSOCIATION':raise ValueError('prior result/plan mismatch')
    for ref in plan['source_bindings']+plan['implementation']+plan['runtime_bindings']:refs.append(verify(ref))
    return plan,decisions,refs


def check_prior(old,row,pair,ids,segments):
    f=row['frame']
    integer(old['input_index']);integer(old['eligible_track_count'])
    for key in ('before_index','after_index'):
        if old.get(key) is not None:integer(old[key])
    if old.get('association') is not None and old.get('association')!=old.get('kernel_evidence'):raise ValueError('prior evidence aliases differ')
    for key,value in [('input_index',f['input_index']),('frame_id',f['frame_id']),('original_status',f['original_mask_status']),('before_index',pair[0] if pair else None),('after_index',pair[1] if pair else None),('eligible_track_count',len(ids))]:
        if old.get(key)!=value:raise ValueError('prior frame/track mismatch: '+key)
    if any(type(v) is not int for v in old.get('track_ids',[])) or old.get('track_ids',[])!=ids or old.get('geometry_segments',[])!=segments:raise ValueError('prior track IDs/segments mismatch')
    if old.get('selection') is not None or old.get('publishable_for_geometry') is not False:raise ValueError('prior selection forbidden')
    expected=[(c['candidate_id'],r['sha256']) for c,r,_ in row['candidates']]
    if [(c['candidate_id'],c['binding']['sha256']) for c in old['candidates']]!=expected:raise ValueError('prior candidates mismatch')
    if old.get('decision') is not None and old['decision']!=old['status']:raise ValueError('prior decision status mismatch')
    if old.get('status') not in ('UNASSESSED','UNRESOLVED','NO_CANDIDATE'):raise ValueError('unsupported prior state')
    evidence=old.get('kernel_evidence')
    if evidence and evidence.get('references'):
        held=ids[::5];fit=[v for j,v in enumerate(ids) if j%5]
        for record in [evidence]+list(evidence['references'].values()):
            for key,wanted in [('fit_track_ids',fit),('holdout_track_ids',held)]:
                got=record.get(key)
                if got!=wanted or any(type(v) is not int for v in got):raise ValueError('prior side IDs mismatch')
    return bool(evidence and evidence.get('references'))


def stats(values):
    return dict(count=len(values),mean_px=float(np.mean(values)) if values else None,p90_px=float(np.percentile(values,90)) if values else None)


def compare_global(evidence,local,points,ids):
    lookup={v:j for j,v in enumerate(ids)};out={}
    for profile,p in local['profiles'].items():
        common=p['common_supported_track_ids']
        if len(set(common))!=len(common) or any(type(v) is not int or v not in ids[::5] for v in common):raise ValueError('invalid local heldout IDs')
        record=dict(selection_conditioned=True,comparison_role='accepted_common_subset_descriptive_only',common_track_ids=common,eligible_holdout_count=len(ids[::5]),common_count=len(common),coverage=len(common)/len(ids[::5]) if ids else 0,references={})
        for side,source in [('before',points[0]),('after',points[2])]:
            old=evidence['references'][side];matrix=np.asarray(old['matrix'],float)
            if matrix.shape!=(2,3) or not np.isfinite(matrix).all():raise ValueError('invalid saved global matrix')
            held=ids[::5];indices=[lookup[v] for v in held]
            residual=np.linalg.norm(source[indices]@matrix[:,:2].T+matrix[:,2]-points[1][indices],axis=1)
            if not np.allclose(residual,old['holdout_residuals_px'],rtol=1e-6,atol=1e-5):raise ValueError('prior global residual mismatch')
            per={v:float(r) for v,r in zip(held,residual)}
            predicted=[];seen=set();cell_rows=[]
            for cell in p['references'][side]['cells']:
                cell_ids=cell['holdout_track_ids']
                if any(type(v) is not int or v not in held for v in cell_ids) or seen.intersection(cell_ids):raise ValueError('invalid local cell heldout IDs')
                seen.update(cell_ids)
                cell_rows.append(dict(cell_id=cell['cell_id'],heldout_count=len(cell_ids),has_fitted_matrix=cell['matrix'] is not None,passed=cell['passed'],reasons=cell['reasons']))
                if cell['matrix'] is None:continue
                lm=np.asarray(cell['matrix'],float)
                if lm.shape!=(2,3) or not np.isfinite(lm).all():raise ValueError('invalid local matrix')
                ci=[lookup[v] for v in cell_ids]
                lr=np.linalg.norm(source[ci]@lm[:,:2].T+lm[:,2]-points[1][ci],axis=1)
                if not np.allclose(lr,cell['holdout_residuals_px'],rtol=1e-6,atol=1e-5):raise ValueError('local residual mismatch')
                predicted.extend(dict(track_id=v,query_point_xy=points[1][lookup[v]].tolist(),cell_id=cell['cell_id'],cell_passed=cell['passed'],global_residual_px=per[v],local_residual_px=float(r)) for v,r in zip(cell_ids,lr))
            predicted.sort(key=lambda r:r['track_id'])
            all_predicted=dict(selection_conditioned=False,comparison_role='all_fitted_cell_predictions_including_rejected',eligible_holdout_count=len(held),predicted_count=len(predicted),missing_or_unfitted_count=len(held)-len(predicted),coverage=len(predicted)/len(held),global_stats=stats([r['global_residual_px'] for r in predicted]),local_stats=stats([r['local_residual_px'] for r in predicted]),per_point=predicted,cells=cell_rows)
            measured=p['measured_query_evidence'];local_values=[];rows=[]
            for item in measured:
                v=item['track_id']
                if type(v) is not int or v not in common or not np.array_equal(item['query_point_xy'],points[1][lookup[v]]):raise ValueError('local query identity mismatch')
                local_values.append(item[side+'_residual_px']);rows.append(dict(track_id=v,query_point_xy=item['query_point_xy'],global_residual_px=per[v],local_residual_px=item[side+'_residual_px']))
            if [r['track_id'] for r in rows]!=common:raise ValueError('local comparison coverage mismatch')
            record['references'][side]=dict(saved_matrix=old['matrix'],all_fitted_local_predictions=all_predicted,all_heldout_global_per_point=[dict(track_id=v,query_point_xy=points[1][lookup[v]].tolist(),global_residual_px=per[v]) for v in held],all_heldout_global=stats(residual.tolist()),common_global=stats([per[v] for v in common]),common_local=stats(local_values),per_point=rows)
        out[profile]=record
    return out


def build(association,coverage,review,output):
    start=time.monotonic();output=fresh_output(output,(association,coverage,review))
    prior,old_rows,refs=sealed_association(association)
    validated,source_refs,_=validate_sources(coverage,review);refs+=source_refs
    if prior['frames']!=[dict(**r['frame'],candidate_bindings=[dict(candidate_id=c['candidate_id'],**b) for c,b,_ in r['candidates']]) for r in validated.values()]:raise ValueError('prior frozen frames mismatch')
    old_map=index_frames(old_rows)
    from integrations.demo_pipeline.temporal_review import load_cv
    from kozlov_lab.stereo.local_motion import evaluate_local_motion,VERSION,CONSTANTS
    cvrefs=[];cv=load_cv(cvrefs)
    if cv is None:raise ValueError('pinned OpenCV unavailable')
    cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
    prepared=[]
    for lo,hi in CLIPS:
        tracks,bindings=verified_tracks(review,f'{lo}_{hi}');refs+=bindings
        if tracks['profile']['profile_id']!='whole-video-temporal-support-v2':raise ValueError('wrong tracking profile')
        originals={i:validated[i]['frame'] for i in range(lo,hi+1)}
        for index in range(lo,hi+1):
            row=validated[index];pair=reference_pair(originals,index);points=[np.empty((0,2),np.float32)]*3;ids=[];segments=[]
            if row['candidates'] and pair:points,ids,segments=trajectories(tracks,pair[0]-lo,index-lo,pair[1]-lo,hi-lo+1,validated[pair[0]]['mask'],validated[pair[1]]['mask'],cv)
            evaluated=check_prior(old_map[index],row,pair,ids,segments)
            prepared.append((index,pair,points,ids,segments,evaluated))
    if sum(p[-1] for p in prepared)!=16:raise ValueError('expected exactly 16 prior evaluations')
    implementation=[bind(ROOT/p) for p in ('scripts/review_local_motion.py','src/kozlov_lab/stereo/local_motion.py','scripts/review_mask_association.py')]
    frozen=[dict(input_index=i,original_frame=validated[i]['frame'],before_index=pair[0] if pair else None,after_index=pair[1] if pair else None,prior_status=old_map[i]['status'],track_ids=ids,geometry_segments=segments,before_points_xy=points[0].tolist(),query_points_xy=points[1].tolist(),after_points_xy=points[2].tolist(),evaluate=evaluated) for i,pair,points,ids,segments,evaluated in prepared]
    output.mkdir(parents=True)
    save(output/'PLAN.json',dict(status='FROZEN_BEFORE_LOCAL_MOTION',version=VERSION,constants=CONSTANTS,profiles=['local_similarity','local_affine'],source_bindings=refs,implementation=implementation,runtime_bindings=cvrefs,frames=frozen,threads=2,wall_limit_seconds=180,memory_limit_bytes=2*1024**3,selected_candidates=0,publishable_for_geometry=False))
    decisions=[]
    for index,pair,points,ids,segments,evaluated in prepared:
        if time.monotonic()-start>180:raise TimeoutError('180 second budget exceeded')
        old=old_map[index];d=dict(input_index=index,frame_id=old['frame_id'],original_status=old['original_status'],prior_status=old['status'],points=dict(before=points[0].tolist(),query=points[1].tolist(),after=points[2].tolist()),before_index=pair[0] if pair else None,after_index=pair[1] if pair else None,track_ids=ids,geometry_segments=segments,selected_candidate_id=None,publishable_for_geometry=False)
        if evaluated:
            local=evaluate_local_motion(validated[pair[0]]['mask'],validated[pair[1]]['mask'],*points,ids,cv)
            d.update(status='DIAGNOSTIC_ONLY',local_motion=local,global_comparison=compare_global(old['kernel_evidence'],local,points,ids))
        else:d['status']='NO_CANDIDATE' if not validated[index]['candidates'] else ('NO_REFERENCES' if not pair else 'NO_ELIGIBLE')
        decisions.append(d);save(output/'DECISIONS.json',decisions)
    for ref in refs+implementation+cvrefs:verify(ref)
    save(output/'SELECTION_LEDGER.json',[])
    result=dict(status='LOCAL_MOTION_DIAGNOSTIC_ONLY',frames=decisions,decisions=28,evaluated_frames=16,selected_candidates=0,source_unchanged=True,model_calls=0,tracking_calls=0,geometry_calls=0,downstream_calls=0,publishable_for_geometry=False,wall_seconds=time.monotonic()-start)
    save(output/'RESULT.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ('association','coverage','tracks-review','output'):parser.add_argument('--'+flag,required=True,type=Path)
    args=parser.parse_args();build(args.association,args.coverage,args.tracks_review,args.output)
