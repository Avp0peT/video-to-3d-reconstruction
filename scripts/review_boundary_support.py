"""Frozen-source CPU boundary and reference-exterior diagnostics; no selection."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT)]
import numpy as np
from PIL import Image
from review_mask_association import read,save,bind,verify,integer,index_frames,validate_sources,reference_pair,CLIPS
from review_local_motion import sealed_association,fresh_output
LOCAL_SUMS_SHA256='aba2c950e7a412ecb9c8cc51db78e595b97a686573f99cd092c9e978e3f40e14'


def sealed_local_review(local_review):
    local=Path(local_review).resolve(strict=True);task=local.parent.parent
    sums=verify(dict(path=str(task/'SHA256SUMS.txt'),sha256=LOCAL_SUMS_SHA256));refs=[sums];entries={}
    for line in Path(sums['path']).read_text(encoding='utf-8').splitlines():
        digest,rel=line.split('  ',1);rel=rel.replace('\\','/')
        p=(task/rel).resolve(strict=True)
        if not p.is_relative_to(task) or rel in entries:raise ValueError('invalid local seal path')
        entries[rel]=digest
        refs.append(verify(dict(path=str(p),sha256=digest)))
    required=['DELIVERY_CHECK.json','INDEPENDENT_AUDIT.json']+[f'real01/local_motion/{n}.json' for n in ('PLAN','RESULT','DECISIONS','SELECTION_LEDGER')]+['real01/START_BINDINGS.json','real01/FINAL_BINDINGS.json']
    if any(p not in entries for p in required):raise ValueError('incomplete local seal')
    for name in ('DELIVERY_CHECK.json','INDEPENDENT_AUDIT.json'):
        check=read(task/name)
        if check.get('status')!='PASS' or check.get('failures')!=[]:raise ValueError('local audit not PASS')
    start=read(task/'real01/START_BINDINGS.json');final=read(task/'real01/FINAL_BINDINGS.json')
    if start!=final or len(start)!=9:raise ValueError('local implementation binding mismatch')
    for rel,digest in start.items():
        refs.append(verify(dict(path=str(ROOT/rel),sha256=digest)))
        snapshot=task/'real01/implementation'/rel
        if snapshot.relative_to(task).as_posix() not in entries:raise ValueError('missing local implementation snapshot')
        refs.append(verify(dict(path=str(snapshot),sha256=digest)))
    plan=read(local/'PLAN.json');result=read(local/'RESULT.json');decisions=read(local/'DECISIONS.json')
    if plan.get('status')!='FROZEN_BEFORE_LOCAL_MOTION' or result.get('status')!='LOCAL_MOTION_DIAGNOSTIC_ONLY' or result.get('source_unchanged') is not True or result.get('frames')!=decisions or len(decisions)!=28:raise ValueError('local result mismatch')
    if read(local/'SELECTION_LEDGER.json')!=[] or result.get('selected_candidates')!=0:raise ValueError('local selection forbidden')
    for ref in plan['source_bindings']+plan['implementation']+plan['runtime_bindings']:refs.append(verify(ref))
    return plan,index_frames(decisions),refs


def prepare_rows(validated,local_plan,local_rows,association_plan,association_rows):
    expected=[dict(**r['frame'],candidate_bindings=[dict(candidate_id=c['candidate_id'],**b) for c,b,_ in r['candidates']]) for r in validated.values()]
    if association_plan['frames']!=expected:raise ValueError('association original source identity mismatch')
    frozen={}
    for item in local_plan['frames']:
        i=integer(item['input_index'])
        if i in frozen:raise ValueError('duplicate frozen frame')
        frozen[i]=item
    old=index_frames(association_rows)
    if set(local_rows)!=set(validated) or set(frozen)!=set(validated) or set(old)!=set(validated):raise ValueError('local frame coverage mismatch')
    statuses=[r['frame']['original_mask_status'] for r in validated.values()]
    if {s:statuses.count(s) for s in set(statuses)}!={'OK':18,'NO_MATCH':6,'NEEDS_SELECTION':4}:raise ValueError('original status coverage mismatch')
    prepared=[];groups=set()
    for lo,hi in CLIPS:
        originals={i:validated[i]['frame'] for i in range(lo,hi+1)}
        for i in range(lo,hi+1):
            row=validated[i];f=row['frame'];prior=local_rows[i];freeze=frozen[i];pair=reference_pair(originals,i)
            for source in (prior,old[i]):
                integer(source['input_index'])
                if source['frame_id']!=f['frame_id'] or source['original_status']!=f['original_mask_status']:raise ValueError('local frame/status identity mismatch')
                if (source['before_index'],source['after_index'])!=(pair if pair else (None,None)):raise ValueError('original nearest OK pair mismatch')
            if freeze['original_frame']!=f or freeze['track_ids']!=prior['track_ids'] or freeze['query_points_xy']!=prior['points']['query'] or freeze['geometry_segments']!=prior['geometry_segments'] or prior.get('selected_candidate_id') is not None or prior.get('publishable_for_geometry') is not False:raise ValueError('local frozen identity mismatch')
            for ident in prior['track_ids']:integer(ident)
            if len(set(prior['track_ids']))!=len(prior['track_ids']):raise ValueError('duplicate local track IDs')
            lookup=dict(zip(prior['track_ids'],prior['points']['query']))
            if len(lookup)!=len(prior['points']['query']):raise ValueError('local query point count mismatch')
            profile=prior.get('local_motion',{}).get('profiles',{}).get('local_affine',{})
            positives=profile.get('measured_query_evidence',[])
            for point in positives:
                ident=integer(point['track_id'])
                xy=np.asarray(point['query_point_xy'])
                if ident not in lookup or xy.shape!=(2,) or not np.isfinite(xy).all() or point['query_point_xy']!=lookup[ident]:raise ValueError('positive measured query identity mismatch')
            if [p['track_id'] for p in positives]!=profile.get('common_supported_track_ids',[]):raise ValueError('positive coverage mismatch')
            exterior_pair=pair if f['original_mask_status']=='NEEDS_SELECTION' else None
            if exterior_pair:groups.add(exterior_pair)
            prepared.append(dict(index=i,pair=pair,positive_evidence=positives,exterior_pair=exterior_pair))
    if groups!={(80,82),(83,87)}:raise ValueError('unexpected exterior pair groups')
    return prepared,sorted(groups)


def query_witnesses(ledger,pair,index):
    out=[];seen=set()
    for witness in ledger['accepted_witnesses']:
        ident=integer(witness['witness_id'])
        if ident in seen:raise ValueError('duplicate witness ID')
        seen.add(ident);observations=witness['observations'];indices=[integer(o['frame_index']) for o in observations]
        if indices!=list(range(pair[0],pair[1]+1)):raise ValueError('witness actual query observations missing')
        item=observations[indices.index(index)];xy=item['xy']
        if np.asarray(xy).shape!=(2,) or not np.isfinite(xy).all() or any(isinstance(v,bool) for v in xy):raise ValueError('invalid witness query point')
        out.append(dict(witness_id=f'{pair[0]}_{pair[1]}:{ident}',query_point_xy=xy,reference_exterior_witness=True,verified_background=False,selection_condition='RETROSPECTIVE_SURVIVOR_CONDITIONED'))
    return out


class CountCV:
    def __init__(self,cv):self.cv=cv;self.flow_calls=0
    def __getattr__(self,name):return getattr(self.cv,name)
    def calcOpticalFlowPyrLK(self,*args,**kwargs):
        self.flow_calls+=1
        return self.cv.calcOpticalFlowPyrLK(*args,**kwargs)


def build(local_review,association,coverage,tracks_review,output):
    start=time.monotonic();output=fresh_output(output,(local_review,association,coverage,tracks_review))
    prior,old,refs=sealed_association(association)
    plan,local,localrefs=sealed_local_review(local_review);refs+=localrefs
    validated,source_refs,_=validate_sources(coverage,tracks_review);refs+=source_refs
    prepared,groups=prepare_rows(validated,plan,local,prior,old)
    from integrations.demo_pipeline.temporal_review import load_cv
    from kozlov_lab.stereo.exterior_witness import track_reference_exterior,PROFILE
    from kozlov_lab.stereo.boundary_evidence import assess_candidate_boundary
    cvrefs=[];cv=load_cv(cvrefs)
    if cv is None:raise ValueError('pinned OpenCV unavailable')
    cv.setNumThreads(2);cv.ocl.setUseOpenCL(False);countcv=CountCV(cv)
    implementation=[bind(ROOT/p) for p in ('scripts/review_boundary_support.py','src/kozlov_lab/stereo/boundary_evidence.py','src/kozlov_lab/stereo/exterior_witness.py','scripts/review_mask_association.py','scripts/review_local_motion.py')]
    def rgb(index):
        with Image.open(validated[index]['frame']['path']) as im:return np.array(im.convert('RGB'),dtype=np.uint8)
    output.mkdir(parents=True)
    try:
        save(output/'PLAN.json',dict(status='FROZEN_BEFORE_BOUNDARY_SUPPORT',source_bindings=refs,implementation=implementation,runtime_bindings=cvrefs,frames=[dict(input_index=p['index'],original_frame=validated[p['index']]['frame'],positive_evidence=p['positive_evidence'],exterior_pair=p['exterior_pair']) for p in prepared],exterior_pairs=groups,exterior_profile=PROFILE,gray_conversion='OpenCV COLOR_RGB2GRAY from PIL RGB uint8',boundary_parameters=dict(blur_kernel=[3,3],blur_sigma=1,canny_thresholds=[40,100],canny_aperture=3,canny_l2gradient=True,alignment_distance_px=3,clipped_unknown_band_px=3),threads=2,wall_limit_seconds=180,memory_limit_bytes=2*1024**3,selected_candidates=0,publishable_for_geometry=False))
        save(output/'SELECTION_LEDGER.json',[]);pairs={};decisions=[]
        def budget():
            if time.monotonic()-start>180:raise TimeoutError('180 second budget exceeded')
        for pair in groups:
            budget();indices=list(range(pair[0],pair[1]+1));images=[cv.cvtColor(rgb(i),cv.COLOR_RGB2GRAY) for i in indices]
            ledger=track_reference_exterior(images,validated[pair[0]]['mask'],validated[pair[1]]['mask'],indices,countcv)
            ledger['actual_flow_calls_cumulative']=countcv.flow_calls
            pairs[f'{pair[0]}_{pair[1]}']=ledger;save(output/'EXTERIOR_PAIRS.json',pairs)
        for item in prepared:
            budget();index=item['index'];row=validated[index];frame=row['frame'];pair=item['exterior_pair']
            exterior=query_witnesses(pairs[f'{pair[0]}_{pair[1]}'],pair,index) if pair else []
            candidates=[dict(candidate_id=c['candidate_id'],binding=b,original_metadata=c) for c,b,_ in row['candidates']]
            assessment=assess_candidate_boundary(rgb(index),[dict(candidate_id=c['candidate_id'],mask=m) for c,b,m in row['candidates']],item['positive_evidence'],exterior,cv) if candidates else None
            decision=dict(input_index=index,frame_id=frame['frame_id'],original_status=frame['original_mask_status'],old_local_status=local[index]['status'],before_index=item['pair'][0] if item['pair'] else None,after_index=item['pair'][1] if item['pair'] else None,positive_evidence=item['positive_evidence'],exterior_evidence=exterior,exterior_pair=pair,exterior_reason='RETROSPECTIVE_SURVIVOR_CONDITIONED' if pair else ('NO_CANDIDATE' if not candidates else 'NOT_REQUESTED_CONTROL'),candidates=candidates,assessment=assessment,status='NO_CANDIDATE' if not candidates else 'EVIDENCE_ONLY',selected_candidate_id=None,publishable_for_geometry=False)
            decisions.append(decision);save(output/'DECISIONS.json',decisions)
        for ref in refs+implementation+cvrefs:verify(ref)
        result=dict(status='BOUNDARY_SUPPORT_DIAGNOSTIC_ONLY',frames=decisions,pair_ledgers=pairs,decisions=28,tracking_pair_calls=len(groups),adjacent_frame_steps=sum(b-a for a,b in groups),flow_calls=countcv.flow_calls,boundary_frames=sum(bool(r['candidates']) for r in decisions),selected_candidates=0,source_unchanged=True,model_calls=0,geometry_calls=0,downstream_calls=0,publishable_for_geometry=False,verified_background=False,endpoint_selection='RETROSPECTIVE_SURVIVOR_CONDITIONED',wall_seconds=time.monotonic()-start)
        save(output/'RESULT.json',result);return result
    except Exception as exc:
        save(output/'FAILURE.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),selected_candidates=0,publishable_for_geometry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ('local-review','association','coverage','tracks-review','output'):parser.add_argument('--'+flag,required=True,type=Path)
    args=parser.parse_args();build(args.local_review,args.association,args.coverage,args.tracks_review,args.output)
