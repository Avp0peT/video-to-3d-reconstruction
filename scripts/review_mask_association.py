"""Read-only, bounded 2-D mask association from frozen original inputs."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from PIL import Image
CLIPS=((49,62),(77,90))
FAILURES={'flow_failed','nonfinite_flow','fb_error','lk_error','outside_patch','patch_low_texture','patch_appearance'}


def read(path):
    def pairs(items):
        obj={}
        for key,value in items:
            if key in obj:raise ValueError('duplicate JSON key: '+key)
            obj[key]=value
        return obj
    def floating(value):
        number=float(value)
        if not math.isfinite(number):raise ValueError('nonfinite JSON number')
        return number
    def invalid(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_float=floating,parse_constant=invalid)

def verified_tracks(review,name):
    review=Path(review).resolve(strict=True);task=review.parent.parent
    audit_path=bounded(task,'INDEPENDENT_RESULT_AUDIT.json');audit=read(audit_path)
    if audit.get('status')!='PASS' or audit.get('failures')!=[]:raise ValueError('prior independent audit did not pass')
    path=bounded(review,f'{name}/after/TRACKS.json');expected=audit['clips'][name]['tracks_sha256']
    ref=verify(dict(path=str(path),sha256=expected));audit_ref=bind(audit_path)
    sums=task/'SHA256SUMS.txt'
    if sums.exists():
        entries={}
        for line in sums.read_text(encoding='utf-8').splitlines():
            digest,relative=line.split('  ',1)
            if relative in entries:raise ValueError('duplicate checksum path')
            entries[relative.replace('\\','/')]=digest
        for r in (ref,audit_ref):
            relative=Path(r['path']).relative_to(task).as_posix()
            if entries.get(relative)!=r['sha256']:raise ValueError('prior checksum manifest mismatch: '+relative)
    return read(path),[ref,audit_ref]+([bind(sums)] if sums.exists() else [])
def save(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def bind(path):
    p=Path(path).resolve(strict=True)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def verify(ref):
    got=bind(ref['path'])
    if got['sha256']!=ref['sha256'] or ('bytes' in ref and got['bytes']!=ref['bytes']): raise ValueError('binding drift: '+str(ref['path']))
    return got

def integer(x):
    if type(x) is not int or x<0: raise ValueError('nonnegative integer required (bool is invalid)')
    return x

def bounded(root,path):
    root=Path(root).resolve(strict=True);p=(root/path).resolve(strict=True)
    if not p.is_relative_to(root): raise ValueError('path escapes verified root')
    return p

def index_frames(frames):
    out={};ids=set()
    for frame in frames:
        i=integer(frame['input_index']);fid=frame['frame_id']
        if type(fid) is not str or not fid:raise ValueError('invalid frame ID')
        if i in out or fid in ids: raise ValueError('duplicate input index or frame ID')
        out[i]=frame;ids.add(fid)
    return out

def reference_pair(frames,index):
    # Only immutable original source OK entries may serve as references.
    before=[i for i,f in frames.items() if 0<index-i<=6 and f['original_mask_status']=='OK']
    after=[i for i,f in frames.items() if 0<i-index<=6 and f['original_mask_status']=='OK']
    return (max(before),min(after)) if before and after else None

def trajectories(tracking,before,query,after,frame_count,before_mask,after_mask,cv,diagnostics=None):
    counts=Counter()
    ids=set();good=[]
    distances=[cv.distanceTransform(mask,cv.DIST_L2,cv.DIST_MASK_PRECISE) for mask in (before_mask,after_mask)]
    for track in tracking['tracks']:
        ident=integer(track['track_id'])
        if ident in ids: raise ValueError('duplicate track IDs')
        ids.add(ident);observations=track['observations'];indices=[integer(o['frame_index']) for o in observations]
        if indices!=sorted(set(indices)) or any(i>=frame_count for i in indices): raise ValueError('unordered/duplicate/out-of-range observations')
        lookup=dict(zip(indices,observations))
        if any(i not in lookup for i in range(before,after+1)):
            counts['missing_observation']+=1;continue
        selected=[lookup[i] for i in range(before,after+1)]
        valid=True
        for o in selected:
            fb,error=o.get('fb_error_px'),o.get('lk_error')
            xy=np.asarray(o['xy'],float)-.5
            if (xy.shape!=(2,) or not np.isfinite(xy).all() or not (0<=xy[0]<=before_mask.shape[1]-1 and 0<=xy[1]<=before_mask.shape[0]-1) or not o.get('strong') or
                fb is None or error is None or not np.isfinite(fb) or not np.isfinite(error) or not 0<=fb<=1 or not 0<=error<=25):valid=False;break
        if not valid:
            counts['ordinary_flow_or_strength']+=1;continue
        for o in selected[1:]:
            proof=o.get('temporal_support') or {};ncc,std=proof.get('patch_ncc'),proof.get('patch_std')
            if (ncc is None or std is None or not np.isfinite(ncc) or not np.isfinite(std) or ncc<.75 or std<5 or FAILURES.intersection(proof.get('reasons',[]))):valid=False;break
        if not valid:
            counts['independent_patch_chain']+=1;continue
        for o,mask,dist in zip((lookup[before],lookup[after]),(before_mask,after_mask),distances):
            xy=np.asarray(o['xy'],float)-.5;x,y=np.floor(xy+.5).astype(int)
            if (o['label']!='foreground' or o.get('target_verified') is not True or not (0<=x<mask.shape[1] and 0<=y<mask.shape[0]) or not mask[y,x] or dist[y,x]<3):valid=False;break
        if not valid:counts['original_endpoint_mask_or_interior']+=1
        if valid:good.append((ident,[lookup[i] for i in (before,query,after)],[(i,lookup[i]['identity_segment']) for i in range(before,after+1)]))
    if diagnostics is not None:diagnostics.update(total_tracks=len(ids),eligible_tracks=len(good),exclusion_counts=dict(counts))
    if len(good)>768:raise ValueError('track budget exceeded')
    good.sort(key=lambda x:x[0])
    points=[np.asarray([np.asarray(row[1][j]['xy'],float)-.5 for row in good],np.float32).reshape(-1,2) for j in range(3)]
    return points,[row[0] for row in good],[dict(track_id=row[0],geometry_segments=[dict(frame_index=i,identity_segment=s) for i,s in row[2]]) for row in good]

def load_mask(ref,shape):
    verify(ref)
    with Image.open(ref['path']) as image:
        if image.mode!='L': raise ValueError('mask must be uint8 L')
        a=np.array(image)
    if a.dtype!=np.uint8 or a.shape!=shape or not np.isin(a,[0,255]).all(): raise ValueError('invalid binary mask or dimensions')
    return a

def validate_sources(coverage,review):
    coverage=Path(coverage).resolve(strict=True);review=Path(review).resolve(strict=True)
    plan=read(review/'PLAN.json');inputs=read(coverage/'INPUTS.json');result=read(coverage/'RESULT.json')
    if plan['clips']!=[list(c) for c in CLIPS] or plan['status']!='FROZEN_BEFORE_TRACKING':raise ValueError('unsupported review plan')
    if result['status']!='PARTIAL_CONDITIONAL_EXPANSION' or not result['source_unchanged']:raise ValueError('unverified coverage source')
    if inputs['track_profile']['profile_id']!='whole-video-incremental-v1':raise ValueError('wrong source profile')
    refs=[]
    for ref in plan['source_bindings']+plan['implementation']+plan.get('runtime_bindings',[]):refs.append(verify(ref))
    for name in ('INPUTS.json','RESULT.json'):
        actual=bind(coverage/name)
        if not any(Path(r['path']).resolve()==Path(actual['path']) and r['sha256']==actual['sha256'] for r in plan['source_bindings']):raise ValueError('coverage not bound by original PLAN')
    manifests=[r for r in inputs['source_bindings'] if Path(r['path']).name=='manifest.json' and Path(r['path']).parent.name=='merged']
    if len(manifests)!=1:raise ValueError('exactly one bound original merged manifest required')
    manifest_ref=verify(manifests[0]);refs.append(manifest_ref);manifest_path=Path(manifest_ref['path']);manifest=read(manifest_path)
    image_root=Path(manifest['source_io']['path']).resolve(strict=True)
    # Root identity itself is bound through its immutable run and frame registry.
    for name,key in [('run.json','run_json_sha256'),('frames.jsonl','frames_jsonl_sha256')]:refs.append(verify(dict(path=str(bounded(image_root,name)),sha256=manifest['source_io'][key])))
    fm=index_frames(inputs['frames']);mm=index_frames(manifest['frames']);validated={}
    for lo,hi in CLIPS:
        frames=[]
        for i in range(lo,hi+1):
            f=fm[i];m=mm[i]
            if (f['frame_id']!=m['frame_id'] or f['original_mask_status']!=m['status'] or f['sha256']!=m['source_image_sha256'] or Path(f['path']).resolve()!=Path(m['source_image']).resolve()):raise ValueError('frame/manifest identity mismatch')
            p=Path(f['path']).resolve(strict=True)
            if not p.is_relative_to(image_root):raise ValueError('unverified image root')
            refs.append(verify(dict(path=str(p),sha256=f['sha256'])))
            with Image.open(p) as im:
                if im.size!=(f['width'],f['height']) or im.size!=(m['width'],m['height']) or max(im.size)>1024:raise ValueError('image dimensions drift')
            for value in (f['width'],f['height'],m['width'],m['height'],f['decode_index'],f['pts']):integer(value)
            if not f['width'] or not f['height']:raise ValueError('empty declared image')
            shape=(f['height'],f['width']);candidates=[];seen=set()
            for candidate in m['candidates']:
                cid=candidate['candidate_id']
                if cid in seen:raise ValueError('duplicate candidate ID')
                seen.add(cid);ref=dict(path=str(bounded(manifest_path.parent,candidate['mask_path'])),sha256=candidate['mask_sha256'])
                mask=load_mask(ref,shape);refs.append(bind(ref['path']));candidates.append((candidate,ref,mask))
            mask=None
            if f['original_mask_status']=='OK':
                selected=m.get('selected_mask');binding=f.get('mask_binding')
                if not selected or not binding:raise ValueError('OK missing original selected binding')
                p=bounded(manifest_path.parent,selected['path'])
                if Path(binding['path']).resolve()!=p or selected['sha256']!=binding['sha256']:raise ValueError('selected mask binding differs')
                mask=load_mask(binding,shape);refs.append(verify(binding))
            elif f.get('mask_binding'):raise ValueError('unverified source promoted to mask')
            validated[i]=dict(frame=f,manifest=m,candidates=candidates,mask=mask);frames.append(f)
        for a,b in zip(frames,frames[1:]):
            if b['decode_index']!=a['decode_index']+1 or b['pts']<=a['pts'] or b['time_base']!=a['time_base']:raise ValueError('nonadjacent decode or PTS')
        if frames!=plan['input_frames'][len(validated)//14-1]:raise ValueError('review frozen frame mapping differs')
    return validated,refs,plan

def _build(coverage,review,output):
    start=time.monotonic();output=Path(output).resolve()
    for source in (Path(coverage).resolve(),Path(review).resolve()):
        if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('fresh nonoverlapping output required')
    validated,refs,old_plan=validate_sources(coverage,review)
    tracks={}
    for lo,hi in CLIPS:
        name=f'{lo}_{hi}';tracks[name],track_refs=verified_tracks(review,name);refs.extend(track_refs)
        if tracks[name]['profile']['profile_id']!='whole-video-temporal-support-v2':raise ValueError('wrong tracking profile')
    from integrations.demo_pipeline.temporal_review import load_cv
    from kozlov_lab.stereo.mask_association import associate_candidates
    cvrefs=[];cv=load_cv(cvrefs)
    if cv is None:raise ValueError('pinned OpenCV unavailable')
    cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
    implementation=[bind(ROOT/p) for p in ('scripts/review_mask_association.py','src/kozlov_lab/stereo/mask_association.py')]
    output.mkdir(parents=True)
    plan=dict(status='FROZEN_BEFORE_ASSOCIATION',clips=CLIPS,source_bindings=refs,implementation=implementation,runtime_bindings=cvrefs,threads=2,wall_limit_seconds=180,memory_limit_bytes=2*1024**3,maximum_queries=28,model_calls=0,tracking_calls=0,geometry_calls=0,downstream_calls=0,publishable_for_geometry=False,source_review_plan=bind(Path(review)/'PLAN.json'),frames=[dict(**row['frame'],candidate_bindings=[dict(candidate_id=c['candidate_id'],**r) for c,r,_ in row['candidates']]) for row in validated.values()])
    save(output/'PLAN.json',plan);decisions=[];ledger=[]
    for lo,hi in CLIPS:
        originals={i:validated[i]['frame'] for i in range(lo,hi+1)};name=f'{lo}_{hi}'
        for index in range(lo,hi+1):
            if time.monotonic()-start>180:raise TimeoutError('180 second budget exceeded')
            row=validated[index];status=row['frame']['original_mask_status'];pair=reference_pair(originals,index)
            decision=dict(input_index=index,frame_id=row['frame']['frame_id'],original_status=status,status='UNRESOLVED',publishable_for_geometry=False,candidates=[dict(candidate_id=c['candidate_id'],binding=r,raw_score=c.get('score'),score_semantics=c.get('score_semantics'),score_used_for_ranking=False) for c,r,_ in row['candidates']])
            decision['reasons']=[];decision['track_rejection']=dict(total_tracks=len(tracks[name]['tracks']),eligible_tracks=0,exclusion_counts={'not_assessed_no_candidate_or_reference':len(tracks[name]['tracks'])})
            if not row['candidates']:
                decision['status']='NO_CANDIDATE';decision['reasons']=['NO_ORIGINAL_CANDIDATE']
            elif pair:
                before,after=pair;diagnostics={};points,ids,segments=trajectories(tracks[name],before-lo,index-lo,after-lo,hi-lo+1,validated[before]['mask'],validated[after]['mask'],cv,diagnostics)
                decision['track_rejection']=diagnostics
                evidence=associate_candidates(validated[before]['mask'],validated[after]['mask'],*points,ids,[dict(candidate_id=c['candidate_id'],mask=m) for c,_,m in row['candidates']],cv) if ids else dict(status='UNRESOLVED',reason='NO_ELIGIBLE_TRACKS',selected_candidate_id=None)
                decision.update(endpoint_frames=[before,after],track_ids=ids,geometry_segments=segments,association=evidence)
                selected_id=evidence.get('selected_candidate_id')
                chosen=next((j for j,(c,_,_) in enumerate(row['candidates']) if c['candidate_id']==selected_id),None)
                if selected_id is not None and chosen is None:raise ValueError('kernel selected invalid candidate ID')
                if chosen is not None:
                    if status=='NEEDS_SELECTION':
                        decision['status']='TEMPORAL_CANDIDATE_SELECTED';candidate,ref,_=row['candidates'][chosen]
                        ledger.append(dict(input_index=index,frame_id=row['frame']['frame_id'],candidate_id=candidate['candidate_id'],candidate_binding=ref,status='TEMPORAL_CANDIDATE_SELECTED',original_status=status,physical_identity='UNVERIFIED',quality_status='NOT_EVALUATED',publishable_for_geometry=False))
                    elif status=='OK':decision['status']='CONSISTENT' if row['candidates'][chosen][0]['candidate_id']==row['manifest']['selected_candidate_id'] else 'FLAGGED'
                elif status=='OK':decision['status']='UNASSESSED'
            else:
                decision['reasons']=['NO_TWO_SIDED_ORIGINAL_REFERENCE']
                if status=='OK':decision['status']='UNASSESSED'
            decision.update(before_index=pair[0] if pair else None,after_index=pair[1] if pair else None,eligible_track_count=len(decision.get('track_ids',[])),decision=decision['status'],kernel_evidence=decision.get('association'),selection=next((v for v in ledger if v['input_index']==index),None))
            decisions.append(decision)
            save(output/'DECISIONS.json',decisions)
    for ref in refs+implementation:verify(ref)
    save(output/'DECISIONS.json',decisions);save(output/'SELECTION_LEDGER.json',ledger)
    result=dict(status='ASSOCIATION_CANDIDATES_ONLY',frames=decisions,decisions=len(decisions),selected_candidates=len(ledger),source_unchanged=True,downstream_calls=0,geometry_calls=0,model_calls=0,publishable_for_geometry=False,wall_seconds=time.monotonic()-start)
    save(output/'RESULT.json',result);return result

def build(coverage,review,output):
    output=Path(output).resolve();fresh=not output.exists()
    safe=fresh and all(not output.is_relative_to(Path(source).resolve()) and not Path(source).resolve().is_relative_to(output) for source in (coverage,review))
    try:return _build(coverage,review,output)
    except Exception as exc:
        if safe:
            output.mkdir(parents=True,exist_ok=True)
            save(output/'FAILURE.json',dict(status='FAILED',exception=type(exc).__name__,message=str(exc),downstream_calls=0))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--coverage',required=True,type=Path);parser.add_argument('--tracks-review',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    fresh=not args.output.exists()
    try: print(json.dumps(build(args.coverage,args.tracks_review,args.output),ensure_ascii=False))
    except Exception as exc:
        if fresh and args.output.is_dir():save(args.output/'FAILURE.json',dict(status='FAILED',exception=type(exc).__name__,message=str(exc),downstream_calls=0))
        raise
