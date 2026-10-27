"""Supplement existing SIFT raw correspondences with current-mask region tracks.

Run only under an external process-tree supervisor (600 second wall limit).
No source database, reconstruction or image is modified.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import os
import sqlite3

THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
PAIR_BASE = 2147483647

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def load(path):
    def invalid(value): raise ValueError('nonfinite JSON '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)

def _expanded_policy(policy):
    if policy.get('id') == 'glove-region-bridge-keyframes-v1':
        return True
    if policy.get('schema') == 'object-region-bridge-policy-v1':
        from .object_region_policy import _configuration
        _configuration(policy)
        return True
    return False


def _validate_frame_limit(count, max_frames, sourceplan):
    if type(max_frames) is not int or not 2<=max_frames<=96:
        raise ValueError('max_frames must be integer 2..96')
    policy=sourceplan.get('endpoint_expansion_policy',{})
    if policy.get('schema')=='object-region-bridge-policy-v1':
        _expanded_policy(policy)
    if max_frames>64 and (max_frames!=96 or not _expanded_policy(policy)):
        raise ValueError('expanded frame limit requires max_frames=96 and versioned bridge policy')
    if type(count) is not int or not 2<=count<=max_frames:
        raise ValueError('frame count outside explicit max_frames')
    if policy.get('schema')=='object-region-bridge-policy-v1' and count>policy['native_max_frames']:
        raise ValueError('frame count outside tightened policy budget')


def prepare(sfm_run, tracks_file, max_frames=64):
    import numpy as np
    from PIL import Image
    source=Path(sfm_run).resolve(strict=True)
    tracks_file=Path(tracks_file).resolve(strict=True)
    doc=load(tracks_file)
    if doc.get('schema')!='glove-region-tracks-v1' or doc.get('coordinate_convention')!='opencv_integer_centers':
        raise ValueError('track schema/coordinate convention mismatch')
    fmap=source/'frames_map.jsonl'
    maps=[json.loads(line) for line in fmap.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    by_index={f['input_index']:f for f in maps}
    if len(by_index)!=len(maps): raise ValueError('duplicate frame map input_index')
    refs={str(p):digest(p) for p in (source/'database.db',source/'run.json',fmap,tracks_file,Path(__file__))}
    for folder in ('models','native_models'):
        for path in (source/folder).rglob('*'):
            if path.is_file():refs[str(path)]=digest(path)
    source_refs=doc.get('source_bindings',[])
    if not source_refs: raise ValueError('bound source plan required')
    for ref in source_refs:
        refpath=Path(ref['path']).resolve(strict=True)
        if digest(refpath)!=ref['sha256']:raise ValueError('track source binding drift')
        refs[str(refpath)]=ref['sha256']
    sourceplan=load(source_refs[0]['path'])
    if sourceplan.get('schema',sourceplan.get('contract_version'))!='glove-region-plan-v1':raise ValueError('source plan schema mismatch')
    identity_policy=sourceplan.get('identity_policy')
    if doc.get('identity_policy') != identity_policy:
        raise ValueError('source/track identity policy differs')
    if identity_policy is not None:
        from .track_identity import validate_policy, require_unique_targets
        from . import track_identity
        validate_policy(identity_policy)
        refs[str(Path(track_identity.__file__).resolve())]=digest(track_identity.__file__)
    endpoints={r['input_index']:r for r in sourceplan['timeline'] if r.get('input_index') is not None}
    _validate_frame_limit(len(doc['frames']),max_frames,sourceplan)
    if len(endpoints)!=len(doc['frames']):raise ValueError('endpoint coverage mismatch')
    frames=[]; seen=set()
    with sqlite3.connect('file:'+str(source/'database.db')+'?mode=ro',uri=True) as db:
        images=dict(db.execute('SELECT image_id,name FROM images'))
        sizes={iid:(rows,cols) for iid,rows,cols in db.execute('SELECT image_id,rows,cols FROM keypoints')}
    for item in doc['frames']:
        idx=item['input_index']
        if type(idx) is not int or idx in seen or idx not in by_index: raise ValueError('invalid/duplicate input_index')
        endpoint=endpoints.get(idx)
        if endpoint is None or any(item.get(k)!=endpoint.get(k) for k in ('decode_index','image_path','image_sha256','mask_path','mask_sha256','width','height')):raise ValueError('endpoint source plan binding mismatch')
        seen.add(idx); bound=by_index[idx]; iid=bound['database_image_id']; name=bound['staged_image_name']
        if images.get(iid)!=name: raise ValueError('database/frame-map name mismatch')
        image=Path(item['image_path']).resolve(strict=True); maskpath=Path(item['mask_path']).resolve(strict=True)
        staged=source/name
        if not staged.exists(): staged=source/'images'/Path(name).name
        if digest(image)!=item['image_sha256'] or digest(staged)!=item['image_sha256'] or bound['canonical_image_sha256']!=item['image_sha256']:
            raise ValueError('image hash binding mismatch')
        if digest(maskpath)!=item['mask_sha256']: raise ValueError('mask hash binding mismatch')
        w,h=item['width'],item['height']
        with Image.open(image) as im:
            if im.size!=(w,h): raise ValueError('image dimensions mismatch')
        with Image.open(maskpath) as im:
            if im.mode!='L' or im.size!=(w,h): raise ValueError('mask domain mismatch')
            mask=np.asarray(im)
        if not np.isin(mask,[0,255]).all() or not mask.any(): raise ValueError('nonempty binary current mask required')
        points={}
        for obs in item['observations']:
            tid=obs['track_id']; xy=obs['xy']
            if type(tid) is not int or tid<0 or tid in points or len(xy)!=2: raise ValueError('invalid/duplicate track')
            x,y=xy
            if not all(type(v) in (float,int) and math.isfinite(v) for v in xy) or not (0<=x<w and 0<=y<h): raise ValueError('invalid point')
            if not mask[min(h-1,int(math.floor(y+.5))),min(w-1,int(math.floor(x+.5)))]: raise ValueError('point outside current mask')
            points[tid]=[x,y]
        for p in (image,maskpath,staged): refs[str(p)]=digest(p)
        frames.append(dict(input_index=idx,decode_index=item['decode_index'],image_id=iid,name=name,image_path=str(staged),points=points,original_rows=sizes[iid][0],cols=sizes[iid][1]))
    if set(f['image_id'] for f in frames)!=set(images):raise ValueError('all baseline database images must be bound')
    _validate_frame_limit(len(frames),max_frames,sourceplan)
    counts=Counter(t for f in frames for t in f['points'])
    for f in frames:
        f['points']={t:xy for t,xy in sorted(f['points'].items()) if counts[t]>=2}
        if identity_policy is not None:
            require_unique_targets(list(f['points'].values()),identity_policy)
        f['row_by_track']={t:f['original_rows']+i for i,t in enumerate(f['points'])}
    return dict(frames=frames,source_bindings=refs,source=source,run=load(source/'run.json'),max_frames=max_frames,endpoint_expansion_policy=sourceplan.get('endpoint_expansion_policy',{}))

def restore_options(obj, values):
    """Restore every serialized option; reject unknown or unrepresentable values."""
    for key,value in values.items():
        current=getattr(obj,key)
        if isinstance(value,dict): restore_options(current,value)
        else:
            if isinstance(current,set) and value=='set()': value=set()
            elif isinstance(value,str) and '.' in value and not isinstance(current,(str,Path)):
                value=getattr(type(current),value.rsplit('.',1)[-1])
            setattr(obj,key,value)
    return obj

def _bind_mapper_image_names(mapper, frames, max_frames, policy):
    """Expand only a versioned bridge selection; preserve every other option."""
    before=list(mapper.image_names);names=[f['name'] for f in frames]
    if any(not isinstance(n,str) or not n for n in names+before) or len(names)!=len(set(names)) or len(before)!=len(set(before)):
        raise ValueError('invalid or duplicate mapper image names')
    if not set(before)<=set(names):raise ValueError('inherited mapper names absent from bound frames')
    expanded=max_frames==96 and _expanded_policy(policy)
    if expanded:
        mapper.image_names=names
    elif set(before)!=set(names):
        raise ValueError('bound frames differ from mapper selection without expansion policy')
    after=list(mapper.image_names)
    if set(after)!=set(names):raise ValueError('mapper selection must cover exactly bound frames')
    return dict(policy='versioned_bridge_expansion' if expanded else 'unchanged_baseline',before=before,after=after,added=sorted(set(after)-set(before)),removed=sorted(set(before)-set(after)))


def _read_keypoint_rows(rows, cols, data):
    """Read native SQLite float32 rows, including COLMAP's NULL empty rows."""
    import numpy as np
    if type(rows) is not int or rows<0 or type(cols) is not int or cols not in (2,4,6):
        raise ValueError('invalid baseline keypoint dimensions')
    if data is None:
        if rows:raise ValueError('nonempty baseline keypoints have NULL data')
        return np.empty((0,cols),np.float32)
    if len(data)!=rows*cols*np.dtype(np.float32).itemsize:
        raise ValueError('baseline keypoint byte length mismatch')
    return np.frombuffer(data,np.float32).reshape(rows,cols)


def supplement_database(plan, destination):
    import numpy as np
    source=plan['source']; destination=Path(destination)
    with sqlite3.connect('file:'+str(source/'database.db')+'?mode=ro',uri=True) as src, sqlite3.connect(destination) as db:
        src.backup(db)
        # No descriptor synthesis: all old keypoint rows and raw matches remain.
        db.execute('DELETE FROM descriptors'); db.execute('DELETE FROM two_view_geometries')
        for f in plan['frames']:
            rows,cols,data=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(f['image_id'],)).fetchone()
            old=_read_keypoint_rows(rows,cols,data)
            new=np.zeros((len(f['points']),cols),np.float32)
            new[:,:2]=np.asarray(list(f['points'].values()),np.float64).reshape(-1,2)+.5
            if cols>=4: new[:,2]=1
            if cols==6: new[:,5]=1
            combined=np.concatenate([old,new])
            db.execute('UPDATE keypoints SET rows=?,data=? WHERE image_id=?',(len(combined),combined.tobytes(),f['image_id']))
        pairs=[]
        for a,fa in enumerate(plan['frames']):
            for fb in plan['frames'][a+1:]:
                tids=sorted(fa['points'].keys()&fb['points'].keys())
                if not tids: continue
                ia,ib=fa['image_id'],fb['image_id']
                new=np.asarray([[fa['row_by_track'][t],fb['row_by_track'][t]] for t in tids],np.uint32)
                if ia>ib: ia,ib=ib,ia; new=new[:,::-1].copy()
                pid=ia*PAIR_BASE+ib
                row=db.execute('SELECT rows,cols,data FROM matches WHERE pair_id=?',(pid,)).fetchone()
                if row is None:
                    old=np.empty((0,2),np.uint32)
                else:
                    rows,cols,data=row
                    if type(rows) is not int or rows<0 or cols!=2:raise ValueError('invalid baseline raw match dimensions')
                    if data is None:
                        if rows!=0:raise ValueError('nonempty baseline raw matches have NULL data')
                        old=np.empty((0,2),np.uint32)
                    else:
                        if len(data)!=rows*cols*np.dtype(np.uint32).itemsize:raise ValueError('baseline raw match byte length mismatch')
                        old=np.frombuffer(data,np.uint32).reshape(rows,cols)
                merged=np.concatenate([old,new])
                db.execute('INSERT OR REPLACE INTO matches VALUES (?,?,?,?)',(pid,len(merged),2,merged.tobytes()))
                pairs.append(dict(image_a=ia,image_b=ib,sift_raw=len(old),region_raw=len(new),region_rows=new.tolist(),track_ids=tids))
        db.commit()
        allpairs=[(pid//PAIR_BASE,pid%PAIR_BASE,rows) for pid,rows in db.execute('SELECT pair_id,rows FROM matches WHERE rows>0 ORDER BY pair_id')]
    return pairs,allpairs

def run(sfm_run, tracks_file, output, max_frames=64):
    if any(os.environ.get(k)!='2' for k in THREADS): raise ValueError('thread environment must be 2 before native imports')
    import numpy as np
    import pycolmap as pc
    if pc.__version__!='4.2.1': raise ValueError('pycolmap 4.2.1 required')
    plan=prepare(sfm_run,tracks_file,max_frames=max_frames); out=Path(output).resolve()
    if out.exists() or any(Path(p).is_relative_to(out) for p in plan['source_bindings']): raise ValueError('fresh disjoint output required')
    nativeopts=plan['run']['effective_options']['native']
    verify=restore_options(pc.TwoViewGeometryOptions(),nativeopts['match_exhaustive']['two_view_geometry'])
    mapper=restore_options(pc.IncrementalPipelineOptions(),nativeopts['incremental_mapping']['pipeline'])
    selection=_bind_mapper_image_names(mapper,plan['frames'],max_frames,plan['endpoint_expansion_policy'])
    out.mkdir(parents=True)
    pairs,allpairs=supplement_database(plan,out/'database.db')
    with sqlite3.connect(out/'database.db') as db: names=dict(db.execute('SELECT image_id,name FROM images'))
    images=out/'images';images.mkdir()
    for iid,name in names.items():
        src=plan['source']/name
        if not src.exists(): src=plan['source']/'images'/Path(name).name
        dest=images/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
    pairfile=out/'pairs.txt';pairfile.write_text(''.join(names[a]+' '+names[b]+'\n' for a,b,_ in allpairs),encoding='utf-8')
    actual=dict(verification=verify.todict(),mapping=mapper.todict(),image_selection_binding=selection)
    def plain(v):
        if isinstance(v,dict): return {k:plain(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [plain(x) for x in v]
        if isinstance(v,(str,int,float,bool)) or v is None:return v
        return str(v)
    save(out/'PLAN.json',dict(max_frames=max_frames,actual_options=plain(actual),frames=plan['frames'],pairs=pairs,source_bindings=plan['source_bindings'],descriptors_policy='removed from independent copy; no extraction or descriptor matching',geometry_accepted=False))
    pc.verify_matches(out/'database.db',pairfile,options=verify)
    db=pc.Database.open(out/'database.db')
    pairmap={(p['image_a'],p['image_b']):p for p in pairs}; verified=[]
    for a,b,raw in allpairs:
        inliers=np.asarray(db.read_two_view_geometry(a,b).inlier_matches,np.uint32).reshape(-1,2)
        rawrows=np.asarray(db.read_matches(a,b),np.uint32).reshape(-1,2)
        if not set(map(tuple,inliers))<=set(map(tuple,rawrows)):raise ValueError('verified rows absent from raw correspondences')
        region=set(map(tuple,pairmap.get((a,b),{}).get('region_rows',[])))
        verified.append(dict(image_a=a,image_b=b,raw=raw,verified=len(inliers),region_raw=len(region),sift_raw=raw-len(region),region_verified=sum(tuple(x) in region for x in inliers),sift_verified=sum(tuple(x) not in region for x in inliers)))
    del db
    native=out/'mapping';native.mkdir()
    models=pc.incremental_mapping(out/'database.db',images,native,options=mapper)
    frames={f['image_id']:f for f in plan['frames']};results=[];registered=set()
    for mid,rec in sorted(models.items()):
        folder=out/'models'/str(mid);binary=folder/'binary';text=folder/'text';binary.mkdir(parents=True);text.mkdir()
        rec.write_binary(binary);rec.write_text(text);rec.export_PLY(folder/'points.ply')
        with (folder/'points.ply').open('rb') as stream:
            header=[]
            for _ in range(100):
                line=stream.readline().decode('ascii').strip();header.append(line)
                if line=='end_header':break
            if 'end_header' not in header or 'element vertex '+str(rec.num_points3D()) not in header:raise ValueError('PLY header readback mismatch')
        back=pc.Reconstruction(binary);bt=pc.Reconstruction(text)
        if set(back.reg_image_ids())!=set(rec.reg_image_ids()) or set(back.points3D)!=set(rec.points3D) or set(bt.points3D)!=set(rec.points3D): raise ValueError('model roundtrip mismatch')
        for pid,p in rec.points3D.items():
            if not np.array_equal(p.xyz,back.points3D[pid].xyz) or not np.allclose(p.xyz,bt.points3D[pid].xyz,rtol=1e-12,atol=1e-12):raise ValueError('XYZ roundtrip mismatch')
        indices=sorted(frames[i]['input_index'] for i in back.reg_image_ids());registered.update(indices)
        lengths=[len(p.track.elements) for p in back.points3D.values()];errors=[float(p.error) for p in back.points3D.values()]
        def stats(v):return dict(count=len(v),mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None,max=max(v) if v else None)
        results.append(dict(model_id=int(mid),registered_input_indices=indices,registered_views=len(indices),points=back.num_points3D(),track_length=stats(lengths),reprojection_error_px=stats(errors),binary_path=str(binary),text_path=str(text),ply_path=str(folder/'points.ply'),roundtrip_verified=True))
    after={p:digest(p) for p in plan['source_bindings']}
    if after!=plan['source_bindings']:raise ValueError('source drift')
    result=dict(schema='glove-region-geometry-v1',max_frames=max_frames,status='RECONSTRUCTED_NOT_QUALITY_ACCEPTED' if results else 'NO_MODEL',models=results,pairs=verified,actual_options=plain(actual),unique_registered_input_indices=sorted(registered),source_before=plan['source_bindings'],source_after=after,source_unchanged=True,components_merged=False,geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown')
    save(out/'RESULT.json',result)
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--sfm-run',required=True);parser.add_argument('--tracks',required=True);parser.add_argument('--output',required=True);parser.add_argument('--max-frames',type=int,default=64)
    args=parser.parse_args();run(args.sfm_run,args.tracks,args.output,max_frames=args.max_frames)
