"""Native consumer of saved, source-bound external image observations.

This is a separate experimental contract, not a claim of physical track identity
or the existing native-SIFT producer. No descriptors or correspondences are made
up. The caller must supervise and reclaim this CPU native process.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import os

SCHEMA = 'balanced-object-observations-v1'
MIN_TRACK_VIEWS = 3
MAX_SEQUENCE_DISTANCE = 3
MIN_PAIR_MATCHES = 15
MIN_MODEL_SIZE = 2  # Existing new.mp4 run's recorded value, not backend default 10.
THREAD_ENV = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def _hash(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    after = path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError('source changed during read: '+str(path))
    return dict(path=str(path),bytes=after.st_size,sha256=h.hexdigest())


def _save(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def _plain(value):
    if isinstance(value,dict): return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)): return [_plain(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)): return value
    return str(value)


def _stats(values):
    import numpy as np
    a = np.asarray(values,float)
    if not len(a): return dict(n=0)
    if not np.isfinite(a).all(): raise ValueError('nonfinite native statistics')
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),max=float(a.max()),mean=float(a.mean()))


def prepare(tracks_path):
    """Pure source/identity validation and deterministic row/pair planning."""
    import numpy as np
    from PIL import Image
    path = Path(tracks_path).resolve(strict=True)
    binding = _hash(path)
    if binding['bytes'] > 32 << 20: raise ValueError('tracks JSON exceeds 32MiB')
    def invalid(value): raise ValueError('nonfinite JSON: '+value)
    doc = json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=invalid)
    if not isinstance(doc,dict) or doc.get('schema') != SCHEMA: raise ValueError('unknown observations schema')
    if doc.get('arm') not in ('balanced','sift_control'): raise ValueError('unknown arm')
    frames = doc.get('frames')
    if not isinstance(frames,list) or not 3 <= len(frames) <= 192: raise ValueError('3..192 complete frames required')
    refs = [binding]; validated = []; seen_frames=set(); seen_indices=set(); last_index=-1;last_decode=-1
    track_views = Counter(); dimensions=None
    for seq,frame in enumerate(frames):
        if not isinstance(frame,dict): raise ValueError('frame must be object')
        i,d = frame.get('input_index'),frame.get('decode_index')
        fid=frame.get('frame_id');w,h=frame.get('width'),frame.get('height')
        if (type(i) is not int or i<0 or i<=last_index or type(d) is not int or d<0 or d<=last_decode
                or not isinstance(fid,str) or not fid or fid in seen_frames or i in seen_indices):
            raise ValueError('invalid/duplicate/nonordered frame identity')
        last_index,last_decode=i,d;seen_frames.add(fid);seen_indices.add(i)
        if type(w) is not int or type(h) is not int or min(w,h)<2 or max(w,h)>1024: raise ValueError('invalid frame dimensions')
        if dimensions is None: dimensions=(w,h)
        if dimensions!=(w,h): raise ValueError('shared camera requires identical dimensions')
        if frame.get('original_mask_status')!='OK': raise ValueError('only original OK masks admitted')
        for key in ('image','mask'):
            p=frame.get(key+'_path')
            if not isinstance(p,str) or not Path(p).is_absolute(): raise ValueError('absolute source paths required')
            ref=_hash(p)
            if ref['sha256']!=frame.get(key+'_sha256'): raise ValueError('source hash mismatch: '+key)
            refs.append(ref)
        with Image.open(frame['image_path']) as im:
            if im.size!=(w,h) or im.mode!='RGB': raise ValueError('canonical image must be matching RGB')
        with Image.open(frame['mask_path']) as im:
            if im.size!=(w,h) or im.mode not in ('L','1'): raise ValueError('matching scalar mask required')
            mask=np.asarray(im)>0
        if not mask.any(): raise ValueError('empty selected mask')
        observations=frame.get('observations')
        if not isinstance(observations,list) or len(observations)>20000: raise ValueError('bounded observation list required')
        found={}
        for observation in observations:
            if not isinstance(observation,dict): raise ValueError('observation must be object')
            tid=observation.get('track_id');xy=observation.get('xy')
            if type(tid) is not int or not 0<=tid<2**63 or tid in found: raise ValueError('invalid or duplicate track identity')
            if (not isinstance(xy,list) or len(xy)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in xy)):
                raise ValueError('invalid finite OpenCV xy')
            x,y=map(float,xy)
            if not 0<=x<w or not 0<=y<h: raise ValueError('observation outside image')
            # Same convention used after +0.5 conversion in the existing native mask consumer.
            px,py=math.floor(x+.5),math.floor(y+.5)
            if not 0<=px<w or not 0<=py<h or not mask[py,px]: raise ValueError('observation outside selected mask')
            found[tid]=(x,y);track_views[tid]+=1
        validated.append(dict(sequence_index=seq,input_index=i,frame_id=fid,decode_index=d,
            image_path=str(Path(frame['image_path']).resolve()),mask_path=str(Path(frame['mask_path']).resolve()),
            width=w,height=h,observations=found,mask=mask))
    eligible={tid for tid,n in track_views.items() if n>=MIN_TRACK_VIEWS}
    plans=[]
    for f in validated:
        tids=sorted(set(f['observations'])&eligible)
        xy=np.asarray([f['observations'][t] for t in tids],np.float64).reshape(-1,2)+.5
        keypoints=xy.astype(np.float32)
        if not np.isfinite(keypoints).all(): raise ValueError('nonfinite converted keypoints')
        plans.append(dict(**f,track_ids=tids,keypoints=keypoints,row_by_track={t:r for r,t in enumerate(tids)}))
    pairs=[];excluded=[]
    for a in range(len(plans)):
        for b in range(a+1,min(a+MAX_SEQUENCE_DISTANCE+1,len(plans))):
            common=sorted(set(plans[a]['track_ids'])&set(plans[b]['track_ids']))
            item=dict(sequence_a=a,sequence_b=b,common_tracks=len(common))
            if len(common)>=MIN_PAIR_MATCHES:
                pairs.append(dict(**item,track_ids=common,matches=np.asarray([[plans[a]['row_by_track'][t],plans[b]['row_by_track'][t]] for t in common],np.uint32)))
            else: excluded.append(item)
    if _hash(path)!=binding: raise ValueError('tracks changed during planning')
    return dict(arm=doc['arm'],frames=plans,pairs=pairs,excluded_pairs=excluded,source_bindings=refs,
                track_view_counts=dict(track_views),eligible_track_count=len(eligible),dimensions=dimensions)


def _grid(xy,mask):
    import numpy as np
    yy,xx=np.where(mask);x0,y0=int(xx.min()),int(yy.min());w,h=int(xx.max()+1-x0),int(yy.max()+1-y0)
    result={}
    for n in (4,8,12):
        support=set(zip(((xx+.5-x0)*n/w).astype(int).tolist(),((yy+.5-y0)*n/h).astype(int).tolist()))
        occupied=set()
        for x,y in xy:
            ix,iy=math.floor(x),math.floor(y)
            if 0<=ix<mask.shape[1] and 0<=iy<mask.shape[0] and mask[iy,ix]:
                occupied.add((min(n-1,int((x-x0)*n/w)),min(n-1,int((y-y0)*n/h))))
        result[str(n)]=dict(occupied=len(occupied&support),nonempty_mask_cells=len(support),fraction=len(occupied&support)/len(support))
    return result


def run(tracks_path,output):
    """One native verify and mapper call in a fresh independent output tree."""
    if any(os.environ.get(k)!='2' for k in THREAD_ENV): raise ValueError('numeric thread environment must be 2 before imports')
    import numpy as np
    import pycolmap as pc
    if pc.__version__!='4.2.1': raise ValueError('pinned pycolmap 4.2.1 required')
    plan=prepare(tracks_path);out=Path(output).resolve()
    if out.exists(): raise ValueError('fresh output required')
    for ref in plan['source_bindings']:
        source=Path(ref['path'])
        if source.is_relative_to(out) or out==source: raise ValueError('output overlaps source')
    out.mkdir(parents=True,exist_ok=False)
    sources_before=plan['source_bindings'];w,h=plan['dimensions']
    # 4.2.1 verify_matches owns a matching thread pool and rejects parallel RANSAC.
    verify=pc.TwoViewGeometryOptions();verify.ransac.random_seed=0;verify.ransac.num_threads=1
    mapper=pc.IncrementalPipelineOptions();mapper.num_threads=2;mapper.random_seed=0
    mapper.multiple_models=True;mapper.min_model_size=MIN_MODEL_SIZE;mapper.max_num_models=len(plan['frames'])
    mapper.ba_refine_focal_length=True;mapper.ba_refine_extra_params=True;mapper.ba_refine_principal_point=False
    mapper.max_runtime_seconds=180
    dbpath=out/'database.db';db=pc.Database.open(dbpath)
    camera=pc.Camera.create_from_model_name(1,'SIMPLE_RADIAL',1.2*max(w,h),w,h)
    camera.params=np.array([1.2*max(w,h),w/2,h/2,0.]);camera.has_prior_focal_length=False
    cid=db.write_camera(camera,use_camera_id=True)
    sensor=pc.sensor_t();sensor.type=pc.SensorType.CAMERA;sensor.id=cid
    rig=pc.Rig();rig.rig_id=1;rig.add_ref_sensor(sensor);rid=db.write_rig(rig,use_rig_id=True)
    backcamera=db.read_camera(cid);backrig=db.read_rig(rid)
    if not np.array_equal(backcamera.params,camera.params) or backrig.ref_sensor_id.id!=cid or backrig.ref_sensor_id.type!=pc.SensorType.CAMERA:
        raise ValueError('camera/reference rig sensor readback mismatch')
    images=out/'images';images.mkdir();ledger=[]
    for f in plan['frames']:
        iid=f['sequence_index']+1;name=f'frame_{f["input_index"]:06d}.png'
        # Read immutable source bytes; copy only inside the new native task.
        (images/name).write_bytes(Path(f['image_path']).read_bytes())
        if _hash(images/name)['sha256']!=_hash(f['image_path'])['sha256']:raise ValueError('staged image bytes differ')
        im=pc.Image(name=name,camera_id=cid,image_id=iid);db.write_image(im,use_image_id=True)
        frame=pc.Frame();frame.frame_id=iid;frame.rig_id=rid;frame.add_data_id(pc.data_t(sensor,iid))
        fid=db.write_frame(frame,use_frame_id=True);im.frame_id=fid;db.update_image(im)
        db.write_keypoints(iid,f['keypoints'])
        back=db.read_keypoints(iid);backim=db.read_image(iid);backframe=db.read_frame(fid)
        if not np.array_equal(back,f['keypoints']) or backim.frame_id!=fid or backframe.rig_id!=rid:
            raise ValueError('DB native keypoint/frame readback mismatch')
        data=next(iter(backframe.data_ids)) if backframe.data_ids else None
        if len(backframe.data_ids)!=1 or data.id!=iid or data.sensor_id.id!=cid or data.sensor_id.type!=pc.SensorType.CAMERA:
            raise ValueError('frame data readback mismatch')
        ledger.append(dict(sequence_index=f['sequence_index'],input_index=f['input_index'],decode_index=f['decode_index'],
            frame_id=f['frame_id'],database_image_id=iid,database_frame_id=fid,image_name=name,track_ids=f['track_ids'],keypoint_rows=len(back)))
    pairlines=[];pairledger=[]
    for pair in plan['pairs']:
        a,b=pair['sequence_a']+1,pair['sequence_b']+1;matrix=pair['matches']
        db.write_matches(a,b,matrix)
        if not np.array_equal(db.read_matches(a,b),matrix): raise ValueError('DB matches readback mismatch')
        pairlines.append(ledger[a-1]['image_name']+' '+ledger[b-1]['image_name'])
        pairledger.append(dict(image_a=a,image_b=b,sequence_a=a-1,sequence_b=b-1,raw_matches=len(matrix),track_ids=pair['track_ids']))
    if db.num_descriptors()!=0: raise ValueError('unexpected fabricated descriptors')
    if len(db.read_all_rigs())!=1 or len(db.read_all_frames())!=len(ledger): raise ValueError('rig/frame count mismatch')
    del db
    pairs_path=out/'pairs.txt';pairs_path.write_text('\n'.join(pairlines)+'\n',encoding='utf-8')
    mapper.image_names=[r['image_name'] for r in ledger]
    actual=dict(verification=_plain(verify.todict()),mapping=_plain(mapper.todict()),camera_initial_params=camera.params.tolist(),
        threads=2,cpu_only=True,descriptors_written=False,minimum_track_samples=3,max_pair_sequence_distance=3,minimum_pair_matches=15,
        verification_ransac_threads=1,verification_thread_constraint='pycolmap 4.2.1 verify_matches rejects RANSAC threads>1 inside matching thread pool; numeric environment and mapper remain 2.',
        synthetic_integration_history='Initial synthetic attempt rejected RANSAC threads=2 before mapper; actual native constraint recorded, no geometry thresholds changed.',
        min_model_size_provenance='existing new01 SfM run effective_options.min_model_size=2')
    _save(out/'PLAN.json',dict(schema='balanced-native-plan-v1',arm=plan['arm'],actual_options=actual,frames=ledger,
        candidate_pairs=pairledger,excluded_pairs=plan['excluded_pairs'],source_bindings=sources_before,
        eligible_track_count=plan['eligible_track_count'],geometry_accepted=False,track_identity_truth=False))
    if plan['pairs']:
        pc.verify_matches(dbpath,pairs_path,options=verify)
    db=pc.Database.open(dbpath);verified=[]
    for pair in pairledger:
        geom=db.read_two_view_geometry(pair['image_a'],pair['image_b'])
        inliers=np.asarray(geom.inlier_matches,dtype=np.uint32)
        raw=db.read_matches(pair['image_a'],pair['image_b'])
        if inliers.ndim!=2 or inliers.shape[1]!=2 or not set(map(tuple,inliers))<=set(map(tuple,raw)):
            raise ValueError('verified native row IDs not a subset of stored raw matches')
        verified.append(dict(image_a=pair['image_a'],image_b=pair['image_b'],raw_matches=pair['raw_matches'],
                             inliers=len(inliers),inlier_feature_rows=inliers.tolist(),config=str(geom.config)))
    if db.num_descriptors()!=0: raise ValueError('descriptor table unexpectedly changed')
    del db
    native=out/'mapping';native.mkdir()
    models=pc.incremental_mapping(dbpath,images,native,options=mapper) if plan['pairs'] else {}
    results=[];registered=set()
    for mid,rec in sorted(models.items()):
        folder=out/'models'/str(mid);binary=folder/'binary';text=folder/'text';binary.mkdir(parents=True);text.mkdir()
        rec.write_binary(binary);rec.write_text(text);ply=folder/'points.ply';rec.export_PLY(ply)
        back=pc.Reconstruction(str(binary));backtext=pc.Reconstruction(str(text))
        if set(back.reg_image_ids())!=set(rec.reg_image_ids()) or set(back.points3D)!=set(rec.points3D) or set(backtext.points3D)!=set(rec.points3D):
            raise ValueError('native binary/text roundtrip identity differs')
        for pid,p in rec.points3D.items():
            if not np.array_equal(p.xyz,back.points3D[pid].xyz):raise ValueError('native XYZ roundtrip differs')
        errors=[];tracklengths=[];views=[];point_records=[]
        for p in back.points3D.values():tracklengths.append(len(p.track.elements))
        for pid,point in back.points3D.items():
            point_records.append(dict(point3D_id=int(pid),xyz=point.xyz.tolist(),color=point.color.tolist(),
                track=[dict(native_image_id=int(e.image_id),feature_row=int(e.point2D_idx),
                    input_index=plan['frames'][e.image_id-1]['input_index'],
                    candidate_track_id=plan['frames'][e.image_id-1]['track_ids'][e.point2D_idx]) for e in point.track.elements]))
        for iid in sorted(back.reg_image_ids()):
            f=plan['frames'][iid-1];im=back.images[iid];pose=im.cam_from_world();points=[];observations=[]
            if im.name!=ledger[iid-1]['image_name']:raise ValueError('native image mapping differs')
            registered.add(f['input_index'])
            for row,q in enumerate(im.points2D):
                if not q.has_point3D():continue
                local=pose.rotation.matrix()@back.points3D[q.point3D_id].xyz+pose.translation
                error=float(np.linalg.norm(back.cameras[im.camera_id].img_from_cam(local)-q.xy));errors.append(error);points.append(q.xy.tolist())
                observations.append(dict(feature_row=row,candidate_track_id=f['track_ids'][row],point3D_id=int(q.point3D_id),native_xy=q.xy.tolist(),reprojection_error_px=error))
            views.append(dict(input_index=f['input_index'],frame_id=f['frame_id'],decode_index=f['decode_index'],native_image_id=int(iid),observation_count=len(points),native_xy=points,observations=observations,grid=_grid(points,f['mask'])))
        results.append(dict(model_id=int(mid),binary_path=str(binary),text_path=str(text),ply_path=str(ply),
            registered_views=int(back.num_reg_images()),registered_input_indices=[v['input_index'] for v in views],points=int(back.num_points3D()),
            binary_text_roundtrip_verified=True,track_length=_stats(tracklengths),reprojection_error_px=_stats(errors),
            cameras=[dict(camera_id=int(cid),model=c.model_name,params=c.params.tolist()) for cid,c in back.cameras.items()],views=views,point_records=point_records))
    drift=[r['path'] for r in sources_before if _hash(r['path'])!=r]
    if drift:raise ValueError('source drift after native execution: '+str(drift))
    result=dict(schema='balanced-native-result-v1',status='MODELS_GENERATED_NOT_ACCEPTED' if results else 'NO_MODEL',arm=plan['arm'],
        input_count=len(ledger),unique_registered_count=len(registered),unique_registered_input_indices=sorted(registered),
        models=results,pairs=verified,actual_options=actual,source_unchanged=True,source_drift=[],
        descriptors_written=False,verification_calls=int(bool(plan['pairs'])),mapping_calls=int(bool(plan['pairs'])),
        components_merged=False,scale_status='unknown',geometry_accepted=False,track_identity_truth=False,
        coverage_scope='2D projected observation occupancy in nonempty mask-bbox cells; not physical surface coverage')
    _save(out/'RESULT.json',result)
    return result
