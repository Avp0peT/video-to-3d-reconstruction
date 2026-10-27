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
MIN_TRACK_VIEWS = 2
BASELINE_SEQUENCE_DISTANCE = 3
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


def prepare(tracks_path,policy='all_shared_tracks'):
    return _prepare_sources(tracks_path,policy,MIN_TRACK_VIEWS,False)


def _prepare_sources(tracks_path,policy,min_required,allow_legacy_unannotated):
    """Pure source/identity validation and deterministic row/pair planning."""
    import numpy as np
    from PIL import Image
    _policy(policy)
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
            measured=observation.get('measured',True if allow_legacy_unannotated else None)
            if measured is not True or observation.get('predicted',False) is not False:
                raise ValueError('explicit measured non-predicted observations required')
            for type_key in ('observed_type','observation_type'):
                if type_key in observation and observation[type_key] not in ('measured','observed'):
                    raise ValueError('observation type contradicts actual measured observation')
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
    if doc.get('predicted_observations',0) != 0: raise ValueError('predicted observations forbidden')
    eligible={tid for tid,n in track_views.items() if n>=min_required}
    plans=[]
    for f in validated:
        tids=sorted(set(f['observations'])&eligible)
        xy=np.asarray([f['observations'][t] for t in tids],np.float64).reshape(-1,2)+.5
        keypoints=xy.astype(np.float32)
        if not np.isfinite(keypoints).all(): raise ValueError('nonfinite converted keypoints')
        plans.append(dict(**f,track_ids=tids,keypoints=keypoints,row_by_track={t:r for r,t in enumerate(tids)}))
    outside_window_edges=sum(bool(set(plans[a]['track_ids'])&set(plans[b]['track_ids']))
        for a in range(len(plans)) for b in range(a+BASELINE_SEQUENCE_DISTANCE+1,len(plans)))
    pairs=[];excluded=[]
    for a in range(len(plans)):
        for b in range(a+1,len(plans)):
            common=sorted(set(plans[a]['track_ids'])&set(plans[b]['track_ids']))
            item=dict(sequence_a=a,sequence_b=b,sequence_distance=b-a,
                input_index_distance=plans[b]['input_index']-plans[a]['input_index'],
                decode_index_distance=plans[b]['decode_index']-plans[a]['decode_index'],common_tracks=len(common))
            if len(common)>=MIN_PAIR_MATCHES:
                pairs.append(dict(**item,track_ids=common,matches=np.asarray([[plans[a]['row_by_track'][t],plans[b]['row_by_track'][t]] for t in common],np.uint32)))
            else: excluded.append(item)
    if _hash(path)!=binding: raise ValueError('tracks changed during planning')
    return dict(policy_id=policy,shared_track_edges_outside_window=outside_window_edges,arm=doc['arm'],frames=plans,pairs=pairs,excluded_pairs=excluded,source_bindings=refs,
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


def run(tracks_path,output,policy='all_shared_tracks'):
    """One native verify and mapper call in a fresh independent output tree."""
    if any(os.environ.get(k)!='2' for k in THREAD_ENV): raise ValueError('numeric thread environment must be 2 before imports')
    import numpy as np
    import pycolmap as pc
    if pc.__version__!='4.2.1': raise ValueError('pinned pycolmap 4.2.1 required')
    plan=prepare(tracks_path,policy);out=Path(output).resolve()
    if out.exists(): raise ValueError('fresh output required')
    for ref in plan['source_bindings']:
        source=Path(ref['path'])
        if source.is_relative_to(out) or out==source: raise ValueError('output overlaps source')
    out.mkdir(parents=True,exist_ok=False)
    sources_before=plan['source_bindings'];w,h=plan['dimensions']
    # 4.2.1 verify_matches owns a matching thread pool and rejects parallel RANSAC.
    verify,mapper=make_options(pc,policy,len(plan['frames']))
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
        pairledger.append(dict(image_a=a,image_b=b,sequence_a=a-1,sequence_b=b-1,sequence_distance=pair['sequence_distance'],
            input_index_distance=pair['input_index_distance'],decode_index_distance=pair['decode_index_distance'],raw_matches=len(matrix),track_ids=pair['track_ids']))
    if db.num_descriptors()!=0: raise ValueError('unexpected fabricated descriptors')
    if len(db.read_all_rigs())!=1 or len(db.read_all_frames())!=len(ledger): raise ValueError('rig/frame count mismatch')
    del db
    pairs_path=out/'pairs.txt';pairs_path.write_text('\n'.join(pairlines)+'\n',encoding='utf-8')
    mapper.image_names=[r['image_name'] for r in ledger]
    actual=dict(verification=_plain(verify.todict()),mapping=_plain(mapper.todict()),camera_initial_params=camera.params.tolist(),
        threads=2,cpu_only=True,descriptors_written=False,minimum_track_samples=MIN_TRACK_VIEWS,policy_id=policy,max_pair_sequence_distance=None,pairing_policy='all_actual_shared_track_pairs',minimum_pair_matches=15,
        effective_local_ba=_plain(mapper.get_local_bundle_adjustment().todict()),
        effective_global_ba=_plain(mapper.get_global_bundle_adjustment().todict()),
        effective_mapper=_plain(mapper.get_mapper().todict()),
        radial_policy='fixed_zero_diagnostic_hypothesis_not_calibration' if _policy(policy) else 'optimized_radial',
        camera_model='SIMPLE_RADIAL',camera_has_prior_focal_length=False,principal_point_fixed=True,
        verification_ransac_threads=1,verification_thread_constraint='pycolmap 4.2.1 verify_matches rejects RANSAC threads>1 inside matching thread pool; numeric environment and mapper remain 2.',
        synthetic_integration_history='Initial synthetic attempt rejected RANSAC threads=2 before mapper; actual native constraint recorded, no geometry thresholds changed.',
        min_model_size_provenance='existing new01 SfM run effective_options.min_model_size=2')
    _save(out/'PLAN.json',dict(schema='balanced-native-plan-v1',arm=plan['arm'],actual_options=actual,frames=ledger,
        candidate_pairs=pairledger,excluded_pairs=plan['excluded_pairs'],source_bindings=sources_before,
        eligible_track_count=plan['eligible_track_count'],policy_id=policy,shared_track_edges_outside_window=plan['shared_track_edges_outside_window'],geometry_accepted=False,track_identity_truth=False))
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
    if _policy(policy):
        for model in results:
            for cam in model['cameras']:
                if cam['model']!='SIMPLE_RADIAL' or cam['params'][-1]!=0.:
                    raise ValueError('fixed zero radial diagnostic drifted')
    drift=[r['path'] for r in sources_before if _hash(r['path'])!=r]
    if drift:raise ValueError('source drift after native execution: '+str(drift))
    result=dict(schema='balanced-native-result-v1',policy_id=policy,status='MODELS_GENERATED_NOT_ACCEPTED' if results else 'NO_MODEL',arm=plan['arm'],
        input_count=len(ledger),unique_registered_count=len(registered),unique_registered_input_indices=sorted(registered),
        models=results,pairs=verified,actual_options=actual,source_unchanged=True,source_drift=[],
        descriptors_written=False,verification_calls=int(bool(plan['pairs'])),mapping_calls=int(bool(plan['pairs'])),
        components_merged=False,scale_status='unknown',geometry_accepted=False,track_identity_truth=False,
        coverage_scope='2D projected observation occupancy in nonempty mask-bbox cells; not physical surface coverage')
    _save(out/'RESULT.json',result)
    audit_existing(out/'RESULT.json',out/'GEOMETRY_EVIDENCE.json')
    return result


def _policy(policy):
    if policy not in ('all_shared_tracks','all_shared_tracks_zero_radial'):
        raise ValueError('unknown evidence policy')
    return policy=='all_shared_tracks_zero_radial'


def make_options(pc,policy,num_frames):
    """Construct pinned native options only; no reconstruction is executed.

    Zero radial is a diagnostic model assumption. It does not provide camera
    calibration or a focal prior. Both local/global BA and absolute-pose
    refinement must honor the same radial freeze.
    """
    zero=_policy(policy)
    if pc.__version__!='4.2.1': raise ValueError('pinned pycolmap 4.2.1 required')
    if type(num_frames) is not int or not 3<=num_frames<=192: raise ValueError('bounded frame count')
    verify=pc.TwoViewGeometryOptions(); verify.ransac.random_seed=0; verify.ransac.num_threads=1
    mapper=pc.IncrementalPipelineOptions(); mapper.num_threads=2; mapper.random_seed=0
    mapper.multiple_models=True; mapper.min_model_size=MIN_MODEL_SIZE; mapper.max_num_models=num_frames
    mapper.ba_refine_focal_length=True; mapper.ba_refine_principal_point=False
    mapper.ba_refine_extra_params=not zero
    mapper.mapper.abs_pose_refine_focal_length=True
    mapper.mapper.abs_pose_refine_extra_params=not zero
    mapper.max_runtime_seconds=180
    mapper.triangulation.ignore_two_view_tracks=False
    _validate_refinement_options(policy,_plain(mapper.todict()),
        _plain(mapper.get_mapper().todict()),_plain(mapper.get_local_bundle_adjustment().todict()),
        _plain(mapper.get_global_bundle_adjustment().todict()))
    return verify,mapper


def _validate_refinement_options(policy,mapping,effective_mapper,local_ba,global_ba):
    extra=not _policy(policy)
    if mapping['triangulation']['ignore_two_view_tracks'] is not False: raise ValueError('two-view tracks must be enabled')
    for ba in (local_ba,global_ba):
        if ba['refine_extra_params'] is not extra or ba['refine_focal_length'] is not True or ba['refine_principal_point'] is not False:
            raise ValueError('effective bundle adjustment refinement policy mismatch')
    for mapper in (mapping['mapper'],effective_mapper):
        if mapper['abs_pose_refine_extra_params'] is not extra or mapper['abs_pose_refine_focal_length'] is not True:
            raise ValueError('absolute-pose refinement policy mismatch')
    if mapping['ba_refine_extra_params'] is not extra or mapping['ba_refine_focal_length'] is not True or mapping['ba_refine_principal_point'] is not False:
        raise ValueError('pipeline refinement policy mismatch')


def _radial_observation(camera_xyz,k):
    """Diagnostic only. No geometric acceptance rule is added here."""
    import numpy as np
    local=np.asarray(camera_xyz,float)
    result=dict(normalized_radius_squared=None,radial_scale=None,radial_derivative=None)
    if local.shape!=(3,) or not np.isfinite(local).all() or local[2]==0 or not math.isfinite(float(k)):
        return result
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        radius=float(np.sum((local[:2]/local[2])**2))
        scale=1.+float(k)*radius; derivative=1.+3.*float(k)*radius
    if np.isfinite([radius,scale,derivative]).all():
        result=dict(normalized_radius_squared=radius,radial_scale=scale,radial_derivative=derivative)
    return result


def _radial_summary(observations):
    valid=[o for o in observations if o['normalized_radius_squared'] is not None]
    return dict(scope='Actual native observation domain; diagnostic only, no added rejection gate.',
        observations=len(observations),finite_observations=len(valid),invalid_observations=len(observations)-len(valid),
        normalized_radius_squared=_stats([o['normalized_radius_squared'] for o in valid]),
        radial_scale=_stats([o['radial_scale'] for o in valid]),
        radial_derivative=_stats([o['radial_derivative'] for o in valid]),
        nonpositive_scale_count=sum(o['radial_scale']<=0 for o in valid),
        nonpositive_derivative_count=sum(o['radial_derivative']<=0 for o in valid))


def audit_point_geometry(xyz, observations, max_reprojection_error=4.0, min_angle_deg=1.5):
    """Independent frozen-model consistency computation, not held-out truth.

    Each observation supplies actual native projection, camera-space XYZ and
    camera center. Effective angle is min(theta,180-theta), as in COLMAP.
    """
    import numpy as np
    world=np.asarray(xyz,float)
    finite_xyz=world.shape==(3,) and bool(np.isfinite(world).all())
    reasons=[]; audited=[]; rays=[]; depths=[]; errors=[]
    if not finite_xyz: reasons.append('nonfinite_xyz')
    seen=set()
    for observation in observations:
        identity=(observation['native_image_id'],observation['feature_row'])
        if identity in seen: raise ValueError('duplicate native observation identity')
        seen.add(identity)
        local=np.asarray(observation['camera_xyz'],float)
        center=np.asarray(observation['camera_center'],float)
        projected=np.asarray(observation['projected_xy'],float)
        measured=np.asarray(observation['native_xy'],float)
        finite=(local.shape==(3,) and center.shape==(3,) and projected.shape==(2,) and measured.shape==(2,)
                and np.isfinite(local).all() and np.isfinite(center).all()
                and np.isfinite(projected).all() and np.isfinite(measured).all())
        depth=float(local[2]) if local.shape==(3,) and np.isfinite(local[2]) else None
        error=float(np.linalg.norm(projected-measured)) if finite else None
        if error is not None and not math.isfinite(error): error=None
        if not finite: reasons.append('nonfinite_observation_geometry')
        if depth is None or depth<=0: reasons.append('nonpositive_depth')
        if error is None or error>max_reprojection_error: reasons.append('reprojection_error')
        if depth is not None: depths.append(depth)
        if error is not None: errors.append(error)
        if finite_xyz and finite:
            ray=world-center; length=float(np.linalg.norm(ray))
            if not math.isfinite(length): reasons.append('nonfinite_camera_ray')
            elif length>1e-12: rays.append(ray/length)
            else: reasons.append('zero_camera_ray')
        audited.append(dict(native_image_id=int(observation['native_image_id']),feature_row=int(observation['feature_row']),
            candidate_track_id=int(observation['candidate_track_id']),native_xy=measured.tolist() if measured.shape==(2,) and np.isfinite(measured).all() else None,
            positive_depth=bool(depth is not None and depth>0),depth=depth,reprojection_error_px=error))
    angles=[]
    for a in range(len(rays)):
        for b in range(a+1,len(rays)):
            cosine=float(np.clip(np.dot(rays[a],rays[b]),-1.,1.))
            theta=math.degrees(math.acos(cosine)); angles.append(min(theta,180.-theta))
    angle=max(angles,default=0.)
    views=len({o['native_image_id'] for o in observations})
    if views<2: reasons.append('fewer_than_two_measured_views')
    if angle<min_angle_deg: reasons.append('insufficient_triangulation_angle')
    passed=not reasons
    label=('two_view_provisional' if views==2 else 'multiview_consistent_not_accepted') if passed else 'geometry_evidence_rejected'
    return dict(finite_xyz=finite_xyz,measured_views=views,pass_frozen_geometry=passed,classification=label,
        max_effective_ray_angle_deg=angle,max_reprojection_error_px=max(errors) if errors else None,
        min_depth=min(depths) if depths else None,rejection_reasons=sorted(set(reasons)),observations=audited,
        quality_accepted=False,track_identity_truth=False)


def audit_existing(native_result_path,output_json):
    """Read frozen native binaries; write exclusive independent component audit.

    Never maps, triangulates, modifies a reconstruction, or deletes raw points.
    """
    import numpy as np
    import pycolmap as pc
    if pc.__version__!='4.2.1': raise ValueError('pinned pycolmap 4.2.1 required')
    result_path=Path(native_result_path).resolve(strict=True); result_binding=_hash(result_path)
    doc=json.loads(result_path.read_text(encoding='utf-8-sig'))
    if doc.get('schema')!='balanced-native-result-v1': raise ValueError('unknown native result schema')
    mapping=doc['actual_options']['mapping']; reproj=float(mapping['mapper']['filter_max_reproj_error'])
    angle=float(mapping['mapper']['filter_min_tri_angle'])
    if reproj!=4. or angle!=1.5: raise ValueError('frozen evidence thresholds must remain 4px / 1.5deg')
    plan_path=result_path.parent/'PLAN.json'
    plan_binding=_hash(plan_path)
    plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
    if plan.get('schema')!='balanced-native-plan-v1': raise ValueError('unknown native plan schema')
    if plan['actual_options']!=doc['actual_options']: raise ValueError('result/plan actual option drift')
    policy_id=doc.get('policy_id','legacy_three_sample_baseline')
    legacy_control=policy_id=='geometry_evidence'
    if not legacy_control: _policy(policy_id)
    required=2
    if legacy_control:
        if doc['actual_options']['max_pair_sequence_distance']!=3 or mapping['triangulation']['ignore_two_view_tracks'] is not False:
            raise ValueError('unexpected legacy control pairing/triangulation policy')
    else:
        _validate_refinement_options(policy_id,mapping,doc['actual_options']['effective_mapper'],
            doc['actual_options']['effective_local_ba'],doc['actual_options']['effective_global_ba'])
        if doc['actual_options']['max_pair_sequence_distance'] is not None: raise ValueError('unexpected restricted pair window')
    if plan['actual_options']['minimum_track_samples']!=required: raise ValueError('source track gate/policy mismatch')
    source_bindings=plan.get('source_bindings')
    if not isinstance(source_bindings,list) or not source_bindings: raise ValueError('native plan source bindings required')
    for binding in source_bindings:
        if _hash(binding['path'])!=binding: raise ValueError('frozen source binding drift')
    rebuilt=_prepare_sources(source_bindings[0]['path'],
        'all_shared_tracks' if legacy_control else policy_id,required,False)
    if rebuilt['source_bindings']!=source_bindings: raise ValueError('source binding order/content differs from actual TRACKS')
    source_pairs=[p for p in rebuilt['pairs'] if not legacy_control or p['sequence_distance']<=3]
    if len(plan['candidate_pairs'])!=len(source_pairs): raise ValueError('submitted shared pair graph differs from sources')
    for expected_pair,submitted in zip(source_pairs,plan['candidate_pairs']):
        expected=dict(image_a=expected_pair['sequence_a']+1,image_b=expected_pair['sequence_b']+1,
            sequence_a=expected_pair['sequence_a'],sequence_b=expected_pair['sequence_b'],
            sequence_distance=expected_pair['sequence_distance'],input_index_distance=expected_pair['input_index_distance'],
            decode_index_distance=expected_pair['decode_index_distance'],raw_matches=expected_pair['common_tracks'],
            track_ids=expected_pair['track_ids'])
        if legacy_control:
            for key in ('sequence_distance','input_index_distance','decode_index_distance'): del expected[key]
        if submitted!=expected: raise ValueError('source shared-track pair/row identity mismatch')
    excluded=[p for p in rebuilt['excluded_pairs'] if not legacy_control or p['sequence_distance']<=3]
    if legacy_control: excluded=[{k:p[k] for k in ('sequence_a','sequence_b','common_tracks')} for p in excluded]
    if plan['excluded_pairs']!=excluded: raise ValueError('excluded under-15 source pair evidence mismatch')
    if len(rebuilt['frames'])!=len(plan['frames']): raise ValueError('source frame count differs from native plan')
    frame_rows={int(f['database_image_id']):f for f in plan['frames']}
    if len(frame_rows)!=len(plan['frames']): raise ValueError('duplicate plan image identity')
    source_rows={}
    for source,ledger in zip(rebuilt['frames'],plan['frames']):
        for key in ('sequence_index','input_index','decode_index','frame_id'):
            if source[key]!=ledger[key]: raise ValueError('actual TRACKS frame identity differs from native plan')
        if ledger['database_image_id']!=source['sequence_index']+1 or ledger['track_ids']!=source['track_ids']:
            raise ValueError('native plan track rows differ from actual TRACKS')
        if ledger['keypoint_rows']!=len(source['keypoints']): raise ValueError('source keypoint row count mismatch')
        source_rows[ledger['database_image_id']]=source
    bindings=[result_binding,plan_binding,*source_bindings]; components=[]
    for model in doc['models']:
        binary=Path(model['binary_path']).resolve(strict=True)
        if Path(output_json).resolve().is_relative_to(binary): raise ValueError('audit output overlaps frozen binary')
        files=sorted(p for p in binary.iterdir() if p.is_file())
        if not files: raise ValueError('empty native binary')
        bindings.extend(_hash(p) for p in files)
        rec=pc.Reconstruction(str(binary))
        if not legacy_control and _policy(policy_id):
            for camera in rec.cameras.values():
                if camera.model_name!='SIMPLE_RADIAL' or camera.params[-1]!=0.:
                    raise ValueError('frozen radial parameter drift in binary')
        records={int(p['point3D_id']):p for p in model['point_records']}
        if len(records)!=len(model['point_records']) or set(records)!=set(rec.points3D): raise ValueError('point identity readback mismatch')
        views={int(v['native_image_id']):v for v in model['views']}
        if set(views)!=set(rec.reg_image_ids()): raise ValueError('registered view identity mismatch')
        for iid,image in rec.images.items():
            source=source_rows.get(iid); ledger=frame_rows.get(iid)
            if source is None or image.name!=ledger['image_name']: raise ValueError('native image/source frame mismatch')
            actual_xy=np.asarray([p.xy for p in image.points2D],np.float64).reshape(-1,2)
            expected_xy=source['keypoints'].astype(np.float64)
            if not np.array_equal(actual_xy,expected_xy): raise ValueError('all native feature coordinates differ from actual TRACKS float32(xy+0.5)')
            if iid in views:
                for key in ('input_index','decode_index','frame_id'):
                    if views[iid][key]!=source[key]: raise ValueError('registered result frame differs from actual TRACKS')
        feature_map={}
        for iid,view in views.items():
            for observation in view['observations']:
                key=(iid,int(observation['feature_row']))
                if key in feature_map: raise ValueError('duplicate result feature row')
                feature_map[key]=observation
        points=[]; radial_observations=[]
        for pid,point in sorted(rec.points3D.items()):
            record=records[pid]
            if not np.array_equal(np.asarray(record['xyz']),point.xyz): raise ValueError('point XYZ readback mismatch')
            expected={(int(o['native_image_id']),int(o['feature_row'])):int(o['candidate_track_id']) for o in record['track']}
            actual={(int(e.image_id),int(e.point2D_idx)) for e in point.track.elements}
            if len(expected)!=len(record['track']) or set(expected)!=actual: raise ValueError('point track feature row mismatch')
            observations=[]
            for iid,row in sorted(actual):
                image=rec.images[iid]; feature=image.points2D[row]; stored=feature_map.get((iid,row))
                if stored is None or int(stored['point3D_id'])!=pid or int(stored['candidate_track_id'])!=expected[(iid,row)]:
                    raise ValueError('candidate/native identity binding mismatch')
                ledger=frame_rows.get(iid)
                if ledger is None or not 0<=row<len(ledger['track_ids']) or ledger['track_ids'][row]!=expected[(iid,row)]:
                    raise ValueError('native feature row differs from frozen source track ledger')
                if image.name!=ledger['image_name']: raise ValueError('native image/source ledger identity mismatch')
                if not feature.has_point3D() or int(feature.point3D_id)!=pid or not np.array_equal(feature.xy,stored['native_xy']):
                    raise ValueError('feature location/point identity mismatch')
                pose=image.cam_from_world(); rotation=pose.rotation.matrix(); translation=pose.translation
                local=rotation@point.xyz+translation; projected=rec.cameras[image.camera_id].img_from_cam(local)
                camera=rec.cameras[image.camera_id]
                if camera.model_name!='SIMPLE_RADIAL': raise ValueError('radial diagnostic requires actual SIMPLE_RADIAL model')
                radial=_radial_observation(local,float(camera.params[-1]))
                radial_observations.append(radial)
                observations.append(dict(native_image_id=iid,feature_row=row,candidate_track_id=expected[(iid,row)],
                    native_xy=feature.xy.tolist(),camera_xyz=local.tolist(),camera_center=(-rotation.T@translation).tolist(),
                    projected_xy=np.asarray(projected if projected is not None else [float('nan'),float('nan')]).tolist(),**radial))
            audited=audit_point_geometry(point.xyz,observations,reproj,angle)
            for stored,source in zip(audited['observations'],observations):
                stored.update({key:source[key] for key in ('normalized_radius_squared','radial_scale','radial_derivative')})
            points.append(dict(point3D_id=int(pid),xyz=point.xyz.tolist() if np.isfinite(point.xyz).all() else None,**audited))
        counts=Counter(p['classification'] for p in points)
        components.append(dict(model_id=model['model_id'],binary_path=str(binary),registered_views=len(views),raw_points=len(points),
            pass_points=sum(p['pass_frozen_geometry'] for p in points),classifications=dict(counts),points=points,
            radial_observation_domain=_radial_summary(radial_observations),components_merged=False,quality_accepted=False))
    if any(_hash(b['path'])!=b for b in bindings): raise ValueError('source drift during frozen model audit')
    audit=dict(schema='frozen-native-geometry-evidence-v1',policy_id=doc.get('policy_id','legacy_three_sample_baseline'),
        method='Independent implementation of frozen-model consistency checks; not held-out validation.',
        native_result_path=str(result_path),thresholds=dict(max_reprojection_error_px=reproj,min_effective_ray_angle_deg=angle,strict_positive_depth=True),
        components=components,components_merged=False,quality_accepted=False,track_identity_truth=False,source_bindings=bindings,source_unchanged=True)
    _save(output_json,audit)
    return audit
