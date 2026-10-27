"""Experimental learned candidate geometry; no physical identity or CAD acceptance.

prepare is pure (read-only, no native/GPU imports). worker must be externally
supervised with a wall/memory/output limit and process-tree reclamation.
"""
from pathlib import Path
from collections import Counter
import json
import math
import os
from .connectivity_native import (_hash, _save, _plain, _stats, _grid, _policy,
    make_options, audit_point_geometry, _radial_observation, _radial_summary,
    THREAD_ENV, MIN_PAIR_MATCHES, MIN_TRACK_VIEWS)


def keyframe_indices(count):
    if type(count) is not int or not 3 <= count <= 1000:
        raise ValueError('3..1000 frames required')
    size=min(count,64)
    return [i*(count-1)//(size-1) for i in range(size)]


def prepare(source_run):
    import numpy as np
    from PIL import Image
    from .learned_point_contract import classify_points
    source=Path(source_run).resolve(strict=True)
    names=('INPUTS','MASKS','TRACKS','QUERIES','PLAN','RESULT')
    refs=[_hash(source/(n+'.json')) for n in names]
    def invalid(v): raise ValueError('nonfinite JSON '+v)
    docs={n:json.loads((source/(n+'.json')).read_text(encoding='utf-8-sig'),parse_constant=invalid) for n in names}
    inputs,masks,tracks,queries,upstream,result=[docs[n] for n in names]
    if (upstream.get('schema')!='learned-point-plan-v1' or upstream.get('model')!='TAPNext++'
        or tracks.get('status')!='TRACKS_GENERATED_NOT_IDENTITY_VERIFIED'
        or tracks.get('physical_identity_verified') is not False
        or tracks.get('geometry_accepted') is not False
        or result.get('status')!='EXPERIMENT_COMPLETE_NOT_IDENTITY_VERIFIED'
        or result.get('source_drift') or result.get('implementation_drift')):
        raise ValueError('completed unverified learned source required')
    frames=inputs['frames']; ms=masks['frames']; ts=tracks['frames']; qs=queries['points']
    selected=keyframe_indices(len(frames))
    if len(ms)!=len(frames) or len(ts)!=len(frames) or upstream.get('frames')!=len(frames):
        raise ValueError('frame count mismatch')
    if upstream.get('dimensions')!=[576,1024] or upstream.get('query_count')!=len(qs) or not 1<=len(qs)<=512:
        raise ValueError('source dimensions/query count mismatch')
    tids=[q['track_id'] for q in qs]
    if any(type(t) is not int or not 0<=t<2**63 for t in tids) or len(set(tids))!=len(tids):
        raise ValueError('duplicate/invalid query identity')
    decodes=[f['decode_index'] for f in frames]
    if any(type(d) is not int or d<0 for d in decodes) or decodes!=sorted(set(decodes)):
        raise ValueError('nonordered decode identity')
    plans=[]; excluded_candidates=Counter()
    for i,(f,m,t) in enumerate(zip(frames,ms,ts)):
        if m['decode_index']!=f['decode_index'] or t['decode_index']!=f['decode_index']:
            raise ValueError('decode identity mismatch')
        if i not in selected: continue
        image=Path(f['path']).resolve(strict=True)
        ref=_hash(image)
        if ref['sha256']!=f['sha256'] or m.get('raw_sha256')!=ref['sha256'] or Path(m['raw_path']).resolve()!=image:
            raise ValueError('raw image binding mismatch')
        refs.append(ref)
        with Image.open(image) as im:
            if im.mode!='RGB' or im.size!=(576,1024): raise ValueError('RGB image domain mismatch')
        mask=None
        if m.get('current_observed') is True:
            if m.get('status')!='OK': raise ValueError('observed mask must be OK')
            ref=_hash(m['mask_path']); refs.append(ref)
            if ref['sha256']!=m['mask_sha256']: raise ValueError('mask hash mismatch')
            with Image.open(m['mask_path']) as im:
                if im.mode!='L' or im.size!=(576,1024): raise ValueError('mask domain mismatch')
                mask=np.asarray(im).copy()
            if not np.isin(mask,[0,255]).all() or not mask.any(): raise ValueError('binary nonempty mask required')
        points=t['points']
        if [p['track_id'] for p in points]!=tids: raise ValueError('query/point identity mismatch')
        statuses=classify_points([p['xy'] for p in points],[p['visibility_probability'] for p in points],
            [p['certainty'] for p in points],mask,576,1024)
        found={}
        for p,status in zip(points,statuses):
            if p.get('status')!=status or p.get('supported') is not (status=='VISIBLE_CANDIDATE'):
                raise ValueError('candidate classification drift')
            if status=='VISIBLE_CANDIDATE': found[p['track_id']]=p['xy']
            else: excluded_candidates[status]+=1
        ids=sorted(found)
        plans.append(dict(sequence_index=len(plans),input_index=i,decode_index=f['decode_index'],
            frame_id='raw_decode_'+str(f['decode_index']),image_path=str(image),mask_path=m.get('mask_path'),
            width=576,height=1024,mask=mask>0 if mask is not None else np.zeros((1024,576),bool),
            track_ids=ids,keypoints=(np.asarray([found[t] for t in ids],np.float64).reshape(-1,2)+.5).astype(np.float32),
            row_by_track={t:r for r,t in enumerate(ids)}))
    counts=Counter(t for f in plans for t in f['track_ids'])
    for f in plans:
        keep=[r for r,t in enumerate(f['track_ids']) if counts[t]>=MIN_TRACK_VIEWS]
        f['track_ids']=[f['track_ids'][r] for r in keep]
        f['keypoints']=f['keypoints'][keep]
        f['row_by_track']={t:r for r,t in enumerate(f['track_ids'])}
    pairs=[];excluded=[]
    for a,fa in enumerate(plans):
        for b in range(a+1,len(plans)):
            fb=plans[b]; common=sorted(set(fa['track_ids'])&set(fb['track_ids']))
            item=dict(sequence_a=a,sequence_b=b,sequence_distance=b-a,input_index_distance=fb['input_index']-fa['input_index'],
                decode_index_distance=fb['decode_index']-fa['decode_index'],common_tracks=len(common))
            if len(common)>=MIN_PAIR_MATCHES:
                pairs.append(dict(**item,track_ids=common,matches=np.asarray([[fa['row_by_track'][t],fb['row_by_track'][t]] for t in common],np.uint32)))
            else: excluded.append(item)
    refs.extend(_hash(p) for p in (Path(__file__),Path(__file__).with_name('connectivity_native.py'),Path(__file__).with_name('learned_point_contract.py')))
    if any(_hash(r['path'])!=r for r in refs): raise ValueError('source changed during planning')
    return dict(schema='learned-candidate-geometry-source-v1',arm='learned_candidates',frames=plans,pairs=pairs,
        excluded_pairs=excluded,source_bindings=refs,dimensions=(576,1024),eligible_track_count=len(set(t for f in plans for t in f['track_ids'])),
        shared_track_edges_outside_window=sum(p['sequence_distance']>3 for p in pairs),selected_input_indices=selected,
        excluded_candidate_counts=dict(excluded_candidates),physical_identity_verified=False,geometry_accepted=False)


def worker(source_run,output):
    policy='all_shared_tracks'
    """One native verify and mapper call in a fresh independent output tree."""
    if any(os.environ.get(k)!='2' for k in THREAD_ENV): raise ValueError('numeric thread environment must be 2 before imports')
    import numpy as np
    import pycolmap as pc
    if pc.__version__!='4.2.1': raise ValueError('pinned pycolmap 4.2.1 required')
    plan=prepare(source_run);out=Path(output).resolve()
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
        keyframe_selection='uniform integer endpoint-inclusive max64; fixed before geometry',selected_input_indices=plan['selected_input_indices'],
        coordinate_conversion='float32(float64(OpenCV xy)+0.5)',physical_identity_verified=False,
        excluded_candidate_counts=plan['excluded_candidate_counts'],threads=2,cpu_only=True,descriptors_written=False,minimum_track_samples=MIN_TRACK_VIEWS,policy_id=policy,max_pair_sequence_distance=None,pairing_policy='all_actual_shared_track_pairs',minimum_pair_matches=15,
        effective_local_ba=_plain(mapper.get_local_bundle_adjustment().todict()),
        effective_global_ba=_plain(mapper.get_global_bundle_adjustment().todict()),
        effective_mapper=_plain(mapper.get_mapper().todict()),
        radial_policy='fixed_zero_diagnostic_hypothesis_not_calibration' if _policy(policy) else 'optimized_radial',
        camera_model='SIMPLE_RADIAL',camera_has_prior_focal_length=False,principal_point_fixed=True,
        verification_ransac_threads=1,verification_thread_constraint='pycolmap 4.2.1 verify_matches rejects RANSAC threads>1 inside matching thread pool; numeric environment and mapper remain 2.',
        candidate_semantics='Image-conditioned learned estimates; physical identity unverified',
        min_model_size_provenance='existing new01 SfM run effective_options.min_model_size=2')
    _save(out/'SOURCE_BINDINGS.json',sources_before)
    _save(out/'PLAN.json',dict(schema='learned-candidate-native-plan-v1',arm=plan['arm'],actual_options=actual,frames=ledger,
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
    result=dict(schema='learned-candidate-native-result-v1',policy_id=policy,status='RECONSTRUCTED_NOT_QUALITY_ACCEPTED' if results else 'NO_MODEL',arm=plan['arm'],
        input_count=len(ledger),unique_registered_count=len(registered),unique_registered_input_indices=sorted(registered),
        models=results,pairs=verified,actual_options=actual,source_unchanged=True,source_drift=[],
        physical_identity_verified=False,quality_status='NOT_EVALUATED',cad_status='NOT_EVALUATED',
        descriptors_written=False,verification_calls=int(bool(plan['pairs'])),mapping_calls=int(bool(plan['pairs'])),
        components_merged=False,scale_status='unknown',geometry_accepted=False,track_identity_truth=False,
        coverage_scope='2D projected observation occupancy in nonempty mask-bbox cells; not physical surface coverage')
    evidence=audit_models(pc,plan,results)
    if any(_hash(r['path'])!=r for r in sources_before): raise ValueError('source drift during native audit')
    _save(out/'GEOMETRY_EVIDENCE.json',evidence)
    _save(out/'RESULT.json',result)
    return result



def audit_models(pc,plan,models):
    """Inspect immutable exported native models without remapping or filtering."""
    import numpy as np
    components=[]
    for model in models:
        rec=pc.Reconstruction(model['binary_path']); points=[];radials=[]
        if len(rec.cameras)!=1: raise ValueError('single shared camera required')
        camera=next(iter(rec.cameras.values()))
        if camera.model_name!='SIMPLE_RADIAL' or (camera.width,camera.height)!=(576,1024):
            raise ValueError('camera model/domain drift')
        for iid,image in rec.images.items():
            expected=plan['frames'][iid-1]['keypoints'].astype(np.float64)
            actual=np.asarray([p.xy for p in image.points2D],np.float64).reshape(-1,2)
            if not np.array_equal(actual,expected): raise ValueError('native candidate xy/row drift')
        for pid,point in rec.points3D.items():
            obs=[]
            for e in point.track.elements:
                image=rec.images[e.image_id]; pose=image.cam_from_world(); r=pose.rotation.matrix();v=pose.translation
                local=r@point.xyz+v; q=image.points2D[e.point2D_idx]; projection=camera.img_from_cam(local)
                radial=_radial_observation(local,float(camera.params[-1]));radials.append(radial)
                obs.append(dict(native_image_id=int(e.image_id),feature_row=int(e.point2D_idx),
                    candidate_track_id=plan['frames'][e.image_id-1]['track_ids'][e.point2D_idx],native_xy=q.xy.tolist(),
                    camera_xyz=local.tolist(),camera_center=(-r.T@v).tolist(),
                    projected_xy=np.asarray(projection if projection is not None else [float('nan')]*2).tolist()))
            evidence=audit_point_geometry(point.xyz,obs)
            evidence['candidate_views']=evidence.pop('measured_views')
            evidence['rejection_reasons']=['fewer_than_two_candidate_views' if x=='fewer_than_two_measured_views' else x for x in evidence['rejection_reasons']]
            points.append(dict(point3D_id=int(pid),**evidence))
        components.append(dict(model_id=model['model_id'],points=points,radial_observation_domain=_radial_summary(radials),
            focal_px=float(camera.params[0]),focal_over_max_dimension=float(camera.params[0])/1024,
            radial_k=float(camera.params[-1]),camera_params=camera.params.tolist(),
            native_binary_bindings=[_hash(p) for p in sorted(Path(model['binary_path']).iterdir()) if p.is_file()]))
    return dict(schema='learned-candidate-geometry-evidence-v1',components=components,
        physical_identity_verified=False,quality_accepted=False,scale_status='unknown',cad_status='NOT_EVALUATED')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=['worker'])
    parser.add_argument('--source-run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    worker(args.source_run,args.output)
