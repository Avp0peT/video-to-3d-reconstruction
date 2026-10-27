"""One bounded current-state intrinsic-constraint counterfactual, never calibration truth."""
import json
import os
from pathlib import Path
import sqlite3
import time

from . import registration_probe as probe


def _assign(obj,name,value):
    if not hasattr(obj,name):raise ValueError('recorded option absent: '+name)
    current=getattr(obj,name)
    if isinstance(value,str) and not isinstance(current,str):
        token=value.rsplit('.',1)[-1]
        if not hasattr(type(current),token):raise ValueError('recorded enum cannot be resolved: '+name)
        value=getattr(type(current),token)
    setattr(obj,name,value)


def _pipeline_options(pc,recorded):
    options=pc.IncrementalPipelineOptions()
    names=('num_threads','random_seed','min_focal_length_ratio','max_focal_length_ratio','max_extra_param')
    for name in names:_assign(options,name,recorded[name])
    for name,value in recorded.items():
        if name.startswith('ba_') and isinstance(value,(str,int,float,bool)):_assign(options,name,value)
    for name,value in recorded['triangulation'].items():
        if not isinstance(value,(str,int,float,bool)):raise ValueError('non-scalar triangulation option unsupported')
        _assign(options.triangulation,name,value)
    return options


def _full_state(rec):
    state=probe._model_state(rec)
    state['images']={str(i):dict(name=image.name,camera_id=int(image.camera_id),has_pose=bool(image.has_pose),
        observations=[dict(xy=probe._plain(p.xy),point3D_id=int(p.point3D_id) if p.has_point3D() else None) for p in image.points2D])
        for i,image in rec.images.items()}
    return state


def _residuals(rec):
    """Bounded original-camera projections, not held-out identity evidence."""
    import numpy as np
    records=[];total=0;angles=[];angle_total=0
    centers={}
    for iid in sorted(rec.reg_image_ids()):
        image=rec.image(iid);pose=image.cam_from_world();camera=rec.cameras[image.camera_id]
        centers[int(iid)]=-np.asarray(pose.rotation.matrix()).T@np.asarray(pose.translation)
        for index,p in enumerate(image.points2D):
            if not p.has_point3D():continue
            total+=1
            if len(records)>=2048:continue
            xyz=rec.points3D[int(p.point3D_id)].xyz;cam=np.asarray(pose*xyz,dtype=float)
            predicted=camera.img_from_cam(cam)
            valid=predicted is not None and np.asarray(predicted).shape==(2,) and np.isfinite(predicted).all()
            records.append(dict(image_id=int(iid),feature_row=index,point3D_id=int(p.point3D_id),raw_pixels=probe._plain(p.xy),
                positive_depth=bool(cam[2]>0),camera_depth=float(cam[2]),projected_pixels=probe._plain(predicted) if valid else None,
                residual_pixels=float(np.linalg.norm(np.asarray(predicted)-p.xy)) if valid else None))
    for pid,point in sorted(rec.points3D.items()):
        views=sorted({int(e.image_id) for e in point.track.elements if int(e.image_id) in centers})
        for i,a in enumerate(views):
            for b in views[i+1:]:
                angle_total+=1
                if len(angles)>=2048:continue
                u=np.asarray(point.xyz)-centers[a];v=np.asarray(point.xyz)-centers[b];norm=float(np.linalg.norm(u)*np.linalg.norm(v))
                angle=None if not norm else float(np.degrees(np.arccos(np.clip(float(u@v)/norm,-1.,1.))))
                angles.append(dict(point3D_id=int(pid),image_ids=[a,b],degrees=angle))
    return dict(observations=records,total_observations=total,omitted_observations=total-len(records),
        angles=angles,total_angle_slots=angle_total,omitted_angle_slots=angle_total-len(angles),
        angle_definition='Camera-center to candidate XYZ rays; conditional geometry, not independent measured identity.',
        world_units='unknown',accuracy_acceptance=False)


def _gate(rec,mapper,cache,options,target_id):
    from . import registration_gate as gate
    thresholds=(options.min_focal_length_ratio,options.max_focal_length_ratio,options.max_extra_param)
    checks={};native_checks={}
    for cid,camera in rec.cameras.items():
        check=gate.bogus_camera(int(camera.width),int(camera.height),probe._plain(camera.params),*thresholds)
        native_value=bool(camera.has_bogus_params(*thresholds))
        if check['bogus']!=native_value:raise ValueError('native/manual camera predicate mismatch')
        checks[int(cid)]=check;native_checks[int(cid)]=native_value
    target=rec.image(target_id);links=[]
    if len(target.points2D)>20000:raise ValueError('bounded target rows required')
    for row in range(len(target.points2D)):
        for corr in cache.correspondence_graph.extract_correspondences(target_id,row):
            image=rec.image(int(corr.image_id));index=int(corr.point2D_idx)
            if not 0<=index<len(image.points2D):raise ValueError('native correspondence row invalid')
            p=image.points2D[index];pid=int(p.point3D_id) if p.has_point3D() else None
            if pid is not None and (pid not in rec.points3D or not any(int(e.image_id)==int(corr.image_id) and int(e.point2D_idx)==index for e in rec.points3D[pid].track.elements)):
                raise ValueError('native correspondence track not reciprocal')
            links.append(dict(source_row=row,target_image_id=int(corr.image_id),target_point_row=index,point3D_id=pid,
                camera_id=int(image.camera_id),has_pose=bool(image.has_pose)))
            if len(links)>200000:raise ValueError('bounded correspondence links required')
    visible=int(mapper.observation_manager.num_visible_points3D(target_id))
    return dict(camera_checks=checks,native_camera_checks=native_checks,links=links,
        visible_points3D=visible,visible_correspondences=int(mapper.observation_manager.num_visible_correspondences(target_id)),
        summary=gate.summarize_gate(checks,links,int(options.abs_pose_min_num_inliers),visible_count=visible))


def run_control(run_json,input_indices_pair,target_index,output):
    """Outer process must impose wall/memory bounds; this runs no automatic retries."""
    if type(input_indices_pair) is not list or len(input_indices_pair)!=2 or len(set(input_indices_pair))!=2 or any(type(i) is not int or i<0 for i in input_indices_pair) or type(target_index) is not int or target_index<0 or target_index in input_indices_pair:
        raise ValueError('two distinct integer input indices and distinct target required')
    run_path=Path(run_json)
    if not run_path.is_absolute():raise ValueError('absolute recorded run required')
    run_path=run_path.resolve(strict=True);root=run_path.parent;output=Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):raise ValueError('fresh nonoverlapping output required')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='initial-pair-calibration-control-v1',status='PREPARING',input_indices_pair=input_indices_pair,target_index=target_index,
        counts=dict(initial_estimation=0,initial_registration=0,triangulation=0,global_ba=0,target_registration=0),arms=[],source_bindings=[],
        identity_verified=False,geometry_acceptance=False,native_injected=False,native_model_writes=0,
        scope='Current counterfactual from DB initial camera and selected pair, not a replay of historical model1.',
        calibration_claim=False,k0_assumption='DB heuristic initial k=0, not independently measured calibration.')
    def save(name,value):
        with (output/name).open('x',encoding='utf-8') as stream:
            json.dump(probe._plain(value),stream,ensure_ascii=False,allow_nan=False,indent=2);stream.flush();os.fsync(stream.fileno())
    def event(phase,**data):
        with (output/'EVENTS.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(phase=phase,monotonic=time.monotonic(),counts=dict(result['counts']),**data))+'\n');stream.flush();os.fsync(stream.fileno())
    def bind(path):result['source_bindings'].append(probe._binding(path))
    def state(name,rec):
        payload=dict(model=_full_state(rec),projection_diagnostics=_residuals(rec));save(name,payload);return payload['model']
    database=None;active_mapper=None;active_begun=False
    event('CREATED')
    try:
        from . import parallel_photo_audit as native
        bind(run_path);run=native.read_json(run_path)
        if run.get('backend',{}).get('name')!='pycolmap' or run.get('backend',{}).get('version')!='4.2.1' or run.get('contract_version')!='sfm-native-object-v1':raise ValueError('frozen pycolmap4.2.1 object source required')
        fmap=native.under(root,'frames_map.jsonl');bind(fmap)
        rows=[native.loads(l) for l in fmap.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        ids=[];selected_names={}
        for index in input_indices_pair+[target_index]:
            found=[r for r in rows if r['input_index']==index]
            if len(found)!=1 or found[0].get('database_image_id') is None or found[0].get('upstream_io_run_id')!=run.get('upstream_io_run_id'):raise ValueError('unique bound native input identities required')
            ids.append(int(found[0]['database_image_id']))
            selected_names[ids[-1]]=found[0]['staged_image_name']
        if len(set(ids))!=3:raise ValueError('distinct native images required')
        pair=ids[:2];target_id=ids[2];result.update(database_pair=pair,target_database_image_id=target_id)
        actual=native.under(root,'logs/incremental_mapping.result.json');bind(actual);recorded=native.read_json(actual)['actual_options']['pipeline']
        db_path=native.under(root,'database.db');bind(db_path)
        wal=Path(str(db_path)+'-wal')
        result['source_wal_bytes_before']=wal.stat().st_size if wal.exists() else 0
        if result['source_wal_bytes_before']:raise ValueError('nonempty source WAL refused')
        event('SNAPSHOT_ENTER')
        snapshot=output/'snapshot.db'
        with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True) as source:
            source.execute('PRAGMA query_only=ON')
            with sqlite3.connect(snapshot) as copy:source.backup(copy)
        result['snapshot_before_native']=probe._binding(snapshot)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
        pc=probe._load_pycolmap()
        if pc.__version__!='4.2.1':raise ValueError('installed version differs')
        pc.logging.verbose_level=2;pc.logging.logtostderr=True
        database=pc.Database.open(snapshot);cache_options=pc.DatabaseCacheOptions()
        for name in ('min_num_matches','ignore_watermarks','image_names'):setattr(cache_options,name,set(recorded[name]) if name=='image_names' else recorded[name])
        cache=pc.DatabaseCache.create(database,cache_options);mapper_options,result['mapper_options']=probe._mapper_options(pc,recorded['mapper'],recorded)
        pipeline_options=_pipeline_options(pc,recorded);result['pipeline_options']=probe._plain(pipeline_options.todict())
        rec=pc.Reconstruction();active_mapper=pc.IncrementalMapper(cache);active_mapper.begin_reconstruction(rec);active_begun=True
        initial=state('COMMON_DB_INITIAL.json',rec)
        for ident,name in selected_names.items():
            if not rec.exists_image(ident) or rec.image(ident).name!=name:raise ValueError('native image name differs from frame map')
            if rec.rig(rec.image(ident).frame.rig_id).num_sensors()!=1:raise ValueError('ordinary single-sensor rig required')
        for cid,camera in rec.cameras.items():
            if str(camera.model_name)!='SIMPLE_RADIAL' or len(camera.params)!=4 or float(camera.params[3])!=0.:raise ValueError('SIMPLE_RADIAL DB initial k must be zero; no reset permitted')
        result['counts']['initial_estimation']=1;event('INITIAL_ESTIMATE_ENTER')
        pose=active_mapper.estimate_initial_two_view_geometry(mapper_options,pair[0],pair[1])
        state('COMMON_POST_ESTIMATE.json',rec)
        if pose is None:raise ValueError('specified pair unsuitable for initialization; no alternate pair or relaxed retry')
        result['counts']['initial_registration']=1;event('INITIAL_REGISTER_ENTER')
        active_mapper.register_initial_image_pair(mapper_options,pair[0],pair[1],pose)
        tri=pipeline_options.get_triangulation();tri.min_angle=mapper_options.init_min_tri_angle;result['triangulation_options']=probe._plain(tri.todict())
        for image_id in pair:
            result['counts']['triangulation']+=1;event('TRIANGULATE_ENTER',image_id=image_id)
            active_mapper.triangulate_image(tri,image_id)
        common=state('COMMON_POST_TRIANGULATION.json',rec)
        for camera in rec.cameras.values():
            if float(camera.params[3])!=0.:raise ValueError('post-estimate common k is not zero; no reset permitted')
        if common['registered_image_ids']!=sorted(pair) or len(common['points3D'])<mapper_options.abs_pose_min_num_inliers:raise ValueError('common initial model lacks two registered views or required points')
        active_begun=False;active_mapper.end_reconstruction(False);active_mapper=None
        if not probe._preserved(common,probe._model_state(rec)):raise ValueError('common model changed during mapper cleanup')
        immutable_common=_full_state(rec);save('COMMON_COPY_SOURCE.json',dict(model=immutable_common))
        def cache_cameras():
            return {str(i):dict(model=c.model_name,params=probe._plain(c.params),width=int(c.width),height=int(c.height)) for i,c in cache.cameras.items()}
        immutable_cache=cache_cameras();save('CACHE_CAMERAS_BEFORE_ARMS.json',immutable_cache)
        def guard_common():
            if _full_state(rec)!=immutable_common:raise ValueError('branch copy is not isolated: immutable common model changed')
            if cache_cameras()!=immutable_cache:raise ValueError('branch modified shared cache cameras')
        effective_ba=[];prepared_states=[]
        for label,refine_extra in (('FREE_K',True),('FIX_DB_K0',False)):
            branch=pc.Reconstruction(rec);active_mapper=pc.IncrementalMapper(cache)
            copied=_full_state(branch)
            if copied!=_full_state(rec):raise ValueError('memory copy changed common model')
            active_mapper.begin_reconstruction(branch);active_begun=True
            guard_common()
            pre=state(label+'_PRE_BA.json',branch)
            if not probe._preserved(common,pre):raise ValueError('branch begin changed common cameras/poses/3D')
            prepared_states.append(pre)
            if len(prepared_states)>1 and pre!=prepared_states[0]:raise ValueError('two arms have different prepared initial states')
            ba=pipeline_options.get_global_bundle_adjustment()
            if not bool(ba.refine_extra_params):raise ValueError('recorded baseline must optimize extra params')
            ba.refine_extra_params=refine_extra
            ba_dict=probe._plain(ba.todict());effective_ba.append(ba_dict)
            compare=dict(ba_dict);compare.pop('refine_extra_params',None)
            if len(effective_ba)>1:
                original=dict(effective_ba[0]);original.pop('refine_extra_params',None)
                if compare!=original:raise ValueError('BA arms differ beyond refine_extra_params')
            arm=dict(name=label,ba_options=ba_dict,refine_extra_params=refine_extra,registration_status='NOT_RUN',identity_verified=False,geometry_acceptance=False,
                fixed_k_scope='BA extra-parameter constraint only; native absolute-pose options remain unchanged and later registration may update intrinsics.',stage_summaries={})
            result['arms'].append(arm)
            def summary(snapshot,filename):
                return dict(snapshot_file=filename,registered_image_ids=snapshot['registered_image_ids'],points3D_count=len(snapshot['points3D']),
                    cameras={cid:camera['params'] for cid,camera in snapshot['cameras'].items()})
            arm['stage_summaries']['pre_ba']=summary(pre,label+'_PRE_BA.json')
            def arm_state(stage,suffix):
                filename=label+'_'+suffix+'.json';snapshot=state(filename,branch)
                arm['stage_summaries'][stage]=summary(snapshot,filename);return snapshot
            result['counts']['global_ba']+=1;event('GLOBAL_BA_ENTER',arm=label)
            arm['native_ba_result']=bool(active_mapper.adjust_global_bundle(mapper_options,ba))
            guard_common()
            if not refine_extra and any(float(c.params[3])!=0. for c in branch.cameras.values()):raise ValueError('fixed-k BA changed zero extra parameter')
            arm_state('post_ba','POST_BA')
            event('NORMALIZE_ENTER',arm=label);branch.normalize();arm_state('post_normalize','POST_NORMALIZE')
            event('FILTER_POINTS_ENTER',arm=label);arm['filter_points_returned_observation_count']=int(active_mapper.filter_points(mapper_options));arm_state('post_filter_points','POST_FILTER_POINTS')
            event('FILTER_FRAMES_ENTER',arm=label);arm['filtered_frames']=int(active_mapper.filter_frames(mapper_options));arm_state('post_filter_frames','POST_FILTER_FRAMES')
            if len(branch.reg_image_ids())>=2 and len(branch.points3D)>=int(mapper_options.abs_pose_min_num_inliers) and branch.exists_image(target_id) and not branch.image(target_id).has_pose:
                arm['pregate']=_gate(branch,active_mapper,cache,mapper_options,target_id);save(label+'_PREGATE.json',arm['pregate'])
                result['counts']['target_registration']+=1;event('TARGET_REGISTER_ENTER',arm=label)
                arm['native_registration_result']=bool(active_mapper.register_next_image(mapper_options,target_id));arm['registration_status']='ONE_NATIVE_CALL_COMPLETED'
                arm_state('post_target_registration','POST_TARGET_REGISTRATION')
            else:arm['registration_not_run_reason']='Post-filter model lacks two registered views/required points/unposed target.'
            active_begun=False;active_mapper.end_reconstruction(False);active_mapper=None
            guard_common()
        result['status']='CONTROL_COMPLETED_NOT_GEOMETRY_ACCEPTED'
    except Exception as exc:
        result.update(status='CONTROL_FAILED',error_type=type(exc).__name__,error=str(exc));event('ERROR',error=str(exc))
    finally:
        cleanup=[]
        if active_mapper is not None and active_begun:
            try:active_mapper.end_reconstruction(False);cleanup.append('mapper.end_reconstruction(False)')
            except Exception as exc:cleanup.append(str(exc));result['status']='CONTROL_FAILED'
        if database is not None:
            try:database.close();cleanup.append('database.close')
            except Exception as exc:cleanup.append(str(exc));result['status']='CONTROL_FAILED'
        drift=[]
        for receipt in result['source_bindings']:
            try:
                if probe._binding(receipt['path'])!=receipt:drift.append(receipt['path'])
            except Exception:drift.append(receipt['path'])
        final_wal=Path(str(root/'database.db')+'-wal')
        result['source_wal_bytes_after']=final_wal.stat().st_size if final_wal.exists() else 0
        if 'source_wal_bytes_before' in result and result['source_wal_bytes_after']!=result['source_wal_bytes_before'] and str(final_wal) not in drift:drift.append(str(final_wal))
        result.update(cleanup=cleanup,source_unchanged=not drift,source_drift_paths=drift)
        if drift:result['status']='SOURCE_DRIFT_REFUSED'
        event('FINALIZED',status=result['status']);save('CONTROL.json',result)
    return probe._plain(result)
