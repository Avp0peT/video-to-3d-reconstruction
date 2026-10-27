"""Single-call native registration gate probe on a DB snapshot and memory model.

Heavy imports are lazy. This module never writes a native reconstruction and
never retries, falls back, or calls an estimator separately.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import time


def _load_pycolmap():return importlib.import_module('pycolmap')


def _plain(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if isinstance(value,set):return sorted(_plain(v) for v in value)
    if value is None or isinstance(value,(str,bool,int,float)):return value
    return str(value)


def _binding(path):
    path=Path(path).resolve(strict=True)
    if path.stat().st_size>160*1024*1024:raise ValueError('source file exceeds 160 MiB bound')
    digest=hashlib.sha256();size=0
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):size+=len(chunk);digest.update(chunk)
    return dict(path=str(path),bytes=size,sha256=digest.hexdigest())


def _pose(image):
    if not image.has_pose:return None
    pose=image.cam_from_world()
    return dict(R=_plain(pose.rotation.matrix()),t=_plain(pose.translation))


def _model_state(reconstruction):
    return dict(registered_image_ids=sorted(int(i) for i in reconstruction.reg_image_ids()),
        cameras={str(i):dict(camera_id=int(i),width=int(c.width),height=int(c.height),params=_plain(c.params)) for i,c in reconstruction.cameras.items()},
        registered_poses={str(i):_pose(reconstruction.image(i)) for i in reconstruction.reg_image_ids()},
        points3D={str(i):dict(xyz=_plain(p.xyz),color=_plain(p.color),error=float(p.error),
            track=[[int(e.image_id),int(e.point2D_idx)] for e in p.track.elements]) for i,p in reconstruction.points3D.items()})


def _preserved(before,after):
    return (before['registered_image_ids']==after['registered_image_ids'] and
        all(after[group].get(k)==v for group in ('cameras','registered_poses','points3D') for k,v in before[group].items()) and
        set(before['points3D'])==set(after['points3D']))


def _mapper_options(pc,recorded,pipeline):
    options=pc.IncrementalMapperOptions()
    applied={}
    for name,value in recorded.items():
        if not hasattr(options,name):raise ValueError('recorded mapper option absent from installed API: '+name)
        current=getattr(options,name)
        if isinstance(current,set):
            if value!='set()' and value!=[]:raise ValueError('unsupported nonempty recorded constant set')
            value=set()
        elif isinstance(value,str) and not isinstance(current,str):
            enum_name=value.rsplit('.',1)[-1]
            if not hasattr(type(current),enum_name):raise ValueError('recorded enum option cannot be resolved: '+name)
            value=getattr(type(current),enum_name)
        setattr(options,name,value);applied[name]=_plain(value)
    overrides=dict(num_threads=int(pipeline['num_threads']),random_seed=int(pipeline['random_seed']),
        min_focal_length_ratio=pipeline['min_focal_length_ratio'],max_focal_length_ratio=pipeline['max_focal_length_ratio'],max_extra_param=pipeline['max_extra_param'])
    if overrides['num_threads']!=2 or overrides['random_seed']!=0:raise ValueError('frozen pipeline threads=2/seed=0 required')
    for name,value in overrides.items():setattr(options,name,value)
    if not options.check():raise ValueError('installed mapper options check refused recorded configuration')
    return options,dict(recorded=recorded,applied_before_pipeline_overrides=applied,pipeline_overrides=overrides,effective=_plain(options.todict()))


def run_probe(run_json,input_index,model_id,output):
    """Run inside an outer bounded process/Job. Exactly one registration maximum."""
    output=Path(output).resolve()
    if output.exists():raise ValueError('fresh registration probe output required')
    run_path=Path(run_json)
    if not run_path.is_absolute() or type(input_index) is not int or input_index<0 or not isinstance(model_id,str) or not model_id:
        raise ValueError('absolute run, integer frame and explicit model ID required')
    root=run_path.resolve(strict=True).parent
    if output.is_relative_to(root) or root.is_relative_to(output):raise ValueError('probe output overlaps immutable source')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema='single-registration-probe-v1',status='PREPARING',input_index=input_index,model_id=model_id,
        registration_call_count=0,extra_estimator_call_count=0,fallback_call_count=0,native_model_writes=0,
        source_bindings=[],identity_verified=False,geometry_acceptance=False,native_injected=False,
        stage_estimation='UNKNOWN',stage_refinement='UNKNOWN',historical_replay=False,
        scope='One current saved-model conditional native attempt; not an exact historical replay.')
    def save(name,data):
        with (output/name).open('x',encoding='utf-8') as stream:
            json.dump(_plain(data),stream,ensure_ascii=False,allow_nan=False,indent=2);stream.flush();os.fsync(stream.fileno())
    def event(phase,**data):
        with (output/'EVENTS.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(phase=phase,monotonic=time.monotonic(),registration_call_count=manifest['registration_call_count'],**data),allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    def bind(path):
        receipt=_binding(path);manifest['source_bindings'].append(receipt);return receipt
    database=None;mapper=None;begun=False
    event('PROBE_CREATED')
    try:
        from . import parallel_photo_audit as native
        from . import registration_gate as gate
        bind(run_path);run=native.read_json(run_path)
        if run.get('contract_version')!='sfm-native-object-v1' or run.get('backend',{}).get('version')!='4.2.1' or run.get('backend',{}).get('name')!='pycolmap':raise ValueError('frozen pycolmap4.2.1 object run required')
        fmap=native.under(root,'frames_map.jsonl');bind(fmap)
        rows=[native.loads(l) for l in fmap.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        selected=[r for r in rows if r['input_index']==input_index]
        if len(selected)!=1 or selected[0].get('upstream_io_run_id')!=run.get('upstream_io_run_id') or selected[0].get('database_image_id') is None:raise ValueError('unique DB-bound frame identity required')
        frame=selected[0];image_id=int(frame['database_image_id']);manifest['database_image_id']=image_id
        models=[m for m in run['models'] if str(m['model_id'])==model_id]
        if len(models)!=1 or models[0].get('usable') is not True or models[0].get('roundtrip_verified') is not True:raise ValueError('unique usable verified model required')
        model=models[0];binary=native.under(root,model['binary_path'])
        for name in ('cameras.bin','images.bin','points3D.bin','rigs.bin','frames.bin'):bind(native.under(root,str((binary/name).relative_to(root))))
        actual_path=native.under(root,'logs/incremental_mapping.result.json');bind(actual_path)
        pipeline=native.read_json(actual_path)['actual_options']['pipeline']
        db_path=native.under(root,'database.db');bind(db_path)
        wal=Path(str(db_path)+'-wal')
        if wal.exists() and wal.stat().st_size:raise ValueError('source database has nonempty WAL; no frozen snapshot proof')
        manifest['source_wal_bytes_before']=0
        snapshot=output/'snapshot.db'
        event('SNAPSHOT_BEGIN')
        with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True) as source:
            source.execute('PRAGMA query_only=ON')
            with sqlite3.connect(snapshot) as target:source.backup(target)
        manifest['snapshot_before_native_open']=_binding(snapshot)
        event('SNAPSHOT_COMPLETE')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
        pc=_load_pycolmap()
        if pc.__version__!='4.2.1':raise ValueError('installed pycolmap version differs')
        manifest['installed_backend']=dict(version=pc.__version__,file=pc.__file__)
        pc.logging.verbose_level=2;pc.logging.logtostderr=True
        database=pc.Database.open(snapshot)
        cache_options=pc.DatabaseCacheOptions()
        for name in ('min_num_matches','ignore_watermarks','image_names'):setattr(cache_options,name,set(pipeline[name]) if name=='image_names' else pipeline[name])
        manifest['database_cache_options']=_plain(cache_options.todict())
        cache=pc.DatabaseCache.create(database,cache_options)
        reconstruction=pc.Reconstruction(binary)
        before=_model_state(reconstruction)
        if before['registered_image_ids']!=sorted(model['registered_image_ids']) or len(before['points3D'])!=model['num_points3D'] or image_id in before['registered_image_ids']:raise ValueError('original registered model identity/count differs or target already posed')
        options,manifest['mapper_options']=_mapper_options(pc,pipeline['mapper'],pipeline)
        mapper=pc.IncrementalMapper(cache)
        event('BEGIN_RECONSTRUCTION_ENTER')
        mapper.begin_reconstruction(reconstruction);begun=True
        after_begin=_model_state(reconstruction)
        manifest['memory_before_begin']=before;manifest['memory_after_begin']=after_begin
        if not _preserved(before,after_begin):raise ValueError('begin_reconstruction altered original camera/pose/3D state')
        if not reconstruction.exists_image(image_id):raise ValueError('target missing from prepared reconstruction')
        target=reconstruction.image(image_id)
        if target.name!=frame['staged_image_name'] or target.has_pose:raise ValueError('prepared target name/pose identity differs')
        sensor_count=int(reconstruction.rig(target.frame.rig_id).num_sensors())
        if sensor_count!=1:raise ValueError('probe is restricted to the ordinary single-sensor rig branch')
        manifest['target_rig_sensor_count']=sensor_count
        camera_checks={};native_checks={}
        thresholds=(options.min_focal_length_ratio,options.max_focal_length_ratio,options.max_extra_param)
        for cid,camera in reconstruction.cameras.items():
            if camera.model_name!='SIMPLE_RADIAL':raise ValueError('probe requires SIMPLE_RADIAL cameras')
            check=gate.bogus_camera(int(camera.width),int(camera.height),_plain(camera.params),*thresholds)
            native_check=bool(camera.has_bogus_params(*thresholds))
            if check['bogus']!=native_check:raise ValueError('manual/native bogus-camera predicates differ')
            camera_checks[int(cid)]=check;native_checks[int(cid)]=native_check
        manager=mapper.observation_manager
        visible=int(manager.num_visible_points3D(image_id));visible_correspondences=int(manager.num_visible_correspondences(image_id))
        links=[];graph=cache.correspondence_graph
        if len(target.points2D)>20000:raise ValueError('target row count exceeds frozen bound')
        for source_row in range(len(target.points2D)):
            for corr in graph.extract_correspondences(image_id,source_row):
                other_id=int(corr.image_id);other_row=int(corr.point2D_idx)
                other=reconstruction.image(other_id)
                if not 0<=other_row<len(other.points2D):raise ValueError('correspondence target row outside native image')
                point=other.points2D[other_row]
                point_id=int(point.point3D_id) if point.has_point3D() else None
                if point_id is not None and point_id not in reconstruction.points3D:raise ValueError('native point correspondence missing point3D')
                if point_id is not None and not any(int(e.image_id)==other_id and int(e.point2D_idx)==other_row for e in reconstruction.points3D[point_id].track.elements):
                    raise ValueError('nonreciprocal native point correspondence')
                links.append(dict(source_row=source_row,target_image_id=other_id,target_point_row=other_row,point3D_id=point_id,camera_id=int(other.camera_id),has_pose=bool(other.has_pose)))
                if len(links)>200000:raise ValueError('correspondence link budget exceeded')
        summary=gate.summarize_gate(camera_checks,links,int(options.abs_pose_min_num_inliers),visible_count=visible)
        pregate=dict(camera_checks=camera_checks,native_camera_checks=native_checks,links=links,gate_summary=summary,
            visible_points3D=visible,visible_correspondences=visible_correspondences,target_pose_before=_pose(target),
            original_memory_preserved_after_begin=True,mapper_options=manifest['mapper_options'])
        save('PREGATE.json',pregate);manifest['pregate_path']=str(output/'PREGATE.json');manifest['gate_summary']=summary
        event('PREGATE_FLUSHED')
        manifest['registration_call_count']=1;event('REGISTER_NEXT_IMAGE_ENTER')
        native_result=mapper.register_next_image(options,image_id)
        manifest['native_registration_result']=bool(native_result);event('REGISTER_NEXT_IMAGE_RETURNED',native_result=bool(native_result))
        manifest['memory_after_registration']=_model_state(reconstruction)
        manifest['target_pose_before']=pregate['target_pose_before'];manifest['target_pose_after']=_pose(reconstruction.image(image_id))
        if summary['structure_based']['gate_reachable'] is False:
            manifest['stage_estimation']='NOT_REACHED';manifest['stage_refinement']='NOT_REACHED'
            manifest['stage_inference_basis']='Current prepared state fails the visible-count or collected-2D3D precheck; native call/log and frozen 4.2.1 source must be reviewed together, not interpreted as historical proof.'
            if native_result:raise ValueError('native result contradicts predicted pre-estimation gate refusal')
        manifest['status']='ONE_NATIVE_CALL_COMPLETED_NOT_GEOMETRY_ACCEPTED'
    except Exception as exc:
        manifest.update(status='PROBE_FAILED',error_type=type(exc).__name__,error=str(exc));event('PROBE_ERROR',error_type=type(exc).__name__,error=str(exc))
    finally:
        cleanup=[]
        if mapper is not None and begun:
            try:event('MAPPER_CLEANUP_ENTER');mapper.end_reconstruction(False);cleanup.append('mapper.end_reconstruction(False)')
            except Exception as exc:cleanup.append('mapper cleanup error: '+str(exc));manifest['status']='PROBE_FAILED'
        if database is not None:
            try:database.close();cleanup.append('database.close')
            except Exception as exc:cleanup.append('database close error: '+str(exc));manifest['status']='PROBE_FAILED'
        manifest['cleanup']=cleanup
        drift=[]
        for receipt in manifest['source_bindings']:
            try:
                if _binding(receipt['path'])!=receipt:drift.append(receipt['path'])
            except Exception:drift.append(receipt['path'])
        manifest['source_unchanged']=not drift;manifest['source_drift_paths']=drift
        if 'source_wal_bytes_before' in manifest:
            wal=Path(str(root/'database.db')+'-wal')
            manifest['source_wal_bytes_after']=wal.stat().st_size if wal.exists() else 0
            if manifest['source_wal_bytes_after']:
                drift.append(str(wal));manifest['source_unchanged']=False
        if drift:manifest['status']='SOURCE_DRIFT_REFUSED'
        event('FINALIZED',status=manifest['status']);save('MANIFEST.json',manifest)
    return _plain(manifest)
