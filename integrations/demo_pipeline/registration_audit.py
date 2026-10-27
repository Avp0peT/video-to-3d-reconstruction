"""Read-only saved COLMAP 4.2.1 registration evidence; never estimate a pose."""
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import re
import sqlite3
import struct

from . import parallel_photo_audit as native
from . import approximate_photo_bridge as bridge
from .registration_gate import bogus_camera,summarize_gate


def parse_registration_attempts(text,image_id):
    """Observed log events only; absent detailed reasons remain UNKNOWN."""
    if text is None:return dict(status='UNKNOWN_LOG_UNAVAILABLE',observed_attempt_count=None,attempts=[])
    lines=text.splitlines();attempts=[]
    start=re.compile(r'Registering image #'+str(image_id)+r'\s')
    boundary=re.compile(r'Registering image #|Retriangulation and|Keeping successful|Finding good initial')
    for i,line in enumerate(lines):
        if not start.search(line):continue
        end=i+1
        while end<len(lines) and not boundary.search(lines[end]):end+=1
        chunk=lines[i:end];joined='\n'.join(chunk)
        points=re.search(r'Image sees (\d+) / (\d+) points',joined)
        correspondences=re.search(r'Image sees (\d+) / (\d+) correspondences',joined)
        reasons=[s for s in chunk if any(x in s for x in ('Insufficient number of 2D-3D','Absolute pose estimation failed','Pose refinement failed','insufficient number of points'))]
        attempts.append(dict(line_1based=i+1,lines=chunk,status='FAILED' if 'Could not register' in joined else 'UNKNOWN',
            structureless_fallback='structure-less fallback' in joined,
            logged_visible_points=None if points is None else dict(numerator=int(points[1]),denominator=int(points[2])),
            logged_fallback_correspondences=None if correspondences is None else dict(numerator=int(correspondences[1]),denominator=int(correspondences[2])),
            internal_reason_status='RECORDED' if reasons else 'UNKNOWN',internal_reason_lines=reasons,
            absolute_pose_inlier_count=None,absolute_pose_inlier_status='UNKNOWN_NOT_RETAINED'))
    return dict(status='OBSERVED_LOG_EVENTS_ONLY',observed_attempt_count=len(attempts),attempts=attempts,
        scope='Observed log events do not certify log completeness or absence of unlogged attempts.')


def audit_registration(run_json,input_index,output,refs):
    if type(input_index) is not int or input_index<0:raise ValueError('nonnegative non-bool input index required')
    run_path=Path(run_json)
    if not run_path.is_absolute():raise ValueError('absolute SfM run.json required')
    run_path=run_path.resolve(strict=True);root=run_path.parent;output=Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):raise ValueError('fresh nonoverlapping audit output required')
    local=[];stream_paths=set()
    def stream_binding(path):
        path=Path(path).resolve(strict=True)
        if path.stat().st_size>160*1024*1024:raise ValueError('database exceeds bounded 160 MiB input')
        digest=hashlib.sha256();size=0
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):size+=len(chunk);digest.update(chunk)
        return dict(path=str(path),bytes=size,sha256=digest.hexdigest())
    def bind(path):
        receipt=stream_binding(path) if Path(path).resolve() in stream_paths else native.binding(path)
        for old in refs+local:
            if Path(old['path']).resolve()==Path(receipt['path']) and old!=receipt:raise ValueError('previous source binding drift')
        local.append(receipt);return receipt
    def optional(relative):
        path=root/relative
        if not path.exists():return None
        path=native.under(root,relative);bind(path);return path
    bind(run_path);run=native.read_json(run_path)
    if run.get('contract_version')!='sfm-native-object-v1' or run.get('backend',{}).get('name')!='pycolmap' or run.get('backend',{}).get('version')!='4.2.1':
        raise ValueError('frozen object SfM / pycolmap 4.2.1 evidence required')
    map_path=native.under(root,'frames_map.jsonl');bind(map_path);mapped={};by_db={}
    for line in map_path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        row=native.loads(line);index=row['input_index']
        if type(index) is not int or index in mapped or not run.get('upstream_io_run_id') or row.get('upstream_io_run_id')!=run['upstream_io_run_id']:raise ValueError('frame map identity differs')
        mapped[index]=row
        if row.get('database_image_id') is not None:
            ident=row['database_image_id']
            if type(ident) is not int or ident in by_db:raise ValueError('duplicate database frame identity unsupported')
            by_db[ident]=row
    if input_index not in mapped:raise ValueError('input index missing from full frame ledger')
    selected=mapped[input_index];ident=selected.get('database_image_id')
    if ident is None:
        result=dict(status='NO_DATABASE_IMAGE',input_index=input_index,frame=selected,registration_failure_cause='NOT_APPLICABLE_NO_DATABASE_IMAGE',attempt_evidence=dict(status='NOT_APPLICABLE',observed_attempt_count=None,attempts=[]),pair_evidence=[],final_snapshot_correspondences=[])
    else:
        database=native.under(root,'database.db');stream_paths.add(database.resolve());bind(database)
        connection=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True);connection.execute('PRAGMA query_only=ON')
        try:
            db_images={i:dict(image_id=i,name=n,camera_id=c) for i,n,c in connection.execute('SELECT image_id,name,camera_id FROM images')}
            if ident not in db_images or db_images[ident]['name']!=selected['staged_image_name']:raise ValueError('database image/frame map name differs')
            for image_id,image in db_images.items():
                if image_id not in by_db or image['name']!=by_db[image_id]['staged_image_name']:raise ValueError('database/map coverage differs')
            camera_rows={}
            for cid,model,width,height,data,prior in connection.execute('SELECT camera_id,model,width,height,params,prior_focal_length FROM cameras'):
                if model!=2 or len(data)!=32:raise ValueError('frozen SIMPLE_RADIAL database required')
                camera_rows[cid]=dict(camera_id=cid,model='SIMPLE_RADIAL',width=width,height=height,params=list(struct.unpack('<4d',data)),prior_focal_length=bool(prior))
            pair_evidence=[];pair_matches={}
            for pid,count,cols,data,config in connection.execute('SELECT pair_id,rows,cols,data,config FROM two_view_geometries'):
                a,b=divmod(pid,2147483647)
                if ident not in (a,b):continue
                if a not in db_images or b not in db_images or cols!=2 or count<0 or count>100000 or len(data or b'')!=count*8:raise ValueError('malformed native geometric match matrix')
                raw=connection.execute('SELECT rows FROM matches WHERE pair_id=?',(pid,)).fetchone()
                target=b if a==ident else a
                matches=list(struct.iter_unpack('<II',data or b''));matches=matches if a==ident else [(j,i) for i,j in matches]
                if count:pair_matches[target]=matches
                pair_evidence.append(dict(target_database_image_id=target,target_input_index=by_db[target]['input_index'],
                    target_camera_id=db_images[target]['camera_id'],shared_database_camera=db_images[target]['camera_id']==db_images[ident]['camera_id'],
                    raw_match_count=None if raw is None else raw[0],geometric_inlier_count=count,configuration_code=config,
                    configuration_name={0:'UNDEFINED',1:'DEGENERATE',2:'CALIBRATED',3:'UNCALIBRATED',4:'PLANAR',5:'PANORAMIC',6:'PLANAR_OR_PANORAMIC',7:'WATERMARK',8:'MULTIPLE',9:'CALIBRATED_RIG'}.get(config,'UNKNOWN')))
            feature=connection.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(ident,)).fetchone()
            if feature is None or feature[1] not in (2,4,6) or not 0<=feature[0]<=20000 or len(feature[2] or b'')!=feature[0]*feature[1]*4:raise ValueError('invalid source feature matrix')
            for target,matches in pair_matches.items():
                if any(i>=feature[0] for i,j in matches):raise ValueError('geometric match source row outside matrix')
            final_snapshots=[]
            models=run.get('models',[])
            if len(models)>21 or len({str(m['model_id']) for m in models})!=len(models):raise ValueError('bounded unique models required')
            for model in models:
                relevant=sorted(set(model['registered_image_ids'])&set(pair_matches))
                if not relevant:continue
                if model.get('usable') is not True or model.get('roundtrip_verified') is not True:raise ValueError('usable roundtrip model required')
                binary=native.under(root,model['binary_path']);paths={n:native.under(root,str((binary/n).relative_to(root))) for n in ('images.bin','points3D.bin','cameras.bin')}
                for path in paths.values():bind(path)
                images,points=native.read_native(paths['images.bin'],paths['points3D.bin']);cameras,poses=bridge.read_calibration(paths['cameras.bin'],paths['images.bin'])
                if sorted(images)!=sorted(model['registered_image_ids']) or set(poses)!=set(images) or len(points)!=model['num_points3D']:raise ValueError('model identity/count differs')
                source_points=defaultdict(set);per_target=[];gate_links=[]
                for target in relevant:
                    image=images[target];pose=poses[target]
                    if image['name']!=db_images[target]['name'] or image['camera_id']!=db_images[target]['camera_id'] or pose['name']!=image['name'] or pose['camera_id']!=image['camera_id']:raise ValueError('native/database image-camera identity differs')
                    matrix=connection.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(target,)).fetchone()
                    if matrix is None or matrix[1] not in (2,4,6) or len(matrix[2] or b'')!=matrix[0]*matrix[1]*4 or matrix[0]!=len(image['observations']):raise ValueError('target native/database feature matrix differs')
                    available=[]
                    for source_row,target_row in pair_matches[target]:
                        if target_row>=matrix[0]:raise ValueError('target feature row outside matrix')
                        obs=image['observations'][target_row];xy=struct.unpack_from('<ff',matrix[2],target_row*matrix[1]*4)
                        if tuple(obs[:2])!=xy:raise ValueError('native/database target row xy differs')
                        gate_links.append(dict(source_row=source_row,target_image_id=target,target_point_row=target_row,
                            point3D_id=None if obs[2]==-1 else int(obs[2]),camera_id=image['camera_id'],has_pose=True))
                        if obs[2]!=-1:available.append((source_row,obs[2]));source_points[source_row].add(obs[2])
                    per_target.append(dict(target_image_id=target,matched_final_native_point_observations=len(available),
                        distinct_source_rows=len({i for i,p in available}),distinct_point3D_ids=len({p for i,p in available})))
                final_snapshots.append(dict(model_id=str(model['model_id']),per_target=per_target,
                    distinct_source_rows=len(source_points),distinct_point3D_ids=len({p for v in source_points.values() for p in v}),
                    source_rows_with_multiple_point3D_ids=sum(len(v)>1 for v in source_points.values()),
                    cameras=[dict(camera_id=cid,width=cam['width'],height=cam['height'],params=list(cam['params'])) for cid,cam in sorted(cameras.items())],
                    posed_correspondence_links=gate_links,
                    scope='Final exported native snapshot, not the state or estimator inliers of any actual registration attempt.'))
        finally:connection.close()
        log_path=optional('logs/incremental_mapping.stderr.txt');attempts=parse_registration_attempts(None if log_path is None else log_path.read_text(encoding='utf-8-sig'),ident)
        actual_path=optional('logs/incremental_mapping.result.json');spec_path=optional('logs/incremental_mapping.spec.json')
        actual=None if actual_path is None else native.read_json(actual_path).get('actual_options',{}).get('pipeline')
        spec=None if spec_path is None else native.read_json(spec_path)
        threshold=None if actual is None else actual.get('max_extra_param');suspicions=[]
        for snapshot in final_snapshots:
            snapshot['static_registration_gate']=dict(status='UNKNOWN_MAPPER_OPTIONS',native_registration_executions=0)
            if actual is not None and all(k in actual for k in ('min_focal_length_ratio','max_focal_length_ratio','max_extra_param','min_num_matches','ignore_watermarks')) and 'mapper' in actual and 'abs_pose_min_num_inliers' in actual['mapper']:
                checks={c['camera_id']:bogus_camera(c['width'],c['height'],c['params'],actual['min_focal_length_ratio'],actual['max_focal_length_ratio'],actual['max_extra_param']) for c in snapshot['cameras']}
                cache_targets={p['target_database_image_id'] for p in pair_evidence if p['geometric_inlier_count']>=actual['min_num_matches'] and not(actual['ignore_watermarks'] and p['configuration_code']==7)}
                links=sorted([v for v in snapshot['posed_correspondence_links'] if v['target_image_id'] in cache_targets],key=lambda v:(v['source_row'],v['target_image_id'],v['target_point_row']))
                snapshot['static_registration_gate']=summarize_gate(checks,links,actual['mapper']['abs_pose_min_num_inliers'])
                snapshot['static_registration_gate'].update(camera_checks=checks,cache_min_num_matches=actual['min_num_matches'],cache_ignore_watermarks=actual['ignore_watermarks'],
                    link_scope='Saved direct geometric links to posed images in this final model, using recorded cache pair filters. Not a historical-attempt snapshot.',
                    historical_failure_proven=False)
            for cam in snapshot['cameras']:
                if threshold is not None and abs(cam['params'][3])>threshold:
                    suspicions.append(dict(kind='FINAL_RADIAL_PARAM_EXCEEDS_RECORDED_MAPPER_GATE',model_id=snapshot['model_id'],
                        camera_id=cam['camera_id'],final_k=cam['params'][3],max_extra_param=threshold,
                        causality='SUSPECTED_MECHANISM_NOT_PROVEN',attempt_camera_snapshot_available=False))
        result=dict(status='READ_ONLY_REGISTRATION_EVIDENCE_COMPLETE',input_index=input_index,database_image_id=ident,frame=selected,
            database_image=db_images[ident],database_camera=camera_rows[db_images[ident]['camera_id']],feature_row_count=feature[0],
            pair_evidence=sorted(pair_evidence,key=lambda r:r['target_database_image_id']),final_snapshot_correspondences=final_snapshots,
            attempt_evidence=attempts,recorded_camera_mode=None if spec is None else spec.get('camera_mode'),
            recorded_mapper_options=None if actual is None else actual.get('mapper'),suspected_mechanisms=suspicions,
            registration_failure_cause='UNKNOWN_INTERNAL_BRANCH_NOT_RETAINED',occlusion_cause='UNKNOWN',motion_cause='UNKNOWN')
    for receipt in local:
        if Path(receipt['path']) in stream_paths:
            if stream_binding(receipt['path'])!=receipt:raise ValueError('database source binding drift')
        else:bridge.verify_binding(receipt)
    attempts=result['attempt_evidence']
    result['counts']=dict(positive_geometric_pairs=sum(p['geometric_inlier_count']>0 for p in result['pair_evidence']),
        stored_geometric_inliers=sum(p['geometric_inlier_count'] for p in result['pair_evidence']),
        observed_registration_attempts=attempts['observed_attempt_count'],
        observed_failed_attempts=None if attempts['observed_attempt_count'] is None else sum(a['status']=='FAILED' for a in attempts['attempts']),
        observed_structureless_attempts=None if attempts['observed_attempt_count'] is None else sum(a['structureless_fallback'] for a in attempts['attempts']),
        final_snapshot_models=len(result['final_snapshot_correspondences']))
    result.update(schema='saved-registration-audit-v2',source_bindings=local,backend_scope='pycolmap4.2.1/sfm-native-object-v1',
        reconstruction_executions=0,parameters_modified=False,identity_verified=False,geometry_acceptance=False,
        limitations=['Stored pair geometric inliers and logged visible points are not absolute-pose RANSAC inliers.',
            'Final camera/point snapshots cannot prove a per-attempt registration rejection branch.',
            'Missing logs or fields remain UNKNOWN; no motion, occlusion or correspondence truth is inferred.'])
    output.mkdir(parents=True,exist_ok=False);bridge.write(output/'REGISTRATION.json',result)
    (output/'index.md').write_text('# Read-only registration evidence\n\nStatus: '+result['status']+'\n\nFinal snapshot correspondence counts are separate from actual attempt logs. Failure cause remains UNKNOWN unless explicitly retained; suspicious final calibration is not causal proof.\n\nSee REGISTRATION.json for bindings and complete evidence.\n',encoding='utf-8')
    for receipt in local:
        if Path(receipt['path']) in stream_paths:
            if stream_binding(receipt['path'])!=receipt:raise ValueError('database source binding drift')
        else:bridge.verify_binding(receipt)
    refs.extend(local);return result
