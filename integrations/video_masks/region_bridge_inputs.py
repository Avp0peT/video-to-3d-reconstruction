"""Prepare measured intermediate keyframes only; no segmentation or SfM run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
import numpy as np
from .region_tracks import PROFILE

POLICY=dict(id='glove-region-bridge-keyframes-v1',native_max_frames=96,
            max_added=24,max_original_endpoint_span=6,
            fractions=[[1,3],[2,3]],ordering='endpoint-span ascending, left endpoint position ascending, interval ascending, fraction ascending')


def select_intermediates(timeline,models):
    endpoints=[r for r in timeline if r.get('input_index') is not None]
    valid=sorted((m for m in models if m.get('label')=='new' and m['camera_domain_status']=='CAMERA_DOMAIN_VALID'),
                 key=lambda m:(-len(m['registered_indices']),m['model_id']))
    groups={}
    for pos,row in enumerate(endpoints):
        groups[pos]=next((m['model_id'] for m in valid if row['input_index'] in m['registered_indices']),None)
    grouped=[p for p in range(len(endpoints)) if groups[p] is not None]
    gaps=[]
    for left,right in zip(grouped,grouped[1:]):
        if groups[left]!=groups[right] and right-left<=6:
            gaps.append((right-left,left,right))
    existing={r['decode_index'] for r in endpoints};available={r['decode_index'] for r in timeline}
    chosen={};ledger=[]
    for span,left,right in sorted(gaps):
        for p in range(left,right):
            a,b=endpoints[p]['decode_index'],endpoints[p+1]['decode_index']
            for numerator in (1,2):
                n=a+(b-a)*numerator//3
                record=dict(left_original_position=left,right_original_position=right,span=span,
                            interval_positions=[p,p+1],decode_index=n,fraction=[numerator,3])
                if n in existing or n in chosen or n not in available:
                    record['decision']='EXISTING_OR_DUPLICATE_OR_UNAVAILABLE'
                elif len(chosen)>=24:
                    record['decision']='CAP_REACHED'
                else:
                    record['decision']='SELECTED';chosen[n]=record
                ledger.append(record)
    return sorted(chosen),dict(policy=POLICY,group_by_original_position=groups,gaps=[list(g) for g in sorted(gaps)],decisions=ledger)


def choose_intermediates(timeline, models, selection_policy=None):
    """Keep the historical policy reproducible; new callers opt into v1."""
    if selection_policy is None:
        return select_intermediates(timeline, models)
    from .object_region_policy import select_intermediates as select_object
    # The assessment deliberately includes both old and new reconstructions.
    current = [m for m in models if m.get('label') == 'new']
    return select_object(timeline, current, policy=selection_policy)


def prepare(source_run,output,selection_policy=None):
    import pycolmap as pc
    source=Path(source_run).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or out==source or source.is_relative_to(out):
        raise FileExistsError('fresh disjoint output required')
    refs={}
    def bind(path,expected=None):
        path=Path(path).resolve(strict=True);data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        if (expected is not None and digest!=expected) or (str(path) in refs and refs[str(path)]!=digest):
            raise ValueError('source binding drift: '+str(path))
        refs[str(path)]=digest
        return data
    def read(path):
        return json.loads(bind(path).decode('utf-8-sig'))
    parent=read(source/'PLAN.json');original_source=Path(parent['source_run']).resolve(strict=True)
    original=read(original_source/'PLAN.json')
    if original.get('schema')!='glove-region-plan-v1' or original.get('profile')!=PROFILE:
        raise ValueError('frozen original region plan required')
    for path,digest in original.get('source_bindings',{}).items():
        bind(path,digest)
    assessment=read(source/'assessment'/'ASSESSMENT.json')
    read(source/'geometry'/'RESULT.json');read(source/'geometry'/'PLAN.json')
    read(source/'tracking'/'REGION_TRACKS.json')
    timeline=deepcopy(original['timeline'])
    if [r['decode_index'] for r in timeline]!=list(range(len(timeline))):
        raise ValueError('complete consecutive original timeline required')
    endpoints=deepcopy([r for r in timeline if r.get('input_index') is not None])
    if len({r['input_index'] for r in endpoints})!=len(endpoints):
        raise ValueError('unique original endpoint identity required')
    added,selection=choose_intermediates(timeline,assessment['models'],selection_policy)
    effective_policy=selection['policy']
    if selection.get('resource_limit_satisfied') is False:
        raise ValueError('existing endpoints exceed selection policy resource limit')
    if len(endpoints)+len(added)>96:
        raise ValueError('native frame cap exceeded')
    baseline=Path(original['baseline_sfm']).resolve(strict=True)
    native_run=read(baseline/'run.json')
    maps=[json.loads(line) for line in bind(baseline/'frames_map.jsonl').decode('utf-8-sig').splitlines() if line.strip()]
    oldmap={r['input_index']:r for r in maps}
    if set(oldmap)!={r['input_index'] for r in endpoints}:
        raise ValueError('original frame map coverage differs')
    original_db=bind(baseline/'database.db')
    with sqlite3.connect('file:'+str(baseline/'database.db')+'?mode=ro',uri=True) as db:
        oldimages={iid:(name,cid) for iid,name,cid in db.execute('SELECT image_id,name,camera_id FROM images')}
        cols={c for c, in db.execute('SELECT DISTINCT cols FROM keypoints')}
        snapshot={t:list(db.execute('SELECT * FROM '+t)) for t in ('images','cameras','keypoints','descriptors','matches','two_view_geometries','rigs','rig_sensors','frames','frame_data')}
    if len(cols)!=1 or next(iter(cols)) not in (4,6):
        raise ValueError('uniform original keypoint columns 4 or 6 required')
    keypoint_cols=next(iter(cols));old_ids=set(oldimages)
    if {r['database_image_id'] for r in maps}!=old_ids:
        raise ValueError('all original database images must be bound')
    out.mkdir(parents=True,exist_ok=False);sfm=out/'source_sfm';sfm.mkdir();(sfm/'images').mkdir()
    (sfm/'database.db').write_bytes(original_db)
    for iid,(name,cid) in oldimages.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('relative staged image name required')
        source_image=baseline/relative
        if not source_image.is_file():source_image=baseline/'images'/relative.name
        data=bind(source_image)
        destination=sfm/relative
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
    newmap=[];added_set=set(added);endpoint_by_decode={r['decode_index']:r for r in endpoints}
    db=pc.Database.open(sfm/'database.db');next_id=max(old_ids,default=0)+1
    try:
        original_frames={f.frame_id:f for f in db.read_all_frames()}
        rigs={r.rig_id:r for r in db.read_all_rigs()}
        image_rig={}
        for iid in old_ids:
            image_record=db.read_image(iid)
            if not image_record.has_frame_id() or image_record.frame_id not in original_frames:
                raise ValueError('original image lacks explicit frame association')
            image_rig[iid]=original_frames[image_record.frame_id].rig_id
        for row in timeline:
            original_index=row.get('input_index')
            if original_index is None and row['decode_index'] not in added_set:continue
            n=row['decode_index']; row['input_index']=n
            image=Path(row['image_path']);mask=Path(row['mask_path'])
            image_bytes=bind(image,row['image_sha256']);bind(mask,row['mask_sha256'])
            if original_index is not None:
                entry=deepcopy(oldmap[original_index]);iid=entry['database_image_id'];name=entry['staged_image_name']
                if entry['canonical_image_sha256']!=row['image_sha256'] or hashlib.sha256((sfm/name).read_bytes()).hexdigest()!=row['image_sha256']:
                    raise ValueError('original canonical bytes differ')
            else:
                if image.suffix.lower() not in ('.jpg','.jpeg'):
                    raise ValueError('new endpoint must use existing decoded JPEG')
                left=max((r for r in endpoints if r['decode_index']<n),key=lambda r:r['decode_index'])
                left_image_id=oldmap[left['input_index']]['database_image_id']
                camera_id=oldimages[left_image_id][1]
                iid=next_id;next_id+=1;name=f'images/bridge_decode_{n:06d}.jpg'
                if name in {v[0] for v in oldimages.values()}:raise ValueError('new staged name collision')
                (sfm/name).write_bytes(image_bytes)
                db.write_image(pc.Image(name=name,camera_id=camera_id,image_id=iid),use_image_id=True)
                db.write_keypoints(iid,np.empty((0,keypoint_cols),np.float32))
                # COLMAP 4.2 does not auto-create missing frame associations when
                # a copied database already contains rigs and frames.
                sensor=pc.sensor_t(pc.SensorType.CAMERA,camera_id)
                rig_id=image_rig[left_image_id]
                if rig_id not in rigs or sensor not in rigs[rig_id].sensor_ids():
                    raise ValueError('source frame rig does not contain camera sensor')
                frame=pc.Frame(rig_id=rig_id)
                data_id=pc.data_t(sensor,iid)
                frame.add_data_id(data_id)
                frame_id=db.write_frame(frame)
                measured=db.read_image(iid)
                frames_now={f.frame_id:f for f in db.read_all_frames()}
                if not measured.has_frame_id() or measured.frame_id!=frame_id or measured.data_id!=data_id or data_id not in frames_now[frame_id].data_ids:
                    raise ValueError('new image frame/sensor association readback failed')
                entry=dict(database_image_id=iid,staged_image_name=name,status='PREPARED_ONLY',registrations=[])
            entry.update(input_index=n,original_input_index=original_index,representative_input_index=n,decode_index=n,
                         canonical_image_sha256=row['image_sha256'],canonical_mask_sha256=row['mask_sha256'],
                         actual_source_image=str(image.resolve()),actual_source_mask=str(mask.resolve()),
                         source_manifest=str(original_source/'PLAN.json'),source_manifest_sha256=refs[str((original_source/'PLAN.json').resolve())])
            newmap.append(entry)
    finally:db.close()
    # Compare all old rows by their primary IDs; additions are confined to images/keypoints.
    with sqlite3.connect(sfm/'database.db') as db:
        for table,rows in snapshot.items():
            current=list(db.execute('SELECT * FROM '+table))
            if not set(rows).issubset(set(current)):raise ValueError('original database rows changed: '+table)
            if table not in ('images','keypoints','frames','frame_data') and len(rows)!=len(current):raise ValueError('unexpected database additions: '+table)
    native_run=dict(schema='glove-region-prepared-native-input-v1',
                    run_id=str(uuid.uuid4()),status='PREPARED_ONLY',
                    source_run_id=native_run.get('run_id'),
                    source_run_sha256=refs[str((baseline/'run.json').resolve())],
                    effective_options=deepcopy(native_run['effective_options']),
                    prepared_only=True,additional_regions_only=True,models=[],geometry_executed=False,
                    preparation=dict(policy=effective_policy,original_sfm=str(baseline),new_images=len(added),sift_extraction_run=False,sfm_executed=False))
    (sfm/'run.json').write_text(json.dumps(native_run,indent=2,allow_nan=False),encoding='utf8')
    (sfm/'frames_map.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in newmap),encoding='utf8')
    for row in timeline:
        for kind in ('image','mask'):bind(row[kind+'_path'],row[kind+'_sha256'])
    for path,digest in list(refs.items()):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('source changed during preparation')
    for path in sorted(sfm.rglob('*')):
        if path.is_file():refs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    refs[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan=deepcopy(original)
    plan.update(baseline_sfm=str(sfm),timeline=timeline,endpoint_count=len(newmap),source_run=str(source),
                endpoint_expansion_policy=effective_policy,source_bindings=refs,prepared_only=True,geometry_accepted=False,
                endpoint_selection=selection,added_decode_indices=added,
                original_endpoint_decode_indices=sorted(endpoint_by_decode))
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2,allow_nan=False),encoding='utf8')
    return plan
