"""Export one point per fitted query into a copied native reconstruction.

Experimental identity constraints are not physical identity verification.
No solving, BA, camera modification, hole filling, or new geometry occurs.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .connectivity_native import _hash, _save


def require(condition, message):
    if not condition: raise ValueError(message)


def unchanged_views(original, child):
    require(set(original.reg_image_ids())==set(child.reg_image_ids()),'registered views changed')
    require(set(original.images)==set(child.images),'image set changed')
    require(set(original.cameras)==set(child.cameras),'camera set changed')
    require(set(original.frames)==set(child.frames) and set(original.rigs)==set(child.rigs),'frame/rig set changed')
    for cid,a in original.cameras.items():
        b=child.cameras[cid]
        require((a.model_name,a.width,a.height)==(b.model_name,b.width,b.height) and
                np.array_equal(a.params,b.params),'camera changed')
    for iid,a in original.images.items():
        b=child.images[iid]
        require((a.name,a.camera_id,a.frame_id)==(b.name,b.camera_id,b.frame_id),'image identity changed')
        require(np.array_equal(np.asarray([p.xy for p in a.points2D]).reshape(-1,2),
                               np.asarray([p.xy for p in b.points2D]).reshape(-1,2)),'keypoint rows changed')
        if iid in original.reg_image_ids():
            require(np.array_equal(a.cam_from_world().rotation.matrix(),b.cam_from_world().rotation.matrix()) and
                    np.array_equal(a.cam_from_world().translation,b.cam_from_world().translation),'pose changed')


def verify_export(original, child, ledger):
    unchanged_views(original,child)
    require(len(child.points3D)==len(ledger),'point count mismatch')
    require(len({r['query_id'] for r in ledger})==len(ledger),'duplicate exported query')
    require({r['point3D_id'] for r in ledger}==set(child.points3D),'native point IDs mismatch')
    expected={}
    for record in ledger:
        pid=record['point3D_id']; p=child.points3D[pid]
        require(np.array_equal(p.xyz,np.asarray(record['xyz'])),'XYZ mismatch')
        require(np.array_equal(p.color,np.asarray(record['color_rgb'])),'RGB mismatch')
        pairs=[(int(e.image_id),int(e.point2D_idx)) for e in p.track.elements]
        wanted=[(r['image_id'],r['feature_row']) for r in record['observations']]
        require(len(pairs)==len(set(pairs)) and sorted(pairs)==sorted(wanted),'track mismatch')
        require(len({iid for iid,row in pairs})==len(pairs),'duplicate image within track')
        for iid,row in pairs:
            require((iid,row) not in expected,'feature assigned twice')
            expected[iid,row]=pid
            require(child.images[iid].points2D[row].point3D_id==pid,'broken track reciprocity')
    for iid,im in child.images.items():
        for row,p in enumerate(im.points2D):
            require(p.has_point3D()==((iid,row) in expected),'unexpected image observation')
            if p.has_point3D(): require(p.point3D_id==expected[iid,row],'broken reverse reciprocity')
    return dict(registered_views=len(child.reg_image_ids()),points=len(child.points3D),
                observations=len(expected),cameras_poses_keypoint_rows_unchanged=True,
                exact_xyz_rgb_tracks_and_reciprocity=True)


def derive(original, identity, plan, image_root, pc):
    require(identity.get('physical_identity_verified') is False and identity.get('geometry_accepted') is False,
            'unverified identity diagnostic required')
    registered=set(int(i) for i in original.reg_image_ids())
    require(sorted(registered)==identity['registered_image_ids'],'diagnostic registered view mismatch')
    frames={f['database_image_id']:f for f in plan['frames']}
    records=identity['records']
    require(len({r['query_id'] for r in records})==len(records),'duplicate query ID')
    require({r['query_id'] for r in records}=={t for f in plan['frames'] for t in f['track_ids']},
            'query coverage mismatch')
    child=pc.Reconstruction(original)
    for pid in list(child.points3D): child.delete_point3D(pid)
    require(not child.points3D,'old points not removed')
    colors={}; used=set(); ledger=[]; rejected=[]
    image_root=Path(image_root).resolve(strict=True)
    for record in sorted(records,key=lambda r:r['query_id']):
        qid=record['query_id']
        if record['xyz'] is None:
            rejected.append(dict(query_id=qid,reason='no_fitted_xyz')); continue
        xyz=np.asarray(record['xyz'],dtype=float)
        require(xyz.shape==(3,) and np.isfinite(xyz).all(),'invalid fitted XYZ')
        observations=[];seen=set();samples=[]
        for obs in record['observations']:
            iid=obs['image_id']; row=obs['feature_row']
            require(iid in registered and iid not in seen,'unknown/duplicate image observation')
            seen.add(iid)
            im=original.images[iid]; f=frames[iid]
            require(im.name==f['image_name'] and 0<=row<len(im.points2D) and
                    len(im.points2D)==len(f['track_ids']) and f['track_ids'][row]==qid,'query/row identity mismatch')
            xy=np.asarray(im.points2D[row].xy)
            require(np.array_equal(xy,np.asarray(obs['xy'])),'diagnostic observation XY mismatch')
            expected_split='heldout' if iid in identity['heldout_image_ids'] else 'train'
            require(obs['split']==expected_split,'observation split mismatch')
            local=im.cam_from_world()*xyz
            projection=original.cameras[im.camera_id].img_from_cam(local) if local[2]>0 else None
            error=float(np.linalg.norm(projection-xy)) if projection is not None else None
            good=error is not None and np.isfinite(error) and error<=4.
            require(obs['positive_depth']==bool(local[2]>0) and obs['within_threshold']==bool(good),
                    'diagnostic eligibility mismatch')
            if not good: continue
            require((iid,row) not in used,'feature assigned to multiple queries')
            used.add((iid,row))
            path=(image_root/im.name).resolve(strict=True)
            require(path.is_relative_to(image_root),'unsafe source image path')
            if iid not in colors:
                with Image.open(path) as image:
                    camera=original.cameras[im.camera_id]
                    require(image.mode=='RGB' and image.size==(camera.width,camera.height),'source image RGB/domain mismatch')
                    colors[iid]=np.asarray(image).copy()
            x,y=np.floor(xy).astype(int)
            require(0<=x<colors[iid].shape[1] and 0<=y<colors[iid].shape[0],'color sample outside source image')
            rgb=colors[iid][y,x].tolist(); samples.append(rgb)
            observations.append(dict(image_id=iid,feature_row=row,split=expected_split,
                error_px=error,color_source_path=str(path),color_pixel_xy=[int(x),int(y)],color_rgb=rgb))
        require(sum(o['split']=='train' for o in observations)>=3,'fewer than three valid training observations')
        color=np.rint(np.median(np.asarray(samples),axis=0)).astype(np.uint8)
        track=pc.Track()
        for o in observations: track.add_element(o['image_id'],o['feature_row'])
        pid=int(child.add_point3D(xyz,track,color))
        ledger.append(dict(query_id=qid,point3D_id=pid,xyz=xyz.tolist(),color_rgb=color.tolist(),observations=observations))
    audit=verify_export(original,child,ledger)
    return child,ledger,rejected,audit


def worker(identity_result, native_run, output):
    import pycolmap as pc
    require(pc.__version__=='4.2.1','pinned pycolmap 4.2.1 required')
    identity_path=Path(identity_result).resolve(strict=True)
    native=Path(native_run).resolve(strict=True); out=Path(output).resolve()
    require(not out.exists() and not native.is_relative_to(out) and not out.is_relative_to(native)
            and not identity_path.is_relative_to(out),'fresh output must not overlap sources')
    refs=[_hash(identity_path),_hash(native/'PLAN.json'),_hash(native/'RESULT.json'),
          _hash(Path(__file__)),_hash(Path(__file__).with_name('connectivity_native.py'))]
    identity=json.loads(identity_path.read_text(encoding='utf-8-sig'))
    require(identity.get('schema')=='learned-identity-geometry-v2','v2 diagnostic required')
    require(identity.get('model_id')==3,'native model 3 required')
    for binding in identity['source_bindings']:
        require(_hash(binding['path'])==binding,'diagnostic source binding changed: '+binding['path'])
        refs.append(binding)
    plan=json.loads((native/'PLAN.json').read_text(encoding='utf-8-sig'))
    upstream=json.loads((native/'RESULT.json').read_text(encoding='utf-8-sig'))
    mask_manifests=[]
    for binding in plan.get('source_bindings',[]):
        if Path(binding['path']).name=='MASKS.json':
            require(_hash(binding['path'])==binding,'native MASKS binding changed')
            refs.append(binding);mask_manifests.append(binding['path'])
    for name in ('PLAN.json','RESULT.json'):
        require(any(Path(b['path']).resolve()==native/name for b in identity['source_bindings']),
                'diagnostic does not bind selected native '+name)
    model=next(m for m in upstream['models'] if m['model_id']==3)
    binary=Path(model['binary_path']).resolve(strict=True)
    model_refs=[_hash(p) for p in sorted(binary.iterdir()) if p.is_file()]
    require(all(b in identity['source_bindings'] for b in model_refs),'selected binary not fully bound by diagnostic')
    refs.extend(model_refs)
    original=pc.Reconstruction(str(binary))
    require(len(original.reg_image_ids())==model['registered_views'],'source registered view count mismatch')
    for iid in sorted(original.reg_image_ids()):
        image=(native/'images'/original.images[iid].name).resolve(strict=True)
        require(image.is_relative_to(native/'images'),'unsafe image name')
        refs.append(_hash(image))
    # Native module itself is an input to the native roundtrip checks.
    import pycolmap._core as core
    refs.append(_hash(core.__file__))
    refs=list({r['path']:r for r in refs}.values())
    child,ledger,rejected,in_memory=derive(original,identity,plan,native/'images',pc)
    require(all(_hash(r['path'])==r for r in refs),'source changed during derivation')
    out.mkdir(parents=True,exist_ok=False)
    binary_out=out/'binary';text_out=out/'text';binary_out.mkdir();text_out.mkdir()
    child.write_binary(str(binary_out));child.write_text(str(text_out));child.export_PLY(str(out/'points.ply'))
    binary_back=pc.Reconstruction(str(binary_out));text_back=pc.Reconstruction(str(text_out))
    binary_audit=verify_export(original,binary_back,ledger)
    text_audit=verify_export(original,text_back,ledger)
    require(all(_hash(r['path'])==r for r in refs),'source changed during export/readback')
    images=[]
    frames={f['database_image_id']:f for f in plan['frames']}
    for iid in sorted(child.reg_image_ids()):
        im=child.images[iid];f=frames[iid];pose=im.cam_from_world()
        images.append(dict(image_id=int(iid),name=im.name,camera_id=int(im.camera_id),frame_id=int(im.frame_id),
            pose_rotation=pose.rotation.matrix().tolist(),pose_translation=pose.translation.tolist(),
            input_index=f['input_index'],decode_index=f['decode_index'],frame_id_source=f['frame_id']))
    final_bindings=[_hash(p) for folder in (binary_out,text_out) for p in sorted(folder.iterdir()) if p.is_file()]
    final_bindings.append(_hash(out/'points.ply'))
    result=dict(schema='learned-identity-native-export-v1',status='IDENTITY_CONSTRAINED_EXPERIMENTAL',
        binary_path=str(binary_out),text_path=str(text_out),ply_path=str(out/'points.ply'),
        native_root=str(native),source_model_id=3,source_identity_result=str(identity_path),
        mask_manifest_paths=mask_manifests,
        image_root=str(native/'images'),model_id=3,registered_views=len(child.reg_image_ids()),images=images,
        registered_image_ids=sorted(int(i) for i in child.reg_image_ids()),points=len(ledger),
        query_native_point_map=ledger,rejected_queries=rejected,source_bindings=refs,final_bindings=final_bindings,source_unchanged=True,
        in_memory_audit=in_memory,binary_readback=binary_audit,text_readback=text_audit,
        color_policy='Median per RGB channel of all retained source observations at floor(native XY); np.rint ties-to-even to uint8. Source RGB and pixel coordinates recorded per observation.',
        downstream_holdout_policy='Retained tracks include qualifying train and heldout observations for downstream modeling. Heldout observations are now consumed by the model; this export is not an independent heldout test.',
        conditional_holdout=identity['conditional_holdout'],
        physical_identity_verified=False,geometry_accepted=False,quality_status='NOT_EVALUATED',
        scale_status='unknown',cad_status='NOT_EVALUATED',bundle_adjustment=False,hole_filling=False,
        original_points_removed=True,old_xyz_used=False,
        upstream_lineage='Diagnostic source bindings live revalidated; earlier upstream lineage remains inherited and not live revalidated.')
    _save(out/'RESULT.json',result)
    return result


def infer_native_root(identity_result):
    path=Path(identity_result).resolve(strict=True); before=_hash(path)
    identity=json.loads(path.read_text(encoding='utf-8-sig'))
    bindings=identity['source_bindings']
    for binding in bindings:
        require(_hash(binding['path'])==binding,'diagnostic source binding changed: '+binding['path'])
    bound={Path(b['path']).resolve() for b in bindings}
    roots=[p.parent for p in bound if p.name=='PLAN.json' and
           p.parent/'RESULT.json' in bound and p.parent/'database.db' in bound]
    require(len(roots)==1,'cannot uniquely locate native root from diagnostic bindings')
    require(_hash(path)==before,'diagnostic changed during native root lookup')
    return roots[0]


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--diagnostic',required=True,type=Path)
    parser.add_argument('--native-run',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    worker(args.diagnostic,args.native_run or infer_native_root(args.diagnostic),args.output)
