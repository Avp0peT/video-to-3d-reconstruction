"""One-query-one-point retriangulation diagnostic, with fixed original cameras.

Image-conditioned query IDs are hypotheses, never physical identity truth.
Run only under an external native-process supervisor. No mapping or BA occurs.
"""
import itertools
import json
from pathlib import Path
import numpy as np
from .connectivity_native import _hash, _save, _stats


def effective_ray_angle_deg(a, b):
    theta=float(np.rad2deg(np.arccos(np.clip(a@b,-1.,1.))))
    return min(theta,180.-theta)


def fit_query(observations, pc, heldout_ids, threshold=4., min_angle_deg=1.5):
    """Fit training observations only; report every observation, including failures."""
    train=[o for o in observations if o['image_id'] not in heldout_ids]
    result=dict(status='REJECTED', xyz=None, observations=[], training_count=len(train),
                heldout_count=len(observations)-len(train), rejection_reasons=[])
    xyz=None
    if len(train)<3:
        result['rejection_reasons'].append('fewer_than_three_training_views')
    else:
        options=pc.EstimateTriangulationOptions()
        options.residual_type=pc.TriangulationResidualType.REPROJECTION_ERROR
        options.min_tri_angle=float(np.deg2rad(min_angle_deg))
        options.ransac.max_error=threshold
        options.ransac.random_seed=0
        options.ransac.num_threads=1
        estimate=pc.estimate_triangulation(np.array([o['xy'] for o in train]),
            [o['pose'] for o in train], [o['camera'] for o in train], options)
        if estimate is None:
            result['rejection_reasons'].append('native_robust_triangulation_failed')
        else:
            candidate=np.asarray(estimate['xyz'],dtype=float)
            if candidate.shape==(3,) and np.isfinite(candidate).all(): xyz=candidate
            else: result['rejection_reasons'].append('nonfinite_native_hypothesis')
    inliers=[]
    for o in observations:
        held=o['image_id'] in heldout_ids
        row=dict(image_id=o['image_id'], feature_row=o['feature_row'], xy=list(o['xy']),
                 split='heldout' if held else 'train', error_px=None, positive_depth=None,
                 within_threshold=False, rejection_reasons=[])
        if xyz is None:
            row['rejection_reasons'].append('no_fitted_hypothesis')
        else:
            local=o['pose'].rotation.matrix()@xyz+o['pose'].translation
            row['positive_depth']=bool(local[2]>0)
            projection=o['camera'].img_from_cam(local) if row['positive_depth'] else None
            if not row['positive_depth']: row['rejection_reasons'].append('nonpositive_depth')
            if projection is None or not np.isfinite(projection).all():
                row['rejection_reasons'].append('projection_invalid')
            else:
                row['error_px']=float(np.linalg.norm(np.asarray(projection)-o['xy']))
                row['within_threshold']=row['error_px']<=threshold
                if not row['within_threshold']: row['rejection_reasons'].append('reprojection_above_threshold')
            if not held and not row['rejection_reasons']: inliers.append(o)
        result['observations'].append(row)
    angle=0.
    if xyz is not None:
        rays=[]
        for o in inliers:
            r=o['pose'].rotation.matrix(); center=-r.T@o['pose'].translation
            v=xyz-center; rays.append(v/np.linalg.norm(v))
        for a,b in itertools.combinations(rays,2):
            angle=max(angle,effective_ray_angle_deg(a,b))
        if len(inliers)<3: result['rejection_reasons'].append('fewer_than_three_training_inliers')
        if angle<min_angle_deg: result['rejection_reasons'].append('insufficient_training_triangulation_angle')
    result.update(training_inliers=len(inliers), max_training_angle_deg=angle,
        training_error_px=_stats([r['error_px'] for r in result['observations'] if r['split']=='train' and r['error_px'] is not None]),
        training_inlier_error_px=_stats([r['error_px'] for r in result['observations'] if r['split']=='train' and not r['rejection_reasons']]),
        heldout_error_px=_stats([r['error_px'] for r in result['observations'] if r['split']=='heldout' and r['error_px'] is not None]))
    if xyz is not None and not result['rejection_reasons']:
        result.update(status='FITTED_NOT_IDENTITY_VERIFIED',xyz=xyz.tolist())
    return result


def worker(native_run, output, model_id=3):
    import pycolmap as pc
    if pc.__version__!='4.2.1': raise ValueError('pinned pycolmap 4.2.1 required')
    source=Path(native_run).resolve(strict=True); out=Path(output).resolve()
    if out.exists() or out==source or source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError('fresh output outside native source required')
    refs=[_hash(source/n) for n in ('PLAN.json','RESULT.json','database.db')]
    plan=json.loads((source/'PLAN.json').read_text(encoding='utf-8'))
    upstream=json.loads((source/'RESULT.json').read_text(encoding='utf-8'))
    if plan.get('schema')!='learned-candidate-native-plan-v1': raise ValueError('native candidate plan required')
    model=next(m for m in upstream['models'] if m['model_id']==model_id)
    binary=Path(model['binary_path']).resolve(strict=True)
    refs += [_hash(p) for p in sorted(binary.iterdir()) if p.is_file()]
    refs += [_hash(Path(__file__)),_hash(Path(__file__).with_name('connectivity_native.py'))]
    rec=pc.Reconstruction(str(binary)); db=pc.Database.open(source/'database.db')
    frames={f['database_image_id']:f for f in plan['frames']}
    registered=sorted(int(i) for i in rec.reg_image_ids())
    heldout=set(registered[4::5]); tracks={t:[] for f in plan['frames'] for t in f['track_ids']}
    for iid in registered:
        im=rec.images[iid]; f=frames[iid]; xy=db.read_keypoints(iid)
        if im.name!=f['image_name'] or len(xy)!=len(f['track_ids']): raise ValueError('image/row binding mismatch')
        native_xy=np.asarray([p.xy for p in im.points2D]).reshape(-1,2)
        if not np.array_equal(xy[:,:2],native_xy): raise ValueError('native keypoints differ from database')
        for row,t in enumerate(f['track_ids']):
            tracks[t].append(dict(image_id=iid,feature_row=row,xy=xy[row,:2].astype(float).tolist(),
                                  camera=rec.cameras[im.camera_id],pose=im.cam_from_world()))
    del db
    records=[dict(query_id=t,**fit_query(obs,pc,heldout)) for t,obs in sorted(tracks.items())]
    accepted=[r for r in records if r['xyz'] is not None]
    if any(_hash(r['path'])!=r for r in refs): raise ValueError('source changed during diagnostic')
    out.mkdir(parents=True,exist_ok=False)
    ply=out/'points.ply'
    with ply.open('w',encoding='ascii',newline='\n') as stream:
        stream.write('ply\nformat ascii 1.0\nelement vertex '+str(len(accepted))+'\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
        for r in accepted: stream.write(' '.join(format(v,'.17g') for v in r['xyz'])+'\n')
    result=dict(schema='learned-identity-geometry-v1',status='RETRIANGULATED_NOT_QUALITY_ACCEPTED',
        model_id=model_id,registered_image_ids=registered,heldout_image_ids=sorted(heldout),
        split_policy='Every fifth sorted registered image, positions 5,10,...; excluded from fit',
        conditional_holdout='Heldout observations are excluded only from retriangulation. Original camera intrinsics and poses were estimated using all views, including heldout views and candidate observations; cameras and data selection were not independently trained. Heldout errors are conditional diagnostics, not end-to-end generalization or ground-truth validation.',
        estimator='pycolmap 4.2.1 estimate_triangulation LO-RANSAC; seed 0; reprojection 4px; angle 1.5deg',
        camera_policy='Original native intrinsics and poses preserved; no bundle adjustment',
        query_count=len(records),fitted_query_count=len(accepted),rejected_query_count=len(records)-len(accepted),
        records=records,ply_path=str(ply),ply_vertex_query_ids=[r['query_id'] for r in accepted],
        source_bindings=refs,physical_identity_verified=False,geometry_accepted=False,
        upstream_lineage='Inherited from native PLAN and RESULT; upstream source_bindings are not live revalidated by this diagnostic.',
        quality_status='NOT_EVALUATED',scale_status='unknown',cad_status='NOT_EVALUATED',
        old_points_used=False,components_merged=False)
    _save(out/'RESULT.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--native-run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--model-id',type=int,default=3)
    args=parser.parse_args()
    worker(args.native_run,args.output,args.model_id)
