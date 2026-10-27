"""One-query-one-point retriangulation diagnostic, with fixed original cameras.

Image-conditioned query IDs are hypotheses, never physical identity truth.
Run only under an external native-process supervisor. No mapping or BA occurs.
"""
import sqlite3
import json
from pathlib import Path
import numpy as np
from .connectivity_native import _hash, _save, _stats
from .learned_identity_geometry import fit_query


def read_keypoints_readonly(connection, image_id):
    row=connection.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(int(image_id),)).fetchone()
    if row is None: raise ValueError('missing keypoints')
    rows,cols,data=row
    if type(rows) is not int or type(cols) is not int or rows<0 or cols not in (2,4,6):
        raise ValueError('invalid keypoint shape')
    if data is None: data=b''
    if len(data)!=rows*cols*4: raise ValueError('invalid keypoint blob size')
    points=np.frombuffer(data,dtype='<f4').reshape(rows,cols)
    if not np.isfinite(points).all(): raise ValueError('nonfinite keypoints')
    return points.copy()

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
    refs += [_hash(Path(__file__)),_hash(Path(__file__).with_name('connectivity_native.py')),_hash(Path(__file__).with_name('learned_identity_geometry.py'))]
    rec=pc.Reconstruction(str(binary)); db=sqlite3.connect((source/'database.db').as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA query_only=ON')
    frames={f['database_image_id']:f for f in plan['frames']}
    registered=sorted(int(i) for i in rec.reg_image_ids())
    heldout=set(registered[4::5]); tracks={t:[] for f in plan['frames'] for t in f['track_ids']}
    for iid in registered:
        im=rec.images[iid]; f=frames[iid]; xy=read_keypoints_readonly(db,iid)
        if im.name!=f['image_name'] or len(xy)!=len(f['track_ids']): raise ValueError('image/row binding mismatch')
        native_xy=np.asarray([p.xy for p in im.points2D]).reshape(-1,2)
        if not np.array_equal(xy[:,:2],native_xy): raise ValueError('native keypoints differ from database')
        for row,t in enumerate(f['track_ids']):
            tracks[t].append(dict(image_id=iid,feature_row=row,xy=xy[row,:2].astype(float).tolist(),
                                  camera=rec.cameras[im.camera_id],pose=im.cam_from_world()))
    db.close()
    records=[dict(query_id=t,**fit_query(obs,pc,heldout)) for t,obs in sorted(tracks.items())]
    accepted=[r for r in records if r['xyz'] is not None]
    if any(_hash(r['path'])!=r for r in refs): raise ValueError('source changed during diagnostic')
    out.mkdir(parents=True,exist_ok=False)
    ply=out/'points.ply'
    with ply.open('w',encoding='ascii',newline='\n') as stream:
        stream.write('ply\nformat ascii 1.0\nelement vertex '+str(len(accepted))+'\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
        for r in accepted: stream.write(' '.join(format(v,'.17g') for v in r['xyz'])+'\n')
    result=dict(schema='learned-identity-geometry-v2',status='RETRIANGULATED_NOT_QUALITY_ACCEPTED',
        model_id=model_id,registered_image_ids=registered,heldout_image_ids=sorted(heldout),
        split_policy='Every fifth sorted registered image, positions 5,10,...; excluded from fit',
        conditional_holdout='Heldout observations are excluded only from retriangulation. Original camera intrinsics and poses were estimated using all views, including heldout views and candidate observations; cameras and data selection were not independently trained. Heldout errors are conditional diagnostics, not end-to-end generalization or ground-truth validation.',
        estimator='pycolmap 4.2.1 estimate_triangulation LO-RANSAC; seed 0; reprojection 4px; angle 1.5deg',
        camera_policy='Original native intrinsics and poses preserved; no bundle adjustment',
        query_count=len(records),fitted_query_count=len(accepted),rejected_query_count=len(records)-len(accepted),
        records=records,ply_path=str(ply),ply_vertex_query_ids=[r['query_id'] for r in accepted],
        database_read_policy='sqlite3 URI mode=ro plus PRAGMA query_only; keypoint rows/cols/data float32 read directly; no native Database.open',
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
