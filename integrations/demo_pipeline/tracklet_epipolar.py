"""Uncalibrated held-out F/H diagnostics from preserved measured prefixes only."""
import math
from collections import defaultdict

RIGHTS=dict(geometry_accepted=False,identity_verified=False,intrinsics_verified=False,
            poses_verified=False,new_cameras=0,new_model_points=0)


def assess_epipolar(ledger, *, budget=lambda:None):
    """Fit on frozen training IDs only; square-root Sampson and forward H pixels.

    NumPy and numerical helpers load only on invocation. Malformed identities
    raise ValueError; budget exceptions propagate. No robust fitting or refill.
    """
    import numpy as np
    from kozlov_lab.stereo.viewpoint_consistency import fit_fundamental, _normalize, _residual
    def need(ok,reason):
        if not ok:raise ValueError(reason)
    def xy(value):
        need(isinstance(value,(list,tuple)) and len(value)==2 and all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=1e9 for v in value),'INVALID_RAW_XY')
        return list(value)
    bins=defaultdict(dict);seen=set()
    for record in ledger['records']:
        budget();tid=record.get('tracklet_id');pid=record.get('point_id');edge=record.get('edge_ordinal')
        need(isinstance(tid,str) and tid and tid not in seen,'DUPLICATE_OR_INVALID_TRACKLET_ID');seen.add(tid)
        need(type(edge) is int and edge>=0,'INVALID_EDGE_ORDINAL')
        need(all(type(record.get(k)) is int and record[k]>=0 for k in ('track_id','identity_segment','source_input_index')),'INVALID_POINT_IDENTITY')
        need(pid==f"track:{record['track_id']}:segment:{record['identity_segment']}:source:{record['source_input_index']}" and tid==f'{pid}:edge:{edge}','POINT_IDENTITY_MISMATCH')
        if record.get('status')!='CANDIDATE_2D_TRACKLET':continue
        obs=record['observations'];legs=record['verified_legs']
        need(type(legs) is int and 2<=legs<=8 and isinstance(obs,list) and len(obs)==legs+1,'INVALID_VERIFIED_PREFIX')
        indices=[o.get('decode_index') for o in obs]
        need(all(type(i) is int and i>=0 for i in indices),'INVALID_DECODE_IDENTITY')
        direction=1 if indices[-1]>indices[0] else -1
        need(all(b-a==direction for a,b in zip(indices,indices[1:])),'NONCONSECUTIVE_PREFIX')
        coords=[xy(o['raw_xy']) for o in obs]
        point=record.get('source_point',{})
        need(point.get('point_id')==pid and point.get('source_decode_index')==indices[0] and point.get('source_input_index')==record['source_input_index'] and xy(point.get('source_raw_xy'))==coords[0],'SOURCE_BINDING_MISMATCH')
        for i,(a,b) in enumerate(zip(indices,indices[1:])):
            key=(edge,a,b);need(pid not in bins[key],'DUPLICATE_PAIR_POINT')
            bins[key][pid]=dict(point_id=pid,tracklet_id=tid,source_raw_xy=coords[i],target_raw_xy=coords[i+1])
    def metrics(values,ids):
        need(np.isfinite(values).all(),'NONFINITE_RESIDUAL')
        return dict(point_ids=ids,residual_pixels=values.tolist(),rms=float(np.sqrt(np.mean(values**2))),median=float(np.median(values)),max=float(values.max()))
    def homography(train):
        a,ta=_normalize(train[:,:2]);b,tb=_normalize(train[:,2:]);rows=[]
        for (x,y),(u,v) in zip(a,b):
            rows.extend([[-x,-y,-1,0,0,0,u*x,u*y,u],[0,0,0,-x,-y,-1,v*x,v*y,v]])
        _,s,vt=np.linalg.svd(np.asarray(rows),full_matrices=True)
        need(s[7]>s[0]*1e-6,'H_DESIGN_DEGENERATE')
        h=np.linalg.inv(tb)@vt[-1].reshape(3,3)@ta
        need(np.isfinite(h).all() and np.linalg.matrix_rank(h)==3,'H_SINGULAR')
        h/=np.linalg.norm(h)
        return dict(H=h.tolist(),design_singular_values=s.tolist())
    def herror(model,points):
        projected=np.column_stack((points[:,:2],np.ones(len(points))))@np.asarray(model['H']).T
        need(np.all(np.abs(projected[:,2])>1e-12),'H_PROJECTION_AT_INFINITY')
        return np.linalg.norm(projected[:,:2]/projected[:,2,None]-points[:,2:],axis=1)
    pairs=[];omitted=[]
    for ordinal,key in enumerate(sorted(bins)):
        budget();edge,a,b=key;items=[bins[key][pid] for pid in sorted(bins[key])]
        need(len(items)<=768,'PAIR_POINT_BOUND')
        if ordinal>=16:
            omitted.append(dict(edge_ordinal=edge,source_decode_index=a,target_decode_index=b,points=len(items),reason='PAIR_BUDGET'));continue
        ids=[r['point_id'] for r in items];pixels=np.asarray([r['source_raw_xy']+r['target_raw_xy'] for r in items],float)
        row=dict(edge_ordinal=edge,source_decode_index=a,target_decode_index=b,status='BLOCKED',reasons=[],points=items,
                 counts=dict(points=len(items),train=0,holdout=0),train_ids=[],holdout_ids=[],F=None,H=None,negative_control=None,**RIGHTS)
        pairs.append(row)
        row['bbox_2d']={name:dict(min_xy=pixels[:,sl].min(0).tolist(),max_xy=pixels[:,sl].max(0).tolist(),span_px=np.ptp(pixels[:,sl],axis=0).tolist(),meaning='2D observed bounding box; not mask or surface coverage') for name,sl in [('source',slice(0,2)),('target',slice(2,4))]}
        row['image_support']={name:dict(centered_rank=int(np.linalg.matrix_rank(pixels[:,sl]-pixels[:,sl].mean(0))),singular_values=np.linalg.svd(pixels[:,sl]-pixels[:,sl].mean(0),compute_uv=False).tolist()) for name,sl in [('source',slice(0,2)),('target',slice(2,4))]}
        if len(items)<12:row['reasons']=['BLOCKED_INSUFFICIENT_HOLDOUT'];continue
        held=[i for i in range(len(items)) if i%4==3];train=[i for i in range(len(items)) if i%4!=3]
        ti=[ids[i] for i in train];hi=[ids[i] for i in held];row.update(train_ids=ti,holdout_ids=hi)
        row['counts'].update(train=len(train),holdout=len(held));tr=pixels[train];ho=pixels[held]
        order=np.roll(np.arange(len(held)),1);control=ho.copy();control[:,2:]=ho[order,2:]
        row['negative_control']=dict(kind='CYCLIC_TARGET_SHIFT_ONE_HELDOUT_ONLY',target_point_ids=[hi[i] for i in order],F=None,H=None,refit=False)
        for label,fit,residual in [('F',fit_fundamental,lambda m,p:_residual('fundamental',m,p)),('H',homography,herror)]:
            budget()
            try:
                model=fit(tr);training=residual(model,tr);testing=residual(model,ho);negative=residual(model,control)
                row[label]=dict(status='EVALUATED',model=model,train=metrics(training,ti),holdout=metrics(testing,hi),residual_definition='sqrt Sampson squared distance in original raw pixels' if label=='F' else 'forward Euclidean pixel reprojection error')
                row['negative_control'][label]=metrics(negative,hi)
            except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                row[label]=dict(status='BLOCKED',reason=str(exc));row['reasons'].append(label+': '+str(exc))
            budget()
        if any(row[k]['status']=='EVALUATED' for k in ('F','H')):row['status']='UNCALIBRATED_EPIPOLAR_DIAGNOSTIC'
    return dict(status='UNCALIBRATED_EPIPOLAR_LEDGER_COMPLETE',pairs=pairs,omitted_pairs=omitted,
                counts=dict(pairs=len(pairs),diagnostic=sum(p['status']=='UNCALIBRATED_EPIPOLAR_DIAGNOSTIC' for p in pairs),blocked=sum(p['status']=='BLOCKED' for p in pairs)),
                split='stable point_id sort; every fourth (index 3 mod 4) held out; no refit or point removal',
                limitations=['Unknown intrinsics/distortion; no recovered R/t/K/XYZ.','Heldout coordinates share upstream tracking and selection; not independent correspondence truth.','F and H use different residual definitions and model dimensions; no numerical pass threshold or geometry acceptance.'],**RIGHTS)
