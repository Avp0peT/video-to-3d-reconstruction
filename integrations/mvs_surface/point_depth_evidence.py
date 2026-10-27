"""Read-only consistency of point candidates against original native depth views.

These correlated reconstruction inputs are not independent physical ground truth.
No points, cameras, masks or native depths are edited or inferred from this audit.
"""
import numpy as np
from .local_depth_search import _view

STATES = {0:'ORIGIN_EXCLUDED', 1:'OUT_OF_VIEW', 2:'NO_NATIVE_DEPTH',
          3:'MASK_BACKGROUND_DIAGNOSTIC', 4:'UNKNOWN_DISCONTINUITY',
          5:'DEPTH_SUPPORT', 6:'OCCLUDED', 7:'FREE_SPACE_CONTRADICTION',
          8:'DEPTH_INCONSISTENT'}
LABELS = {0:'UNVERIFIED', 1:'SUPPORTED', 2:'CONFLICT'}
POLICY = dict(max_points=200000,max_views=128,max_state_cells=25600000,chunk_points=8192,
              stable_depth_relative_spread=.02,axial_relative_tolerance=.02,
              world_distance_semantics='Diagnostic only; not an additional support gate',min_distinct_supports=2,
              min_support_angle_degrees=1.,native_sampling='Full valid foreground 2x2; inverse-depth bilinear at pixel centers; no boundary clamping',
              conflict_semantics='Evidence conflict channels include diagnostic mask background; none establish independent physical truth',
              depth_inconsistent_state_used=False,mask_background_is_separate_diagnostic_channel=True,mixed_mask_boundary_is_unknown=True,explicit_validity_required=True,
              depth_semantics='Only existing native depth arrays; zero is unknown; inferred shell is never an input replacement')


def assess_points(points,views,source_view_ids=None,check=lambda:None):
    """Classify unchanged finite world points, excluding each point's own depth view.

    A SUPPORT label requires two selected distinct source IDs and ray angle >=1deg,
    with no free-space or all-known-background diagnostic flag. Mixed-mask
    boundary patches remain UNKNOWN_DISCONTINUITY, never background conflicts.
    Occlusion and unavailable evidence remain UNVERIFIED rather than incorrect.
    """
    check()
    if not isinstance(points,np.ndarray) or points.dtype.kind!='f' or points.ndim!=2 or points.shape[1]!=3 or len(points)>200000 or not np.isfinite(points).all():
        raise ValueError('finite floating Nx3 points, at most 200000, required')
    if not isinstance(views,list) or not 1<=len(views)<=128:
        raise ValueError('one to 128 native views required')
    selected=[]
    for raw in views:
        check();v=_view(raw)
        validity=v.get('validity')
        if not isinstance(validity,np.ndarray) or validity.dtype!=bool or validity.shape!=v['mask'].shape:
            raise ValueError('invalid validity mask')
        available=v.get('depth_available',True)
        if type(available) is not bool: raise ValueError('depth_available must be boolean')
        v['validity']=validity;v['depth_available']=available
        selected.append(v)
    ids=[v['view_id'] for v in selected]
    if len(set(ids))!=len(ids): raise ValueError('duplicate view IDs cannot supply independent support')
    n=len(points);m=len(selected)
    if source_view_ids is None:
        origin=np.full(n,-1,np.int64)
    else:
        if not isinstance(source_view_ids,np.ndarray) or source_view_ids.shape!=(n,) or source_view_ids.dtype.kind not in 'iu' or (source_view_ids.dtype.kind=='i' and (source_view_ids < -1).any()):
            raise ValueError('source_view_ids must be integer N with -1 for unknown')
        if source_view_ids.dtype.kind=='u' and (source_view_ids>np.iinfo(np.int64).max).any(): raise ValueError('source ID outside int64')
        origin=source_view_ids.astype(np.int64,copy=True)
    states=np.full((n,m),1,np.uint8)
    axial_error=np.full((n,m),np.nan,np.float32)
    world_error=np.full((n,m),np.nan,np.float32)
    signed_residual=np.full((n,m),np.nan,np.float32)
    sampled_depth=np.full((n,m),np.nan,np.float32)
    max_angle=np.zeros(n,np.float64)
    for start in range(0,n,8192):
        check();end=min(start+8192,n);p=points[start:end].astype(np.float64,copy=False)
        support_directions=np.zeros((len(p),m,3),np.float64)
        for j,v in enumerate(selected):
            check();state=states[start:end,j]
            excluded=origin[start:end]==v['view_id'];state[excluded]=0
            camera=(p-v['C'])@v['R'].T
            pix=camera@v['K'].T
            positive=camera[:,2]>1e-9
            uv=pix[:,:2]/np.where(positive,camera[:,2],1.)[:,None]
            h,w=v['mask'].shape
            # x0+1,y0+1 must genuinely exist, even at an integer boundary sample.
            inside=~excluded & positive & np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<w-1)&(uv[:,1]<h-1)
            rows=np.flatnonzero(inside)
            if not len(rows): continue
            u=uv[rows,0];y=uv[rows,1]
            x0=np.floor(u).astype(int);y0=np.floor(y).astype(int)
            ix=np.column_stack((x0,x0+1,x0,x0+1))
            iy=np.column_stack((y0,y0,y0+1,y0+1))
            valid=v['validity'][iy,ix].all(axis=1)
            maskpatch=v['mask'][iy,ix]
            fg=maskpatch.all(axis=1)
            ds=v['depth'][iy,ix].astype(np.float64)
            state[rows]=4
            background=valid&(~maskpatch).all(axis=1);state[rows[background]]=3
            known=valid&fg
            if not v['depth_available']:
                state[rows[known]]=2;continue
            depth_known=(ds>0).all(axis=1)&np.isfinite(ds).all(axis=1)
            state[rows[known&~depth_known]]=2
            stable=known&depth_known&((ds.max(axis=1)/np.maximum(ds.min(axis=1),1e-30)-1)<=.02)
            q=np.flatnonzero(stable)
            if not len(q): continue
            rows=rows[q];dx=u[q]-x0[q];dy=y[q]-y0[q]
            weights=np.column_stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy))
            depth=1./np.sum(weights/ds[q],axis=1)
            zs=camera[rows,2]
            delta=(zs-depth)/depth
            axial_error[start+rows,j]=np.abs(delta)
            signed_residual[start+rows,j]=delta
            sampled_depth[start+rows,j]=depth
            near=np.abs(delta)<=.02
            state[rows[delta>.02]]=6
            state[rows[delta<-.02]]=7
            rays=np.column_stack((uv[rows],np.ones(len(rows))))@np.linalg.inv(v['K']).T
            backed=(rays*depth[:,None])@v['R']+v['C']
            distance=np.linalg.norm(p[rows]-backed,axis=1)
            world_error[start+rows,j]=distance
            # World distance is correlated with axial residual, not independent validation.
            good=near
            state[rows[good]]=5
            directions=p[rows[good]]-v['C']
            support_directions[rows[good],j]=directions/np.linalg.norm(directions,axis=1)[:,None]
        chunk_support=states[start:end]==5
        # Pairwise selected view IDs are distinct; only actual depth supports vote.
        for a in range(m):
            for b in range(a+1,m):
                both=chunk_support[:,a]&chunk_support[:,b]
                if both.any():
                    dot=np.sum(support_directions[both,a]*support_directions[both,b],axis=1)
                    angle=np.degrees(np.arccos(np.clip(dot,-1.,1.)))
                    subset=np.flatnonzero(both)+start
                    max_angle[subset]=np.maximum(max_angle[subset],angle)
    flags={name.lower():states==key for key,name in STATES.items()}
    state_counts={name.lower()+'_count':np.sum(states==key,axis=1,dtype=np.uint8) for key,name in STATES.items()}
    contradictions=(states==3)|(states==7)|(states==8)
    mask_conflict=(states==3).any(axis=1)
    native_depth_conflict=((states==7)|(states==8)).any(axis=1)
    conflict=contradictions.any(axis=1)
    supported=(state_counts['depth_support_count']>=2)&(max_angle>=1.)&~conflict
    labels=np.zeros(n,np.uint8);labels[supported]=1;labels[conflict]=2
    per_view={str(iid):{name:int(np.count_nonzero(states[:,j]==key)) for key,name in STATES.items()} for j,iid in enumerate(ids)}
    return dict(state=states,state_legend=dict(STATES),classification=labels,classification_legend=dict(LABELS),
        supported=supported,conflict=conflict,mask_background_diagnostic=mask_conflict,native_depth_conflict=native_depth_conflict,unverified=labels==0,max_support_angle_degrees=max_angle,
        source_view_ids=origin,origin_unknown=origin==-1,axial_relative_error=axial_error,world_distance_error=world_error,signed_relative_residual=signed_residual,sampled_depth=sampled_depth,
        **state_counts,**{'flag_'+name:value for name,value in flags.items()},support_count=state_counts['depth_support_count'],view_ids=ids,per_view=per_view,
        counts=dict(points=n,views=m,supported=int(supported.sum()),conflict=int(conflict.sum()),unverified=int((labels==0).sum()),origin_unknown=int((origin==-1).sum()),mask_background_diagnostic=int(mask_conflict.sum()),native_depth_conflict=int(native_depth_conflict.sum()),known_origin_not_selected=int(((origin>=0)&~np.isin(origin,ids)).sum())),
        policy=dict(POLICY),points_modified=False,geometry_accepted=False,physical_truth='NOT_ESTABLISHED',
        semantics='Native depth/mask evidence consistency only. Shared calibration and reconstruction are correlated. MASK_BACKGROUND_DIAGNOSTIC is a conflict channel, not proof of incorrect physical geometry. Unknown and occlusion never count as negative evidence.')
