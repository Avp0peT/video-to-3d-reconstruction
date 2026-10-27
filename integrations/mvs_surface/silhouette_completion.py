"""Explicitly inferred silhouette distance envelope; never observed geometry."""
import cv2
import numpy as np
from .projective_tsdf import _corners


def _views(views,check):
    if not isinstance(views,list) or not 3<=len(views)<=128:raise ValueError('three to128 silhouette cameras required')
    result=[];ids=[]
    for raw in views:
        check();iid=raw.get('view_id');mask=raw.get('mask');valid=raw.get('validity')
        if type(iid) is not int or iid<0 or iid in ids:raise ValueError('distinct nonnegative view IDs required')
        if not isinstance(mask,np.ndarray) or mask.dtype!=bool or mask.ndim!=2 or min(mask.shape)<5 or mask.size>16000000:raise ValueError('bounded binary foreground mask required')
        if not isinstance(valid,np.ndarray) or valid.dtype!=bool or valid.shape!=mask.shape:raise ValueError('explicit validity required')
        k,r,c=[np.asarray(raw.get(key),np.float64) for key in ('K','R','C')]
        if k.shape!=(3,3) or r.shape!=(3,3) or c.shape!=(3,) or not all(np.isfinite(x).all() for x in (k,r,c)):raise ValueError('finite K R C required')
        if k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1],atol=1e-9,rtol=0) or abs(np.linalg.det(k))<1e-12:raise ValueError('invalid K')
        if not np.allclose(r@r.T,np.eye(3),atol=1e-6,rtol=0) or not np.isclose(np.linalg.det(r),1,atol=1e-6,rtol=0):raise ValueError('invalid rotation')
        fg=mask&valid;bg=~mask&valid
        # Unknown pixels do not become distance-transform background seeds.
        # If a sign is absent, image diagonal is a finite lower-information cap.
        cap=float(np.hypot(*mask.shape))
        to_bg=cv2.distanceTransform((~bg).astype(np.uint8),cv2.DIST_L2,5) if bg.any() else np.full(mask.shape,cap,np.float32)
        to_fg=cv2.distanceTransform((~fg).astype(np.uint8),cv2.DIST_L2,5) if fg.any() else np.full(mask.shape,cap,np.float32)
        distance=(to_bg-to_fg).astype(np.float32)
        result.append(dict(view_id=iid,K=k,R=r,C=c,mask=mask,validity=valid,distance=distance));ids.append(iid)
    centers=np.asarray([v['C'] for v in result]);_,groups=np.unique(centers,axis=0,return_inverse=True)
    if len(np.unique(groups))<3:raise ValueError('three distinct camera centers required for two distinct baselines')
    return result,groups


def infer_silhouette_completion(views,bounds,*,max_axis=160,quantile=.1,smoothing_sigma=0.,check=lambda:None):
    """Compute strict q0 or robust q.1 lower empirical silhouette envelope.

    Positive 2D signed distances are converted approximately by axial z/min(fx,fy).
    Unknown/outside/behind projections provide no signed vote. At least3 known
    distinct camera centers are required at every eligible lattice corner.
    No box padding, morphological closing, camera adjustment, or hidden-cavity claim.
    """
    check()
    if type(max_axis) is not int or not 8<=max_axis<=160:raise ValueError('max_axis integer8..160 required')
    if type(quantile) not in (int,float) or quantile not in (0.,.1):raise ValueError('only fixed quantile0 or0.1 supported')
    if type(smoothing_sigma) not in (int,float) or smoothing_sigma not in (0.,1.):raise ValueError('only explicit smoothing sigma0 or1 supported')
    selected,groups=_views(views,check)
    box=np.asarray(bounds)
    if box.shape!=(2,3) or box.dtype.kind not in 'fiu' or not np.isfinite(box).all():raise ValueError('finite world AABB required')
    box=box.astype(np.float64);extent=box[1]-box[0]
    if not np.isfinite(extent).all() or (extent<=0).any():raise ValueError('positive world AABB extents required')
    pitch=float(extent.max()/max_axis)
    if pitch<=0 or not np.isfinite(pitch):raise ValueError('invalid grid pitch')
    dimensions=np.maximum(2,np.ceil(extent/pitch).astype(np.int64)+1);shape=tuple(map(int,dimensions));spacing=extent/(dimensions-1)
    total=int(np.prod(dimensions))
    if total>5000000:raise ValueError('5000000 lattice sample limit')
    field=np.full(total,np.nan,np.float32);known=np.zeros(total,np.uint8);distinct=known.copy();fgvotes=known.copy();bgvotes=known.copy()
    per_view={str(v['view_id']):dict(known_lattice_votes=0,foreground_votes=0,background_votes=0,unknown_lattice_votes=0) for v in selected}
    for start in range(0,total,8192):
        check();end=min(total,start+8192);grid=np.column_stack(np.unravel_index(np.arange(start,end),shape));points=box[0]+grid*spacing
        distances=np.full((len(points),len(selected)),np.nan,np.float32)
        signs=np.zeros((len(points),len(selected)),np.int8)
        for col,v in enumerate(selected):
            check();camera=(points-v['C'])@v['R'].T;positive=camera[:,2]>1e-9;pix=camera@v['K'].T
            uv=pix[:,:2]/np.where(positive,camera[:,2],1)[:,None];h,w=v['mask'].shape
            inside=positive&np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
            rows=np.flatnonzero(inside);stats=per_view[str(v['view_id'])]
            if len(rows):
                u,y=uv[rows,0],uv[rows,1];x=np.floor(u).astype(int);iy=np.floor(y).astype(int)
                ix=np.column_stack((x,x+1,x,x+1));jy=np.column_stack((iy,iy,iy+1,iy+1))
                valid=v['validity'][jy,ix].all(axis=1);keep=np.flatnonzero(valid);rows=rows[keep]
                if len(rows):
                    dx=u[keep]-x[keep];dy=y[keep]-iy[keep]
                    weights=np.column_stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy))
                    signed=np.sum(weights*v['distance'][jy[keep],ix[keep]],axis=1)
                    world=signed*camera[rows,2]/min(v['K'][0,0],v['K'][1,1])
                    if not np.isfinite(world).all() or (np.abs(world)>np.finfo(np.float32).max).any():raise ValueError('world distance outside finite float32')
                    distances[rows,col]=world
                    signs[rows,col]=np.where(signed>=0,1,-1)
            stats['known_lattice_votes']+=len(rows);stats['unknown_lattice_votes']+=len(points)-len(rows)
            stats['foreground_votes']+=int((signs[:,col]>0).sum());stats['background_votes']+=int((signs[:,col]<0).sum())
        available=np.isfinite(distances);counts=available.sum(axis=1)
        groupcounts=np.zeros(len(points),np.uint8)
        for group in np.unique(groups):groupcounts+=available[:,groups==group].any(axis=1)
        usable=(counts>=3)&(groupcounts>=3)
        if usable.any():field[start:end][usable]=np.nanquantile(distances[usable],quantile,axis=1,method='lower')
        known[start:end]=counts;distinct[start:end]=groupcounts
        fgvotes[start:end]=(signs>0).sum(axis=1);bgvotes[start:end]=(signs<0).sum(axis=1)
    field=field.reshape(shape);known=known.reshape(shape);distinct=distinct.reshape(shape);fgvotes=fgvotes.reshape(shape);bgvotes=bgvotes.reshape(shape)
    trusted=np.isfinite(field)
    if smoothing_sigma:
        check()
        from scipy.ndimage import gaussian_filter
        numerator=gaussian_filter(np.where(trusted,field,0.),sigma=1.,mode='constant',cval=0.)
        denominator=gaussian_filter(trusted.astype(np.float32),sigma=1.,mode='constant',cval=0.)
        smoothed=numerator/np.maximum(denominator,1e-12)
        field=np.where(trusted,smoothed,np.nan).astype(np.float32)
    boundary=np.zeros(shape,bool);boundary[[0,-1],:,:]=True;boundary[:,[0,-1],:]=True;boundary[:,:,[0,-1]]=True
    boundary_safe=bool((trusted[boundary]&(field[boundary]<0)).all())
    eligible=_corners(trusted,np.logical_and)
    vertices=np.empty((0,3),np.float64);faces=np.empty((0,3),np.int64);face_cells=np.empty((0,3),np.int64)
    raw_count=0;removed=0
    if eligible.any() and (field[trusted]>0).any() and (field[trusted]<0).any():
        check();from skimage.measure import marching_cubes
        numerical=np.where(trusted,field,-1.).astype(np.float32)
        raw_vertices,raw_faces,_,_=marching_cubes(numerical,level=0.,allow_degenerate=False,gradient_direction='descent')
        raw_count=len(raw_faces)
        if raw_count>2000000:raise ValueError('raw extraction budget exceeded; no truncation')
        triangles=raw_vertices[raw_faces];centers=triangles.mean(axis=1);keep=np.zeros(raw_count,bool);selected_cell=np.full((raw_count,3),-1,np.int64)
        for ox in (0,-1e-5):
            for oy in (0,-1e-5):
                for oz in (0,-1e-5):
                    check();candidate=np.floor(centers+[ox,oy,oz]).astype(np.int64)
                    inside=((candidate>=0)&(candidate<dimensions-1)).all(axis=1);rows=np.flatnonzero(inside&~keep)
                    if not len(rows):continue
                    cells=candidate[rows];within=((triangles[rows]>=cells[:,None,:]-1e-5)&(triangles[rows]<=cells[:,None,:]+1+1e-5)).all(axis=(1,2))
                    good=within&eligible[cells[:,0],cells[:,1],cells[:,2]];rows=rows[good];keep[rows]=True;selected_cell[rows]=candidate[rows]
        retained=raw_faces[keep];face_cells=selected_cell[keep];removed=int((~keep).sum())
        if len(retained)>400000:raise ValueError('400000 face limit; no truncation')
        if len(retained):
            used,inverse=np.unique(retained,return_inverse=True)
            if len(used)>200000:raise ValueError('200000 vertex limit; no truncation')
            vertices=box[0]+raw_vertices[used].astype(float)*spacing;faces=inverse.reshape(-1,3).astype(np.int64)
    min_known=_corners(known,np.minimum);min_fg=_corners(fgvotes,np.minimum);max_bg=_corners(bgvotes,np.maximum)
    if len(faces):
        cells=tuple(face_cells.T)
        edges=np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1);_,edgecounts=np.unique(edges,axis=0,return_counts=True)
        boundary_edges=int((edgecounts==1).sum());nonmanifold=int((edgecounts>2).sum())
        face_known=min_known[cells];face_fg=min_fg[cells];face_bg=max_bg[cells]
    else:boundary_edges=0;nonmanifold=0;face_known=np.empty(0,np.uint8);face_fg=face_known.copy();face_bg=face_known.copy()
    check()
    closed=bool(len(faces) and boundary_edges==0 and nonmanifold==0)
    return dict(vertices=vertices,faces=faces,colors=np.tile(np.array([255,150,30],np.uint8),(len(vertices),1)),field=field,known_count=known,distinct_center_count=distinct,foreground_votes=fgvotes,background_votes=bgvotes,trusted_mask=trusted,eligible_cells=eligible,face_grid_cell=face_cells,face_min_known=face_known,face_min_foreground_votes=face_fg,face_max_background_votes=face_bg,
        source_ids=[v['view_id'] for v in selected],per_view=per_view,geometry_accepted=False,complete=False,kind='INFERRED_SILHOUETTE_COMPLETION_NOT_OBSERVED',status='INFERRED_ENVELOPE_NOT_GEOMETRY_ACCEPTED' if boundary_safe else 'UNKNOWN_VOLUME_BOUNDARY',
        boundary_known_negative=boundary_safe,closed_mesh_topology=closed,closed_envelope_with_safe_boundary=closed and boundary_safe,
        counts=dict(lattice_samples=total,trusted_samples=int(trusted.sum()),positive_samples=int((field>0).sum()),unknown_samples=int((~trusted).sum()),raw_faces=raw_count,removed_untrusted_faces=removed,vertices=len(vertices),faces=len(faces),boundary_edges=boundary_edges,nonmanifold_edges=nonmanifold),
        parameters=dict(world_aabb=box.tolist(),dimensions=list(shape),spacing=spacing.tolist(),max_axis=max_axis,quantile=float(quantile),quantile_method='lower',smoothing_sigma=float(smoothing_sigma),min_known_views=3,min_distinct_centers=3,two_distinct_nonzero_baselines=True,max_views=128,chunk_points=8192,max_vertices=200000,max_faces=400000,world_distance='Signed 2D image distance times axial z/min(fx,fy), approximate',unknown_caps=False,volume_caps=False),
        semantics='Explicit inferred outer envelope; q0 intersects available silhouettes and q0.1 permits minority known silhouette conflicts. Unknown/outside/behind provide no background vote. Boundaries unknown or touching box cannot establish a closed object. Not measured depth, hidden concavity, identity/rigidity guarantee, gap preservation, independent truth, CAD or completeness. Sigma1 is optional declared field smoothing and can erase subgrid gaps.')
