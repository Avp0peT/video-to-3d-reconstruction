"""Bounded projective native-depth fusion, with no surfaces across unknown cells."""
import numpy as np
from .local_depth_search import _view


def _corners(a, operation):
    values=[a[x:a.shape[0]-1+x,y:a.shape[1]-1+y,z:a.shape[2]-1+z]
            for x in (0,1) for y in (0,1) for z in (0,1)]
    result=values[0].copy()
    for value in values[1:]:operation(result,value,out=result)
    return result


def fuse_depth_views(views,bounds,*,max_axis=128,min_views=2,truncation_voxels=3,check=lambda:None):
    """Fuse axial native depths; reject extraction cells without near evidence.

    Positive free-space distances are clipped to +1 and counted once per view.
    Samples more than tau behind a measured surface are ignored as occluded.
    A mesh cell needs all eight lattice samples supported by min_views, including
    min_views NEAR-band votes at every corner. This deliberately leaves holes.
    """
    check()
    if type(max_axis) is not int or not 8<=max_axis<=192:raise ValueError('max_axis must be integer 8..192')
    if type(min_views) is not int or not 2<=min_views<=64:raise ValueError('min_views must be integer 2..64')
    if type(truncation_voxels) not in (int,float) or not np.isfinite(truncation_voxels) or not 0<truncation_voxels<=10:raise ValueError('truncation_voxels must be finite (0,10]')
    if not isinstance(views,list) or not 1<=len(views)<=64:raise ValueError('one to sixty-four views required')
    selected=[]
    for raw in views:
        check();v=_view(raw);valid=v.get('validity');available=v.get('depth_available',True)
        if not isinstance(valid,np.ndarray) or valid.dtype!=bool or valid.shape!=v['mask'].shape:raise ValueError('explicit validity mask required')
        if type(available) is not bool:raise ValueError('depth_available must be bool')
        v['validity']=valid;v['depth_available']=available;selected.append(v)
    ids=[v['view_id'] for v in selected]
    if len(set(ids))!=len(ids):raise ValueError('duplicate view IDs')
    box=np.asarray(bounds)
    if box.shape!=(2,3) or box.dtype.kind not in 'fiu' or not np.isfinite(box).all():raise ValueError('finite world AABB [2,3] required')
    box=box.astype(np.float64);extent=box[1]-box[0]
    if not np.isfinite(extent).all() or (extent<=0).any():raise ValueError('positive finite bounds extents required')
    pitch=float(extent.max()/max_axis)
    dimensions=np.maximum(2,np.ceil(extent/pitch).astype(np.int64)+1)
    shape=tuple(map(int,dimensions));total=int(np.prod(dimensions))
    if total>8000000:raise ValueError('8000000 lattice sample budget exceeded')
    spacing=extent/(dimensions-1);tau=float(truncation_voxels*spacing.max())
    sums=np.zeros(total,np.float32);squared_sums=np.zeros(total,np.float32);weight=np.zeros(total,np.uint8);near=np.zeros(total,np.uint8)
    background=np.zeros(total,np.uint8);bits=np.zeros(total,np.uint64);near_bits=np.zeros(total,np.uint64)
    per_view={str(v['view_id']):dict(available=v['depth_available'],stable_foreground_samples=0,integrated_samples=0,near_samples=0,occluded_samples=0,mask_background_samples=0) for v in selected}
    for start in range(0,total,32768):
        check();end=min(start+32768,total);indices=np.arange(start,end)
        grid=np.column_stack(np.unravel_index(indices,shape));points=box[0]+grid*spacing
        for column,v in enumerate(selected):
            check();stats=per_view[str(v['view_id'])]
            camera=(points-v['C'])@v['R'].T;positive=camera[:,2]>1e-9
            pix=camera@v['K'].T;uv=pix[:,:2]/np.where(positive,camera[:,2],1)[:,None]
            h,w=v['mask'].shape
            inside=positive & np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
            rows=np.flatnonzero(inside)
            if not len(rows):continue
            u,y=uv[rows,0],uv[rows,1];x0=np.floor(u).astype(int);y0=np.floor(y).astype(int)
            ix=np.column_stack((x0,x0+1,x0,x0+1));iy=np.column_stack((y0,y0,y0+1,y0+1))
            valid=v['validity'][iy,ix].all(axis=1);mask=v['mask'][iy,ix]
            bg=valid&(~mask).all(axis=1);background[start+rows[bg]]+=1;stats['mask_background_samples']+=int(bg.sum())
            if not v['depth_available']:continue
            ds=v['depth'][iy,ix].astype(np.float64)
            stable=valid&mask.all(axis=1)&(ds>0).all(axis=1)&((ds.max(axis=1)/np.maximum(ds.min(axis=1),1e-30)-1)<=.02)
            q=np.flatnonzero(stable);stats['stable_foreground_samples']+=len(q)
            if not len(q):continue
            r=rows[q];dx=u[q]-x0[q];dy=y[q]-y0[q]
            interpolation=np.column_stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy))
            depth=1/np.sum(interpolation/ds[q],axis=1);signed=depth-camera[r,2]
            integrate=signed>=-tau;close=np.abs(signed)<=tau
            stats['occluded_samples']+=int((~integrate).sum());stats['integrated_samples']+=int(integrate.sum());stats['near_samples']+=int(close.sum())
            target=start+r[integrate];contribution=np.clip(signed[integrate]/tau,-1,1).astype(np.float32)
            sums[target]+=contribution;squared_sums[target]+=contribution*contribution;weight[target]+=1
            bits[target]|=np.uint64(1<<column)
            target=start+r[close];near[target]+=1;near_bits[target]|=np.uint64(1<<column)
    observed=weight>=min_views;near_observed=near>=min_views
    tsdf=np.ones(total,np.float32);np.divide(sums,weight,out=tsdf,where=weight>0)
    variance=np.zeros(total,np.float32);np.divide(squared_sums,weight,out=variance,where=weight>0)
    variance=np.where(weight>0,np.maximum(variance-tsdf*tsdf,0),0).astype(np.float32).reshape(shape)
    tsdf=tsdf.reshape(shape);weight=weight.reshape(shape);near=near.reshape(shape)
    bits=bits.reshape(shape);near_bits=near_bits.reshape(shape);background=background.reshape(shape)
    trusted=(observed&near_observed).reshape(shape)
    eligible=_corners(trusted,np.logical_and)
    cell_weights=_corners(weight,np.minimum);cell_near=_corners(near,np.minimum)
    cell_bits=_corners(bits,np.bitwise_and);cell_near_bits=_corners(near_bits,np.bitwise_and)
    vertices=np.empty((0,3),np.float64);faces=np.empty((0,3),np.int64);face_cells=np.empty((0,3),np.int64)
    raw_faces=0;removed_unknown=0
    check()
    if eligible.any() and (tsdf<0).any() and (tsdf>0).any():
        from skimage.measure import marching_cubes
        # Unknowns have +1 for numerical storage only; no face crossing an
        # untrusted cell may survive the explicit all-eight-corner filter.
        # Explicitly freeze the tested skimage winding: for volume axes (x,y,z)
        # and negative-inside TSDF, descent gives right-hand triangle normals
        # toward positive free space. See the directed planar regression test.
        raw_vertices,raw_triangles,_,_=marching_cubes(tsdf,level=0.,allow_degenerate=False,gradient_direction='descent')
        check();raw_faces=len(raw_triangles)
        triangles=raw_vertices[raw_triangles];centers=triangles.mean(axis=1)
        selected_cell=np.full((len(raw_triangles),3),-1,np.int64);keep=np.zeros(len(raw_triangles),bool)
        # Flat surfaces exactly on a lattice plane may belong to either side.
        for ox in (0,-1e-5):
            for oy in (0,-1e-5):
                for oz in (0,-1e-5):
                    check();candidate=np.floor(centers+[ox,oy,oz]).astype(np.int64)
                    inside=((candidate>=0)&(candidate<dimensions-1)).all(axis=1)
                    rows=np.flatnonzero(inside&~keep)
                    if not len(rows):continue
                    cell=candidate[rows];in_cell=((triangles[rows]>=cell[:,None,:]-1e-5)&(triangles[rows]<=cell[:,None,:]+1+1e-5)).all(axis=(1,2))
                    good=in_cell&eligible[cell[:,0],cell[:,1],cell[:,2]]
                    rows=rows[good];keep[rows]=True;selected_cell[rows]=candidate[rows]
        retained=raw_triangles[keep];face_cells=selected_cell[keep];removed_unknown=int((~keep).sum())
        if len(retained)>400000:raise ValueError('400000 face budget exceeded; no truncation')
        if len(retained):
            used,inverse=np.unique(retained,return_inverse=True)
            if len(used)>200000:raise ValueError('200000 vertex budget exceeded; no truncation')
            vertices=box[0]+raw_vertices[used].astype(np.float64)*spacing;faces=inverse.reshape(-1,3).astype(np.int64)
    if not np.isfinite(vertices).all():raise ValueError('nonfinite surface vertices')
    if len(faces):
        fc=tuple(face_cells.T);face_weight=cell_weights[fc];face_near=cell_near[fc]
        face_bits=cell_bits[fc];face_near_bits=cell_near_bits[fc]
        vertex_weight=np.full(len(vertices),255,np.uint8);vertex_near=vertex_weight.copy()
        for index in range(3):
            np.minimum.at(vertex_weight,faces[:,index],face_weight);np.minimum.at(vertex_near,faces[:,index],face_near)
    else:
        face_weight=np.empty(0,np.uint8);face_near=face_weight.copy();face_bits=np.empty(0,np.uint64);face_near_bits=face_bits.copy()
        vertex_weight=np.empty(0,np.uint8);vertex_near=vertex_weight.copy()
    check()
    return dict(vertices=vertices,faces=faces,colors=np.tile(np.array([70,175,225],np.uint8),(len(vertices),1)),
        tsdf=tsdf,weight=weight,tsdf_vote_variance=variance,near_surface_votes=near,observed_mask=observed.reshape(shape),trusted_mask=trusted,eligible_cells=eligible,
        mask_background_count=background,mask_background_votes=background,view_support_bits=bits,near_view_support_bits=near_bits,
        face_grid_cell=face_cells,face_min_weight=face_weight,face_min_near_votes=face_near,
        face_common_support_bits=face_bits,face_common_near_bits=face_near_bits,
        vertex_min_weight=vertex_weight,vertex_min_near_votes=vertex_near,
        source_ids=ids,per_view=per_view,status='PROJECTIVE_TSDF_SURFACE_NOT_GEOMETRY_ACCEPTED' if len(faces) else 'NO_SUPPORTED_TSDF_SURFACE',
        geometry_accepted=False,watertight='NOT_ESTABLISHED',complete=False,
        counts=dict(lattice_samples=total,observed_samples=int(observed.sum()),trusted_samples=int(trusted.sum()),eligible_cells=int(eligible.sum()),raw_extracted_faces=raw_faces,removed_untrusted_cell_faces=removed_unknown,vertices=len(vertices),faces=len(faces)),
        parameters=dict(world_aabb=box.tolist(),dimensions=list(shape),spacing=spacing.tolist(),truncation_distance=tau,max_axis=max_axis,min_views=min_views,truncation_voxels=truncation_voxels,
            max_views=64,view_support_dtype='uint64',vote_count_dtype='uint8 (at most64 distinct views)',stable_depth_relative_spread=.02,sampling='Full valid foreground positive 2x2, inverse-depth bilinear',positive_free_space='Clip signed distances greater than tau to +1; one weight per native view',negative_occlusion='Ignore s<-tau',
            variance_semantics='Population variance of clipped normalized signed-distance votes, zero when no votes; inspect weight separately',color_semantics='Uniform display blue, not recovered photo color',
            face_winding='Marching cubes descent: tested right-hand normals toward positive free space for negative-inside TSDF',
            extraction='All eight cell corners require min_views total AND near-band votes; no unknown or AABB caps',max_vertices=200000,max_faces=400000),
        semantics='Cross-view fused surface estimate, not new measured points or independent truth. Native depth and calibration may share errors. Missing depth, background and unknown never write negative TSDF. Near votes and common-view bit intersections are separate; corner support may come from different view subsets. No pose updates, silhouette filling, smoothing or guaranteed closure. Thin gaps below grid/truncation scale may be missed; no physical error bound.')
