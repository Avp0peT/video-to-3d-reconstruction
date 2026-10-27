"""Independent sampled silhouette boundary proposal; never observed geometry."""
import numpy as np


def _views(views, reference_id, roi):
    if not isinstance(views, list) or not 3 <= len(views) <= 15:
        raise ValueError('three to fifteen calibrated views required')
    result = []
    for v in views:
        if not isinstance(v, dict) or type(v.get('view_id')) is not int:
            raise ValueError('invalid view identity')
        mask = v.get('mask'); valid = v.get('validity')
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.ndim != 2 or not mask.size:
            raise ValueError('invalid foreground mask')
        if not isinstance(valid, np.ndarray) or valid.dtype != bool or valid.shape != mask.shape:
            raise ValueError('explicit matching validity mask required')
        matrices = []
        for name, shape in [('K',(3,3)), ('R',(3,3)), ('C',(3,))]:
            a = np.asarray(v.get(name))
            if a.shape != shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
                raise ValueError('invalid camera '+name)
            matrices.append(a.astype(np.float64))
        k,r,c = matrices
        if not np.allclose(k[2], [0,0,1], atol=1e-8, rtol=0) or k[0,0] <= 0 or k[1,1] <= 0 or abs(np.linalg.det(k)) < 1e-10:
            raise ValueError('invalid calibration')
        if not np.allclose(r@r.T,np.eye(3),atol=1e-6,rtol=0) or not np.isclose(np.linalg.det(r),1,atol=1e-6,rtol=0):
            raise ValueError('invalid rotation')
        result.append(dict(view_id=v['view_id'], mask=mask, validity=valid, K=k, R=r, C=c))
    ids = [v['view_id'] for v in result]
    if len(set(ids)) != len(ids) or type(reference_id) is not int or reference_id not in ids:
        raise ValueError('distinct IDs and a present reference required')
    ref = result[ids.index(reference_id)]
    if not isinstance(roi,np.ndarray) or roi.dtype != bool or roi.shape != ref['mask'].shape:
        raise ValueError('invalid reference ROI')
    if (roi & ~(ref['mask'] & ref['validity'])).any():
        raise ValueError('ROI must lie in known reference foreground')
    return result, ref


def _project(view, points, roi=None):
    camera = (points-view['C'])@view['R'].T
    positive = camera[:,2] > 1e-9
    homogeneous = camera@view['K'].T
    uv = homogeneous[:,:2]/np.where(positive,camera[:,2],1)[:,None]
    h,w = view['mask'].shape
    inside = positive & np.isfinite(uv).all(axis=1) & (uv[:,0]>=0) & (uv[:,0]<=w-1) & (uv[:,1]>=0) & (uv[:,1]<=h-1)
    ix = np.rint(np.clip(uv[:,0],0,w-1)).astype(np.int64)
    iy = np.rint(np.clip(uv[:,1],0,h-1)).astype(np.int64)
    known = inside & view['validity'][iy,ix]
    fg = known & view['mask'][iy,ix]
    bg = known & ~view['mask'][iy,ix]
    domain = fg & roi[iy,ix] if roi is not None else fg
    return fg,bg,domain


def _classify(views, ref, roi, points, check):
    # 0 UNKNOWN, 1 occupied, 2 known outside; outside reference ROI is UNKNOWN.
    _,_,domain = _project(ref,points,roi)
    support = np.zeros(len(points),np.uint8)
    background = np.zeros(len(points),bool)
    for v in views:
        check()
        fg,bg,_ = _project(v,points)
        support += fg
        background |= bg
    state = np.zeros(len(points),np.uint8)
    state[domain & background] = 2
    state[domain & ~background & (support>=3)] = 1
    return state, domain


def build_local_visual_hull(views, reference_id, roi, bounds, check=lambda:None):
    """Emit occupied-to-known-outside voxel faces, never UNKNOWN/domain caps.

    Occupancy is center-sampled. Eight additional corners quantify classification
    disagreement; they DO NOT certify continuous silhouettes or alter the center
    policy. Mixed-corner faces remain explicitly approximate, independently
    labelled proposals. No old depth is read or fused and no smoothing is done.
    """
    check()
    views,ref = _views(views,reference_id,roi)
    box = np.asarray(bounds)
    if box.shape != (2,3) or box.dtype.kind not in 'fiu' or not np.isfinite(box).all():
        raise ValueError('finite world AABB [2,3] required')
    box = box.astype(np.float64)
    extent = box[1]-box[0]
    if not np.isfinite(extent).all() or (extent<=0).any(): raise ValueError('positive finite AABB extents required')
    pitch = float(extent.max()/96)
    dimensions = np.maximum(2,np.ceil(extent/pitch).astype(np.int64))
    if np.prod(dimensions)>1000000: raise ValueError('voxel budget exceeded')
    spacing = extent/dimensions
    shape = tuple(map(int,dimensions))
    total = int(np.prod(dimensions))
    state = np.zeros(total,np.uint8)
    domain = np.zeros(total,bool)
    corner_disagreement = np.zeros(total,bool)
    corner_unknown = np.zeros(total,bool)
    corners = np.array([[a,b,c] for a in (0,1) for b in (0,1) for c in (0,1)],float)
    for start in range(0,total,16384):
        check(); end = min(total,start+16384)
        ijk = np.column_stack(np.unravel_index(np.arange(start,end),shape))
        center = box[0]+(ijk+.5)*spacing
        s,d = _classify(views,ref,roi,center,check)
        state[start:end],domain[start:end] = s,d
        for corner in corners:
            check()
            other,_ = _classify(views,ref,roi,box[0]+(ijk+corner)*spacing,check)
            corner_disagreement[start:end] |= other!=s
            corner_unknown[start:end] |= other==0
    state = state.reshape(shape)
    domain = domain.reshape(shape)
    corner_disagreement = corner_disagreement.reshape(shape)
    corner_unknown = corner_unknown.reshape(shape)
    quads = []; ambiguous_faces = []; omitted_unknown = 0; omitted_volume = 0
    # Corner winding is outward from each occupied cell, axis x/y/z respectively.
    positive_offsets = [np.array(v,np.int64) for v in (
        [[1,0,0],[1,1,0],[1,1,1],[1,0,1]],
        [[0,1,0],[0,1,1],[1,1,1],[1,1,0]],
        [[0,0,1],[1,0,1],[1,1,1],[0,1,1]])]
    for axis in range(3):
        for sign in (-1,1):
            check()
            at_edge = [slice(None)]*3; at_edge[axis] = 0 if sign<0 else -1
            omitted_volume += int((state[tuple(at_edge)]==1).sum())
            own = [slice(None)]*3; neighbor = [slice(None)]*3
            own[axis] = slice(1,None) if sign<0 else slice(None,-1)
            neighbor[axis] = slice(None,-1) if sign<0 else slice(1,None)
            own,neighbor = tuple(own),tuple(neighbor)
            occupied = state[own]==1
            omitted_unknown += int((occupied & (state[neighbor]==0)).sum())
            cells = np.column_stack(np.nonzero(occupied & (state[neighbor]==2)))
            if sign<0: cells[:,axis] += 1
            if not len(cells): continue
            if 2*(sum(len(q) for q in quads)+len(cells))>300000:
                raise ValueError('face budget exceeded; no truncation')
            offsets = positive_offsets[axis].copy()
            if sign<0:
                offsets[:,axis] = 0
                offsets = offsets[::-1]
            quads.append(cells[:,None,:]+offsets[None,:,:])
            mixed = corner_disagreement[own] | corner_disagreement[neighbor]
            ambiguous_faces.append(mixed[occupied & (state[neighbor]==2)])
    quad_count = sum(len(q) for q in quads)
    if 2*quad_count>300000: raise ValueError('face budget exceeded; no truncation')
    if quads:
        grid,indices = np.unique(np.concatenate(quads).reshape(-1,3),axis=0,return_inverse=True)
        if len(grid)>200000: raise ValueError('vertex budget exceeded; no truncation')
        q = indices.reshape(-1,4)
        faces = np.concatenate((q[:,[0,1,2]],q[:,[0,2,3]]))
        vertices = box[0]+grid*spacing
        mixed = np.concatenate(ambiguous_faces)
        face_ambiguous = np.concatenate((mixed,mixed))
    else:
        vertices = np.empty((0,3),float); faces = np.empty((0,3),np.int64)
        face_ambiguous = np.empty(0,bool)
    check()
    return dict(vertices=vertices,colors=np.tile(np.array([255,150,30],np.uint8),(len(vertices),1)),faces=faces,
        voxel_state=state,reference_domain=domain,corner_disagreement=corner_disagreement,
        corner_unknown=corner_unknown,face_corner_ambiguous=face_ambiguous,
        status='INFERRED_SILHOUETTE_BOUNDARY',geometry_accepted=False,surface_accepted=False,
        source_ids=[v['view_id'] for v in views],reference_id=reference_id,
        voxel_info=dict(world_aabb=box.tolist(),dimensions=list(shape),spacing=spacing.tolist(),longest_axis_cells=96,
            total_voxels=total,voxel_diagonal=float(np.linalg.norm(spacing)),sampling='Center occupancy; fixed eight-corner disagreement diagnostics',
            discretization='Cell-face localization and center quantization are voxel-scale approximations. Voxel diagonal is grid scale, NOT a bound on true-surface error; subvoxel holes/intervals can be missed.'),
        counts=dict(occupied=int((state==1).sum()),known_outside=int((state==2).sum()),unknown=int((state==0).sum()),
            reference_domain=int(domain.sum()),outside_reference_domain=int((~domain).sum()),
            mixed_corner_cells=int(corner_disagreement.sum()),corner_unknown_cells=int(corner_unknown.sum()),
            emitted_quads=quad_count,vertices=len(vertices),faces=len(faces),
            omitted_unknown_adjacencies=omitted_unknown,omitted_volume_boundary_adjacencies=omitted_volume,
            ambiguous_corner_triangles=int(face_ambiguous.sum())),
        policy=dict(minimum_known_foreground_views=3,any_known_background_veto=True,reference_roi_required=True,
            max_vertices=200000,max_faces=300000,max_voxels=1000000,unknown_caps=False,volume_caps=False,corner_samples=8),
        semantics='Independent outer-shape proposal, not observed surface or hidden-concavity recovery. Historical masks/poses are not independent truth. UNKNOWN and AABB/ROI cuts remain open; no smoothing, depth fusion or sparse-endpoint stitching.')
