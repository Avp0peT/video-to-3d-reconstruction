"""Conditional, per-child silhouette evidence; no mesh edits or acceptance.

A negative vote proves either a default enclosing disk or opt-in closed-cell
triangle footprint is wholly in the supplied bounded gap and more than three
pixels from foreground/unknown, then exhibits actual integer pixel centres in
the triangle. Original supported probes protect every child
containing them as a closed set. The caller must aggregate views and veto the
union of their core flags; an UNKNOWN result never approves geometry.
"""
from dataclasses import dataclass
import math
import numpy as np

from .face_edge_gate import EdgePolicy, _project
from .face_mask_gate import _mesh, _camera, _view_id, _cv, UNKNOWN, CONFLICT


@dataclass(frozen=True)
class LocalPolicy(EdgePolicy):
    boundary_px: float = 3.
    min_background_pixels: int = 4
    max_raster_pixels_per_face: int = 200000
    min_angle_degrees: float = 8.
    min_views: int = 2

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.boundary_px <= 100 or not 0 < self.min_angle_degrees <= 180:
            raise ValueError('invalid local background margin/angle')
        for name in ('min_background_pixels', 'max_raster_pixels_per_face', 'min_views'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('positive local integer required: '+name)
        if not 4 <= self.min_background_pixels <= 64 or self.min_views < 2:
            raise ValueError('at least four pixel centres and two views required')


REASONS = dict(enumerate((
    'NOT_EVALUATED', 'INELIGIBLE_PARENT', 'CAMERA_PLANE', 'OUT_OF_FRAME',
    'INVALID_VERTEX_DOMAIN', 'SUBPIXEL_OR_DEGENERATE', 'NO_KNOWN_DOMAIN_DISK',
    'NO_WHOLE_GAP_PROOF', 'NO_DEEP_BACKGROUND_PROOF', 'PIXEL_BUDGET',
    'INSUFFICIENT_ACTUAL_PIXEL_CENTERS', 'WHOLE_GAP_NEGATIVE',
    'NO_KNOWN_DOMAIN_FOOTPRINT')))
_R = {value:key for key,value in REASONS.items()}
_PROBES = np.array([[0.,0.], [1.,0.], [0.,1.], [1/3,1/3]])


def closed_footprint_cells(triangle, width, height, budget, check=lambda:None):
    """Conservative closed triangle/unit-cell intersection using separating axes.

    Integer (x,y) denotes the closed square [x-.5,x+.5] x [y-.5,y+.5].
    Outward numerical tolerances add cells, never discard boundary contact.
    The complete bounding box is budgeted before allocation; no clipping.
    """
    t = np.asarray(triangle, dtype=float)
    if t.shape != (3,2) or not np.isfinite(t).all():
        raise ValueError('finite projected triangle required')
    for value in (width,height,budget):
        if type(value) is not int or value < 1:
            raise ValueError('positive integer footprint domain and budget required')
    if (t.min(axis=0) <= -.5).any() or (t.max(axis=0) >= np.array([width,height])-.5).any():
        return None, 'OUT_OF_FRAME'
    scale = max(1., float(np.abs(t).max()))
    eps = 64*np.finfo(float).eps*scale
    ab, ac = t[1]-t[0], t[2]-t[0]
    if abs(ab[0]*ac[1]-ab[1]*ac[0]) <= eps*scale:
        return None, 'SUBPIXEL_OR_DEGENERATE'
    lo = np.ceil(t.min(axis=0)-.5-eps).astype(np.int64)
    hi = np.floor(t.max(axis=0)+.5+eps).astype(np.int64)
    if (lo < 0).any() or (hi >= [width,height]).any():
        return None, 'OUT_OF_FRAME'
    nx, ny = hi-lo+1
    total = int(nx)*int(ny)
    if total > budget:
        return None, 'PIXEL_BUDGET'
    chunks = []
    for start in range(0,total,1024):
        check()
        flat = np.arange(start,min(start+1024,total),dtype=np.int64)
        cells = np.column_stack((lo[0]+flat%nx,lo[1]+flat//nx))
        hit = np.ones(len(cells),bool)
        for a,b in ((0,1),(1,2),(2,0)):
            edge = t[b]-t[a]; axis = np.array([-edge[1],edge[0]])
            projected = t@axis; centers = cells@axis
            radius = .5*np.abs(axis).sum()
            tolerance = eps*max(1.,float(np.abs(axis).sum()))
            hit &= ((centers+radius >= projected.min()-tolerance)
                    & (centers-radius <= projected.max()+tolerance))
        chunks.append(cells[hit])
    return np.concatenate(chunks), None


def _parents(parent_ids, face_count, original_count):
    ids = np.asarray(parent_ids)
    if (ids.shape != (face_count,) or not np.issubdtype(ids.dtype, np.integer)
            or (ids.size and (ids.min() < 0 or ids.max() >= original_count))):
        raise ValueError('one valid original face ID per child required')
    return ids


def prepare_probe_membership(vertices, faces, original_vertices, original_faces,
                             parent_ids, *, check=lambda:None):
    """Return bool[N,4] for the original vertices/centroid in each closed child.

    Precompute once on the exact pre-export coordinates. Plane/parent containment
    is checked. A dimensionless 1e-9 barycentric tolerance can only add protection.
    Degenerate child faces conservatively contain all probes and cannot produce a
    nondegenerate projection. No new vertex or protected isolated point is made.
    """
    v,f = _mesh(vertices, faces); ov,of = _mesh(original_vertices, original_faces)
    ids = _parents(parent_ids, len(f), len(of))
    result = np.ones((len(f),4), bool)
    for first in range(0, len(f), 16384):
        check(); last = min(first+16384, len(f))
        parent = ov[of[ids[first:last]]]; child = v[f[first:last]]
        u, w = parent[:,1]-parent[:,0], parent[:,2]-parent[:,0]
        normal = np.cross(u,w); norm2 = np.einsum('ij,ij->i',normal,normal)
        edge_scale = np.maximum(np.linalg.norm(u,axis=1), np.linalg.norm(w,axis=1))
        nonzero = norm2 > 0
        if np.any(~nonzero):
            # A rank-one parent still permits a collinear child on its closed segment.
            for j in np.flatnonzero(~nonzero):
                pairs = [(0,1),(1,2),(2,0)]
                a,b = max(pairs, key=lambda ab: np.linalg.norm(parent[j,ab[1]]-parent[j,ab[0]]))
                direction = parent[j,b]-parent[j,a]; scale2 = float(direction@direction)
                tolerance = 1e-9*max(math.sqrt(scale2),1e-12) + 64*np.finfo(float).eps*max(1.,float(np.abs(parent[j]).max()))
                if scale2:
                    t = (child[j]-parent[j,a])@direction/scale2
                    residual = child[j]-parent[j,a]-t[:,None]*direction
                    valid = (t >= -1e-9).all() and (t <= 1+1e-9).all() and (np.linalg.norm(residual,axis=1) <= tolerance).all()
                else:
                    valid = (np.linalg.norm(child[j]-parent[j,a],axis=1) <= tolerance).all()
                if not valid:
                    raise ValueError('child escapes degenerate original parent')
        rows = np.flatnonzero(nonzero)
        if not len(rows):
            continue
        delta = child[rows]-parent[rows,0,None,:]
        n = normal[rows]; den = norm2[rows]
        b1 = np.einsum('nki,ni->nk', np.cross(delta,w[rows,None,:]), n)/den[:,None]
        b2 = np.einsum('nki,ni->nk', np.cross(u[rows,None,:],delta), n)/den[:,None]
        bc = np.stack((1-b1-b2,b1,b2),axis=2)
        plane = np.abs(np.einsum('nki,ni->nk',delta,n))/np.sqrt(den)[:,None]
        coordinate_scale = np.maximum(1.,np.max(np.abs(parent[rows]),axis=(1,2)))
        tolerance = edge_scale[rows]*1e-9 + 64*np.finfo(float).eps*coordinate_scale
        if not ((bc >= -1e-9).all() and (bc <= 1+1e-9).all() and (plane <= tolerance[:,None]).all()):
            raise ValueError('child must lie in its original triangle plane and closed footprint')
        tri = np.stack((b1,b2),axis=2)
        ab,ac = tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
        det = ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
        usable = np.abs(det) > 1e-15
        # All three edge half-planes are expressed in original-parent coordinates.
        signed = []
        for a,b in ((0,1),(1,2),(2,0)):
            edge = tri[:,b]-tri[:,a]
            d = _PROBES[None,:,:]-tri[:,a,None,:]
            signed.append(edge[:,0,None]*d[:,:,1]-edge[:,1,None]*d[:,:,0])
        signed = np.stack(signed,axis=1)
        epsilon = np.maximum(np.abs(det),1e-15)[:,None,None]*1e-9
        inside = (signed >= -epsilon).all(axis=1) | (signed <= epsilon).all(axis=1)
        result[first+rows[usable]] = inside[usable]
    check()
    return result


def _pixel_witnesses(triangle, width, height, minimum, budget, check):
    """Count real integer centres, stopping after minimum; never substitute area."""
    lo = np.maximum(np.ceil(triangle.min(axis=0)).astype(int), [0,0])
    hi = np.minimum(np.floor(triangle.max(axis=0)).astype(int), [width-1,height-1])
    nx,ny = np.maximum(0,hi-lo+1)
    total = int(nx)*int(ny)
    witnesses = np.full((minimum,2),-1,np.int32)
    if total > budget:
        return 0, False, witnesses, 0, 'PIXEL_BUDGET'
    if not total:
        return 0, False, witnesses, 0, 'INSUFFICIENT_ACTUAL_PIXEL_CENTERS'
    ab,ac = triangle[1]-triangle[0],triangle[2]-triangle[0]
    orientation = ab[0]*ac[1]-ab[1]*ac[0]
    count = 0; examined = 0
    # A positive inward tolerance rejects borderline roundoff rather than adding
    # exterior centres to the certificate. Pixel cells are already disk-bounded.
    epsilon = 64*np.finfo(float).eps*max(1.,float(np.max(np.abs(triangle)))**2)
    for start in range(0,total,1024):
        check(); flat = np.arange(start,min(start+1024,total),dtype=np.int64)
        x = lo[0]+flat%nx; y = lo[1]+flat//nx
        inside = np.ones(len(flat),bool)
        for a,b in ((0,1),(1,2),(2,0)):
            edge = triangle[b]-triangle[a]
            values = edge[0]*(y-triangle[a,1])-edge[1]*(x-triangle[a,0])
            inside &= values >= epsilon if orientation > 0 else values <= -epsilon
        found = np.flatnonzero(inside); take = min(minimum-count,len(found))
        if take:
            witnesses[count:count+take] = np.column_stack((x[found[:take]],y[found[:take]]))
            count += take
        examined += len(flat)
        if count == minimum:
            return count, True, witnesses, examined, None
    return count, False, witnesses, examined, 'INSUFFICIENT_ACTUAL_PIXEL_CENTERS'


def evaluate_local_view(vertices, faces, *, K, R, C, mask, validity, depth,
                        gap_mask, view_id, original_vertices, original_faces,
                        parent_ids, eligible, policy=LocalPolicy(),
                        probe_membership=None, check=lambda:None, footprint_mode='disk'):
    """Independent per-child observations; aggregate negative votes separately.

    actual_pixel_count is a lower bound when pixel_count_is_lower_bound is true.
    The whole-footprint gap fraction is certified by the default disk proof or
    opt-in ``closed_cells`` proof, never estimated from those few witnesses.
    Closed cells conservatively include edge/vertex contact and require every
    touched square to be valid, gap-labelled and wholly beyond the same 3px
    foreground/unknown margin. Disk arrays remain diagnostics in either mode.
    No missing depth becomes negative evidence. View aggregation and protection
    vetoes remain the caller's responsibility in either mode.
    """
    if not isinstance(policy,LocalPolicy):
        raise ValueError('LocalPolicy required')
    if footprint_mode not in ('disk','closed_cells'):
        raise ValueError('footprint_mode must be disk or closed_cells')
    _view_id(view_id); check()
    v,f = _mesh(vertices,faces); ov,of = _mesh(original_vertices,original_faces)
    if len(f) > 400000:
        raise ValueError('bounded child mesh required')
    ids = _parents(parent_ids,len(f),len(of)); k,r,c = _camera(K,R,C)
    mask,valid,gap = np.asarray(mask),np.asarray(validity),np.asarray(gap_mask)
    if (mask.ndim != 2 or not all(mask.shape) or mask.size > 4_194_304
            or any(a.dtype != np.bool_ or a.shape != mask.shape for a in (mask,valid,gap))
            or np.any(mask & ~valid) or np.any(gap & (mask | ~valid))
            or (gap.any() and not mask.any())):
        raise ValueError('same-domain bool masks, valid foreground and background-only gap required')
    eligible = np.asarray(eligible)
    if eligible.dtype != np.bool_ or eligible.shape != (len(f),):
        raise ValueError('one bool negative-eligibility flag per child required')
    if depth is not None:
        depth = np.asarray(depth)
        if depth.shape != mask.shape or depth.dtype.kind not in 'fiu':
            raise ValueError('real depth in exact mask domain required')
    if probe_membership is None:
        membership = prepare_probe_membership(v,f,ov,of,ids,check=check)
    else:
        membership = np.asarray(probe_membership)
        if membership.dtype != np.bool_ or membership.shape != (len(f),4):
            raise ValueError('precomputed bool[N,4] probe membership required')
    n = len(f); h,w = mask.shape
    out = dict(view_id=view_id, status=np.zeros(n,np.uint8), negative=np.zeros(n,bool),
        reason=np.zeros(n,np.uint8), reason_names=REASONS.copy(), face_indices=np.arange(n,dtype=np.int64),
        parent_ids=ids.copy(), core_protected=np.zeros(n,bool), local_core_protected=np.zeros(n,bool),
        original_probe_protected=np.zeros(n,bool), original_supported_probe_count=np.zeros(n,np.uint8),
        local_supported_sample_count=np.zeros(n,np.uint8), actual_pixel_count=np.zeros(n,np.int16),
        pixel_count_is_lower_bound=np.zeros(n,bool), pixel_count_evaluated=np.zeros(n,bool),
        pixel_centers_examined=np.zeros(n,np.int32), pixel_witness_xy=np.full((n,policy.min_background_pixels,2),-1,np.int32),
        whole_gap_disk=np.zeros(n,bool), deep_background_disk=np.zeros(n,bool), known_domain_disk=np.zeros(n,bool),
        circle_envelope_pixels=np.zeros(n,float), circle_gap_clearance_pixels=np.zeros(n,float),
        circle_background_clearance_pixels=np.zeros(n,float), projected_area_pixels=np.zeros(n,float),
        certified_gap_fraction=np.zeros(n,float), geometry_accepted=False, source_mask_changed=False,
        status_scope='CONDITIONAL_LOCAL_EVIDENCE_NOT_ACCEPTED',
        core_scope='SAMPLED_ORIGINAL_PROBES_AND_LOCAL_VERTICES_CENTROID',
        physical_visibility='NOT_EVALUATED')
    if footprint_mode == 'closed_cells':
        out.update(footprint_mode=footprint_mode, footprint_cells=np.zeros(n,np.int32),
                   footprint_evaluated=np.zeros(n,bool), whole_gap_footprint=np.zeros(n,bool),
                   known_domain_footprint=np.zeros(n,bool), deep_background_footprint=np.zeros(n,bool),
                   footprint_vertex_prefilter_rejected=np.zeros(n,bool))
    out['reason'][~eligible] = _R['INELIGIBLE_PARENT']
    yy,xx = np.nonzero(mask)
    extent = int(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1)) if len(xx) else 0
    band = float(np.clip(extent*policy.band_extent_fraction,policy.minimum_band_px,policy.maximum_band_px))
    out['edge_band_pixels'] = band; out['core_margin_pixels'] = band*policy.core_band_multiplier
    if not n:
        return out
    cv = _cv()
    to_bg = (cv.distanceTransform((mask | ~valid).astype(np.uint8),cv.DIST_L2,cv.DIST_MASK_PRECISE)
             if np.any(~mask & valid) else np.full(mask.shape,np.inf))
    z,uv,pix,inframe = _project(v,k,r,c,mask.shape)
    if depth is not None:
        def supported(points, projection=None):
            zz,_,p,inside = projection if projection is not None else _project(points,k,r,c,mask.shape)
            x,y = p[:,0],p[:,1]; d = depth[y,x]
            return (inside & valid[y,x] & mask[y,x] & (to_bg[y,x] > out['core_margin_pixels'])
                    & np.isfinite(d) & (d > 0) & (np.abs(zz-d) <= policy.depth_relative_tolerance*d))
        local_v = supported(v,(z,uv,pix,inframe))
        local_c = supported(v[f].mean(axis=1))
        local_count = local_v[f].sum(axis=1,dtype=np.uint8)+local_c.astype(np.uint8)
        original_v = supported(ov)
        original_c = supported(ov[of].mean(axis=1))
        original_hits = np.column_stack((original_v[of[ids]],original_c[ids]))
        original_count = (original_hits & membership).sum(axis=1,dtype=np.uint8)
        out['local_supported_sample_count'] = local_count
        out['original_supported_probe_count'] = original_count
        out['local_core_protected'] = local_count > 0
        out['original_probe_protected'] = original_count > 0
        out['core_protected'] = (local_count > 0) | (original_count > 0)
    check()
    pending = eligible.copy()
    def reject(condition, reason):
        select = pending & condition
        out['reason'][select] = _R[reason]; pending[select] = False
    front = (z > 1e-12) & np.isfinite(uv).all(axis=1)
    reject(~front[f].all(axis=1),'CAMERA_PLANE')
    reject(~inframe[f].all(axis=1),'OUT_OF_FRAME')
    good = inframe & valid[pix[:,1],pix[:,0]]
    reject(~good[f].all(axis=1),'INVALID_VERTEX_DOMAIN')
    selected = np.flatnonzero(pending)
    if len(selected):
        tri = uv[f[selected]]
        ab,ac = tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
        area = .5*np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])
        out['projected_area_pixels'][selected] = area
        reject(~np.isfinite(out['projected_area_pixels']) | (out['projected_area_pixels'] < .5),'SUBPIXEL_OR_DEGENERATE')
    selected = np.flatnonzero(pending)
    if len(selected):
        deep = cv.distanceTransform(np.pad((~mask & valid).astype(np.uint8),1),cv.DIST_L2,cv.DIST_MASK_PRECISE)[1:-1,1:-1]
        gap_distance = cv.distanceTransform(np.pad(gap.astype(np.uint8),1),cv.DIST_L2,cv.DIST_MASK_PRECISE)[1:-1,1:-1]
        domain = cv.distanceTransform(np.pad(valid.astype(np.uint8),1),cv.DIST_L2,cv.DIST_MASK_PRECISE)[1:-1,1:-1]
        tri = uv[f[selected]]; center = tri.mean(axis=1)
        p = np.floor(center+.5).astype(np.int64); x,y = p[:,0],p[:,1]
        radius = np.linalg.norm(tri-center[:,None,:],axis=2).max(axis=1)
        envelope = radius+np.linalg.norm(center-p,axis=1)+math.sqrt(.5)
        out['circle_envelope_pixels'][selected] = envelope
        out['circle_gap_clearance_pixels'][selected] = gap_distance[y,x]-envelope
        out['circle_background_clearance_pixels'][selected] = deep[y,x]-envelope
        out['known_domain_disk'][selected] = domain[y,x] > envelope
        out['whole_gap_disk'][selected] = gap_distance[y,x] > envelope
        out['deep_background_disk'][selected] = deep[y,x] > policy.boundary_px+envelope
        if footprint_mode == 'disk':
            reject(~out['known_domain_disk'],'NO_KNOWN_DOMAIN_DISK')
            reject(~out['whole_gap_disk'],'NO_WHOLE_GAP_PROOF')
            reject(~out['deep_background_disk'],'NO_DEEP_BACKGROUND_PROOF')
        else:
            # Each nearest-centre vertex cell necessarily intersects the closed
            # triangle. Failure is therefore a necessary-condition rejection,
            # never a substitute for SAT: other cells touched on an edge or a
            # half-integer vertex boundary still require the complete test.
            vertex_cells = pix[f[selected]]
            vx,vy = vertex_cells[:,:,0],vertex_cells[:,:,1]
            vertex_known = valid[vy,vx].all(axis=1)
            vertex_gap = gap[vy,vx].all(axis=1)
            vertex_deep = (deep[vy,vx] > policy.boundary_px+math.sqrt(2)).all(axis=1)
            prefilter = vertex_known & vertex_gap & vertex_deep
            out['footprint_vertex_prefilter_rejected'][selected] = ~prefilter
            prefilter_reasons = np.where(~vertex_known, _R['NO_KNOWN_DOMAIN_FOOTPRINT'],
                np.where(~vertex_gap, _R['NO_WHOLE_GAP_PROOF'], _R['NO_DEEP_BACKGROUND_PROOF']))
            rejected = selected[~prefilter]
            out['reason'][rejected] = prefilter_reasons[~prefilter]
            pending[rejected] = False
            for fid in selected[prefilter]:
                cells,error = closed_footprint_cells(uv[f[fid]],w,h,
                    policy.max_raster_pixels_per_face,check)
                if error:
                    out['reason'][fid] = _R[error]; pending[fid] = False
                    continue
                x,y = cells.T
                out['footprint_evaluated'][fid] = True
                out['footprint_cells'][fid] = len(cells)
                out['known_domain_footprint'][fid] = valid[y,x].all()
                out['whole_gap_footprint'][fid] = gap[y,x].all()
                # Both the tested cell and a foreground/unknown cell have a
                # half-diagonal sqrt(.5). Their sum certifies the full closed
                # cell, rather than only its centre, at the original margin.
                out['deep_background_footprint'][fid] = (deep[y,x] > policy.boundary_px+math.sqrt(2)).all()
            reject(~out['known_domain_footprint'],'NO_KNOWN_DOMAIN_FOOTPRINT')
            reject(~out['whole_gap_footprint'],'NO_WHOLE_GAP_PROOF')
            reject(~out['deep_background_footprint'],'NO_DEEP_BACKGROUND_PROOF')
    for ordinal,fid in enumerate(np.flatnonzero(pending)):
        if ordinal % 128 == 0:
            check()
        count,lower,witnesses,examined,error = _pixel_witnesses(uv[f[fid]],w,h,
            policy.min_background_pixels,policy.max_raster_pixels_per_face,check)
        out['actual_pixel_count'][fid] = count
        out['pixel_count_is_lower_bound'][fid] = lower
        out['pixel_count_evaluated'][fid] = error != 'PIXEL_BUDGET'
        out['pixel_centers_examined'][fid] = examined
        out['pixel_witness_xy'][fid] = witnesses
        if error:
            out['reason'][fid] = _R[error]
            continue
        known = witnesses[:count]
        # Separate readback assertion prevents a numerical certificate from hiding
        # a contradictory actual pixel or incorrect incoming gap domain.
        if not (gap[known[:,1],known[:,0]] & valid[known[:,1],known[:,0]] & ~mask[known[:,1],known[:,0]]).all():
            raise ValueError('disk proof contradicts actual pixel witness')
        out['status'][fid] = CONFLICT; out['negative'][fid] = True
        out['reason'][fid] = _R['WHOLE_GAP_NEGATIVE']; out['certified_gap_fraction'][fid] = 1.
    check()
    return out
