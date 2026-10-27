"""Open local depth-grid surfaces, with conservative full-footprint mask checks."""
from collections import Counter
import numpy as np
from integrations.mvs_surface.local_depth_search import _view


def _known_foreground_bbox(triangle,shape,integral,max_cells=4096):
    """Sufficient support test only; every failure falls back to original SAT."""
    h,w=shape
    if not np.isfinite(triangle).all(): return False
    a,b=triangle[1]-triangle[0],triangle[2]-triangle[0]
    if abs(float(a[0]*b[1]-a[1]*b[0]))<1e-9: return False
    complete=bool((triangle[:,0]>=-1e-8).all() and (triangle[:,0]<=w-1+1e-8).all() and (triangle[:,1]>=-1e-8).all() and (triangle[:,1]<=h-1+1e-8).all())
    if not complete: return False
    lo=np.ceil(triangle.min(axis=0)-.5-1e-8).astype(int); hi=np.floor(triangle.max(axis=0)+.5+1e-8).astype(int)
    lo=np.maximum(lo,[0,0]); hi=np.minimum(hi,[w-1,h-1])
    if (hi<lo).any(): return False
    cells=int(np.prod(hi-lo+1))
    if cells>max_cells: return False
    x0,y0=lo; x1,y1=hi+1
    known=int(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])
    return known==cells


def _grid_templates():
    yy,xx=np.mgrid[0:4,0:4]
    q=np.column_stack([xx.ravel(),yy.ravel()])
    result=[]
    for pixels in (np.array([[0,0],[3,0],[3,3]]),np.array([[0,0],[3,3],[0,3]])):
        bary=np.column_stack([q,np.ones(len(q))])@np.linalg.inv(np.vstack([pixels.T,np.ones(3)])).T
        interior=(bary>=-1e-9).all(axis=1)
        result.append((pixels,q[interior],bary[interior]))
    return result


def _covered_pixels(triangle, shape, max_cells=4096):
    """All native pixel squares touched by the projected triangle (SAT test).

    Returns None when too large/degenerate, otherwise in-frame cells and a flag
    indicating the entire triangle lies within the native pixel-centre domain.
    """
    h,w=shape
    if not np.isfinite(triangle).all(): return None
    a,b=triangle[1]-triangle[0],triangle[2]-triangle[0]
    area=a[0]*b[1]-a[1]*b[0]
    if abs(float(area))<1e-9: return None
    lo=np.ceil(triangle.min(axis=0)-.5-1e-8).astype(int); hi=np.floor(triangle.max(axis=0)+.5+1e-8).astype(int)
    lo=np.maximum(lo,[0,0]); hi=np.minimum(hi,[w-1,h-1])
    complete=bool((triangle[:,0]>=-1e-8).all() and (triangle[:,0]<=w-1+1e-8).all() and (triangle[:,1]>=-1e-8).all() and (triangle[:,1]<=h-1+1e-8).all())
    if (hi<lo).any(): return np.empty((0,2),int),False
    if int(np.prod(hi-lo+1))>max_cells: return None
    yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
    centers=np.column_stack([xx.ravel(),yy.ravel()]); touched=np.ones(len(centers),bool)
    for i in range(3):
        edge=triangle[(i+1)%3]-triangle[i]; axis=np.array([-edge[1],edge[0]])
        projected=triangle@axis; center=centers@axis; radius=.5*np.abs(axis).sum()
        touched &= (center+radius>=projected.min()-1e-9)&(center-radius<=projected.max()+1e-9)
    return centers[touched],complete


def _project_world(view,xyz):
    cam=(xyz-view['C'])@view['R'].T
    if (cam[:,2]<=1e-9).any(): return None
    uv=cam@view['K'].T
    return uv[:,:2]/cam[:,2,None]


def _dispersed(sources,center):
    """Require support from two distinct directions separated by >=1 degree."""
    for i,a in enumerate(sources):
        da=a['C']-center; na=np.linalg.norm(da)
        if na<=1e-12: continue
        for b in sources[i+1:]:
            db=b['C']-center; nb=np.linalg.norm(db)
            if nb<=1e-12 or np.linalg.norm(a['C']-b['C'])<=1e-9: continue
            if np.dot(da,db)/(na*nb)<=np.cos(np.deg2rad(1.)): return True
    return False


def build_depth_surface(views,check=lambda:None):
    """Faces interpolate existing native depths only; holes remain unfilled.

    Every local 3x3 cell is checked over its full pixel rectangle. Other-camera
    checks cover every pixel square touched by each projected triangle, including
    all edges/interior. A known-background footprint vetoes the face even where
    other cameras support it. This deliberately risks rejecting real occluded
    geometry; silhouettes do not establish depth visibility.
    """
    if not isinstance(views,list) or not 3<=len(views)<=16: raise ValueError('three to sixteen views required')
    checked=[]
    for view in views:
        check(); v=_view(view); valid=v.get('validity')
        if not isinstance(valid,np.ndarray) or valid.dtype!=bool or valid.shape!=v['mask'].shape: raise ValueError('explicit bool validity required')
        checked.append(v)
    if len({v['view_id'] for v in checked})!=len(checked): raise ValueError('duplicate view ID')
    vertices=[]; colors=[]; faces=[]; provenance=[]; vertex_sources=[]; support_ids=[]
    counts=Counter(); per_view={}; capped=False
    templates=_grid_templates()
    foreground_integrals={}
    for view in checked:
        check()
        foreground=(view['mask'] & view['validity']).astype(np.int64)
        foreground_integrals[view['view_id']]=np.pad(foreground.cumsum(axis=0).cumsum(axis=1),((1,0),(1,0)))
    for ref in checked:
        check(); local=Counter(); vertex_index={}; h,w=ref['mask'].shape
        inverse_k=np.linalg.inv(ref['K'])
        allowed=ref['mask'] & ref['validity'] & (ref['depth']>0)
        others=[v for v in checked if v['view_id']!=ref['view_id']]
        for y in range(0,h-3,3):
            if capped: break
            check()
            for x in range(0,w-3,3):
                if capped: break
                local['candidate_cells']+=1
                square=allowed[y:y+4,x:x+4]
                if not square.all(): local['ref_missing_or_background_cells']+=1; continue
                square_depth=ref['depth'][y:y+4,x:x+4]
                if square_depth.max()/square_depth.min()-1>.02: local['ref_depth_jump_cells']+=1; continue
                for pixel_offsets,interior_offsets,bary in templates:
                    check(); local['candidate_faces']+=1
                    pixels=pixel_offsets+[x,y]; z=ref['depth'][pixels[:,1],pixels[:,0]].astype(np.float64)
                    rays=np.column_stack([pixels,np.ones(3)])@inverse_k.T; camera=rays*z[:,None]; xyz=camera@ref['R']+ref['C']
                    # Full reference pixel-domain depth versus perspective-correct planar interpolation.
                    q=interior_offsets+[x,y]
                    interpolated=1/(bary@(1/z))
                    actual=ref['depth'][q[:,1],q[:,0]]
                    if (np.abs(interpolated-actual)/actual>.02).any(): local['ref_interpolation_rejected']+=1; continue
                    edge_ok=True
                    for a,b in ((0,1),(1,2),(2,0)):
                        footprint=np.mean(z)*np.linalg.norm(rays[a]-rays[b])
                        if footprint<=0 or np.linalg.norm(camera[a]-camera[b])>2*footprint: edge_ok=False
                    if not edge_ok: local['edge_footprint_rejected']+=1; continue
                    supporters=[]; background=False
                    for source_index,source in enumerate(others):
                        check(); local['source_checks_processed']+=1; uv=_project_world(source,xyz)
                        if uv is not None and _known_foreground_bbox(uv,source['mask'].shape,foreground_integrals[source['view_id']]):
                            supporters.append(source); local['source_bbox_fast_accepts']+=1
                            continue
                        covered=None if uv is None else _covered_pixels(uv,source['mask'].shape)
                        if covered is None: local['source_unknown_checks']+=1; continue
                        cells,complete=covered
                        sx,sy=cells[:,0],cells[:,1]
                        known=source['validity'][sy,sx]
                        if (known & ~source['mask'][sy,sx]).any():
                            background=True; local['source_background_checks']+=1
                            local['source_checks_skipped_after_veto']+=len(others)-source_index-1
                            break
                        if complete and len(cells) and (known & source['mask'][sy,sx]).all(): supporters.append(source)
                        else: local['source_unknown_checks']+=1
                    if background: local['background_veto_faces']+=1; continue
                    if len(supporters)<2: local['insufficient_support_faces']+=1; continue
                    if not _dispersed(supporters,xyz.mean(axis=0)): local['insufficient_view_spread_faces']+=1; continue
                    needed=sum(tuple(p) not in vertex_index for p in pixels)
                    if len(vertices)+needed>200000 or len(faces)>=300000:
                        capped=True; local['capacity_stops']+=1; break
                    face=[]
                    for p,world in zip(pixels,xyz):
                        key=tuple(p)
                        if key not in vertex_index:
                            vertex_index[key]=len(vertices); vertices.append(world); colors.append(ref['image'][p[1],p[0]].copy()); vertex_sources.append(ref['view_id'])
                        face.append(vertex_index[key])
                    faces.append(face); provenance.append(ref['view_id']); support_ids.append([s['view_id'] for s in supporters]); local['retained_faces']+=1
        per_view[str(ref['view_id'])]=dict(local); counts.update(local)
        if capped: break
    world=np.asarray(vertices,np.float64).reshape(-1,3); triangles=np.asarray(faces,np.int32).reshape(-1,3)
    if not np.isfinite(world).all(): raise ValueError('nonfinite output vertex')
    # Separate view layers retain separate indices even at identical world positions.
    return dict(worldvertices=world,colors=np.asarray(colors,np.uint8).reshape(-1,3),faces=triangles,
                sourceview=np.asarray(provenance,np.int32),vertex_sourceview=np.asarray(vertex_sources,np.int32),
                face_support_view_ids=support_ids,counts=dict(counts,vertices=len(world),faces=len(triangles)),per_view=per_view,
                status='OBSERVED_DEPTH_LOCAL_SURFACE_NOT_GEOMETRY_ACCEPTED' if len(triangles) else 'NO_EVIDENCE_SUPPORTED_FACES',
                capacity_limited=capped,geometry_accepted=False,watertight=False,
                source_check_count_semantics='Processed source checks stop at first known-background veto. Skipped-after-veto checks are not classified as unknown; counters are not an exhaustive all-camera diagnostic on rejected faces.',
                policy=dict(grid_step=3,max_relative_depth_difference=.02,max_edge_pixel_footprint_factor=2.,minimum_other_views=2,minimum_support_angle_degrees=1.,max_projected_footprint_cells=4096,max_vertices=200000,max_faces=300000),
                semantics='Triangles linearly interpolate original depth-grid vertices; no silhouette-endpoint faces, no welding across views, no gap filling. Full reference cell must have valid foreground depth. Projected full pixel-square footprints gate edges and interiors. Silhouette support is not depth visibility; known-background veto can reject true geometry under occlusion, mask errors, or view-dependent appearance. Open layered surfaces may overlap or self-intersect; topology and physical scale are not accepted.')
