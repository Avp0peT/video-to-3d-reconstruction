"""Indexed topology and centered volume audit; closure is not shape completeness."""
import numpy as np


def _validate(vertices,faces):
    if not isinstance(vertices,np.ndarray) or vertices.dtype.kind!='f' or vertices.dtype.itemsize>8 or vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():raise ValueError('finite floating Nx3 vertices required')
    if not isinstance(faces,np.ndarray) or faces.dtype.kind not in 'iu' or faces.ndim!=2 or faces.shape[1]!=3:raise ValueError('integer Fx3 faces required')
    if len(vertices)>200000 or len(faces)>400000:raise ValueError('200000 vertex /400000 face audit budget exceeded')
    if faces.size and ((faces<0).any() or (faces>=len(vertices)).any()):raise ValueError('invalid face indices')
    return vertices.astype(np.float64,copy=False),faces.astype(np.int64,copy=False)


def audit_closed_mesh(vertices,faces):
    """Read-only exact-index audit, with no welding or self-intersection test.

    Orientability here means the supplied face winding already consistently
    orients every manifold shared edge; inconsistent winding does NOT prove
    that the underlying surface is topologically nonorientable.
    """
    v,f=_validate(vertices,faces)
    vertex_component=np.full(len(v),-1,np.int64)
    if len(f):
        used=np.unique(f);edges=f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
        sorted_edges=np.sort(edges,axis=1)
        unique_edges,inverse,edge_counts=np.unique(sorted_edges,axis=0,return_inverse=True,return_counts=True)
        direction=np.where(edges[:,0]<edges[:,1],1.,-1.)
        orientation_sum=np.bincount(inverse,weights=direction,minlength=len(unique_edges))
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        graph=coo_matrix((np.ones(len(unique_edges),np.uint8),(unique_edges[:,0],unique_edges[:,1])),shape=(len(v),len(v))).tocsr()
        _,labels=connected_components(graph,directed=False)
        _,component=np.unique(labels[used],return_inverse=True);vertex_component[used]=component
        face_component=vertex_component[f[:,0]];edge_component=vertex_component[unique_edges[:,0]]
        with np.errstate(over='ignore',invalid='ignore'):
            cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
        if not np.isfinite(cross).all():raise ValueError('triangle area computation overflow')
        degenerate=(cross==0).all(axis=1)
        _,face_group,face_group_counts=np.unique(np.sort(f,axis=1),axis=0,return_inverse=True,return_counts=True)
        duplicate=face_group_counts[face_group]>1
        duplicate_excess=len(f)-len(face_group_counts)
        ncomponents=int(component.max())+1
    else:
        used=np.empty(0,np.int64);unique_edges=np.empty((0,2),np.int64);edge_counts=np.empty(0,np.int64);orientation_sum=np.empty(0)
        face_component=np.empty(0,np.int64);edge_component=np.empty(0,np.int64);degenerate=np.empty(0,bool);duplicate=np.empty(0,bool);duplicate_excess=0;ncomponents=0
    components=[]
    for index in range(ncomponents):
        vertex_ids=np.flatnonzero(vertex_component==index);face_ids=np.flatnonzero(face_component==index);edge_ids=np.flatnonzero(edge_component==index)
        coords=v[vertex_ids];lo=coords.min(0);hi=coords.max(0);center=lo+(hi-lo)/2
        tri=v[f[face_ids]]-center
        with np.errstate(over='ignore',invalid='ignore'):
            terms=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))/6
            volume=float(terms.sum(dtype=np.float64))
            tolerance=float(64*np.finfo(np.float64).eps*float((hi-lo).max())**3*max(1,len(face_ids)))
        if not np.isfinite(volume) or not np.isfinite(tolerance):raise ValueError('signed volume computation overflow')
        counts=edge_counts[edge_ids]
        closed=bool(len(face_ids) and (counts==2).all() and not degenerate[face_ids].any() and not duplicate[face_ids].any())
        inconsistent=(counts==2)&(orientation_sum[edge_ids]!=0)
        consistent=bool(not inconsistent.any() and not (counts>2).any())
        nonzero=abs(volume)>tolerance
        components.append(dict(component_id=index,vertices=len(vertex_ids),faces=len(face_ids),edges=len(edge_ids),
            boundary_edges=int((counts==1).sum()),two_face_edges=int((counts==2).sum()),nonmanifold_edges=int((counts>2).sum()),
            degenerate_faces=int(degenerate[face_ids].sum()),duplicate_group_faces=int(duplicate[face_ids].sum()),
            inconsistent_two_face_edges=int(inconsistent.sum()),edge_orientation_consistent=consistent,
            closed_indexed_surface=closed,orientable_with_current_winding=closed and consistent,
            signed_volume=volume,volume_center=center.tolist(),volume_numerical_tolerance=tolerance,volume_nonzero=bool(nonzero),
            volume_sign='POSITIVE' if volume>tolerance else ('NEGATIVE' if volume < -tolerance else 'NUMERICALLY_ZERO'),
            closed_oriented_nonzero_volume=bool(closed and consistent and nonzero)))
    all_closed=bool(components and all(c['closed_indexed_surface'] for c in components))
    all_consistent=bool(components and all(c['edge_orientation_consistent'] for c in components))
    all_valid=bool(components and all(c['closed_oriented_nonzero_volume'] for c in components))
    return dict(vertex_component=vertex_component,face_component=face_component,unique_edges=unique_edges,edge_face_count=edge_counts,
        edge_direction_sum=orientation_sum,face_degenerate=degenerate,face_duplicate_group=duplicate,components=components,
        counts=dict(vertices=len(v),faces=len(f),connected_components=ncomponents,isolated_vertices=len(v)-len(used),
            boundary_edges=int((edge_counts==1).sum()),two_face_edges=int((edge_counts==2).sum()),nonmanifold_edges=int((edge_counts>2).sum()),
            degenerate_faces=int(degenerate.sum()),duplicate_faces=int(duplicate_excess),duplicate_group_faces=int(duplicate.sum()),
            inconsistent_two_face_edges=int(((edge_counts==2)&(orientation_sum!=0)).sum())),
        closed=all_closed,edge_orientation_consistent=all_consistent,closed_oriented_nonzero_volume=all_valid,
        self_intersections='NOT_EVALUATED',complete_object='NOT_ESTABLISHED',geometry_accepted=False,mesh_modified=False,
        semantics='Connectivity uses exact shared vertex indices only. Isolated vertices are reported separately, not surface components. Supplied winding consistency is not a general topological orientability solver. Centered signed volumes are algebraic, not physical solids when self-intersections exist. Closed/oriented/nonzero volume does not establish completeness, no self-intersections, accurate shape, scale or CAD acceptance.')


def orient_components(vertices,faces):
    """Flip negative, already closed and consistently wound components only."""
    before=audit_closed_mesh(vertices,faces);result=faces.copy();flipped=[]
    for component in before['components']:
        if component['closed_oriented_nonzero_volume'] and component['volume_sign']=='NEGATIVE':
            index=component['component_id'];rows=before['face_component']==index
            result[rows]=result[rows][:,[0,2,1]];flipped.append(index)
    after=audit_closed_mesh(vertices,result)
    return dict(vertices=vertices.copy(),faces=result,flipped_component_ids=np.asarray(flipped,np.int64),before=before,after=after,
        vertices_modified=False,faces_reoriented=bool(flipped),geometry_accepted=False,
        semantics='Only whole-component winding reversal; no coordinates, indices, topology, holes or self-intersections repaired. Before evidence retained.')


def split_by_face_labels(vertices,faces,labels):
    """Disjoint face compaction, with exact original face/vertex identity ledgers."""
    v,f=_validate(vertices,faces)
    if not isinstance(labels,np.ndarray) or labels.ndim!=1 or labels.shape!=(len(f),) or labels.dtype.kind not in 'biuUS':raise ValueError('one bool/integer/string label per face required')
    unique=np.unique(labels);layers=[];face_layer=np.full(len(f),-1,np.int64);covered=[]
    for index,label in enumerate(unique):
        rows=np.flatnonzero(labels==label);used,inverse=np.unique(f[rows],return_inverse=True)
        compact=inverse.reshape(-1,3).astype(np.int64)
        if not np.array_equal(v[used][compact],v[f[rows]]):raise AssertionError('layer geometry differs from original')
        layers.append(dict(label=label.item(),vertices=vertices[used].copy(),faces=compact,original_vertex_ids=used,original_face_ids=rows))
        face_layer[rows]=index;covered.append(rows)
    merged=np.concatenate(covered) if covered else np.empty(0,np.int64)
    exact=np.array_equal(np.sort(merged),np.arange(len(f)))
    if not exact:raise AssertionError('layer face identity partition incomplete or duplicated')
    return dict(layers=layers,face_layer=face_layer,face_identity_coverage_exact=exact,
        original_face_count=len(f),partition_face_count=len(merged),geometry_accepted=False,
        semantics='Every original face belongs to exactly one compact layer. Boundary vertices may be copied between layers; no shared-index weld, new face, face deletion or averaging.')
