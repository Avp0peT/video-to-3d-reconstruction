"""Select a dominant already-closed envelope, retaining discarded evidence."""
import numpy as np
from .completion_mesh_audit import audit_closed_mesh


def _metadata(audit):return {key:value for key,value in audit.items() if not isinstance(value,np.ndarray)}


def select_dominant_envelope(vertices,faces,bounds,spacing,check=lambda:None):
    """Single-object proposal selection; no welding, filling or coordinate edits.

    The largest indexed face component must contain >=90 percent of ALL input
    faces, already pass the closed/oriented/nonzero-volume audit and be at least
    one grid spacing from each AABB plane. All other geometry is retained as a
    separate contamination/diagnostic layer, never relabelled physically false.
    """
    check();box=np.asarray(bounds);step=np.asarray(spacing)
    if box.shape!=(2,3) or box.dtype.kind not in 'fiu' or not np.isfinite(box).all() or not (box[1]>box[0]).all():raise ValueError('finite positive world AABB [2,3] required')
    if step.dtype.kind not in 'fiu' or step.shape not in ((),(3,)) or not np.isfinite(step).all() or not (step>0).all():raise ValueError('finite positive scalar or XYZ spacing required')
    step=np.broadcast_to(step,(3,)).astype(np.float64);box=box.astype(np.float64)
    audit=audit_closed_mesh(vertices,faces);check()
    if not audit['components']:raise ValueError('no surface component available')
    component=max(audit['components'],key=lambda row:row['faces'])
    fraction=component['faces']/len(faces)
    if fraction<.9:raise ValueError('largest component has less than90percent of all faces')
    if not component['closed_oriented_nonzero_volume']:raise ValueError('dominant component is not closed, consistently oriented and nonzero volume')
    component_id=component['component_id'];face_ids=np.flatnonzero(audit['face_component']==component_id)
    vertex_ids,inverse=np.unique(faces[face_ids],return_inverse=True)
    selected_vertices=vertices[vertex_ids].copy();selected_faces=inverse.reshape(-1,3).astype(np.int64)
    lower=selected_vertices.min(0)-box[0];upper=box[1]-selected_vertices.max(0)
    # Permit only machine-scale arithmetic tolerance, never a fraction of a cell.
    tolerance=16*np.finfo(np.float64).eps*np.maximum(1,np.maximum(np.abs(box).max(0),step))
    if (lower+tolerance<step).any() or (upper+tolerance<step).any():raise ValueError('dominant envelope touches the one-cell AABB margin')
    discarded_face_ids=np.flatnonzero(audit['face_component']!=component_id)
    retained_vertex_mask=np.zeros(len(vertices),bool);retained_vertex_mask[vertex_ids]=True
    discarded_vertex_ids=np.flatnonzero(~retained_vertex_mask)
    discarded_vertices=vertices[discarded_vertex_ids].copy()
    remap=np.full(len(vertices),-1,np.int64);remap[discarded_vertex_ids]=np.arange(len(discarded_vertex_ids))
    discarded_faces=remap[faces[discarded_face_ids]]
    if (discarded_faces<0).any():raise AssertionError('indexed components unexpectedly share retained vertices')
    if not np.array_equal(selected_vertices[selected_faces],vertices[faces[face_ids]]):raise AssertionError('selected geometry changed')
    if not np.array_equal(np.sort(np.concatenate((face_ids,discarded_face_ids))),np.arange(len(faces))):raise AssertionError('face partition incomplete')
    selected_audit=audit_closed_mesh(selected_vertices,selected_faces);check()
    return dict(vertices=selected_vertices,faces=selected_faces,original_face_ids=face_ids,original_vertex_ids=vertex_ids,
        discarded_face_ids=discarded_face_ids,discarded_vertex_ids=discarded_vertex_ids,
        discarded_vertices=discarded_vertices,discarded_faces=discarded_faces,
        original_vertex_component=audit['vertex_component'],original_face_component=audit['face_component'],
        counts=dict(original_vertices=len(vertices),original_faces=len(faces),original_components=len(audit['components']),
            dominant_component_id=component_id,retained_vertices=len(selected_vertices),retained_faces=len(selected_faces),retained_face_fraction=fraction,
            discarded_vertices=len(discarded_vertices),discarded_faces=len(discarded_faces),
            isolated_input_vertices=audit['counts']['isolated_vertices']),
        audit=_metadata(selected_audit),original_audit=_metadata(audit),
        contamination_layer=dict(label='UNSELECTED_COMPONENTS_NOT_PHYSICALLY_CLASSIFIED',vertices_array='discarded_vertices',faces_array='discarded_faces',
            original_vertex_ids_array='discarded_vertex_ids',original_face_ids_array='discarded_face_ids',retained_separately=True,
            semantics='Other components and isolated vertices remain available as diagnostic geometry. They are not proven erroneous or deleted from the source.'),
        policy=dict(minimum_face_fraction=.9,selection='Largest face-count exact-index component only',single_object_assumption=True,
            world_aabb=box.tolist(),spacing=step.tolist(),minimum_boundary_margin_cells=1,
            actual_lower_margin=lower.tolist(),actual_upper_margin=upper.tolist(),welding=False,filling=False,coordinates_changed=False),
        status='DOMINANT_CLOSED_INFERRED_ENVELOPE_NOT_GEOMETRY_ACCEPTED',geometry_accepted=False,complete_object='NOT_ESTABLISHED',
        semantics='Single-object inferred-envelope proposal. Indexed closure and dominant face count do not establish physical completeness, no self-intersections, accurate fingers/concavities, scale or CAD validity. Full original and unselected layers remain diagnostic evidence.')
