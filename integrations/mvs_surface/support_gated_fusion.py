"""Pure reversible support selection and exact-coordinate pooling; no IO."""
import numpy as np


def _points(points, colors, supported, name):
    if not isinstance(points,np.ndarray) or points.ndim!=2 or points.shape[1]!=3 or points.dtype.kind!='f' or points.dtype.itemsize>8 or not np.isfinite(points).all():
        raise ValueError(name+' points must be finite floating Nx3')
    if not isinstance(colors,np.ndarray) or colors.dtype!=np.uint8 or colors.shape!=points.shape:
        raise ValueError(name+' colors must be uint8 Nx3')
    if not isinstance(supported,np.ndarray) or supported.dtype!=bool or supported.shape!=(len(points),):
        raise ValueError(name+' support must be bool N')
    return points.astype(np.float64,copy=False)


def assemble_supported(global_points, global_colors, global_supported,
                       local_vertices, local_colors, local_supported,
                       local_faces, local_face_supported, local_vertex_source,
                       check=lambda:None):
    """Pool supported global/local vertices and retain existing admissible faces.

    Support labels are supplied by the caller, NOT verified here. Only exact
    numeric XYZ equality shares an index (including signed zero equality). The
    first occurrence supplies display color; every origin retains its own color
    and source identity in CSR arrays. There is no inferred-shell input.
    """
    check()
    gp=_points(global_points,global_colors,global_supported,'global')
    lp=_points(local_vertices,local_colors,local_supported,'local')
    if not isinstance(local_faces,np.ndarray) or local_faces.dtype.kind not in 'iu' or local_faces.ndim!=2 or local_faces.shape[1]!=3:
        raise ValueError('local faces must be integer Fx3')
    if local_faces.size and ((local_faces<0).any() or (local_faces>=len(lp)).any()):
        raise ValueError('local face index outside vertices')
    if not isinstance(local_face_supported,np.ndarray) or local_face_supported.dtype!=bool or local_face_supported.shape!=(len(local_faces),):
        raise ValueError('local face support must be bool F')
    if not isinstance(local_vertex_source,np.ndarray) or local_vertex_source.dtype.kind not in 'iu' or local_vertex_source.shape!=(len(lp),) or (local_vertex_source<0).any():
        raise ValueError('local vertex source must be nonnegative integer M')
    if local_vertex_source.size and (local_vertex_source>np.iinfo(np.int64).max).any():
        raise ValueError('source ID exceeds int64')
    vertices=[];colors=[];origins=[];keys={}
    gmap=np.full(len(gp),-1,np.int64);lmap=np.full(len(lp),-1,np.int64)
    global_keys=set();local_new_unique=0;local_duplicate_global=0;local_duplicate_local=0
    for kind,(points,rgb,support,mapping) in enumerate(((gp,global_colors,global_supported,gmap),(lp,local_colors,local_supported,lmap))):
        for position,index in enumerate(np.flatnonzero(support)):
            if position%2048==0:check()
            key=tuple(points[index]);out=keys.get(key)
            if out is None:
                if len(vertices)>=200000:raise ValueError('200000 vertex budget exceeded; no truncation')
                out=len(vertices);keys[key]=out;vertices.append(points[index].copy());colors.append(rgb[index].copy());origins.append([])
                if kind==1:local_new_unique+=1
            elif kind==1:
                if key in global_keys:local_duplicate_global+=1
                else:local_duplicate_local+=1
            if kind==0:global_keys.add(key)
            mapping[index]=out
            origins[out].append((kind,int(index),-1 if kind==0 else int(local_vertex_source[index]),rgb[index].copy()))
    world=np.asarray(vertices,np.float64).reshape(-1,3)
    faces=[];face_original=[];face_sources=[]
    face_output=np.full(len(local_faces),-1,np.int64)
    face_reason=np.zeros(len(local_faces),np.uint8)
    # 1 unsupported face; 2 unsupported vertex; 3 collapsed; 4 zero area; 5 kept.
    for index,face in enumerate(local_faces):
        if index%2048==0:check()
        if not local_face_supported[index]:face_reason[index]=1;continue
        mapped=lmap[face]
        if (mapped<0).any():face_reason[index]=2;continue
        if len(set(map(int,mapped)))!=3:face_reason[index]=3;continue
        triangle=world[mapped]
        with np.errstate(over='ignore',invalid='ignore'):
            cross=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
        if not np.isfinite(cross).all():raise ValueError('triangle area computation overflow')
        if (cross==0).all():face_reason[index]=4;continue
        if len(faces)>=400000:raise ValueError('400000 face budget exceeded; no truncation')
        face_output[index]=len(faces);face_reason[index]=5
        faces.append(mapped.copy());face_original.append(index);face_sources.append(local_vertex_source[face].astype(np.int64))
    triangles=np.asarray(faces,np.int64).reshape(-1,3)
    face_sources=np.asarray(face_sources,np.int64).reshape(-1,3)
    ptr=np.zeros(len(world)+1,np.int64)
    kinds=[];indices=[];sources=[];origin_colors=[]
    global_member=np.zeros(len(world),bool);local_member=global_member.copy()
    for out,items in enumerate(origins):
        if out%2048==0:check()
        for kind,index,source,color in items:
            kinds.append(kind);indices.append(index);sources.append(source);origin_colors.append(color)
            if kind==0:global_member[out]=True
            else:local_member[out]=True
        ptr[out+1]=len(kinds)
    if len(triangles):
        edges=np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1)
        _,edge_count=np.unique(edges,axis=0,return_counts=True)
        canonical=np.sort(triangles,axis=1)
        duplicate_faces=len(canonical)-len(np.unique(canonical,axis=0))
        used=np.unique(triangles)
    else:edge_count=np.empty(0,np.int64);duplicate_faces=0;used=np.empty(0,np.int64)
    check()
    return dict(vertices=world,colors=np.asarray(colors,np.uint8).reshape(-1,3),faces=triangles,
        global_to_output=gmap,local_to_output=lmap,local_face_to_output=face_output,
        output_global_member=global_member,output_local_member=local_member,
        origin_offsets=ptr,origin_kind=np.asarray(kinds,np.uint8),origin_index=np.asarray(indices,np.int64),
        origin_source_id=np.asarray(sources,np.int64),origin_colors=np.asarray(origin_colors,np.uint8).reshape(-1,3),
        face_original_ids=np.asarray(face_original,np.int64),face_vertex_source_ids=face_sources,
        face_source_id=np.array([int(s[0]) if (s==s[0]).all() else -1 for s in face_sources],np.int64),
        face_reason=face_reason,face_reason_legend={0:'NOT_EVALUATED',1:'FACE_UNSUPPORTED',2:'VERTEX_UNSUPPORTED',3:'COLLAPSED_INDEX',4:'ZERO_AREA',5:'RETAINED_ORIGINAL_FACE'},
        counts=dict(input_global_points=len(gp),supported_global_origins=int(global_supported.sum()),
            input_local_vertices=len(lp),supported_local_origins=int(local_supported.sum()),
            retained_unique_global_points=len(global_keys),local_new_unique_vertices=local_new_unique,
            local_origins_exact_duplicate_global=local_duplicate_global,local_origins_exact_duplicate_local=local_duplicate_local,
            global_origins_exact_duplicate_global=int(global_supported.sum())-len(global_keys),
            output_vertices=len(world),retained_faces=len(triangles),original_faces=len(local_faces),
            excluded_global_origins=int((~global_supported).sum()),excluded_local_origins=int((~local_supported).sum()),
            rejected_unsupported_faces=int((face_reason==1).sum()),rejected_vertex_unsupported_faces=int((face_reason==2).sum()),
            rejected_collapsed_faces=int((face_reason==3).sum()),rejected_zero_area_faces=int((face_reason==4).sum()),
            isolated_output_vertices=len(world)-len(used),retained_original_faces=len(triangles)),
        topology=dict(boundary_edges=int((edge_count==1).sum()),nonmanifold_edges=int((edge_count>2).sum()),
            exact_duplicate_faces=int(duplicate_faces),self_intersections='NOT_EVALUATED',accepted=False),
        policy=dict(exact_coordinates_only=True,averaging=False,snapping=False,retriangulation=False,
            max_vertices=200000,max_faces=400000,display_color='First supported origin; originals preserved in origin_colors',
            origin_kind_legend={0:'GLOBAL',1:'LOCAL'},global_source_id=-1),
        status='SUPPORTED_HYBRID_MODEL_NOT_GEOMETRY_ACCEPTED',geometry_accepted=False,complete=False,
        semantics='Caller-supplied support selection, not independent truth. Exact-coordinate pooling preserves every supported origin; close points and opposing nearby layers remain separate. No inferred silhouettes, gap filling or new triangles. New unique coordinates do not prove increased physical coverage. Exact sharing can create nonmanifold or duplicate faces; diagnostics are not topology acceptance.')


def assemble_with_policy(global_points, global_colors, global_supported,
                         local_vertices, local_colors, local_supported,
                         local_faces, local_face_supported, local_vertex_source,
                         *, local_policy='supported-only', global_policy='supported-only',
                         face_policy='supported-only', check=lambda:None):
    """Separate point admission from evidence, without loosening face evidence.

    Point policies control admission only, never evidence labels. Original-face
    admission requires all local points and restores existing faces only, still
    subject to index, collapse, zero-area and capacity checks in the old kernel.
    Conflict versus unverified distinctions remain the caller's audit ledger.
    """
    if type(local_policy) is not str or local_policy not in ('supported-only','all-points'):
        raise ValueError('local_policy must be supported-only or all-points')
    if type(global_policy) is not str or global_policy not in ('supported-only','all-points'):
        raise ValueError('global_policy must be supported-only or all-points')
    if type(face_policy) is not str or face_policy not in ('supported-only','original'):
        raise ValueError('face_policy must be supported-only or original')
    if face_policy=='original' and local_policy!='all-points':
        raise ValueError('original faces require local_policy all-points')
    check()
    if not isinstance(local_supported,np.ndarray) or local_supported.dtype!=bool or local_supported.shape!=(len(local_vertices),):
        raise ValueError('local support must be bool M')
    # Validate before indexing for the additional face evidence guard.
    if not isinstance(local_faces,np.ndarray) or local_faces.dtype.kind not in 'iu' or local_faces.ndim!=2 or local_faces.shape[1]!=3:
        raise ValueError('local faces must be integer Fx3')
    if local_faces.size and ((local_faces<0).any() or (local_faces>=len(local_vertices)).any()):
        raise ValueError('local face index outside vertices')
    if not isinstance(local_face_supported,np.ndarray) or local_face_supported.dtype!=bool or local_face_supported.shape!=(len(local_faces),):
        raise ValueError('local face support must be bool F')
    admitted=local_supported.copy() if local_policy=='supported-only' else np.ones(len(local_vertices),bool)
    if not isinstance(global_supported,np.ndarray) or global_supported.dtype!=bool or global_supported.shape!=(len(global_points),):
        raise ValueError('global support must be bool N')
    global_admitted=global_supported.copy() if global_policy=='supported-only' else np.ones(len(global_points),bool)
    face_vertex_evidence=local_supported[local_faces].all(axis=1)
    effective_face_support=local_face_supported & face_vertex_evidence
    if face_policy=='original':effective_face_support=np.ones(len(local_faces),bool)
    # Default forwards the original arguments, preserving original face reasons.
    result=assemble_supported(global_points,global_colors,global_admitted,
        local_vertices,local_colors,admitted,local_faces,
        local_face_supported if local_policy=='supported-only' else effective_face_support,
        local_vertex_source,check=check)
    result['global_admitted']=global_admitted.copy()
    result['local_admitted']=admitted.copy()
    result['local_evidence_supported']=local_supported.copy()
    result['global_evidence_supported']=global_supported.copy()
    result['local_face_input_supported']=local_face_supported.copy()
    result['local_face_vertices_evidence_supported']=face_vertex_evidence
    result['local_face_admitted']=result['local_face_to_output']>=0
    counts=result['counts']
    counts['admitted_local_origins']=counts.pop('supported_local_origins')
    counts['admitted_global_origins']=int(global_admitted.sum())
    counts['supported_global_origins']=int(global_supported.sum())
    counts['evidence_supported_global_origins']=int(global_supported.sum())
    counts['evidence_unsupported_global_origins']=int((~global_supported).sum())
    counts['admitted_evidence_unsupported_global_origins']=int((global_admitted & ~global_supported).sum())
    counts['evidence_supported_local_origins']=int(local_supported.sum())
    counts['evidence_unsupported_local_origins']=int((~local_supported).sum())
    counts['admitted_evidence_unsupported_local_origins']=int((admitted & ~local_supported).sum())
    counts['restored_failed_face_audit_faces']=int((result['local_face_admitted'] & ~local_face_supported).sum())
    counts['restored_faces_without_full_evidence']=int((result['local_face_admitted'] & ~(local_face_supported & face_vertex_evidence)).sum())
    result['policy']=dict(result['policy'],local_policy=local_policy,global_policy=global_policy,face_policy=face_policy,
        global_point_admission='Evidence-supported only' if global_policy=='supported-only' else 'All global point origins; support flags unchanged',
        local_point_admission='Evidence-supported only' if local_policy=='supported-only' else 'All local point origins; support flags unchanged',
        face_admission='All existing original faces, subject to basic geometry checks; evidence labels unchanged' if face_policy=='original' else 'Original face support AND original evidence support of all three vertices; no policy relaxation')
    if global_policy=='all-points' and local_policy=='all-points' and face_policy=='original':
        result['status']='ALL_ORIGINAL_COMPARISON_NOT_GEOMETRY_ACCEPTED'
    elif global_policy=='all-points' or face_policy=='original':
        result['status']='RELAXED_ADMISSION_COMPARISON_NOT_GEOMETRY_ACCEPTED'
    elif local_policy=='all-points':result['status']='ALL_LOCAL_COMPARISON_NOT_GEOMETRY_ACCEPTED'
    result['semantics']=('Admission and evidence support are separate. Global and local point admission follow their explicit policies; '
        'unsupported origins are never relabelled supported. Face policy may retain only evidence-approved original faces '
        'or restore original faces regardless of evidence; no new faces are generated. Basic index, zero-area and budget checks remain. Exact-coordinate pooling, no averaging, '
        'snapping, inferred shell ingestion, new triangles or gap filling. Point admission does not establish physical correctness '
        'or increased coverage; caller retains conflict/unverified distinctions. Exact sharing may create nonmanifold topology.')
    return result
