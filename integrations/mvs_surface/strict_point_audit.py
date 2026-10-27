"""Versioned stricter selection over unchanged correlated native-depth evidence."""
import numpy as np
from .point_depth_evidence import assess_points
from .fused_surface_mask_gate import gate_surface

VERTEX_REASONS={0:'BASELINE_NOT_SUPPORTED',1:'STRICT_FREE_SPACE_CONTRADICTION',2:'FEWER_THAN_THREE_ONE_PERCENT_SOURCES',3:'STRICT_RAY_ANGLE_BELOW_TWO_DEGREES',4:'STRICT_SUPPORTED'}
FACE_REASONS={0:'VERTEX_NOT_STRICT',1:'FEWER_THAN_THREE_COMMON_NEAR_SOURCES',2:'COMMON_CENTROID_RAY_ANGLE_BELOW_TWO_DEGREES',3:'FULL_MASK_GATE_FAILED',4:'FEWER_THAN_THREE_FULL_FOREGROUND_VIEWS',5:'STRICT_FACE_SUPPORTED'}


def _angle(points,support,centers,check):
    result=np.zeros(len(points),np.float64)
    for start in range(0,len(points),4096):
        check();end=min(start+4096,len(points));p=points[start:end]
        rays=p[:,None,:]-centers[None,:,:]
        rays/=np.maximum(np.linalg.norm(rays,axis=2),1e-30)[:,:,None]
        minimum=np.ones(len(p),np.float64);use=support[start:end]
        for a in range(len(centers)):
            if a%16==0:check()
            for b in range(a+1,len(centers)):
                both=use[:,a]&use[:,b]
                if both.any():minimum[both]=np.minimum(minimum[both],np.sum(rays[both,a]*rays[both,b],axis=1))
        result[start:end]=np.degrees(np.arccos(np.clip(minimum,-1.,1.)))
    return result


def _bits(support):
    words=(support.shape[1]+63)//64
    result=np.zeros((len(support),words),np.uint64)
    for j in range(support.shape[1]):result[:,j//64]|=support[:,j].astype(np.uint64)*np.uint64(1<<(j%64))
    return result


def audit_native_mesh(vertices,faces,views,check=lambda:None):
    """Fixed3-source/2degree/1percent subset; never move points or add triangles.

    Source identity is unknown for this generic mesh API. Baseline classification
    is retained verbatim. All depth/mask evidence shares reconstruction/calibration;
    strict consistency is not independent physical truth or geometry acceptance.
    """
    check()
    if not isinstance(faces,np.ndarray) or faces.dtype.kind not in 'iu' or faces.ndim!=2 or faces.shape[1]!=3 or len(faces)>400000:raise ValueError('integer Fx3 faces at most400000 required')
    evidence=assess_points(vertices,views,check=check)
    if faces.size and ((faces<0).any() or (faces>=len(vertices)).any()):raise ValueError('face index outside native vertices')
    faceindices=faces.astype(np.int64,copy=False);centers=np.asarray([v['C'] for v in views],np.float64)
    state=evidence['state'];residual=evidence['signed_relative_residual']
    baseline=evidence['classification']==1
    near=(state==5)&np.isfinite(residual)&(np.abs(residual)<=.01)
    # Residual is recorded only for stable known native foreground2x2 samples.
    # Positive residual (candidate behind measured depth) is not free-space.
    free=np.isfinite(residual)&(residual<-.01)
    count=near.sum(axis=1,dtype=np.uint8);angle=_angle(vertices,near,centers,check)
    vertex_reason=np.zeros(len(vertices),np.uint8)
    vertex_reason[baseline]=2
    vertex_reason[baseline&(count>=3)]=3
    vertex_reason[baseline&(count>=3)&(angle>=2.)]=4
    vertex_reason[baseline&free.any(axis=1)]=1
    keep_vertices=vertex_reason==4
    if (keep_vertices&~baseline).any():raise AssertionError('strict selection expanded baseline')
    face_count=np.zeros(len(faces),np.uint8);face_angle=np.zeros(len(faces),np.float64)
    words=(len(views)+63)//64;face_bits=np.zeros((len(faces),words),np.uint64)
    face_reason=np.zeros(len(faces),np.uint8)
    vertex_eligible=keep_vertices[faceindices].all(axis=1)
    face_reason[vertex_eligible]=1
    for start in range(0,len(faces),8192):
        check();ids=np.flatnonzero(vertex_eligible[start:start+8192])+start
        if not len(ids):continue
        common=near[faceindices[ids]].all(axis=1)
        face_count[ids]=common.sum(axis=1,dtype=np.uint8);face_bits[ids]=_bits(common)
        eligible=face_count[ids]>=3;chosen=ids[eligible]
        if len(chosen):
            face_reason[chosen]=2
            centroid=vertices[faceindices[chosen]].mean(axis=1)
            face_angle[chosen]=_angle(centroid,common[eligible],centers,check)
    candidate=vertex_eligible&(face_count>=3)&(face_angle>=2.)
    face_reason[candidate]=3;candidate_ids=np.flatnonzero(candidate)
    foreground_count=np.zeros(len(faces),np.uint8);mask_keep=np.zeros(len(faces),bool)
    mask_state=np.full((len(faces),len(views)),255,np.uint8)
    gate_metadata=None
    if len(candidate_ids):
        gate=gate_surface(vertices,faceindices[candidate_ids],views,check=check)
        mask_keep[candidate_ids]=gate['keep'];foreground_count[candidate_ids]=gate['full_foreground_view_count'];mask_state[candidate_ids]=gate['view_state']
        face_reason[candidate_ids[gate['keep']]]=4
        good=gate['keep']&(gate['full_foreground_view_count']>=3)
        face_reason[candidate_ids[good]]=5
        gate_metadata=dict(counts=gate['counts'],policy=gate['policy'],semantics=gate['semantics'],view_state_legend=gate['view_state_legend'])
    keep_faces=face_reason==5
    return dict(keep_vertices=keep_vertices,keep_faces=keep_faces,baseline_supported=baseline,baseline_classification=evidence['classification'].copy(),baseline_support_count=evidence['support_count'].copy(),baseline_max_angle_degrees=evidence['max_support_angle_degrees'].copy(),baseline_mask_background_diagnostic=evidence['mask_background_diagnostic'].copy(),
        vertex_reasons=vertex_reason,vertex_strict_support_count=count,vertex_strict_max_angle_degrees=angle,vertex_strict_near_view_bits=_bits(near),vertex_strict_free_space_view_bits=_bits(free),vertex_strict_free_space_conflict=free.any(axis=1),
        face_vertex_eligible=vertex_eligible,face_candidate=candidate,face_reasons=face_reason,face_common_support_count=face_count,face_common_max_angle_degrees=face_angle,face_common_near_view_bits=face_bits,face_full_foreground_view_count=foreground_count,face_mask_gate_keep=mask_keep,face_mask_view_state=mask_state,
        original_vertex_ids=np.arange(len(vertices),dtype=np.int64),original_face_ids=np.arange(len(faces),dtype=np.int64),retained_vertex_ids=np.flatnonzero(keep_vertices),retained_face_ids=np.flatnonzero(keep_faces),candidate_face_ids=candidate_ids,
        vertex_reason_legend=dict(VERTEX_REASONS),face_reason_legend=dict(FACE_REASONS),baseline_classification_legend=evidence['classification_legend'],view_ids=evidence['view_ids'],mask_gate=gate_metadata,mask_gate_not_evaluated=255,
        counts=dict(vertices=len(vertices),faces=len(faces),baseline_supported_vertices=int(baseline.sum()),strict_supported_vertices=int(keep_vertices.sum()),strict_supported_faces=int(keep_faces.sum()),candidate_faces=len(candidate_ids),vertex_reasons={name:int((vertex_reason==code).sum()) for code,name in VERTEX_REASONS.items()},face_reasons={name:int((face_reason==code).sum()) for code,name in FACE_REASONS.items()}),
        policy=dict(version='strict-native-mesh-v1',baseline=evidence['policy'],min_strict_sources=3,strict_axial_relative_tolerance=.01,min_strict_ray_angle_degrees=2.,strict_free_space_signed_residual_below=-.01,min_face_common_near_sources=3,min_face_common_centroid_ray_angle_degrees=2.,min_full_foreground_views=3,full_triangle_mask_gate=True,source_origin='UNKNOWN',source_bit_words=words,source_bit_mapping='View column j maps to uint64word j//64 bit j%64',parameters_frozen=True),
        status='STRICT_NATIVE_MESH_AUDIT_NOT_GEOMETRY_ACCEPTED',geometry_accepted=False,mesh_modified=False,semantics='Strict selection is a subset of unchanged baseline support. Stable native foreground residuals only; missing depth, unknown and occlusion are not free-space contradictions. Mask background remains correlated diagnostic evidence, not physical truth. Original indexed geometry is not moved or repaired; empty strict output is valid.')
