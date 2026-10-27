"""Reversible full projected pixel-square mask veto for an existing surface."""
import numpy as np
from .local_depth_search import _view
from .evidence_surface import _covered_pixels, _known_foreground_bbox

STATES={0:'FULL_FOREGROUND',1:'KNOWN_BACKGROUND',2:'OUT_OF_VIEW',3:'PARTIAL_OFF_IMAGE',
        4:'UNKNOWN_MASK_VALIDITY',5:'UNKNOWN_CAMERA_PLANE',6:'UNKNOWN_DEGENERATE_PROJECTION',7:'UNKNOWN_FOOTPRINT_BUDGET'}


def gate_surface(vertices,faces,views,check=lambda:None):
    """Veto only triangles touching known BG, retaining exact original indices.

    Every touched native pixel square is considered, including edges and tiny
    gaps. Unknown/occluded/off-image evidence never becomes a background veto.
    No minimum foreground-view requirement is imposed; pass means no evaluated
    known-background footprint, not verified surface or visibility.
    """
    check()
    if not isinstance(vertices,np.ndarray) or vertices.dtype.kind!='f' or vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all() or len(vertices)>200000:raise ValueError('finite floating Nx3 vertices required, <=200000')
    if not isinstance(faces,np.ndarray) or faces.dtype.kind not in 'iu' or faces.ndim!=2 or faces.shape[1]!=3 or len(faces)>400000:raise ValueError('integer Fx3 faces required, <=400000')
    if faces.size and ((faces<0).any() or (faces>=len(vertices)).any()):raise ValueError('face index out of range')
    if not isinstance(views,list) or not 1<=len(views)<=128:raise ValueError('one to 128 views required')
    checked=[];integrals=[]
    for raw in views:
        check();v=_view(raw);valid=v.get('validity')
        if not isinstance(valid,np.ndarray) or valid.dtype!=bool or valid.shape!=v['mask'].shape:raise ValueError('explicit matching bool validity required')
        available=v.get('depth_available',True)
        if type(available) is not bool:raise ValueError('depth_available must be bool')
        v['validity']=valid;v['depth_available']=available;checked.append(v)
        fg=(v['mask']&valid).astype(np.int64)
        integrals.append(np.pad(fg.cumsum(0).cumsum(1),((1,0),(1,0))))
    ids=[v['view_id'] for v in checked]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate view IDs')
    state=np.full((len(faces),len(checked)),4,np.uint8)
    off_image=np.zeros(state.shape,bool);occluded=np.zeros(state.shape,bool);occlusion_evaluated=np.zeros(state.shape,bool)
    touched_count=np.zeros(state.shape,np.uint16);bbox_fast=np.zeros(state.shape,bool)
    for column,v in enumerate(checked):
        check();h,w=v['mask'].shape
        for start in range(0,len(faces),1024):
            check();triangles=vertices[faces[start:start+1024]].astype(np.float64)
            camera=(triangles-v['C'])@v['R'].T
            for row,cam in enumerate(camera):
                if row%128==0:check()
                index=start+row
                if (cam[:,2]<=1e-9).all():state[index,column]=2;off_image[index,column]=True;continue
                if (cam[:,2]<=1e-9).any():state[index,column]=5;continue
                projected=cam@v['K'].T;uv=projected[:,:2]/cam[:,2,None]
                if not np.isfinite(uv).all():state[index,column]=5;continue
                a,b=uv[1]-uv[0],uv[2]-uv[0]
                if abs(float(a[0]*b[1]-a[1]*b[0]))<1e-9:state[index,column]=6;continue
                if _known_foreground_bbox(uv,v['mask'].shape,integrals[column]):
                    state[index,column]=0;bbox_fast[index,column]=True
                    lo=np.maximum(np.ceil(uv.min(0)-.5-1e-8).astype(int),[0,0])
                    hi=np.minimum(np.floor(uv.max(0)+.5+1e-8).astype(int),[w-1,h-1])
                    # Strong sufficient occlusion diagnostic over the entire bbox;
                    # it cannot admit a face or establish visibility when false.
                    ds=v['depth'][lo[1]:hi[1]+1,lo[0]:hi[0]+1]
                    if v['depth_available'] and ds.size and (ds>0).all():
                        occlusion_evaluated[index,column]=True
                        occluded[index,column]=bool(ds.max()<cam[:,2].min()*.98)
                    continue
                covered=_covered_pixels(uv,v['mask'].shape)
                if covered is None:state[index,column]=7;continue
                cells,complete=covered;off_image[index,column]=not complete
                touched_count[index,column]=len(cells)
                if not len(cells):state[index,column]=2;continue
                x,y=cells.T;valid=v['validity'][y,x];fg=v['mask'][y,x]
                if (valid&~fg).any():state[index,column]=1;continue
                if not complete:state[index,column]=3;continue
                if not (valid&fg).all():state[index,column]=4;continue
                state[index,column]=0
                ds=v['depth'][y,x]
                if v['depth_available'] and (ds>0).all():
                    occlusion_evaluated[index,column]=True
                    occluded[index,column]=bool(ds.max()<cam[:,2].min()*.98)
    background=state==1;keep=~background.any(axis=1)
    words=(len(checked)+63)//64
    bit_matrix=np.zeros((len(faces),words),np.uint64)
    for column in range(len(checked)):
        bit_matrix[:,column//64]|=background[:,column].astype(np.uint64)*np.uint64(1<<(column%64))
    bits=bit_matrix[:,0] if words==1 else bit_matrix
    bit_schema=dict(version='background-view-bitset-u64-v1',dtype='uint64',word_bits=64,words=words,
        layout='face' if words==1 else 'face,word',view_ids=ids,
        mapping='View column j maps to word j//64, bit j%64, least-significant bit first; <=64views retain one-dimensional output')
    fg_count=np.sum(state==0,axis=1,dtype=np.uint8)
    per_view={str(iid):dict(**{name:int((state[:,j]==code).sum()) for code,name in STATES.items()},
        occlusion_evaluated=int(occlusion_evaluated[:,j].sum()),conservative_all_footprint_occluded=int(occluded[:,j].sum()),
        off_image_flag=int(off_image[:,j].sum()),bbox_fast_accepts=int(bbox_fast[:,j].sum())) for j,iid in enumerate(ids)}
    check()
    return dict(keep=keep,reason=np.where(keep,0,1).astype(np.uint8),reason_legend={0:'NO_KNOWN_BACKGROUND_VETO',1:'KNOWN_BACKGROUND_FOOTPRINT'},
        view_state=state,view_state_legend=dict(STATES),view_ids=ids,background_view_bits=bits,background_view_bits_schema=bit_schema,
        full_foreground_view_count=fg_count,known_background_view_count=background.sum(axis=1,dtype=np.uint8),
        off_image=off_image,occluded=occluded,occlusion_evaluated=occlusion_evaluated,
        sat_touched_pixel_count=touched_count,bbox_fast_accept=bbox_fast,per_view=per_view,
        retained_face_ids=np.flatnonzero(keep),rejected_face_ids=np.flatnonzero(~keep),
        counts=dict(original_faces=len(faces),retained_faces=int(keep.sum()),rejected_known_background_faces=int((~keep).sum()),
            retained_without_two_full_foreground_views=int((keep&(fg_count<2)).sum())),
        policy=dict(max_views=128,max_footprint_cells=4096,minimum_full_foreground_views=0,known_background_veto=True,
            footprint='All native pixel squares touched by the full projected triangle; original SAT with sufficient all-FG bbox shortcut',
            occlusion='Diagnostic only when all touched/bbox native positive depths are at least2percent nearer than every triangle vertex; otherwise visibility unestablished',
            touched_count='SAT count only; zero when bbox shortcut is used, not proof of empty footprint'),
        geometry_accepted=False,status='FULL_FOOTPRINT_MASK_GATE_NOT_GEOMETRY_ACCEPTED',mesh_modified=False,
        semantics='Selection mask only; original raw candidate is unchanged. Unknown/off-image/camera-plane/budget states are not background. No minimum FG count or view-angle gate: weaker than historical native-depth surface acceptance. Occlusion diagnostic never overrides the known-background mask veto; visible-only masks and occluders may reject real hidden surfaces. Not a full depth-consistency, visibility, topology or completeness acceptance.')
