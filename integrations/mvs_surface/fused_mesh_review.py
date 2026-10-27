"""Unmodified mesh topology and correlated four-sample projection diagnostics."""
import numpy as np
from .local_depth_search import _view
from .point_depth_evidence import assess_points, STATES, LABELS


def review_mesh(vertices,faces,views,check=lambda:None):
    """Review only: no face removal, geometry correction, or independent truth.

    Each face has three vertex samples plus its centroid, not full-face visibility.
    Face votes count cameras: foreground requires all four known foreground;
    background requires at least one known-background sample; otherwise unknown.
    Native depth at the centroid supplies a separate correlated consistency label.
    """
    check()
    if not isinstance(vertices,np.ndarray) or vertices.ndim!=2 or vertices.shape[1]!=3 or vertices.dtype.kind!='f' or len(vertices)>200000 or not np.isfinite(vertices).all():raise ValueError('finite floating Nx3 vertices, at most 200000, required')
    if not isinstance(faces,np.ndarray) or faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or len(faces)>400000 or (faces.size and (faces.min()<0 or faces.max()>=len(vertices))):raise ValueError('valid integer Fx3 faces, at most 400000, required')
    if not isinstance(views,list) or not 1<=len(views)<=128:raise ValueError('one to 128 views required')
    selected=[];ids=[]
    for raw in views:
        check();v=_view(raw);valid=v.get('validity')
        if not isinstance(valid,np.ndarray) or valid.dtype!=bool or valid.shape!=v['mask'].shape:raise ValueError('explicit validity required')
        if type(v.get('depth_available')) is not bool:raise ValueError('explicit depth_available required')
        if v['view_id'] in ids:raise ValueError('duplicate view IDs')
        selected.append(v);ids.append(v['view_id'])
    v=vertices.astype(np.float64,copy=False);f=faces.astype(np.int64,copy=False)
    if len(f):
        triangles=v[f]
        with np.errstate(over='ignore',invalid='ignore'):
            cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        if not np.isfinite(cross).all():raise ValueError('triangle area overflow')
        degenerate=np.all(cross==0,axis=1)
        canonical=np.sort(f,axis=1)
        _,inverse,counts=np.unique(canonical,axis=0,return_inverse=True,return_counts=True)
        duplicates=counts[inverse]>1
        edges=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
        unique_edges,edgecounts=np.unique(edges,axis=0,return_counts=True)
        used=np.unique(f)
    else:
        triangles=np.empty((0,3,3));degenerate=np.zeros(0,bool);duplicates=np.zeros(0,bool)
        unique_edges=np.empty((0,2),np.int64);edgecounts=np.empty(0,np.int64);used=np.empty(0,np.int64)
    # Real indexed connectivity: touching at a shared vertex is one component;
    # equal coordinates with distinct indices are intentionally not welded.
    parent=np.arange(len(v),dtype=np.int64)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=int(parent[x])
        return x
    for start in range(0,len(unique_edges),8192):
        check()
        for a,b in unique_edges[start:start+8192]:
            ra,rb=find(int(a)),find(int(b))
            if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
    roots=np.asarray([find(int(i)) for i in used],np.int64)
    vertex_component=np.full(len(v),-1,np.int64)
    _,labels=np.unique(roots,return_inverse=True)
    vertex_component[used]=labels
    face_component=vertex_component[f[:,0]] if len(f) else np.empty(0,np.int64)
    ncomponents=len(np.unique(labels))
    component_details=[]
    edge_components=vertex_component[unique_edges[:,0]] if len(unique_edges) else np.empty(0,np.int64)
    component_vertices=np.bincount(labels,minlength=ncomponents)
    component_faces=np.bincount(face_component,minlength=ncomponents)
    component_boundary=np.bincount(edge_components[edgecounts==1],minlength=ncomponents)
    component_nonmanifold=np.bincount(edge_components[edgecounts>2],minlength=ncomponents)
    component_degenerate=np.bincount(face_component[degenerate],minlength=ncomponents)
    component_duplicates=np.bincount(face_component[duplicates],minlength=ncomponents)
    for index in range(ncomponents):
        if index%8192==0:check()
        component_details.append(dict(component_id=index,vertices=int(component_vertices[index]),faces=int(component_faces[index]),boundary_edges=int(component_boundary[index]),nonmanifold_edges=int(component_nonmanifold[index]),degenerate_faces=int(component_degenerate[index]),duplicate_group_faces=int(component_duplicates[index])))
    topology=dict(vertices=len(v),faces=len(f),connected_components=ncomponents,connectivity='Shared vertex indices, not proximity or coordinate welding',isolated_vertices=len(v)-len(used),boundary_edges=int((edgecounts==1).sum()),nonmanifold_edges=int((edgecounts>2).sum()),degenerate_faces=int(degenerate.sum()),duplicate_faces=(len(f)-len(np.unique(np.sort(f,axis=1),axis=0))) if len(f) else 0,components=component_details,self_intersections='NOT_EVALUATED',hole_loops='NOT_EVALUATED',geometry_accepted=False)
    foreground=np.zeros(len(f),np.uint8);background=foreground.copy();unknown=foreground.copy()
    centroid_states=np.empty((len(f),len(selected)),np.uint8);centroid_classification=np.zeros(len(f),np.uint8)
    per_view=[]
    for view in selected:
        check();stats=dict(view_id=view['view_id'],faces=len(f),evaluated_samples=4*len(f),finite_samples=0,in_front_samples=0,in_image_samples=0,foreground_samples=0,background_samples=0,unknown_samples=0,face_all_foreground=0,face_any_known_background=0,face_unknown=0)
        for start in range(0,len(f),8192):
            check();end=min(start+8192,len(f));tri=triangles[start:end]
            points=np.concatenate((tri,tri.mean(axis=1)[:,None]),axis=1).reshape(-1,3)
            camera=(points-view['C'])@view['R'].T;pix=camera@view['K'].T
            finite=np.isfinite(camera).all(axis=1)&np.isfinite(pix).all(axis=1);positive=finite&(camera[:,2]>1e-9)
            uv=pix[:,:2]/np.where(positive,camera[:,2],1)[:,None];h,w=view['mask'].shape
            inside=positive&np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
            safe=np.where(inside[:,None],uv,0);ix=np.rint(safe[:,0]).astype(int);iy=np.rint(safe[:,1]).astype(int)
            known=inside&view['validity'][iy,ix];fg=known&view['mask'][iy,ix];bg=known&~view['mask'][iy,ix];unk=~known
            good=fg.reshape(-1,4).all(axis=1);bad=bg.reshape(-1,4).any(axis=1);uncertain=~(good|bad)
            foreground[start:end]+=good;background[start:end]+=bad;unknown[start:end]+=uncertain
            for key,value in [('finite_samples',finite),('in_front_samples',positive),('in_image_samples',inside),('foreground_samples',fg),('background_samples',bg),('unknown_samples',unk),('face_all_foreground',good),('face_any_known_background',bad),('face_unknown',uncertain)]:stats[key]+=int(value.sum())
        per_view.append(stats)
    for start in range(0,len(f),100000):
        check();end=min(start+100000,len(f));centroids=triangles[start:end].mean(axis=1)
        evidence=assess_points(centroids,selected,check=check)
        centroid_states[start:end]=evidence['state'];centroid_classification[start:end]=evidence['classification']
    return dict(topology=topology,per_view=per_view,face_component=face_component,vertex_component=vertex_component,face_degenerate=degenerate,face_duplicate_group=duplicates,face_foreground_votes=foreground,face_known_background_votes=background,face_unknown_votes=unknown,centroid_state_legend=dict(STATES),centroid_classification_legend=dict(LABELS),centroid_states=centroid_states,centroid_classification=centroid_classification,view_ids=ids,geometry_accepted=False,mesh_modified=False,status='FUSED_MESH_COMPARISON_NOT_GEOMETRY_ACCEPTED',sampling='Three vertices plus centroid per face per view; not an entire-face or visibility test. Counts overlap for finite/in-front/in-image stages; FG/BG/unknown samples partition samples.',centroid_semantics='Native-depth consistency with unknown generating source; calibration and reconstruction evidence are shared, not independent truth. Background votes are diagnostics, not proof of incorrect physical geometry.')
