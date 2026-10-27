"""Read-only learned surface audit; diagnostics never imply quality acceptance."""
from __future__ import annotations
import argparse
import json
import html
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from integrations.demo_pipeline.learned_silhouette_hull import record
from integrations.viewer.ply_data import read_ply
from integrations.meshprep.geometry import edge_components

def need(ok,message):
    if not ok: raise ValueError(message)

def sample_faces(count,budget):
    need(type(budget) is int and 1<=budget<=100000,'face preview budget must be 1..100000')
    if count<=budget:return np.arange(count,dtype=np.int64)
    return np.arange(budget,dtype=np.int64)*count//budget

def topology(vertices,faces):
    v=np.asarray(vertices,dtype=float);f=np.asarray(faces,dtype=np.int64)
    need(v.ndim==2 and v.shape[1]==3 and len(v)>0 and np.isfinite(v).all(),'finite nonempty vertices required')
    need(f.ndim==2 and f.shape[1]==3 and len(f)>0 and f.min()>=0 and f.max()<len(v),'indexed triangle faces required')
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    parent=np.arange(len(v));size=np.ones(len(v),dtype=np.int64)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in unique:
        ra,rb=root(a),root(b)
        if ra!=rb:
            if size[ra]<size[rb]:ra,rb=rb,ra
            parent[rb]=ra;size[ra]+=size[rb]
    roots=np.array([root(i) for i in f[:,0]])
    _,component_faces=np.unique(roots,return_counts=True)
    used=np.unique(f)
    area2=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)
    face_components=edge_components(f.tolist())
    return {'vertices':len(v),'faces':len(f),'unique_edges':len(unique),'boundary_edges':int((counts==1).sum()),'nonmanifold_edges':int((counts>2).sum()),'vertex_connected_components':len(component_faces),'vertex_component_face_counts_descending':sorted(component_faces.tolist(),reverse=True),'face_edge_connected_components':len(face_components),'face_edge_component_face_counts_descending':sorted([len(c) for c in face_components],reverse=True),'isolated_vertices':int(len(v)-len(used)),'degenerate_faces':int((area2==0).sum()),'bbox':[v.min(axis=0).tolist(),v.max(axis=0).tolist()],'scope':'all stored vertices and faces; no geometry modified','self_intersections':'NOT_EVALUATED','watertight_acceptance':False}

def projected_sample(vertices,faces,camera,image,scope="sampled triangles"):
    pose=image.cam_from_world()
    q=np.asarray(vertices)@np.asarray(pose.rotation.matrix()).T+np.asarray(pose.translation)
    valid=(q[:,2]>1e-12)&np.isfinite(q).all(axis=1)
    uv=np.full((len(q),2),np.nan)
    uv[valid]=np.asarray(camera.img_from_cam(q[valid]))
    valid &= np.isfinite(uv).all(axis=1)
    good=valid[faces].all(axis=1)
    canvas=Image.new('L',(int(camera.width),int(camera.height)))
    draw=ImageDraw.Draw(canvas)
    drawn=0
    for tri in uv[faces[good]]:
        lo,hi=tri.min(axis=0),tri.max(axis=0)
        if hi[0]<0 or hi[1]<0 or lo[0]>=camera.width or lo[1]>=camera.height:continue
        draw.polygon((tri-.5).ravel().tolist(),fill=255);drawn+=1
    return np.array(canvas)>0,{'selected_faces':len(faces),'drawn_faces':drawn,'skipped_near_camera_or_invalid_faces':int((~good).sum()),'clipping':'faces crossing camera plane omitted, not clipped','raster':f'1x opaque triangle union; {scope}; no ground-truth silhouette or IoU claim'}

def preview(vertices,faces,angle,scope="sampled triangles"):
    center=(vertices.min(axis=0)+vertices.max(axis=0))/2
    a=np.deg2rad(angle);e=np.deg2rad(25.)
    R=np.array([[np.cos(a),0,np.sin(a)],[-np.sin(e)*np.sin(a),np.cos(e),np.sin(e)*np.cos(a)],[-np.cos(e)*np.sin(a),-np.sin(e),np.cos(e)*np.cos(a)]])
    q=(vertices-center)@R.T
    extent=max(float(np.ptp(q[:,:2],axis=0).max()),1e-12)
    xy=q[:,:2]*(650/extent)+[400,380];xy[:,1]=760-xy[:,1]
    out=Image.new('RGB',(800,800),(23,27,35));draw=ImageDraw.Draw(out)
    tri=q[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(normal,axis=1)
    shade=.3+.7*np.abs(normal[:,2])/np.maximum(norm,1e-20)
    for i in np.argsort(tri[:,:,2].mean(axis=1),kind='stable'):
        color=tuple(int(c*shade[i]) for c in (205,164,103))
        draw.polygon(xy[faces[i]].ravel().tolist(),fill=color)
    draw.text((15,15),f'{scope.upper()} | yaw {angle} | NOT QUALITY ACCEPTED',fill='white')
    return out

def run(surface_run,previous_run,output,face_budget=20000):
    import pycolmap as pc
    need(pc.__version__=='4.2.1','pycolmap 4.2.1 required')
    bindings={}
    def bind(p,expected=None):
        r=record(p)
        if expected is not None:need(all(r[k]==expected[k] for k in ('bytes','sha256')),'declared source digest differs: '+str(p))
        bindings[r['path']]=r;return Path(r['path'])
    def read(p):return json.loads(bind(p).read_text(encoding='utf-8'))
    bind(__file__)
    import integrations.viewer.ply_data as ply_module
    import integrations.demo_pipeline.learned_silhouette_hull as hash_module
    import pycolmap._core as core
    import integrations.meshprep.geometry as mesh_geometry
    for p in (ply_module.__file__,hash_module.__file__,core.__file__,mesh_geometry.__file__):bind(p)
    surface_run=Path(surface_run).resolve();previous_run=Path(previous_run).resolve()
    surface=surface_run/'surface'
    parent=read(surface_run/'RESULT.json');result=read(surface/'result.json');plan=read(surface_run/'SURFACE_PLAN.json')
    for name in ('LEARNED_SOURCE_CHAIN.json','OUTPUT_BINDINGS.json'):
        p=surface_run/name
        if p.exists():bind(p)
    mesh_records=[g for stage in result.get('stages',[]) if stage.get('name')=='mesh' and stage.get('status')=='GEOMETRY_READBACK_PASSED' for g in stage.get('outputs',{}).get('geometry',[]) if g.get('role')=='mesh']
    need(len(mesh_records)==1,'unique completed mesh readback declaration required; failed/no mesh is not a surface')
    declared=mesh_records[0];mesh_path=bind(declared['path'],declared)
    need(mesh_path.is_relative_to(surface),'mesh outside selected surface run')
    ply=read_ply(mesh_path,max_bytes=128*1024**2,max_vertices=500000,max_faces=1000000)
    need(ply['sha256']==record(mesh_path)['sha256'],'PLY changed during read')
    v=np.array(ply['vertices'],dtype=float);f=np.array(ply['faces'],dtype=np.int64)
    stats=topology(v,f)
    need(stats['vertices']==declared['vertices'] and stats['faces']==declared['triangles'],'declared geometry counts differ')
    bridge=previous_run/'bridge';masks=read(previous_run/'masks'/'RESULT.json')
    need(masks['schema']=='learned-experimental-mvs-masks-v1' and masks['status']=='AVAILABLE','learned mask adapter unavailable')
    need(Path(plan['arguments']['scene_path']).resolve()==bridge/'scene.mvs','surface plan/bridge scene differs')
    bind(bridge/'scene.mvs',plan['scene'])
    for name in ('PLAN.json','RESULT.json','INPUT_BINDINGS.json','UNDISTORT_COMPARISON.json','MVS_COMPARISON.json'):bind(bridge/name)
    sparse=bridge/'undistorted'/'sparse'
    for p in sparse.iterdir():
        if p.is_file():bind(p)
    model=pc.Reconstruction(str(sparse))
    images={model.images[i].name:model.images[i] for i in model.reg_image_ids()}
    mask_rows={r['native_name']:r for r in masks['outputs']}
    need(set(images)==set(mask_rows) and len(images)>=6,'all-view mask/camera relation differs')
    selected_names=[sorted(images)[int(i)] for i in np.linspace(0,len(images)-1,6,dtype=int)]
    prepared=[]
    for name in selected_names:
        im=images[name];cam=model.cameras[im.camera_id]
        need(cam.model_name in ('PINHOLE','SIMPLE_PINHOLE'),'undistorted pinhole required')
        row=mask_rows[name];mp=bind(row['mask']['path'],row['mask']);vp=bind(row['validity']['path'],row['validity'])
        rgb=bind(bridge/'undistorted'/'images'/name)
        planned=[row for row in plan['frames'] if Path(row['image_file']['path']).resolve()==rgb]
        need(len(planned)==1,'surface plan image missing/ambiguous')
        for role,path in [('image_file',rgb),('mask',mp),('validity',vp)]:
            need(Path(planned[0][role]['path']).resolve()==path,'surface plan frame identity differs')
            bind(path,planned[0][role])
        with Image.open(rgb) as image:need(image.size==(int(cam.width),int(cam.height)),'RGB/camera size differs')
        prepared.append((name,im,cam,mp,vp,rgb))
    chosen=sample_faces(len(f),face_budget);sample=f[chosen]
    scope='all stored triangles' if len(chosen)==len(f) else 'sampled triangles'
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    np.save(output/'sample_face_indices.npy',chosen,allow_pickle=False)
    for angle in (0,60,120,180,240,300):preview(v,sample,angle,scope).save(output/f'preview_yaw_{angle:03d}.png')
    metrics=[]
    for name,im,cam,mp,vp,rgb in prepared:
        projected,diagnostic=projected_sample(v,sample,cam,im,scope)
        with Image.open(mp) as image:
            need(image.mode=='L','mask must be L');labels=np.array(image)
        with Image.open(vp) as image:
            need(image.mode=='L','validity must be L');valid=np.array(image)
        need(labels.shape==projected.shape==valid.shape and np.isin(labels,[0,255]).all() and np.isin(valid,[0,255]).all(),'mask raster domain differs')
        labels=labels==255;valid=valid==255
        with Image.open(rgb) as image:arr=np.array(image.convert('RGB'))
        canvas=Image.new('RGB',(arr.shape[1]*2,arr.shape[0]));canvas.paste(Image.fromarray(arr),(0,0))
        arr[projected]=((arr[projected].astype(float)+[0,255,255])/2).astype('uint8')
        canvas.paste(Image.fromarray(arr),(arr.shape[1],0));canvas.save(output/(name+'.rgb_projection.png'))
        Image.fromarray(projected.astype('uint8')*255).save(output/(name+'.sample_projection.png'))
        denom=int((labels&valid).sum());covered=int((projected&labels&valid).sum())
        metrics.append({'name':name,'sample_projection_mask_coverage':covered/denom if denom else None,'covered_label_pixels':covered,'valid_label_pixels':denom,'sample_projection_outside_label_pixels':int((projected&~labels&valid).sum()),**diagnostic})
    drift=[r['path'] for r in bindings.values() if record(r['path'])!=r]
    report={'schema':'learned-surface-readonly-audit-v1','status':'DIAGNOSTIC_ONLY','geometry_accepted':False,'quality_status':'NOT_EVALUATED','source_surface_status':result['status'],'source_continuation_status':parent['status'],'mesh':record(mesh_path),'topology':stats,'visualization':{'face_budget':face_budget,'sample_faces':len(chosen),'stored_faces':len(f),'triangle_scope':scope,'rule':'indices floor(arange(budget)*face_count/budget); full range when under budget','previews':scope+', orthographic painter rendering; not a full z-buffer render','projection_scope':'six uniformly spaced views by sorted native name; '+scope+'; no ground-truth silhouette/IoU'},'views':metrics,'source_bindings':list(bindings.values()),'input_drift':drift,'limitations':['Occlusion and ground truth unknown; predictions and source geometry are not independent.','No physical scale or full-object correctness inferred.','No repair, closure, face removal or geometry export performed.','Boundary/nonmanifold counts do not evaluate self-intersections or prove CAD validity.']}
    if drift:report['status']='INPUT_DRIFT'
    (output/'RESULT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    write_html(output,report)
    need(not drift,'source drift during audit')
    return report


def write_html(output,report):
    esc=lambda value:html.escape(str(value),quote=True)
    scope=report['visualization']['triangle_scope']
    pieces=['<!doctype html><meta charset="utf-8"><title>Surface diagnostic</title>',
        '<style>body{background:#181c24;color:#edf1f6;font:16px system-ui;margin:2rem}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}pre{white-space:pre-wrap}a{color:#80d7ff}figure{margin:0}</style>',
        '<h1>Surface diagnostic — NOT QUALITY ACCEPTED</h1>',
        '<p>'+esc(scope)+'. Orthographic painter previews and 1x projected triangle unions; camera-plane crossing faces omitted. Occlusion and ground truth unknown. No physical-scale or CAD acceptance.</p>',
        '<p><a href="RESULT.json">RESULT.json</a> · <a href="sample_face_indices.npy">Reproducible face indices</a></p>',
        '<h2>All stored geometry statistics</h2><pre>'+esc(json.dumps(report['topology'],indent=2))+'</pre>',
        '<h2>Visualization budget</h2><pre>'+esc(json.dumps(report['visualization'],indent=2))+'</pre>',
        '<h2>Six angles</h2><div class="grid">']
    for angle in (0,60,120,180,240,300):
        pieces.append(f'<figure><img src="preview_yaw_{angle:03d}.png"><figcaption>Yaw {angle}: {esc(scope)}</figcaption></figure>')
    pieces.append('</div><h2>Six source views: RGB left, projection overlay right</h2><div class="grid">')
    for view in report['views']:
        name=esc(view['name'])
        pieces.append(f'<figure><img src="{name}.rgb_projection.png"><figcaption>{name}: {esc(scope)}; projected coverage diagnostic only</figcaption></figure>')
    pieces.append('</div>')
    (Path(output)/'index.html').write_text('\n'.join(pieces),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface-run',required=True);p.add_argument('--previous-run',required=True);p.add_argument('--output',required=True);p.add_argument('--face-budget',type=int,default=20000)
    a=p.parse_args();r=run(a.surface_run,a.previous_run,a.output,a.face_budget)
    print(json.dumps({'status':r['status'],'topology':r['topology'],'geometry_accepted':False}))
