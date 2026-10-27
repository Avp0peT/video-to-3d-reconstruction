"""Reversible global/local evidence selection in their existing calibrated frame.

This does not prove points physically correct. Original native depths provide
correlated consistency evidence; inferred silhouette shells never supply depth.
"""
from pathlib import Path
import html,time,uuid
import numpy as np
from PIL import Image,ImageDraw
from .global_local_pipeline import binding,load,save,_binary,_write_cloud
from .boundary_alignment_pipeline import _npz
from .global_local_model_pipeline import write_mesh
from .mask_consumption_probe import authenticate_depth
from .point_depth_evidence import assess_points
from .support_gated_fusion import assemble_with_policy
from integrations.openmvs_depth.dmap import read_dmap
from integrations.viewer.ply_data import read_ply
from integrations.viewer.export import export_html


def load_model_inputs(source,check):
    source=Path(source).resolve(strict=True);refs={}
    def bind(path,expected=None):
        check();value=binding(path)
        if expected is not None and any(value[k]!=expected[k] for k in ('path','bytes','sha256')):raise ValueError('source binding changed: '+str(path))
        refs[value['path']]=value;return value
    def read(path):
        before=bind(path);value=load(path)
        if binding(path)!=before:raise ValueError('JSON changed during read')
        return value
    # Old manifest authenticates generating-view arrays as well as geometry.
    # Mesh equality alone would not detect a changed per-vertex source ID.
    manifest=source.parent/'SHA256SUMS.txt';manifest_binding=bind(manifest);sealed={}
    for line in manifest.read_text('utf8').splitlines():
        digest,relative=line.split('  ',1)
        if len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest) or relative in sealed:raise ValueError('invalid source manifest')
        sealed[relative]=digest
    for name in ('RESULT.json','PLAN.json','SURFACE.json','SURFACE_PROVENANCE.npz'):
        item=bind(source/name);rel=(source/name).relative_to(source.parent).as_posix()
        if sealed.get(rel)!=item['sha256']:raise ValueError('source result/provenance seal mismatch: '+name)
    if binding(manifest)!=manifest_binding:raise ValueError('source manifest changed')
    result=read(source/'RESULT.json');plan=read(source/'PLAN.json')
    if result.get('status')!='MODEL_GENERATED_NOT_GEOMETRY_ACCEPTED' or 'wall_seconds' not in result:raise ValueError('completed source model required')
    for b in result['source_bindings']:bind(b['path'],b)
    for b in result['artifacts'].values():
        bind(b['path'],b)
        if 'obj' in b:bind(b['obj']['path'],b['obj'])
    historic=plan['source_plan_historical_binding'];bind(historic['path'],historic);camera_plan=read(historic['path'])
    frames={f['image']['global_id']:f for f in camera_plan['frames']};ids=plan['view_ids']
    if len(frames)!=len(camera_plan['frames']) or len(set(ids))!=len(ids) or not 3<=len(ids)<=16 or any(type(i) is not int or i not in frames for i in ids):raise ValueError('unique bound view IDs required')
    depth={}
    for b in result['source_bindings']:
        if Path(b['path']).suffix.lower()=='.dmap':
            d=read_dmap(b['path']);iid=d.view_ids[0]
            if iid in depth or iid not in frames:raise ValueError('native depth identity mismatch')
            authenticate_depth(d,frames[iid]['image']);depth[iid]=d
    views=[]
    for iid in ids:
        check();f=frames[iid];cam=f['image'];shape=(cam['height'],cam['width']);bind(f['image_file']['path'],f['image_file'])
        with Image.open(f['image_file']['path']) as im:image=np.array(im.convert('RGB'))
        if image.shape[:2]!=shape:raise ValueError('image domain mismatch')
        mask=_binary(f['mask'],shape,refs);valid=_binary(f['validity'],shape,refs);d=depth.get(iid)
        if d is not None and d.depth.shape!=shape:raise ValueError('native depth domain mismatch')
        views.append(dict(view_id=iid,image=image,mask=mask,validity=valid,depth_available=d is not None,
            depth=d.depth if d is not None else np.zeros(shape,np.float32),K=d.K if d is not None else cam['K'],R=d.R if d is not None else cam['R'],C=d.C if d is not None else cam['C']))
    whole_refs=[b for b in result['source_bindings'] if Path(b['path']).name=='__mvs_dense.ply']
    if len(whole_refs)!=1:raise ValueError('unique unchanged whole cloud required')
    whole=read_ply(whole_refs[0]['path'],max_bytes=16<<20,max_vertices=100000,max_faces=1)
    surface=result['artifacts'].get('observed_surface')
    if not surface:raise ValueError('depth-derived local surface required; inferred shell cannot substitute')
    local=read_ply(surface['path'],max_bytes=16<<20,max_vertices=200000,max_faces=400000)
    archive=bind(source/'SURFACE_PROVENANCE.npz');read(source/'SURFACE.json')
    with np.load(archive['path'],allow_pickle=False) as z:
        vertices=z['worldvertices'].copy();faces=z['faces'].copy();colors=z['colors'].copy();origins=z['vertex_sourceview'].copy();face_origin=z['sourceview'].copy()
    if binding(archive['path'])!=archive or not np.array_equal(vertices,np.asarray(local['vertices'])) or not np.array_equal(faces,np.asarray(local['faces'])) or not np.array_equal(colors,np.asarray(local['colors'])):raise ValueError('local provenance differs from mesh bytes')
    if origins.shape!=(len(vertices),) or face_origin.shape!=(len(faces),) or not np.isin(origins,ids).all() or not np.array_equal(origins[faces],np.broadcast_to(face_origin[:,None],faces.shape)):raise ValueError('native generating-view identity unavailable')
    return dict(global_points=np.asarray(whole['vertices'],np.float64),global_colors=np.asarray(whole['colors'],np.uint8),local_vertices=vertices,local_colors=colors,local_faces=faces,local_vertex_source=origins,face_source=face_origin,views=views,reference_id=plan['reference_id'],bindings=list(refs.values()),historical_plan_binding=historic)


def audit_face_samples(vertices,faces,source_ids,vertex_supported,views,check=lambda:None):
    """Four extra samples, not an exhaustive depth check over every pixel.

    Prior source mesh already passed its original full mask-footprint rule.
    A missing vertex or failed additional sample prevents adoption of that face.
    """
    eligible=vertex_supported[faces].all(axis=1);kept=np.zeros(len(faces),bool)
    state=np.full((len(faces),4,len(views)),255,np.uint8)
    labels=np.full((len(faces),4),255,np.uint8)
    for start in range(0,len(faces),25000):
        check();ids=np.flatnonzero(eligible[start:start+25000])+start
        if not len(ids):continue
        tri=vertices[faces[ids]]
        samples=np.stack(((tri[:,0]+tri[:,1])/2,(tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2,tri.mean(axis=1)),axis=1)
        evidence=assess_points(samples.reshape(-1,3),views,np.repeat(source_ids[ids],4),check=check)
        state[ids]=evidence['state'].reshape(len(ids),4,len(views));labels[ids]=evidence['classification'].reshape(len(ids),4)
        kept[ids]=evidence['supported'].reshape(len(ids),4).all(axis=1)
    return dict(supported=kept,vertex_eligible=eligible,sample_state=state,sample_classification=labels,
        counts=dict(original_faces=len(faces),vertex_eligible=int(eligible.sum()),sampled_points=int(eligible.sum())*4,retained_faces=int(kept.sum()),sample_rejected=int((eligible&~kept).sum())),
        not_evaluated=255,view_ids=[v['view_id'] for v in views],geometry_accepted=False,
        semantics='Three edge midpoints plus barycenter; not exhaustive depth visibility. Prior full-mask-footprint checks are inherited through unchanged source mesh. Unknown rejects adoption but does not prove a face wrong.')


def _scene(path,title,note):
    return dict(ply=path.name,sha256=binding(path)['sha256'],title=title,data_kind='real_photos',declared_unit=None,note=note)


def _overlay(path,view,points,labels):
    rgb=view['image'].copy();r=np.asarray(view['R']);c=np.asarray(view['C']);k=np.asarray(view['K']);cam=(points-c)@r.T;positive=cam[:,2]>1e-9
    uv=cam@k.T;xy=uv[:,:2]/np.where(positive,cam[:,2],1)[:,None];h,w=rgb.shape[:2]
    inside=positive&np.isfinite(xy).all(axis=1)&(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
    out=Image.fromarray(rgb);draw=ImageDraw.Draw(out);palette={0:(255,192,30),1:(40,235,100),2:(250,50,65)}
    for label in (0,2,1):
        for i in np.flatnonzero(inside&(labels==label)):
            x,y=map(float,xy[i]);draw.ellipse((x-1,y-1,x+1,y+1),fill=palette[label])
    out.save(path)
    return dict(view_id=view['view_id'],projected=int(inside.sum()),counts={str(x):int((inside&(labels==x)).sum()) for x in (0,1,2)},semantics='All projected audit points; image overlays can occlude one another. Green=supported, red=evidence conflict, yellow=unverified. Not physical ground truth or visibility proof.')


def run(source_model,output,*,blender=False,seconds=600,local_policy='supported-only',global_policy='supported-only',face_policy='supported-only'):
    start=time.monotonic();source=Path(source_model).resolve(strict=True);out=Path(output).resolve()
    if type(seconds)is not int or not 1<=seconds<=1200 or type(blender)is not bool:raise ValueError('invalid bounded options')
    if local_policy not in ('supported-only','all-points'):raise ValueError('invalid local policy')
    if global_policy not in ('supported-only','all-points'):raise ValueError('invalid global policy')
    if face_policy not in ('supported-only','original'):raise ValueError('invalid face policy')
    if face_policy=='original' and local_policy!='all-points':raise ValueError('original faces require all-points local policy')
    all_local=local_policy=='all-points'
    all_global=global_policy=='all-points';original_faces=face_policy=='original'
    all_original=all_local and all_global and original_faces
    relaxed=all_global or original_faces
    prefix='all_original' if all_original else 'relaxed_admission'
    point_name=(prefix+'_points') if relaxed else ('local_all_combined_points' if all_local else 'supported_combined_points')
    model_name=(prefix+'_model') if relaxed else ('local_all_model' if all_local else 'supported_model')
    title='整体点＋局部点＋已有面全部保留' if all_original else ('自定义点／面采用策略' if relaxed else ('局部全量点＋整体审核点' if all_local else '整体与局部共同支持点'))
    mesh_note=('All original local faces retained subject only to structural checks; failed or unknown evidence is NOT promoted. No new triangles, surface for isolated global points, inferred shell or gap filling.' if original_faces else 'Only evidence-supported original faces retained; isolated points are not surfaces. No shell inference or new bridge triangles.')
    point_note=('Global points admitted without evidence filtering.' if all_global else 'Global points require evidence support.')+(' All local points admitted with unchanged evidence labels.' if all_local else ' Local points require evidence support.')
    status_name=('ALL_ORIGINAL_COMPARISON_NOT_GEOMETRY_ACCEPTED' if all_original else 'RELAXED_ADMISSION_COMPARISON_NOT_GEOMETRY_ACCEPTED') if relaxed else ('ALL_LOCAL_COMPARISON_NOT_GEOMETRY_ACCEPTED' if all_local else 'SUPPORTED_MODEL_GENERATED_NOT_GEOMETRY_ACCEPTED')
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('fresh disjoint output required')
    out.mkdir(parents=True);status=dict(status='RUNNING',geometry_accepted=False,complete=False,scale_status='unknown');save(out/'RESULT.json',status)
    def check():
        if time.monotonic()-start>seconds:raise TimeoutError('support selection CPU deadline exceeded')
    try:
        data=load_model_inputs(source,check);refs=data['bindings'];root=Path(__file__).resolve().parents[2]
        implementation=[binding(root/x) for x in ('integrations/mvs_surface/mutual_model_pipeline.py','integrations/mvs_surface/point_depth_evidence.py','integrations/mvs_surface/support_gated_fusion.py','scripts/run_support_gated_model.py','integrations/blender_export/isolated.py','integrations/blender_export/isolated_launcher.py')]
        save(out/'PLAN.json',dict(schema='mutual-support-model-v1',source_model=str(source),input_bindings=refs,implementation=implementation,view_ids=[v['view_id'] for v in data['views']],reference_id=data['reference_id'],historical_plan_binding=data['historical_plan_binding'],seconds=seconds,threads=2,blender_requested=blender,
            points_never_moved=True,source_model_never_modified=True,inferred_shell_adopted=False,global_origin='UNKNOWN',local_policy=local_policy,global_policy=global_policy,face_policy=face_policy,
            fusion=point_note,
            face_admission=mesh_note,
            coordinate_policy='Exact XYZ pooling only; no proximity weld, averaging, new triangles or silhouette filling.',
            data_dependence='Global and local estimates share calibration and native depths; consistency is not independent truth.'))
        print('MUTUAL_STAGE global_evidence',flush=True)
        g=assess_points(data['global_points'],data['views'],check=check);gm=_npz(out/'GLOBAL_EVIDENCE.npz',g);save(out/'GLOBAL_EVIDENCE.json',gm)
        print('MUTUAL_STAGE local_evidence',flush=True)
        l=assess_points(data['local_vertices'],data['views'],data['local_vertex_source'],check=check);lm=_npz(out/'LOCAL_EVIDENCE.npz',l);save(out/'LOCAL_EVIDENCE.json',lm)
        print('MUTUAL_STAGE face_evidence',flush=True)
        f=audit_face_samples(data['local_vertices'],data['local_faces'],data['face_source'],l['supported'],data['views'],check);fm=_npz(out/'FACE_EVIDENCE.npz',f);save(out/'FACE_EVIDENCE.json',fm)
        combined=assemble_with_policy(data['global_points'],data['global_colors'],g['supported'],data['local_vertices'],data['local_colors'],l['supported'],data['local_faces'],f['supported'],data['local_vertex_source'],local_policy=local_policy,global_policy=global_policy,face_policy=face_policy,check=check)
        combined['global_evidence_classification']=g['classification'].copy()
        combined['local_evidence_classification']=l['classification'].copy()
        cm=_npz(out/'FUSION_PROVENANCE.npz',combined);save(out/'FUSION_PROVENANCE.json',cm)
        artifacts={};scenes=[]
        def cloud(name,points,colors,title,note):
            if not len(points):return
            p=out/(name+'.ply');artifacts[name]=_write_cloud(p,points,colors)
            spec=out/(name+'_SPEC.json');save(spec,dict(schema='ply-viewer-scenes-v1',scenes=[_scene(p,title,note)]));export_html(spec,out/(name+'_viewer.html'),profile='surface400k')
        cloud('global_supported',data['global_points'][g['supported']],data['global_colors'][g['supported']],'复核后保留的整体点','Correlated native depth support only; not independent truth.')
        audit_colors=np.array([[255,192,30],[40,235,100],[250,50,65]],np.uint8)[g['classification']]
        cloud('global_audit',data['global_points'],audit_colors,'整体点证据：绿支持/红冲突/黄待定','No original points deleted. Red is evidence conflict, not certified physical error; yellow is not rejected as false.')
        cloud(point_name,combined['vertices'],combined['colors'],title,'Exact-coordinate pool; every origin recorded. Admission is not evidence approval. Unique additional coordinates do not establish improved physical coverage.')
        local_audit_colors=np.array([[255,192,30],[40,235,100],[250,50,65]],np.uint8)[l['classification']]
        cloud('local_evidence_audit',data['local_vertices'],local_audit_colors,'局部全量证据：绿支持/红冲突/黄待定','All original local points with unchanged evidence labels, independent of admission policy.')
        isolated=np.ones(len(combined['vertices']),bool);isolated[np.unique(combined['faces'])]=False
        if len(combined['faces']):
            p=out/(model_name+'.ply');artifacts[model_name]=write_mesh(p,combined['vertices'],combined['faces'],combined['colors']);scenes.append(_scene(p,title,mesh_note))
        elif len(combined['vertices']):scenes.append(_scene(out/(point_name+'.ply'),'整体与局部点集','No retained surface; pointcloud only.'))
        colors=np.tile(np.array([55,180,245],np.uint8),(len(combined['vertices']),1));colors[combined['output_global_member']]=[50,220,100]
        cloud('origin_comparison',combined['vertices'],colors,'来源：绿整体/蓝局部补充','Colors encode provenance, not accuracy; exact duplicates retain all origins in ledger.')
        (out/'overlays').mkdir();overlays=[]
        for view in data['views']:
            check();overlays.append(_overlay(out/'overlays'/f'global_{view["view_id"]}.png',view,data['global_points'],g['classification']))
        save(out/'OVERLAYS.json',overlays)
        save(out/'MODEL_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=scenes))
        if scenes:export_html(out/'MODEL_SPEC.json',out/'model_viewer.html',profile='surface400k')
        status.update(status=status_name if scenes else 'NO_SUPPORTED_MODEL',local_policy=local_policy,global_policy=global_policy,face_policy=face_policy,global_counts=g['counts'],local_counts=l['counts'],face_counts=f['counts'],fusion_counts=combined['counts'],topology=combined['topology'],artifacts=artifacts,input_bindings=refs,implementation=implementation,inferred_shell_adopted=False,original_files_changed=False,new_meshes=int(bool(len(combined['faces']))),blender=dict(status='NOT_RUN'),compute_seconds=time.monotonic()-start,
            limitations=['No independent ground truth; shared native depth and calibration can share systematic errors.',
            point_note,
            mesh_note,
            '2x2 mixed mask/depth boundaries are unknown; not background errors.','Face depth audit is vertices plus4samples, not complete per-pixel visibility.','No near-point welding, hole filling, concavity recovery, metric scale or CAD acceptance.'])
        save(out/'RESULT.json',status)
        if blender and scenes:
            from integrations.blender_export.isolated import export_blend_isolated
            scratch=root/'outputs/runtime_scratch'/('blender_'+uuid.uuid4().hex)
            status['blender_scratch']=dict(path=str(scratch),observation='Bounded endpoint snapshots; not continuous scratch disk monitoring',reason='Blender deletes its temporary subdirectories normally; retain strict artifact-tree observation without misclassifying those deletions.')
            save(out/'RESULT.json',status)
            print('MUTUAL_STAGE blender',flush=True);status['blender']=export_blend_isolated(out/'MODEL_SPEC.json',out/'blender',scratch_root=scratch,profile='surface400k')
        for b in refs+implementation:
            if binding(b['path'])!=b:raise ValueError('bound input/code changed during support selection')
        status['wall_seconds']=time.monotonic()-start;save(out/'RESULT.json',status)
        links=[]
        for name,label in [(point_name+'_viewer.html','查看全部采用点'),('model_viewer.html','查看原有全部表面' if original_faces else '查看审核表面'),('global_audit_viewer.html','整体点证据'),('local_evidence_audit_viewer.html','局部点原有证据'),('origin_comparison_viewer.html','整体与局部贡献'),(model_name+'.obj','OBJ模型'),('blender/model.blend','Blender模型')]:
            if (out/name).is_file():links.append(f'<a href="{name}">{label}</a>')
        summary=('整体与局部的输入点及原有三角面全部采用；原审核标记保留，不把冲突或待定改成通过。全量恢复也可能带回粘连、错点和坏面；它不会填出原本缺失的数据。' if all_original else (point_note+' '+mesh_note if relaxed else ('局部点全部采用；整体点仅采用审核通过部分。局部红色冲突和黄色待定标记原样保留。三角面仍须通过原有证据审核，新增孤立点并不自动生成表面。' if all_local else '支持点进入模型；冲突和证据不足的点保留在审查层。')))
        page='<!doctype html><meta charset="utf-8"><title>整体与局部点集策略</title><style>body{max-width:1280px;margin:30px auto;background:#152434;color:#eee;font:17px system-ui}a{color:#8cd6ff}pre{white-space:pre-wrap}img{max-width:23%}</style><h1>'+title+'</h1><p>'+status['status']+'</p><p>'+summary+'原模型不删改。轮廓推定壳没有用来填点。证据存在相关性，不能保证没有错点。</p><p>'+' · '.join(links)+'</p>'
        for label,counts in [('整体点证据',g['counts']),('局部点证据（排除其生成视图）',l['counts']),('三角面证据（与采用数量分列）',f['counts']),('实际采用与合并',combined['counts']),('拓扑',combined['topology'])]:page+='<h2>'+label+'</h2><pre>'+html.escape(str(counts))+'</pre>'
        page+='<h2>逐视图复核：绿色支持、红色冲突、黄色待定</h2>'
        for view in data['views']:page+=f'<a href="overlays/global_{view["view_id"]}.png"><img src="overlays/global_{view["view_id"]}.png" loading="lazy"></a>'
        page+='<h2>记录</h2><p><a href="PLAN.json">输入和策略</a> · <a href="RESULT.json">结果</a> · <a href="GLOBAL_EVIDENCE.json">整体证据</a> · <a href="LOCAL_EVIDENCE.json">局部证据</a> · <a href="FUSION_PROVENANCE.json">来源账本</a></p><pre>'+html.escape('\n'.join(status['limitations']))+'</pre>'
        (out/'index.html').write_text(page,encoding='utf8');return status
    except Exception as exc:
        status.update(status='FAILED_SUPPORT_MODEL',error=type(exc).__name__+': '+str(exc),wall_seconds=time.monotonic()-start);save(out/'RESULT.json',status);raise
