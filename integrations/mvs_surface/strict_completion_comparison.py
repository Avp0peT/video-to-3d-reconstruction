"""Matched input point audit and before/after silhouette completion comparison."""
from pathlib import Path
import time,uuid
import numpy as np
from .global_local_pipeline import binding,load,save,_write_cloud
from .boundary_alignment_pipeline import _npz
from .full_depth_inputs import load_full_inputs
from .dominant_completion import select_dominant_envelope
from .completion_mesh_audit import orient_components
from .inferred_mesh_io import write_mesh
from .strict_point_audit import audit_native_mesh
from .comparison_visuals import render_comparison,write_projection_comparison
from .depth_fusion_visuals import project_footprint
from integrations.viewer.ply_data import read_ply
from integrations.viewer.export import export_html

COLORS=((50,165,235),(255,150,30),(75,215,145))
NAMES=('before_depth','after_robust','after_strict')
TITLES=('Before: depth reconstruction','After: tolerant inference','After: strict silhouette')

def _read_mesh(ref):
    actual=binding(ref['path'])
    if any(actual[key]!=ref[key] for key in ('path','bytes','sha256')):raise ValueError('mesh source drift')
    p=read_ply(ref['path'],max_bytes=16<<20,max_vertices=200000,max_faces=400000)
    return np.asarray(p['vertices'],np.float64),np.asarray(p['faces'],np.int64)

def _compact(v,f,keep):
    chosen=f[keep];used,indices=np.unique(chosen,return_inverse=True)
    return v[used].copy(),indices.reshape(-1,3).astype(np.int64),used

def run(inferred_run,output,*,baseline_delivery=None,blender=False,seconds=600):
    start=time.monotonic();source=Path(inferred_run).resolve(strict=True);out=Path(output).resolve()
    if type(seconds) is not int or not 1<=seconds<=1200 or type(blender) is not bool:raise ValueError('invalid comparison budget')
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('fresh disjoint output required')
    if baseline_delivery is not None:
        base=Path(baseline_delivery).resolve(strict=True)
        if out.is_relative_to(base) or base.is_relative_to(out):raise ValueError('output must be disjoint from baseline')
    def check():
        if time.monotonic()-start>seconds:raise TimeoutError('strict comparison shared deadline')
    refs=[]
    def bound(path,expected=None):
        value=binding(path)
        if expected is not None and any(value[k]!=expected[k] for k in ('path','bytes','sha256')):raise ValueError('comparison source drift')
        refs.append(value);return value
    bound(source/'RESULT.json');bound(source/'PLAN.json');sr=load(source/'RESULT.json');sp=load(source/'PLAN.json')
    if not sr['status'].endswith('INFERRED_ENVELOPE_NOT_GEOMETRY_ACCEPTED'):raise ValueError('completed inferred source required')
    data=load_full_inputs(sp['full_source'],check);refs+=data['bindings'];views=data['views']
    if [v['view_id'] for v in views]!=sp['view_ids']:raise ValueError('view order mismatch')
    raw=Path(sp['fusion_run']);observed_dir=raw.parent/'final';bound(observed_dir/'RESULT.json');orr=load(observed_dir/'RESULT.json')
    if orr['status']!='ORIGINAL_PLUS_AUDITED_FUSION_NOT_GEOMETRY_ACCEPTED':raise ValueError('observed source status unsupported')
    observed_ref=orr['artifacts']['combined'];bound(observed_ref['path'],observed_ref);dv,df=_read_mesh(observed_ref)
    meshes=[dict(title=TITLES[0],vertices=dv,faces=df,color=COLORS[0])];selections={};topologies={}
    for variant,q in (('robust',.1),('strict',0.)):
        if variant=='strict' and sr['variants'][variant]['status']=='NO_ENVELOPE':
            if variant in sr['artifacts']:raise ValueError('absent strict envelope has unexpected artifact')
            v=np.empty((0,3),np.float64);f=np.empty((0,3),np.int64)
            selections[variant]=dict(vertices=v,faces=f,status='NO_ENVELOPE',reason='strict_envelope_absent',counts={'retained_vertices':0,'retained_faces':0})
            topologies[variant]=orient_components(v,f)['after'];meshes.append(dict(title=TITLES[2]+' (no envelope)',vertices=v,faces=f,color=COLORS[2]));continue
        ref=sr['artifacts'][variant];bound(ref['path'],ref);v,f=_read_mesh(ref);params=sr['variants'][variant]['parameters']
        if params['quantile']!=q or params['smoothing_sigma']!=0:raise ValueError('matched frozen quantile variants required')
        if params['world_aabb']!=sr['variants']['robust']['parameters']['world_aabb'] or params['spacing']!=sr['variants']['robust']['parameters']['spacing']:raise ValueError('variant grid differs')
        try:
            selected=select_dominant_envelope(v,f,np.asarray(params['world_aabb']),np.asarray(params['spacing']),check)
        except ValueError as exc:
            # Keep the actual full diagnostic surface when it cannot be adopted.
            selected=dict(vertices=v,faces=f,original_vertex_ids=np.arange(len(v)),original_face_ids=np.arange(len(f)),status='NO_DOMINANT_ADOPTION',selection_error=str(exc),counts={'retained_vertices':len(v),'retained_faces':len(f)},policy={'adopted':False})
        oriented=orient_components(selected['vertices'],selected['faces']);selected['vertices']=oriented['vertices'];selected['faces']=oriented['faces'];selections[variant]=selected;topologies[variant]=oriented['after']
        j=len(meshes);meshes.append(dict(title=TITLES[j],vertices=selected['vertices'],faces=selected['faces'],color=COLORS[j]))
    baseline_checks=None
    if baseline_delivery is not None:
        base=Path(baseline_delivery).resolve(strict=True);bound(base/'RESULT.json');br=load(base/'RESULT.json');checks={}
        for name,mesh in (('complete_envelope',meshes[1]),('observed_depth',meshes[0])):
            ref=br['artifacts'][name];bound(ref['path'],ref);v,f=_read_mesh(ref);checks[name]=bool(np.array_equal(v,mesh['vertices']) and np.array_equal(f,mesh['faces']))
        if not all(checks.values()):raise ValueError('current baseline geometry does not match frozen variants')
        baseline_checks=dict(path=str(base),geometry_identical=checks)
    out.mkdir(parents=True);events=[];artifacts={}
    def stage(name):
        check();events.append(dict(stage=name,elapsed_seconds=time.monotonic()-start));save(out/'STAGES.json',events);print(name,flush=True)
    save(out/'PLAN.json',dict(schema='strict-completion-comparison-v1',inferred_source=str(source),baseline=baseline_checks,source_bindings=refs,seconds=seconds,blender=blender,scope='Independent experiments: fixed stronger native point/face review and fixed q0 versus q0.1 silhouette inference. Not a changed point threshold causing the silhouette change.',camera_ids=[v['view_id'] for v in views],no_new_native_reconstruction=True,no_geometry_motion=True,auto_replace_previous=False,physical_scale='unknown'))
    try:
        stage('stricter_native_point_and_face_review');audit=audit_native_mesh(dv,df,views,check);save(out/'POINT_AUDIT.json',_npz(out/'POINT_AUDIT.npz',audit))
        ids=np.flatnonzero(audit['keep_vertices']);colors=np.tile(np.array(COLORS[2],np.uint8),(len(ids),1))
        artifacts['strict_points']=_write_cloud(out/'strict_depth_points.ply',dv[ids],colors)
        if audit['keep_faces'].any():
            pv,pf,used=_compact(dv,df,audit['keep_faces']);artifacts['strict_depth_surface']=write_mesh(out/'strict_depth_surface.ply',pv,pf,np.tile(np.array(COLORS[2],np.uint8),(len(pv),1)),provenance_note='Subset of existing depth-derived geometry passing strict3view2degree1percent plus full triangle mask gate. Correlated inputs; not physical truth.')
        scene_defs=[]
        for name,mesh in zip(NAMES,meshes):
            if not len(mesh['faces']):
                artifacts[name]=dict(status='NO_ENVELOPE',vertices=0,faces=0,reason='strict_envelope_absent')
                (out/(name+'_viewer.html')).write_text('<!doctype html><meta charset="utf-8"><p>Strict constraints produced no envelope. No mesh was fabricated.</p>',encoding='utf8');continue
            note='Unchanged depth-derived baseline; historical conflicts and gaps remain.' if name==NAMES[0] else 'Inferred silhouette envelope; same frozen masks and cameras. Not measured geometry or independent truth.'
            ref=write_mesh(out/(name+'.ply'),mesh['vertices'],mesh['faces'],np.tile(np.array(mesh['color'],np.uint8),(len(mesh['vertices']),1)),provenance_note=note);artifacts[name]=ref
            scene=dict(ply=Path(ref['path']).name,sha256=ref['sha256'],title=mesh['title'],data_kind='real_photos',declared_unit=None,note=note);scene_defs.append(scene)
            save(out/(name+'_SPEC.json'),dict(schema='ply-viewer-scenes-v1',scenes=[scene]));export_html(out/(name+'_SPEC.json'),out/(name+'_viewer.html'),profile='surface400k')
        for variant in selections:
            save(out/(variant+'_SELECTION.json'),_npz(out/(variant+'_SELECTION.npz'),selections[variant]));save(out/(variant+'_TOPOLOGY.json'),_npz(out/(variant+'_TOPOLOGY.npz'),topologies[variant]))
        save(out/'COMPARE_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=scene_defs));export_html(out/'COMPARE_SPEC.json',out/'layers_viewer.html',profile='surface400k')
        stage('shared_camera_orthographic_comparison')
        # Generic stable sampling across camera order, never hand-specific IDs.
        picked=sorted(set((0,len(views)//2,len(views)-1)));rotations=[views[i]['R'] for i in picked]
        preview=render_comparison(meshes,out/'comparison.png',rotations=rotations,reference_vertices=meshes[1]['vertices'],check=check);save(out/'PREVIEW.json',dict(preview,selected_view_ids=[views[i]['view_id'] for i in picked]))
        stage('every_calibrated_view_projection_comparison');folder=out/'projections';folder.mkdir();panels=[]
        for view in views:
            check();footprints=[project_footprint(m['vertices'],m['faces'],view,check) if len(m['faces']) else np.zeros(view['mask'].shape,bool) for m in meshes]
            row=write_projection_comparison(folder/('view_'+str(view['view_id'])+'.png'),view,footprints);panels.append(row);save(out/'PROJECTIONS.json',dict(panels=panels,finished=len(panels),requested=len(views)))
        totals=[]
        for j,name in enumerate(NAMES):
            counts={key:sum(p['counts'][j][key] for p in panels) for key in ('fg','bg','missing','invalid')};den=counts['fg']+counts['missing'];counts.update(name=name,foreground_coverage=counts['fg']/den if den else None,iou=counts['fg']/(den+counts['bg']) if den+counts['bg'] else None);totals.append(counts)
        blend={'status':'NOT_RUN'}
        if blender and not len(meshes[2]['faces']):blend=dict(status='NOT_APPLICABLE',reason='strict_envelope_absent')
        elif blender:
            stage('strict_blender_create_and_independent_readback');from integrations.blender_export.isolated import export_blend_isolated
            root=Path(__file__).resolve().parents[2];blend=export_blend_isolated(out/'after_strict_SPEC.json',out/'blender_strict',scratch_root=root/'outputs/runtime_scratch'/('strict_comp_'+uuid.uuid4().hex),profile='surface400k');save(out/'BLENDER.json',blend)
        if any(binding(r['path'])!=r for r in refs):raise ValueError('source changed during comparison')
        strict_closed=bool(topologies['strict']['closed_oriented_nonzero_volume'] and selections['strict']['status']!='NO_DOMINANT_ADOPTION')
        result=dict(status='STRICT_COMPARISON_DELIVERED',strict_closed=strict_closed,geometry_accepted=False,complete_measured_object=False,scale_status='unknown',quality_status='NOT_EVALUATED',artifacts=artifacts,source_bindings=refs,baseline=baseline_checks,point_audit=audit['counts'],point_policy=audit['policy'],topologies={k:{key:value for key,value in v.items() if not isinstance(value,np.ndarray)} for k,v in topologies.items()},projection_totals=totals,preview=preview,per_view=panels,blender=blend,automatic_adoption=False,wall_seconds=time.monotonic()-start,interpretation='Stricter point support and silhouette constraint are separate experiments. Reduced occupied region or background overlap does not establish better physical accuracy; missing anatomy may increase. No inferred shell vertex is relabelled as measured.')
        save(out/'RESULT.json',result);_report(out,result);return result
    except BaseException as exc:
        save(out/'FAILURE.json',dict(status='INCOMPLETE',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-start,stages=events));raise

def _report(out,r):
    rows=''.join(f'<tr><td>{title}</td><td>{r["artifacts"][name]["vertices"]:,}</td><td>{r["artifacts"][name]["faces"]:,}</td><td>{p["foreground_coverage"]:.1%}</td><td>{p["bg"]:,}</td></tr>' for name,title,p in zip(NAMES,('推算前：原深度重建','推算后：当前宽容轮廓','推算后：严格轮廓'),r['projection_totals']))
    point=r['point_audit'];images=''.join(f'<a href="projections/{Path(p["path"]).name}"><img loading="lazy" src="projections/{Path(p["path"]).name}"></a>' for p in r['per_view'])
    page=f'''<!doctype html><meta charset="utf-8"><title>严格审核与推算前后对照</title><style>body{{font:17px/1.7 system-ui;max-width:1300px;margin:30px auto;background:#101725;color:#e2ecf7}}a{{color:#83d8ff}}img{{max-width:100%;margin:12px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #476079;padding:9px}}.warn{{background:#49391e;padding:16px}}</style><h1>严格审核 + 推算前后对照</h1><p class="warn">三列是同坐标、同尺度、同观察方向：原深度重建 → 当前推算 → 严格轮廓推算。严格点筛选是另一项独立实验，不能说后两列差异由点审核造成。所有推算仍是估计外形；缺少真实尺度和独立真值。</p><p><a href="before_depth_viewer.html">推算前</a> · <a href="after_robust_viewer.html">当前推算</a> · <a href="after_strict_viewer.html">严格推算</a> · <a href="layers_viewer.html">同坐标切换三方案</a> · <a href="blender_strict/model.blend">严格版 Blender</a> · <a href="after_strict.obj">严格版 OBJ</a> · <a href="RESULT.json">机器证据</a></p><table><tr><th>方案</th><th>顶点</th><th>三角面</th><th>前景投影覆盖</th><th>背景投影像素（越低越少越界）</th></tr>{rows}</table><p>投影统计为{len(r["per_view"])}个已标定视角的共享输入一致性，不是准确率。当前模型不自动被严格版覆盖。</p><a href="comparison.png"><img src="comparison.png"></a><h2>原深度点的严格审核</h2><p>2视角/1°/2% → 3视角/2°/1%。严格点是基线支持点的子集；严格面需要三顶点和共同视角通过，再检查完整三角形掩膜。遮挡、未知和缺深度不算负票。缺失或被拒不等于物理错误。</p><pre>{point}</pre><p><a href="POINT_AUDIT.json">逐项判据</a> · <a href="strict_depth_points.ply">保留点</a> · 原索引与原因见 POINT_AUDIT.npz。无新SfM/MVS/SAM3，不移动点或人为焊接面。</p><h2>逐视角图像对照</h2><p>原图与三种模型投影并列，洋红色轮廓为输入掩膜。用于排查指缝及拇指缺失，不能保证每个细节已修正。</p>{images}'''
    if r['artifacts']['after_strict'].get('status')=='NO_ENVELOPE':
        page=page.replace('<a href="blender_strict/model.blend">严格版 Blender</a>','严格版没有可导出的包络').replace('<a href="after_strict.obj">严格版 OBJ</a>','OBJ 未生成')
    elif r['blender']['status']!='SUCCESS':
        page=page.replace('<a href="blender_strict/model.blend">严格版 Blender</a>','Blender 未完成，见机器证据')
    (out/'index.html').write_text(page,encoding='utf8')
