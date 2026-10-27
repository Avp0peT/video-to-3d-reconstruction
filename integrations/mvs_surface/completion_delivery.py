"""Select a closed dominant inferred component, preserving all source evidence."""
from pathlib import Path
import time,uuid
import numpy as np
from .global_local_pipeline import binding,load,save
from .boundary_alignment_pipeline import _npz
from .completion_mesh_audit import orient_components
from .dominant_completion import select_dominant_envelope
from .inferred_mesh_io import write_mesh
from .full_depth_inputs import load_full_inputs
from .depth_fusion_visuals import project_footprint,write_coverage_panel
from integrations.viewer.ply_data import read_ply
from integrations.viewer.export import export_html

def finalize_completion(inferred_run,output,*,blender=False,seconds=600):
    start=time.monotonic();source=Path(inferred_run).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or source.is_relative_to(out) or out.is_relative_to(source):raise ValueError('fresh disjoint completion-delivery directory required')
    if type(seconds) is not int or not 1<=seconds<=900 or type(blender) is not bool:raise ValueError('invalid budget')
    def check():
        if time.monotonic()-start>seconds:raise TimeoutError('completion delivery deadline')
    refs=[]
    def bound(path,expected=None):
        r=binding(path)
        if expected is not None and any(r[k]!=expected[k] for k in ('path','bytes','sha256')):raise ValueError('changed completion source')
        refs.append(r);return r
    bound(source/'RESULT.json');bound(source/'PLAN.json');r=load(source/'RESULT.json');plan=load(source/'PLAN.json')
    if not r['status'].endswith('INFERRED_ENVELOPE_NOT_GEOMETRY_ACCEPTED'):raise ValueError('completed inferred envelope source required')
    ref=r['artifacts']['robust'];bound(ref['path'],ref);ply=read_ply(ref['path'],max_bytes=16<<20,max_vertices=200000,max_faces=400000)
    v=np.array(ply['vertices'],np.float64);f=np.array(ply['faces'],np.int64);parameters=r['variants']['robust']['parameters']
    data=load_full_inputs(plan['full_source'],check);refs+=data['bindings']
    raw=Path(plan['fusion_run']);bound(raw/'RESULT.json');rr=load(raw/'RESULT.json');coverage_ref=rr['artifacts']['coverage']['arrays'];bound(coverage_ref['path'],coverage_ref)
    with np.load(coverage_ref['path'],allow_pickle=False) as z:coverage={k:z[k].copy() for k in z.files if k.startswith('coverage_view_')}
    final=raw.parent/'final';bound(final/'RESULT.json');observed=load(final/'RESULT.json')['artifacts']['combined'];bound(observed['path'],observed)
    depth=read_ply(observed['path'],max_bytes=16<<20,max_vertices=200000,max_faces=400000);dv=np.asarray(depth['vertices'],np.float64);df=np.asarray(depth['faces'],np.int64)
    out.mkdir(parents=True);events=[]
    def stage(name):
        check();events.append(dict(stage=name,elapsed_seconds=time.monotonic()-start));save(out/'STAGES.json',events);print(name,flush=True)
    try:
        stage('dominant_closed_component_selection');selection=select_dominant_envelope(v,f,np.asarray(parameters['world_aabb']),np.asarray(parameters['spacing']),check)
        save(out/'SELECTION.json',_npz(out/'SELECTION.npz',selection));oriented=orient_components(selection['vertices'],selection['faces']);sv=oriented['vertices'];sf=oriented['faces']
        save(out/'TOPOLOGY_BEFORE.json',_npz(out/'TOPOLOGY_BEFORE.npz',oriented['before']));top=_npz(out/'TOPOLOGY.npz',oriented['after']);save(out/'TOPOLOGY.json',top)
        note='Inferred complete outer envelope; silhouette estimate, not measured details. Unknown scale; source components and depth reconstruction preserved separately.'
        mesh=write_mesh(out/'complete_envelope.ply',sv,sf,np.tile(np.array([255,150,30],np.uint8),(len(sv),1)),provenance_note=note)
        artifacts={'complete_envelope':mesh,'observed_depth':write_mesh(out/'observed_depth.ply',dv,df,np.tile(np.array([50,165,235],np.uint8),(len(dv),1)),provenance_note='Unchanged source depth-derived reconstruction. Contains historical conflicts and gaps; not ground truth or an accepted complete model.')}
        if len(selection['discarded_faces']):artifacts['unselected_components']=write_mesh(out/'unselected_components.ply',selection['discarded_vertices'],selection['discarded_faces'],np.tile(np.array([180,100,190],np.uint8),(len(selection['discarded_vertices']),1)),provenance_note='Unselected disconnected inferred components preserved for diagnosis. Not independently proven incorrect.')
        def scene(name,title,note):
            a=artifacts[name];return dict(ply=Path(a['path']).name,sha256=a['sha256'],title=title,data_kind='real_photos',declared_unit=None,note=note)
        inferred_scene=scene('complete_envelope','Inferred complete envelope',note);observed_scene=scene('observed_depth','Observed-depth reconstruction','Depth-derived source kept unchanged; gaps and conflicts remain. Hidden by default in presentation Blender. Never used as truth.')
        save(out/'MODEL_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=[inferred_scene,observed_scene]));save(out/'ENVELOPE_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=[inferred_scene]));save(out/'OBSERVED_SPEC.json',dict(schema='ply-viewer-scenes-v1',scenes=[observed_scene]))
        stage('viewers_and_all_view_comparison');export_html(out/'ENVELOPE_SPEC.json',out/'model_viewer.html',profile='surface400k');export_html(out/'MODEL_SPEC.json',out/'layers_viewer.html',profile='surface400k');export_html(out/'OBSERVED_SPEC.json',out/'observed_viewer.html',profile='surface400k')
        panels=[];folder=out/'coverage';folder.mkdir()
        for view in data['views']:
            before=project_footprint(dv,df,view,check);after=project_footprint(sv,sf,view,check)
            panel=write_coverage_panel(folder/('view_'+str(view['view_id'])+'.png'),view,coverage['coverage_view_'+str(view['view_id'])],before,after);panel['semantics']='Depth-derived reconstruction versus estimated closed silhouette envelope. Shared input projection consistency only.';panels.append(panel)
            save(out/'PROJECTIONS.json',dict(panels=panels,finished_views=len(panels),requested_views=len(data['views'])))
        blend={'status':'NOT_RUN'};styled={'status':'NOT_RUN'}
        if blender:
            stage('base_blender_export');from integrations.blender_export.isolated import export_blend_isolated
            root=Path(__file__).resolve().parents[2];blend=export_blend_isolated(out/'MODEL_SPEC.json',out/'blender_base',scratch_root=root/'outputs/runtime_scratch'/('closed_model_'+uuid.uuid4().hex),profile='surface400k');save(out/'BLENDER_BASE.json',blend)
            if blend['status']!='SUCCESS':raise ValueError('Blender base export failed')
            stage('source_layer_presentation_and_readback');from integrations.blender_export.completion_presentation import style_completion_blend
            styled=style_completion_blend(out/'blender_base/model.blend',out/'blender_base/payload.json',out/'blender',scratch_root=root/'outputs/runtime_scratch'/('closed_layers_'+uuid.uuid4().hex));save(out/'BLENDER.json',styled)
            if styled['status']!='SUCCESS':raise ValueError('Blender presentation failed')
        totals={key:sum(p['counts'][key] for p in panels) for key in ('valid_foreground_pixels','foreground_native_depth','old_foreground_covered','new_foreground_covered','old_background_projection','new_background_projection')}
        totals['old_foreground_coverage']=totals['old_foreground_covered']/max(1,totals['valid_foreground_pixels']);totals['estimated_foreground_coverage']=totals['new_foreground_covered']/max(1,totals['valid_foreground_pixels'])
        if any(binding(x['path'])!=x for x in refs):raise ValueError('delivery input drift')
        result=dict(status='CLOSED_ESTIMATED_OUTER_MODEL_DELIVERED',closed_estimated_outer_model=top['closed_oriented_nonzero_volume'],complete_measured_object=False,geometry_accepted=False,physical_scale='unknown',length_unit=None,quality_status='NOT_EVALUATED',self_intersections='NOT_EVALUATED',source_bindings=refs,source_inferred_run=str(source),selection=selection['counts'],topology=top,artifacts=artifacts,projection_totals=totals,per_view=panels,blender=styled,blender_base=blend,estimated_surface_has_directly_measured_vertices=False,depth_support_of_envelope=r['layer_records'],observed_depth_geometry_unchanged=True,world_coordinates_unchanged=True,wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',result);_report(out,result);return result
    except BaseException as exc:
        save(out/'FAILURE.json',dict(status='INCOMPLETE',error_type=type(exc).__name__,error=str(exc),stages=events,elapsed_seconds=time.monotonic()-start));raise

def _report(out,r):
    t=r['topology']['counts'];n=r['selection'];p=r['projection_totals'];images=''.join(f'<a href="coverage/view_{x["counts"]["view_id"]}.png"><img loading="lazy" src="coverage/view_{x["counts"]["view_id"]}.png"></a>' for x in r['per_view'])
    page=f'''<!doctype html><meta charset="utf-8"><title>完整外形估计模型交付</title><style>body{{font:17px/1.7 system-ui;background:#101725;color:#e2ecf7;max-width:1250px;margin:30px auto}}a{{color:#83d8ff}}.warn{{padding:18px;background:#49391e}}img{{max-width:100%;margin:10px 0}}</style><h1>完整外形估计模型 · 已生成</h1><p class="warn">这是你授权的推算补全版本：橙色为轮廓推算出的完整外壳；蓝色为原深度重建，单独保留。完整指封闭外形，不代表精细真实形状或精确尺寸。指缝、凹陷和薄片仍可能有误差；没有把推算点说成实测点。</p><p><a href="model_viewer.html">旋转查看完整外形</a> · <a href="blender/model.blend">下载 Blender（分层）</a> · <a href="complete_envelope.obj">完整 OBJ</a> · <a href="complete_envelope.ply">完整 PLY</a> · <a href="layers_viewer.html">切换原重建与推算层</a> · <a href="observed_viewer.html">仅原深度重建</a> · <a href="RESULT.json">机器记录</a></p><p>{t['vertices']:,} 个顶点，{t['faces']:,} 个三角面，{t['connected_components']} 个连通主体；开放边 {t['boundary_edges']}，非流形边 {t['nonmanifold_edges']}，退化面 {t['degenerate_faces']}。主组件占推算表面 {n['retained_face_fraction']:.2%}；{n['discarded_faces']:,} 个其余组件面保留在 <a href="unselected_components.ply">诊断 PLY</a>，没有焊接或人为封盖。</p><p>67 个已标定视角的共享掩膜投影覆盖：原深度模型 {p['old_foreground_coverage']:.1%} → 推算外形 {p['estimated_foreground_coverage']:.1%}。这是二维覆盖，不是独立准确率。背景投影像素 {p['old_background_projection']:,} → {p['new_background_projection']:,}；此值保留以显示放宽轮廓冲突的代价。</p><p>通用规则：完整输入校验 → 全视角深度融合 → 固定轮廓场推算 → 主连通组件占比≥90%、已闭合、离体积框至少一格 → 来源分层 → Blender 原生回读。没有手工指定手指或拇指。</p><p>自交与 CAD 实体未验收，真实尺度未知。蓝色源层本身仍有不完整和冲突；橙色外形未被证明是照片真实恢复细节。</p>{images}'''
    (out/'index.html').write_text(page,encoding='utf8')
