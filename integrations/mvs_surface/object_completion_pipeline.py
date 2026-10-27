"""Explicit opt-in inferred envelope, with correlated depth-support layers."""
from pathlib import Path
import time,uuid,html
import numpy as np
from .global_local_pipeline import binding,load,save
from .full_depth_inputs import load_full_inputs
from .silhouette_completion import infer_silhouette_completion
from .completion_mesh_audit import orient_components,split_by_face_labels
from .boundary_alignment_pipeline import _npz
from .inferred_mesh_io import write_mesh
from .point_depth_evidence import assess_points
from .depth_fusion_visuals import project_footprint,write_coverage_panel
from integrations.viewer.export import export_html

NOTE='Inferred silhouette envelope, not directly measured geometry; unknown scale. Depth-supported estimate and unsupported completion are separate layers.'

def validate_options(full_source,fusion_run,output,blender,seconds):
    spec=Path(full_source).resolve(strict=True);source=Path(fusion_run).resolve(strict=True);out=Path(output).resolve()
    if type(blender) is not bool or type(seconds) is not int or not 1<=seconds<=1200:raise ValueError('invalid bounded execution options')
    if out.exists() or out==source or out.is_relative_to(source) or source.is_relative_to(out) or spec.is_relative_to(out):raise ValueError('fresh disjoint completion output required')
    return spec,source,out

def _scene(ref,title,note):
    return dict(ply=Path(ref['path']).name,sha256=ref['sha256'],title=title,data_kind='real_photos',declared_unit=None,note=note)

def run(full_source,fusion_run,output,*,blender=False,seconds=720):
    spec,source,out=validate_options(full_source,fusion_run,output,blender,seconds);start=time.monotonic()
    def check():
        if time.monotonic()-start>seconds:raise TimeoutError('shared silhouette-completion deadline')
    refs=[binding(spec),binding(source/'PLAN.json'),binding(source/'RESULT.json')]
    plan=load(source/'PLAN.json');raw=load(source/'RESULT.json')
    if raw.get('status')!='FUSED_SURFACE_GENERATED_NOT_GEOMETRY_ACCEPTED' or Path(plan.get('full_source','')).resolve()!=spec:raise ValueError('completed full-source fusion must bind this exact input specification')
    for key in ('fusion','coverage'):
        for ref in (raw['artifacts'][key]['arrays'],):
            if binding(ref['path'])!=ref:raise ValueError('fusion artifact drift')
            refs.append(ref)
    fm_ref=raw['artifacts']['fusion']['metadata'];fusion_meta=Path(fm_ref['path'])
    if binding(fusion_meta)!=fm_ref:raise ValueError('fusion metadata drift')
    refs.append(fm_ref);fm=load(fusion_meta)
    data=load_full_inputs(spec,check);refs+=data['bindings'];views=data['views']
    if [v['view_id'] for v in views]!=plan['view_ids']:raise ValueError('view order mismatch')
    bounds=np.asarray(fm['parameters']['world_aabb'],np.float64)
    with np.load(raw['artifacts']['coverage']['arrays']['path'],allow_pickle=False) as z:coverage={k:z[k].copy() for k in z.files if k.startswith('coverage_view_')}
    out.mkdir(parents=True);events=[];artifacts={};variants={}
    def stage(name):
        check();events.append(dict(stage=name,elapsed_seconds=time.monotonic()-start));save(out/'STAGES.json',events);print(name,flush=True)
    save(out/'PLAN.json',dict(schema='inferred-completion-plan-v1',full_source=str(spec),fusion_run=str(source),source_bindings=refs,view_ids=plan['view_ids'],bounds=bounds.tolist(),max_axis=160,quantiles=[0.,.1],smoothing_sigma=0.,primary_quantile=.1,explicit_inference=True,classification='Correlated native-depth centroid support; not direct measured vertices or truth.',seconds=seconds,blender=blender,physical_scale='unknown',physical_accuracy='NOT_EVALUATED',geometry_accepted=False))
    try:
        for name,q in (('strict',0.),('robust',.1)):
            stage('silhouette_'+name);envelope=infer_silhouette_completion(views,bounds,max_axis=160,quantile=q,smoothing_sigma=0.,check=check)
            save(out/(name+'_FIELD.json'),_npz(out/(name+'_FIELD.npz'),envelope))
            if not len(envelope['faces']):
                variants[name]=dict(status='NO_ENVELOPE',counts=envelope['counts']);continue
            oriented=orient_components(envelope['vertices'],envelope['faces']);v=oriented['vertices'];f=oriented['faces']
            before=_npz(out/(name+'_TOPOLOGY_BEFORE.npz'),oriented['before']);after=_npz(out/(name+'_TOPOLOGY.npz'),oriented['after'])
            save(out/(name+'_TOPOLOGY_BEFORE.json'),before);save(out/(name+'_TOPOLOGY.json'),after)
            ref=write_mesh(out/(name+'_envelope.ply'),v,f,envelope['colors'],provenance_note=NOTE);artifacts[name]=ref
            variants[name]=dict(status=envelope['status'],counts=envelope['counts'],topology=after,boundary_known_negative=envelope['boundary_known_negative'],closed_envelope_with_safe_boundary=bool(envelope['boundary_known_negative'] and after['closed_oriented_nonzero_volume']),flipped_component_ids=oriented['flipped_component_ids'].tolist(),parameters=envelope['parameters'])
            save(out/'VARIANTS.json',variants)
            if name=='robust':primary=(v,f)
            del envelope,oriented
        if 'robust' not in artifacts:raise ValueError('no robust inferred envelope; strict diagnostic retained')
        v,f=primary
        stage('native_depth_evidence_partition')
        if len(f)>200000:raise ValueError('face audit budget exceeded; no truncation')
        evidence=assess_points(v[f].mean(axis=1),views,check=check);save(out/'DEPTH_EVIDENCE.json',_npz(out/'DEPTH_EVIDENCE.npz',evidence))
        labels=(evidence['classification']==1).astype(np.uint8);partition=split_by_face_labels(v,f,labels)
        layer_records=[];scenes=[]
        for layer in partition['layers']:
            supported=bool(layer['label']);name='depth_supported_estimate' if supported else 'inferred_completion';color=[50,170,245] if supported else [255,150,30]
            ref=write_mesh(out/(name+'.ply'),layer['vertices'],layer['faces'],np.tile(np.array(color,np.uint8),(len(layer['vertices']),1)),provenance_note=NOTE);artifacts[name]=ref
            save(out/(name+'_IDENTITY.json'),_npz(out/(name+'_IDENTITY.npz'),{k:layer[k] for k in ('original_vertex_ids','original_face_ids')}))
            layer_records.append(dict(name=name,label=layer['label'],vertices=len(layer['vertices']),faces=len(layer['faces'])))
            scenes.append(_scene(ref,'Depth-supported estimate' if supported else 'Inferred completion',NOTE))
        save(out/'LAYER_PARTITION.json',_npz(out/'LAYER_PARTITION.npz',dict(face_layer=partition['face_layer'],face_identity_coverage_exact=partition['face_identity_coverage_exact'],layers=layer_records,depth_evidence_counts=evidence['counts'],original_face_count=len(f),individual_layers_need_not_be_closed=True,estimated_vertices_not_observations=True)))
        del evidence
        save(out/'VIEWER.json',dict(schema='ply-viewer-scenes-v1',scenes=scenes));(out/'viewer').mkdir();viewer=export_html(out/'VIEWER.json',out/'viewer/viewer.html',profile='surface400k')
        for name in artifacts.keys()&{'strict','robust'}:
            save(out/(name+'_VIEWER.json'),dict(schema='ply-viewer-scenes-v1',scenes=[_scene(artifacts[name],name+' inferred envelope',NOTE)]));export_html(out/(name+'_VIEWER.json'),out/(name+'_viewer.html'),profile='surface400k')
        stage('all_view_projection_report');folder=out/'coverage';folder.mkdir();panels=[]
        for view in views:
            check();old=project_footprint(data['local_vertices'],data['local_faces'],view,check);new=project_footprint(v,f,view,check)
            panel=write_coverage_panel(folder/('view_'+str(view['view_id'])+'.png'),view,coverage['coverage_view_'+str(view['view_id'])],old,new)
            panel['semantics']='Estimated envelope projection; shared calibrated silhouettes, not independent quality. Old local depth surface is unchanged comparison.'
            panels.append(panel);save(out/'PROJECTIONS.json',dict(panels=panels,finished_views=len(panels),requested_views=len(views)))
        blend={'status':'NOT_RUN'}
        if blender:
            stage('blender_export_and_native_readback');from integrations.blender_export.isolated import export_blend_isolated
            root=Path(__file__).resolve().parents[2];blend=export_blend_isolated(out/'VIEWER.json',out/'blender',scratch_root=root/'outputs/runtime_scratch'/('completion_'+uuid.uuid4().hex),profile='surface400k');save(out/'BLENDER.json',blend)
        if any(binding(ref['path'])!=ref for ref in refs):raise ValueError('bound inputs changed during completion')
        safe=variants['robust']['closed_envelope_with_safe_boundary']
        result=dict(status='CLOSED_INFERRED_ENVELOPE_NOT_GEOMETRY_ACCEPTED' if safe else 'PARTIAL_INFERRED_ENVELOPE_NOT_GEOMETRY_ACCEPTED',geometry_accepted=False,closed_estimated_outer_envelope=safe,complete_observed_object=False,quality_status='NOT_EVALUATED',scale_status='unknown',length_unit=None,source_bindings=refs,artifacts=artifacts,variants=variants,layer_records=layer_records,layer_face_identity_exact=partition['face_identity_coverage_exact'],viewer=viewer,blender=blend,per_view=panels,wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',result);_report(out,result);return result
    except BaseException as exc:
        save(out/'FAILURE.json',dict(status='INCOMPLETE',error_type=type(exc).__name__,error=str(exc),stages=events,elapsed_seconds=time.monotonic()-start));raise

def _report(out,r):
    counts=r['variants']['robust']['counts'];top=r['variants']['robust']['topology']['counts'];panels=''.join(f'<a href="coverage/view_{p["counts"]["view_id"]}.png"><img loading="lazy" src="coverage/view_{p["counts"]["view_id"]}.png"></a>' for p in r['per_view'])
    layers='；'.join(html.escape(x['name'])+': '+format(x['faces'],',')+' 面' for x in r['layer_records'])
    page=f'''<!doctype html><meta charset="utf-8"><title>完整外形推算模型</title><style>body{{font:17px/1.7 system-ui;max-width:1250px;margin:30px auto;background:#101725;color:#e0eaf6}}a{{color:#81d5ff}}img{{max-width:100%;margin:12px 0}}.warn{{background:#49391e;padding:16px}}</style><h1>多视角轮廓补全 · 分层模型</h1><p class="warn">已授权推算补全。蓝色为有原生深度支持的估计表面；橙色为其余推算表面。所有顶点均由轮廓场生成，不冒充照片直接恢复点。拓扑封闭不代表真实形状完整，凹部、指缝和尺寸仍需质量核验。</p><p><a href="viewer/viewer.html">可切换来源层的模型</a> · <a href="blender/model.blend">Blender 文件</a> · <a href="robust_envelope.obj">完整索引 OBJ</a> · <a href="robust_envelope.ply">完整 PLY</a> · <a href="strict_viewer.html">严格轮廓交集对照</a> · <a href="RESULT.json">完整机器记录</a></p><p>状态：{r['status']}。{counts['vertices']:,} 顶点 / {counts['faces']:,} 面，{top['connected_components']} 个连通分量，边界边 {top['boundary_edges']}，非流形边 {top['nonmanifold_edges']}。{layers}。</p><p>复用 {len(r['per_view'])} 个现有标定视角；固定160网格上限。严格版 q=0，主展示版 q=0.1、允许少数轮廓冲突；未平滑、未按手套部位手工造型。未知域不当作背景，未沿体积框封盖。两来源层的面恰好组成完整表面；单层边界允许开放。</p><p>下图依次为原照片、现有深度证据、旧局部表面投影、新推算外形投影。投影覆盖率仅是共享输入一致性，不是独立准确率。自交未检验、尺度未知、CAD 未验收。</p>{panels}'''
    (out/'index.html').write_text(page,encoding='utf8')
