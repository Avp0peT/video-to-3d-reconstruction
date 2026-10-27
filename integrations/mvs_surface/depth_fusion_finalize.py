"""Keep the original surface and admit only audited TSDF additions.

No source face is removed. New triangles need entire-footprint masks and depth
consistency, so a raw fusion coverage increase cannot hide a background regression.
"""
from pathlib import Path
import time,uuid,html
import numpy as np
from .global_local_pipeline import binding,load,save
from .mutual_model_pipeline import load_model_inputs
from .boundary_alignment_pipeline import _npz
from .global_local_model_pipeline import write_mesh
from .depth_fusion_pipeline import review_summary
from .fused_mesh_review import review_mesh
from .depth_fusion_visuals import project_footprint,write_coverage_panel
from integrations.viewer.export import export_html


def select_additions(gate,common_near_bits,centroid_classification):
    keep=np.asarray(gate['keep']);fg=np.asarray(gate['full_foreground_view_count']);bits=np.asarray(common_near_bits);labels=np.asarray(centroid_classification)
    if keep.ndim!=1:raise ValueError('one-dimensional keep mask required')
    n=len(keep)
    if keep.dtype!=bool or keep.shape!=(n,) or fg.shape!=(n,) or fg.dtype.kind not in 'iu' or bits.shape!=(n,) or bits.dtype.kind not in 'iu' or labels.shape!=(n,) or labels.dtype.kind not in 'iu':raise ValueError('matching face evidence arrays required')
    if not np.isin(labels,[0,1,2]).all() or (fg<0).any() or (fg>128).any() or (bits<0).any() or any(int(x)>2**64-1 for x in bits):raise ValueError('invalid evidence vocabulary/range')
    common=np.array([int(x).bit_count() for x in bits],np.uint8)
    accepted=keep&(fg>=2)&(common>=2)&(labels==1)
    return dict(accepted=accepted,common_near_view_count=common,
        counts=dict(candidates=n,mask_gate_pass=int(keep.sum()),two_complete_foreground_views=int((fg>=2).sum()),two_common_near_views=int((common>=2).sum()),centroid_supported=int((labels==1).sum()),accepted=int(accepted.sum())),
        policy='Full triangle mask veto + two complete foreground views + two common near-depth sources throughout the cell + supported centroid. Correlated evidence, not independent truth.')


def finalize(fusion_run,output,*,blender=False,seconds=720):
    source=Path(fusion_run).resolve(strict=True);out=Path(output).resolve();start=time.monotonic()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('fresh disjoint output required')
    if type(blender) is not bool or type(seconds) is not int or not 1<=seconds<=1200:raise ValueError('invalid budget/options')
    def check():
        if time.monotonic()-start>seconds:raise TimeoutError('shared finalization deadline')
    refs=[]
    def read(path,expected=None):
        actual=binding(path)
        if expected is not None and any(actual[k]!=expected[k] for k in ('path','bytes','sha256')):raise ValueError('changed fusion artifact')
        refs.append(actual);return actual
    raw_result_ref=read(source/'RESULT.json');raw=load(source/'RESULT.json')
    if raw.get('status')!='FUSED_SURFACE_GENERATED_NOT_GEOMETRY_ACCEPTED':raise ValueError('completed fusion source required')
    read(source/'PLAN.json');plan=load(source/'PLAN.json')
    for item in raw['source_bindings']:read(item['path'],item)
    fusion_ref=raw['artifacts']['fusion']['arrays'];read(fusion_ref['path'],fusion_ref)
    review_ref=raw['artifacts']['new_review']['arrays'];read(review_ref['path'],review_ref)
    coverage_ref=raw['artifacts']['coverage']['arrays'];read(coverage_ref['path'],coverage_ref)
    with np.load(fusion_ref['path'],allow_pickle=False) as a:
        v=a['vertices'].copy();f=a['faces'].copy();colors=a['colors'].copy();bits=a['face_common_near_bits'].copy()
    with np.load(review_ref['path'],allow_pickle=False) as a:labels=a['centroid_classification'].copy()
    with np.load(coverage_ref['path'],allow_pickle=False) as a:coverage={k:a[k].copy() for k in a.files if k.startswith('coverage_view_')}
    if plan.get('full_source'):
        from .full_depth_inputs import load_full_inputs
        data=load_full_inputs(plan['full_source'],check)
        if Path(load(plan['full_source'])['baseline_model']).resolve()!=Path(plan['source']).resolve():raise ValueError('full source baseline differs')
    else:data=load_model_inputs(plan['source'],check)
    refs+=data['bindings'];views=data['views']
    if [x['view_id'] for x in views]!=plan['view_ids']:raise ValueError('view order drift')
    out.mkdir(parents=True);events=[]
    def stage(name):
        check();events.append(dict(stage=name,elapsed_seconds=time.monotonic()-start));save(out/'STAGES.json',events);print(name,flush=True)
    try:
        stage('full_triangle_mask_gate');from .fused_surface_mask_gate import gate_surface
        gate=gate_surface(v,f,views,check);meta=_npz(out/'MASK_GATE.npz',gate);save(out/'MASK_GATE.json',meta)
        selection=select_additions(gate,bits,labels);save(out/'ADMISSION.json',_npz(out/'ADMISSION.npz',selection))
        accepted=f[selection['accepted']];used,inverse=np.unique(accepted,return_inverse=True)
        new_v=v[used];new_f=inverse.reshape(-1,3).astype(np.int64);new_colors=colors[used]
        old_v=data['local_vertices'];old_f=data['local_faces'];old_c=data['local_colors']
        combined_v=np.concatenate((old_v,new_v));combined_f=np.concatenate((old_f,new_f+len(old_v)));combined_c=np.concatenate((old_c,new_colors))
        if len(combined_v)>200000 or len(combined_f)>400000:raise ValueError('combined geometry budget exceeded; no truncation')
        save(out/'PLAN.json',dict(fusion_source=str(source),source_bindings=refs,policy=selection['policy'],original_surface_preserved=True,no_cross_source_welding=True,scale_status='unknown',length_unit=None,seconds=seconds,blender=blender))
        stage('combined_geometry_review');review=review_mesh(combined_v,combined_f,views,check);save(out/'REVIEW.json',_npz(out/'REVIEW.npz',review))
        artifacts={'combined':write_mesh(out/'combined_surface.ply',combined_v,combined_f,combined_c),'original':write_mesh(out/'original_surface.ply',old_v,old_f,old_c)}
        if len(new_f):artifacts['additions']=write_mesh(out/'audited_additions.ply',new_v,new_f,new_colors)
        # Exact source provenance retained: final triangles start with every old face.
        save(out/'PROVENANCE.json',_npz(out/'PROVENANCE.npz',dict(original_vertex_count=len(old_v),original_face_count=len(old_f),accepted_raw_face_ids=np.flatnonzero(selection['accepted']),accepted_raw_vertex_ids=used,old_vertices_exact=np.array_equal(combined_v[:len(old_v)],old_v),old_faces_exact=np.array_equal(combined_f[:len(old_f)],old_f),new_vertices_are_interpolated=True)))
        stage('projection_regression_check');folder=out/'coverage';folder.mkdir();panels=[]
        for view in views:
            before=project_footprint(old_v,old_f,view,check);after=project_footprint(combined_v,combined_f,view,check)
            panels.append(write_coverage_panel(folder/('view_'+str(view['view_id'])+'.png'),view,coverage['coverage_view_'+str(view['view_id'])],before,after))
        regressions=[]
        for p in panels:
            c=p['counts']
            if c['new_foreground_covered']<c['old_foreground_covered'] or c['new_background_projection']>c['old_background_projection']:regressions.append(c['view_id'])
        save(out/'PROJECTION_COMPARISON.json',dict(panels=panels,regression_views=regressions,gate='No lower projected foreground coverage and no increased known-background projection in the supplied views. Not shape correctness.'))
        if regressions:raise ValueError('projection regression at views '+str(regressions))
        scenes=[]
        for name,title,note in [('original','Original depth surface','Unchanged original local surface; known evidence conflicts retained and disclosed.'),('additions','Audited TSDF additions','Interpolated TSDF surface; correlated depth and full-footprint mask audit; not measured truth.')]:
            if name in artifacts:scenes.append(dict(ply=Path(artifacts[name]['path']).name,sha256=artifacts[name]['sha256'],title=title,data_kind='real_photos',declared_unit=None,note=note))
        save(out/'VIEWER.json',dict(schema='ply-viewer-scenes-v1',scenes=scenes));(out/'viewer').mkdir();viewer=export_html(out/'VIEWER.json',out/'viewer/viewer.html',profile='surface400k')
        blend={'status':'NOT_RUN'}
        if blender:
            stage('blender_create_and_readback');from integrations.blender_export.isolated import export_blend_isolated
            root=Path(__file__).resolve().parents[2];blend=export_blend_isolated(out/'VIEWER.json',out/'blender',scratch_root=root/'outputs/runtime_scratch'/('fusion_final_'+uuid.uuid4().hex),profile='surface400k');save(out/'BLENDER.json',blend)
        if any(binding(x['path'])!=x for x in refs):raise ValueError('input changed during finalization')
        result=dict(status='ORIGINAL_PLUS_AUDITED_FUSION_NOT_GEOMETRY_ACCEPTED',geometry_accepted=False,complete_object=False,scale_status='unknown',length_unit=None,artifacts=artifacts,viewer=viewer,blender=blend,admission=selection['counts'],review=review_summary(review),source_bindings=refs,projection_regression_views=regressions,per_view=panels,wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',result);_report(out,result,source);return result
    except BaseException as exc:
        save(out/'FAILURE.json',dict(status='INCOMPLETE',error_type=type(exc).__name__,error=str(exc),stages=events,elapsed_seconds=time.monotonic()-start));raise


def _report(out,r,source):
    c=r['admission'];t=r['review']['topology'];images=''.join(f'<a href="coverage/{p.name}"><img loading="lazy" src="coverage/{p.name}" width="1100"></a>' for p in sorted((out/'coverage').glob('*.png')))
    page=f'''<!doctype html><meta charset="utf-8"><title>自动融合与边界收口模型</title><style>body{{font:17px/1.65 system-ui;max-width:1200px;margin:32px auto;background:#101725;color:#dae7f7}}a{{color:#82d6ff}}img{{max-width:100%;margin:16px 0}}.warn{{padding:16px;background:#463b20}}</style><h1>原表面保留 + 审核后融合补面</h1><p class="warn">自动流程已生成可查看模型，完整物体仍未验收。缺少可靠深度的拇指、指尖等区域不会凭空补成实测形状；绝对尺度未知。</p><p><a href="viewer/viewer.html">旋转查看／切换两个来源层</a> · <a href="combined_surface.obj">组合 OBJ</a> · <a href="combined_surface.ply">组合 PLY</a> · <a href="blender/model.blend">Blender</a> · <a href="RESULT.json">机器证据</a></p><p>原始局部三角面全部保持原字节坐标和索引顺序；{c['candidates']:,} 个融合候选面中，{c['accepted']:,} 个通过完整投影掩膜、共同深度来源和重投影深度检查后添加。其余保留在原始候选对照中，不冒充正确面。</p><p>交付 {t['vertices']:,} 顶点／{t['faces']:,} 三角面；无新增掩膜背景投影、没有降低已覆盖的前景像素。上述检查限当前视图，不证明真实物体完整或误差达标。</p><p>两层在同一坐标系中叠加，未强行焊接、填洞或跨指缝封面。局部层仍可能有原始深度冲突，已在 REVIEW 中保留。蓝色为审核后的插值面，原照片色为原有局部面。</p>{images}'''
    (out/'index.html').write_text(page,encoding='utf8')
