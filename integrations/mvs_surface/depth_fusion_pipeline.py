"""Native-depth fusion and diagnostic export, retaining missing observations.

This is an engineering surface estimate, not a Kozlov theorem implementation,
independent shape validation, or a measured-scale CAD solid.
"""
from pathlib import Path
import html,time,uuid
import numpy as np
from .global_local_pipeline import binding,save
from .boundary_alignment_pipeline import _npz
from .mutual_model_pipeline import load_model_inputs
from .depth_coverage import classify_depth_coverage
from .projective_tsdf import fuse_depth_views
from .fused_mesh_review import review_mesh
from .global_local_model_pipeline import write_mesh
from integrations.viewer.export import export_html


def native_bounds(views,max_axis,check=lambda:None):
    """All valid masked native depths, without quantile cropping or old mesh AABB."""
    if type(max_axis) is not int or not 8<=max_axis<=192:raise ValueError('axis must be integer 8..192')
    lo=np.full(3,np.inf);hi=np.full(3,-np.inf);rows=[]
    for view in views:
        check();valid=view['mask']&view['validity']&(view['depth']>0)&view['depth_available']
        y,x=np.nonzero(valid);count=len(x);vmin=np.full(3,np.inf);vmax=np.full(3,-np.inf)
        inv=np.linalg.inv(np.asarray(view['K'],np.float64))
        for start in range(0,count,32768):
            check();xx=x[start:start+32768];yy=y[start:start+32768]
            rays=np.column_stack((xx,yy,np.ones(len(xx))))@inv.T
            p=(rays*view['depth'][yy,xx,None])@np.asarray(view['R'])+view['C']
            if not np.isfinite(p).all():raise ValueError('nonfinite native point')
            vmin=np.minimum(vmin,p.min(0));vmax=np.maximum(vmax,p.max(0))
        if count:lo=np.minimum(lo,vmin);hi=np.maximum(hi,vmax)
        rows.append(dict(view_id=view['view_id'],native_foreground_pixels=count,
            bounds=[vmin.tolist(),vmax.tolist()] if count else None))
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (hi<=lo).any():raise ValueError('nondegenerate native depth bounds unavailable')
    # Four unpadded voxel pitches on each side; no observed foreground point is
    # removed to fit an older mesh, and no padding is treated as surface evidence.
    padding=float((hi-lo).max()/max_axis*4)
    return np.stack((lo-padding,hi+padding)),dict(method='AABB of every positive valid masked native-depth pixel; no quantile crop',raw_bounds=[lo.tolist(),hi.tolist()],padding=padding,per_view=rows,unit=None)


def review_summary(review):
    topology={k:v for k,v in review['topology'].items() if k!='components'}
    labels=review['centroid_classification_legend']
    return dict(topology=topology,
        centroid_labels={name:int(np.count_nonzero(review['centroid_classification']==int(code))) for code,name in labels.items()},
        faces_with_any_sampled_background=int((review['face_known_background_votes']>0).sum()),
        faces_with_at_least_two_foreground_views=int((review['face_foreground_votes']>=2).sum()),
        sampling=review['sampling'],geometry_accepted=False)


def _store_arrays(out,name,value):
    meta=_npz(out/(name+'.npz'),value);save(out/(name+'.json'),meta)
    return dict(arrays=binding(out/(name+'.npz')),metadata=binding(out/(name+'.json')))


def coverage_stride(views,max_points=180000):
    """Choose and disclose one regular grid; never truncate a sampled point set."""
    if type(max_points) is not int or not 1<=max_points<=200000:raise ValueError('invalid sample capacity')
    for stride in range(3,1025):
        count=sum(int(np.count_nonzero(v['mask'][::stride,::stride]&v['validity'][::stride,::stride]&(v['depth'][::stride,::stride]>0)&v['depth_available'])) for v in views)
        if count<=max_points:return stride,count
    raise ValueError('regular sampling capacity unavailable')


def run(source_model,output,*,blender=False,max_axis=160,min_views=2,truncation_voxels=3,seconds=720,full_source=None):
    start=time.monotonic();source=Path(source_model).resolve(strict=True);out=Path(output).resolve()
    if type(blender) is not bool or type(seconds) is not int or not 1<=seconds<=1200:raise ValueError('invalid budget/options')
    if type(max_axis) is not int or not 8<=max_axis<=192 or type(min_views) is not int or not 2<=min_views<=64:raise ValueError('invalid grid/support settings')
    if type(truncation_voxels) not in (int,float) or not np.isfinite(truncation_voxels) or not 0<truncation_voxels<=10:raise ValueError('invalid truncation')
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('fresh disjoint output required')
    def check():
        if time.monotonic()-start>seconds:raise TimeoutError('shared pipeline deadline exceeded')
    out.mkdir(parents=True);events=[]
    def stage(name):
        check();events.append(dict(stage=name,elapsed_seconds=time.monotonic()-start));save(out/'STAGES.json',events);print(name,flush=True)
    try:
        stage('authenticate_inputs')
        if full_source is None:data=load_model_inputs(source,check)
        else:
            from .full_depth_inputs import load_full_inputs
            data=load_full_inputs(full_source,check)
            from .global_local_pipeline import load
            if Path(load(full_source)['baseline_model']).resolve()!=source:raise ValueError('full source baseline differs')
        views=data['views'];native_views=[v for v in views if v['depth_available']]
        stride,sample_count=coverage_stride(views) if full_source else (3,None)
        bounds,box=native_bounds(views,max_axis,check)
        plan=dict(source=str(source),source_bindings=data['bindings'],max_axis=max_axis,min_views=min_views,truncation_voxels=truncation_voxels,bounds=bounds.tolist(),bounds_provenance=box,view_ids=[v['view_id'] for v in views],depth_views=[v['view_id'] for v in views if v['depth_available']],reference_id=data['reference_id'],sampling_stride=3,seconds=seconds,threads=2,blender=blender,scale_status='unknown',length_unit=None,
            semantics='One projective TSDF from unchanged native masked depth and fixed poses. No inferred silhouette shell, no source point merging, no implicit unobserved closure.')
        plan.update(full_source=str(Path(full_source).resolve()) if full_source else None,full_input_provenance=data.get('full_input_provenance'),sampling_stride=stride,planned_sample_count=sample_count)
        save(out/'PLAN.json',plan)
        stage('classify_native_depth_coverage');coverage=classify_depth_coverage(views,stride=stride,check=check);artifacts={'coverage':_store_arrays(out,'COVERAGE',coverage)}
        if coverage['status']=='CAPACITY_LIMIT':raise ValueError('coverage sampling capacity exceeded; no silent subset')
        stage('fuse_native_depths');fused=fuse_depth_views(native_views if full_source else views,bounds,max_axis=max_axis,min_views=min_views,truncation_voxels=truncation_voxels,check=check)
        artifacts['fusion']=_store_arrays(out,'FUSION',fused)
        if not len(fused['faces']):raise ValueError('no supported TSDF surface; no fallback hull generated')
        stage('review_old_and_new');old=review_mesh(data['local_vertices'],data['local_faces'],views,check);new=review_mesh(fused['vertices'],fused['faces'],views,check)
        artifacts['old_review']=_store_arrays(out,'OLD_REVIEW',old);artifacts['new_review']=_store_arrays(out,'NEW_REVIEW',new)
        stage('export_geometry');artifacts['mesh']=write_mesh(out/'fused_surface.ply',fused['vertices'],fused['faces'],fused['colors'])
        spec=dict(schema='ply-viewer-scenes-v1',scenes=[dict(ply='fused_surface.ply',sha256=artifacts['mesh']['sha256'],title='Cross-view native-depth TSDF surface',data_kind='real_photos',declared_unit=None,note='Interpolated TSDF surface estimate. Unknown regions remain open. Uniform diagnostic color; not photo texture. Shape/scale/CAD acceptance pending.')])
        save(out/'VIEWER.json',spec);(out/'viewer').mkdir();viewer=export_html(out/'VIEWER.json',out/'viewer/viewer.html',profile='surface400k')
        stage('per_view_diagnostics');from .depth_fusion_visuals import project_footprint,write_coverage_panel
        panels=[];folder=out/'coverage';folder.mkdir()
        for view in views:
            check();before=project_footprint(data['local_vertices'],data['local_faces'],view,check)
            after=project_footprint(fused['vertices'],fused['faces'],view,check)
            panels.append(write_coverage_panel(folder/('view_'+str(view['view_id'])+'.png'),view,coverage['coverage_view_'+str(view['view_id'])],before,after))
        save(out/'PROJECTION_COMPARISON.json',panels)
        common=fused['face_common_near_bits'];common_counts=np.array([int(x).bit_count() for x in common],np.uint8)
        comparison=dict(old=review_summary(old),new=review_summary(new),native_coverage=coverage['summary'],fusion_counts=fused['counts'],bounds=box,
            common_near_views_per_cell=dict(zero=int((common_counts==0).sum()),one=int((common_counts==1).sum()),two_or_more=int((common_counts>=2).sum())),
            old_isolated_global_points_not_in_surface=len(data['global_points']),per_view=panels,
            limitations=['Old and new share cameras/depths, not independent truth.','Projection footprint is not visibility or completeness.','Per-corner two-view support need not be the same two views throughout a cell.','Background/vote conflicts are reported, not silently declared correct.','Voxel/truncation smoothing can merge thin features; no complete-shape claim.','Self intersections, metric scale and CAD solid validity not evaluated.'])
        save(out/'COMPARISON.json',comparison)
        blend={'status':'NOT_RUN'}
        if blender:
            stage('blender_create_and_readback');from integrations.blender_export.isolated import export_blend_isolated
            root=Path(__file__).resolve().parents[2]
            blend=export_blend_isolated(out/'VIEWER.json',out/'blender',scratch_root=root/'outputs/runtime_scratch'/('depth_fusion_'+uuid.uuid4().hex),profile='surface400k')
            save(out/'BLENDER.json',blend)
        stage('verify_sources_unchanged')
        if any(binding(item['path'])!=item for item in data['bindings']):raise ValueError('source input changed during execution')
        result=dict(status='FUSED_SURFACE_GENERATED_NOT_GEOMETRY_ACCEPTED',geometry_accepted=False,complete_object=False,scale_status='unknown',length_unit=None,artifacts=artifacts,viewer=viewer,blender=blend,comparison=comparison,source_bindings=data['bindings'],wall_seconds=time.monotonic()-start)
        save(out/'RESULT.json',result)
        _report(out,result)
        return result
    except BaseException as exc:
        save(out/'FAILURE.json',dict(error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-start,stages=events,status='INCOMPLETE',geometry_accepted=False))
        raise


def _report(out,result):
    c=result['comparison'];old=c['old'];new=c['new'];cover=c['native_coverage']
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{old["topology"].get(k)}</td><td>{new["topology"].get(k)}</td></tr>' for k in ('vertices','faces','connected_components','boundary_edges','nonmanifold_edges','degenerate_faces'))
    # Panel schemas are saved unchanged; links are obtained from actual files.
    images=''.join(f'<a href="coverage/{html.escape(p.name)}"><img loading="lazy" src="coverage/{html.escape(p.name)}" width="900"></a>' for p in sorted((out/'coverage').glob('*.png')))
    page=f'''<!doctype html><meta charset="utf-8"><title>跨视图深度融合</title><style>body{{font:17px/1.6 system-ui;background:#101623;color:#dce7f5;max-width:1180px;margin:32px auto}}a{{color:#7bd9ff}}table{{border-collapse:collapse}}td,th{{padding:8px 20px;border:1px solid #52617b}}img{{max-width:100%;margin:16px 0}}.warn{{background:#483e21;padding:16px}}</style><h1>跨视图深度融合：统一表面</h1><p class="warn">已生成可查看的表面估计；仍未通过完整物体、几何精度或 CAD 实体验收。无可靠深度的区域保持开放。颜色为诊断色，不是照片纹理。</p><p><a href="viewer/viewer.html">旋转查看新模型</a> · <a href="fused_surface.obj">OBJ</a> · <a href="fused_surface.ply">PLY</a> · <a href="COMPARISON.json">完整对照证据</a></p><table><tr><th>指标</th><th>旧局部面片</th><th>新融合面</th></tr>{rows}</table><p>有效物体掩膜内共 {cover['total_valid_mask_foreground_pixels']:,} 个像素观测，其中 {cover['total_native_depth_foreground_pixels']:,} 有原生深度，{cover['total_foreground_without_native_depth']:,} 无原生深度。这是跨视图像素观测总数，不是物体表面积或独立点数。</p><p>下面逐视角显示照片、深度覆盖与新旧投影。投影覆盖率不代表三维正确率，背景冲突、细缝粘连仍需检查。</p>{images}<p>输入、网格参数、支持来源、缺失与冲突分类均已保存；旧输出未改写。实体尺寸未知，未自动封闭缺口。</p>'''
    (out/'index.html').write_text(page,encoding='utf8')
