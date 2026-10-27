"""One-pass partial surface from a bound regional reconstruction.

Never merges components or closes holes. Explicit whole-point derivations,
image conversion and mask remapping remain inspectable separate stages.
"""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import sys

from integrations.demo_pipeline.pipeline import (
    ROOT, LIGHT_REL, MVS_REL, _invoke, entry_code, read, save, sha, utc,
)
from integrations.demo_pipeline.balanced_point_run import binding, check_bindings


BOUNDARY_GEOMETRY_STATUS = 'BOUNDARY_GEOMETRY_DIAGNOSTIC_NOT_ACCEPTED'
SHAPE_SUPPORT_STATUSES = ('REVIEW_REQUIRED', 'NO_FLAGGED_REGRESSION_NOT_QUALITY_ACCEPTANCE')


def validate_shape_support(record, surface_run, candidate_run, output):
    """Require bound, read-only coverage evidence before candidate promotion."""
    if (not isinstance(record, dict) or record.get('schema') != 'shape-support-audit-v1'
            or record.get('status') not in SHAPE_SUPPORT_STATUSES
            or record.get('source_unchanged') is not True
            or record.get('geometry_changed') is not False
            or type(record.get('candidate_promotable')) is not bool
            or 'candidate_run' not in record):
        raise ValueError('shape support diagnostic contract failed')
    source = record.get('source_surface_run')
    candidate = record.get('candidate_run')
    if (not isinstance(source, str) or Path(source).resolve() != Path(surface_run).resolve()
            or (candidate_run is None and candidate is not None)
            or (candidate_run is not None and (not isinstance(candidate, str)
                or Path(candidate).resolve() != Path(candidate_run).resolve()))):
        raise ValueError('shape support request binding differs')
    if candidate_run is None and record['candidate_promotable']:
        raise ValueError('shape support cannot promote an absent candidate')
    if record['candidate_promotable'] and (record['status'] != SHAPE_SUPPORT_STATUSES[1]
            or 'error' in record or 'drift_error' in record):
        raise ValueError('shape support promotion contradicts review or error evidence')
    if not (Path(output)/'index.html').is_file():
        raise ValueError('shape support diagnostic report missing')
    return record['status']


def validate_boundary_geometry(record):
    """A completed diagnostic never authorizes geometry or mask replacement."""
    if (not isinstance(record, dict)
            or record.get('status') not in (BOUNDARY_GEOMETRY_STATUS, 'NOT_APPLICABLE')
            or record.get('source_unchanged') is not True
            or record.get('geometry_accepted') is not False
            or record.get('geometry_changed') is not False):
        raise ValueError('boundary geometry diagnostic contract failed')
    if record['status'] == 'NOT_APPLICABLE' and (
            not isinstance(record.get('reason'), str) or not record['reason'].strip()):
        raise ValueError('boundary geometry not-applicable reason required')
    return record['status']


def choose_model(models, model_id=None):
    if not models:
        raise ValueError('no reconstructed component')
    if model_id is not None:
        if type(model_id) is not int or model_id < 0:
            raise ValueError('component ID must be a nonnegative integer')
        selected = [m for m in models if m['model_id'] == model_id]
        if len(selected) != 1:
            raise ValueError('selected component not uniquely present')
        return selected[0]
    return sorted(models, key=lambda m: (-m['registered_views'], -m['points'], m['model_id']))[0]


def exclusion_candidates(model):
    """Only structural whole-point defects; no residual/appearance filtering."""
    bad = []
    for pid, point in model.points3D.items():
        views = [int(e.image_id) for e in point.track.elements]
        if len(set(views)) != len(views) or len(set(views)) < 2:
            bad.append(int(pid))
    if len(bad) * 100 > len(model.points3D):
        raise ValueError('structural exclusion exceeds 1 percent; upstream repair required')
    return sorted(bad)


def code_bindings():
    paths = {Path(__file__), ROOT/'scripts/build_region_surface.py', ROOT/'scripts/run_photo_demo.py',
        ROOT/'scripts/prepare_mvs.py', ROOT/'scripts/prepare_mvs_masks.py', ROOT/'scripts/build_mvs_surface.py',
        ROOT/'scripts/audit_camera_domain.py', ROOT/'scripts/export_blender.py',
        ROOT/'integrations/demo_pipeline/camera_domain_gate.py',
        ROOT/'integrations/demo_pipeline/pipeline.py', ROOT/'integrations/demo_pipeline/supervision.py',
        ROOT/'integrations/demo_pipeline/balanced_point_run.py',
        ROOT/'integrations/demo_pipeline/shape_support_audit.py',
        ROOT/'integrations/demo_pipeline/silhouette_review.py',
        ROOT/'integrations/video_masks/region_surface_export.py',
        ROOT/'integrations/video_masks/surface_support.py',
        ROOT/'integrations/video_masks/face_mask_adapter.py',
        ROOT/'integrations/video_masks/face_mask_gate.py',
        ROOT/'integrations/video_masks/face_gap_gate.py',
        ROOT/'integrations/video_masks/face_edge_gate.py',
        ROOT/'integrations/video_masks/local_subdivision.py',
        ROOT/'integrations/video_masks/local_face_evidence.py',
        ROOT/'integrations/video_masks/local_surface_adapter.py',
        ROOT/'integrations/video_masks/boundary_precision.py',
        ROOT/'integrations/video_masks/boundary_geometry.py',
        ROOT/'integrations/video_masks/boundary_geometry_binding.py',
        ROOT/'integrations/video_masks/boundary_geometry_evidence.py',
        ROOT/'integrations/video_masks/auto_boundary.py',
        ROOT/'integrations/video_masks/native_gap_evidence.py',
        ROOT/'integrations/video_masks/narrow_gap.py',
        ROOT/'integrations/video_masks/narrow_gap_review.py',
        ROOT/'integrations/openmvs_depth/dmap.py',
        ROOT/'integrations/video_masks/region_surface_masks.py'}
    paths.update(ROOT/p for p in ('scripts/audit_point_density.py',
        'integrations/video_masks/density_report.py','integrations/video_masks/density_sampling.py',
        'integrations/video_masks/point_density.py','integrations/video_masks/region_tracks.py'))
    for folder in ('integrations/mvs_surface', 'integrations/mvs_masks', 'integrations/viewer', 'integrations/blender_export'):
        paths.update((ROOT/folder).rglob('*.py'))
    return [binding(p) for p in sorted(paths)]


def render(out, result):
    esc = html.escape
    preferred = result.get('preferred_viewer')
    preferred_link = ('<p><a href="'+esc(Path(preferred).relative_to(out).as_posix(), quote=True)
        +'">默认查看：'+('局部候选（未验收）' if result.get('candidate_promoted') else '原始表面（未验收）')+'</a></p>') if preferred else ''
    links = []
    for label, path in result.get('artifacts', {}).items():
        p = Path(path)
        if p.is_file():
            links.append('<li><a href="'+esc(p.relative_to(out).as_posix(), quote=True)+'">'+esc(label)+'</a></li>')
    rows = ''.join('<tr><td>'+esc(x['name'])+'</td><td>'+esc(x['status'])+'</td><td>'
        +esc(str(x.get('exit_code', '')) )+'</td><td><a href="logs/'+esc(x['name'])+'/stdout.txt">stdout</a> '
        +'<a href="logs/'+esc(x['name'])+'/stderr.txt">stderr</a></td></tr>' for x in result['stages'])
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>局部表面</title>'
        '<style>body{font:17px system-ui;max-width:1050px;margin:40px auto;background:#f4f6f8;color:#142032}'
        'td,th{padding:10px;border-bottom:1px solid #ccd}pre{white-space:pre-wrap}a{color:#075cba}</style>'
        '<h1>区域点云 → 局部表面</h1><p>'+esc(result['status'])+'</p>'+preferred_link+
        '<p>只使用一个分量；孔洞保留。原始模型未覆盖。几何质量、物理尺度和完整物体均未验收。</p>'
        '<p>整点排除：'+esc(str(result.get('excluded_point_ids',[])))+'</p>'
        '<p>边界几何诊断：'+esc(result.get('boundary_geometry_status','NOT_EVALUATED'))
        +' '+esc(result.get('boundary_geometry_reason',''))+'；只诊断，不替换默认候选。</p>'
        '<p>形状覆盖审计：'+esc(result.get('shape_support_status','NOT_EVALUATED'))
        +'；无警示也不代表完整性或质量验收。</p><pre>'+esc('\n'.join(result.get('warnings',[])))+'</pre>'
        '<table><tr><th>步骤</th><th>状态</th><th>退出码</th><th>日志</th></tr>'+rows+'</table>'
        '<ul>'+''.join(links)+'</ul><pre>'+esc(result.get('error',''))+'</pre>'
        '<p><a href="PLAN.json">冻结计划</a> · <a href="RESULT.json">完整结果</a> · '
        '<a href="REGION_SOURCE_CHAIN.json">来源链</a></p>', encoding='utf-8')


def run(source_run, output, model_id=None):
    source = Path(source_run).resolve(strict=True)
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'outputs') or source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError('fresh disjoint output under repository outputs required')
    source_result = source/'geometry/RESULT.json'
    chosen = choose_model(read(source_result)['models'], model_id)
    binary = Path(chosen['binary_path']).resolve(strict=True)
    refs = [binding(source_result), binding(source/'geometry/PLAN.json'), binding(source/'input/PLAN.json')]
    refs += [binding(p) for p in sorted(binary.iterdir()) if p.is_file()]
    import pycolmap
    model = pycolmap.Reconstruction(str(binary))
    excluded = exclusion_candidates(model)
    before = dict(views=model.num_reg_images(), points=model.num_points3D(), observations=model.compute_num_observations())
    del model
    python = (ROOT/LIGHT_REL).resolve(strict=True)
    tools = (ROOT/MVS_REL).resolve(strict=True)
    refs += code_bindings() + [binding(python)]
    refs += [binding(p) for p in sorted(tools.iterdir()) if p.suffix.lower() in ('.exe','.dll')]
    if check_bindings(refs):
        raise ValueError('precondition bindings changed')
    out.mkdir(parents=True)
    save(out/'PLAN.json', dict(schema='regional-partial-surface-v1', source_run=str(source),
        model_id=chosen['model_id'], before=before, expected_excluded_point_ids=excluded,
        exclusion_policy='whole structural-invalid points only, max 1 percent, exact frozen IDs',
        new_segmentation_executions=0, new_sfm_executions=0, maximum_surface_attempts=1,
        automatic_retry=False, threads=2, mvs_total_seconds=1200, mvs_stage_seconds=900,
        automatic_face_mask_gate=True, face_mask_gate_seconds=240, face_mask_gate_memory_bytes=3<<30,
        automatic_face_local=True,face_local_seconds=480,face_local_memory_bytes=4<<30,
        face_local_cooperative_seconds=420,
        automatic_boundary_geometry=True,boundary_geometry_seconds=600,
        boundary_geometry_memory_bytes=4<<30,boundary_geometry_max_frames=16,
        automatic_shape_support=True,shape_support_seconds=600,shape_support_memory_bytes=4<<30,
        mvs_commit_bytes=8<<30, mvs_output_bytes=2<<30, hole_filling=False, component_merging=False,
        outer_entire_run_threshold_bytes=2<<30, outer_total_roots_threshold_bytes=3<<30,
        budget_note='Outer sampled disk threshold includes export/bridge/masks/logs and may stop before inner surface allowance.',
        source_and_code_bindings=refs))
    refs.append(binding(out/'PLAN.json'))
    result = dict(schema='regional-partial-surface-result-v1', status='RUNNING', stages=[], artifacts={},
        started_utc=utc(), model_id=chosen['model_id'], before=before, excluded_point_ids=excluded,
        geometry_accepted=False, quality_status='NOT_EVALUATED', scale_status='unknown',
        mask_consumption='NOT_VERIFIED', cad_status='NOT_EVALUATED', errors=[],
        point_density_status='NOT_EVALUATED',point_density_target_met=False,
        surface_support_status='NOT_EVALUATED',face_mask_gate_status='NOT_EVALUATED',face_local_status='NOT_EVALUATED',
        boundary_geometry_status='NOT_EVALUATED',shape_support_status='NOT_EVALUATED',
        candidate_promotable=False,candidate_promoted=False,warnings=[])
    surface_ready = False
    candidate_run = None
    def persist():
        save(out/'RESULT.json', result)
        render(out, result)
    def unchanged():
        if check_bindings(refs):
            raise ValueError('bound source/code/stage output drift')
    def launch(name, args, seconds, memory):
        unchanged()
        if any(s['name']==name for s in result['stages']):
            raise RuntimeError('stage already attempted')
        item = dict(name=name, status='STARTING', started_utc=utc(),
            argv=[str(python),'-B','-X','utf8',*map(str,args)], seconds=seconds,memory_bytes=memory)
        result['stages'].append(item);persist()
        print('STAGE_BEGIN '+name, flush=True)
        entry = _invoke(ROOT, out, name, item['argv'], seconds=seconds, memory=memory)
        code = entry_code(entry)
        item.update(status='PROCESS_COMPLETED' if code==0 else 'FAILED', exit_code=code,
            reclaim_confirmed=entry.get('reclaim_confirmed'), finished_utc=utc())
        persist();print('STAGE_END '+name+' rc='+str(code), flush=True)
        if code != 0:
            raise RuntimeError(name+' execution failed; see preserved logs')
        unchanged()
    def accept(folder):
        refs.extend(binding(p) for p in sorted(folder.rglob('*')) if p.is_file())
        result['stages'][-1]['status']='BUSINESS_GATE_PASSED'
        save(out/'STAGE_BINDINGS.json',refs)
    persist()
    try:
        exported = out/'export'
        launch('export',[ROOT/'scripts/build_region_surface.py','export-worker',
            '--source',source,'--output',exported,'--model-id',chosen['model_id'],
            '--exclusion-plan',out/'PLAN.json'],120,4<<30)
        export = read(exported/'EXPORT.json')
        if export.get('status')!='EXPORTED_NOT_GEOMETRY_ACCEPTED':
            raise ValueError('export business gate failed')
        refs.extend(export['source_bindings'])
        unchanged()
        accept(exported)
        result['derived_counts']=read(exported/'DERIVATION.json')['after']
        result['artifacts']['派生记录'] = str(exported/'DERIVATION.json')
        launch('camera_domain',[ROOT/'scripts/audit_camera_domain.py','--model',exported/'binary',
            '--output',out/'CAMERA_DOMAIN.json'],90,2<<30)
        domain = read(out/'CAMERA_DOMAIN.json')
        if domain.get('status')!='CAMERA_DOMAIN_VALID' or domain.get('source_unchanged') is not True:
            raise ValueError('numerical camera domain rejected')
        refs.append(binding(out/'CAMERA_DOMAIN.json'))
        result['stages'][-1]['status']='BUSINESS_GATE_PASSED'
        bridge = out/'bridge'
        launch('bridge',[ROOT/'scripts/prepare_mvs.py','--sparse',exported/'binary','--image-root',exported/'images',
            '--interface-colmap',tools/'InterfaceCOLMAP.exe','--output',bridge,
            '--wall-seconds','180','--max-output-mib','1024'],240,4<<30)
        bridged = read(bridge/'RESULT.json')
        if bridged.get('status')!='PASS' or bridged.get('input_bindings_unchanged') is not True:
            raise ValueError('bridge business gate failed')
        accept(bridge)
        masks = out/'masks'
        launch('masks',[ROOT/'scripts/build_region_surface.py','masks-worker',
            '--source',exported,'--bridge',bridge,'--output',masks],180,4<<30)
        consumed = read(masks/'RESULT.json')
        if (consumed.get('coverage_status')!='COMPLETE' or consumed.get('adapter_source_chain')!='VERIFIED'
                or consumed.get('available_views')!=before['views']):
            raise ValueError('complete bound mask coverage required')
        refs.extend(consumed['source_bindings'])
        unchanged()
        accept(masks)
        from integrations.mvs_surface.runner import make_plan, save_json
        surface_plan = make_plan(bridge/'scene.mvs', Path(consumed['surface_image_root']),
            masks/'masks', masks/'validity',tools,total_seconds=1200,stage_seconds=900,
            memory_bytes=8<<30,output_bytes=2<<30,input_origin='unverified',ignore_label=0,photo_source=None)
        plan_path = out/'SURFACE_PLAN.json'
        save_json(plan_path,surface_plan,exclusive=True)
        refs.append(binding(plan_path))
        chain = dict(schema='regional-surface-source-chain-v1',adapter_source_chain='VERIFIED',
            export=binding(exported/'EXPORT.json'),derivation=binding(exported/'DERIVATION.json'),
            bridge=binding(bridge/'RESULT.json'),masks=binding(masks/'RESULT.json'),surface_plan=binding(plan_path),
            runner_input_origin='unverified',legacy_producer_claimed=False,physical_identity_verified=False,
            geometry_accepted=False,mask_consumption='NOT_VERIFIED')
        save(out/'REGION_SOURCE_CHAIN.json',chain);refs.append(binding(out/'REGION_SOURCE_CHAIN.json'))
        save(out/'STAGE_BINDINGS.json',refs)
        launch('surface',[ROOT/'scripts/build_mvs_surface.py','run','--plan',plan_path,
            '--plan-sha256',sha(plan_path),'--output',out/'surface'],1230,9<<30)
        surf = read(out/'surface/result.json')
        if surf.get('status')!='SURFACE_GENERATED_NOT_ACCEPTED':
            raise ValueError('nonempty finite surface output not established')
        result['stages'][-1]['status']='BUSINESS_GATE_PASSED'
        result['status']='SURFACE_GENERATED_NOT_ACCEPTED'
        result['surface_result']=surf
        surface_ready = True
        from integrations.viewer.export import export_html
        for label,filename in [('dense','__mvs_dense.ply'),('mesh','__mvs_surface_raw.ply')]:
            ply = out/'surface/work'/filename
            result['artifacts'][label+' PLY']=str(ply)
            spec = out/(label+'_scenes.json')
            save(spec,dict(schema='ply-viewer-scenes-v1',scenes=[dict(ply=str(ply),sha256=sha(ply),title=label,
                data_kind='real_photos',declared_unit=None,note='Partial native geometry; holes preserved; not quality accepted.',
                evidence=[str(out/'surface/result.json'),str(out/'REGION_SOURCE_CHAIN.json')])]))
            try:
                viewer=out/(label+'_viewer.html')
                exported_view=export_html(spec,viewer,profile='surface400k')
                save(out/(label+'_viewer.json'),exported_view)
                result['artifacts'][label+' 三维查看']=str(viewer)
                if label == 'mesh':
                    result['preferred_viewer']=str(viewer)
            except Exception as exc:
                result['errors'].append(label+' viewer: '+str(exc))
        launch('blender',[ROOT/'scripts/export_blender.py','--spec',out/'mesh_scenes.json',
            '--output',out/'blender','--profile','surface400k'],210,5<<30)
        blend=read(out/'blender/RESULT.json')
        if blend.get('status')=='SUCCESS':
            result['artifacts']['Blender模型']=str(out/'blender/model.blend')
            result['stages'][-1]['status']='BUSINESS_GATE_PASSED'
        else:
            raise ValueError('Blender independent readback failed')
        launch('point_density',[ROOT/'scripts/audit_point_density.py','--export',exported,
            '--output',out/'point_density'],240,3<<30)
        density=read(out/'point_density/RESULT.json')
        if density.get('source_unchanged') is not True:
            raise ValueError('density source binding not established')
        result['point_density_status']=density['status']
        result['point_density_target_met']=density['status']=='DENSITY_TARGET_MET'
        result['dense_visibility_coverage']=density['dense_visibility_coverage']
        result['artifacts']['物体取点密度检查']=str(out/'point_density/index.html')
        result['stages'][-1]['status']='DENSITY_AUDITED_NOT_QUALITY_ACCEPTED'
        launch('surface_support',[ROOT/'scripts/build_region_surface.py','support-worker',
            '--source',out,'--output',out/'support_audit'],180,3<<30)
        support=read(out/'support_audit/RESULT.json')
        if (support.get('status')!='DIAGNOSTIC_ONLY' or support.get('source_unchanged') is not True
                or support.get('geometry_accepted') is not False):
            raise ValueError('surface support diagnostic contract failed')
        accept(out/'support_audit')
        result['surface_support_status']=support['status']
        result['artifacts']['面片支撑诊断']=str(out/'support_audit/index.html')
        result['stages'][-1]['status']='SUPPORT_AUDITED_NOT_QUALITY_ACCEPTED'
        launch('face_mask_gate',[ROOT/'scripts/build_region_surface.py','face-mask-worker',
            '--source',out,'--output',out/'face_mask_gate'],240,3<<30)
        gated=read(out/'face_mask_gate/RESULT.json')
        if (gated.get('status')!='CANDIDATE_SURFACE_NOT_ACCEPTED'
                or gated.get('source_unchanged') is not True or gated.get('geometry_accepted') is not False):
            raise ValueError('face mask candidate contract failed')
        for name in ('index.html','comparison_viewer.html','retained_viewer.html','retained.ply','quarantined.ply','FACE_LEDGER.csv'):
            if not (out/'face_mask_gate'/name).is_file():
                raise ValueError('face mask candidate artifact missing: '+name)
        accept(out/'face_mask_gate')
        result['face_mask_gate_status']=gated['status']
        result['artifacts']['上一阶段边缘候选（未验收）']=str(out/'face_mask_gate/retained_viewer.html')
        result['artifacts']['原始面与候选隔离面对照（未验收）']=str(out/'face_mask_gate/comparison_viewer.html')
        result['artifacts']['自动面掩膜约束记录']=str(out/'face_mask_gate/index.html')
        result['artifacts']['保留面候选 PLY（未验收）']=str(out/'face_mask_gate/retained.ply')
        result['artifacts']['隔离面候选 PLY（未验收）']=str(out/'face_mask_gate/quarantined.ply')
        result['stages'][-1]['status']='CANDIDATE_DERIVED_NOT_QUALITY_ACCEPTED'
        launch('face_local',[ROOT/'scripts/build_region_surface.py','face-local-worker',
            '--source',out,'--gate',out/'face_mask_gate','--output',out/'face_local'],480,4<<30)
        local=read(out/'face_local/RESULT.json')
        if (local.get('status')!='LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED'
                or local.get('source_unchanged') is not True or local.get('geometry_accepted') is not False):
            raise ValueError('local face candidate contract failed')
        for name in ('index.html','retained_viewer.html','comparison_viewer.html','retained.ply',
                     'quarantined.ply','FACE_LEDGER.csv','LINEAGE.npz'):
            if not (out/'face_local'/name).is_file():
                raise ValueError('local face candidate artifact missing: '+name)
        accept(out/'face_local')
        result['face_local_status']=local['status']
        candidate_run=out/'face_local'
        result['artifacts']['单独修正版候选（未验收）']=str(candidate_run/'retained_viewer.html')
        result['artifacts']['局部边界处理记录']=str(out/'face_local/index.html')
        result['artifacts']['原始面与局部边界候选对照（未验收）']=str(out/'face_local/comparison_viewer.html')
        result['artifacts']['局部保留面候选 PLY（未验收）']=str(out/'face_local/retained.ply')
        result['artifacts']['局部隔离面候选 PLY（未验收）']=str(out/'face_local/quarantined.ply')
        result['stages'][-1]['status']='LOCAL_CANDIDATE_DERIVED_NOT_QUALITY_ACCEPTED'
        result['boundary_geometry_status']='RUNNING'
        boundary = out/'boundary_geometry'
        launch('boundary_geometry',[ROOT/'scripts/build_region_surface.py','boundary-geometry-worker',
            '--source',out,'--gate',out/'face_local','--output',boundary],600,4<<30)
        boundary_record=read(boundary/'RESULT.json')
        boundary_status=validate_boundary_geometry(boundary_record)
        if not (boundary/'index.html').is_file():
            raise ValueError('boundary geometry diagnostic report missing')
        accept(boundary)
        result['boundary_geometry_status']=boundary_status
        if boundary_status=='NOT_APPLICABLE':
            result['boundary_geometry_reason']=boundary_record['reason']
        result['artifacts']['边界几何诊断（不改变模型）']=str(boundary/'index.html')
        result['artifacts']['边界几何诊断记录']=str(boundary/'RESULT.json')
        result['stages'][-1]['status']=boundary_status
    except Exception as exc:
        result['error']=type(exc).__name__+': '+str(exc)
        if result['stages'] and result['stages'][-1]['status']=='PROCESS_COMPLETED':
            result['stages'][-1].update(status='BUSINESS_GATE_FAILED',error=str(exc))
        if result['status'] in ('RUNNING','SURFACE_GENERATED_NOT_ACCEPTED'):
            result['status']='PARTIAL_OR_FAILED'
        if result['boundary_geometry_status']=='RUNNING':
            result['boundary_geometry_status']='FAILED'
            result['status']='PARTIAL_OR_FAILED'
    finally:
        # Even an unavailable candidate leaves useful raw-surface coverage evidence.
        # Never replace a prior failure with a later diagnostic failure or promote
        # geometry after any failed stage. No diagnostic is retried.
        if surface_ready:
            audit=out/'shape_support'
            result['shape_support_status']='RUNNING'
            try:
                args=[ROOT/'scripts/build_region_surface.py','shape-support-worker',
                      '--source',out,'--output',audit]
                if candidate_run is not None:args.extend(['--candidate-run',candidate_run])
                launch('shape_support',args,600,4<<30)
                record=read(audit/'RESULT.json')
                status=validate_shape_support(record,out,candidate_run,audit)
                accept(audit)
                result['shape_support_status']=status
                result['candidate_promotable']=record['candidate_promotable']
                result['artifacts']['形状覆盖与退化审计']=str(audit/'index.html')
                result['artifacts']['形状覆盖与退化审计记录']=str(audit/'RESULT.json')
                result['stages'][-1]['status']=status
                if (record['candidate_promotable'] and candidate_run is not None
                        and result['status']=='SURFACE_GENERATED_NOT_ACCEPTED' and not result.get('error')):
                    result['preferred_viewer']=str(candidate_run/'retained_viewer.html')
                    result['candidate_promoted']=True
                else:
                    result['warnings'].append('覆盖审计未授权候选晋升或此前步骤失败；保留原始表面查看。')
            except Exception as exc:
                message=type(exc).__name__+': '+str(exc)
                result['shape_support_status']='FAILED'
                result['shape_support_error']=message
                result['warnings'].append('覆盖审计失败或证据不完整；候选未晋升。')
                if result['stages'] and result['stages'][-1]['name']=='shape_support' and result['stages'][-1]['status']=='PROCESS_COMPLETED':
                    result['stages'][-1].update(status='BUSINESS_GATE_FAILED',error=message)
                if not result.get('error'):result['error']=message
                if result['status'] in ('RUNNING','SURFACE_GENERATED_NOT_ACCEPTED'):
                    result['status']='PARTIAL_OR_FAILED'
        result['binding_drift']=check_bindings(refs)
        if result['binding_drift']:
            result['status']='INPUT_DRIFT_REFUSED'
            result['candidate_promoted']=False
            if (out/'mesh_viewer.html').is_file():result['preferred_viewer']=str(out/'mesh_viewer.html')
        result['finished_utc']=utc();persist()
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','export-worker','masks-worker','support-worker','face-mask-worker','face-local-worker','boundary-geometry-worker','shape-support-worker'])
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--model-id',type=int)
    p.add_argument('--bridge',type=Path)
    p.add_argument('--exclusion-plan',type=Path)
    p.add_argument('--gate',type=Path)
    p.add_argument('--candidate-run',type=Path)
    p.add_argument('--footprint-mode',choices=['disk','closed_cells'],default=None)
    a=p.parse_args(argv)
    if a.footprint_mode is not None and a.action != 'face-local-worker':
        p.error('--footprint-mode only applies to face-local-worker')
    if a.action=='shape-support-worker':
        from integrations.demo_pipeline.shape_support_audit import run as audit_shape_support
        result=audit_shape_support(a.source,a.output,candidate_run=a.candidate_run)
        try:
            validate_shape_support(result,a.source,a.candidate_run,a.output)
        except ValueError as exc:
            print(str(exc),file=sys.stderr,flush=True)
            return 1
        print(result['status'],flush=True)
        return 0
    if a.action=='boundary-geometry-worker':
        if a.gate is None:p.error('--gate required for boundary-geometry-worker')
        from .boundary_geometry import run as audit_boundary_geometry
        result=audit_boundary_geometry(a.source,a.gate,a.output,max_frames=16)
        try:
            validate_boundary_geometry(result)
        except ValueError as exc:
            print(str(exc),file=sys.stderr,flush=True)
            return 1
        print(result['status'],flush=True)
        return 0
    if a.action=='face-local-worker':
        if a.gate is None:p.error('--gate required for face-local-worker')
        from .local_surface_adapter import run as constrain_local
        result=constrain_local(a.source,a.gate,a.output,**(
            {'footprint_mode':a.footprint_mode} if a.footprint_mode is not None else {}))
        print(result['status'],flush=True)
        return 0 if result.get('status')=='LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED' else 1
    if a.action=='face-mask-worker':
        from .face_mask_adapter import run as constrain_faces
        result=constrain_faces(a.source,a.output)
        print(result['status'],flush=True)
        return 0 if result.get('status')=='CANDIDATE_SURFACE_NOT_ACCEPTED' else 1
    if a.action=='support-worker':
        from .surface_support import run as audit_support
        result=audit_support(a.source,a.output)
        print(result['status'],flush=True)
        return 0
    if a.action=='export-worker':
        from .region_surface_export import export
        plan=read(a.exclusion_plan)
        if Path(plan['source_run']).resolve()!=a.source.resolve() or plan['model_id']!=a.model_id:
            raise ValueError('export request differs from frozen plan')
        export(a.source,a.output,a.model_id,expected_excluded_point_ids=plan['expected_excluded_point_ids'])
        return 0
    if a.action=='masks-worker':
        from .region_surface_masks import prepare
        prepare(a.source,a.bridge,a.output)
        return 0
    result=run(a.source,a.output,a.model_id)
    print(result['status'],flush=True)
    return 0 if result['status']=='SURFACE_GENERATED_NOT_ACCEPTED' else 1


if __name__=='__main__':
    raise SystemExit(main())
