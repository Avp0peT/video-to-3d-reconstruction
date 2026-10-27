"""Automatic region review: generic motion windows or calibrated diagnostics.

Calibrated model mode emits separate experimental surface/proposal layers.
It never modifies a historical plan/model or declares geometry accepted.
"""
import os
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
from integrations.mvs_surface import mask_consumption_probe as probe
from integrations.mvs_surface import overlapping_domains as domains
from integrations.mvs_surface import region_dense_experiment as regional
from integrations.mvs_surface.runner import make_plan, binding, save_json


def execute_gated(probe_prepared, region_prepared):
    """A failed empirical consumption proof must never start regional native work."""
    p = probe.execute(probe_prepared['path'], probe_prepared['sha256'])
    if p.get('status') != 'REFERENCE_MASK_CONSUMPTION_VERIFIED_NOT_GEOMETRY_ACCEPTANCE':
        return dict(status='STOPPED_AT_MASK_CONSUMPTION', probe_status=p.get('status'),
                    regional_status='NOT_RUN', regional_launches=0)
    r = regional.execute(region_prepared['path'], region_prepared['sha256'])
    return dict(status=r['status'], probe_status=p['status'], regional_status=r['status'],
                regional_launches=r['new_native_launch_attempts'])


def run(source_plan, subset_root, output, baseline_depth_work=None, prepare_only=False):
    start = time.monotonic()
    source_plan = Path(source_plan).resolve(strict=True)
    subset_root = Path(subset_root).resolve(strict=True)
    output = Path(output).resolve()
    probe.need(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output)
        for p in (source_plan.parent, subset_root)), 'fresh disjoint output required')
    old = probe.load(source_plan)
    current = make_plan(**old['arguments'])
    probe.need({k:v for k,v in old.items() if k!='implementation'} ==
               {k:v for k,v in current.items() if k!='implementation'},
               'historical inputs, cameras, tools or recipe changed; only code rebinding allowed')
    output.mkdir(parents=True)
    save_json(output/'INTENT.json', dict(source=binding(source_plan), old_code=old['implementation'],
        current_code=current['implementation'], orchestrator=binding(__file__),
        scope='12-view diagnostic; candidate domains are not semantic parts', retries=0,
        geometry_adoption=False, budgets=dict(probe_seconds=600, region_seconds=900,
        prepare_domains_seconds=300, per_native_seconds=180, native_entries_maximum=7,
        per_job_committed_bytes=8<<30, per_arm_output_stop_bytes=1<<30, threads=2)), exclusive=True)
    (output/'rebound_source').mkdir()
    rebound = output/'rebound_source/plan.json'
    save_json(rebound, current, exclusive=True)
    result = dict(status='PREPARING', geometry_accepted=False, default_replaced=False)
    try:
        domains.prepare(subset_root, rebound, output/'domains')
        p = probe.prepare(rebound, output/'mask_probe',
                          baseline_depth_work or source_plan.parent/'work')
        r = regional.prepare(rebound, output/'domains/RESULT.json',
                             output/'mask_probe/RESULT.json', output/'regional')
        result.update(dict(status='PREPARED_NOT_NATIVE_EXECUTED', probe_prepared=p,
            regional_prepared=r, native_launches=0) if prepare_only else execute_gated(p, r))
    except Exception as exc:
        result.update(status='DIAGNOSTIC_ERROR', error=type(exc).__name__+': '+str(exc))
    result['elapsed_seconds'] = time.monotonic()-start
    save_json(output/'RESULT.json', result)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>自动分区诊断</h1><p>'+
        result['status']+'</p><p>不自动接收几何，不修改历史模型。未执行项以各阶段RESULT为准。</p>'+
        '<a href="domains/index.html">候选区域</a> · <a href="mask_probe/index.html">掩膜消费对照</a> · '+
        '<a href="regional/index.html">分区深度对照</a> · <a href="RESULT.json">状态</a>', encoding='utf-8')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-plan','subset-root'): parser.add_argument('--'+key)
    parser.add_argument('--output',required=True)
    parser.add_argument('--motion-manifest', help='Generic object-motion mask stage; no native 3D execution')
    parser.add_argument('--alignment-review', help='Saved multiview silhouette run to diagnose image/mask/projection alignment; no geometry edits')
    parser.add_argument('--model-from-multiview', help='Saved multiview run to track neighbors and export separate depth/silhouette mesh layers')
    parser.add_argument('--blender', action='store_true', help='Model mode only: supervised Blender creation plus independent readback')
    parser.add_argument('--raw-motion-only', action='store_true', help='Motion mode diagnostic control only')
    parser.add_argument('--baseline-depth-work')
    parser.add_argument('--global-local-experiment', help='Completed WHOLE experiment for additive local CPU depth search')
    parser.add_argument('--reference-image-id',type=int)
    parser.add_argument('--roi-spec')
    parser.add_argument('--local-method',choices=('small-patch-v1','regional-plane-v2','silhouette-interval-v1','silhouette-multiview-v2'))
    parser.add_argument('--prepare-only', action='store_true', help='Exercise real preparation and bindings without native execution')
    args=parser.parse_args(argv)
    if args.model_from_multiview:
        if any((args.alignment_review,args.motion_manifest,args.raw_motion_only,args.source_plan,args.subset_root,args.baseline_depth_work,args.global_local_experiment,args.roi_spec,args.local_method,args.prepare_only)) or args.reference_image_id is not None:
            parser.error('model mode cannot be mixed with diagnostic or input selection modes')
        from integrations.mvs_surface.global_local_model_pipeline import run as model_run
        result=model_run(args.model_from_multiview,args.output,blender=args.blender)
        print(json.dumps({k:result[k] for k in ('status','new_meshes','new_triangulated_points','wall_seconds')},ensure_ascii=False))
        return 0 if result['new_meshes'] and (not args.blender or result['blender']['status']=='SUCCESS') else 1
    if args.blender: parser.error('--blender requires --model-from-multiview')
    if args.alignment_review:
        if any((args.motion_manifest,args.raw_motion_only,args.source_plan,args.subset_root,args.baseline_depth_work,args.global_local_experiment,args.roi_spec,args.local_method,args.prepare_only)) or args.reference_image_id is not None:
            parser.error('alignment review cannot be mixed with reconstruction or ROI options')
        from integrations.mvs_surface.boundary_alignment_pipeline import run as alignment_run
        result=alignment_run(args.alignment_review,args.output)
        print(json.dumps({k:result[k] for k in ('status','wall_seconds')},ensure_ascii=False))
        return 0
    if args.global_local_experiment:
        if args.motion_manifest or args.raw_motion_only or args.source_plan or args.subset_root or args.baseline_depth_work or args.prepare_only:
            parser.error('global/local mode cannot be mixed with other processing modes')
        if args.reference_image_id is None or not args.roi_spec:
            parser.error('global/local mode requires --reference-image-id and --roi-spec')
        if args.local_method=='silhouette-multiview-v2':
            from integrations.mvs_surface.silhouette_multiview_pipeline import run as local_run
            kwargs={}
        elif args.local_method=='silhouette-interval-v1':
            from integrations.mvs_surface.silhouette_pipeline import run as local_run
            kwargs={}
        else:
            from integrations.mvs_surface.global_local_pipeline import run as local_run
            kwargs={} if args.local_method is None else {'method':args.local_method}
        result=local_run(args.global_local_experiment,args.reference_image_id,args.roi_spec,args.output,**kwargs)
        print(json.dumps({k:result[k] for k in ('status','metrics','wall_seconds')},ensure_ascii=False))
        return 0
    if args.reference_image_id is not None or args.roi_spec or args.local_method:
        parser.error('local ROI options require --global-local-experiment')
    if args.motion_manifest:
        if args.source_plan or args.subset_root or args.baseline_depth_work or args.prepare_only:
            parser.error('--motion-manifest cannot be mixed with calibrated/native options')
        from integrations.video_masks.motion_part_pipeline import run as motion_run
        result=motion_run(args.motion_manifest,args.output,coherence=not args.raw_motion_only)
        print(json.dumps({k:result[k] for k in ('status','frame_count','window_count')},ensure_ascii=False))
        return 0
    if args.raw_motion_only or not args.source_plan or not args.subset_root:
        parser.error('provide --motion-manifest OR both --source-plan and --subset-root')
    values=vars(args); values.pop('motion_manifest'); values.pop('raw_motion_only')
    for key in ('global_local_experiment','reference_image_id','roi_spec','local_method','alignment_review','model_from_multiview','blender'):values.pop(key)
    result=run(**values); print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status'] in ('REGIONAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED','PREPARED_NOT_NATIVE_EXECUTED') else 2


if __name__ == '__main__':raise SystemExit(main())
