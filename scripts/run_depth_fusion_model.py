"""Automatically fuse authenticated native depth views and export a reviewed surface."""
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time
from integrations.mvs_surface.depth_fusion_pipeline import run
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-model',required=True);p.add_argument('--output',required=True);p.add_argument('--full-depth-parent',help='Include every bound calibrated parent view and available historical native depth');p.add_argument('--blender',action='store_true');p.add_argument('--allow-inferred-completion',action='store_true',help='Explicitly add a separate silhouette-inferred completion layer; requires full parent and normal finalization');p.add_argument('--raw-only',action='store_true',help='Diagnostic TSDF only; not the default model delivery');p.add_argument('--max-axis',type=int,default=160);p.add_argument('--min-views',type=int,default=2);p.add_argument('--truncation-voxels',type=float,default=3);p.add_argument('--seconds',type=int,default=720);a=p.parse_args(argv)
    if a.allow_inferred_completion and (not a.full_depth_parent or a.raw_only):p.error('--allow-inferred-completion requires --full-depth-parent and cannot use --raw-only')
    started=time.monotonic();destination=Path(a.output).resolve()
    if destination.exists():p.error('fresh output root required')
    options={}
    if a.full_depth_parent:
        if a.raw_only:p.error('--full-depth-parent requires the normal raw/final layout')
        from integrations.mvs_surface.full_depth_inputs import prepare_full_inputs
        options['full_source']=destination/'FULL_INPUTS.json'
        def preparation_check():
            if time.monotonic()-started>a.seconds:raise TimeoutError('shared full-source preparation deadline')
        prepare_full_inputs(a.full_depth_parent,a.source_model,options['full_source'],preparation_check)
    remaining=int(a.seconds-(time.monotonic()-started)) if a.full_depth_parent else a.seconds
    if remaining<1:raise TimeoutError('shared automatic pipeline deadline exceeded')
    r=run(a.source_model,destination if a.raw_only else destination/'raw',blender=a.blender and a.raw_only,max_axis=a.max_axis,min_views=a.min_views,truncation_voxels=a.truncation_voxels,seconds=remaining,**options)
    if not a.raw_only:
        from integrations.mvs_surface.depth_fusion_finalize import finalize
        remaining=int(a.seconds-(time.monotonic()-started))
        if remaining<1:raise TimeoutError('shared automatic pipeline deadline exceeded')
        r=finalize(destination/'raw',destination/'final',blender=a.blender,seconds=remaining)
        observed_status=r['status']
        blender_ok=not a.blender or r['blender']['status']=='SUCCESS'
        completion_link=''
        if a.allow_inferred_completion:
            from integrations.mvs_surface.object_completion_pipeline import run as complete_object
            remaining=int(a.seconds-(time.monotonic()-started))
            if remaining<1:raise TimeoutError('shared automatic pipeline deadline exceeded before inferred completion')
            r=complete_object(destination/'FULL_INPUTS.json',destination/'raw',destination/'completion',blender=False,seconds=remaining)
            candidate_status=r['status']
            from integrations.mvs_surface.completion_delivery import finalize_completion
            remaining=int(a.seconds-(time.monotonic()-started))
            if remaining<1:raise TimeoutError('shared automatic pipeline deadline exceeded before closed model delivery')
            r=finalize_completion(destination/'completion',destination/'completed_model',blender=a.blender,seconds=remaining)
            closed_ok=r.get('closed_estimated_outer_model') is True
            blender_ok=blender_ok and (not a.blender or r['blender']['status']=='SUCCESS')
            from integrations.mvs_surface.strict_completion_comparison import run as compare_completion
            remaining=int(a.seconds-(time.monotonic()-started))
            if remaining<1:raise TimeoutError('shared automatic pipeline deadline exceeded before strict completion comparison')
            comparison=compare_completion(destination/'completion',destination/'strict_comparison',baseline_delivery=destination/'completed_model',blender=a.blender,seconds=remaining)
            comparison_ok=comparison.get('status')=='STRICT_COMPARISON_DELIVERED'
            comparison_blender=comparison['blender']
            absent_strict=(comparison_blender.get('status')=='NOT_APPLICABLE' and comparison_blender.get('reason')=='strict_envelope_absent' and comparison.get('strict_closed') is False)
            blender_ok=blender_ok and (not a.blender or comparison_blender.get('status')=='SUCCESS' or absent_strict)
            completion_link='<p><a href="strict_comparison/index.html">推荐：严格审核与推算前后对照 / Strict completion comparison</a></p><p><a href="completed_model/index.html">封闭推算外形模型 / Closed estimated outer model</a></p><p><a href="completion/index.html">完整候选与分层诊断 / Inferred candidate diagnostics</a></p><p>推算不是新增观测或完整几何证明 / Inference is not an observation or completeness guarantee.</p>'
        (destination/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Depth fusion delivery</title>'+completion_link+'<p><a href="final/index.html">审核后模型 / Audited model</a> · <a href="raw/index.html">原始融合诊断 / Raw diagnostic</a></p>',encoding='utf8')
    summary=dict(status=r['status'],wall_seconds=time.monotonic()-started)
    if a.allow_inferred_completion:summary.update(observed_status=observed_status,inferred_candidate_status=candidate_status,inferred_completion_enabled=True,closed_estimated_outer_model=closed_ok,strict_comparison_status=comparison.get('status'),strict_closed=comparison.get('strict_closed'))
    print(json.dumps(summary,ensure_ascii=False))
    return 0 if ((blender_ok and (not a.allow_inferred_completion or (closed_ok and comparison_ok))) if not a.raw_only else (not a.blender or r['blender']['status']=='SUCCESS')) else 1
if __name__=='__main__':raise SystemExit(main())
