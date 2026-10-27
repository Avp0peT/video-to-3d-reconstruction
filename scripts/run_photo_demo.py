"""One command for photos/video -> segmentation, reconstructed surface and Blender."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from integrations.demo_pipeline.pipeline import Options,plan,run_pipeline,report_existing,read,source_info


def deployment_options(args):
    """A deployment record supplies defaults only after a fresh bound audit."""
    config = args.pop('deployment_config', None)
    digest = args.pop('deployment_config_sha256', None)
    if (config is None) != (digest is None):
        raise ValueError('--deployment-config and --deployment-config-sha256 must be supplied together')
    if config is None:
        return args
    from integrations.demo_pipeline.deployment import options_overrides
    overrides = options_overrides(config, digest, expected_input_kind=source_info(args['source'])['kind'])
    for key, value in overrides.items():
        if key in args and args[key] is not None:
            explicit = args[key]
            if isinstance(value, Path):
                explicit = Path(explicit).expanduser().resolve()
            if explicit != value:
                raise ValueError('explicit option conflicts with bound deployment: ' + key)
        args[key] = value
    return args


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('run','plan'):
        p=sub.add_parser(command)
        p.add_argument('--source',type=Path,required=True)
        p.add_argument('--prompt',required=True,help='目标物体，如 beverage can / cup / toy car')
        p.add_argument('--output',type=Path)
        p.add_argument('--object-id',default='object-01')
        p.add_argument('--frames',type=int,default=24,help='视频最多抽取帧数，2–96；照片读取全目录')
        p.add_argument('--all-frames',action='store_true',help='逐帧处理整个视频，明确上限192帧；超限拒绝，不静默截断。SAM每批最多96帧。')
        p.add_argument('--point-selection',choices=['native','rcover-v1'],default='native',help='SAM3 mask 内使用原生特征，或先做覆盖选点再匹配重建（实验）')
        recovery=p.add_mutually_exclusive_group()
        recovery.add_argument('--auto-recover',dest='automatic_recovery',action='store_true',help='自动执行至多一次未匹配帧提示词恢复及一次失效相机模型对照（默认开启）')
        recovery.add_argument('--no-auto-recover',dest='automatic_recovery',action='store_false')
        p.set_defaults(automatic_recovery=True)
        p.add_argument('--recovery-prompt',help='同一目标的备用描述；仅用于尚未唯一识别的帧，默认在原提示词前加 the')
        p.add_argument('--ffmpeg');p.add_argument('--ffprobe')
        p.add_argument('--python',type=Path);p.add_argument('--sam3-python',type=Path)
        p.add_argument('--sam3-package',type=Path);p.add_argument('--checkpoint',type=Path,default=argparse.SUPPRESS)
        p.add_argument('--target',choices=['blender','report','cad'],default=argparse.SUPPRESS,help='blender 生成表面与 Blender；cad 继续验证 FCStd/STEP；report 仅分割和稀疏报告')
        p.add_argument('--deployment-config',type=Path,help='已创建的本地部署配置；使用前重新核验绑定')
        p.add_argument('--deployment-config-sha256',help='部署配置的预期 SHA-256')
        p.add_argument('--freecad-python',type=Path,help='CAD target 使用已有 FreeCAD Python')
        p.add_argument('--git-exe',type=Path,help='SAM3源码核验使用已有Git绝对路径')
        p.add_argument('--model-id',help='明确选择 SfM 分量；默认选择注册视图最多的分量并记录其余分量')
        p.add_argument('--mvs-tool-root',type=Path)
        p.add_argument('--blender-exe',type=Path)
        diagnostic=p.add_mutually_exclusive_group()
        diagnostic.add_argument('--view-geometry-audit',dest='view_geometry_audit',action='store_true',help='自动选视图与几何诊断（默认开启；非精度验收）')
        diagnostic.add_argument('--no-view-geometry-audit',dest='view_geometry_audit',action='store_false')
        p.set_defaults(view_geometry_audit=True)
        research=p.add_mutually_exclusive_group()
        research.add_argument('--research-code6',dest='research_code6',action='store_true',help='生成固定六点研究表示（默认开启）')
        research.add_argument('--no-research-code6',dest='research_code6',action='store_false',help='关闭独立研究侧支')
        p.set_defaults(research_code6=True)
        post=p.add_mutually_exclusive_group()
        post.add_argument('--post-review',dest='post_review',action='store_true',help='自动离线复核与诊断报告（默认开启；非建模验收）')
        post.add_argument('--no-post-review',dest='post_review',action='store_false')
        p.set_defaults(post_review=True)
    for command in ('report','status'):
        p=sub.add_parser(command);p.add_argument('--run',type=Path,required=True)
    p=sub.add_parser('propagate-masks',help='自动选连续短片段、双向传播并生成复核报告；不覆盖原掩膜')
    p.add_argument('--pipeline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-frames',type=int,help='每区间上限；短片默认48，多区间默认96')
    maskmode=p.add_mutually_exclusive_group()
    maskmode.add_argument('--coverage',action='store_true',help='自动选择多个漏检区间并生成完整覆盖账本')
    maskmode.add_argument('--full-video',action='store_true',help='覆盖整个时间轴并导出掩膜/叠加视频')
    p.add_argument('--reuse-run',type=Path,help='完整视频模式可复用已核验的多区间结果')
    p.add_argument('--mask-fusion',choices=['none','union','supported-holes'],default='union',help='完整视频模式自动融合，默认并集；none保留旧选择模式')
    p.add_argument('--max-intervals',type=int,default=6)
    p.add_argument('--max-total-frames',type=int,default=384)
    p.add_argument('--execute',action='store_true',help='执行一次真实视频跟踪；省略则仅生成计划')
    p=sub.add_parser('fuse-masks',help='复用已完成的图像/视频掩膜，在新目录自动融合并导出视频，不运行模型')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['union','supported-holes'],default='union')
    p=sub.add_parser('review-mask-boundaries',help='在新目录复核已有并集掩膜边界；默认至多16帧，不自动推广候选')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+')
    p=sub.add_parser('reconstruct-masks',help='复用已绑定的全视频并集掩膜，自动完成新SfM、MVS、Blender及逐步诊断')
    p.add_argument('--run',type=Path,required=True,help='已完成融合导出的目录')
    p.add_argument('--output',type=Path,required=True,help='新的建模与诊断目录')
    p=sub.add_parser('report-mask-reconstruction',help='只刷新融合掩膜建模报告，不重跑任何求解器')
    p.add_argument('--run',type=Path,required=True)
    p=sub.add_parser('continue-mask-surface',help='复用已核验的融合掩膜/相机/稀疏模型，在新目录继续表面及Blender')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('reconstruct-regions',help='完整视频掩膜与既有SfM：区域跟踪、几何和对比；物体类别无关')
    p.add_argument('--mask-run',type=Path,required=True)
    p.add_argument('--sfm-run',type=Path,required=True)
    p.add_argument('--baseline-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--no-refine',action='store_true',help='只完成初次区域重建；默认自动继续反向与一次补帧')
    p=sub.add_parser('refine-object',help='自动反向补充并按当前分量补入视频中间帧')
    p.add_argument('--run',type=Path,required=True,help='已完成的区域重建目录')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-geometry',help='仅一个几何结果时自动识别；多结果须明确目录')
    p.add_argument('--prepare-only',action='store_true')
    p=sub.add_parser('execute-object-refinement',help='执行已冻结且未尝试的通用区域增强计划')
    p.add_argument('--run',type=Path,required=True)
    p=sub.add_parser('surface-regions',help='单个区域重建分量自动生成局部表面；保留孔洞，不合并分量')
    p.add_argument('--run',type=Path,required=True,help='含 input/PLAN 和 geometry/RESULT 的区域重建目录')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-id',type=int,help='默认注册视角最多的分量')
    p=sub.add_parser('audit-point-density',help='现有区域表面导出：局部取点密度与补点对比，不重跑建模')
    p.add_argument('--run',type=Path,required=True,help='包含 export/EXPORT.json 的区域表面目录')
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('rebuild-density-model',help='同一完整区域时间轴：密度优先重跟踪、SfM、表面及Blender')
    p.add_argument('--run',type=Path,required=True,help='含 input/PLAN.json 和 geometry/RESULT.json 的旧区域目录')
    p.add_argument('--output',type=Path,required=True)
    for command in ('plan-resume','resume'):
        p=sub.add_parser(command, help='复用已提交上游，仅在新目录继续表面/Blender')
        p.add_argument('--run',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--python',type=Path)
        p.add_argument('--mvs-tool-root',type=Path)
        p.add_argument('--blender-exe',type=Path)
        p.add_argument('--model-id')
    for command in ('plan-cad','continue-cad'):
        p=sub.add_parser(command, help='从已核验统一任务或表面续跑，在新目录交付CAD')
        p.add_argument('--run',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--python',type=Path)
        p.add_argument('--freecad-python',type=Path)
    args=vars(parser.parse_args(argv));command=args.pop('command')
    try:
        if command=='rebuild-density-model':
            from integrations.video_masks.density_reconstruction import prepare,execute
            prepare(args['run'],args['output'])
            result=execute(args['output'])
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result['status']=='SURFACE_GENERATED_NOT_ACCEPTED' else 1
        if command=='audit-point-density':
            from integrations.video_masks.density_report import audit_export
            result=audit_export(args['run']/'export',args['output'])
            print(json.dumps({k:result[k] for k in ('status','frame_count','mean_frame_met_area_fraction')},ensure_ascii=False))
            return 0
        if command=='surface-regions':
            from integrations.video_masks.region_surface import run as regional_surface
            result=regional_surface(args['run'],args['output'],args['model_id'])
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result['status']=='SURFACE_GENERATED_NOT_ACCEPTED' else 1
        if command=='reconstruct-regions':
            from integrations.video_masks import region_experiment as regions
            result=regions.prepare(args['mask_run'],args['sfm_run'],args['baseline_report'],args['output'])
            if not args['prepare_only']:result=regions.execute(args['output'])
            success=args['prepare_only'] or result.get('status')=='EXPERIMENT_COMPLETE_NOT_ACCEPTED'
            if success and not args['prepare_only'] and not args['no_refine']:
                from integrations.video_masks import region_refinement as refinement
                from integrations.demo_pipeline.pipeline import save
                models=read(args['output']/'geometry/RESULT.json').get('models',[])
                if models:
                    continued=refinement.run(args['output'],args['output']/'refinement','geometry')
                    success=continued['status']=='REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED'
                else:
                    continued={'status':'NOT_APPLICABLE_NO_RECONSTRUCTION','geometry_accepted':False}
                save(args['output']/'AUTOMATIC_REFINEMENT.json',continued)
                result=dict(result,automatic_refinement=continued)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if success else 1
        if command in ('refine-object','execute-object-refinement'):
            from integrations.video_masks import region_refinement as refinement
            if command=='refine-object':
                result=refinement.prepare(args['run'],args['output'],args['source_geometry'])
                if not args['prepare_only']:result=refinement.execute(args['output'])
            else:result=refinement.execute(args['run'])
            print(json.dumps(result,ensure_ascii=False,indent=2))
            prepared=command=='refine-object' and args['prepare_only'] and result['status']=='PREPARED'
            return 0 if prepared or result['status']=='REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED' else 1
        if command=='continue-mask-surface':
            from integrations.video_masks.reconstruction import continue_surface
            result=continue_surface(args['run'],args['output'])
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result['status']=='SURFACE_GENERATED_NOT_ACCEPTED' else 1
        if command=='reconstruct-masks':
            from integrations.video_masks.reconstruction import execute
            result=execute(args['run'],args['output'])
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result['status']=='SURFACE_GENERATED_NOT_ACCEPTED' else 1
        if command=='report-mask-reconstruction':
            from integrations.video_masks.reconstruction import render
            print(json.dumps(render(args['run']),ensure_ascii=False,indent=2));return 0
        if command=='review-mask-boundaries':
            from integrations.video_masks.auto_boundary import execute_auto_boundary
            result=execute_auto_boundary(args['run'],args['output'],indices=args['indices'],max_frames=16)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result.get('status')=='AUTOMATIC_BOUNDARY_REVIEW_READY_NOT_ACCEPTED' else 1
        if command=='fuse-masks':
            from integrations.video_masks.fusion import execute
            from integrations.video_masks.full_pipeline import automatic_boundary_complete
            result=execute(args['run'],args['output'],args['mode'])
            print(json.dumps(result,ensure_ascii=False,indent=2))
            success=result['status']=='FULL_TIMELINE_FUSED_EXPORTED'
            return 0 if success and automatic_boundary_complete(result,args['mode']=='union') else 1
        if command=='propagate-masks':
            if args['full_video']:
                from integrations.video_masks.full_pipeline import execute
                result=execute(args['pipeline'],args['output'],args['reuse_run'],args['execute'],args['mask_fusion'])
            elif args['coverage']:
                from integrations.video_masks.coverage import execute
                result=execute(args['pipeline'],args['output'],args['max_frames'] if args['max_frames'] is not None else 96,
                               args['execute'],args['max_intervals'],args['max_total_frames'])
            else:
                from integrations.video_masks.runner import execute
                result=execute(args['pipeline'],args['output'],args['max_frames'] if args['max_frames'] is not None else 48,args['execute'])
            print(json.dumps({k:v for k,v in result.items() if k not in ('review','summary')},ensure_ascii=False,indent=2))
            from integrations.video_masks.full_pipeline import automatic_boundary_complete
            success=result.get('status') in ('PREPARED','REVIEW_READY','NOT_APPLICABLE','FULL_TIMELINE_EXPORTED')
            required=args['full_video'] and args['mask_fusion']=='union' and result.get('status')=='FULL_TIMELINE_EXPORTED'
            return 0 if success and automatic_boundary_complete(result,required) else 1
        if command in ('plan-cad','continue-cad'):
            from integrations.demo_pipeline import cad
            frozen=cad.plan(**args)
            if command=='plan-cad':
                print(json.dumps(frozen,ensure_ascii=False,indent=2));return 0
            result=cad.execute(frozen)
            print(json.dumps(result,ensure_ascii=False,indent=2));return result['exit_code']
        if command in ('plan-resume','resume'):
            from integrations.demo_pipeline import resume as continuation
            from integrations.demo_pipeline.pipeline import LIGHT_REL,MVS_REL
            from integrations.blender_export.export import find_blender
            frozen = continuation.plan(args['run'],args['output'],
                python=args['python'] or ROOT/LIGHT_REL,
                tool_root=args['mvs_tool_root'] or ROOT/MVS_REL,
                blender_exe=find_blender(args['blender_exe']),model_id=args['model_id'])
            if command=='plan-resume':
                print(json.dumps(frozen,ensure_ascii=False,indent=2));return 0
            result=continuation.execute(frozen)
            print(json.dumps(result,ensure_ascii=False,indent=2));return result['exit_code']
        if command in ('plan','run'):args=deployment_options(args)
        if command=='plan':print(json.dumps(plan(Options(**args)),ensure_ascii=False,indent=2));return 0
        if command=='status':
            record=args['run']/'pipeline.json'
            if not record.is_file():record=args['run']/'CONTINUATION.json'
            print(json.dumps(read(record),ensure_ascii=False,indent=2));return 0
        if command=='report':print(report_existing(args['run']));return 0
        output,code=run_pipeline(Options(**args))
        print(json.dumps({'run':str(output),'exit_code':code,'report':str(output/'report/index.html')},ensure_ascii=False));return code
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f'INPUT_ERROR: {exc}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
