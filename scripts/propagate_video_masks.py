"""Plan or run bounded bidirectional video mask propagation from a pipeline run."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONUTF8'] = '1'
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'src'))
from integrations.video_masks.runner import execute


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-frames',type=int)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--coverage',action='store_true')
    mode.add_argument('--full-video',action='store_true')
    parser.add_argument('--reuse-run',type=Path)
    parser.add_argument('--mask-fusion',choices=['none','union','supported-holes'],default='union',
                        help='full-video后自动融合；none保留旧的选择模式')
    parser.add_argument('--max-intervals',type=int,default=6)
    parser.add_argument('--max-total-frames',type=int,default=384)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    if args.full_video:
        from integrations.video_masks.full_pipeline import execute as execute_full
        result = execute_full(args.pipeline,args.output,args.reuse_run,args.execute,args.mask_fusion)
    elif args.coverage:
        from integrations.video_masks.coverage import execute as execute_coverage
        result = execute_coverage(args.pipeline,args.output,args.max_frames if args.max_frames is not None else 96,
                                  args.execute,args.max_intervals,args.max_total_frames)
    else:
        result = execute(args.pipeline,args.output,args.max_frames if args.max_frames is not None else 48,args.execute)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    from integrations.video_masks.full_pipeline import automatic_boundary_complete
    success=result.get('status') in ('PREPARED','REVIEW_READY','NOT_APPLICABLE','FULL_TIMELINE_EXPORTED')
    required=args.full_video and args.mask_fusion=='union' and result.get('status')=='FULL_TIMELINE_EXPORTED'
    return 0 if success and automatic_boundary_complete(result,required) else 1


if __name__ == '__main__':
    raise SystemExit(main())
