"""Prepare a photo directory or a local video for the 3D reconstruction workflow."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]

def main(argv: list[str] | None = None) -> int:
    from integrations.object_input.session import prepare_photos,prepare_video,adopt_video,verify_input
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='action',required=True)
    for action in ('photos','video'):
        p=subs.add_parser(action);p.add_argument('--source',required=True);p.add_argument('--dataset-id',required=True)
        p.add_argument('--data-kind',choices=('real','synthetic'),required=True);p.add_argument('--output',required=True)
        if action=='video':
            p.add_argument('--all-frames',action='store_true');p.add_argument('--count',type=int,default=48);p.add_argument('--ffmpeg');p.add_argument('--ffprobe')
    p=subs.add_parser('adopt-video');p.add_argument('--video-run',required=True);p.add_argument('--output',required=True)
    p=subs.add_parser('inspect');p.add_argument('--run',required=True)
    args=vars(parser.parse_args(argv));action=args.pop('action')
    try:
        if action=='inspect':result=verify_input(args['run']);code=0
        elif action=='adopt-video':result=adopt_video(**args);code=result['exit_code']
        else:
            source=args.pop('source')
            result=(prepare_photos if action=='photos' else prepare_video)(source,**args);code=result['exit_code']
        print(json.dumps(result if action=='inspect' else {k:result.get(k) for k in ('status','exit_code','output','error')},ensure_ascii=False))
        return code
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f'INPUT_ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
