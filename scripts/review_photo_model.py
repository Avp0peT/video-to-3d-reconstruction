"""One command: automatically review existing photo/video modelling evidence."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True,type=Path)
    parser.add_argument('--output',type=Path,help='fresh directory; omitted creates a dated sibling of the pipeline')
    args=parser.parse_args()
    from integrations.demo_pipeline.post_review import review
    try:result=review(args.pipeline,args.output)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f'INPUT_ERROR: {exc}',file=sys.stderr);return 2
    print(json.dumps({k:result[k] for k in ('status','reason','result_path','report_path')},ensure_ascii=False))
    return {'DIAGNOSTIC_COMPLETE':0,'INSUFFICIENT_DATA':1}.get(result['status'],2)


if __name__=='__main__':raise SystemExit(main())
