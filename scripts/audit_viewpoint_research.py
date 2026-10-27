"""Compare original photo correspondence constraints; no geometry mutation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); started=time.monotonic()
    from integrations.demo_pipeline.viewpoint_research import run
    try:
        result=run(args.prior_input,args.output)
    except Exception as exc:
        print(f'DIAGNOSTIC_ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2
    print(json.dumps(dict(status=result['status'],pairs=len(result['pairs']),
        elapsed_seconds=time.monotonic()-started,output=str(args.output.resolve())),ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
