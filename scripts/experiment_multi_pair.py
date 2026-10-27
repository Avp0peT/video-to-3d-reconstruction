"""Bounded multi-pair engineering ablation using existing photo observations."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();started=time.monotonic()
    from integrations.demo_pipeline.multi_pair_ablation import experiment
    try:result=experiment(args.prior_input,args.output)
    except Exception as exc:
        print(f'EXPERIMENT_ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);return 2
    print(json.dumps(dict(status=result['status'],groups=result['group_count'],counts=result['counts'],
                         elapsed_seconds=time.monotonic()-started),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
