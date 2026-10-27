"""Saved-observation correspondence identifiability diagnostic, without refitting."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='2'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); start=time.monotonic()
    from integrations.demo_pipeline.mismatch_identifiability import experiment
    try: result=experiment(args.prior_run,args.output)
    except Exception as exc:
        print(f'DIAGNOSTIC_ERROR: {type(exc).__name__}: {exc}',file=sys.stderr); return 2
    print(json.dumps(dict(status=result['status'],selected=result['selected_count'],
        counts=result['interpretation_counts'],elapsed_seconds=time.monotonic()-start)))
    return 0

if __name__=='__main__': raise SystemExit(main())
