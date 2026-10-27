"""Read-only automatic view geometry diagnostic for an existing native SfM run."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sfm-run',type=Path,required=True)
    p.add_argument('--model-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    from integrations.demo_pipeline.view_geometry import audit
    try:
        result=audit(args.sfm_run,args.model_id,args.output)
    except Exception as exc:
        print(f'DIAGNOSTIC_ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2
    print(json.dumps(dict(status=result['status'],counts=result['counts'],geometry_acceptance=False),ensure_ascii=False))
    return 0  # Successful diagnostic execution is not geometry acceptance.


if __name__=='__main__':raise SystemExit(main())
