"""Run the automatic, bounded mask-boundary review without model inference."""
import argparse
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
def main(argv=None):
    from integrations.video_masks.auto_boundary import execute_auto_boundary,READY
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+')
    a=p.parse_args(argv)
    result=execute_auto_boundary(a.run,a.output,a.indices)
    print(result['status'],flush=True)
    return 0 if result['status']==READY else 1
if __name__=='__main__':raise SystemExit(main())
