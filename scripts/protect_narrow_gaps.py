"""Review and preserve narrow background gaps in an existing bound fusion run."""
import argparse
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
def main(argv=None):
    from integrations.video_masks.narrow_gap_review import prepare,execute
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','execute','run'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--indices',type=int,nargs='+')
    args=parser.parse_args(argv)
    if args.action in ('prepare','run'):
        if args.source is None:parser.error('--source required')
        prepare(args.source,args.output,args.indices)
    if args.action in ('execute','run'):
        result=execute(args.output);print(result['status'],flush=True)
        return 0 if result['status']=='NARROW_GAP_CANDIDATES_NOT_ACCEPTED' and result['summary']['refused_frames']==0 else 1
    return 0
if __name__=='__main__':raise SystemExit(main())
