from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from integrations.video_masks.boundary_refine import prepare,execute

def main(argv=None):
    p=argparse.ArgumentParser(description='Fixed CPU trimap-constrained boundary experiment, not accepted segmentation')
    p.add_argument('action',choices=['prepare','execute','run'])
    p.add_argument('--review',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+')
    a=p.parse_args(argv)
    if a.action in ('prepare','run'):
        if a.review is None:p.error('--review required')
        prepare(a.review,a.output,a.indices)
    if a.action in ('execute','run'):
        result=execute(a.output);print(result['status'],flush=True)
        return 0 if result['status']=='BOUNDARY_REFINEMENT_CANDIDATE_NOT_ACCEPTED' else 1
    return 0

if __name__=='__main__':raise SystemExit(main())
