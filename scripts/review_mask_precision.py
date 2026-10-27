"""Independent diagnostic trimaps; never overwrite binary reconstruction masks."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from integrations.video_masks.boundary_precision import prepare,execute

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','execute','run'])
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+')
    a=p.parse_args(argv)
    if a.action in ('prepare','run'):
        if a.source is None:p.error('--source required')
        prepare(a.source,a.output,a.indices)
    if a.action in ('execute','run'):
        result=execute(a.output);print(result['status'],flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
