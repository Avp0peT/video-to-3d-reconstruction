"""Experimental whole-preserving local depth search; no geometry acceptance."""
import os
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from integrations.mvs_surface.global_local_pipeline import run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment',required=True)
    parser.add_argument('--reference-image-id',type=int,required=True)
    parser.add_argument('--roi-spec',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=int,default=300)
    parser.add_argument('--method',choices=('small-patch-v1','regional-plane-v2'),default='small-patch-v1')
    args=parser.parse_args(argv)
    result=run(args.source_experiment,args.reference_image_id,args.roi_spec,args.output,seconds=args.seconds,method=args.method)
    print(json.dumps({k:result[k] for k in ('status','metrics','wall_seconds')},ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
