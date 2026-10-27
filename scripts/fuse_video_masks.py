"""Fuse existing SAM3 image/tracking masks and publish a new full video."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'src'))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['union','supported-holes'],default='union')
    args=parser.parse_args(argv)
    from integrations.video_masks.fusion import execute
    from integrations.video_masks.full_pipeline import automatic_boundary_complete
    result=execute(args.run,args.output,args.mode)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    success=result['status']=='FULL_TIMELINE_FUSED_EXPORTED'
    return 0 if success and automatic_boundary_complete(result,args.mode=='union') else 1

if __name__=='__main__':
    raise SystemExit(main())
