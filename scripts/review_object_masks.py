"""Create a new read-only-input mask review package; never run inference."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='2'
sys.dont_write_bytecode=True
_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_root))


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask-manifest','--manifest',action='append',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sfm-run',type=Path)
    parser.add_argument('--model-id')
    args=parser.parse_args(argv)
    from integrations.mask_review.review import create_review
    try:
        report,code=create_review(args.mask_manifest,args.output,sfm_run=args.sfm_run,model_id=args.model_id)
    except (ValueError,OSError) as exc:
        print('REFUSED: '+str(exc),file=sys.stderr);return 2
    print(json.dumps({'status':report['status'],'counts':report['counts'],'quality_status':report['quality_status'],
                      'output':str(args.output.resolve())},ensure_ascii=False,allow_nan=False))
    return code


if __name__=='__main__':
    raise SystemExit(main())
