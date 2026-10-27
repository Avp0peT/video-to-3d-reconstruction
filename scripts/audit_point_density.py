"""Read existing regional export and compare bounded mask-local sampling."""
import argparse
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    from integrations.video_masks.density_report import audit_export
    result=audit_export(args.export,args.output)
    print(result['status'])
    # 0 means audit completed. The business coverage verdict is separate.
    return 0


if __name__=='__main__':
    raise SystemExit(main())
