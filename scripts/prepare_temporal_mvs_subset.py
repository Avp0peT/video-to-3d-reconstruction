"""Prepare calibrated temporal MVS inputs without launching external tools."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import sys
from pathlib import Path
import argparse
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from integrations.mvs_surface.regional_temporal_subset import prepare

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','analysis','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--window-seconds',type=int,choices=[2],default=2)
    a=p.parse_args()
    print(json.dumps(prepare(a.source,a.analysis,a.window_seconds,a.output)))
