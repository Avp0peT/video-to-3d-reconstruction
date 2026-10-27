"""Analyze existing depth candidates in fixed temporal windows; no native work."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import sys
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from integrations.demo_pipeline.temporal_depth_report import run

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','ledger-run','region-json','mask-run','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    run(a.source,a.ledger_run,a.region_json,a.mask_run,a.output)
