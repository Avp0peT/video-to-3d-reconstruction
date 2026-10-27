"""Reusable regional reconstruction to local surface entry."""
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from integrations.video_masks.region_surface import main
if __name__=='__main__':
    raise SystemExit(main())
