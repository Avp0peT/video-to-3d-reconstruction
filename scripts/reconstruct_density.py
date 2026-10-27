"""Replay existing region inputs with the frozen mask-local density policy."""
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from integrations.video_masks.density_reconstruction import main
if __name__=='__main__':raise SystemExit(main())
