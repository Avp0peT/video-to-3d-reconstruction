"""CLI for fixed-reference texture tracking and permanent loss handling."""
from pathlib import Path
import os
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline.texture_point_run import main
if __name__ == '__main__':
    raise SystemExit(main())
