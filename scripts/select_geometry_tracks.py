"""Source-bound two-sample admission and geometric evidence comparison."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
sys.dont_write_bytecode = True
from integrations.demo_pipeline.evidence_point_run import main
if __name__ == '__main__':
    raise SystemExit(main())
