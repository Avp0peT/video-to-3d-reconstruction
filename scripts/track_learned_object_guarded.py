"""Guarded v2 learned point entry point."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from integrations.demo_pipeline.learned_point_guarded import main
if __name__ == '__main__':
    raise SystemExit(main())
