import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = '2'
import sys
from pathlib import Path
import argparse
sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from integrations.mvs_surface.overlapping_domains import prepare
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Prepare candidate domains; no native execution')
    for k in ('subset-root', 'plan', 'output'): p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args(); prepare(a.subset_root, a.plan, a.output)
