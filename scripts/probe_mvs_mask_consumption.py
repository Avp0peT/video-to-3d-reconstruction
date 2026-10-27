"""Prepare then explicitly execute an independent, one-attempt native mask assay."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.mvs_surface.mask_consumption_probe import prepare,execute


def main():
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true');action.add_argument('--execute',action='store_true')
    p.add_argument('--source-plan');p.add_argument('--output');p.add_argument('--prepared');p.add_argument('--prepared-sha256')
    p.add_argument('--baseline-depth-work', help='Explicit authenticated prior depth maps; source recipe may be freshly rebound.')
    a=p.parse_args()
    if a.prepare:
        if not a.source_plan or not a.output:p.error('--prepare requires --source-plan and --output')
        result=prepare(a.source_plan,a.output,a.baseline_depth_work)
    else:
        if not a.prepared or not a.prepared_sha256:p.error('--execute requires --prepared and --prepared-sha256')
        result=execute(a.prepared,a.prepared_sha256)
    summary = result if a.prepare else {
        'status': result['status'],
        'source_unchanged': result.get('source_unchanged'),
        'native_launch_attempts': result.get('new_native_launch_attempts'),
        'elapsed_seconds': result.get('elapsed_seconds'),
        'arms': {name: {'counts': row['counts'], 'reclaimed': row['reclaimed'],
                       'execution_ok': row['execution_ok']}
                 for name, row in result['arms'].items()},
        'error': result.get('error'),
    }
    print(json.dumps(summary,ensure_ascii=False))
    return 0 if a.prepare or result['status'].startswith('REFERENCE_MASK_CONSUMPTION_VERIFIED') else 2


if __name__=='__main__':raise SystemExit(main())
