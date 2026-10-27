"""Prepare/execute frozen whole-vs-three-regions dense-only diagnostic."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.mvs_surface.region_dense_experiment import prepare,execute


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    for name in ('source-plan','regions-result','probe-result','output','prepared','prepared-sha256'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.prepare:
        if not all((a.source_plan,a.regions_result,a.probe_result,a.output)):p.error('prepare needs source-plan, regions-result, probe-result, output')
        r=prepare(a.source_plan,a.regions_result,a.probe_result,a.output);print(json.dumps(r));return 0
    if not all((a.prepared,a.prepared_sha256)):p.error('execute needs prepared and prepared-sha256')
    r=execute(a.prepared,a.prepared_sha256);print(json.dumps(dict(status=r['status'],views=len(r['views']),launch_attempts=r['new_native_launch_attempts'],source_unchanged=r['source_unchanged'])))
    return 0 if r['status']=='REGIONAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ACCEPTED' else 2


if __name__=='__main__':raise SystemExit(main())
