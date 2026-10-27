"""Supervised single-variable two-view calibration experiment; no model export."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'

CHILD_CODE='''import sys,json
from integrations.demo_pipeline.calibration_control import run_control
r=run_control(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
print(r['status'])
raise SystemExit(0 if r['status']=='CONTROL_COMPLETED_NOT_GEOMETRY_ACCEPTED' else 2)
'''

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sfm-run',type=Path,required=True)
    parser.add_argument('--pair',type=int,nargs=2,required=True,help='two original input indices')
    parser.add_argument('--target',type=int,required=True,help='original input index to register after BA')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    from integrations.demo_pipeline.registration_probe import _binding
    from integrations.demo_pipeline.parallel_photo_audit import read_json
    from integrations.demo_pipeline.pipeline import _invoke,entry_code
    source=args.sfm_run.resolve(strict=True);output=args.output.resolve()
    if len(set(args.pair+[args.target]))!=3 or min(args.pair+[args.target])<0:
        raise ValueError('three distinct nonnegative input indices required')
    if output.exists() or output.is_relative_to(source.parent) or source.parent.is_relative_to(output):
        raise ValueError('fresh nonoverlapping output required')
    paths=[source,source.parent/'database.db',source.parent/'frames_map.jsonl',
        source.parent/'logs/incremental_mapping.result.json',Path(__file__),
        *[ROOT/'integrations/demo_pipeline'/name for name in (
            'calibration_control.py','registration_probe.py','registration_gate.py',
            'parallel_photo_audit.py','pipeline.py','supervision.py')]]
    bindings=[_binding(p) for p in paths]
    command=[sys.executable,'-B','-X','utf8','-c',CHILD_CODE,str(source),json.dumps(args.pair),str(args.target),str(output/'native')]
    plan=dict(schema='calibration-control-entry-v1',argv=command,cwd=str(ROOT),
        input_pair=args.pair,target_input_index=args.target,wall_seconds=120,job_commit_bytes=1<<30,
        source_bindings=bindings,maximum_calls=dict(initial_estimation=1,triangulation=2,bundle_adjustment=2,registration=2),
        arms=['free_radial','fixed_initial_radial'],only_arm_difference='BundleAdjustmentOptions.refine_extra_params',
        historical_replay=False,native_model_writes=0,physical_calibration_known=False)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'PLAN.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2,allow_nan=False)
    entry=_invoke(ROOT,output,'calibration_control',command,seconds=120,memory=1<<30)
    drift=[]
    for receipt in bindings:
        try:
            if _binding(receipt['path'])!=receipt:drift.append(receipt['path'])
        except OSError:drift.append(receipt['path'])
    code=entry_code(entry)
    result_path=output/'native/CONTROL.json'
    native=read_json(result_path) if result_path.is_file() else None
    completed=bool(native and native.get('status')=='CONTROL_COMPLETED_NOT_GEOMETRY_ACCEPTED'
        and native.get('source_unchanged') is True)
    result=dict(process_code=code,source_unchanged=not drift,source_drift=drift,
        native_result=str(result_path),native_completion_confirmed=completed,geometry_acceptance=False)
    with (output/'RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))
    return code if code else (2 if drift or not completed else 0)

if __name__=='__main__':raise SystemExit(main())
