"""Supervised read-only intrinsics diagnostic; no calibration acceptance."""
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

CHILD_CODE="""import sys
from integrations.demo_pipeline.intrinsics_sensitivity import audit_sensitivity
r=audit_sensitivity(sys.argv[1],sys.argv[2])
print(r['status'])
raise SystemExit(0 if r.get('status')=='INTRINSICS_SENSITIVITY_COMPLETE' and r.get('source_unchanged') is True else 2)
"""


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intrinsics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    from integrations.demo_pipeline.registration_probe import _binding
    from integrations.demo_pipeline.pipeline import _invoke,entry_code
    source=args.intrinsics.resolve(strict=True);output=args.output.resolve()
    def reject_constant(value):raise ValueError('nonfinite JSON constant: '+value)
    before=_binding(source)
    record=json.loads(source.read_text(encoding='utf-8-sig'),parse_constant=reject_constant)
    if not isinstance(record,dict) or not isinstance(record.get('sfm_run'),str):
        raise ValueError('intrinsics source must identify sfm_run')
    sfm=Path(record['sfm_run']).resolve(strict=True)
    sfm_root=sfm if sfm.is_dir() else sfm.parent
    protected=[source.parent,sfm_root]
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('fresh nonoverlapping output required')
    if _binding(source)!=before:raise ValueError('intrinsics source drift during read')
    paths=[source,Path(__file__),
        *[ROOT/'integrations/demo_pipeline'/n for n in ('intrinsics_sensitivity.py','intrinsics_sensitivity_math.py',
            'focal_profile.py','intrinsics_stability.py','parallel_photo_audit.py','pipeline.py','supervision.py')]]
    bindings=[_binding(p) for p in paths]
    command=[sys.executable,'-B','-X','utf8','-c',CHILD_CODE,str(source),str(output/'report')]
    plan=dict(schema='intrinsics-sensitivity-entry-v1',argv=command,cwd=str(ROOT),wall_seconds=120,
        job_commit_bytes=1<<30,threads=2,source_bindings=bindings,report_output=str(output/'report'),
        native_model_executions=0,calibration_accepted=False)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'PLAN.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2,allow_nan=False)
    entry=_invoke(ROOT,output,'intrinsics_sensitivity',command,seconds=120,memory=1<<30)
    drift=[]
    for receipt in bindings:
        try:
            if _binding(receipt['path'])!=receipt:drift.append(receipt['path'])
        except OSError:drift.append(receipt['path'])
    code=entry_code(entry);result_path=output/'report/SENSITIVITY.json';detail=None;error=None
    try:
        if result_path.is_file():detail=json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:error=f'{type(exc).__name__}: {exc}'
    completed=bool(isinstance(detail,dict) and detail.get('status')=='INTRINSICS_SENSITIVITY_COMPLETE'
        and detail.get('source_unchanged') is True and (output/'report/index.html').is_file())
    result=dict(process_code=code,source_unchanged=not drift,source_drift=drift,
        diagnostic_result=str(result_path),diagnostic_completion_confirmed=completed,
        result_read_error=error,calibration_accepted=False,native_model_executions=0)
    with (output/'RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))
    return code if code else (2 if drift or not completed else 0)

if __name__=='__main__':raise SystemExit(main())
