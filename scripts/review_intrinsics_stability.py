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
from integrations.demo_pipeline.intrinsics_stability import audit_intrinsics
r=audit_intrinsics(sys.argv[1],sys.argv[2],pipeline=sys.argv[3] or None)
print(r['status'])
raise SystemExit(0 if r.get('status')=='INTRINSICS_DIAGNOSTIC_COMPLETE' and r.get('source_unchanged') is True else 2)
"""


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sfm-run',type=Path,required=True)
    parser.add_argument('--pipeline',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    from integrations.demo_pipeline.registration_probe import _binding
    from integrations.demo_pipeline.pipeline import _invoke,entry_code
    source=args.sfm_run.resolve(strict=True);output=args.output.resolve()
    if source.is_dir():source=source/'run.json'
    pipeline=args.pipeline.resolve(strict=True) if args.pipeline else None
    protected=[source.parent]+([pipeline] if pipeline else [])
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('fresh nonoverlapping output required')
    paths=[source,source.parent/'database.db',source.parent/'frames_map.jsonl',Path(__file__),
        *[ROOT/'integrations/demo_pipeline'/n for n in ('intrinsics_stability.py','focal_profile.py','parallel_photo_audit.py','pipeline.py','supervision.py')]]
    if pipeline:
        paths.extend(pipeline/n for n in ('pipeline.json','input/video/RESULT.json','input/video/timeline.json') if (pipeline/n).is_file())
    bindings=[_binding(p) for p in paths]
    command=[sys.executable,'-B','-X','utf8','-c',CHILD_CODE,str(source),str(output/'report'),str(pipeline) if pipeline else '']
    plan=dict(schema='intrinsics-stability-entry-v1',argv=command,cwd=str(ROOT),wall_seconds=120,
        job_commit_bytes=1<<30,threads=2,source_bindings=bindings,report_output=str(output/'report'),
        native_model_executions=0,calibration_accepted=False)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'PLAN.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2,allow_nan=False)
    entry=_invoke(ROOT,output,'intrinsics_stability',command,seconds=120,memory=1<<30)
    drift=[]
    for receipt in bindings:
        try:
            if _binding(receipt['path'])!=receipt:drift.append(receipt['path'])
        except OSError:drift.append(receipt['path'])
    code=entry_code(entry);result_path=output/'report/INTRINSICS.json';detail=None;error=None
    try:
        if result_path.is_file():detail=json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:error=f'{type(exc).__name__}: {exc}'
    completed=bool(isinstance(detail,dict) and detail.get('status')=='INTRINSICS_DIAGNOSTIC_COMPLETE'
        and detail.get('source_unchanged') is True and (output/'report/index.html').is_file())
    result=dict(process_code=code,source_unchanged=not drift,source_drift=drift,
        diagnostic_result=str(result_path),diagnostic_completion_confirmed=completed,
        result_read_error=error,calibration_accepted=False,native_model_executions=0)
    with (output/'RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))
    return code if code else (2 if drift or not completed else 0)

if __name__=='__main__':raise SystemExit(main())
