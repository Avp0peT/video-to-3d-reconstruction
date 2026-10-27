"""One supervised COLMAP registration diagnostic on a private DB snapshot."""
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

CHILD_CODE='''import sys
from integrations.demo_pipeline.registration_probe import run_probe
r=run_probe(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4])
print(r['status'])
raise SystemExit(0 if r['status']=='ONE_NATIVE_CALL_COMPLETED_NOT_GEOMETRY_ACCEPTED' else 2)
'''

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sfm-run',type=Path,required=True,help='existing run.json')
    parser.add_argument('--input-index',type=int,required=True)
    parser.add_argument('--model-id',required=True)
    parser.add_argument('--output',type=Path,required=True,help='new directory, never reused')
    args=parser.parse_args(argv)
    from integrations.demo_pipeline.registration_probe import _binding
    from integrations.demo_pipeline.parallel_photo_audit import read_json,under
    from integrations.demo_pipeline.pipeline import _invoke,entry_code
    source=args.sfm_run.resolve(strict=True);output=args.output.resolve()
    if args.input_index<0 or output.exists() or output.is_relative_to(source.parent) or source.parent.is_relative_to(output):
        raise ValueError('nonnegative index and fresh nonoverlapping output required')
    source_run=read_json(source)
    models=[m for m in source_run['models'] if str(m['model_id'])==args.model_id]
    if len(models)!=1:raise ValueError('one explicit source model required')
    binary=under(source.parent,models[0]['binary_path'])
    paths=[source,source.parent/'database.db',source.parent/'frames_map.jsonl',
        source.parent/'logs/incremental_mapping.result.json',*sorted(binary.glob('*.bin')),
        Path(__file__),ROOT/'integrations/demo_pipeline/registration_probe.py',
        ROOT/'integrations/demo_pipeline/registration_gate.py',ROOT/'integrations/demo_pipeline/pipeline.py',
        ROOT/'integrations/demo_pipeline/supervision.py']
    bindings=[_binding(p) for p in paths]
    command=[sys.executable,'-B','-X','utf8','-c',CHILD_CODE,str(source),str(args.input_index),args.model_id,str(output/'native')]
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(schema='registration-gate-entry-v1',argv=command,cwd=str(ROOT),
        allowed_registration_calls=1,wall_seconds=90,job_commit_bytes=1<<30,
        source_bindings=bindings,historical_replay=False,native_model_writes=0)
    with (output/'PLAN.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2,allow_nan=False)
    entry=_invoke(ROOT,output,'registration',command,seconds=90,memory=1<<30)
    drift=[b['path'] for b in bindings if _binding(b['path'])!=b]
    code=entry_code(entry)
    native_path=output/'native/MANIFEST.json'
    native=read_json(native_path) if native_path.is_file() else None
    confirmed=bool(native and native.get('status')=='ONE_NATIVE_CALL_COMPLETED_NOT_GEOMETRY_ACCEPTED'
        and native.get('registration_call_count')==1 and native.get('source_unchanged') is True)
    result=dict(process_code=code,source_unchanged=not drift,source_drift=drift,
        native_manifest=str(native_path),native_completion_confirmed=confirmed,geometry_acceptance=False)
    with (output/'RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))
    return code if code else (2 if drift or not confirmed else 0)

if __name__=='__main__':raise SystemExit(main())
