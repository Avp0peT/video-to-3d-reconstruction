"""Bind a reference measurement to a prepared PLY and emit a mm scale ledger."""
from __future__ import annotations
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="2"
import argparse
from datetime import datetime,timezone
import time
from integrations.mesh_scale.ledger import create_ledger,save

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    output=args.output.resolve()
    if output.exists() or (ROOT/"outputs").resolve() not in output.parents:
        print("output must be new and strictly under project outputs",file=sys.stderr)
        return 2
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    result=dict(status="STARTED",exit_code=None,argv=[sys.executable,*sys.argv],
                started_utc=datetime.now(timezone.utc).isoformat())
    save(output/"RESULT.json",result)
    try:
        value=create_ledger(args.mesh,args.reference,args.source_run,output)
        result.update(status="PASS_REFERENCE_SCALING",exit_code=0,
                      data_kind=value["data_kind"],ledger=str(output/"scale_ledger.json"),
                      next_cad_entry=str(ROOT/"scripts/scaled_mesh_to_cad.py"),
                      next_cad_arguments=["--ledger",str(output/"scale_ledger.json"),
                                          "--output","<NEW_CAD_OUTPUT>"])
    except (ValueError,OSError,KeyError,TypeError) as error:
        result.update(status="REJECTED",exit_code=2,error=type(error).__name__+": "+str(error))
    except Exception as error:
        result.update(status="INTERNAL_ERROR",exit_code=4,error=type(error).__name__+": "+str(error))
    finally:
        result.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.monotonic()-start)
        save(output/"RESULT.json",result)
    print(str(output/"RESULT.json"))
    return result["exit_code"]

if __name__=="__main__":
    raise SystemExit(main())
