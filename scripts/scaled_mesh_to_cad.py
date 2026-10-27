"""Verify bound scale/measurements before the unchanged FreeCAD converter."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
sys.dont_write_bytecode=True
import argparse
from datetime import datetime,timezone
from integrations.mesh_scale.ledger import verify_ledger,file_binding,verify_binding,save
import mesh_to_cad as cad

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-faces",type=cad.positive_int,default=10000)
    args=p.parse_args(argv)
    output=args.output.resolve()
    if output.exists() or (ROOT/"outputs").resolve() not in output.parents:
        print("output must be new and under project outputs",file=sys.stderr)
        return 2
    record=dict(status="STARTED",exit_code=None,native_conversion_started=False,
                started_utc=datetime.now(timezone.utc).isoformat())
    try:
        bound=file_binding(args.ledger,1024*1024)
        value=verify_ledger(args.ledger)
        if any(row["within_threshold"] is False for row in value["independent_checks"]):
            raise ValueError("a declared independent distance tolerance failed; CAD is not started")
        record.update(ledger=bound,data_kind=value["data_kind"],scale_verified=True,
                      converter=file_binding(ROOT/"scripts/mesh_to_cad.py"))
        arguments=["--mesh",value["output_mesh"]["path"],"--output",str(output),
                   "--source-run",str(args.ledger.resolve()),"--scale-status","reference_scaled",
                   "--length-unit","mm","--scale-reference",str(args.ledger.resolve()),
                   "--max-faces",str(args.max_faces)]
        if value["data_kind"]=="synthetic": arguments.append("--synthetic")
        record.update(converter_arguments=arguments,native_conversion_started=True)
        code=cad.main(arguments)
        verify_binding(bound,1024*1024)
        verify_ledger(args.ledger)
        verify_binding(record["converter"])
        record.update(status="PASS_VERIFIED_SCALE_AND_CAD" if code==0 else "CAD_REJECTED",exit_code=code)
    except (ValueError,OSError,KeyError,TypeError) as error:
        record.update(status="REJECTED",exit_code=2,error=type(error).__name__+": "+str(error))
    except Exception as error:
        record.update(status="INTERNAL_ERROR",exit_code=4,error=type(error).__name__+": "+str(error))
    finally:
        output.mkdir(parents=True,exist_ok=True)
        record["completed_utc"]=datetime.now(timezone.utc).isoformat()
        save(output/"scale_verification.json",record)
    print(str(output/"scale_verification.json"))
    return record["exit_code"]

if __name__=="__main__":
    raise SystemExit(main())
