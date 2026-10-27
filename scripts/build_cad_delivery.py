"""Build or inspect a CAD delivery from an existing accepted prepared mesh."""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,math,os,time
from integrations.cad_delivery.delivery import (save,prepared_input,implementation_bindings,check_bindings,file_binding,verify_binding,read_json,coordinate,supervise_request,verify_delivery,verify_bundle,native_bindings,output_paths)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run",type=Path)
    parser.add_argument("--freecad-python",type=Path)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--wall-seconds",type=float,default=120)
    parser.add_argument("--verify-run",type=Path)
    parser.add_argument("--internal-request",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--request-sha",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.internal_request:
        if not args.request_sha:parser.error("internal request needs SHA")
        return coordinate(args.internal_request,args.request_sha)
    if args.verify_run:
        if any((args.prepared_run,args.freecad_python,args.reference,args.output)):parser.error("verify-run cannot be combined with build inputs")
        try:print(json.dumps(verify_delivery(args.verify_run),ensure_ascii=False));return 0
        except (ValueError,OSError,KeyError,TypeError) as error:print(json.dumps({"status":"INVALID_OR_STALE","error":str(error)},ensure_ascii=False));return 2
    if not all((args.prepared_run,args.freecad_python,args.output)):parser.error("build requires prepared-run, freecad-python and output")
    output=args.output.resolve()
    if output.exists() or (ROOT/"outputs").resolve() not in output.parents:
        print("output must be new and strictly under project outputs",file=sys.stderr);return 2
    if not math.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=120:
        print("wall-seconds must be finite within [1,120]",file=sys.stderr);return 2
    output.mkdir(parents=True,exist_ok=False)
    result={"status":"PREFLIGHT","exit_code":None,"GUI":"NOT_RUN","photo_lineage_status":"NOT_VERIFIED","physical_accuracy":"NOT_EVALUATED"}
    save(output/"RESULT.json",result);started=time.monotonic()
    try:
        if os.name!="nt":raise ValueError("this delivery supervisor currently requires Windows")
        prepared=prepared_input(args.prepared_run)
        native=args.freecad_python.resolve(strict=True)
        if native.name.lower()!="python.exe":raise ValueError("freecad-python must identify bundled python.exe")
        bindings=implementation_bindings()+native_bindings(native)+[file_binding(Path(sys.executable))]
        reference=str(args.reference.resolve(strict=True)) if args.reference else None
        if reference:bindings.append(file_binding(reference))
        request={"schema":"cad-delivery-request-v1","prepared_run":str(args.prepared_run.resolve()),"prepared":prepared,"freecad_python":str(native),"coordinator_python":str(Path(sys.executable).resolve()),"reference":reference,"output":str(output),"wall_seconds":args.wall_seconds,"bindings":bindings}
        save(output/"request.json",request)
        request_binding=file_binding(output/"request.json")
        result["status"]="RUNNING";save(output/"RESULT.json",result)
        supervisor=supervise_request(output/"request.json")
        verify_binding(request_binding)
        if supervisor.get("status")!="completed" or supervisor.get("root_exit_code")!=0 or supervisor.get("reclaim_confirmed") is not True:
            if supervisor.get("status") in ("wall_clock_exceeded","run_dir_threshold","total_threshold"):
                result.update(status="RESOURCE_LIMIT",exit_code=5,error="Job resource budget reached; inspect supervisor_result.json")
                return 5
            if (output/"stages.json").is_file():
                failed=read_json(output/"stages.json")
                if failed.get("exit_code") in (2,3,4,5):
                    result.update(status=failed.get("status","STAGE_FAILED"),exit_code=failed["exit_code"],error=failed.get("error","stage failed"))
                    return failed["exit_code"]
            raise ValueError("coordinator or supervision failed; inspect supervisor_result.json and stages.json")
        stages=read_json(output/"stages.json")
        if stages.get("status")!="COMPLETED_GEOMETRIC_DELIVERY" or type(stages.get("exit_code")) is not int or stages["exit_code"]!=0:
            raise ValueError("delivery stages did not complete")
        check_bindings(bindings)
        paths=output_paths(output,reference is not None)
        result.update(status="COMPLETED_GEOMETRIC_DELIVERY",exit_code=0,synthetic_input=prepared["synthetic_input"],scale_status=stages["scale_status"],
                      viewer_status="EXPORTED_NOT_UI_VERIFIED",request_and_outputs=[file_binding(path) for path in paths],
                      resource_scope="coordinator/children supervised as a Job with 2GiB limit, shared wall budget and 128MiB sampled output threshold; legacy raw peak memory is not promoted to validated peak")
        verify_bundle(output,result)
    except (ValueError,OSError,KeyError,TypeError) as error:
        result.update(status="REJECTED",exit_code=2,error=f"{type(error).__name__}: {error}")
    except Exception as error:result.update(status="BACKEND_ERROR",exit_code=4,error=f"{type(error).__name__}: {error}")
    finally:
        result["wall_seconds"]=time.monotonic()-started;save(output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"result":str(output/"RESULT.json"),"exit_code":result["exit_code"]},ensure_ascii=False))
    return result["exit_code"]


if __name__=="__main__":raise SystemExit(main())
