"""One bounded delivery step from a verified prepared mesh, not photos to CAD."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from integrations.mesh_scale.ledger import file_binding, verify_binding, read_json, source_semantics, verify_ledger
from integrations.viewer.export import export_html
from integrations.meshprep.provenance import validate_source

ROOT=Path(__file__).resolve().parents[2]
SUPERVISOR_SHA="077a4a021741d1b432d6312bcedbab1bc31696cc6621b66d61dc65dda0d7e4be"
PREP_CODE=("scripts/prepare_mesh.py","scripts/mesh_to_cad.py","integrations/meshprep/geometry.py","integrations/meshprep/ply_io.py","integrations/meshprep/provenance.py")


class StageFailure(RuntimeError):
    def __init__(self,name,code):
        super().__init__(f"{name} exited {code}; later stages not started")
        self.code=code if code in (2,3,4,5) else 4


def native_bindings(python):
    python=Path(python)
    base=python.parent
    paths=[python,base/"python311.dll",base/"FreeCAD.pyd",base/"FreeCADApp.dll",base/"FreeCADBase.dll",base.parent/"lib/Part.pyd",base.parent/"lib/Mesh.pyd"]
    return [file_binding(path,1024**3) for path in paths]


def output_paths(directory,scaled):
    paths=[Path(directory)/name for name in ("request.json","stages.json","supervisor_result.json","cad/conversion_report.json","cad/model.FCStd","cad/model.step","viewer.html","viewer_export.json","viewer_scenes.json")]
    if scaled:paths += [Path(directory)/name for name in ("scale/RESULT.json","scale/scaled.ply","scale/scale_ledger.json","cad/scale_verification.json")]
    return paths


def validate_wall(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 1<=value<=120:
        raise ValueError("wall-seconds must be finite within [1,120]")


def save(path,value):
    path=Path(path)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    tmp=path.with_name(path.name+".pending")
    tmp.write_text(text,encoding="utf-8");os.replace(tmp,path)


def prepared_input(directory):
    directory=Path(directory).resolve(strict=True)
    record_path=directory/"mesh_prep.json";mesh=directory/"prepared.ply"
    record=read_json(record_path)
    if not isinstance(record,dict) or record.get("contract_version")!="mesh-prep-v1":
        raise ValueError("prepared input must be a mesh-prep-v1 directory")
    flag=record.get("synthetic_input")
    if type(flag) is not bool:raise ValueError("prepared synthetic flag must be boolean")
    data_kind="synthetic" if flag else "real"
    bound=file_binding(mesh)
    source_semantics(record_path,bound,data_kind)
    implementation=record.get("implementation")
    if not isinstance(implementation,dict):raise ValueError("missing prepared implementation")
    normalized={k.replace("\\","/"):v for k,v in implementation.items()}
    if len(normalized)!=len(implementation) or set(normalized)!=set(PREP_CODE):
        raise ValueError("prepared implementation set differs")
    for name in PREP_CODE:
        if file_binding(ROOT/name)["sha256"]!=normalized[name]:
            raise ValueError("prepared implementation drift: "+name)
    artifacts=record.get("artifacts")
    if not isinstance(artifacts,list) or len(artifacts)!=3:
        raise ValueError("prepared artifacts must contain the expected three files")
    expected={str((directory/name).resolve()) for name in ("selected_before.ply","prepared.ply","selection_map.json")}
    if {row.get("path") for row in artifacts}!=expected:
        raise ValueError("prepared artifact set differs")
    for row in artifacts:verify_binding(row)
    inputs=record.get("inputs")
    if not isinstance(inputs,dict) or set(inputs)!={"mesh","source_record"}:raise ValueError("missing source bindings")
    upstream=[]
    for row in inputs.values():
        actual=file_binding(row["path"])
        if actual["sha256"]!=row["sha256"]:raise ValueError("prepared upstream input drift")
        upstream.append(actual)
    declared=record.get("source_validation")
    if not isinstance(declared,dict):raise ValueError("prepared producer-output source validation missing")
    checked=validate_source(inputs["mesh"]["path"],inputs["source_record"]["path"],synthetic=flag,role=declared.get("source_role"))
    if checked!=declared:raise ValueError("prepared producer-output source validation differs")
    return {"mesh":bound,"record":file_binding(record_path),"artifacts":artifacts,"upstream":upstream,
            "synthetic_input":flag,"data_kind":data_kind,"photo_lineage_status":"NOT_VERIFIED",
            "reuse_scope":"existing prepared bytes and current implementation bindings; topology is checked again during native CAD conversion"}


def implementation_bindings():
    paths=[ROOT/"scripts/build_cad_delivery.py",ROOT/"scripts/mesh_to_cad.py",ROOT/"scripts/scale_mesh.py",ROOT/"scripts/scaled_mesh_to_cad.py",ROOT/"scripts/prepare_mesh.py"]
    for folder in ("cad_delivery","mesh_scale","meshprep","viewer"):
        paths.extend(sorted((ROOT/"integrations"/folder).glob("*.py")))
    paths.extend(sorted((ROOT/"integrations/viewer/assets").glob("*")))
    return [file_binding(path) for path in paths]


def check_bindings(rows):
    if not isinstance(rows,list) or not rows:raise ValueError("empty binding set")
    for row in rows:verify_binding(row,1024*1024*1024)


def validate_scale_link(request,ledger):
    """Join an independently verified ledger to this exact delivery request.

    Both endpoints must be current files. An internally valid foreign ledger is
    not a substitute for the selected prepared mesh and measurement request.
    """
    prepared=request.get("prepared")
    if not isinstance(prepared,dict) or request.get("reference") is None:
        raise ValueError("scale link requires prepared inputs and a reference request")
    flag=prepared.get("synthetic_input")
    if type(flag) is not bool or prepared.get("data_kind")!=("synthetic" if flag else "real"):
        raise ValueError("scale link prepared data_kind mismatch")
    if ledger.get("data_kind")!=prepared["data_kind"]:
        raise ValueError("scale link data_kind mismatch")
    expected={"source_mesh":prepared.get("mesh"),"source_record":prepared.get("record"),
              "request":file_binding(request["reference"])}
    for name,row in expected.items():
        verify_binding(row)
        actual=ledger.get(name)
        if not isinstance(actual,dict):raise ValueError("scale link missing "+name)
        verify_binding(actual)
        if actual!=row:raise ValueError("scale link mismatched "+name)


def _reference_matches(record,expected,description):
    if not isinstance(record,dict) or not isinstance(record.get("path"),str):
        raise ValueError(description+" reference missing")
    if (Path(record["path"]).resolve()!=Path(expected["path"])
            or record.get("sha256")!=expected["sha256"]):
        raise ValueError(description+" reference mismatch")


def validate_cad(path,synthetic,mesh,scaled=False,*,source_record=None,scale_reference=None):
    path=Path(path)
    report=read_json(path/"conversion_report.json")
    expected="PASS_SYNTHETIC_CAD_CONVERSION" if synthetic else "PASS_GEOMETRIC_CAD_CONVERSION"
    if report.get("status")!=expected or type(report.get("script_exit_code")) is not int or report["script_exit_code"]!=0:
        raise ValueError("native CAD report did not pass")
    if report.get("synthetic_input") is not synthetic:raise ValueError("CAD data kind changed")
    actual=file_binding(mesh)
    if report.get("input",{}).get("mesh_sha256")!=actual["sha256"] or Path(report["input"]["mesh"]).resolve()!=Path(actual["path"]):
        raise ValueError("CAD input does not match selected mesh")
    unit=report.get("scale",{})
    if unit.get("source_scale_status")!=("reference_scaled" if scaled else "unknown"):
        raise ValueError("CAD scale status differs")
    if unit.get("declared_length_unit")!=("mm" if scaled else "arbitrary"):
        raise ValueError("CAD unit differs")
    artifacts=report.get("artifacts",[])
    expected_paths={str((path/name).resolve()) for name in ("model.FCStd","model.step")}
    if len(artifacts)!=2 or {row.get("path") for row in artifacts}!=expected_paths:
        raise ValueError("CAD artifact set differs")
    check_bindings(artifacts)
    if source_record is None:raise ValueError("CAD expected source_record is required")
    source=file_binding(source_record)
    _reference_matches({"path":report.get("input",{}).get("source_run"),
                        "sha256":report.get("input",{}).get("source_run_sha256")},source,"CAD source")
    if not scaled:
        if scale_reference is not None or unit.get("scale_reference") is not None:
            raise ValueError("unscaled CAD must not declare a scale reference")
    else:
        if scale_reference is None:raise ValueError("scaled CAD expected scale_reference is required")
        reference=file_binding(scale_reference)
        if reference!=source:raise ValueError("scaled CAD source and reference must be the same ledger")
        _reference_matches(unit.get("scale_reference"),reference,"CAD scale")
    if scaled:
        outer=read_json(path/"scale_verification.json")
        if outer.get("status")!="PASS_VERIFIED_SCALE_AND_CAD" or type(outer.get("exit_code")) is not int or outer["exit_code"]!=0:
            raise ValueError("scale wrapper post-validation did not pass")
        if outer.get("ledger")!=source:raise ValueError("scale wrapper ledger mismatch")
        verify_binding(outer["ledger"])
        if (outer.get("data_kind")!=("synthetic" if synthetic else "real") or outer.get("scale_verified") is not True
                or outer.get("native_conversion_started") is not True):
            raise ValueError("scale wrapper state/data_kind mismatch")
        if outer.get("converter")!=file_binding(ROOT/"scripts/mesh_to_cad.py"):
            raise ValueError("scale wrapper converter mismatch")
        face_limit=report.get("max_faces")
        if type(face_limit) is not int or face_limit<=0:raise ValueError("CAD max_faces invalid")
        expected_args=["--mesh",actual["path"],"--output",str(path.resolve()),"--source-run",source["path"],
                       "--scale-status","reference_scaled","--length-unit","mm","--scale-reference",source["path"],
                       "--max-faces",str(face_limit)]+(["--synthetic"] if synthetic else [])
        if outer.get("converter_arguments")!=expected_args:raise ValueError("scale wrapper converter arguments mismatch")
        # The converter is called in-process by the wrapper; sys.argv records
        # that wrapper invocation, not converter_arguments. Preserve this fact.
        invocation=report.get("argv")
        expected_wrapper=["--ledger",source["path"],"--output",str(path.resolve())]
        if (not isinstance(invocation,list) or len(invocation)<2 or not all(isinstance(x,str) for x in invocation)
                or Path(invocation[1]).resolve()!=(ROOT/"scripts/scaled_mesh_to_cad.py").resolve()
                or not (invocation[2:]==expected_wrapper+["--max-faces",str(face_limit)]
                        or (face_limit==10000 and invocation[2:]==expected_wrapper))):
            raise ValueError("scale wrapper recorded invocation mismatch")
    return report


def environment(native_bin=None,temp=None):
    env={key:os.environ[key] for key in ("SystemRoot","WINDIR","COMSPEC") if key in os.environ}
    bins=[str(Path(sys.executable).parent),str(Path(os.environ.get("SystemRoot",r"C:\Windows"))/"System32")]
    if native_bin is not None:bins.insert(0,str(native_bin))
    env["PATH"]=os.pathsep.join(bins)
    env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8",PYTHONNOUSERSITE="1")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS"):env[key]="2"
    if temp:
        Path(temp).mkdir(parents=True,exist_ok=True);env["TEMP"]=env["TMP"]=str(temp)
    return env


def coordinate(request_path,expected_sha):
    request_path=Path(request_path)
    if file_binding(request_path)["sha256"]!=expected_sha:raise ValueError("request changed before child launch")
    request=read_json(request_path);output=Path(request["output"])
    validate_wall(request["wall_seconds"])
    record={"status":"STARTED","stages":[],"started_utc":datetime.now(timezone.utc).isoformat(),"exit_code":None,
            "synthetic_input":request["prepared"]["synthetic_input"],"photo_lineage_status":"NOT_VERIFIED","GUI":"NOT_RUN"}
    journal=output/"stages.json";save(journal,record)
    def stage(name,argv,native=False):
        row={"name":name,"argv":list(map(str,argv)),"cwd":str(ROOT),"status":"STARTING","returncode":None,"started_utc":datetime.now(timezone.utc).isoformat()}
        record["stages"].append(row);save(journal,record)
        env=environment(Path(request["freecad_python"]).parent if native else None,output/"temp"/name)
        row["env_overrides"]=env
        start=time.monotonic()
        with (output/(name+".stdout.txt")).open("wb") as stdout,(output/(name+".stderr.txt")).open("wb") as stderr:
            # Outer Windows Job owns this coordinator and all child processes.
            proc=subprocess.Popen(row["argv"],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            row["pid"]=proc.pid;row["status"]="RUNNING";save(journal,record)
            row["returncode"]=proc.wait()
        row.update(status="COMPLETED" if row["returncode"]==0 else "FAILED",wall_seconds=time.monotonic()-start)
        save(journal,record)
        if row["returncode"]!=0:raise StageFailure(name,row["returncode"])
    try:
        check_bindings(request["bindings"])
        prepared=prepared_input(request["prepared_run"])
        if prepared!=request["prepared"]:raise ValueError("prepared state changed")
        mesh=Path(prepared["mesh"]["path"]);source=Path(prepared["record"]["path"])
        scaled=bool(request["reference"])
        if scaled:
            stage("scale",[sys.executable,"-B",str(ROOT/"scripts/scale_mesh.py"),"--mesh",str(mesh),"--reference",request["reference"],"--source-run",str(source),"--output",str(output/"scale")])
            scale_result=read_json(output/"scale/RESULT.json")
            if scale_result.get("status")!="PASS_REFERENCE_SCALING" or type(scale_result.get("exit_code")) is not int or scale_result["exit_code"]!=0:raise ValueError("scale report did not pass")
            ledger=verify_ledger(output/"scale/scale_ledger.json",synthetic=prepared["synthetic_input"])
            validate_scale_link(request,ledger)
            mesh=Path(ledger["output_mesh"]["path"])
            argv=[request["freecad_python"],"-I","-B",str(ROOT/"scripts/scaled_mesh_to_cad.py"),"--ledger",str(output/"scale/scale_ledger.json"),"--output",str(output/"cad")]
        else:
            argv=[request["freecad_python"],"-I","-B",str(ROOT/"scripts/mesh_to_cad.py"),"--mesh",str(mesh),"--source-run",str(source),"--output",str(output/"cad"),"--scale-status","unknown","--length-unit","arbitrary"]
            if prepared["synthetic_input"]:argv.append("--synthetic")
        stage("cad",argv,native=True)
        cad_source=output/"scale/scale_ledger.json" if scaled else source
        validate_cad(output/"cad",prepared["synthetic_input"],mesh,scaled,source_record=cad_source,
                     scale_reference=cad_source if scaled else None)
        record["stages"].append({"name":"viewer","status":"RUNNING","kind":"in_process_export"});save(journal,record)
        spec={"schema":"ply-viewer-scenes-v1","scenes":[{"ply":str(mesh),"sha256":file_binding(mesh)["sha256"],
               "title":"合成 CAD 输入网格" if prepared["synthetic_input"] else "CAD 输入网格（照片来源未认证）",
               "data_kind":"synthetic" if prepared["synthetic_input"] else "unspecified","declared_unit":"mm" if scaled else None,
               "note":"这是已通过几何转换及原生往返检查的输入 PLY 预览，不是 STEP/FCStd 导入；照片保真、物理精度和 GUI 均未验收。",
               "evidence":[str(output/"cad/conversion_report.json")]+([str(output/"scale/scale_ledger.json")] if scaled else [])}]}
        save(output/"viewer_scenes.json",spec)
        viewer=export_html(output/"viewer_scenes.json",output/"viewer.html");save(output/"viewer_export.json",viewer)
        record["stages"][-1].update(status="EXPORTED_NOT_UI_VERIFIED")
        check_bindings(request["bindings"])
        if prepared_input(request["prepared_run"])!=prepared:raise ValueError("prepared input changed during work")
        if file_binding(request_path)["sha256"]!=expected_sha:raise ValueError("request changed during work")
        record.update(status="COMPLETED_GEOMETRIC_DELIVERY",exit_code=0,scale_status="reference_scaled" if scaled else "unknown",physical_accuracy_status="NOT_EVALUATED")
    except StageFailure as error:
        record.update(status="RESOURCE_LIMIT" if error.code==5 else "STAGE_FAILED",exit_code=error.code,error=str(error))
    except (ValueError,OSError,KeyError,TypeError) as error:
        record.update(status="REJECTED",exit_code=2,error=f"{type(error).__name__}: {error}")
    except Exception as error:
        record.update(status="BACKEND_ERROR",exit_code=4,error=f"{type(error).__name__}: {error}")
    finally:
        record["completed_utc"]=datetime.now(timezone.utc).isoformat();save(journal,record)
    return record["exit_code"]


def supervise_request(request_path):
    request=read_json(request_path);output=Path(request["output"])
    validate_wall(request["wall_seconds"])
    path=ROOT/"integrations/cad_delivery/supervise.py"
    if file_binding(path)["sha256"]!=SUPERVISOR_SHA:raise ValueError("copied supervisor differs from accepted source")
    spec=importlib.util.spec_from_file_location("cad_delivery_job",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.supervise([sys.executable,"-B",str(ROOT/"scripts/build_cad_delivery.py"),"--internal-request",str(request_path),"--request-sha",file_binding(request_path)["sha256"]],
        cwd=ROOT,env_overrides=environment(temp=output/"temp/owner"),inherit_env=False,
        log_path=output/"supervisor.jsonl",stdout_path=output/"coordinator.stdout.txt",stderr_path=output/"coordinator.stderr.txt",
        job_memory_limit_bytes=2*1024**3,wall_clock_seconds=request["wall_seconds"],poll_seconds=.1,reclamation_seconds=5,
        run_dir=output,run_dir_threshold_bytes=128*1024**2,label="cad_delivery",script_sha256=file_binding(ROOT/"scripts/build_cad_delivery.py")["sha256"])
    save(output/"supervisor_result.json",result)
    return result


def verify_bundle(directory,result):
    directory=Path(directory).resolve(strict=True)
    if result.get("status")!="COMPLETED_GEOMETRIC_DELIVERY" or type(result.get("exit_code")) is not int or result["exit_code"]!=0:
        raise ValueError("delivery did not finish successfully")
    request=read_json(directory/"request.json")
    validate_wall(request["wall_seconds"])
    if request.get("schema")!="cad-delivery-request-v1" or request.get("output")!=str(directory):
        raise ValueError("request is not bound to this delivery directory")
    output_rows=result.get("request_and_outputs")
    paths=output_paths(directory,request["reference"] is not None)
    if not isinstance(output_rows,list) or len(output_rows)!=len(paths) or {row.get("path") for row in output_rows}!={str(p) for p in paths}:
        raise ValueError("delivery artifact binding set is incomplete or mismatched")
    required=implementation_bindings()+native_bindings(request["freecad_python"])+[file_binding(Path(request["coordinator_python"]))]
    if request["reference"] is not None:required.append(file_binding(request["reference"]))
    rows=request.get("bindings")
    if not isinstance(rows,list) or len(rows)!=len(required) or {row.get("path") for row in rows}!={row["path"] for row in required}:
        raise ValueError("implementation/tool/input binding set is incomplete")
    check_bindings(result["request_and_outputs"]);check_bindings(request["bindings"])
    if prepared_input(request["prepared_run"])!=request["prepared"]:raise ValueError("prepared record changed")
    stages=read_json(directory/"stages.json")
    if stages.get("status")!="COMPLETED_GEOMETRIC_DELIVERY" or type(stages.get("exit_code")) is not int or stages["exit_code"]!=0:raise ValueError("stage result was not successful")
    expected_names=(["scale"] if request["reference"] is not None else [])+["cad","viewer"]
    rows=stages.get("stages")
    if not isinstance(rows,list) or [row.get("name") for row in rows]!=expected_names:
        raise ValueError("delivery stage order or set differs")
    for row in rows[:-1]:
        if row.get("status")!="COMPLETED" or type(row.get("returncode")) is not int or row["returncode"]!=0:
            raise ValueError("a child stage did not complete")
    if rows[-1].get("status")!="EXPORTED_NOT_UI_VERIFIED":raise ValueError("viewer stage not exported")
    supervisor=read_json(directory/"supervisor_result.json")
    if supervisor.get("status")!="completed" or type(supervisor.get("root_exit_code")) is not int or supervisor["root_exit_code"]!=0 or supervisor.get("reclaim_confirmed") is not True:
        raise ValueError("supervision/cleanup did not complete")
    expected_argv=[request["coordinator_python"],"-B",str(ROOT/"scripts/build_cad_delivery.py"),"--internal-request",str(directory/"request.json"),"--request-sha",file_binding(directory/"request.json")["sha256"]]
    if supervisor.get("argv")!=expected_argv or supervisor.get("cwd")!=str(ROOT) or supervisor.get("wall_clock_seconds")!=request["wall_seconds"]:
        raise ValueError("supervisor execution does not match the original request")
    scaled=request["reference"] is not None
    if result.get("synthetic_input") is not request["prepared"]["synthetic_input"] or result.get("scale_status")!=("reference_scaled" if scaled else "unknown"):
        raise ValueError("delivery metadata differs from request")
    if scaled:
        ledger=verify_ledger(directory/"scale/scale_ledger.json",synthetic=request["prepared"]["synthetic_input"])
        validate_scale_link(request,ledger)
        mesh=ledger["output_mesh"]["path"]
    else:mesh=request["prepared"]["mesh"]["path"]
    cad_source=directory/"scale/scale_ledger.json" if scaled else request["prepared"]["record"]["path"]
    validate_cad(directory/"cad",request["prepared"]["synthetic_input"],mesh,scaled,source_record=cad_source,
                 scale_reference=cad_source if scaled else None)
    viewer=read_json(directory/"viewer_export.json")
    if viewer.get("status")!="EXPORTED_NOT_UI_VERIFIED" or viewer.get("html_sha256")!=file_binding(directory/"viewer.html")["sha256"]:
        raise ValueError("viewer report/hash differs")
    if viewer.get("spec_sha256")!=file_binding(directory/"viewer_scenes.json")["sha256"]:
        raise ValueError("viewer specification changed")
    if len(viewer.get("scenes",[]))!=1 or viewer["scenes"][0]["sha256"]!=file_binding(mesh)["sha256"]:
        raise ValueError("viewer does not show the CAD input mesh")
    return {"status":"BINDINGS_AND_REPORTS_VALID","native_revalidation":"NOT_RUN","GUI":"NOT_RUN","physical_accuracy":"NOT_EVALUATED",
            "cad_source_record":"BOUND","scale_request_link":"BOUND" if scaled else "NOT_APPLICABLE","photo_lineage":"NOT_VERIFIED",
            "delivery":str(directory),"notice":"Existing successful artifacts remain internally bound; this does not rerun CAD or authorize resuming a failed stage."}


def verify_delivery(directory):
    directory=Path(directory).resolve(strict=True)
    return verify_bundle(directory,read_json(directory/"RESULT.json"))
