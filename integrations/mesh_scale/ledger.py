"""Bound file evidence and reversible geometry; measurement truth remains declared."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
from integrations.meshprep.geometry import write_double_ply
from integrations.meshprep.ply_io import read_ply
from .calibration import fit_transform, apply_transform, measure_checks

ROOT = Path(__file__).resolve().parents[2]
REQUEST = "mesh-scale-request-v1"
LEDGER = "mesh-scale-ledger-v1"
MAX_JSON = 1024 * 1024
MAX_EVIDENCE = 16 * 1024 * 1024

def same_json(left,right):
    """Type-aware JSON equality: False must never compare equal to numeric zero."""
    return json.dumps(left,sort_keys=True,allow_nan=False,separators=(",",":")) == json.dumps(right,sort_keys=True,allow_nan=False,separators=(",",":"))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    p=Path(path)
    with p.open("rb") as stream:
        raw=stream.read(MAX_JSON+1)
    if len(raw)>MAX_JSON:
        raise ValueError("JSON exceeds 1 MiB")
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError("duplicate JSON key: "+key)
            result[key]=value
        return result
    def nonfinite(value): raise ValueError("nonfinite JSON constant: "+value)
    def finite_float(text):
        result=float(text)
        if not math.isfinite(result): raise ValueError("nonfinite JSON number")
        return result
    value=json.loads(raw.decode("utf-8-sig"),object_pairs_hook=unique,parse_constant=nonfinite,parse_float=finite_float)
    if not isinstance(value,dict): raise ValueError("JSON root must be object")
    return value

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def file_binding(path,max_bytes=None):
    p=Path(path).resolve(strict=True)
    if not p.is_file(): raise ValueError("evidence must be a file")
    if max_bytes is not None and p.stat().st_size>max_bytes:
        raise ValueError("evidence exceeds byte limit")
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def verify_binding(binding,max_bytes=None):
    if not isinstance(binding,dict) or not isinstance(binding.get("path"),str):
        raise ValueError("invalid file binding")
    actual=file_binding(binding["path"],max_bytes)
    if not same_json(actual,binding): raise ValueError("file binding changed: "+binding["path"])
    return Path(actual["path"])

def evidence(measurement,request_path):
    if not isinstance(measurement,dict): raise ValueError("measurement must be object")
    allowed={"vertex_indices","length","unit","label","correspondence","evidence","allowed_absolute_error_mm"}
    if set(measurement)-allowed: raise ValueError("unknown measurement field")
    for key in ("label","correspondence"):
        if not isinstance(measurement.get(key),str) or not measurement[key].strip():
            raise ValueError("measurement "+key+" must explain physical feature and endpoint correspondence")
    ref=measurement.get("evidence")
    if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:
        raise ValueError("measurement evidence requires path and sha256")
    if not isinstance(ref["path"],str) or not isinstance(ref["sha256"],str):
        raise ValueError("measurement evidence path/hash must be strings")
    path=Path(ref["path"])
    if not path.is_absolute(): path=request_path.parent/path
    bound=file_binding(path,MAX_EVIDENCE)
    if bound["sha256"]!=ref["sha256"]: raise ValueError("measurement evidence hash mismatch")
    return bound

def load_request(path,mesh):
    request_path=Path(path).resolve(strict=True)
    request=read_json(request_path)
    allowed={"schema_version","source_mesh_sha256","data_kind","reference","checks","origin_vertex"}
    if set(request)-allowed: raise ValueError("unknown request field")
    if request.get("schema_version")!=REQUEST: raise ValueError("unsupported scale request schema")
    if request.get("data_kind") not in ("real","synthetic"): raise ValueError("data_kind must be explicit")
    if not isinstance(request.get("source_mesh_sha256"),str) or not re.fullmatch("[0-9a-f]{64}",request["source_mesh_sha256"]):
        raise ValueError("source mesh hash must be full lowercase SHA256")
    if sha(mesh)!=request["source_mesh_sha256"]: raise ValueError("reference is bound to a different mesh")
    checks=request.get("checks",[])
    if not isinstance(checks,list) or len(checks)>64: raise ValueError("checks must be a list of at most 64")
    ref=request.get("reference")
    if isinstance(ref,dict) and "allowed_absolute_error_mm" in ref:
        raise ValueError("fit reference is not an independent error check")
    proofs=[evidence(ref,request_path)]+[evidence(row,request_path) for row in checks]
    return request,proofs

def geometry_gate(source,scaled,faces,factor):
    if not faces or len(source)<3 or len(source)!=len(scaled):
        raise ValueError("nonempty triangle mesh required")
    if len(set(source))!=len(set(scaled)):
        raise ValueError("scaling collapsed distinct vertices")
    edges={tuple(sorted((a,b))) for face in faces for a,b in zip(face,(face[1],face[2],face[0]))}
    maximum=0.0
    for a,b in edges:
        expected=math.dist(source[a],source[b])*factor
        actual=math.dist(scaled[a],scaled[b])
        if not math.isfinite(expected) or not math.isfinite(actual) or expected<=0 or actual<=0:
            raise ValueError("nonfinite or collapsed mesh edge length")
        relative=abs(actual-expected)/expected
        if not math.isfinite(relative) or relative>1e-12:
            raise ValueError("mesh edge length changed beyond uniform-scale numeric tolerance")
        maximum=max(maximum,relative)
    for points in (source,scaled):
        for a,b,c in faces:
            p,q,r=points[a],points[b],points[c]
            u=[q[i]-p[i] for i in range(3)]
            v=[r[i]-p[i] for i in range(3)]
            cross=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
            if not all(math.isfinite(x) for x in cross) or not any(cross):
                raise ValueError("degenerate or nonfinite triangle before/after scaling")
    return dict(edge_count=len(edges),max_relative_edge_length_error=maximum,
                relative_edge_tolerance=1e-12,absolute_tolerance=0,
                reference="each original triangle edge length times factor; not world-coordinate magnitude")

def implementation_bindings():
    paths=[ROOT/"integrations/mesh_scale"/name for name in ("calibration.py","ledger.py")]
    paths += [ROOT/"integrations/meshprep"/name for name in ("geometry.py","ply_io.py")]
    return [file_binding(p) for p in paths]

def source_semantics(path,mesh_binding,data_kind):
    """Check known immediate provenance only; never certify the upstream photo chain."""
    record=read_json(path)
    if record.get("contract_version")=="mesh-prep-v1":
        if record.get("status")!="PASS_PREPARED_TOPOLOGY" or type(record.get("script_exit_code")) is not int or record["script_exit_code"]!=0:
            raise ValueError("source mesh-prep run was not successful")
        if type(record.get("synthetic_input")) is not bool or record["synthetic_input"]!=(data_kind=="synthetic"):
            raise ValueError("source synthetic flag conflicts with requested data_kind")
        artifacts=record.get("artifacts")
        if not isinstance(artifacts,list) or not any(same_json(row,mesh_binding) for row in artifacts):
            raise ValueError("mesh is not a bound artifact of source mesh-prep run")
        return dict(schema="mesh-prep-v1",immediate_binding_verified=True,data_kind=data_kind)
    if record.get("schema_version")=="mesh-scale-fixture-source-v1" and record.get("data_kind")==data_kind=="synthetic":
        if not same_json(record.get("mesh"),mesh_binding):
            raise ValueError("synthetic fixture mesh binding mismatch")
        return dict(schema="mesh-scale-fixture-source-v1",immediate_binding_verified=True,data_kind="synthetic")
    raise ValueError("unsupported source record; real inputs require a successful bound mesh-prep-v1 record")

def stored_reference_gate(points,transform):
    a,b=transform["reference"]["vertex_indices"]
    actual_fit=math.dist(points[a],points[b])
    if not math.isclose(actual_fit,transform["reference"]["length_mm"],rel_tol=1e-12,abs_tol=0):
        raise ValueError("reference distance was not achieved in stored coordinates")
    return actual_fit

def create_ledger(mesh_path,request_path,source_record,output):
    mesh=Path(mesh_path).resolve(strict=True)
    record=Path(source_record).resolve(strict=True)
    output=Path(output)
    before_mesh=file_binding(mesh,64*1024**2)
    before_request=file_binding(request_path,MAX_JSON)
    before_record=file_binding(record,4*1024**2)
    request,proofs=load_request(request_path,mesh)
    source_check=source_semantics(record,before_mesh,request["data_kind"])
    before_implementation=implementation_bindings()
    points,faces,meta=read_ply(mesh)
    transform=fit_transform(points,request["reference"],request.get("origin_vertex"))
    scaled,numeric=apply_transform(points,transform)
    edge_check=geometry_gate(points,scaled,faces,transform["factor_to_mm"])
    checks=measure_checks(scaled,request.get("checks",[]),request["reference"]["vertex_indices"])
    out=output/"scaled.ply"
    write_double_ply(out,scaled,faces)
    actual,actual_faces,_=read_ply(out)
    if actual!=scaled or actual_faces!=faces:
        raise ValueError("disk PLY changed coordinates, face order or winding")
    actual_fit=stored_reference_gate(actual,transform)
    for bound,cap in [(before_mesh,64*1024**2),(before_request,MAX_JSON),(before_record,4*1024**2)]:
        verify_binding(bound,cap)
    for bound in proofs: verify_binding(bound,MAX_EVIDENCE)
    if before_implementation!=implementation_bindings():
        raise ValueError("scale implementation changed")
    value=dict(schema_version=LEDGER,status="REFERENCE_SCALED",data_kind=request["data_kind"],
               source_mesh=before_mesh,request=before_request,source_record=before_record,
               source_validation=source_check,stored_reference_length_mm=actual_fit,
               measurement_evidence=proofs,output_mesh=file_binding(out),
               scale_status="reference_scaled",source_length_unit="arbitrary",length_unit="mm",
               transform=transform,numeric_roundtrip=numeric,edge_length_check=edge_check,independent_checks=checks,
               physical_accuracy_status="NOT_EVALUATED",
               physical_accuracy_reason="Input endpoint correspondences and measurement truth are declared by the caller; checks apply only to listed distances.",
               reference_fit_is_independent_check=False,
               photo_lineage_status="NOT_VERIFIED",upstream_transform_history="NOT_VERIFIED",
               cad_validity_status="NOT_EVALUATED",
               properties_discarded=meta["ignored_scalar_properties"],
               vertex_order_preserved=True,face_order_and_winding_preserved=True,
               implementation=before_implementation)
    save(output/"scale_ledger.json",value)
    return value

def verify_ledger(path,synthetic=None):
    """Recompute against immutable source/request, rather than trusting a PASS flag."""
    path=Path(path).resolve(strict=True)
    value=read_json(path)
    if value.get("schema_version")!=LEDGER or value.get("status")!="REFERENCE_SCALED":
        raise ValueError("scale ledger is not a supported successful record")
    if value.get("data_kind") not in ("real","synthetic"):
        raise ValueError("invalid ledger data_kind")
    if synthetic is not None and (value["data_kind"]=="synthetic")!=synthetic:
        raise ValueError("synthetic/real mismatch")
    source=verify_binding(value.get("source_mesh"),64*1024**2)
    target=verify_binding(value.get("output_mesh"),64*1024**2)
    request_path=verify_binding(value.get("request"),MAX_JSON)
    source_record=verify_binding(value.get("source_record"),4*1024**2)
    request,proofs=load_request(request_path,source)
    source_check=source_semantics(source_record,value["source_mesh"],request["data_kind"])
    if not same_json(value.get("source_validation"),source_check):
        raise ValueError("ledger source validation differs from recomputed provenance")
    if request["data_kind"]!=value["data_kind"] or not same_json(proofs,value.get("measurement_evidence")):
        raise ValueError("ledger measurement provenance differs from request")
    if not same_json(value.get("implementation"),implementation_bindings()):
        raise ValueError("ledger needs the bound scaling implementation")
    points,faces,_=read_ply(source)
    transform=fit_transform(points,request["reference"],request.get("origin_vertex"))
    scaled,numeric=apply_transform(points,transform)
    actual,out_faces,_=read_ply(target)
    edge_check=geometry_gate(points,actual,faces,transform["factor_to_mm"])
    if not same_json(value.get("edge_length_check"),edge_check):
        raise ValueError("ledger mesh edge length report changed")
    if scaled!=actual or faces!=out_faces or not same_json(value.get("transform"),transform):
        raise ValueError("ledger or output disagrees with recalculated transform")
    if not same_json(value.get("stored_reference_length_mm"),stored_reference_gate(actual,transform)):
        raise ValueError("ledger stored reference length changed")
    checks=measure_checks(actual,request.get("checks",[]),request["reference"]["vertex_indices"])
    if not same_json(value.get("independent_checks"),checks) or not same_json(value.get("numeric_roundtrip"),numeric):
        raise ValueError("ledger distance/check results were changed")
    fixed=dict(length_unit="mm",scale_status="reference_scaled",source_length_unit="arbitrary",
               physical_accuracy_status="NOT_EVALUATED",photo_lineage_status="NOT_VERIFIED",
               upstream_transform_history="NOT_VERIFIED",cad_validity_status="NOT_EVALUATED",
               reference_fit_is_independent_check=False,vertex_order_preserved=True,
               face_order_and_winding_preserved=True)
    if not same_json({k:value.get(k) for k in fixed},fixed):
        raise ValueError("ledger scale/evidence semantics disagree")
    value["independent_checks"]=checks
    return value
