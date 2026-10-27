from copy import deepcopy
import json
from pathlib import Path
import pytest
from integrations.meshprep.geometry import write_double_ply
from integrations.meshprep.ply_io import read_ply
from integrations.mesh_scale.ledger import create_ledger,verify_ledger,sha,save,read_json,file_binding

POINTS=[(0,0,0),(2,0,0),(0,3,0),(0,0,4)]
FACES=[(0,2,1),(0,1,3),(0,3,2),(1,2,3)]

def fixture(tmp):
    mesh=tmp/"source.ply"
    write_double_ply(mesh,POINTS,FACES)
    proof=tmp/"measurement.txt"
    proof.write_text("Synthetic fixture: x edge 20 mm; independent y edge 30 mm.\n")
    provenance=tmp/"source.json"; save(provenance,{"schema_version":"mesh-scale-fixture-source-v1",
                                                  "data_kind":"synthetic","mesh":file_binding(mesh)})
    def measure(indices,length):
        return dict(vertex_indices=indices,length=length,unit="mm",label="fixture-edge",
                    correspondence="These exact source-mesh vertices are synthetic measurement endpoints.",
                    evidence=dict(path="measurement.txt",sha256=sha(proof)))
    request=tmp/"reference.json"
    save(request,dict(schema_version="mesh-scale-request-v1",source_mesh_sha256=sha(mesh),
                      data_kind="synthetic",reference=measure([0,1],20),
                      checks=[dict(measure([0,2],30),allowed_absolute_error_mm=0.01)]))
    output=tmp/"scaled"; output.mkdir()
    return mesh,request,provenance,output

def make(tmp):
    mesh,request,source,out=fixture(tmp)
    value=create_ledger(mesh,request,source,out)
    return mesh,request,source,out,value

def test_real_file_roundtrip_binds_scale_and_independent_distance(tmp_path):
    mesh,request,source,out,value=make(tmp_path)
    assert value["transform"]["factor_to_mm"]==10
    assert value["independent_checks"][0]["absolute_error_mm"]==0
    assert value["independent_checks"][0]["within_threshold"] is True
    assert value["physical_accuracy_status"]=="NOT_EVALUATED"
    assert value["photo_lineage_status"]=="NOT_VERIFIED"
    points,faces,_=read_ply(out/"scaled.ply")
    assert points==[(0,0,0),(20,0,0),(0,30,0),(0,0,40)] and faces==FACES
    assert verify_ledger(out/"scale_ledger.json",synthetic=True)==value
    assert json.loads(json.dumps(value,allow_nan=False))==value

@pytest.mark.parametrize("changed",["mesh","measurement","source_record","request"],ids=["mesh","proof","source","request"])
def test_later_input_mutation_is_detected(tmp_path,changed):
    mesh,request,source,out,value=make(tmp_path)
    paths=dict(mesh=mesh,measurement=tmp_path/"measurement.txt",source_record=source,request=request)
    with paths[changed].open("ab") as f: f.write(b"\n")
    expected="evidence hash mismatch" if changed=="measurement" else "changed"
    with pytest.raises(ValueError,match=expected):
        verify_ledger(out/"scale_ledger.json")

def test_other_mesh_cannot_reuse_reference(tmp_path):
    mesh,request,source,out=fixture(tmp_path)
    with mesh.open("ab") as f: f.write(b"\n")
    with pytest.raises(ValueError,match="different mesh"):
        create_ledger(mesh,request,source,out)
    assert not (out/"scaled.ply").exists()

def test_missing_evidence_hash_rejected(tmp_path):
    mesh,request,source,out=fixture(tmp_path)
    data=read_json(request); data["reference"]["evidence"]["sha256"]="0"*64; save(request,data)
    with pytest.raises(ValueError,match="evidence hash"):
        create_ledger(mesh,request,source,out)

def test_forged_output_hash_does_not_bypass_geometric_recompute(tmp_path):
    mesh,request,source,out,value=make(tmp_path)
    other=out/"other.ply"
    write_double_ply(other,[(0,0,0),(21,0,0),(0,30,0),(0,0,40)],FACES)
    value["output_mesh"]=file_binding(other)
    save(out/"scale_ledger.json",value)
    with pytest.raises(ValueError,match="edge length|recalculated"):
        verify_ledger(out/"scale_ledger.json")

def test_forged_numeric_result_rejected(tmp_path):
    mesh,request,source,out,value=make(tmp_path)
    value["independent_checks"][0]["absolute_error_mm"]=9
    save(out/"scale_ledger.json",value)
    with pytest.raises(ValueError,match="results were changed"):
        verify_ledger(out/"scale_ledger.json")

def test_wrong_synthetic_flag_rejected(tmp_path):
    *_,out,value=make(tmp_path)
    with pytest.raises(ValueError,match="synthetic/real"):
        verify_ledger(out/"scale_ledger.json",synthetic=False)

@pytest.mark.parametrize("text",['[]','{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'],ids=["root","duplicate","nan","overflow"])
def test_strict_json(tmp_path,text):
    p=tmp_path/"bad.json"; p.write_text(text)
    with pytest.raises(ValueError): read_json(p)

def test_preexisting_output_never_overwritten(tmp_path):
    mesh,request,source,out,value=make(tmp_path)
    before=sha(out/"scaled.ply")
    with pytest.raises(FileExistsError): create_ledger(mesh,request,source,out)
    assert sha(out/"scaled.ply")==before

def test_fit_pair_cannot_claim_independent_check(tmp_path):
    mesh,request,source,out=fixture(tmp_path)
    data=read_json(request); data["checks"][0]["vertex_indices"]=[1,0]; save(request,data)
    with pytest.raises(ValueError):
        create_ledger(mesh,request,source,out)

def test_declared_distance_failure_stops_before_cad(tmp_path,monkeypatch):
    from scripts import scaled_mesh_to_cad as wrapper
    mesh,request,source,out=fixture(tmp_path)
    data=read_json(request); data["checks"][0]["length"]=35; save(request,data)
    value=create_ledger(mesh,request,source,out)
    assert value["independent_checks"][0]["within_threshold"] is False
    monkeypatch.setattr(wrapper,"ROOT",tmp_path)
    def forbidden(*args,**kwargs): raise AssertionError("native conversion must not start")
    monkeypatch.setattr(wrapper.cad,"main",forbidden)
    target=tmp_path/"outputs"/"cad_rejected"
    code=wrapper.main(["--ledger",str(out/"scale_ledger.json"),"--output",str(target)])
    result=read_json(target/"scale_verification.json")
    assert code==2 and result["native_conversion_started"] is False
    assert "tolerance failed" in result["error"]
    assert not (target/"model.step").exists()

def test_unknown_request_fields_rejected(tmp_path):
    mesh,request,source,out=fixture(tmp_path)
    data=read_json(request); data["physical_accuracy_certified"]=True; save(request,data)
    with pytest.raises(ValueError,match="unknown request"):
        create_ledger(mesh,request,source,out)

def test_numeric_zero_cannot_impersonate_failed_boolean(tmp_path):
    mesh,request,source,out=fixture(tmp_path)
    data=read_json(request); data["checks"][0]["length"]=35; save(request,data)
    value=create_ledger(mesh,request,source,out)
    assert value["independent_checks"][0]["within_threshold"] is False
    value["independent_checks"][0]["within_threshold"]=0
    save(out/"scale_ledger.json",value)
    with pytest.raises(ValueError,match="results were changed"):
        verify_ledger(out/"scale_ledger.json")

def test_uncalibrated_accuracy_claim_cannot_be_added(tmp_path):
    *_,out,value=make(tmp_path)
    value["physical_accuracy_status"]="CERTIFIED"
    save(out/"scale_ledger.json",value)
    with pytest.raises(ValueError,match="semantics disagree"):
        verify_ledger(out/"scale_ledger.json")

@pytest.mark.parametrize("kind",["real","failed","foreign"],ids=["false-real","failed-run","unrelated-mesh"])
def test_meshprep_source_must_agree_with_data_and_mesh(tmp_path,kind):
    mesh,request,source,out=fixture(tmp_path)
    record=dict(contract_version="mesh-prep-v1",status="PASS_PREPARED_TOPOLOGY",script_exit_code=0,
                synthetic_input=True,artifacts=[file_binding(mesh)])
    if kind=="real":
        data=read_json(request); data["data_kind"]="real"; save(request,data)
    elif kind=="failed": record["status"]="REJECTED"
    else: record["artifacts"][0]["sha256"]="0"*64
    save(source,record)
    with pytest.raises(ValueError): create_ledger(mesh,request,source,out)

def test_far_origin_reference_distance_is_checked_independently():
    from integrations.mesh_scale.calibration import fit_transform,apply_transform
    from integrations.mesh_scale.ledger import stored_reference_gate
    points=[(2**50,0,0),(2**50+0.25,0,0),(2**50,1,0)]
    transform=fit_transform(points,{"vertex_indices":[0,1],"length":1.1,"unit":"mm"})
    scaled,_=apply_transform(points,transform)
    assert scaled[1][0]-scaled[0][0]==1.0
    with pytest.raises(ValueError,match="reference distance"):
        stored_reference_gate(scaled,transform)

def test_replayed_ledger_cannot_skip_stored_distance_gate(tmp_path):
    from integrations.mesh_scale.calibration import fit_transform,apply_transform
    mesh,request,source,out,value=make(tmp_path)
    far=tmp_path/"far.ply"
    points=[(2**50,0,0),(2**50+0.25,0,0),(2**50,1,0),(2**50,0,1)]
    write_double_ply(far,points,FACES)
    new_request=tmp_path/"far_request.json"
    data=read_json(request); data.update(source_mesh_sha256=sha(far),checks=[])
    data["reference"]["length"]=1.1; save(new_request,data)
    new_source=tmp_path/"far_source.json"
    save(new_source,{"schema_version":"mesh-scale-fixture-source-v1","data_kind":"synthetic","mesh":file_binding(far)})
    transform=fit_transform(points,data["reference"])
    scaled,numeric=apply_transform(points,transform)
    new_target=tmp_path/"far_scaled.ply"; write_double_ply(new_target,scaled,FACES)
    value.update(source_mesh=file_binding(far),request=file_binding(new_request),source_record=file_binding(new_source),
                 output_mesh=file_binding(new_target),transform=transform,numeric_roundtrip=numeric,
                 independent_checks=[],measurement_evidence=value["measurement_evidence"][:1],stored_reference_length_mm=1.0)
    save(out/"scale_ledger.json",value)
    with pytest.raises(ValueError,match="edge length|reference distance"):
        verify_ledger(out/"scale_ledger.json")

def test_global_extent_cannot_hide_short_edge_distortion():
    from integrations.mesh_scale.ledger import geometry_gate
    source=[(2**50,0,0),(2**50+0.25,0,0),(2**50,1,0)]
    scaled=[tuple(4.4*x for x in p) for p in source]
    with pytest.raises(ValueError,match="edge length"):
        geometry_gate(source,scaled,[(0,1,2)],4.4)

def test_per_axis_rounding_does_not_imply_a_collapsed_vertex():
    from integrations.mesh_scale.ledger import geometry_gate
    import math
    source=[(0.1,0,0),(0.1+math.ulp(0.1),1,0),(1,0,1)]
    # A one-ULP coordinate component change with distinct XYZ is below the fixed edge tolerance.
    scaled=[(0.1,0,0),(0.1,1,0),(1,0,1)]
    assert geometry_gate(source,scaled,[(0,1,2)],1)["max_relative_edge_length_error"]<1e-12
