"""Pure metric regressions; no FreeCAD import, process, mesh or CAD output."""
import copy
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/"scripts/mesh_to_cad.py"
spec=importlib.util.spec_from_file_location("cad_numeric_under_test",PATH)
cad=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cad)

def box(origin=0.,size=1.):
    return {"bbox":[origin]*3+[origin+size]*3,"volume":size**3}

@pytest.mark.parametrize("origin",[0.,1e9,-1e9])
def test_same_size_solid_shifted_one_unit_is_rejected_at_every_origin(origin):
    source=box(origin)
    target={"bbox":[x+1 for x in source["bbox"]],"volume":source["volume"]}
    result=cad.compare(source,target)
    assert result["pass"] is False
    assert result["linear_reference_length"]==1.
    assert result["linear_tolerance"]==1e-7
    assert result["bbox_absolute_differences"]==[1.]*6

def test_bbox_shape_change_cannot_hide_behind_large_origin_and_same_volume():
    source=box(1e9)
    target=copy.deepcopy(source)
    target["bbox"][3]+=0.5
    assert cad.compare(source,target)["pass"] is False

@pytest.mark.parametrize("origin",[0.,1e9,-1e9])
def test_identical_roundtrip_remains_accepted(origin):
    result=cad.compare(box(origin),box(origin))
    assert result["pass"] is True
    assert result["comparison_policy"]=="source_extent_v1"

@pytest.mark.parametrize("size",[1.,100.,10000.])
def test_error_budget_follows_object_size(size):
    source=box(size=size)
    close=copy.deepcopy(source)
    close["bbox"][3]+=0.25*cad.RTOL*size
    far=copy.deepcopy(source)
    far["bbox"][3]+=10*cad.RTOL*size
    assert cad.compare(source,close)["pass"] is True
    assert cad.compare(source,far)["pass"] is False

def test_tolerance_uses_largest_source_extent_not_world_coordinate():
    source={"bbox":[1e9,1e9,1e9,1e9+2,1e9+3,1e9+4],"volume":24.}
    result=cad.compare(source,copy.deepcopy(source))
    assert result["linear_reference_length"]==4.
    assert result["linear_tolerance"]==cad.RTOL*4

def test_volume_remains_an_independent_gate():
    target=box();target["volume"]=1.01
    result=cad.compare(box(),target)
    assert result["pass"] is False
    assert result["bbox_absolute_differences"]==[0.]*6

def test_absolute_floor_for_small_geometry_is_explicit():
    source=box(size=0.001)
    result=cad.compare(source,copy.deepcopy(source))
    assert result["linear_tolerance"]==cad.ATOL

@pytest.mark.parametrize("bad",[
    {"bbox":[0,0,0,1,1],"volume":1.},
    {"bbox":[0,0,0,1,1,1,float("nan")],"volume":1.},
    {"bbox":[0,0,0,1,1,float("inf")],"volume":1.},
    {"bbox":[0,0,0,-1,1,1],"volume":1.},
    {"bbox":[0,0,0,1,1,1],"volume":float("nan")},
    {"bbox":[0,0,0,1,1,1],"volume":0.},
    {"bbox":[False,0,0,1,1,1],"volume":1.},
    None,
])
def test_malformed_metrics_do_not_pass_vacuous_zip_or_nonfinite_checks(bad):
    with pytest.raises(cad.Rejected) as error:cad.compare(box(),bad)
    assert error.value.reason=="INVALID_NUMERIC_METRICS"
    assert error.value.exit_code==4

def test_zero_source_extent_cannot_define_a_solid_error_budget():
    source={"bbox":[0,0,0,0,1,1],"volume":1.}
    with pytest.raises(cad.Rejected):cad.compare(source,box())

def test_comparison_does_not_change_input_metrics():
    source=box();target=box(0.25)
    frozen=copy.deepcopy((source,target))
    cad.compare(source,target)
    assert (source,target)==frozen
