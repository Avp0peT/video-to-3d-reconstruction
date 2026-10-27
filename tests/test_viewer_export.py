"""Behavior boundaries for an offline preview, not reconstruction acceptance."""
import hashlib
import json
import re
from pathlib import Path
import pytest
from integrations.viewer.export import export_html, load_scenes


def setup_spec(tmp_path, title="real </script><script>alert(1)</script> /*VIEWER_JS*/"):
    ply=tmp_path/"points.ply"
    ply.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 2\nproperty double x\nproperty double y\nproperty double z\nend_header\n0 0 0\n1 2 3\n")
    data={"schema":"ply-viewer-scenes-v1","scenes":[{"ply":"points.ply","sha256":hashlib.sha256(ply.read_bytes()).hexdigest(),"title":title,"data_kind":"unspecified","declared_unit":None,"note":"No photo lineage or measurement claim."}]}
    spec=tmp_path/"scenes.json";spec.write_text(json.dumps(data),encoding="utf-8")
    return spec,data,ply


def test_standalone_html_preserves_points_and_escapes_script(tmp_path):
    spec,data,ply=setup_spec(tmp_path)
    dest=tmp_path/"viewer.html"; result=export_html(spec,dest)
    text=dest.read_text(encoding="utf-8")
    payload=re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',text,re.S)[1]
    decoded=json.loads(payload)
    assert decoded[0]["title"]==data["scenes"][0]["title"]
    assert decoded[0]["vertices"]==[[0,0,0],[1,2,3]]
    assert "<script>" not in payload
    assert result["html_sha256"]==hashlib.sha256(dest.read_bytes()).hexdigest()
    assert result["scenes"][0]["faces"]==0
    assert result["status"]=="EXPORTED_NOT_UI_VERIFIED"
    assert result["scenes"][0]["declared_unit"] is None
    assert 'src="http' not in text and 'href="http' not in text
    assert "/*SCENE_DATA*/" not in text
    assert "connect-src 'none'" in text


def test_hash_drift_creates_no_html(tmp_path):
    spec,_,ply=setup_spec(tmp_path);ply.write_bytes(ply.read_bytes().replace(b"1 2 3",b"1 2 4"))
    dest=tmp_path/"viewer.html"
    with pytest.raises(ValueError,match="hash mismatch"):export_html(spec,dest)
    assert not dest.exists()


def test_no_overwrite(tmp_path):
    spec,_,_=setup_spec(tmp_path);dest=tmp_path/"viewer.html";dest.write_bytes(b"retain")
    with pytest.raises(ValueError,match="overwrite"):export_html(spec,dest)
    assert dest.read_bytes()==b"retain"


@pytest.mark.parametrize("key,value",[("data_kind","verified_real"),("declared_unit","metres"),("title",None),("sha256","F"*64),("note",""),("evidence",[3])])
def test_bad_labels_rejected(tmp_path,key,value):
    spec,data,_=setup_spec(tmp_path);data["scenes"][0][key]=value;spec.write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(ValueError):load_scenes(spec)


def test_evidence_is_bound_not_declared_validated(tmp_path):
    spec,data,_=setup_spec(tmp_path);ep=tmp_path/"report.json";ep.write_bytes(b'{"status":"FAIL"}')
    data["scenes"][0]["evidence"]=["report.json"];spec.write_text(json.dumps(data),encoding="utf-8")
    result=export_html(spec,tmp_path/"viewer.html")
    assert result["scenes"][0]["evidence"][0]["sha256"]==hashlib.sha256(ep.read_bytes()).hexdigest()
    assert "not semantically revalidated" in result["boundary"]


def test_duplicate_spec_keys_rejected(tmp_path):
    spec,_,_=setup_spec(tmp_path);spec.write_text('{"schema":"x","schema":"ply-viewer-scenes-v1","scenes":[]}',encoding="utf-8")
    with pytest.raises(ValueError,match="duplicate"):load_scenes(spec)


def test_display_bbox_equal_scale_does_not_modify_source(tmp_path):
    spec,_,ply=setup_spec(tmp_path);raw=ply.read_bytes();scenes,_=load_scenes(spec)
    assert scenes[0]["bbox"]==[[0,0,0],[1,2,3]]
    assert scenes[0]["view_scale"]==2/3
    assert scenes[0]["center"]==[.5,1,1.5]
    assert ply.read_bytes()==raw

@pytest.mark.parametrize('profile,vertices,faces', [('standard',100000,200000),('surface400k',200000,400000)])
def test_profile_is_passed_to_reader_and_recorded(tmp_path,monkeypatch,profile,vertices,faces):
    from integrations.viewer import export as module
    spec,_,_=setup_spec(tmp_path);original=module.read_ply;calls=[]
    def reader(path,**kwargs):calls.append(kwargs);return original(path,**kwargs)
    monkeypatch.setattr(module,'read_ply',reader)
    result=module.export_html(spec,tmp_path/'profile.html',profile=profile)
    expected=dict(max_bytes=16*1024*1024,max_vertices=vertices,max_faces=faces)
    assert calls==[expected] and result['geometry_limits']==expected and result['profile']==profile

@pytest.mark.parametrize('profile,limit', [('standard',100000),('surface400k',200000)])
def test_aggregate_profile_budget_not_only_reader_limit(tmp_path,monkeypatch,profile,limit):
    from integrations.viewer import export as module
    spec,data,ply=setup_spec(tmp_path);data['scenes']*=2;spec.write_text(json.dumps(data),encoding='utf-8')
    def reader(path,**kwargs):
        return dict(vertices=[[0,0,0]]*(limit//2+1),faces=[],sha256=data['scenes'][0]['sha256'],colors=None)
    monkeypatch.setattr(module,'read_ply',reader)
    with pytest.raises(ValueError,match='aggregate'):module.load_scenes(spec,profile=profile)

def test_surface_profile_does_not_bypass_reader_header_budget(tmp_path):
    spec,data,ply=setup_spec(tmp_path)
    ply.write_bytes(ply.read_bytes().replace(b'element vertex 2',b'element vertex 200001'))
    data['scenes'][0]['sha256']=hashlib.sha256(ply.read_bytes()).hexdigest();spec.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='budget'):load_scenes(spec,profile='surface400k')

def test_profile_must_be_explicit_and_standard_still_default(tmp_path):
    spec,_,_=setup_spec(tmp_path)
    assert export_html(spec,tmp_path/'default.html')['profile']=='standard'
    with pytest.raises(ValueError,match='unsupported'):load_scenes(spec,profile='unbounded')

def test_surface_profile_preserves_triangle_indices_and_hash_gate(tmp_path):
    spec,data,ply=setup_spec(tmp_path)
    ply.write_bytes(b'ply\nformat ascii 1.0\nelement vertex 3\nproperty double x\nproperty double y\nproperty double z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
    raw=ply.read_bytes();data['scenes'][0]['sha256']=hashlib.sha256(raw).hexdigest();spec.write_text(json.dumps(data),encoding='utf-8')
    dest=tmp_path/'surface.html';export_html(spec,dest,profile='surface400k')
    decoded=json.loads(re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',dest.read_text(encoding='utf-8'),re.S)[1])
    assert decoded[0]['faces']==[[0,1,2]] and decoded[0]['vertices']==[[0,0,0],[1,0,0],[0,1,0]] and ply.read_bytes()==raw
    ply.write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='hash mismatch'):export_html(spec,tmp_path/'drift.html',profile='surface400k')
