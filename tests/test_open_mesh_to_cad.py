import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('open_cad',Path(__file__).parents[1]/'scripts/open_mesh_to_cad.py')
cad=importlib.util.module_from_spec(spec); spec.loader.exec_module(cad)

def mesh(points,faces):
    return SimpleNamespace(Topology=([SimpleNamespace(x=x,y=y,z=z) for x,y,z in points],faces))

TRI=[(0,0,0),(1,0,0),(0,1,0)]

def test_open_triangle_allowed():
    stats=cad.topology(mesh(TRI,[(0,1,2)])); cad.require_mesh(stats)
    assert stats['boundary_edges']==3 and stats['area']==0.5

def test_vertex_bowtie_rejected():
    stats=cad.topology(mesh(TRI+[(0,0,1),(0,-1,0)],[(0,1,2),(0,3,4)]))
    assert stats['nonmanifold_vertices']==1
    with pytest.raises(cad.Rejected): cad.require_mesh(stats)

def test_nonmanifold_edge_rejected():
    stats=cad.topology(mesh(TRI+[(0,0,1),(0,-1,0)],[(0,1,2),(1,0,3),(0,1,4)]))
    assert stats['nonmanifold_edges']==1
    with pytest.raises(cad.Rejected): cad.require_mesh(stats)

def test_orientation_rejected():
    stats=cad.topology(mesh(TRI+[(1,1,0)],[(0,1,2),(1,2,3)]))
    assert stats['orientation_errors']==1
    with pytest.raises(cad.Rejected): cad.require_mesh(stats)

@pytest.mark.parametrize('points,faces',[(TRI,[(0,1,1)]), (TRI,[(0,1,2),(0,1,2)]), ([(0,0,0),(1,0,0),(2,0,0)],[(0,1,2)])])
def test_bad_triangles_rejected(points,faces):
    with pytest.raises(cad.Rejected): cad.require_mesh(cad.topology(mesh(points,faces)))

def test_numeric_boundary_and_area_gate():
    source=dict(bbox=[0,0,0,1,1,0],area=.5,face_count=1,boundary_edges=3)
    assert cad.compare(source,dict(source,solid_count=0))
    assert not cad.compare(source,dict(source,boundary_edges=0))
    assert not cad.compare(source,dict(source,area=.6))
    assert not cad.compare(source,dict(source,solid_count=1))

def test_native_self_intersection_rejected():
    stats=cad.topology(mesh(TRI,[(0,1,2)])); stats['native_has_self_intersections']=True
    with pytest.raises(cad.Rejected): cad.require_mesh(stats)

def test_source_binding(tmp_path):
    original=tmp_path/'original.ply'; diagnostic=tmp_path/'diagnostic.json'; mapping=tmp_path/'map.json'; target=tmp_path/'repaired.ply'
    for path in (original,diagnostic,mapping,target): path.write_text('data')
    def record(path): return dict(path=str(path),sha256=cad.sha256(path),bytes=path.stat().st_size)
    mapping.write_text(json.dumps(dict(schema_version='open-mesh-face-map-v1',output_face_to_source_face=[0],deleted_source_faces=[1])))
    upstream=tmp_path/'upstream.json'; upstream.write_text(json.dumps({'artifacts':[record(original)]}))
    report=dict(upstream_report=record(upstream),retained_count=1,removed_count=1,deleted_face_ids=[1],schema='open-mesh-repair-v1',status='CANDIDATE_WRITTEN_NATIVE_VALIDATION_PENDING',input=dict(mesh=record(original),diagnostic=record(diagnostic)),artifacts=dict(mesh=record(target),face_map=record(mapping)),source_surface_type='simplified_reconstruction',gap_origin='repair_created_excision',scale_status='unknown')
    source=tmp_path/'report.json'; source.write_text(json.dumps(report))
    args=SimpleNamespace(mesh=target,mesh_sha256=cad.sha256(target),source_report=source,source_report_sha256=cad.sha256(source))
    assert cad.bind_source(args)[0]==target
    report['artifacts']['mesh']['path']=str(original); source.write_text(json.dumps(report)); args.source_report_sha256=cad.sha256(source)
    with pytest.raises(cad.Rejected,match='exact mesh'): cad.bind_source(args)

def test_existing_output_refused(tmp_path):
    assert cad.main(['--mesh',str(tmp_path/'x'),'--mesh-sha256','0'*64,'--source-report',str(tmp_path/'r'),'--source-report-sha256','0'*64,'--output',str(tmp_path)])==2

class Edge:
    def __init__(self,identity): self.identity=identity
    def hashCode(self): return 1 # intentional collision
    def isSame(self,other): return self.identity==other.identity

def test_brep_boundary_collision_safe():
    shape=SimpleNamespace(Faces=[SimpleNamespace(Edges=[Edge(1),Edge(2),Edge(3)]),SimpleNamespace(Edges=[Edge(2),Edge(4),Edge(5)])])
    assert cad.boundary_count(shape)==4

def test_solid_shape_rejected():
    shape=SimpleNamespace(isNull=lambda:False,BoundBox=SimpleNamespace(XMin=0,YMin=0,ZMin=0,XMax=1,YMax=1,ZMax=0),Area=.5,Faces=[1],Solids=[1],isValid=lambda:True)
    with pytest.raises(cad.Rejected,match='solid_count'): cad.metrics(shape)

@pytest.mark.parametrize('raw',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}','{"a":1e999}','[]'])
def test_strict_document_rejection(tmp_path,raw):
    path=tmp_path/'bad.json'; path.write_text(raw)
    with pytest.raises(cad.Rejected): cad.document(path)

def fake_shape(check):
    return SimpleNamespace(isNull=lambda:False,BoundBox=SimpleNamespace(XMin=0,YMin=0,ZMin=0,XMax=1,YMax=1,ZMax=0),Area=.5,Faces=[SimpleNamespace(Edges=[Edge(1),Edge(2),Edge(3)])],Solids=[],isValid=lambda:True,check=check)

def test_bop_called_on_open_surface():
    calls=[]
    result=cad.metrics(fake_shape(lambda flag:calls.append(flag)))
    assert calls==[True] and result['boundary_edges']==3 and result['detailed_bop_check']=='PASS'

@pytest.mark.parametrize('error',[False,True])
def test_bop_error_or_bad_return_rejected(error):
    def check(flag):
        if error: raise ValueError('native BOP found interference')
        return []
    with pytest.raises(cad.Rejected): cad.metrics(fake_shape(check))

def test_face_map_loaded_count_rejected(tmp_path):
    mapping=tmp_path/'map.json'; mapping.write_text(json.dumps(dict(schema_version='open-mesh-face-map-v1',output_face_to_source_face=[0],deleted_source_faces=[1])))
    report=dict(artifacts=dict(face_map=dict(path=str(mapping))),retained_count=1,removed_count=1,deleted_face_ids=[1])
    with pytest.raises(cad.Rejected,match='Counts'): cad.verify_face_map(report,2)
