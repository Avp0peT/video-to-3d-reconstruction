"""Synthetic data deletion protocol only; no FreeCAD or intersection algorithm."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts import repair_open_mesh as repair

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,value):path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')

@pytest.fixture
def fixture(tmp_path):
    points=[(float(i*2+x),float(y),float(z)) for i in range(6) for x,y,z in [(0,0,0),(1,0,0),(0,1,0)]]
    faces=[(i*3,i*3+1,i*3+2) for i in range(6)]
    mesh=tmp_path/'TEST_DATA.ply';repair.geo.write_double_ply(mesh,points,faces)
    pair=lambda a,b:dict(face_a=a,face_b=b,endpoint_a=[0.,0.,0.],endpoint_b=[1.,0.,0.])
    diagnostic=dict(status='READONLY_DIAGNOSTIC_COMPLETE',input={'path':str(mesh.resolve()),'sha256':digest(mesh),'faces':6},
        api_exists=True,input_unchanged=True,loaded_face_order_matches_source_rows=True,
        face_index_scope='loaded FreeCAD mesh; exact source row mapping proven',loaded_faces=6,
        has_self_intersections=True,pair_count=3,pairs_written=3,truncated=False,pairs=[pair(1,2),pair(3,4),pair(3,4)])
    record=tmp_path/'TEST_DATA_DIAGNOSTIC.json';save(record,diagnostic)
    return mesh,record,diagnostic,points,faces

def run(fixture,output):
    mesh,record,*_=fixture
    return repair.repair(mesh,record,digest(mesh),digest(record),output)

def test_duplicate_pairs_delete_four_faces_keep_all_vertices_and_source_order(fixture,tmp_path):
    mesh,record,diagnostic,points,faces=fixture;original=(mesh.read_bytes(),record.read_bytes())
    output=tmp_path/'candidate';report=run(fixture,output)
    assert report['status']=='CANDIDATE_WRITTEN_NATIVE_VALIDATION_PENDING'
    assert report['schema']=='open-mesh-repair-v1'
    assert report['deleted_face_ids']==[1,2,3,4] and report['removed_count']==4 and report['retained_count']==2
    assert set(report['artifacts'])=={'mesh','face_map'}
    assert report['source_surface_type']=='simplified_reconstruction' and report['gap_origin']=='repair_created_excision'
    assert report['original_pairs']==diagnostic['pairs']
    assert report['original_pair_count']==3 and report['unique_pair_count']==2
    assert report['unique_pairs']==[[1,2],[3,4]] and report['deleted_source_faces']==[1,2,3,4]
    result_points,result_faces,_=repair.ply.read_ply(output/'open_surface.ply')
    assert result_points==points and result_faces==[faces[0],faces[5]]
    assert json.loads((output/'face_map.json').read_text())['output_face_to_source_face']==[0,5]
    assert (mesh.read_bytes(),record.read_bytes())==original
    assert report['deleted_area']==2. and report['deleted_area_fraction']==pytest.approx(2/3)
    assert report['native_self_intersection_status']=='NOT_EVALUATED'
    assert report['volume_status']=='NOT_EVALUATED_OPEN_SURFACE'

def test_reversed_duplicate_pair_is_still_one_pair(fixture,tmp_path):
    fixture[2]['pairs'][2].update(face_a=4,face_b=3);save(fixture[1],fixture[2])
    assert run(fixture,tmp_path/'candidate')['unique_pair_count']==2

@pytest.mark.parametrize('fault',['bool-face','negative','overflow','same-face','bool-count','pair-count','pairs-written',
    'truncated','missing-truncated','mapping','scope','unchanged','loaded-count','source-count','path','source-sha','status','has','api','endpoint-bool','endpoint-missing','endpoint-overflow'])
def test_unproven_or_incomplete_diagnostic_is_refused(fixture,tmp_path,fault):
    value=fixture[2]
    if fault=='bool-face':value['pairs'][0]['face_a']=True
    elif fault=='negative':value['pairs'][0]['face_a']=-1
    elif fault=='overflow':value['pairs'][0]['face_a']=6
    elif fault=='same-face':value['pairs'][0]['face_a']=value['pairs'][0]['face_b']
    elif fault=='bool-count':value['pair_count']=True
    elif fault=='pair-count':value['pair_count']=2
    elif fault=='pairs-written':value['pairs_written']=2
    elif fault=='truncated':value['truncated']=True
    elif fault=='missing-truncated':del value['truncated']
    elif fault=='mapping':value['loaded_face_order_matches_source_rows']=False
    elif fault=='scope':value['face_index_scope']='not proven'
    elif fault=='unchanged':value['input_unchanged']=False
    elif fault=='loaded-count':value['loaded_faces']=True
    elif fault=='source-count':value['input']['faces']=5
    elif fault=='path':value['input']['path']=str(tmp_path/'other.ply')
    elif fault=='source-sha':value['input']['sha256']='0'*64
    elif fault=='status':value['status']='INCOMPLETE'
    elif fault=='has':value['has_self_intersections']=1
    elif fault=='api':value['api_exists']=False
    elif fault=='endpoint-bool':value['pairs'][0]['endpoint_a'][0]=False
    elif fault=='endpoint-missing':del value['pairs'][0]['endpoint_a']
    elif fault=='endpoint-overflow':value['pairs'][0]['endpoint_a'][0]=10**400
    save(fixture[1],value);out=tmp_path/'candidate'
    with pytest.raises(ValueError):run(fixture,out)
    assert not (out/'open_surface.ply').exists()

@pytest.mark.parametrize('suffix',[',"bad":NaN',',"bad":1e999',',"truncated":false'])
def test_json_nonfinite_and_duplicate_keys_refused(fixture,tmp_path,suffix):
    record=fixture[1];text=record.read_text();record.write_text(text[:-1]+suffix+'}')
    with pytest.raises(ValueError):run(fixture,tmp_path/'candidate')

@pytest.mark.parametrize('which',['mesh','diagnostic'])
def test_cli_expected_hash_is_required_and_checked(fixture,tmp_path,which):
    mesh,record,*_=fixture
    with pytest.raises(ValueError):repair.repair(mesh,record,'0'*64 if which=='mesh' else digest(mesh),
        '0'*64 if which=='diagnostic' else digest(record),tmp_path/'candidate')

def test_existing_output_refused_without_overwriting(fixture,tmp_path):
    out=tmp_path/'candidate';out.mkdir();sentinel=out/'sentinel';sentinel.write_bytes(b'KEEP')
    with pytest.raises(FileExistsError):run(fixture,out)
    assert sentinel.read_bytes()==b'KEEP'

def test_binding_drift_during_readback_refused(fixture,tmp_path,monkeypatch):
    actual=repair.ply.read_ply;mesh=fixture[0]
    def read(path,*a,**kw):
        result=actual(path,*a,**kw)
        if Path(path).name=='open_surface.ply':mesh.write_bytes(mesh.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(repair.ply,'read_ply',read)
    with pytest.raises(ValueError,match='drift'):run(fixture,tmp_path/'candidate')
    assert not (tmp_path/'candidate/repair_report.json').exists()

def test_cli_refuses_missing_explicit_hashes(fixture,tmp_path):
    with pytest.raises(SystemExit):repair.main(['--mesh',str(fixture[0]),'--diagnostic',str(fixture[1]),'--output',str(tmp_path/'candidate')])

@pytest.mark.parametrize('fault',[None,'hash','unbound','missing-sha'])
def test_explicit_upstream_binding_does_not_upgrade_prior_status(fixture,tmp_path,fault):
    mesh,record,*_=fixture;upstream=tmp_path/'upstream.json'
    value={'status':'TEST_DATA_PREVIOUS_FAILURE','artifacts':[repair.identity.binding(mesh,64*1024**2)]}
    if fault=='unbound':value['artifacts']=[]
    save(upstream,value)
    kw=dict(upstream_report=upstream,expected_upstream_sha='0'*64 if fault=='hash' else digest(upstream))
    if fault=='missing-sha':kw['expected_upstream_sha']=None
    if fault:
        with pytest.raises(ValueError):repair.repair(mesh,record,digest(mesh),digest(record),tmp_path/'candidate',**kw)
    else:
        report=repair.repair(mesh,record,digest(mesh),digest(record),tmp_path/'candidate',**kw)
        assert report['upstream_original_status']=='TEST_DATA_PREVIOUS_FAILURE'
        assert report['upstream_report']['sha256']==digest(upstream)
