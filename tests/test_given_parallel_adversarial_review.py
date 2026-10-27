from copy import deepcopy
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import struct
import pytest
from kozlov_lab.stereo.official import given_parallel_correspondence as g

BRIDGE=Path(__file__).resolve().parents[1]/'integrations/demo_pipeline/parallel_photo_audit.py'
spec=importlib.util.spec_from_file_location('parallel_audit_adversarial',BRIDGE)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def rank4():
    x=dict(enumerate([(0,0),(1,0),(0,1),(1,1),(2,0)]))
    y=dict(enumerate([(0,0),(1,0),(0,1),(2,1),(1,2)]))
    return x,y,{i:i for i in x}

def constructed():
    body=[(0,0,0),(1,0,0),(0,1,0),(1,1,1),(2,3,2),(3,1,-1)]
    x={i:(p[0],p[1]) for i,p in enumerate(body)}
    y={i:(p[0]+p[2],p[1]+2*p[2]) for i,p in enumerate(body)}
    return x,y,{i:i for i in x}

@pytest.mark.parametrize('attack',['drop_coordinate','bool_rank','dependent_basis','wrong_origin','status_rank3','extra_coordinate'])
def test_rank4_determinant_certificate_forgeries_reject(attack):
    x,y,m=rank4();r=g.assess_given_parallel_input(x,y,m)
    assert r['status']=='INCOMPATIBLE_EXACT_PARALLEL'
    c=deepcopy(r['certificate']);w=c['candidate']['rank_certificate']
    if attack=='drop_coordinate':w['coordinates']=w['coordinates'][:-1]
    elif attack=='bool_rank':w['rank']=True
    elif attack=='dependent_basis':w['basis']=(1,1,2,3)
    elif attack=='wrong_origin':w['origin']=999
    elif attack=='status_rank3':c['candidate']['status']='UNKNOWN_BRANCH';c['verdict']='UNKNOWN'
    else:w['coordinates']=w['coordinates']+(w['coordinates'][0],)
    assert g.assess_given_parallel_certificate(x,y,m,c)['status']=='INVALID'

@pytest.mark.parametrize('bad',[{}, {0:0,1:1,2:2,3:3,4:3}, ((0,0),(1,1),(2,2),(3,3),(4,4)), {0:0,1:1,2:2,3:3,4:True}])
def test_complete_mapping_required(bad):
    x,y,m=rank4()
    assert g.assess_given_parallel_input(x,y,bad)['status']=='INVALID_INPUT'

def test_raw_binary64_is_preserved_not_quantized():
    values=(123.456,-0.125)
    images={1:{'observations':[(values[0],values[1],3)]},2:{'observations':[(1.25,2.5,3)]}}
    points={3:{'track':{1:0,2:0}}}
    first,second=audit.views_for((1,2),[3],images,points)
    assert first[3]==tuple(F.from_float(x) for x in values)
    assert float(first[3][0]).hex()==values[0].hex()
    assert first[3][0]!=F('123.456')

def test_native_full_reciprocal_track_checks(tmp_path):
    ip=tmp_path/'images.bin';pp=tmp_path/'points3D.bin'
    data=struct.pack('<Q',2)
    for i in [1,2]:
        data+=struct.pack('<idddddddi',i,1.,0.,0.,0.,0.,0.,0.,1)
        data+=f'{i}.jpg'.encode()+b'\0'+struct.pack('<Q',1)+struct.pack('<ddq',1.25,2.5,3)
    ip.write_bytes(data)
    point=struct.pack('<Q',1)+struct.pack('<QdddBBBd',3,0.,0.,0.,1,2,3,0.1)+struct.pack('<Q',2)
    pp.write_bytes(point+struct.pack('<ii',1,0)+struct.pack('<ii',2,0))
    images,points=audit.read_native(ip,pp)
    assert points[3]['track']=={1:0,2:0}
    pp.write_bytes(point+struct.pack('<ii',1,0)+struct.pack('<ii',2,1))
    with pytest.raises(ValueError,match='track reference'):audit.read_native(ip,pp)
    pp.write_bytes(point+struct.pack('<ii',1,0)+struct.pack('<ii',1,0))
    with pytest.raises(ValueError,match='duplicate native track element'):audit.read_native(ip,pp)
    pp.write_bytes(point+struct.pack('<ii',1,0)+struct.pack('<ii',2,0)+b'junk')
    with pytest.raises(ValueError,match='trailing'):audit.read_native(ip,pp)

def test_route_forgery_reproduction_is_currently_accepted():
    # Reproduction, not an acceptance requirement: rank/projection remains valid,
    # but no claim that this made-up AlgT execution route was verified is justified.
    x,y,m=constructed();c=deepcopy(g.assess_given_parallel_input(x,y,m)['certificate'])
    traces=list(c['candidate']['certificate']['branch_traces']);t=list(traces[0]);r=list(t[-1])
    r[2]=('MADE_UP_ROUTE',);t[-1]=tuple(r);traces[0]=tuple(t)
    c['candidate']['certificate']['branch_traces']=tuple(traces)
    check=g.assess_given_parallel_certificate(x,y,m,c)
    assert check['status']=='VERIFIED' # documents current semantic limitation
    assert check['branch_trace_validation']=='STRUCTURE_ONLY_NOT_EXECUTION_REPLAY'
