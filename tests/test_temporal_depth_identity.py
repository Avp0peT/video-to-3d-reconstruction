from copy import deepcopy
from fractions import Fraction
import pytest
from integrations.demo_pipeline.temporal_depth_report import timeline_join


def data(tmp_path):
    f=[dict(decode_index=5,canonical_mask_path=str(tmp_path/'m.png'),canonical_mask_sha256='a'*64)]
    m=[dict(decode_index=5,path=str(tmp_path/'m.png'),sha256='a'*64,pts=3214,time_base=[1,90000])]
    t=[dict(decode_index=5,pts=3214,time_base=[1,90000])]
    return f,m,t


def test_exact_join_ignores_input_index(tmp_path):
    f,m,t=data(tmp_path);f[0]['input_index']=999;m[0]['input_index']=None
    result,_=timeline_join(f,m,t,[5]);assert result==[Fraction(1607,45000)]


@pytest.mark.parametrize('change',['sha','path','pts','duplicate','missing','bool'])
def test_rejects_identity_change(tmp_path,change):
    f,m,t=data(tmp_path)
    if change=='sha':m[0]['sha256']='b'*64
    if change=='path':m[0]['path']=str(tmp_path/'different.png')
    if change=='pts':m[0]['pts']+=1
    if change=='duplicate':t.append(deepcopy(t[0]))
    if change=='missing':t=[]
    if change=='bool':m[0]['pts']=t[0]['pts']=True
    with pytest.raises((ValueError,RuntimeError)):timeline_join(f,m,t,[5])
