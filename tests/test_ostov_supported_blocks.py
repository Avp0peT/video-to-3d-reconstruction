import copy
from fractions import Fraction as F
import pytest
from kozlov_lab.kozlov2025.ostov_supported_blocks import given_cover_supported_block_ostov as solve, verify_supported_block_ostov as verify
from kozlov_lab.kozlov2025.exact_triangle_ellipse import exact_triangle_enclosing_ellipse,verify_exact_triangle_ellipse


P = {0:(0,0),1:(3,0),2:(0,3),3:(10,0),4:(12,0),5:(20,0),6:(1,1)}
PARTS = [[0,1,2,6],[3,4],[5]]


def test_noncollinear_given_cover_all_labels():
    r = solve(P,PARTS)
    assert r['status'] == 'EXACT_GIVEN_COVER_OSTOV'
    assert r['centers'] == [(1,1),(11,0),(20,0)]
    assert r['pieces'][0]['size']['expression'] == 'sqrt(2*(trace+sqrt(discriminant)))'
    assert r['pieces'][1]['size']['squared'] == 4
    assert r['cover_size']['expression'] == 'max(nonnegative_lengths)'
    assert verify(P,r)


def test_complete_binding_and_tampering():
    r = solve(P,PARTS)
    for key,value in [('centers',[(0,0)]),('source_coordinate_sha256','fake'),('complete',False),('m',2),('cover_size',{'expression':'max(nonnegative_lengths)','lengths':[]})]:
        bad=copy.deepcopy(r);bad[key]=value
        assert not verify(P,bad)
    bad=copy.deepcopy(r);bad['pieces'][1]['center']=(10,0)
    assert not verify(P,bad)
    bad=copy.deepcopy(r);bad['pieces'][0]['certificate']['center']=(0,0)
    assert not verify(P,bad)
    assert not verify({k:v for k,v in P.items() if k!=6},r)
    assert not verify(dict(P,**{}),dict(r,parts=[[0,1,2],[3,4],[5]]))


def test_partial_and_nonindependent():
    assert solve(P,[[0,1,2],[3,4],[5]])['status']=='INCOMPLETE_COVER'
    q={0:(0,0),1:(4,4),2:(0,4),3:(4,0)}
    r=solve(q,[[0,1],[2,3]])
    assert r['status']=='NOT_INDEPENDENT' and not r['complete']
    r=solve(q,[[0,1,2,3]])
    assert r['status']=='UNSUPPORTED_DOMAIN' and not r['exact_centers']
    assert 'centers' not in r


@pytest.mark.parametrize('kw', [{'max_work':0},{'seconds':0},{'max_work':20}])
def test_shared_budget(kw):
    r=solve(P,PARTS,**kw)
    assert r['status']=='PARTIAL_UNKNOWN' and not r['complete']


@pytest.mark.parametrize('kw', [{'max_work':True},{'seconds':True},{'seconds':31}])
def test_invalid_budget(kw):
    assert solve(P,PARTS,**kw)['status']=='UNSUPPORTED_INPUT'


def test_bool_identity_rejected_in_certificates():
    p={0:(0,0),1:(3,0),2:(0,3)}
    z=exact_triangle_enclosing_ellipse(p)['certificate']
    for key in ('declared_labels','hull_labels'):
        bad=copy.deepcopy(z);bad[key]=[False if i==0 else True if i==1 else i for i in bad[key]]
        assert not verify_exact_triangle_ellipse(p,bad)
    bad=copy.deepcopy(z);bad['barycentric']={False if i==0 else i:w for i,w in bad['barycentric'].items()}
    assert not verify_exact_triangle_ellipse(p,bad)
    r=solve(P,PARTS)
    bad=copy.deepcopy(r);bad['declared_labels'][0]=False
    assert not verify(P,bad)
    bad=copy.deepcopy(r);bad['pieces'][0]['labels'][0]=False
    assert not verify(P,bad)


def test_binary64_segment_size_full_length():
    p={0:(0.1,0),1:(0.2,0),2:(3,1)}
    r=solve(p,[[0,1],[2]])
    assert r['pieces'][0]['center']==((F(0.1)+F(0.2))/2,0)
    assert r['pieces'][0]['size']['squared']==(F(0.2)-F(0.1))**2
    assert verify(p,r)
