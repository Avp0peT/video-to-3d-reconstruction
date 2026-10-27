from copy import deepcopy
from fractions import Fraction as Q
import pytest
from kozlov_lab.kozlov2025 import four_center_fits as fit
from kozlov_lab.kozlov2025.independent_cover import Budget

A={0:(0,0),1:(3,0),2:(0,2),3:(4,2)}
B={10:(0,0),11:(1,0),12:(0,1),13:(1,1)}
AP=((0,),(1,),(2,),(3,));BP=((10,),(11,),(12,),(13,))

@pytest.fixture(scope='module')
def certificate():
    r=fit.four_center_fits(A,AP,B,BP)
    assert r['status']=='FOUR_CENTER_FITS_SUFFICIENT_WITNESS'
    return r['certificate']

def independent_rows(source,parts,centers,row):
    out=[]
    for owner,part in enumerate(parts):
        target=row['permutation'][owner]
        for label in part:
            point=source[label]
            q=tuple(sum(row['matrix'][k][j]*Q(point[j]) for j in range(2))+row['translation'][k] for k in range(2))
            distances=tuple(sum((q[k]-Q(c[k]))**2 for k in range(2)) for c in centers)
            out.append((label,owner,target,q,distances))
    return tuple(out)

def test_complete_multipt_block_and_independent_strict_checks():
    b={**B,14:(Q(-1,5),0),15:(Q(1,5),0)}
    bp=((10,14,15),(11,),(12,),(13,))
    r=fit.four_center_fits(A,AP,b,bp)
    assert r['status']=='FOUR_CENTER_FITS_SUFFICIENT_WITNESS',r
    c=r['certificate']; ac=c['alignment_certificate']; row=ac['ledger'][c['selected_index']]
    actual=independent_rows(b,bp,c['primary_certificate']['centers'],row)
    assert actual==c['point_checks'] and len(actual)==6
    for _,_,target,_,dist in actual:
        assert dist[target]<c['primary_certificate']['radius_squared']
        assert all(dist[target]<value for i,value in enumerate(dist) if i!=target)

def test_source_to_target_direction_with_noninvolutive_permutation(certificate):
    c=deepcopy(certificate); ac=c['alignment_certificate']
    index=next(i for i,row in enumerate(ac['ledger']) if row['error_squared']==ac['error_squared'] and any(row['permutation'][row['permutation'][j]]!=j for j in range(4)))
    c['selected_index']=index
    c['point_checks']=independent_rows(B,BP,c['primary_certificate']['centers'],ac['ledger'][index])
    assert fit.assess_four_center_fits_certificate(A,B,c)['status']=='VERIFIED'
    bad=deepcopy(c)
    bad['point_checks']=tuple((label,owner,ac['ledger'][index]['permutation'].index(owner),q,d) for label,owner,target,q,d in c['point_checks'])
    assert fit.assess_four_center_fits_certificate(A,B,bad)['status']=='INVALID'

def test_every_global_minimum_tried_when_complete_blocks_fail(certificate):
    b={**B,14:(-100,0),15:(100,0)}; bp=((10,14,15),(11,),(12,),(13,))
    r=fit.four_center_fits(A,AP,b,bp)
    ac=certificate['alignment_certificate']
    expected=tuple(i for i,z in enumerate(ac['ledger']) if z['error_squared']==ac['error_squared'] and z['matrix'][0][0]*z['matrix'][1][1]!=z['matrix'][0][1]*z['matrix'][1][0])
    assert r['status']=='NO_SUPPORTED_WITNESS' and r['tried_ledger_indices']==expected
    assert not r['general_fits_decided']

@pytest.mark.parametrize('point,centers',[
    ((Q(1,2),0),((0,0),(1,0),(0,3),(3,3))),
    ((-1,0),((0,0),(3,0),(0,3),(3,3))),
])
def test_exact_voronoi_and_circle_boundary_rejected(point,centers):
    transform=dict(matrix=((Q(1),Q(0)),(Q(0),Q(1))),translation=(Q(0),Q(0)),mapping=((0,0),))
    valid,checks=fit._containment({10:point},((10,),),centers,Q(1),transform,Budget(work=1000,seconds=10))
    assert not valid and checks==()

def test_verifier_forbids_all_fit_constructors(certificate,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('construction called')
    for name in ('four_center_fits','align_four_centers','solve_exact_center','assess_primary_futlyar'):
        monkeypatch.setattr(fit,name,forbidden)
    assert fit.assess_four_center_fits_certificate(A,B,certificate)['status']=='VERIFIED'

@pytest.mark.parametrize('mutation',('index_bool','parts_bool','point_bool','missing_point','extra_field','epsilon1','delta'))
def test_strict_complete_bound_fields(certificate,mutation):
    c=deepcopy(certificate)
    if mutation=='index_bool': c['selected_index']=False
    elif mutation=='parts_bool': c['reference_parts']=((False,),(1,),(2,),(3,))
    elif mutation=='point_bool':
        row=c['point_checks'][0]; c['point_checks']=((row[0],False,*row[2:]),)+c['point_checks'][1:]
    elif mutation=='missing_point': c['point_checks']=c['point_checks'][:-1]
    elif mutation=='extra_field': c['unbound_claim']='full_group'
    elif mutation=='epsilon1': c['alignment_certificate']['epsilon1']=Q(2)
    else: c['alignment_certificate']['epsilon_point_displacement']*=2
    assert fit.assess_four_center_fits_certificate(A,B,c)['status']=='INVALID'

def test_complete_identity_cannot_drop_unassigned_new_point(certificate):
    assert fit.assess_four_center_fits_certificate(A,{**B,14:(2,3)},certificate)['status']=='INVALID'

def test_common_point_bounded_perturbation_keeps_original_center_minimum(certificate):
    ac=certificate['alignment_certificate']; t=ac['epsilon_point_displacement']/16
    move=lambda points:{i:((1+t)*Q(p[0]),(1-t)*Q(p[1])) for i,p in points.items()}
    aa,bb=move(A),move(B)
    for old,new in ((A,aa),(B,bb)):
        assert all(sum((new[i][k]-Q(p[k]))**2 for k in range(2))<ac['epsilon_point_displacement']**2 for i,p in old.items())
    r=fit.four_center_fits(aa,AP,bb,BP)
    assert r['status']=='FOUR_CENTER_FITS_SUFFICIENT_WITNESS',r
    transformed=r['certificate']['alignment_certificate']
    assert transformed['ledger'][certificate['selected_index']]['error_squared']==transformed['error_squared']
    assert ac['epsilon1']==Q(1)
