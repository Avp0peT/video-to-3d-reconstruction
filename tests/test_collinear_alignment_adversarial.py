"""Hand-computed LP certificates; no producer oracle."""
from fractions import Fraction as F
from copy import deepcopy
import pytest
from kozlov_lab.kozlov2025 import collinear_center_alignment as module


def nonzero_certificate():
    target={0:(F(0),F(0)),1:(F(1),F(0)),2:(F(4),F(0))}
    source={10:(F(0),F(0)),11:(F(0),F(2)),12:(F(0),F(4))}
    dual=((1,F(1,4)),(2,F(1,2)),(5,F(1,4)))
    cert=dict(schema=module.SCHEMA,target=tuple(sorted(target.items())),source=tuple(sorted(source.items())),
              branches=(dict(sign=1,primal=(F(1),F(-1,8),F(1,8)),dual=dual),
                        dict(sign=-1,primal=(F(1),F(7,8),F(1,8)),dual=dual)),chosen=0,
              matrix=((F(0),F(1)),(F(-1),F(0))),translation=(F(-1,2),F(0)),
              mapping=((10,0),(11,1),(12,2)),error_squared=F(1,4),epsilon1=F(1),epsilon2=F(1))
    return target,source,cert


@pytest.mark.parametrize('huge',[F(1<<20000),F(1,1<<20000)])
def test_huge_off_line_witness_coefficient_is_bounded(huge):
    target,source,cert=nonzero_certificate()
    assert module.assess_collinear_alignment_certificate(target,source,cert)['status']=='VERIFIED'
    # This unrestricted off-source-line coefficient changes neither determinant
    # nor any full source point image. It still exceeds witness machine bounds.
    cert['matrix']=((huge,F(1)),(F(-1),F(0)))
    assert module.assess_collinear_alignment_certificate(target,source,cert)['status'] in ('INVALID','UNKNOWN')


def reverse_certificate():
    target,_,cert=nonzero_certificate()
    source={10:(F(0),F(0)),11:(F(0),F(3)),12:(F(0),F(4))}
    cert.update(source=tuple(sorted(source.items())),branches=(
        dict(sign=1,primal=(F(1),F(-1,4),F(1,4)),dual=((1,F(1,8)),(2,F(1,2)),(5,F(3,8)))),
        dict(sign=-1,primal=(F(1),F(1),F(0)),dual=((0,F(0)),(1,F(0)),(7,F(1))))),
        chosen=1,matrix=((F(0),F(-1)),(F(-1),F(0))),translation=(F(4),F(0)),
        mapping=((12,0),(11,1),(10,2)),error_squared=F(0))
    return target,source,cert


def test_hand_nonzero_both_direction_primal_dual_and_no_solver(monkeypatch):
    target,source,cert=nonzero_certificate()
    def forbidden(*a,**k): raise AssertionError('constructor/LP solve called')
    for name in ('align_collinear_centers','_branch','_linear3'):
        monkeypatch.setattr(module,name,forbidden)
    assert module.assess_collinear_alignment_certificate(target,source,cert)['status']=='VERIFIED'
    # Analytic bound: r_mid-(r_first+r_last)/2=1 for physical coordinates,
    # hence every permutation/affine fit after uncrossing has e>=1/2.
    images=[(-F(1,2),F(0)),(F(3,2),F(0)),(F(7,2),F(0))]
    residual=[images[k][0]-target[k][0] for k in (0,1,2)]
    assert residual[1]-(residual[0]+residual[2])/2==1
    assert max(abs(r) for r in residual)==F(1,2)
    assert cert['error_squared']==F(1,4)
    assert module.assess_collinear_alignment_certificate(target,source,cert,max_work=0)['status']=='UNKNOWN'
    assert module.assess_collinear_alignment_certificate(target,source,cert,seconds=0)['status']=='UNKNOWN'


def test_hand_reverse_better_than_forward_exact_dual():
    target,source,cert=reverse_certificate()
    assert module.assess_collinear_alignment_certificate(target,source,cert)['status']=='VERIFIED'
    # Forward physical optimum1, reverse optimum0: both certified independently.
    assert cert['branches'][0]['primal'][2]*4==1
    assert cert['branches'][1]['primal'][2]==0
    bad=deepcopy(cert); bad['chosen']=0
    assert module.assess_collinear_alignment_certificate(target,source,bad)['status']=='INVALID'


def test_common_nonorthogonal_affine_stability_using_hand_certificate():
    target,source,cert=nonzero_certificate()
    # Apply common H to (X,T(Y)), not independently to the initial raw pair.
    def h(point): return (point[0]+2*point[1]+7,2*point[0]+point[1]-3)
    aligned={label:(point[1]-F(1,2),F(0)) for label,point in source.items()}
    target={label:h(point) for label,point in target.items()}
    source={label:h(point) for label,point in aligned.items()}
    cert.update(target=tuple(sorted(target.items())),source=tuple(sorted(source.items())),
                matrix=((F(1),F(0)),(F(0),F(1))),translation=(F(0),F(0)),error_squared=F(5,4))
    assert module.assess_collinear_alignment_certificate(target,source,cert)['status']=='VERIFIED'
    assert cert['error_squared']==F(1,4)*(1**2+2**2)


@pytest.mark.parametrize('case',['sign','primal_bool','dual_bool','duplicate_dual','negative_dual',
                                 'missing_branch','singular','mapping_bool','epsilon_bool','error_bool'])
def test_hand_certificate_tampering_rejected(case):
    target,source,cert=nonzero_certificate()
    if case=='sign': cert['branches'][0]['sign']=True
    elif case=='primal_bool': cert['branches'][0]['primal']=(True,F(-1,8),F(1,8))
    elif case=='dual_bool': cert['branches'][0]['dual']=((True,F(1,4)),(2,F(1,2)),(5,F(1,4)))
    elif case=='duplicate_dual': cert['branches'][0]['dual']=((1,F(1,4)),(1,F(1,2)),(5,F(1,4)))
    elif case=='negative_dual': cert['branches'][0]['dual']=((1,F(-1,4)),(2,F(1,2)),(5,F(1,4)))
    elif case=='missing_branch': cert['branches']=cert['branches'][:1]
    elif case=='singular': cert['matrix']=((F(0),F(1)),(F(0),F(0)))
    elif case=='mapping_bool': cert['mapping']=((10,False),(11,1),(12,2))
    elif case=='epsilon_bool': cert['epsilon1']=True
    else: cert['error_squared']=False
    assert module.assess_collinear_alignment_certificate(target,source,cert)['status']=='INVALID'


def test_complete_input_mismatch_and_transverse_freedom():
    target,source,cert=nonzero_certificate()
    assert module.assess_collinear_alignment_certificate({**target,3:(F(5),F(0))},source,cert)['status']=='INVALID'
    assert module.assess_collinear_alignment_certificate(target,{**source,13:(F(0),F(5))},cert)['status']=='INVALID'
    assert module.assess_collinear_alignment_certificate({**target,1:(F(1),F(1))},source,cert)['status']=='INVALID'
    # Off-line extension is not unique, which is harmless for center-only proof.
    cert['matrix']=((F(7),F(1)),(F(8),F(0)))
    assert module.assess_collinear_alignment_certificate(target,source,cert)['status']=='VERIFIED'
