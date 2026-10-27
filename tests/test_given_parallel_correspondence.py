from copy import deepcopy
from fractions import Fraction as F
import json
import pytest
from kozlov_lab.stereo.official import given_parallel_correspondence as g
from kozlov_lab.stereo.official import bounded_parallel_correspondence as b
from kozlov_lab.stereo.official import parallel_construction as pc
from kozlov_lab.kozlov2025.exact_center import encode_exact,decode_exact


def fixture(n=6):
    body=[(0,0,0),(1,0,0),(0,1,0),(1,1,1),(2,3,2),(3,1,-1)]
    ids=[70,20,90,40,30,80]
    x={10+i:(p[0],p[1]) for i,p in enumerate(body[:n])}
    y={ids[i]:(p[0]+p[2],p[1]+2*p[2]) for i,p in enumerate(body[:n])}
    return x,y,{10+i:ids[i] for i in range(n)}


@pytest.mark.parametrize('n',[5,6])
def test_real_algt_for_all_points_both_views_no_permutation_search(n,monkeypatch):
    x,y,m=fixture(n);calls=[];original=pc.construct_representative
    def tracked(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    def forbidden(*a,**k):raise AssertionError('unknown mapping search called')
    monkeypatch.setattr(pc,'construct_representative',tracked)
    monkeypatch.setattr(b,'search_bounded_parallel_correspondence',forbidden)
    result=g.assess_given_parallel_input(x,y,m)
    assert result['status']=='CONSTRUCTED' and result['complete_given_mapping']
    assert result['affine_rank']==3 and calls==[1,1]
    assert result['unknown_mapping_decided'] is False and result['photo_scope_decided'] is False
    c=result['certificate'];proof=c['candidate']['certificate']
    assert len(proof['body'])==n and len(proof['branch_traces'][0])==n
    restored=decode_exact(json.loads(json.dumps(encode_exact(c))))
    assert g.assess_given_parallel_certificate(x,y,m,restored)['status']=='VERIFIED'


def test_independent_verifier_uses_neither_rank_search_nor_construction(monkeypatch):
    x,y,m=fixture();c=g.assess_given_parallel_input(x,y,m)['certificate']
    def forbidden(*args,**kwargs):raise AssertionError('search or constructor called')
    for name in ('_rank_witness','_construct','_anchors','search_bounded_parallel_correspondence'):
        monkeypatch.setattr(b,name,forbidden)
    for name in ('prepare_views','construct_representative'):
        monkeypatch.setattr(pc,name,forbidden)
    monkeypatch.setattr(g,'assess_given_parallel_input',forbidden)
    assert g.assess_given_parallel_certificate(x,y,m,c)['status']=='VERIFIED'


def test_four_points_diagnostic_warns_information_limit():
    x,y,m=fixture(4);r=g.assess_given_parallel_input(x,y,m)
    assert r['status']=='CONSTRUCTED' and r['four_point_information_insufficient'] is True
    assert g.assess_given_parallel_certificate(x,y,m,r['certificate'])['four_point_information_insufficient'] is True


def test_rank4_rejects_only_given_exact_model_and_planar_is_unsupported():
    x=dict(enumerate([(0,0),(1,0),(0,1),(1,1),(2,0)]))
    y=dict(enumerate([(0,0),(1,0),(0,1),(2,1),(1,2)]));m={i:i for i in x}
    r=g.assess_given_parallel_input(x,y,m)
    assert r['status']=='INCOMPATIBLE_EXACT_PARALLEL' and r['affine_rank']==4
    assert r['complete'] and not r['unknown_mapping_decided'] and not r['photo_scope_decided']
    assert g.assess_given_parallel_certificate(x,y,m,r['certificate'])['status']=='VERIFIED'
    square={i:p for i,p in enumerate([(0,0),(1,0),(0,1),(1,1)])};m={i:i for i in square}
    r=g.assess_given_parallel_input(square,square,m)
    assert r['status']=='UNSUPPORTED_PLANAR' and r['affine_rank']==2
    assert g.assess_given_parallel_certificate(square,square,m,r['certificate'])['status']=='VERIFIED'


@pytest.mark.parametrize('kind',['rank','rank_extra','projection','point_drop','mapping','bool_mapping','branch','scope','four_bool','extra','binding'])
def test_forged_complete_certificates_rejected(kind):
    x,y,m=fixture();c=deepcopy(g.assess_given_parallel_input(x,y,m)['certificate'])
    row=c['candidate'];proof=row['certificate']
    if kind=='rank':row['rank_certificate']['rank']=4
    elif kind=='rank_extra':row['rank_certificate']['global_nonaffine']=True
    elif kind=='projection':proof['projections']=(((F(0),)*4,)*2,)+proof['projections'][1:]
    elif kind=='point_drop':proof['body']=proof['body'][:-1]
    elif kind=='mapping':c['mapping']=c['mapping'][:-1]
    elif kind=='bool_mapping':c['mapping']=((False,70),)+c['mapping'][1:]
    elif kind=='branch':proof['branch_traces']=proof['branch_traces'][:1]
    elif kind=='scope':c['photo_scope_decided']=True
    elif kind=='four_bool':c['four_point_information_insufficient']=0
    elif kind=='binding':c['source']=c['source'][:-1]
    else:c['mesh_generated']=True
    assert g.assess_given_parallel_certificate(x,y,m,c)['status']=='INVALID'


@pytest.mark.parametrize('kind',['few','unequal','duplicate','float','bool_coord','bool_label','large_den','large_num','mapping','bool_target','collinear','budget'])
def test_invalid_inputs_never_quantized(kind):
    x,y,m=fixture()
    if kind=='few':x={10:(0,0)}
    elif kind=='unequal':y.pop(70)
    elif kind=='duplicate':x[11]=x[10]
    elif kind=='float':x[10]=(0.0,0)
    elif kind=='bool_coord':x[10]=(False,0)
    elif kind=='bool_label':x[False]=x.pop(10)
    elif kind=='large_den':x[10]=(F(1,10**12+1),0)
    elif kind=='large_num':x[10]=(10**12+1,0)
    elif kind=='mapping':m.pop(10)
    elif kind=='bool_target':m[10]=False
    elif kind=='collinear':x={i:(i,0) for i in x}
    r=g.assess_given_parallel_input(x,y,m,**({'max_work':True} if kind=='budget' else {}))
    assert r['status']=='INVALID_INPUT' and r['certificate'] is None


def test_budget_exhaustion_and_unresolved_branch_are_unknown(monkeypatch):
    x,y,m=fixture()
    for args in ({'max_work':0},{'seconds':0},{'max_work':1}):
        r=g.assess_given_parallel_input(x,y,m,**args)
        assert r['status']=='UNKNOWN' and r['complete'] is False
    c=g.assess_given_parallel_input(x,y,m)['certificate']
    assert g.assess_given_parallel_certificate(x,y,m,c,max_work=0)['status']=='UNKNOWN'
    def blocked(*a,**k):raise ValueError('unsupported branch')
    monkeypatch.setattr(b,'_construct',blocked)
    r=g.assess_given_parallel_input(x,y,m)
    assert r['status']=='UNKNOWN' and not r['complete'] and r['certificate']['candidate']['status']=='UNKNOWN_BRANCH'
    assert g.assess_given_parallel_certificate(x,y,m,r['certificate'])['status']=='VERIFIED'


def test_unreturned_prepare_budget_accounting_is_conservative(monkeypatch):
    x,y,m=fixture()
    def blocked(*a,**k):raise pc.BudgetExceeded('forced pre-return')
    monkeypatch.setattr(pc,'prepare_views',blocked)
    r=g.assess_given_parallel_input(x,y,m,max_work=100000)
    assert r['status']=='UNKNOWN' and r['work_used']==100000
    assert r['work_accounting']['actual_counter_available'] is False
    assert g.assess_given_parallel_certificate(x,y,m,r['certificate'])['status']=='VERIFIED'



def test_trace_routes_are_structure_only_not_historical_execution_proof():
    x,y,m=fixture();original=g.assess_given_parallel_input(x,y,m)['certificate']
    c=deepcopy(original);proof=c['candidate']['certificate']
    trace=proof['branch_traces'][0];label,branch,route,completeness=trace[0]
    proof['branch_traces']=(((label,branch,('MADE_UP_ROUTE',),completeness),)+trace[1:],)+proof['branch_traces'][1:]
    result=g.assess_given_parallel_certificate(x,y,m,c)
    # Route text has no effect on the checked rank/projection equations. This
    # acceptance is intentionally scoped and cannot authenticate AlgT execution.
    assert result['status']=='VERIFIED'
    assert result['verification_scope']=='exact_input_binding_rank_and_full_projection_geometry'
    assert result['branch_trace_validation']=='STRUCTURE_ONLY_NOT_EXECUTION_REPLAY'
    proof['projections']=(((F(0),)*4,)*2,)+proof['projections'][1:]
    assert g.assess_given_parallel_certificate(x,y,m,c)['status']=='INVALID'
