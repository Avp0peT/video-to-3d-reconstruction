"""Independent four-point certificates, paired-rank facts and constructor tampering."""
from fractions import Fraction as F
from itertools import permutations,combinations
from copy import deepcopy
from kozlov_lab.stereo.official import bounded_parallel_correspondence as module
from kozlov_lab.stereo.official import parallel_construction as pc


def determinant3(m):
    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
            -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
            +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))


def inputs():
    return ({0:(F(0),F(0)),1:(F(1),F(0)),2:(F(0),F(1)),3:(F(1),F(1))},
            {10:(F(0),F(0)),11:(F(1),F(0)),12:(F(0),F(1)),13:(F(2),F(2))})


def hand_row(source,target,mapping):
    paired={i:target[j] for i,j in mapping}
    joint={i:source[i]+paired[i] for i in source}
    differences=[tuple(joint[i][k]-joint[0][k] for k in range(4)) for i in (1,2,3)]
    cols=next(cols for cols in combinations(range(4),3)
              if determinant3([[row[k] for k in cols] for row in differences]))
    rank=dict(rank=3,origin=0,basis=(1,2,3),columns=cols,
              coordinates=((0,(F(0),F(0),F(0))),(1,(F(1),F(0),F(0))),
                           (2,(F(0),F(1),F(0))),(3,(F(0),F(0),F(1)))))
    body=((0,(F(0),F(0),F(0))),(1,(F(1),F(0),F(0))),
          (2,(F(0),F(1),F(0))),(3,(F(1),F(1),F(1))))
    project2=tuple((paired[1][k]-paired[0][k],paired[2][k]-paired[0][k],
                    paired[3][k]-paired[1][k]-paired[2][k]+paired[0][k],paired[0][k]) for k in (0,1))
    project1=((F(1),F(0),F(0),F(0)),(F(0),F(1),F(0),F(0)))
    align=((F(1),F(0),F(0),F(0)),(F(0),F(1),F(0),F(0)),(F(0),F(0),F(1),F(0)))
    trace=tuple((i,pc.BRANCH_INITIALIZATION,(),'CONSTRUCTED') for i in source)
    cert=dict(body=body,second_body=body,anchors=(0,1,2,3),projections=(project1,project2),
              second_to_first=align,tokens=tuple((i,'L:'+str(i)) for i in source),branch_traces=(trace,trace),
              algorithm='AlgT_prime_both_views',scale_status='unknown')
    return dict(mapping=mapping,rank_certificate=rank,status='ACCEPTED',certificate=cert)


def hand_result(complete=False):
    source,target=inputs()
    mappings=[tuple(zip(source,p)) for p in permutations(target)] if complete else [tuple(zip(source,target))]
    rows=tuple(hand_row(source,target,mapping) for mapping in mappings)
    result=dict(schema=module.SCHEMA,scope=module.SCOPE,status='SEARCH_COMPLETE' if complete else 'PARTIAL_UNKNOWN',
                complete=complete,source=tuple(source.items()),target=tuple(target.items()),true_unique_correspondence=False,
                ledger=rows,accepted=rows,solutions_count=len(rows),unsupported_count=0,work_used=0)
    return source,target,result


def test_partial_conservative_account_cannot_claim_measured_work():
    source,target,result=hand_result()
    row=deepcopy(result['ledger'][0]); del row['certificate'];row['status']='UNKNOWN_BRANCH'
    row['work_accounting']=dict(kind='conservative_unreturned_prepare_allocation',charged_upper_bound=123,
                                actual_counter_available=True)
    result.update(ledger=(row,),accepted=(),solutions_count=0)
    assert module.assess_bounded_parallel_correspondence_result(source,target,result)['status']=='INVALID'


def test_hand_24_mappings_complete_math_oracle_no_search_or_constructor(monkeypatch):
    source,target,result=hand_result(complete=True)
    # Y is not a parallelogram for any opposite-pair choice; therefore none of
    # the 24 correspondences is affine-equivalent to the square X. Four paired
    # points have rank<=3, so every mapping has exact paired rank3.
    y=list(target.values())
    for pairs in (((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))):
        a,b=pairs
        assert tuple(y[a[0]][k]+y[a[1]][k] for k in (0,1))!=tuple(y[b[0]][k]+y[b[1]][k] for k in (0,1))
    def forbidden(*a,**k): raise AssertionError('search/rank/AlgT constructor called')
    for name in ('search_bounded_parallel_correspondence','_rank_witness','_construct','_anchors'):
        monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(pc,'prepare_views',forbidden)
    monkeypatch.setattr(pc,'construct_representative',forbidden)
    assert module.assess_bounded_parallel_correspondence_result(source,target,result)['status']=='VERIFIED'
    assert len(result['ledger'])==len(result['accepted'])==24
    assert result['true_unique_correspondence'] is False
    bad=deepcopy(result);bad.update(ledger=bad['ledger'][:-1],accepted=bad['accepted'][:-1],solutions_count=23)
    assert module.assess_bounded_parallel_correspondence_result(source,target,bad)['status']=='INVALID'
    bad=deepcopy(result);bad['ledger']=bad['ledger'][:-1]+bad['ledger'][:1]
    assert module.assess_bounded_parallel_correspondence_result(source,target,bad)['status']=='INVALID'


def test_actual_constructor_result_tampering_and_cached_types():
    source,target,_=hand_result()
    result=module.search_bounded_parallel_correspondence(source,target,max_candidates=1)
    assert result['status']=='PARTIAL_UNKNOWN' and result['solutions_count']==1
    assert module.assess_bounded_parallel_correspondence_result(source,target,result)['status']=='VERIFIED'
    bad=deepcopy(result);bad['scope']='photos_general_perspective'
    assert module.assess_bounded_parallel_correspondence_result(source,target,bad)['status']=='INVALID'
    bad=deepcopy(result);bad['complete']=True;bad['status']='SEARCH_COMPLETE'
    assert module.assess_bounded_parallel_correspondence_result(source,target,bad)['status']=='INVALID'
    for field in ('body','second_body'):
        bad=deepcopy(result);cached=deepcopy(bad['accepted'][0]);cert=cached['certificate']
        row=list(cert[field][0][1]);row[0]=False
        cert[field]=((cert[field][0][0],tuple(row)),)+cert[field][1:]
        bad['accepted']=(cached,)
        assert module.assess_bounded_parallel_correspondence_result(source,target,bad)['status']=='INVALID'
    bad=deepcopy(result);cached=deepcopy(bad['accepted'][0]);cached['rank_certificate']['rank']=True
    bad['accepted']=(cached,)
    assert module.assess_bounded_parallel_correspondence_result(source,target,bad)['status']=='INVALID'
    bad=deepcopy(result);cert=bad['ledger'][0]['certificate'];rows=list(cert['projections'][1]);row=list(rows[0]);row[-1]+=1
    rows[0]=tuple(row);cert['projections']=(cert['projections'][0],tuple(rows))
    assert module.assess_bounded_parallel_correspondence_result(source,target,bad)['status']=='INVALID'
    bad=deepcopy(result);cert=bad['ledger'][0]['certificate'];row=list(cert['second_to_first'][0]);row[-1]+=1
    cert['second_to_first']=(tuple(row),)+cert['second_to_first'][1:]
    assert module.assess_bounded_parallel_correspondence_result(source,target,bad)['status']=='INVALID'


def test_unreturned_prepare_budget_is_conservative_not_actual(monkeypatch):
    source,target,_=hand_result()
    def exhausted(*a,**k):raise pc.BudgetExceeded('forced audit preparation cap')
    monkeypatch.setattr(pc,'prepare_views',exhausted)
    result=module.search_bounded_parallel_correspondence(source,target,max_work=5000)
    assert result['status']=='PARTIAL_UNKNOWN' and result['complete'] is False
    assert result['work_used']==5000 and result['solutions_count']==0
    row=result['ledger'][0]
    assert row['status']=='UNKNOWN_BRANCH' and 'certificate' not in row
    account=row['work_accounting']
    assert account['kind']=='conservative_unreturned_prepare_allocation'
    assert account['actual_counter_available'] is False
    assert 0<=account['charged_upper_bound']<=result['work_used']
    assert module.assess_bounded_parallel_correspondence_result(source,target,result)['status']=='VERIFIED'


def test_planar_is_outside_domain_and_budget_unknown():
    source,_,_=hand_result()
    target={i+10:p for i,p in source.items()}
    result=module.search_bounded_parallel_correspondence(source,target,max_candidates=1)
    assert result['ledger'][0]['status']=='UNSUPPORTED_PLANAR'
    assert result['unsupported_count']==1 and result['solutions_count']==0
    assert module.assess_bounded_parallel_correspondence_result(source,target,result)['status']=='VERIFIED'
    source,target,result=hand_result()
    assert module.assess_bounded_parallel_correspondence_result(source,target,result,max_work=0)['status']=='UNKNOWN'
    assert module.assess_bounded_parallel_correspondence_result(source,target,result,seconds=0)['status']=='UNKNOWN'
    assert module.assess_bounded_parallel_correspondence_result({**source,4:(F(3),F(1))},target,result)['status']=='INVALID'


import pytest


@pytest.mark.parametrize('case',['extra_field','bad_kind','charged_bool','charged_fraction','negative_charge',
                                 'excess_charge','wrong_status','actual_true'])
def test_conservative_account_strict_types_fields_and_state(case):
    source,target,result=hand_result()
    original=deepcopy(result['ledger'][0])
    row=deepcopy(original);del row['certificate'];row['status']='UNKNOWN_BRANCH'
    account=dict(kind='conservative_unreturned_prepare_allocation',charged_upper_bound=123,actual_counter_available=False)
    row['work_accounting']=account
    result.update(ledger=(row,),accepted=(),solutions_count=0,work_used=123)
    assert module.assess_bounded_parallel_correspondence_result(source,target,result)['status']=='VERIFIED'
    if case=='extra_field':account['measured']=123
    elif case=='bad_kind':account['kind']='measured_cpu_work'
    elif case=='charged_bool':account['charged_upper_bound']=True
    elif case=='charged_fraction':account['charged_upper_bound']=F(123)
    elif case=='negative_charge':account['charged_upper_bound']=-1
    elif case=='excess_charge':account['charged_upper_bound']=124
    elif case=='actual_true':account['actual_counter_available']=True
    else:
        original['work_accounting']=account
        result.update(ledger=(original,),accepted=(original,),solutions_count=1)
    assert module.assess_bounded_parallel_correspondence_result(source,target,result)['status']=='INVALID'


@pytest.mark.parametrize('bad_work',[False,-1,F(0)])
def test_top_work_used_cannot_be_bool_negative_or_fraction(bad_work):
    source,target,result=hand_result()
    result['work_used']=bad_work
    assert module.assess_bounded_parallel_correspondence_result(source,target,result)['status']=='INVALID'
