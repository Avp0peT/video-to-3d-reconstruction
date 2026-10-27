"""Given-correspondence exact parallel diagnostic; never searches permutations.

S-1996 AlgT′ is called on both views for rank-three inputs. Joint rank and
its finite certificate are source-derived project controls. Verification proves
input binding, rank and full projection geometry; trace routes receive structure
checks only and do not certify historical execution. The complete flag
only decides the supplied mapping on the complete declared 4..7 point subset.
"""
from . import bounded_parallel_correspondence as bounded
from . import parallel_construction as pc
from ...kozlov2025.independent_cover import BudgetExceeded
from ...kozlov2025.exact_center import _tree

SCHEMA='given-parallel-correspondence-certificate-v1'
RESULT_SCHEMA='given-parallel-correspondence-result-v1'
SCOPE=bounded.SCOPE


def _inputs(source,target,mapping):
    first,second=bounded._input(source),bounded._input(target)
    if type(mapping) is not dict or len(first)!=len(second):
        raise ValueError('explicit complete equal-sized mapping dictionary required')
    if any(type(i) is not int or type(j) is not int for i,j in mapping.items()):
        raise ValueError('integer mapping labels; bool rejected')
    if set(mapping)!=set(first) or set(mapping.values())!=set(second):
        raise ValueError('complete source-to-target bijection required')
    return first,second,tuple(sorted(mapping.items()))


def _same(a,b):
    if type(a) is not type(b):return False
    if type(a) is tuple:return len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    if type(a) is dict:return set(a)==set(b) and all(_same(a[k],b[k]) for k in a)
    return a==b


def _verdict(row):
    return {'REJECTED_RANK4':'INCOMPATIBLE_EXACT_PARALLEL',
            'UNSUPPORTED_PLANAR':'UNSUPPORTED_PLANAR',
            'ACCEPTED':'CONSTRUCTED','UNKNOWN_BRANCH':'UNKNOWN'}[row['status']]


def _certificate(first,second,pairs,row):
    return dict(schema=SCHEMA,scope=SCOPE,source=tuple(first.items()),target=tuple(second.items()),
                mapping=pairs,candidate=row,verdict=_verdict(row),
                four_point_information_insufficient=len(first)==4,
                unknown_mapping_decided=False,photo_scope_decided=False)


def _verify(first,second,pairs,c,budget):
    _tree(c,budget)
    keys={'schema','scope','source','target','mapping','candidate','verdict',
          'four_point_information_insufficient','unknown_mapping_decided','photo_scope_decided'}
    if type(c) is not dict or set(c)!=keys or c['schema']!=SCHEMA or c['scope']!=SCOPE:
        raise ValueError('given mapping schema/fields/scope')
    for key,want in (('source',tuple(first.items())),('target',tuple(second.items())),('mapping',pairs)):
        if not _same(c[key],want):raise ValueError('complete source/target/mapping binding')
    if (type(c['four_point_information_insufficient']) is not bool
            or c['four_point_information_insufficient']!=(len(first)==4)
            or c['unknown_mapping_decided'] is not False or c['photo_scope_decided'] is not False):
        raise ValueError('typed diagnostic scope flags')
    row=c['candidate']
    if type(row) is not dict or row.get('status') not in ('REJECTED_RANK4','UNSUPPORTED_PLANAR','ACCEPTED','UNKNOWN_BRANCH'):
        raise ValueError('candidate status')
    rowkeys={'mapping','rank_certificate','status'} | ({'certificate'} if row['status']=='ACCEPTED' else set())
    if set(row)!=rowkeys or not _same(row['mapping'],pairs):raise ValueError('one complete given mapping candidate')
    rank=row['rank_certificate']
    if type(rank) is not dict or set(rank)!={'rank','origin','basis','columns','coordinates'}:
        raise ValueError('exact rank witness fields')
    if row['status']=='ACCEPTED':
        proof=row['certificate']
        if type(proof) is not dict or set(proof)!={'body','second_body','anchors','projections','second_to_first','tokens','branch_traces','algorithm','scale_status'}:
            raise ValueError('complete AlgT witness fields')
    if c['verdict']!=_verdict(row) or not bounded._verify_candidate(first,second,row,budget):
        raise ValueError('rank/projection/input-binding certificate invalid')
    return c


def assess_given_parallel_input(source_xy,target_xy,mapping,*,max_work=20000000,seconds=30):
    """Rank diagnosis plus actual AlgT′ under one explicitly supplied bijection.

    Exactness is about the declared rational subset only. Four points have no
    additional rank consistency constraint. Rank four does not identify a cause.
    A constructed affine representative has unknown physical scale; no mesh/CAD.
    """
    budget=None
    def finish(status,reason='',certificate=None,**extra):
        decided=status in ('INCOMPATIBLE_EXACT_PARALLEL','UNSUPPORTED_PLANAR','CONSTRUCTED')
        return dict(schema=RESULT_SCHEMA,status=status,reason=reason,complete=decided,
                    complete_given_mapping=decided,unknown_mapping_decided=False,
                    photo_scope_decided=False,certificate=certificate,
                    work_used=budget.used if budget else 0,**extra)
    try:
        budget=bounded._budget(max_work,seconds);budget.spend(0)
        first,second,pairs=_inputs(source_xy,target_xy,mapping)
        paired={i:second[j] for i,j in pairs};joint={i:first[i]+paired[i] for i in first}
        rank=bounded._rank_witness(joint,budget)
        row=dict(mapping=pairs,rank_certificate=rank,
                 status='REJECTED_RANK4' if rank['rank']==4 else 'UNSUPPORTED_PLANAR' if rank['rank']<3 else 'UNKNOWN_BRANCH')
        reason=''
        if rank['rank']==3:
            try:
                row['certificate']=bounded._construct(first,paired,joint,budget)
                row['status']='ACCEPTED'
                if not bounded._verify_candidate(first,second,row,budget):
                    raise ArithmeticError('actual AlgT projection verification failed')
            except bounded.UnreturnedPreparationBudget as exc:
                row.pop('certificate',None);row['status']='UNKNOWN_BRANCH'
                reason=str(exc)
                c=_certificate(first,second,pairs,row)
                return finish('UNKNOWN',reason,c,work_accounting=dict(kind='conservative_unreturned_prepare_allocation',
                    charged_upper_bound=exc.allocated,actual_counter_available=False),
                    four_point_information_insufficient=len(first)==4)
            except (pc.ValidationError,ValueError,ArithmeticError) as exc:
                row.pop('certificate',None);row['status']='UNKNOWN_BRANCH';reason=str(exc)
        c=_certificate(first,second,pairs,row);_verify(first,second,pairs,c,budget)
        return finish(c['verdict'],reason,c,four_point_information_insufficient=len(first)==4,
                      affine_rank=rank['rank'],algorithm_called='AlgT_prime_both_views' if row['status']=='ACCEPTED' else 'NOT_CONSTRUCTED')
    except (BudgetExceeded,pc.BudgetExceeded) as exc:
        return finish('UNKNOWN',str(exc))
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError) as exc:
        return finish('INVALID_INPUT',str(exc))


def assess_given_parallel_certificate(source_xy,target_xy,mapping,certificate,*,max_work=20000000,seconds=30):
    """Independent exact input/rank/projection geometry verification.

    Trace routes are checked structurally, without replaying source branches or
    authenticating historical execution. VERIFIED is not an execution-trace proof.
    No rank search or AlgT construction runs in this verifier.
    """
    budget=None
    try:
        budget=bounded._budget(max_work,seconds);budget.spend(0)
        first,second,pairs=_inputs(source_xy,target_xy,mapping)
        c=_verify(first,second,pairs,certificate,budget)
        return dict(status='VERIFIED',verdict=c['verdict'],work_used=budget.used,
                    verification_scope='exact_input_binding_rank_and_full_projection_geometry',
                    branch_trace_validation='STRUCTURE_ONLY_NOT_EXECUTION_REPLAY',
                    four_point_information_insufficient=c['four_point_information_insufficient'],
                    unknown_mapping_decided=False,photo_scope_decided=False)
    except BudgetExceeded as exc:return dict(status='UNKNOWN',reason=str(exc),work_used=budget.used)
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,AttributeError) as exc:
        return dict(status='INVALID',reason=str(exc),work_used=budget.used if budget else 0)
