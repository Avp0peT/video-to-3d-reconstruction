"""Granularity bounds from complete given m-compatible witnesses (S-2025 PDF12).

An m=u witness proves the exact maximum without deciding smaller m. A lower
witness alone is never promoted to a maximum, etalon set, full or closed image.
"""
from .given_m_compatibility import _group,assess_given_m_compatibility_certificate
from .zero_error_fits import _budget,_call
from .exact_center import _tree
from .independent_cover import BudgetExceeded
from .finite_environment import _canonical_rows

SCHEMA='certified-granularity-v1'


def granularity_from_witnesses(environment,group,witnesses,*,max_work=2000000,seconds=30):
    b=None
    try:
        b=_budget(max_work,seconds);b.spend()
        raw,members=_group(environment,group,0);upper=min(map(len,members))
        if type(witnesses) not in (list,tuple) or len(witnesses)>7:
            raise ValueError('at most seven complete m-compatible witnesses required')
        seen={};certs=[]
        for witness in witnesses:
            check=_call(b,assess_given_m_compatibility_certificate,raw,witness)
            if check['status']=='UNKNOWN':raise BudgetExceeded('compatibility witness unresolved')
            if check['status']!='VERIFIED' or witness['group']!=members:
                raise ValueError('complete witness must certify this exact group')
            m=witness['m']
            if m in seen:raise ValueError('one witness per m required')
            if not 1<=m<=upper:raise ValueError('m exceeds minimum member cardinality')
            seen[m]=witness;certs.append(witness)
        lower=max(seen,default=None)
        exact=lower==upper
        result=dict(schema=SCHEMA,status='EXACT_GRANULARITY_CERTIFIED' if exact else 'BOUNDS_ONLY',
            lower_bound=lower,upper_bound=upper,granularity=upper if exact else None,
            witnessed_m=sorted(seen),undecided_m=[m for m in range(1,upper+1) if m not in seen],
            work_used=b.used,existential_search_performed=False)
        if exact:
            result['certificate']=dict(schema=SCHEMA,environment=tuple(sorted(raw.items())),group=members,
                upper_bound=upper,granularity=upper,upper_bound_reason='nonempty_m_partition_has_m_le_minimum_member_cardinality',
                compatible_witness=seen[upper])
        b.spend(0)
        return result
    except BudgetExceeded as exc:
        return dict(schema=SCHEMA,status='UNKNOWN',reason=str(exc),work_used=b.used if b else 0)
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc:
        return dict(schema=SCHEMA,status='INVALID_WITNESS',reason=str(exc),work_used=b.used if b else 0)


def assess_granularity_certificate(environment,certificate,*,max_work=2000000,seconds=30):
    """Only a tight cardinality upper bound and one independently verified witness."""
    b=None
    try:
        b=_budget(max_work,seconds);b.spend();c=certificate;_tree(c,b)
        if type(c) is not dict or set(c)!={'schema','environment','group','upper_bound','granularity','upper_bound_reason','compatible_witness'} or c['schema']!=SCHEMA:
            raise ValueError('strict granularity certificate fields required')
        raw,members=_group(environment,c['group'],0)
        if not _canonical_rows(c['environment']) or c['environment']!=tuple(sorted(raw.items())) or c['group']!=members:
            raise ValueError('complete environment/group identity changed')
        upper=min(map(len,members))
        if type(c['upper_bound']) is not int or type(c['granularity']) is not int or c['upper_bound']!=upper or c['granularity']!=upper:
            raise ValueError('claimed maximum must attain exact cardinality upper bound')
        if c['upper_bound_reason']!='nonempty_m_partition_has_m_le_minimum_member_cardinality':raise ValueError('upper bound proof missing')
        witness=c['compatible_witness']
        check=_call(b,assess_given_m_compatibility_certificate,raw,witness)
        if check['status']=='UNKNOWN':raise BudgetExceeded('compatibility verification incomplete')
        if check['status']!='VERIFIED' or witness['group']!=members or witness['m']!=upper:
            raise ValueError('tight compatible witness invalid')
        b.spend(0)
        return dict(status='VERIFIED',work_used=b.used)
    except BudgetExceeded as exc:
        return dict(status='UNKNOWN',reason=str(exc),work_used=b.used if b else 0)
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,ArithmeticError) as exc:
        return dict(status='INVALID',reason=str(exc),work_used=b.used if b else 0)
