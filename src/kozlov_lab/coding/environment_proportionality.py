"""S-2019 p58 area proportionality under an explicitly supplied bijection.

Finite exact definition-level implementation, without unknown-mapping search
or construction of an affine transformation. Cross ratios are source/target.
"""
from fractions import Fraction as F
from itertools import combinations
import time

from . import environment_code as ec
from ..registration.fixed_orientation import _Budget, _Exhausted, _label

SCHEMA='environment-proportionality-given-mapping-v1'
SOURCE='S-2019-CODES:58'


def _input(environment,source_labels,target_labels,mapping):
    points=ec._points(environment)
    source=ec._labels(source_labels,points);target=ec._labels(target_labels,points)
    if type(mapping) is not dict:raise ValueError('explicit complete mapping dictionary required')
    for i,j in mapping.items():_label(i);_label(j)
    if (set(mapping)!=set(source) or set(mapping.values())!=set(target)
            or len(source)!=len(target)):
        raise ValueError('complete source-to-target bijection required')
    return points,source,target,tuple(sorted(mapping.items()))


def _remaining(budget):
    return dict(max_work=max(0,budget.maximum-budget.used),seconds=max(0,budget.deadline-time.monotonic()))


def _charge(result,budget):
    budget.spend(result['work_used'])
    if result['status']=='UNKNOWN':raise _Exhausted('nested environment budget exhausted')
    return result


def _area(coords,triple):
    a,b,c=(coords[i] for i in triple)
    return abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))


def _construct_ledger(env,source,mapping,budget):
    ledger=[];constant=None;positive=True;defined=0;preserved=True
    for triple in combinations(source,3):
        budget.spend(20)
        mapped=tuple(sorted(mapping[i] for i in triple))
        sk,tk=ec._key(triple),ec._key(mapped)
        sa,ta=env['double_areas'][sk],env['double_areas'][tk]
        pair=sk+'/'+tk
        preserved=preserved and ((sa==0)==(ta==0))
        if sa==0:
            ratio='NOT_QUANTIFIED';reference='SOURCE_DEGENERATE_NOT_QUANTIFIED'
        elif ta==0:
            defined+=1;ratio='UNDEFINED';reference='TARGET_ZERO_DENOMINATOR';positive=False
        else:
            defined+=1;ratio=sa/ta
            reference='SELF_PAIR_EXCLUDED' if triple==mapped else pair
            if constant is None:constant=ratio
            if ta==0 or ratio!=constant:positive=False
        ledger.append(dict(source_triple=triple,mapped_triple=mapped,pair=pair,
                           source_double_area=sa,target_double_area=ta,ratio=ratio,
                           environment_ratio_reference=reference))
    return tuple(ledger),constant if constant is not None else 'NO_POSITIVE_RATIO',positive,defined,preserved


def _verify(points,source,target,pairs,c,budget):
    ec._tree(c,budget)
    fields={'schema','source','environment','source_labels','target_labels','mapping',
            'ledger','constant','source_domain','target_domain','theorem_eligible','verdict',
            'paper_proportionality_given_mapping','degeneracy_preserved','full_area_proportionality'}
    if type(c) is not dict or set(c)!=fields or c['schema']!=SCHEMA or c['source']!=SOURCE:
        raise ValueError('proportionality schema/fields/source')
    for key,value in (('source_labels',source),('target_labels',target),('mapping',pairs)):
        if not ec._same(c[key],value):raise ValueError('complete global part/mapping binding')
    checked=_charge(ec.assess_environment_code(dict(points),c['environment'],**_remaining(budget)),budget)
    if checked['status']!='VERIFIED':raise ValueError('invalid embedded environment code')
    coords=dict(points);mapping=dict(pairs)
    expected=[];constant=None;positive=True;defined=0;preserved=True
    # Independent expanded determinant identities; do not invoke ledger builder.
    for triple in combinations(source,3):
        budget.spend(30)
        mapped=tuple(sorted(mapping[i] for i in triple))
        def expanded(ids):
            (ax,ay),(bx,by),(cx,cy)=(coords[i] for i in ids)
            return abs(ax*by+bx*cy+cx*ay-ay*bx-by*cx-cy*ax)
        sa,ta=expanded(triple),expanded(mapped)
        sk,tk=ec._key(triple),ec._key(mapped);pair=sk+'/'+tk
        preserved=preserved and ((sa==0)==(ta==0))
        if sa==0:
            ratio='NOT_QUANTIFIED';reference='SOURCE_DEGENERATE_NOT_QUANTIFIED'
        elif ta==0:
            defined+=1;ratio='UNDEFINED';reference='TARGET_ZERO_DENOMINATOR';positive=False
            if pair in c['environment']['ratios']:raise ValueError('zero denominator unexpectedly stored')
        else:
            defined+=1;ratio=sa/ta
            if constant is None:constant=ratio
            if ratio!=constant or ta==0:positive=False
            if triple==mapped:
                reference='SELF_PAIR_EXCLUDED'
                if pair in c['environment']['ratios'] or ratio!=1:
                    raise ValueError('self pair is excluded; identical nonzero triple ratio must be one')
            else:
                reference=pair
                if c['environment']['ratios'].get(pair)!=ratio:
                    raise ValueError('cross ratio missing or wrong in global environment')
        expected.append(dict(source_triple=triple,mapped_triple=mapped,pair=pair,
            source_double_area=sa,target_double_area=ta,ratio=ratio,environment_ratio_reference=reference))
    if not defined:raise ValueError('source has no nonzero source triangle; no predicate evidence')
    sd=ec._domain(tuple(row for row in points if row[0] in source),budget)
    td=ec._domain(tuple(row for row in points if row[0] in target),budget)
    verdict='PROPORTIONAL_GIVEN_MAPPING' if positive else 'NOT_PROPORTIONAL_GIVEN_MAPPING'
    expected_fields=dict(ledger=tuple(expected),constant=constant if constant is not None else 'NO_POSITIVE_RATIO',
        source_domain=sd,target_domain=td,theorem_eligible=positive and sd=='plane' and td=='plane',verdict=verdict,
        paper_proportionality_given_mapping=positive,degeneracy_preserved=preserved,
        full_area_proportionality=positive and preserved)
    if any(not ec._same(c[k],v) for k,v in expected_fields.items()):
        raise ValueError('complete required ledger/constant/domain/verdict mismatch')
    return c


def paper_proportionality_given_mapping(environment,source_labels,target_labels,mapping,*,max_work=2000000,seconds=30):
    """Exact given-mapping area predicate, preserving complete global indices."""
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        points,source,target,pairs=_input(environment,source_labels,target_labels,mapping)
        if not any(_area(dict(points),triple)>0 for triple in combinations(source,3)):
            raise ValueError('source has no nonzero area triple; predicate unsupported')
        env=_charge(ec.build_environment_code(dict(points),**_remaining(budget)),budget)
        if env['status']!='EXACT_ENVIRONMENT_CODE':raise ValueError('environment unsupported')
        ledger,constant,positive,defined,preserved=_construct_ledger(env['certificate'],source,dict(pairs),budget)
        sd=ec._domain(tuple(row for row in points if row[0] in source),budget)
        td=ec._domain(tuple(row for row in points if row[0] in target),budget)
        verdict='PROPORTIONAL_GIVEN_MAPPING' if positive else 'NOT_PROPORTIONAL_GIVEN_MAPPING'
        c=dict(schema=SCHEMA,source=SOURCE,environment=env['certificate'],source_labels=source,target_labels=target,
            mapping=pairs,ledger=ledger,constant=constant,source_domain=sd,target_domain=td,
            theorem_eligible=positive and sd=='plane' and td=='plane',verdict=verdict,
            paper_proportionality_given_mapping=positive,degeneracy_preserved=preserved,
            full_area_proportionality=positive and preserved)
        _verify(points,source,target,pairs,c,budget)
        return dict(status=verdict,certificate=c,work_used=budget.used,
                    theorem_eligible=c['theorem_eligible'],affine_matrix_constructed=False,
                    unknown_mapping_decided=False,paper_proportionality_given_mapping=positive,
                    degeneracy_preserved=preserved,full_area_proportionality=positive and preserved)
    except _Exhausted as e:
        return dict(status='UNKNOWN',reason=str(e),certificate=None,work_used=budget.used)
    except (ValueError,TypeError,KeyError,ArithmeticError) as e:
        return dict(status='UNSUPPORTED',reason=str(e),certificate=None,work_used=budget.used if budget else 0)


# Explicitly documented alias: status is the paper's one-way predicate; consume
# full_area_proportionality when the stronger all-triple engineering gate is needed.
proportionality_given_mapping=paper_proportionality_given_mapping


def assess_proportionality_certificate(environment,source_labels,target_labels,mapping,certificate,*,max_work=2000000,seconds=30):
    """Independent complete-input verifier; never invokes any constructor."""
    budget=None
    try:
        budget=_Budget(max_work,seconds);budget.spend(0)
        points,source,target,pairs=_input(environment,source_labels,target_labels,mapping)
        c=_verify(points,source,target,pairs,certificate,budget)
        return dict(status='VERIFIED',verdict=c['verdict'],theorem_eligible=c['theorem_eligible'],work_used=budget.used,
            paper_proportionality_given_mapping=c['paper_proportionality_given_mapping'],
            degeneracy_preserved=c['degeneracy_preserved'],full_area_proportionality=c['full_area_proportionality'])
    except _Exhausted as e:
        return dict(status='UNKNOWN',reason=str(e),work_used=budget.used)
    except (ValueError,TypeError,KeyError,ArithmeticError) as e:
        return dict(status='INVALID',reason=str(e),work_used=budget.used if budget else 0)
