"""Hand-known whole-group witnesses; no compatibility or fits producer oracle."""
from copy import deepcopy
from fractions import Fraction as F
import pytest
from kozlov_lab.kozlov2025 import certified_granularity as granularity
from kozlov_lab.kozlov2025.exact_triangle_ellipse import _digest
from kozlov_lab.kozlov2025.zero_error_fits import PROOF


def degenerate_result(raw):
    ids=sorted(raw); endpoints=[ids[0]] if len(ids)==1 else [ids[0],ids[-1]]
    a,b=raw[ids[0]],raw[ids[-1]]
    center=tuple((a[k]+b[k])/2 for k in (0,1))
    size=dict(expression='sqrt(length_squared)',length_squared=sum((a[k]-b[k])**2 for k in (0,1)))
    cert=dict(schema='exact-center-degenerate-certificate-v1',kind='point' if len(ids)==1 else 'segment',
              declared_labels=ids,source_coordinate_sha256=_digest(raw),endpoint_labels=endpoints,
              center=center,major_axis_length=deepcopy(size))
    return dict(schema='exact-center-v1',status='EXACT_DEGENERATE_COMPLETE_INPUT',complete=True,
                exact_optimum=True,declared_labels=ids,source_coordinate_sha256=_digest(raw),
                profile=cert['kind'],number_context={'domain':'rational'},center=center,
                major_axis_length=size,certificate=cert)


def hand_group_witness(lower=False):
    values=[0,10,20] if lower else [0,10]
    n=len(values); reference={i:(F(x),F(0)) for i,x in enumerate(values)}
    other={i+n:(F(x+100),F(0)) for i,x in enumerate(values)}
    environment={**reference,**other}
    parts=((0,1),(2,)) if lower else ((0,),(1,))
    centers=[(F(5),F(0)),(F(20),F(0))] if lower else [(F(0),F(0)),(F(10),F(0))]
    radius=F(225) if lower else F(100)
    cover=dict(schema='independent-cover-certificate-v1',contract='independent-cover-v1',
               status='VALID_INDEPENDENT_COVER',coverage_complete=True,decision_complete=True,
               pairwise_independence=True,missing_labels=[],pair_certificates=[
                   dict(left_part=0,right_part=1,kind='strict_separation',axis=(F(1),F(0)))])
    blocks=[dict(labels=list(part),result=degenerate_result({i:reference[i] for i in part})) for part in parts]
    sizes=[dict(kind='squared',squared=row['result']['major_axis_length']['length_squared']) for row in blocks]
    checks=[]
    for owner,part in enumerate(parts):
        for label in part:
            distances=[sum((reference[label][k]-c[k])**2 for k in (0,1)) for c in centers]
            checks.append(dict(label=label,owner=owner,distances_squared=distances,uniquely_nearest=True,strictly_inside=True))
    primary=dict(schema='primary-futlyar-certificate-v1',status='TRUE',parts=[list(p) for p in parts],
                 declared_labels=sorted(reference),source_coordinate_sha256=_digest(reference),cover_certificate=cover,
                 blocks=blocks,centers=centers,radius_squared=radius,size_descriptors=sizes,
                 radius_screens=[True,True],point_checks=checks)
    fits=[]
    for source,shift in ((reference,F(0)),(other,F(-100))):
        source_parts=parts if shift==0 else tuple(tuple(i+n for i in p) for p in parts)
        fits.append(dict(schema='zero-error-fits-v1',proof=PROOF,epsilon1=F(1),epsilon2=F(1),
                         reference_points=tuple(sorted(reference.items())),source_points=tuple(sorted(source.items())),
                         reference_parts=parts,source_parts=source_parts,matrix=((F(1),F(0)),(F(0),F(1))),
                         translation=(shift,F(0)),part_bijection=((0,0),(1,1)),primary_certificate=deepcopy(primary),
                         source_blocks=tuple(degenerate_result({i:source[i] for i in p}) for p in source_parts)))
    group=(tuple(sorted(reference)),tuple(sorted(other)))
    witness=dict(schema='given-m-compatibility-v1',environment=tuple(sorted(environment.items())),group=group,
                 reference_index=0,reference_parts=parts,m=2,shared_primary=primary,member_certificates=tuple(fits))
    return environment,group,witness


def tight_certificate():
    environment,group,witness=hand_group_witness()
    return environment,dict(schema=granularity.SCHEMA,environment=tuple(sorted(environment.items())),group=group,
                            upper_bound=2,granularity=2,
                            upper_bound_reason='nonempty_m_partition_has_m_le_minimum_member_cardinality',
                            compatible_witness=witness)


def test_hand_tight_witness_exact_upper_without_search(monkeypatch):
    environment,group,witness=hand_group_witness()
    answer=granularity.granularity_from_witnesses(environment,group,[witness])
    assert answer['status']=='EXACT_GRANULARITY_CERTIFIED'
    assert answer['granularity']==answer['lower_bound']==answer['upper_bound']==2
    assert answer['undecided_m']==[1]
    assert answer['existential_search_performed'] is False
    environment,cert=tight_certificate()
    def forbidden(*a,**k): raise AssertionError('granularity producer invoked')
    monkeypatch.setattr(granularity,'granularity_from_witnesses',forbidden)
    assert granularity.assess_granularity_certificate(environment,cert)['status']=='VERIFIED'


def test_hand_lower_witness_and_no_witness_are_only_bounds():
    environment,group,witness=hand_group_witness(lower=True)
    answer=granularity.granularity_from_witnesses(environment,group,[witness])
    assert answer['status']=='BOUNDS_ONLY'
    assert answer['lower_bound']==2 and answer['upper_bound']==3
    assert answer['granularity'] is None and 'certificate' not in answer
    assert answer['undecided_m']==[1,3]
    empty=granularity.granularity_from_witnesses(environment,group,[])
    assert empty['status']=='BOUNDS_ONLY' and empty['lower_bound'] is None
    assert empty['undecided_m']==[1,2,3]


@pytest.mark.parametrize('case',['upper_bool','maximum_bool','lower_claim','extra','wrong_group',
                                 'missing_member','m_bool','m1','different_reference'])
def test_hand_granularity_witness_tampering_rejected(case):
    environment,cert=tight_certificate()
    if case=='upper_bool': cert['upper_bound']=True
    elif case=='maximum_bool': cert['granularity']=True
    elif case=='lower_claim': cert['granularity']=1
    elif case=='extra': cert['trust_me']=True
    elif case=='wrong_group': cert['group']=((0,2),(1,3))
    elif case=='missing_member': cert['compatible_witness']['member_certificates']=cert['compatible_witness']['member_certificates'][:1]
    elif case=='m_bool': cert['compatible_witness']['m']=True
    elif case=='m1': cert['compatible_witness']['m']=1
    else: cert['compatible_witness']['reference_index']=1
    assert granularity.assess_granularity_certificate(environment,cert)['status']=='INVALID'


def test_group_identity_environment_empty_m1_and_budget():
    environment,group,witness=hand_group_witness()
    assert granularity.granularity_from_witnesses(environment,list(reversed(group)),[witness])['status']=='INVALID_WITNESS'
    assert granularity.granularity_from_witnesses(environment,group,[witness,witness])['status']=='INVALID_WITNESS'
    assert granularity.granularity_from_witnesses(environment,[],[])['status']=='INVALID_WITNESS'
    singleton={0:(0,0),1:(10,0)}
    answer=granularity.granularity_from_witnesses(singleton,[[0],[1]],[])
    assert answer['status']=='BOUNDS_ONLY' and answer['upper_bound']==1
    assert answer['lower_bound'] is None and answer['undecided_m']==[1]
    environment,cert=tight_certificate()
    assert granularity.assess_granularity_certificate({**environment,6:(F(200),F(0))},cert)['status']=='INVALID'
    assert granularity.assess_granularity_certificate(environment,cert,max_work=0)['status']=='UNKNOWN'
    assert granularity.granularity_from_witnesses(environment,group,[witness],seconds=0)['status']=='UNKNOWN'
