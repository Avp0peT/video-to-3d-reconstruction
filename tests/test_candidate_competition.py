"""Frozen-F competitor evidence, identity bookkeeping and conservative budgets."""
import copy

import pytest

from integrations.demo_pipeline.candidate_competition import assess_competitors
from integrations.demo_pipeline.multiview_review import _edge
from integrations.demo_pipeline.partial_review import _candidate_geometry
from test_multiview_review import feature, frozen_geometry


SUPPORTED='EXISTING_F_SUPPORT_CANDIDATE'


def fixture(*, support=False, duplicate_case=False, retained=False):
    features=[feature(i,4) for i in range(3)]
    if support:features[2]['points'][1]['xy'][1]=features[0]['points'][0]['xy'][1]
    geometry=frozen_geometry()
    main=dict(source_id='1:0',target_id='3:0',descriptor_distance=.01)
    alternative=dict(source_id='1:0',target_id='3:1',descriptor_distance=.3)
    edges=[main]+([alternative] if retained else [])
    source,target=features[0]['points'],features[2]['points']
    row=dict(frames=[0,2],source_points=source,target_points=target,proposal_edges=edges,
             proposal_receipt=dict(omitted_edges=[] if retained else [alternative]),
             proposal_geometry=_candidate_geometry(dict(mapping=edges),0,2,source,target,geometry))
    case=dict(wedge_index=0,endpoint_a=dict(frame=0,feature_id='1:0'),center=dict(frame=1,feature_id='2:0'),
              endpoint_b=dict(frame=2,feature_id='3:0'),descriptor_cycle_closed=True,
              direct_geometry=dict(status=SUPPORTED),competition=[dict(frames=[0,2],alternatives=[copy.deepcopy(alternative)])])
    cases=[case]
    if duplicate_case:cases.append(dict(copy.deepcopy(case),wedge_index=1))
    supported=[_edge(0,1,'1:0','2:0'),_edge(1,2,'2:0','3:0'),_edge(0,2,'1:0','3:0')]
    return cases,[row],supported,geometry


def test_real_omitted_f_check_unique_vs_incidence_and_input_immutability():
    args=fixture(support=True,duplicate_case=True);before=copy.deepcopy(args);calls=[]
    def actual(*a,**kw):
        calls.append(copy.deepcopy(a[0]['mapping']))
        return _candidate_geometry(*a,**kw)
    result=assess_competitors(*args,geometry_check=actual)
    assert args==before
    assert len(calls)==1 and len(calls[0])==1
    assert result['counts']['occurrences']==2 and result['counts']['unique_competitors']==1
    assert result['counts']['completed_new_checks']==1
    record=result['unique_competitors'][0]
    assert record['key']==[0,2,'1:0','3:1'] and record['geometry']['sampson_px']==0
    assert record['status']==SUPPORTED and record['origin']=='omitted'
    assert all(c['component_conflict'] and c['status']=='SUPPORTED_GRAPH_COMPONENT_CONFLICT' for c in result['cases'])
    assert any(c['same_frame_conflicts'] for c in result['supported_graph']['components'])
    assert all(result[k] is False for k in ('identity_verified','geometry_refit','native_injected','observations_synthesized','eligible_for_triangulation','exhaustive_correspondence_search'))


def test_retained_geometry_is_reused_without_callback():
    args=fixture(retained=True)
    def forbidden(*a,**kw):pytest.fail('retained geometry was redundantly checked')
    result=assess_competitors(*args,geometry_check=forbidden)
    assert result['counts']['reused_checks']==1 and result['counts']['attempted_new_checks']==0
    assert result['unique_competitors'][0]['check_source']=='reused_retained'


def test_actual_f_disagreement_removes_competition_only_in_tested_scope():
    result=assess_competitors(*fixture(),geometry_check=_candidate_geometry)
    assert result['counts']['geometry']=={'GEOMETRY_DISAGREES':1}
    assert result['cases'][0]['status']=='NO_SURVIVING_COMPETITOR_IN_TESTED_SCOPE'
    assert result['cases'][0]['identity_verified'] is False
    assert result['cases'][0]['uniqueness']=='NOT_PROVEN'


@pytest.mark.parametrize('failure',['training','missingF','zeroF','holdout'])
def test_actual_unusable_f_evidence_remains_unresolved(failure):
    args=fixture();closing=next(p for p in args[3]['pairs'] if p['frames']==[0,2])
    if failure=='training':closing['candidates'][0]['xy_first']=args[1][0]['source_points'][0]['xy'][:]
    elif failure=='missingF':closing.pop('F')
    elif failure=='zeroF':closing['F']=[[0]*3 for _ in range(3)]
    else:closing['status']='HOLDOUT_RESIDUAL_LARGE'
    result=assess_competitors(*args,geometry_check=_candidate_geometry)
    assert result['cases'][0]['status']=='COMPETITION_UNRESOLVED'
    assert result['cases'][0]['unresolved_competitors']==1
    assert not result['cases'][0]['identity_verified']


def test_budget_does_not_exclude_unchecked_competitors():
    cases,rows,edges,g=fixture();row=rows[0]
    for i in (2,3):
        alternative=dict(source_id='1:0',target_id=f'3:{i}',descriptor_distance=.3+i*.01)
        row['proposal_receipt']['omitted_edges'].append(alternative)
        cases[0]['competition'][0]['alternatives'].append(copy.deepcopy(alternative))
    result=assess_competitors(cases,rows,edges,g,geometry_check=_candidate_geometry,max_new_edges=1)
    assert result['counts']['requested_new_checks']==3 and result['counts']['attempted_new_checks']==1
    assert result['counts']['budget_omitted']==2
    assert result['new_check_selection']['selected']==[[0,2,'1:0','3:1']]
    assert sum(r['status']=='NOT_CHECKED_BUDGET' for r in result['unique_competitors'])==2
    assert result['cases'][0]['status']=='COMPETITION_UNRESOLVED'
    assert result['cases'][0]['unresolved_competitors']==2


@pytest.mark.parametrize('mode',['wrong_identity','duplicate','missing','exception'])
def test_callback_wrong_identity_or_failure_fails_closed(mode):
    def broken(*a,**kw):
        if mode=='exception':raise RuntimeError('read-only synthetic failure')
        checks=_candidate_geometry(*a,**kw)
        if mode=='wrong_identity':checks['links'][0]['to_track_id']='other:999'
        elif mode=='duplicate':checks['links'].append(copy.deepcopy(checks['links'][0]))
        else:checks['links']=[]
        return checks
    result=assess_competitors(*fixture(),geometry_check=broken)
    assert result['counts']['completed_new_checks']==0
    assert result['counts']['geometry']=={'GEOMETRY_EVALUATION_FAILED':1}
    assert result['cases'][0]['status']=='COMPETITION_UNRESOLVED'
    assert not result['identity_verified']


@pytest.mark.parametrize('budget',[0,257,True,1.5,-1])
def test_invalid_budget_never_reaches_callback(budget):
    def forbidden(*a,**kw):pytest.fail('invalid budget reached geometry')
    with pytest.raises(ValueError,match='max_new_edges'):
        assess_competitors(*fixture(),geometry_check=forbidden,max_new_edges=budget)


def test_unbound_or_changed_alternative_rejected():
    args=fixture();args[0][0]['competition'][0]['alternatives'][0]['descriptor_distance']=.9
    with pytest.raises(ValueError,match='descriptor record differs'):
        assess_competitors(*args,geometry_check=_candidate_geometry)
    args=fixture();args[0][0]['competition'][0]['alternatives'][0]['target_id']='not-enumerated'
    with pytest.raises(ValueError,match='absent'):
        assess_competitors(*args,geometry_check=_candidate_geometry)
