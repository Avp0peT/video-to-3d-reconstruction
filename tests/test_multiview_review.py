"""Actual native-descriptor/F closure plus bounded graph/report regressions."""
import copy
import json

import numpy as np
import pytest
from PIL import Image

from integrations.demo_pipeline.multiview_review import analyze_multiview, review_multiview
from integrations.demo_pipeline.partial_review import descriptor_proposals, _candidate_geometry


SUPPORTED='EXISTING_F_SUPPORT_CANDIDATE'


def feature(frame, count=2):
    descriptors=np.zeros((count,128),dtype=np.uint8)
    for i in range(count):descriptors[i,i]=128
    return dict(frame_index=frame,points=[dict(id=f'{frame+1}:{i}',feature_row=i,xy=[20.5+3*frame+15*i,30.5+20*i]) for i in range(count)],descriptors=descriptors)


def frozen_geometry(count=3):
    # y'=y epipolar model, independent training endpoints far from tested points.
    return dict(pairs=[dict(frames=[a,b],status='CANDIDATE_SUPPORT_NOT_VERIFIED',
        F=[[0,0,0],[0,0,-1],[0,1,0]],train_ids=['independent'],holdout_ids=[],
        candidates=[dict(stable_id='independent',xy_first=[300.,400.],xy_second=[310.,400.])])
        for a in range(count) for b in range(a+1,count)])


def real_pair(features,a,b,geometry):
    source,target,edges,receipt=descriptor_proposals(features[a],features[b])
    return dict(frames=[a,b],source_points=source,target_points=target,proposal_edges=edges,proposal_receipt=receipt,
        proposal_geometry=_candidate_geometry(dict(mapping=edges),a,b,source,target,geometry))


def check(candidate,first,second,source,target,geometry,**kwargs):
    links=[dict(from_track_id=e['source_id'],to_track_id=e['target_id'],status=SUPPORTED,
                correspondence_verified=False,geometry_acceptance=False,eligible_for_triangulation=False) for e in candidate['mapping']]
    return dict(links=links,counts={SUPPORTED:len(links)})


def simple_pair(a,b,edges,omitted=()):
    edges=[dict(source_id=s,target_id=t,descriptor_distance=.1) for s,t in edges]
    omitted=[dict(source_id=s,target_id=t,descriptor_distance=.2) for s,t in omitted]
    return dict(frames=[a,b],proposal_edges=edges,proposal_receipt=dict(omitted_edges=omitted),
                proposal_geometry=check(dict(mapping=edges),a,b,[],[],{}))


def proposer(edges,omitted=()):
    def propose(a,b):
        row=simple_pair(a['frame_index'],b['frame_index'],edges,omitted)
        return a['points'],b['points'],row['proposal_edges'],row['proposal_receipt']
    return propose


def simple_seed():
    return [simple_pair(0,1,[('1:0','2:0')]),simple_pair(1,2,[('2:0','3:0')])]


def test_real_native_descriptors_and_f_close_without_truth_or_mutation(tmp_path):
    features=[feature(i) for i in range(3)];geometry=frozen_geometry()
    pairs=[real_pair(features,0,1,geometry),real_pair(features,1,2,geometry)]
    before=copy.deepcopy((features,pairs,geometry));frames=[dict(input_index=7+10*i) for i in range(3)]
    grays=[np.arange(80*90,dtype=np.uint8).reshape(80,90) for _ in frames]
    result=review_multiview(features,pairs,geometry,frames,grays,tmp_path/'report',
                             descriptor_proposals=descriptor_proposals,geometry_check=_candidate_geometry)
    assert result['counts']['closure_status']=={'CYCLE_SUPPORTED_CANDIDATE':2}
    assert result['counts']['new_direct_pairs']==1 and result['counts']['unique_closed_seed_triples']==2
    assert result['closing_pair_selection']['selected']==[[0,2]]
    assert all(case['direct_geometry']['sampson_px']==0 for case in result['closures'])
    assert result['competition_geometry_review']['counts']['unique_competitors']==0
    assert result['counts']['competition_case_resolution']=={'NO_SURVIVING_COMPETITOR_IN_TESTED_SCOPE':2}
    assert all(result[k] is False for k in ('identity_verified','geometry_acceptance','native_injected','geometry_refit','observations_synthesized','feature_ids_merged','eligible_for_triangulation'))
    for actual,old in zip(features,before[0]):
        assert actual['points']==old['points'] and np.array_equal(actual['descriptors'],old['descriptors'])
    assert pairs==before[1] and geometry==before[2]
    assert json.loads((tmp_path/'report/MULTIVIEW.json').read_text())['status']=='CANDIDATE_DIAGNOSTIC_COMPLETE'
    assert '未生成新的三维点' in (tmp_path/'report/index.html').read_text(encoding='utf-8')
    with Image.open(next((tmp_path/'report').glob('triple_*.png'))) as image:
        image.load();assert image.size==(990,650)


def test_wrong_direct_edge_never_fills_composed_closure():
    result=analyze_multiview([feature(i) for i in range(3)],simple_seed(),{},
        descriptor_proposals=proposer([('1:0','3:1')]),geometry_check=check)
    assert result['counts']['closed_path_checks']==0
    assert result['closures'][0]['status']=='NO_RETAINED_DIRECT_DESCRIPTOR_EDGE'
    assert result['closures'][0]['direct_match'] is None
    assert not result['observations_synthesized']


@pytest.mark.parametrize('failure',['zeroF','holdout','train_endpoint'])
def test_real_closing_edge_requires_f_evidence(failure):
    features=[feature(i) for i in range(3)];geometry=frozen_geometry()
    pairs=[real_pair(features,0,1,geometry),real_pair(features,1,2,geometry)]
    closing=next(p for p in geometry['pairs'] if p['frames']==[0,2])
    if failure=='zeroF':closing['F']=[[0]*3 for _ in range(3)]
    elif failure=='holdout':closing['status']='HOLDOUT_RESIDUAL_LARGE'
    else:closing['candidates'][0]['xy_first']=features[0]['points'][0]['xy'][:]
    result=analyze_multiview(features,pairs,geometry,descriptor_proposals=descriptor_proposals,geometry_check=_candidate_geometry)
    assert result['counts']['closed_path_checks']==2
    affected=result['closures'] if failure!='train_endpoint' else [c for c in result['closures'] if c['endpoint_a']['feature_id']=='1:0']
    assert all(c['status']=='DESCRIPTOR_CYCLE_WITHOUT_THIRD_EDGE_GEOMETRY_SUPPORT' for c in affected)


def test_omitted_eligible_competition_cannot_be_hidden_by_top48():
    retained=[(f'1:{i}',f'3:{i}') for i in range(48)]
    result=analyze_multiview([feature(i,49) for i in range(3)],simple_seed(),{},
        descriptor_proposals=proposer(retained,[('1:0','3:48')]),geometry_check=check)
    case=result['closures'][0]
    assert case['descriptor_cycle_closed'] and case['status']=='CYCLE_SUPPORTED_BUT_AMBIGUOUS'
    assert case['observed_descriptor_competition']
    assert len(result['direct_pairs'][0]['proposal_edges'])==48
    assert case['competition'][-1]['alternatives'][0]['target_id']=='3:48'


def test_expected_edge_omitted_budget_is_not_absence_or_closure():
    result=analyze_multiview([feature(i) for i in range(3)],simple_seed(),{},
        descriptor_proposals=proposer([],[('1:0','3:0')]),geometry_check=check)
    case=result['closures'][0]
    assert case['status']=='DIRECT_EDGE_OMITTED_BUDGET' and case['expected_edge_eligible_but_omitted']
    assert not case['descriptor_cycle_closed'] and case['direct_match'] is None


def test_same_frame_component_conflict_survives_supported_cycle():
    pairs=simple_seed();pairs[0]=simple_pair(0,1,[('1:0','2:0'),('1:1','2:0')])
    result=analyze_multiview([feature(i) for i in range(3)],pairs,{},
        descriptor_proposals=proposer([('1:0','3:0'),('1:1','3:0')]),geometry_check=check)
    assert all(c['component_conflict_after_closing'] for c in result['closures'])
    assert set(c['status'] for c in result['closures'])=={'CLOSED_CYCLE_COMPONENT_CONFLICT'}


def test_closing_pair_budget_is_frozen_and_does_not_recurse():
    features=[feature(i) for i in range(5)]
    seeds=[simple_pair(i,i+1,[(f'{i+1}:0',f'{i+2}:0')]) for i in range(4)]
    calls=[]
    def direct(a,b):
        calls.append([a['frame_index'],b['frame_index']])
        return proposer([(a['points'][0]['id'],b['points'][0]['id'])])(a,b)
    result=analyze_multiview(features,seeds,{},descriptor_proposals=direct,geometry_check=check,max_closing_pairs=1)
    assert calls==[[0,2]]
    assert result['closing_pair_selection']['needed']==[[0,2],[1,3],[2,4]]
    assert result['closing_pair_selection']['omitted']==[[1,3],[2,4]]
    assert result['counts']['closure_status']['CLOSING_PAIR_NOT_TESTED_BUDGET']==2


@pytest.mark.parametrize('budget',[0,9,True,1.5])
def test_invalid_budget_rejected_before_callback(budget):
    def forbidden(*args):pytest.fail('invalid budget reached descriptor search')
    with pytest.raises(ValueError,match='max_closing_pairs'):
        analyze_multiview([],[],{},descriptor_proposals=forbidden,geometry_check=check,max_closing_pairs=budget)


def test_failed_report_preserves_upstream_and_existing_output(tmp_path):
    features=[feature(i) for i in range(3)];pairs=simple_seed();before=copy.deepcopy(pairs)
    upstream=tmp_path/'upstream.json';upstream.write_text(json.dumps(pairs));raw=upstream.read_bytes()
    def fail(*args):raise ValueError('synthetic descriptor failure')
    result=review_multiview(features,pairs,{},[],[],tmp_path/'failed',descriptor_proposals=fail,geometry_check=check)
    assert result['status']=='FAILED' and 'synthetic descriptor failure' in result['reason']
    assert pairs==before and upstream.read_bytes()==raw
    assert (tmp_path/'failed/index.html').is_file()
    with pytest.raises(FileExistsError):
        review_multiview(features,pairs,{},[],[],tmp_path/'failed',descriptor_proposals=fail,geometry_check=check)
