from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.overlap_regions import build_overlap_plan,assess_overlap


def source_plan(size=80,sources=1):
    domains={};regions=[]
    for source in range(sources):
        start=source*10
        for i in range(3):domains[str(start+i)]=dict(width=size,height=size,path=f'{start+i}.png',
            original_mask_status='OK' if i==0 else 'UNKNOWN_NOT_SEGMENTED',mask_rle=[[[0,size]] for _ in range(size)] if i==0 else None)
        regions.append(dict(source_input_index=27+source,source_decode_index=start,decode_path=[start,start+1,start+2]))
    return dict(regions=regions,frame_domains=domains)


def areas(plan,*,dx=0.,targets=2):
    output=[]
    for region in plan['regions']:
        center=region['center_raw_xy'];roi=dict(left=int(center[0]-.5)-16,top=int(center[1]-.5)-16,width=33,height=33)
        output.append(dict(region_id=region['region_id'],depth=0,tracklet_id=None,
            source_input_index=region['source_input_index'],source_decode_index=region['source_decode_index'],
            decode_path=region['decode_path'],source_center_raw_xy=center,template_roi=roi,
            verified_targets=min(targets,len(region['decode_path'])-1),status='REGION_TRACKLET_2D' if targets>=2 else 'INCOMPLETE',
            region_legs=[dict(source_decode_index=region['source_decode_index'],target_decode_index=t,status='ACCEPTED',
                forward=dict(template_roi=roi,matrix_global_raw3x3=[[1.,0.,dx],[0.,1.,0.],[0.,0.,1.]])) for t in region['decode_path'][1:targets+1]]))
    return output


def test_plan_is_stable_does_not_read_old_motion_and_records_partial_coverage():
    source=source_plan();before=deepcopy(source)
    class Poison:
        def __iter__(self):raise AssertionError('old measurements read')
        def __getitem__(self,key):raise AssertionError('old measurements read')
    source['regions'][0].update(points=Poison(),center_raw_xy=Poison(),motion=Poison())
    plan=build_overlap_plan(source)
    assert plan==build_overlap_plan(before)
    assert all(r['tracklet_id'] is None and r['depth']==0 and 'points' not in r for r in plan['regions'])
    summary=plan['sources'][0]
    assert summary['foreground_pixels']==6400 and summary['planned_union_pixels']<=summary['eligible_union_pixels']<6400
    assert summary['uncovered_foreground_pixels']>0
    assert summary['candidate_count']==25 and summary['candidate_eligible_count']==9
    assert len(plan['regions'])%2==0 and len(plan['regions'])<=64
    for edge in plan['edges']:
        assert edge['shared_foreground_pixels']>=64 and len(edge['control_raw_xy'])==64
        assert all(x%1==.5 and y%1==.5 for x,y in edge['control_raw_xy'])
        assert 'point_id' not in edge
    assert plan['planned_connected_components'][0]['component_count']>=1
    json.dumps(plan,allow_nan=False)


def test_two_source_round_robin_and_global_sixty_four_limit():
    plan=build_overlap_plan(source_plan(size=256,sources=2))
    assert len(plan['regions'])==64
    assert [r['source_input_index'] for r in plan['regions'][:8]]==[27,27,28,28,27,27,28,28]
    assert [s['selected_count'] for s in plan['sources']]==[32,32]
    assert len(plan['planned_connected_components'])==2


def test_same_warp_network_support_and_union_deduplication():
    plan=build_overlap_plan(source_plan());measured=areas(plan,dx=1.25)
    result=assess_overlap(plan,measured)
    assert result['counts']['network_tracklets']==len(plan['regions'])
    assert all(t['status']=='CONSISTENT' and t['max_error_px']==0 for e in result['edges'] for t in e['targets'])
    row=result['coverage'][0]
    assert row['planned_union_pixels']==row['local_two_target_union_pixels']==row['overlap_two_target_union_pixels']
    assert row['planned_union_pixels']<len(plan['regions'])*1089 and row['planned_fraction']<1
    assert result['geometry_accepted'] is False and result['independent_point_matches']==0


def test_translation_difference_two_point_zero_one_conflicts_without_guessing_winner():
    plan=build_overlap_plan(source_plan());edge=plan['edges'][0]
    measured=[a for a in areas(plan) if a['region_id'] in (edge['region_a'],edge['region_b'])]
    for leg in measured[1]['region_legs']:leg['forward']['matrix_global_raw3x3'][0][2]=2.01
    result=assess_overlap(plan,measured)
    target=next(e for e in result['edges'] if e['edge_id']==edge['edge_id'])['targets'][0]
    assert target['status']=='CONFLICT' and target['max_error_px']>2
    assert all(not r['network_tracklet'] for r in result['regions'])
    assert result['coverage'][0]['overlap_two_target_union_pixels']==0


def test_unmeasured_or_second_target_missing_never_becomes_supported_prefix():
    plan=build_overlap_plan(source_plan());result=assess_overlap(plan,[])
    assert len(result['edges'])==len(plan['edges']) and all(t['status']=='UNTESTED' for e in result['edges'] for t in e['targets'])
    assert result['coverage'][0]['planned_union_pixels']>0 and result['coverage'][0]['local_two_target_union_pixels']==0
    one=assess_overlap(plan,areas(plan,targets=1))
    assert one['counts']['network_tracklets']==0 and one['counts']['local_tracklets']==0
    assert all(len(r['overlap_prefix_targets'])==1 for r in one['regions'])


@pytest.mark.parametrize('mutation',['id','source','decode','route','roi','duplicate'])
def test_measurement_identity_injection_is_rejected(mutation):
    plan=build_overlap_plan(source_plan());measured=areas(plan)
    if mutation=='id':measured[0]['region_id']='foreign'
    if mutation=='source':measured[0]['source_input_index']=999
    if mutation=='decode':measured[0]['source_decode_index']=999
    if mutation=='route':measured[0]['decode_path']=[0,2,3]
    if mutation=='roi':measured[0]['template_roi']=dict(left=1,top=1,width=33,height=33)
    if mutation=='duplicate':measured.append(deepcopy(measured[0]))
    with pytest.raises(ValueError):assess_overlap(plan,measured)


def test_conflicting_same_source_route_unknown_mask_and_candidate_bound():
    source=source_plan();source['regions'].append(dict(source['regions'][0],source_decode_index=1,decode_path=[1,2,3]))
    with pytest.raises(ValueError,match='CONFLICTING_SAME_SOURCE_ROUTE'):build_overlap_plan(source)
    source=source_plan();source['frame_domains']['0']['original_mask_status']='UNKNOWN_NOT_SEGMENTED'
    with pytest.raises(ValueError,match='KNOWN_SOURCE_MASK_REQUIRED'):build_overlap_plan(source)
    with pytest.raises(ValueError,match='CANDIDATE_BOUND_4096_EXCEEDED'):build_overlap_plan(source_plan(size=1040))


def test_compatible_prefix_selects_only_longest_declared_route_order_independently():
    source=source_plan();original=deepcopy(source['regions'][0])
    source['frame_domains']['3']=dict(width=80,height=80,path='3.png',original_mask_status='UNKNOWN_NOT_SEGMENTED',mask_rle=None)
    longer=dict(original,decode_path=[0,1,2,3])
    source['regions']=[longer,deepcopy(original),deepcopy(longer),deepcopy(longer)]
    before=deepcopy(source);plan=build_overlap_plan(source)
    assert source==before
    source['regions'].reverse()
    assert plan==build_overlap_plan(source)
    assert before['regions'][1]['decode_path']==[0,1,2]
    assert all(r['decode_path']==[0,1,2,3] for r in plan['regions'])
    provenance=plan['sources'][0]['route_selection']
    assert provenance==dict(policy='LONGEST_DECLARED_COMPATIBLE_PREFIX',selected_decode_path=[0,1,2,3],
        declared_routes=[dict(decode_path=[0,1,2],declaration_count=1),
            dict(decode_path=[0,1,2,3],declaration_count=3)],extended_beyond_declared_route=False)
    assess_overlap(plan,areas(plan))


def test_compatible_prefix_never_creates_missing_frame_domains():
    source=source_plan();source['regions'].append(dict(source['regions'][0],decode_path=[0,1,2,3]))
    with pytest.raises(ValueError,match='MISSING_ROUTE_DOMAIN'):build_overlap_plan(source)


@pytest.mark.parametrize('other_decode,other_path',[(10,[10,9,8]),(11,[11,12,13])])
def test_same_source_reverse_or_different_decode_is_a_real_conflict(other_decode,other_path):
    source=source_plan(sources=2);source['regions']=source['regions'][1:]
    source['regions'].append(dict(source['regions'][0],source_decode_index=other_decode,decode_path=other_path))
    with pytest.raises(ValueError,match='CONFLICTING_SAME_SOURCE_ROUTE'):build_overlap_plan(source)


def test_nonconsecutive_divergence_remains_rejected():
    source=source_plan();source['regions'].append(dict(source['regions'][0],decode_path=[0,1,3]))
    with pytest.raises(ValueError,match='SOURCE_ROUTE_NOT_CONSECUTIVE'):build_overlap_plan(source)


def test_pair_requirement_can_leave_eligible_single_candidate_unselected():
    plan=build_overlap_plan(source_plan(size=33))
    assert plan['regions']==[] and plan['sources'][0]['candidate_eligible_count']==1
    result=assess_overlap(plan,[])
    assert result['coverage'][0]['foreground_pixels']==1089 and result['coverage'][0]['planned_fraction']==0


def test_edge_controls_must_remain_true_shared_source_foreground():
    plan=build_overlap_plan(source_plan());plan['edges'][0]['control_raw_xy'][0]=[79.5,79.5]
    with pytest.raises(ValueError,match='EDGE_REAL_FOREGROUND_CONTROLS_CHANGED'):assess_overlap(plan,areas(plan))


@pytest.mark.parametrize('mutation',['count','status','duplicate_edge','too_many'])
def test_summary_labels_and_graph_bounds_cannot_contradict_evidence(mutation):
    plan=build_overlap_plan(source_plan());measured=areas(plan)
    if mutation=='count':measured[0]['verified_targets']=1
    if mutation=='status':measured[0]['status']='INCOMPLETE'
    if mutation=='duplicate_edge':plan['edges'].append(dict(plan['edges'][0],edge_id='duplicate-pair'))
    if mutation=='too_many':plan['regions']*=9
    with pytest.raises(ValueError):assess_overlap(plan,measured)


@pytest.mark.parametrize('value',[True,float('nan'),float('inf')])
def test_invalid_accepted_transform_cannot_create_agreement(value):
    plan=build_overlap_plan(source_plan());measured=areas(plan)
    measured[0]['region_legs'][0]['forward']['matrix_global_raw3x3'][0][0]=value
    with pytest.raises(ValueError):assess_overlap(plan,measured)


def test_empty_plan_and_budget_propagation():
    plan=build_overlap_plan(dict(regions=[],frame_domains={}))
    assert plan['regions']==[] and assess_overlap(plan,[])['coverage']==[]
    def budget():raise TimeoutError('shared deadline')
    with pytest.raises(TimeoutError,match='shared deadline'):build_overlap_plan(source_plan(),budget=budget)
