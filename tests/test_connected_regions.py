from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.connected_regions import build_connected_plan,prefix_plan
from integrations.demo_pipeline.overlap_regions import assess_overlap


def source_plan(size=256,sources=1,split=False):
    regions=[];domains={}
    for s in range(sources):
        start=10*s
        regions.append(dict(source_input_index=27+s,source_decode_index=start,decode_path=[start,start+1,start+2]))
        for t in range(start,start+3):
            runs=[[0,size]] if not split else [[0,64],[96,size]]
            domains[str(t)]=dict(width=size,height=size,path=f'{t}.png',original_mask_status='OK' if t==start else 'UNKNOWN_NOT_SEGMENTED',
                mask_rle=[deepcopy(runs) for _ in range(size)] if t==start else None)
    return dict(regions=regions,frame_domains=domains)


def test_spatial_seeds_reach_top_bottom_left_right_before_bridge_or_fill():
    plan=build_connected_plan(source_plan())
    seeds=[s for s in plan['selection_steps'] if s['stage']=='SEED']
    assert len(seeds)==9 and seeds[0]['stratum']==4
    assert {s['stratum'] for s in seeds}==set(range(9))
    assert all(s['stage']=='SEED' for s in plan['selection_steps'][:9])
    selected={r['region_id']:r for r in plan['regions']}
    xy=np.array([selected[s['region_id']]['center_raw_xy'] for s in seeds])
    assert xy[:,0].min()<64 and xy[:,0].max()>192 and xy[:,1].min()<64 and xy[:,1].max()>192
    assert plan['profile']['selection_policy']=='spatial-seeds-bridges-frontier-v1'
    assert len(plan['sources'][0]['spatial_strata'])==9
    assert plan['sources'][0]['selected_component_count']==1
    json.dumps(plan,allow_nan=False)


def test_bridge_paths_reuse_selected_nodes_and_every_hop_shares_real_foreground():
    plan=build_connected_plan(source_plan());items={i['candidate_id']:i for c in plan['candidates'] for i in c['items']}
    selected=set();bridge_count=0;frontier_started=False
    for step in plan['selection_steps']:
        cid=step['candidate_id'];assert cid not in selected
        if step['stage']=='FRONTIER':frontier_started=True
        if step['stage']=='BRIDGE':
            assert not frontier_started;bridge_count+=1
            path=step['declared_path'];assert path[0] in selected and path[-1] in selected
            assert step['parent_candidate_id'] in selected and cid in path
            for a,b in zip(path,path[1:]):
                ra=items[a]['roi'];rb=items[b]['roi']
                w=max(0,min(ra['left']+33,rb['left']+33)-max(ra['left'],rb['left']))
                h=max(0,min(ra['top']+33,rb['top']+33)-max(ra['top'],rb['top']))
                assert w*h>=64 and abs(ra['left']-rb['left'])<=16 and abs(ra['top']-rb['top'])<=16
        if step['stage']=='FRONTIER':assert step['parent_candidate_id'] in selected
        selected.add(cid)
    assert bridge_count>0


def test_disconnected_original_mask_does_not_invent_a_bridge():
    plan=build_connected_plan(source_plan(size=160,split=True))
    assert plan['full_candidate_components'][0]['component_count']==2
    assert plan['planned_connected_components'][0]['component_count']==2
    for edge in plan['edges']:
        assert edge['shared_foreground_pixels']>=64
        assert all(x<64 or x>=96 for x,y in edge['control_raw_xy'])


def test_sixty_four_is_exact_frozen_prefix_and_two_sources_are_balanced():
    source=source_plan(sources=2);before=deepcopy(source)
    full=build_connected_plan(source);prefix=prefix_plan(full,64)
    assert source==before and full['profile']['max_regions']==256
    assert len(full['regions'])==256 and [s['selected_count'] for s in full['sources']]==[128,128]
    assert prefix==build_connected_plan(source,max_regions=64)
    assert prefix['regions']==full['regions'][:64] and prefix['selection_steps']==full['selection_steps'][:64]
    assert prefix['profile']['prefix_only'] and len(prefix['regions'])==64
    ids={r['region_id'] for r in prefix['regions']}
    assert all(e['region_a'] in ids and e['region_b'] in ids for e in prefix['edges'])
    assert full['selection_steps'][:18:2][0]['source_input_index']==27
    assert [s['source_input_index'] for s in full['selection_steps'][:18]]==[27,28]*9
    with pytest.raises(ValueError,match='PREFIX_CANNOT_EXPAND_PLAN'):prefix_plan(prefix,256)


def test_original_input_order_or_old_motion_does_not_change_selection():
    source=source_plan(sources=2);expected=build_connected_plan(source)
    class Poison:
        def __iter__(self):raise AssertionError('old measurements read')
        def __getitem__(self,key):raise AssertionError('old measurements read')
    source['regions'].reverse()
    for region in source['regions']:region.update(points=Poison(),center_raw_xy=Poison(),motion=Poison())
    assert expected==build_connected_plan(source)


def test_coverage_union_is_deduplicated_and_unmeasured_never_supported():
    plan=build_connected_plan(source_plan(),max_regions=64)
    mask=np.zeros((256,256),dtype=bool)
    for r in plan['regions']:
        x=int(r['center_raw_xy'][0]-.5)-16;y=int(r['center_raw_xy'][1]-.5)-16
        mask[y:y+33,x:x+33]=True
    assert plan['sources'][0]['planned_union_pixels']==int(mask.sum())<len(plan['regions'])*1089
    result=assess_overlap(plan,[])
    assert result['coverage'][0]['planned_union_pixels']==int(mask.sum())
    assert result['coverage'][0]['local_two_target_union_pixels']==0
    assert result['coverage'][0]['overlap_two_target_union_pixels']==0
    assert result['counts']['network_tracklets']==0
    assert all(t['status']=='UNTESTED' for e in result['edges'] for t in e['targets'])
    assert plan['geometry_accepted'] is False and plan['independent_point_matches']==0


def test_unknown_source_and_route_drift_remain_blocked():
    source=source_plan();source['frame_domains']['0']['original_mask_status']='UNKNOWN_NOT_SEGMENTED'
    with pytest.raises(ValueError,match='KNOWN_SOURCE_MASK_REQUIRED'):build_connected_plan(source)
    source=source_plan();source['regions'].append(dict(source['regions'][0],source_decode_index=1,decode_path=[1,2,3]))
    with pytest.raises(ValueError,match='CONFLICTING_SAME_SOURCE_ROUTE'):build_connected_plan(source)


@pytest.mark.parametrize('limit',[True,0,65,128,257])
def test_only_frozen_region_budgets_are_allowed(limit):
    with pytest.raises(ValueError,match='CONNECTED_REGION_BOUND'):build_connected_plan(source_plan(),max_regions=limit)


def test_budget_exception_propagates_and_empty_plan_is_honest():
    class Expired(Exception):pass
    calls=[]
    def budget():
        calls.append(1)
        if len(calls)==5:raise Expired()
    with pytest.raises(Expired):build_connected_plan(source_plan(),budget=budget)
    assert len(calls)==5
    plan=build_connected_plan(dict(regions=[],frame_domains={}))
    assert plan['regions']==plan['edges']==plan['selection_steps']==[]
    assert plan['selection_counts']['remaining_region_slots']==256
