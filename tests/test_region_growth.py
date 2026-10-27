from copy import deepcopy
import numpy as np
import pytest
from integrations.demo_pipeline.region_growth import next_region_plan
from integrations.demo_pipeline.region_tracking import make_region_points


def initial(roots=1):
    width=1500;height=600
    domain=dict(width=width,height=height,path='source.png',original_mask_status='OK',mask_rle=[[[0,width]] for _ in range(height)])
    domains={'0':domain,'1':dict(domain,path='next.png',original_mask_status='UNKNOWN_NOT_SEGMENTED',mask_rle=None),
             '2':dict(domain,path='last.png',original_mask_status='UNKNOWN_NOT_SEGMENTED',mask_rle=None)}
    regions=[]
    for i in range(roots):
        center=[200.5+i*150,300.5];rid=f'region:{i}:track:{i}:edge:0'
        regions.append(dict(region_id=rid,depth=0,parent_region_id=None,tracklet_id=f'track:{i}:edge:0',
            source_input_index=27,source_decode_index=0,decode_path=[0,1,2],center_raw_xy=center,
            points=make_region_points(rid,center,domain),selection_cell=i))
    return dict(status='REGION_PLAN_FROZEN',regions=regions,frame_domains=domains,
                profile=dict(max_regions=8,max_legs=4,grid_width=9,grid_step_px=4,lk_width=21,patch_width=13))


def measured(plan):
    regions=deepcopy(plan['regions'])
    delta=np.array([.25,.5])
    for region in regions:
        for p in region['points']:
            if p['status']!='READY':continue
            source=np.array(p['source_raw_xy']);p.update(status='TRACKLET_2D',verified_legs=2,steps=[],observations=[])
            for i,index in enumerate(region['decode_path']):
                raw=(source+i*delta).tolist();p['observations'].append(dict(decode_index=index,raw_xy=raw))
                if not i:continue
                start=source+(i-1)*delta;end=source+i*delta
                def record(value):return dict(xy=(value-.5).tolist(),observed_raw_xy=value.tolist(),status=1,error=0.)
                p['steps'].append(dict(from_decode_index=region['decode_path'][i-1],to_decode_index=index,
                    forward=record(end),backward=record(start),fb_error_px=0.,status='PASSED',
                    destination_mask_status=plan['frame_domains'][str(index)]['original_mask_status']))
        region['affine_diagnostics']=[]
        for a,b in zip(region['decode_path'],region['decode_path'][1:]):
            train=[p for p in region['points'] if p['status']=='TRACKLET_2D' and p['grid_ordinal']%4!=3]
            hold=[p for p in region['points'] if p['status']=='TRACKLET_2D' and p['grid_ordinal']%4==3]
            region['affine_diagnostics'].append(dict(from_decode_index=a,to_decode_index=b,status='EVALUATED',
                train_ids=[p['point_id'] for p in train],holdout_ids=[p['point_id'] for p in hold],rank=3,
                matrix=[[1.,0.,.25],[0.,1.,.5]],
                train=dict(max_px=0.,median_px=0.,residual_pixels=[0.]*len(train)),
                holdout=dict(max_px=0.,median_px=0.,residual_pixels=[0.]*len(hold))))
    return dict(status='REGION_MEASUREMENTS_COMPLETE',regions=regions)


def test_bfs_four_children_fresh_points_identity_and_input_unchanged():
    base=initial();result=measured(base);before=deepcopy((base,result))
    child=next_region_plan(base,[result])
    assert len(child['regions'])==4 and child['growth_statistics']['planned_total']==5
    assert [r['growth_offset_xy'] for r in child['regions']]==[[0,-32],[32,0],[0,32],[-32,0]]
    for region in child['regions']:
        assert region['depth']==1 and region['parent_region_id']==base['regions'][0]['region_id']
        assert region['root_tracklet_id']==base['regions'][0]['tracklet_id'] and region['source_input_index']==27
        assert region['measurement_inherited'] is False and region['physical_sample_independence'] is False
        assert all(p['status']=='READY' and 'observations' not in p and 'verified_legs' not in p for p in region['points'])
    assert child['frame_domains']==base['frame_domains'] and (base,result)==before


def test_depth_two_then_empty_termination_without_new_lk():
    base=initial();r0=measured(base);p1=next_region_plan(base,[r0]);r1=measured(p1)
    p2=next_region_plan(p1,[r0,r1]);assert p2['regions'] and all(r['depth']==2 for r in p2['regions'])
    r2=measured(p2);terminal=next_region_plan(p2,[r0,r1,r2])
    assert terminal['regions']==[] and terminal['growth_statistics']['terminal'] is True
    assert terminal['growth_statistics']['stop_reasons']==['MAX_DEPTH_REACHED']
    assert all(r['depth']<=2 for r in terminal['growth_state']['planned_regions'])


def test_total_twenty_four_cap_with_no_result_refill():
    base=initial(8);result=measured(base);child=next_region_plan(base,[result])
    assert len(child['regions'])==16 and child['growth_statistics']['planned_total']==24
    assert child['growth_statistics']['remaining_regions']==0
    assert any(r['reason']=='TOTAL_REGION_BUDGET_REACHED' for r in child['omitted_centers'])
    for region in child['regions']:
        assert len(region['points'])==81 and len(region['decode_path'])<=5
    rejected=measured(child)
    for region in rejected['regions']:
        for p in region['points']:p['status']='REJECTED'
    terminal=next_region_plan(child,[result,rejected])
    assert terminal['regions']==[] and 'TOTAL_REGION_BUDGET_REACHED' in terminal['growth_statistics']['stop_reasons']


@pytest.mark.parametrize('mutation',['center','status','raw','FB','error','affine','holdrank','missing'])
def test_no_parent_success_by_labels_or_bad_affine_evidence(mutation):
    base=initial();r=measured(base);parent=r['regions'][0]
    if mutation=='center':parent['points'][40]['steps'][1]['forward']['status']=0
    if mutation=='status':parent['points'][40]['status']='REJECTED'
    if mutation=='raw':parent['points'][40]['steps'][1]['forward']['xy'][0]+=1.
    if mutation=='FB':parent['points'][40]['steps'][1]['backward']['xy'][0]+=2.;parent['points'][40]['steps'][1]['backward']['observed_raw_xy'][0]+=2.
    if mutation=='error':parent['points'][40]['steps'][1]['forward']['error']=True
    if mutation=='affine':parent['affine_diagnostics'][1]['holdout']['max_px']=0.1
    if mutation=='holdrank':
        # Three heldouts on one grid column have enough count but rank one.
        keep={3,39,75}
        for point in parent['points']:
            if point['grid_ordinal']%4==3 and point['grid_ordinal'] not in keep:
                point['steps'][0]['forward']['status']=0;point['status']='REJECTED';point['verified_legs']=0
        for diagnostic in parent['affine_diagnostics']:
            diagnostic['holdout_ids']=[p['point_id'] for p in parent['points'] if p['grid_ordinal'] in keep]
            diagnostic['holdout']['residual_pixels']=[0.]*3
    if mutation=='missing':parent['affine_diagnostics'].pop()
    child=next_region_plan(base,[r])
    assert child['regions']==[] and child['growth_statistics']['eligible_parents']==0
    assert 'NO_ELIGIBLE_FRONTIER_PARENTS' in child['growth_statistics']['stop_reasons']
    if mutation=='holdrank':assert all(a['reason']=='AFFINE_SPLIT_RANK' for a in child['growth_statistics']['parent_eligibility'][0]['affine_legs'])


def test_at_least_twelve_final_two_leg_points_required():
    base=initial();r=measured(base)
    for p in r['regions'][0]['points']:
        if p['grid_ordinal'] not in set(range(10))|{40}:p['status']='REJECTED';p['verified_legs']=0
    child=next_region_plan(base,[r])
    eligibility=child['growth_statistics']['parent_eligibility'][0]
    assert eligibility['unique_two_leg_points']==11 and eligibility['center_two_legs'] is True
    assert not child['regions'] and 'FEWER_THAN_TWELVE_UNIQUE_TWO_LEG_POINTS' in eligibility['reasons']


def test_source_mask_boundary_terminates_without_inventing_children():
    base=initial();domain=base['frame_domains']['0'];domain.update(width=100,height=100,
        mask_rle=[[[20,80]] if 20<=y<80 else [] for y in range(100)])
    for d in ('1','2'):base['frame_domains'][d].update(width=100,height=100)
    parent=base['regions'][0];parent['center_raw_xy']=[50.5,50.5]
    parent['points']=make_region_points(parent['region_id'],parent['center_raw_xy'],domain)
    child=next_region_plan(base,[measured(base)])
    assert child['regions']==[] and child['growth_statistics']['eligible_parents']==1
    assert 'SOURCE_MASK_OR_IMAGE_BOUNDARY' in child['growth_statistics']['stop_reasons']


def test_repeated_parent_is_not_reexpanded_and_unplanned_measurement_rejected():
    base=initial();r=measured(base);child=next_region_plan(base,[r])
    again=next_region_plan(child,[r])
    assert again['regions']==[] and 'FRONTIER_EXHAUSTED' in again['growth_statistics']['stop_reasons']
    bad=deepcopy(r);bad['regions'][0]['region_id']='unplanned'
    with pytest.raises(ValueError,match='UNPLANNED_MEASURED_REGION'):next_region_plan(base,[bad])


def test_budget_error_is_not_relabelled_as_bad_parent():
    base=initial();r=measured(base)
    def budget():raise ValueError('shared deadline')
    with pytest.raises(ValueError,match='shared deadline'):next_region_plan(base,[r],budget=budget)


def test_incomplete_measurement_cannot_seed_children():
    base=initial();r=measured(base);r['status']='INTERRUPTED'
    child=next_region_plan(base,[r])
    assert child['regions']==[] and child['growth_statistics']['parent_eligibility'][0]['reasons']==['MEASUREMENT_NOT_COMPLETE']
