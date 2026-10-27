from copy import deepcopy
import pytest
from integrations.demo_pipeline.short_tracklets import extract_tracklets


def fixture(legs=2):
    pid='track:4:segment:2:source:27'
    point=dict(point_id=pid,group_id=pid,track_id=4,identity_segment=2,source_input_index=27,
               source_decode_index=10,source_raw_xy=[5.5,6.5],birth_kind='NEW_SEED',original_strong=False)
    path=list(range(10,11+legs)); steps=[]
    def flow(a,b,start,end):
        return dict(from_decode_index=a,to_decode_index=b,input_raw_xy=start,observed_raw_xy=end,
                    xy=[[[end[0]-.5,end[1]-.5]]],status=[[1]],error=[[0.]])
    for a,b in zip(path,path[1:]):
        steps.append(dict(forward=flow(a,b,[5.5,6.5],[5.5,6.5]),
                          backward=flow(b,a,[5.5,6.5],[5.5,6.5]),fb_error_px=0.))
    edge=dict(points=[point],decode_path=path,source_decode_index=10,target_decode_index=path[-1],
              source_input_index=27,target_input_index=53)
    row=dict(group_id=pid,source_raw_xy=[5.5,6.5],dense=dict(steps=steps,
        fullcycle=dict(status='REJECTED'),reasons=['WHOLE_CYCLE_THRESHOLD']))
    population=dict(plan=dict(edges=[edge]),edges=[dict(rows=[row],source_input_index=27,
                      target_input_index=53,decode_path=path)])
    meta={i:dict(width=20,height=20,path=f'{i}.png',time_seconds=float(i),
                original_mask_status='OK' if i in (path[0],path[-1]) else 'UNKNOWN_NOT_SEGMENTED') for i in path}
    return population,meta


@pytest.mark.parametrize('legs',[2,3])
def test_three_and_four_frames_wholecycle_failure_still_only_candidate(legs):
    args=fixture(legs); original=deepcopy(args)
    record=extract_tracklets(*args)['records'][0]
    assert record['status']=='CANDIDATE_2D_TRACKLET' and len(record['observations'])==legs+1
    assert record['wholecycle_status']=='REJECTED' and not record['geometry_accepted']
    assert record['source_point']['original_strong'] is False
    assert record['observations'][1]['original_mask_status']=='UNKNOWN_NOT_SEGMENTED'
    assert args == original


def test_second_leg_failure_excludes_later_leg_failure_preserves_prefix():
    for failed in (1,2):
        population,meta=fixture(3)
        population['edges'][0]['rows'][0]['dense']['steps'][failed]['forward']['status']=[[0]]
        record=extract_tracklets(population,meta)['records'][0]
        assert record['verified_legs']==failed
        assert record['status']==('EXCLUDED' if failed==1 else 'CANDIDATE_2D_TRACKLET')
        assert record['termination_reason']=='LK_STATUS_FAILED'


@pytest.mark.parametrize('mutation,reason',[
    ('endpoint','FLOW_ENDPOINT_IDENTITY_MISMATCH'),('input','FLOW_INPUT_COORDINATES_MISMATCH'),
    ('nan','INVALID_RAW_LK_OUTPUT'),('error','INVALID_OR_EXCESSIVE_LK_ERROR'),
    ('fb','RECOMPUTED_FB_THRESHOLD'),('missing','MISSING_STEP_OR_INTERRUPTED')])
def test_raw_numeric_and_identity_counterexamples(mutation,reason):
    population,meta=fixture(); dense=population['edges'][0]['rows'][0]['dense']; step=dense['steps'][1]
    if mutation=='endpoint': step['backward']['from_decode_index']=99
    if mutation=='input': step['backward']['input_raw_xy']=[6.5,6.5]
    if mutation=='nan': step['forward']['xy']=[[[float('nan'),6.]]]
    if mutation=='error': step['forward']['error']=[[-1.]]
    if mutation=='fb':
        step['backward']['xy']=[[[7.,6.]]];step['backward']['observed_raw_xy']=[7.5,6.5]
    if mutation=='missing': dense['steps']=dense['steps'][:1]
    record=extract_tracklets(population,meta)['records'][0]
    assert record['status']=='EXCLUDED' and record['termination_reason']==reason


def test_absent_measurement_and_empty_plan():
    population,meta=fixture();population['edges']=[]
    record=extract_tracklets(population,meta)['records'][0]
    assert record['termination_reason']=='NOT_MEASURED_OR_INTERRUPTED'
    assert extract_tracklets(dict(plan=dict(edges=[]),edges=[]),{})['records']==[]


@pytest.mark.parametrize('xy',[[True,False],[True,6.5],['5.5','6.5'],[object(),object()]])
def test_raw_coordinates_require_numeric_types(xy):
    population,meta=fixture()
    population['edges'][0]['rows'][0]['source_raw_xy']=xy
    assert extract_tracklets(population,meta)['records'][0]['termination_reason']=='INVALID_RAW_COORDINATES'


@pytest.mark.parametrize('field',['track_id','identity_segment','source_input_index'])
def test_bool_source_identity_is_excluded(field):
    population,meta=fixture()
    population['plan']['edges'][0]['points'][0][field]=True
    assert extract_tracklets(population,meta)['records'][0]['termination_reason']=='INVALID_SOURCE_IDENTITY_TYPE'


@pytest.mark.parametrize('mutation',['foreign','duplicate','edge'])
def test_structural_foreign_measurement_fails_closed(mutation):
    population,meta=fixture()
    if mutation=='foreign': population['edges'][0]['rows'][0]['group_id']='foreign'
    if mutation=='duplicate': population['edges'][0]['rows']*=2
    if mutation=='edge': population['edges']*=2
    with pytest.raises(ValueError): extract_tracklets(population,meta)


def test_budget_checked_at_edge_point_and_every_leg():
    calls=[]
    extract_tracklets(*fixture(3),budget=lambda:calls.append(1))
    assert len(calls)==5
