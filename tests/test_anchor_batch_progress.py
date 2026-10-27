"""Production batch evidence survives interruption and cannot publish early."""
from copy import deepcopy
import json
import pytest
from integrations.demo_pipeline.batch_evidence import BatchEvidence
from integrations.demo_pipeline import original_anchor_review as geometry
from integrations.demo_pipeline import seed_pair_validation as seed
from test_original_anchor_review import scene
from test_seed_pair_validation import fixture, StubCV


def plan():
    return dict(kind='PLAN',profile=dict(batch_size=32,max_batches=3,max_groups=96),
        selected_groups=['a','b'],omitted_groups=[],batches=[
            dict(batch_id=0,group_ids=['a'],status='NOT_STARTED'),
            dict(batch_id=1,group_ids=['b'],status='NOT_STARTED')])


def event(kind,i):
    return dict(kind=kind,batch_id=i,group_ids=[['a','b'][i]],
        status={'BATCH_BEGIN':'RUNNING','BATCH_END':'COMPLETE','BATCH_FAILED':'FAILED'}[kind])


def test_incomplete_and_failure_never_grant_closing_bindings(tmp_path):
    root=tmp_path/'batch';rec=BatchEvidence(root);rec(plan())
    rec(event('BATCH_BEGIN',0));rec(event('BATCH_END',0));rec(event('BATCH_BEGIN',1))
    state=json.loads((root/'BATCH_STATUS.json').read_text())
    assert state['status']=='INCOMPLETE' and state['completed_batches']==1
    assert (root/'batch_000_end.json').is_file()
    with pytest.raises(ValueError,match='incomplete'):rec.bindings()
    rec(dict(event('BATCH_FAILED',1),failure_type='TimeoutError'))
    assert rec.summary()['status']=='FAILED' and rec.summary()['completed_batches']==1
    with pytest.raises(ValueError):rec(event('BATCH_END',1))
    assert not (root/'batch_001_end.json').exists()


@pytest.mark.parametrize('kind',['duplicate','out_of_order','wrong_groups','bad_partition','bad_budget'])
def test_event_mutations_rejected(tmp_path,kind):
    rec=BatchEvidence(tmp_path/'batch');p=plan()
    if kind=='bad_partition':p['batches'][1]['group_ids']=['a']
    if kind=='bad_budget':p['profile']['max_batches']=4
    if kind in ('bad_partition','bad_budget'):
        with pytest.raises(ValueError):rec(p)
        return
    rec(p)
    if kind=='duplicate':
        with pytest.raises(ValueError):rec(p)
    elif kind=='out_of_order':
        with pytest.raises(ValueError):rec(event('BATCH_BEGIN',1))
    else:
        with pytest.raises(ValueError):rec(dict(event('BATCH_BEGIN',0),group_ids=['c']))


def test_completed_events_are_immutable_and_bound(tmp_path):
    root=tmp_path/'batch';rec=BatchEvidence(root);rec(plan())
    for i in range(2):rec(event('BATCH_BEGIN',i));rec(event('BATCH_END',i))
    assert rec.summary()['status']=='COMPLETE' and rec.summary()['completed_batches']==2
    assert len(rec.bindings())==6
    first=(root/'batch_000_end.json').read_bytes()
    with pytest.raises(ValueError):rec(event('BATCH_END',0))
    assert (root/'batch_000_end.json').read_bytes()==first
    (root/'batch_000_end.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):rec.bindings()


def test_real_geometry_batch_writer_retains_finished_batch_on_deadline(tmp_path):
    f,m,t,c,_=scene(3,tracks=34);rec=BatchEvidence(tmp_path/'geometry');stop=False
    def progress(e):
        nonlocal stop
        rec(e)
        if e['kind']=='BATCH_END':stop=True
    def budget():
        if stop:raise TimeoutError('one shared deadline')
    with pytest.raises(TimeoutError,match='shared deadline'):
        geometry.review_original_anchors(f,m,t,c,budget=budget,progress=progress)
    summary=rec.summary()
    assert summary['completed_batches']==1 and summary['batches'][1]['status']=='FAILED'
    end=json.loads((rec.root/'batch_000_end.json').read_text())
    assert len(end['group_results'])==32
    assert all(g['fit'] is not None for g in end['group_results'])
    assert not (rec.root/'batch_001_end.json').exists()


def test_real_seed_batch_writer_retains_certificates_on_deadline(tmp_path):
    data=fixture();base=data[4]['groups'][0];camera=data[5]['models'][0]
    data[5]['models']=[dict(deepcopy(camera),model_key=f'{i:03d}') for i in range(34)]
    data[4]['groups']=[dict(deepcopy(base),model_key=f'{i:03d}',group_id=f'{i:03d}:7:0') for i in range(34)]
    rec=BatchEvidence(tmp_path/'seed');stop=False
    def progress(e):
        nonlocal stop
        rec(e)
        if e['kind']=='BATCH_END':stop=True
    def budget():
        if stop:raise TimeoutError('one shared deadline')
    cv=StubCV()
    with pytest.raises(TimeoutError,match='shared deadline'):
        seed.validate_seed_pairs(*data[:5],cv,budget,camera_result=data[5],progress=progress)
    assert len(cv.calls)==2  # Same measured pair deduplicated in first batch.
    assert rec.summary()['completed_batches']==1 and rec.summary()['status']=='FAILED'
    end=json.loads((rec.root/'batch_000_end.json').read_text())
    assert len(end['certificates'])==32 and end['flow_calls']==2
    assert not (rec.root/'batch_001_end.json').exists()
