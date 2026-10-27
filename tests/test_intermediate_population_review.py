import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from integrations.demo_pipeline.intermediate_population import build_population_plan
from integrations.demo_pipeline.intermediate_population_review import review_population


class CV:
    TERM_CRITERIA_COUNT=1; TERM_CRITERIA_EPS=2
    def __init__(self,fail_after=None):self.calls=0;self.fail_after=fail_after
    def calcOpticalFlowPyrLK(self,a,b,xy,unused,**kw):
        self.calls+=1
        if self.fail_after is not None and self.calls>self.fail_after:raise RuntimeError('injected native failure')
        delta=int(b[0,0])-int(a[0,0]);out=xy.copy();out[0,0,0]+=delta
        return out,np.ones((1,1),np.uint8),np.zeros((1,1),np.float32)


def fixture(tmp_path,count=3):
    frames=[dict(input_index=x,decode_index=d,frame_id=str(x),original_mask_status='OK',width=64,height=64) for x,d in [(10,0),(20,2)]]
    masks=[np.full((64,64),255,np.uint8) for _ in frames]
    tracks=[dict(track_id=i,observations=[dict(frame_index=0,identity_segment=0,xy=[5.5+i%30,5.5+i//30],label='foreground',target_verified=True,
            tracking_status='seed' if i%2==0 else 'flow_tracked',strong=i%2==1,fb_error_px=None if i%2==0 else .1)]) for i in range(count)]
    edge=dict(model_key='m',source_input_index=10,target_input_index=20,source_decode_index=0,target_decode_index=2,decode_path=[0,1,2],points=[])
    plan=build_population_plan(frames,masks,dict(tracks=tracks),dict(edges=[edge]))
    decoded={}
    for i in range(3):
        p=tmp_path/f'{i}.png';Image.new('RGB',(64,64),(i,i,i)).save(p)
        decoded[i]=dict(path=str(p),width=64,height=64,gray=np.full((64,64),i,np.uint8),mask=None,original_mask_status='UNKNOWN_NOT_SEGMENTED',endpoint_verified=i in (0,2),time_seconds=i/24)
    return plan,frames,masks,decoded


def test_all_frozen_batches_execute_and_publish_bindings(tmp_path):
    p,f,m,d=fixture(tmp_path,65);refs=[];cv=CV()
    r=review_population(p,f,m,d,tmp_path/'review',refs,cv)
    assert r['status']=='POPULATION_COMPARISON_COMPLETE' and r['points']==65
    assert r['counts']=={'BOTH_PASS':65} and cv.calls==585
    assert r['summaries'][0]['by_birth_kind']['NEW_SEED']['evaluated']==33
    assert not r['geometry_accepted'] and not r['intermediate_masks_verified']
    saved=json.loads((tmp_path/'review/RESULT.json').read_text())
    assert len(saved['edges'][0]['batches'])==2
    assert [row['group_id'] for row in saved['edges'][0]['rows']]==[x['point_id'] for x in p['edges'][0]['points']]
    from integrations.demo_pipeline.temporal_review import fingerprint
    assert all(fingerprint(x['path'])==x for x in refs+r['source_bindings'])
    assert (tmp_path/'review/index.html').is_file()


def test_failure_retains_finished_batch_and_real_partial_count(tmp_path):
    p,f,m,d=fixture(tmp_path,65);cv=CV(fail_after=576)
    with pytest.raises(RuntimeError,match='injected native failure'):
        review_population(p,f,m,d,tmp_path/'review',[],cv)
    r=json.loads((tmp_path/'review/FAILURE.json').read_text())
    assert r['flow_calls']==577 and r['counts']=={'BOTH_PASS':64,'INCOMPLETE':1}
    assert len(r['edges'][0]['batches'])==1
    assert (tmp_path/'review/edge_000_batch_000.json').is_file()
    assert not (tmp_path/'review/RESULT.json').exists()
    assert not (tmp_path/'review/index.html').exists()


def test_deadline_is_not_reset_between_batches(tmp_path):
    p,f,m,d=fixture(tmp_path,65);cv=CV()
    def budget():
        if cv.calls>=576:raise TimeoutError('shared deadline')
    with pytest.raises(TimeoutError,match='shared deadline'):
        review_population(p,f,m,d,tmp_path/'review',[],cv,budget=budget)
    r=json.loads((tmp_path/'review/FAILURE.json').read_text())
    assert r['status']=='RESOURCE_LIMIT' and r['flow_calls']==576
    assert len(r['edges'][0]['rows'])==64


def test_empty_plan_still_has_honest_report(tmp_path):
    r=review_population(dict(edges=[]),[],[],{},tmp_path/'review',[],CV())
    assert r['status']=='NOT_APPLICABLE_NO_ELIGIBLE_INTERVALS' and r['points']==0 and r['flow_calls']==0
