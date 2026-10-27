from copy import deepcopy
import json
import time
import numpy as np
import pytest
from integrations.demo_pipeline import intermediate_review as mod


def case():
    frames=[dict(input_index=i,decode_index=i*3,frame_id=str(i),original_mask_status='OK') for i in range(3)]
    def point(g, a, b, status):
        return dict(group_id=g,routes=[dict(source_input_index=a,source_raw_xy=[10.5,10.5],path_input_indices=[a,b],
            steps=[dict(source_input_index=a,target_input_index=b,source_raw_xy=[10.5,10.5],status=status,reason=None)])])
    row=dict(model_key='m',points=[point('a',1,0,'REJECTED'),point('b',1,0,'PASSED')])
    extension=dict(status='VIEW_EXTENSION_LEDGER_COMPLETE',query_rows=[row,deepcopy(row)])
    return frames,extension


def test_plan_deduplicates_but_keeps_both_passing_and_failed_sources():
    frames,ext=case();p=mod.plan_intervals(frames,ext)
    assert len(p['edges'])==1
    e=p['edges'][0]
    assert e['decode_path']==[3,2,1,0] and e['prior_unique_points']==2 and e['prior_failed_points']==1
    assert [v['group_id'] for v in e['points']]==['a','b']
    assert p['decode_indices']==[0,1,2,3]
    assert not p['geometry_accepted'] and not p['thresholds_changed']


def test_inconsistent_duplicate_or_drifted_source_rejected():
    frames,ext=case();ext['query_rows'][1]['points'][0]['routes'][0]['steps'][0]['status']='PASSED'
    with pytest.raises(ValueError,match='inconsistent'):mod.plan_intervals(frames,ext)
    frames,ext=case();ext['query_rows'][0]['points'][0]['routes'][0]['steps'][0]['source_raw_xy']=[11.,11.]
    with pytest.raises(ValueError,match='coordinates'):mod.plan_intervals(frames,ext)


@pytest.mark.parametrize('change,status',[
    (lambda fs:fs[0].update(original_mask_status='NO_MATCH'),'NO_ORIGINAL_ENDPOINT_MASKS'),
    (lambda fs:fs[1].update(decode_index=1),'NO_SUPPORTED_ORIGINAL_GAP')])
def test_unsupported_edges_are_ledgered_not_silently_replaced(change,status):
    fs,e=case();change(fs);p=mod.plan_intervals(fs,e)
    assert not p['edges'] and p['edge_ledger'][0]['status']==status


def test_empty_automatic_stage_never_decodes(tmp_path,monkeypatch):
    from integrations.demo_pipeline import intermediate_decode
    monkeypatch.setattr(intermediate_decode,'decode_intermediates',lambda *a,**k:pytest.fail('must not decode'))
    r=mod.review_intermediates({},[],[],dict(status='VIEW_EXTENSION_LEDGER_COMPLETE',query_rows=[]),tmp_path/'out',[],None,deadline=time.monotonic()+5)
    assert r['status']=='NOT_APPLICABLE_NO_ELIGIBLE_INTERVALS' and r['original_frames_decoded']==0
    assert (tmp_path/'out/index.html').is_file()


def test_orchestration_shared_deadline_masks_partial_and_report(tmp_path,monkeypatch):
    from integrations.demo_pipeline import intermediate_decode,intermediate_tracking,intermediate_report
    fs,ext=case();masks=[np.full((32,32),255,np.uint8) for _ in fs];calls=[];deadline=time.monotonic()+10
    def decode(d,frames,indices,path,refs,**kw):
        assert kw['deadline']==deadline
        calls.append('decode');assert (path.parent/'PLAN.json').is_file()
        path.mkdir();images={}
        from PIL import Image
        for i in indices:
            p=path/f'{i}.png';Image.new('RGB',(32,32)).save(p)
            images[i]=dict(path=str(p),width=32,height=32,gray=np.zeros((32,32),np.uint8),mask=None,
                original_mask_status='UNKNOWN_NOT_SEGMENTED',endpoint_verified=i in (0,3))
        return images,dict(status='fixture')
    def compare(edge,domain,cv,**kw):
        calls.append('compare');assert domain[0]['mask'] is masks[0] and domain[3]['mask'] is masks[1]
        assert domain[1]['mask'] is None and domain[2]['mask'] is None
        return dict(rows=[],counts=dict(RECOVERED=1,LOST=0,BOTH_PASS=0,BOTH_FAIL=1),flow_calls=12)
    monkeypatch.setattr(intermediate_decode,'decode_intermediates',decode)
    monkeypatch.setattr(intermediate_tracking,'compare_edge',compare)
    result=mod.review_intermediates({},fs,masks,ext,tmp_path/'out',[],None,deadline=deadline)
    assert calls==['decode','compare'] and result['counts']['RECOVERED']==1
    assert result['new_intermediate_frames']==2 and not result['geometry_accepted']
    assert all(mod.inputs.fingerprint(x['path'])==x for x in result['source_bindings'])
    saved=json.loads((tmp_path/'out/RESULT.json').read_text())
    assert saved['source_unchanged'] is True
