import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest
from integrations.demo_pipeline.pipeline import sha
from integrations.video_masks.full_pipeline import assemble, gather_predictions


def fixture(tmp_path):
    timeline=tmp_path/'timeline.json'
    timeline.write_text(json.dumps({'frames':[dict(decode_index=i,pts=i,time_base=[1,28]) for i in range(5)]}))
    mask=tmp_path/'a.png';Image.new('L',(4,6),255).save(mask)
    original=dict(decode_index=0,input_index=0,original_status='OK',mask_path=str(mask),mask_sha256=sha(mask))
    plan=dict(width=4,height=6,expected_decode_indexes=list(range(5)),timeline_path=str(timeline),
              original_anchors=[original],coverage_ledger=[original],intervals=[],tasks=[])
    return plan,mask


def prediction(mask,direction='forward',root=0,depth=0):
    return dict(direction=direction,root_seed_decode_index=root,chain_depth=depth,task_id='task_000',
                evidence_origin='TEST_DATA',mask_path=str(mask),mask_sha256=sha(mask))


def test_full_assembly_preserves_unknown_single_and_independent(tmp_path):
    plan,mask=fixture(tmp_path)
    pool={1:[prediction(mask),prediction(mask,'reverse',3)],2:[prediction(mask)],
          3:[prediction(mask),prediction(mask,'reverse',0)]}
    with patch('integrations.video_masks.full_pipeline.gather_predictions',return_value=pool):
        rows=assemble(tmp_path,plan)
    assert [r['status'] for r in rows]==['ORIGINAL_RETAINED','BIDIRECTIONAL_CANDIDATE',
        'SINGLE_DIRECTION_CANDIDATE','SINGLE_DIRECTION_CANDIDATE','UNKNOWN']
    assert rows[0]['mask_sha256']==sha(mask)
    assert rows[4]['mask_path'] is None
    assert rows[3]['independent_bidirectional'] is False


def test_disagreement_not_hidden_by_choosing_mask(tmp_path):
    plan,mask=fixture(tmp_path)
    other=tmp_path/'b.png';Image.fromarray(np.array([[255,0,0,0]]*6,dtype=np.uint8)).save(other)
    with patch('integrations.video_masks.full_pipeline.gather_predictions',
               return_value={1:[prediction(mask),prediction(other,'reverse',3)]}):
        rows=assemble(tmp_path,plan)
    assert rows[1]['status']=='DISAGREEMENT_CANDIDATE'
    assert rows[1]['bidirectional_metrics']['iou']==0.25
    assert rows[1]['mask_path']==str(mask)


def test_mask_drift_stops_assembly(tmp_path):
    plan,mask=fixture(tmp_path)
    p=prediction(mask);mask.write_bytes(b'corrupt')
    with patch('integrations.video_masks.full_pipeline.gather_predictions',return_value={1:[p]}):
        with pytest.raises(ValueError,match='hash drift'):assemble(tmp_path,plan)


def test_incomplete_event_tail_and_no_final_result_keep_commits(tmp_path):
    plan,mask=fixture(tmp_path)
    folder=tmp_path/'windows/task_000/worker';folder.mkdir(parents=True)
    target=folder/'000000.png';target.write_bytes(mask.read_bytes())
    plan['tasks']=[dict(task_id='task_000',frame_count=2,direction='forward',start_decode_index=1,
                       root_seed_decode_index=0,chain_depth=1,interval_id='i')]
    events=tmp_path/'worker';events.mkdir()
    commit=dict(seq=1,event='frame_commit',task_id='task_000',direction='forward',frame_id=0,
                mask_path=str(target),mask_sha256=sha(target))
    (events/'events.jsonl').write_text(json.dumps(dict(seq=0,event='ready'))+'\n'+json.dumps(commit)+'\n{"seq":')
    pool=gather_predictions(tmp_path,plan)
    assert pool[1][0]['chain_depth']==1
    assert pool[1][0]['mask_sha256']==sha(target)


@pytest.mark.parametrize('standalone',[True,False])
def test_full_public_dispatch(tmp_path,standalone):
    if standalone:
        from scripts.propagate_video_masks import main
        args=[]
    else:
        from scripts.run_photo_demo import main
        args=['propagate-masks']
    args+=['--full-video','--pipeline',str(tmp_path/'p'),'--output',str(tmp_path/'o'),
           '--reuse-run',str(tmp_path/'r'),'--execute']
    with patch('integrations.video_masks.full_pipeline.execute',return_value=dict(status='FULL_TIMELINE_EXPORTED',
               automatic_boundary_review=dict(required=True,status='AUTOMATIC_BOUNDARY_REVIEW_READY_NOT_ACCEPTED'))) as call:
        assert main(args)==0
        assert call.call_args.args==(tmp_path/'p',tmp_path/'o',tmp_path/'r',True,'union')
