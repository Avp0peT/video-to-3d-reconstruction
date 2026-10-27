"""Coverage counts are evidence labels, never reconstructed geometry claims."""
from pathlib import Path
from unittest.mock import patch
import pytest

from integrations.video_masks.coverage import summarize, execute


def ledger(index, original='NO_MATCH'):
    return dict(input_index=index, decode_index=index, original_status=original,
                reason=None, propagation_status='NOT_RUN')


def row(index, status='PROPAGATED_CANDIDATE', sampled=True):
    return dict(local_frame_index=index,decode_index=index,input_index=index if sampled else None,
                decision=dict(status=status,metrics={'iou':0.9}),sources={'forward':{},'reverse':{}})


def plans():
    return dict(intervals=[dict(interval_id='clip_000',start_decode_index=0,end_decode_index=3,frame_count=4),
                           dict(interval_id='clip_001',start_decode_index=3,end_decode_index=5,frame_count=3)],
                coverage_ledger=[ledger(0,'OK'),ledger(1),ledger(2),ledger(3,'OK'),ledger(5)])


def test_unique_and_sampled_counts_are_separate():
    report=summarize(plans(),{
        'clip_000':dict(rows=[row(0,'ORIGINAL_RETAINED'),row(1),row(2,'UNRESOLVED'),row(3,'ORIGINAL_RETAINED')],candidate_count=1),
        'clip_001':dict(rows=[row(3,'ORIGINAL_RETAINED'),row(4,sampled=False)],candidate_count=1)})
    assert report['unique_decode_frames_reviewed']==5
    assert report['unique_candidate_count']==2
    assert report['sampled_candidate_count']==1
    assert report['sampled_unresolved_count']==1
    assert report['sampled_not_run_count']==1
    assert report['original_ok_count']==2
    assert report['coverage_ledger'][1]['original_status']=='NO_MATCH'
    assert report['published_for_reconstruction'] is False


def test_missing_batch_is_not_candidate():
    report=summarize(plans(),{})
    assert report['sampled_not_run_count']==3
    assert report['sampled_candidate_count']==0
    assert all(x['worker_status']=='NOT_RUN' for x in report['intervals'])


def test_decoded_but_unstarted_is_not_run():
    r=row(1,'UNRESOLVED');r['sources']={}
    report=summarize(plans(),{'clip_000':dict(rows=[r])})
    assert report['coverage_ledger'][1]['propagation_status']=='NOT_RUN'
    assert report['sampled_unresolved_count']==0


def test_interrupt_recovery_ignores_uncommitted_tail(tmp_path):
    import json
    from integrations.video_masks.coverage import recover_commits
    from integrations.video_masks.runner import review
    from PIL import Image
    from integrations.demo_pipeline.pipeline import sha
    root=tmp_path/'worker';root.mkdir()
    clip=tmp_path/'clips/clip_000'
    masks=clip/'worker/forward';masks.mkdir(parents=True)
    images=clip/'frames';images.mkdir()
    mask=masks/'000000.png';Image.new('L',(4,6),255).save(mask)
    Image.new('RGB',(4,6),'gray').save(images/'000000.jpg')
    record=dict(seq=1,event='frame_commit',clip_id='clip_000',direction='forward',frame_id=0,
                mask_path=str(mask),mask_sha256=sha(mask))
    (root/'events.jsonl').write_text(json.dumps(dict(seq=0,event='ready'))+'\n'+json.dumps(record)+'\n'+ '{"seq":2')
    plan=dict(interval_id='clip_000',frame_count=1,width=4,height=6,start_decode_index=1,end_decode_index=1,
              targets=[dict(local_frame_index=0,input_index=1,original_status='NO_MATCH')])
    recover_commits(tmp_path,plan)
    restored=json.loads((clip/'worker/result.json').read_text())
    assert restored['record_origin']=='parent_recovery_from_fsynced_events'
    assert restored['status']=='INTERRUPTED_WITH_COMMITS'
    result=review(clip,plan)
    assert result['rows'][0]['sources']['forward']['mask_sha256']==sha(mask)
    assert result['rows'][0]['decision']['status']=='UNRESOLVED'
    assert result['candidate_count']==0


def test_prelaunch_binding_failure_leaves_report(tmp_path):
    plan=plans()
    output=tmp_path/'new'
    from integrations.demo_pipeline.pipeline import read
    def selective_read(path):
        if path==tmp_path/'PLAN.json':
            raise ValueError('source binding missing')
        return read(path)
    with patch('integrations.video_masks.coverage.make_coverage_plan',return_value=plan), \
         patch('integrations.video_masks.coverage.read',side_effect=selective_read):
        result=execute(tmp_path,output,execute_model=True)
    import json
    assert json.loads((output/'RESULT.json').read_text())['status']=='FAILED'
    assert result['model_entries']==0


def test_conflicting_duplicate_evidence_not_accepted():
    plan=plans();plan['coverage_ledger'][1]=ledger(3)
    result=summarize(plan,{'clip_000':dict(rows=[row(3)]),
                           'clip_001':dict(rows=[row(3,'UNRESOLVED')])})
    assert result['coverage_ledger'][1]['propagation_status']=='UNRESOLVED'


def test_existing_output_rejected_before_planning(tmp_path):
    with patch('integrations.video_masks.coverage.make_coverage_plan') as planning:
        with pytest.raises(ValueError,match='new'):
            execute(tmp_path,tmp_path,execute_model=True)
        planning.assert_not_called()


@pytest.mark.parametrize('standalone',[False,True])
def test_public_coverage_dispatch_does_not_load_model(tmp_path,standalone):
    if standalone:
        from scripts.propagate_video_masks import main
        args=[]
    else:
        from scripts.run_photo_demo import main
        args=['propagate-masks']
    args += ['--pipeline',str(tmp_path/'source'),'--output',str(tmp_path/'new'),'--coverage']
    with patch('integrations.video_masks.coverage.execute',return_value={'status':'PREPARED'}) as call:
        assert main(args)==0
        assert call.call_args.args==(tmp_path/'source',tmp_path/'new',96,False,6,384)


def test_zero_bound_not_replaced_with_default(tmp_path):
    from scripts.run_photo_demo import main
    with patch('integrations.video_masks.coverage.execute',side_effect=ValueError('bad frame limit')) as call:
        assert main(['propagate-masks','--pipeline',str(tmp_path),'--output',str(tmp_path/'new'),
                     '--coverage','--max-frames','0']) != 0
        assert call.call_args.args[2]==0
