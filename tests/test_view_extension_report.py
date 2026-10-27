import json
from pathlib import Path
import pytest
from PIL import Image
from integrations.demo_pipeline.view_extension_report import ExtensionEvidence


def test_empty_completion_keeps_frozen_events(tmp_path):
    out=tmp_path/'evidence';rec=ExtensionEvidence(out)
    rec(dict(event='PLAN',plan=[]))
    result=dict(status='VIEW_EXTENSION_LEDGER_COMPLETE',plan=[],query_rows=[],counts={},flow_calls=0)
    delivered=rec.finish(result,[])
    assert delivered['pose_fits']==delivered['new_cameras']==0
    assert rec.snapshot()['status']=='COMPLETE'
    assert json.loads((out/'EVENTS.jsonl').read_text())['seq']==0
    assert len(delivered['source_bindings'])==4


@pytest.mark.parametrize('name',['EVENTS.jsonl','PLAN.json'])
def test_late_byte_drift_blocks_completed_publication(tmp_path,name):
    rec=ExtensionEvidence(tmp_path/'e');rec(dict(event='PLAN',plan=[]))
    with (tmp_path/'e'/name).open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='changed'):
        rec.finish(dict(status='VIEW_EXTENSION_LEDGER_COMPLETE',query_rows=[],counts={},flow_calls=0),[])
    assert not (tmp_path/'e'/'RESULT.json').exists()


def test_failed_records_preserve_partial_and_do_not_publish_complete(tmp_path):
    rec=ExtensionEvidence(tmp_path/'e');rec(dict(event='PLAN',plan=[]))
    rec(dict(event='FAILED',partial=dict(flow_calls=1,status='FAILED')))
    assert rec.snapshot()['status']=='FAILED'
    assert json.loads((tmp_path/'e'/'FAILURE.json').read_text())['partial']['flow_calls']==1
    with pytest.raises(ValueError):rec.finish(dict(status='VIEW_EXTENSION_LEDGER_COMPLETE'),[])
    assert not (tmp_path/'e'/'RESULT.json').exists()


def test_overlay_and_identity_stay_per_model(tmp_path):
    source=tmp_path/'frame.png';Image.new('L',(32,32),70).save(source)
    before=source.read_bytes();rec=ExtensionEvidence(tmp_path/'e');rec(dict(event='PLAN',plan=[]))
    points=[dict(status='ACCEPTED_CONDITIONAL_CORRESPONDENCE',routes=[dict(query_raw_xy=[12.5,13.5]),dict(query_raw_xy=[12.6,13.5])])]
    rows=[dict(model_key=key,query_input_index=7,status='CORRESPONDENCES_MEASURED',points=points,summary=dict(correspondence_count=1,registration_reasons=['INTRINSICS_UNESTABLISHED']),reasons=[]) for key in ['a','b']]
    data=dict(status='VIEW_EXTENSION_LEDGER_COMPLETE',plan=[],query_rows=rows,counts=dict(accepted_correspondences=2),flow_calls=12)
    final=rec.finish(data,[dict(input_index=7,path=str(source),width=32,height=32)])
    assert len(list((tmp_path/'e').glob('query_*.png')))==2
    assert len(final['source_bindings'])==6 and source.read_bytes()==before
    assert 'INTRINSICS_UNESTABLISHED' in (tmp_path/'e'/'index.html').read_text(encoding='utf-8')


def test_recorder_rejects_nonfinite_without_partial_line(tmp_path):
    rec=ExtensionEvidence(tmp_path/'e')
    with pytest.raises(ValueError):rec(dict(event='PLAN',bad=float('nan')))
    assert not rec.events.exists() and rec.sequence==0


def test_delivery_timeout_records_stage_failure(tmp_path):
    rec=ExtensionEvidence(tmp_path/'e');rec(dict(event='PLAN',plan=[]))
    data=dict(status='VIEW_EXTENSION_LEDGER_COMPLETE',plan=[],query_rows=[{}],counts={},flow_calls=0)
    def expired():raise TimeoutError('synthetic delivery budget')
    with pytest.raises(TimeoutError,match='synthetic delivery'):
        rec.finish(data,[],expired)
    failure=json.loads((tmp_path/'e'/'FAILURE.json').read_text())
    assert failure['stage']=='delivery' and failure['partial_files_not_publishable']
    assert rec.snapshot()['status']=='FAILED'


def test_in_memory_result_plan_tamper_refused(tmp_path):
    rec=ExtensionEvidence(tmp_path/'e');rec(dict(event='PLAN',plan=[]))
    with pytest.raises(ValueError,match='differs'):
        rec.finish(dict(status='VIEW_EXTENSION_LEDGER_COMPLETE',plan=[{'forged':True}]),[])
