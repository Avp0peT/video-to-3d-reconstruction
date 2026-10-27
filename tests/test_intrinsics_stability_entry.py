import json
from pathlib import Path
import runpy
import pytest
from integrations.demo_pipeline import pipeline, registration_probe

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('evidence,expected',[
    ('complete',0),('missing',2),('malformed',2),('wrong_status',2),
    ('changed',2),('missing_report',2),('unreclaimed',4),('drift',2)])
def test_entry_requires_diagnostic_evidence(tmp_path,monkeypatch,evidence,expected):
    main=runpy.run_path(str(ROOT/'scripts/review_intrinsics_stability.py'))['main']
    source=tmp_path/'source';source.mkdir()
    run=source/'run.json'
    for p in (run,source/'database.db',source/'frames_map.jsonl'):p.write_bytes(b'{}')
    # The numerical module is built concurrently; this test isolates the CLI contract.
    original=registration_probe._binding
    core=ROOT/'integrations/demo_pipeline/intrinsics_stability.py'
    def binding(path):
        if Path(path)==core:return dict(path=str(core),bytes=0,sha256='fixture-module')
        return original(path)
    monkeypatch.setattr(registration_probe,'_binding',binding)
    calls=[]
    def invoke(root,out,stage,argv,**budgets):
        plan=json.loads((out/'PLAN.json').read_text())
        assert plan['threads']==2 and plan['native_model_executions']==0
        assert plan['report_output']==str(out/'report')
        assert stage=='intrinsics_stability' and budgets==dict(seconds=120,memory=1<<30)
        assert argv[-2:]==[str(out/'report'),'']
        calls.append(argv)
        if evidence!='missing':
            report=out/'report';report.mkdir()
            detail=dict(status='INTRINSICS_DIAGNOSTIC_COMPLETE',source_unchanged=True,calibration_accepted=False)
            if evidence=='wrong_status':detail['status']='INSUFFICIENT_DATA'
            if evidence=='changed':detail['source_unchanged']=False
            (report/'INTRINSICS.json').write_text('invalid' if evidence=='malformed' else json.dumps(detail))
            if evidence!='missing_report':(report/'index.html').write_text('diagnostic')
        if evidence=='drift':run.write_text('changed')
        return dict(reclaim_confirmed=evidence!='unreclaimed',status='completed',root_exit_code=0)
    monkeypatch.setattr(pipeline,'_invoke',invoke)
    out=tmp_path/'out';args=['--sfm-run',str(run),'--output',str(out)]
    assert main(args)==expected
    assert len(calls)==1
    result=json.loads((out/'RESULT.json').read_text())
    assert result['calibration_accepted'] is False
    with pytest.raises(ValueError,match='fresh'):main(args)
    assert len(calls)==1


def test_entry_rejects_source_overlap_before_launch(tmp_path,monkeypatch):
    main=runpy.run_path(str(ROOT/'scripts/review_intrinsics_stability.py'))['main']
    run=tmp_path/'run.json';run.write_text('{}')
    monkeypatch.setattr(pipeline,'_invoke',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='nonoverlapping'):
        main(['--sfm-run',str(run),'--output',str(tmp_path/'inside')])
