import json
from pathlib import Path
import runpy
import pytest
from integrations.demo_pipeline import pipeline, registration_probe

ROOT=Path(__file__).resolve().parents[1]


def load_main():
    return runpy.run_path(str(ROOT/'scripts/review_intrinsics_sensitivity.py'))['main']


def source_fixture(tmp_path):
    report=tmp_path/'old_report';report.mkdir()
    sfm=tmp_path/'sfm';sfm.mkdir();(sfm/'run.json').write_text('{}')
    source=report/'INTRINSICS.json'
    source.write_text(json.dumps(dict(sfm_run=str(sfm/'run.json'))))
    return source,sfm


@pytest.mark.parametrize('evidence,expected',[
    ('complete',0),('missing',2),('malformed',2),('wrong_status',2),
    ('changed',2),('missing_report',2),('unreclaimed',4),('drift',2)])
def test_entry_requires_sensitivity_evidence(tmp_path,monkeypatch,evidence,expected):
    main=load_main();source,sfm=source_fixture(tmp_path)
    original=registration_probe._binding
    core=ROOT/'integrations/demo_pipeline/intrinsics_sensitivity.py'
    def binding(path):
        if Path(path)==core:return dict(path=str(core),bytes=0,sha256='fixture-module')
        return original(path)
    monkeypatch.setattr(registration_probe,'_binding',binding)
    calls=[]
    def invoke(root,out,stage,argv,**budgets):
        plan=json.loads((out/'PLAN.json').read_text())
        assert plan['threads']==2 and plan['native_model_executions']==0
        assert plan['report_output']==str(out/'report')
        assert stage=='intrinsics_sensitivity' and budgets==dict(seconds=120,memory=1<<30)
        assert argv[-2:]==[str(source),str(out/'report')]
        names={Path(ref['path']).name for ref in plan['source_bindings']}
        assert {'INTRINSICS.json','intrinsics_sensitivity_math.py','intrinsics_stability.py',
                'focal_profile.py','parallel_photo_audit.py','supervision.py','pipeline.py'}<=names
        calls.append(argv)
        if evidence!='missing':
            report=out/'report';report.mkdir()
            detail=dict(status='INTRINSICS_SENSITIVITY_COMPLETE',source_unchanged=True,calibration_accepted=False)
            if evidence=='wrong_status':detail['status']='INTRINSICS_DIAGNOSTIC_COMPLETE'
            if evidence=='changed':detail['source_unchanged']=False
            (report/'SENSITIVITY.json').write_text('invalid' if evidence=='malformed' else json.dumps(detail))
            if evidence!='missing_report':(report/'index.html').write_text('diagnostic')
        if evidence=='drift':source.write_text('{}')
        return dict(reclaim_confirmed=evidence!='unreclaimed',status='completed',root_exit_code=0)
    monkeypatch.setattr(pipeline,'_invoke',invoke)
    out=tmp_path/'out';args=['--intrinsics',str(source),'--output',str(out)]
    assert main(args)==expected
    assert len(calls)==1
    result=json.loads((out/'RESULT.json').read_text())
    assert result['calibration_accepted'] is False
    with pytest.raises(ValueError):main(args)
    assert len(calls)==1


@pytest.mark.parametrize('where',['report_child','report_parent','sfm_child','sfm_parent'])
def test_entry_rejects_source_overlap_before_launch(tmp_path,monkeypatch,where):
    main=load_main();source,sfm=source_fixture(tmp_path)
    outputs=dict(report_child=source.parent/'inside',report_parent=source.parent.parent,
                 sfm_child=sfm/'inside',sfm_parent=sfm.parent)
    monkeypatch.setattr(pipeline,'_invoke',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='nonoverlapping'):
        main(['--intrinsics',str(source),'--output',str(outputs[where])])
