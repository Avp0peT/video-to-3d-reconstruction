import pytest
from integrations.video_masks import region_bridge_experiment as bridge


@pytest.mark.parametrize('failure,expected',[(None,['input','tracking','geometry','assessment']),('input',['input']),('geometry',['input','tracking','geometry']),('scope',['input','tracking','geometry','assessment'])])
def test_one_shot_stages_stop_on_error(tmp_path,monkeypatch,failure,expected):
    bridge.save(tmp_path/'RUN_PLAN.json',dict(source_bindings={},code_bindings={}))
    bridge.save(tmp_path/'SUMMARY.json',dict(status='PREPARED',stages={}))
    seen=[]
    def invoke(root,run,stage,argv,**kw):
        seen.append(stage)
        if stage=='assessment':
            (run/'assessment').mkdir()
            bridge.save(run/'assessment/BRIDGE_ASSESSMENT.json',{'geometry_accepted':False,'bridge_effect_evaluatable':failure!='scope'})
        return dict(status='completed',root_exit_code=int(failure==stage),reclaim_confirmed=True,errors=[],log_errors=[])
    monkeypatch.setattr(bridge,'_invoke',invoke)
    result=bridge.execute(tmp_path)
    assert seen==expected
    assert result['status']==('PARTIAL_DIAGNOSTIC' if failure else 'BRIDGE_EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED')
    with pytest.raises(ValueError,match='already attempted'):bridge.execute(tmp_path)
