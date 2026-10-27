"""TEST_DATA: pipeline admission, ordering, failures and no hidden restarts."""
import pytest
from integrations.video_masks import region_experiment as exp
from integrations.video_masks import region_comparison as comp
from integrations.video_masks import region_report as report


@pytest.mark.parametrize('failing_stage,expected', [
    (None, ['tracking', 'geometry', 'control', 'analyse', 'report']),
    ('tracking', ['tracking']),
    ('geometry', ['tracking', 'geometry'])])
def test_automatic_stages_stop_on_failure(tmp_path, monkeypatch, failing_stage, expected):
    exp.save(tmp_path/'PLAN.json', {'source_bindings': {}})
    exp.save(tmp_path/'SUMMARY.json', {'status': 'PREPARED', 'stages': {}})
    called=[]
    def invoke(root, run, name, argv, **kw):
        called.append(name)
        if name=='geometry' and failing_stage != name:
            (run/'geometry').mkdir()
            exp.save(run/'geometry'/'RESULT.json', {'status':'NO_MODEL'})
        return {'status':'completed','root_exit_code':1 if failing_stage==name else 0,
                'reclaim_confirmed':True,'errors':[],'log_errors':[]}
    monkeypatch.setattr(exp,'_invoke',invoke)
    monkeypatch.setattr(comp,'run_control',lambda run:called.append('control'))
    monkeypatch.setattr(comp,'analyse',lambda run,geometry_name:called.append('analyse'))
    monkeypatch.setattr(report,'render',lambda run:called.append('report'))
    result=exp.execute(tmp_path)
    assert called==expected
    assert result['status']==('PARTIAL_DIAGNOSTIC' if failing_stage else 'EXPERIMENT_COMPLETE_NOT_ACCEPTED')
    with pytest.raises(ValueError,match='already attempted'):
        exp.execute(tmp_path)
