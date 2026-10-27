"""Only CLI orchestration; no reconstruction, inference, renderer or GPU."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import json,sys
import pytest
from scripts import run_depth_fusion_model as cli


def stages(tmp_path,monkeypatch,final_elapsed=15.,final_blender='SUCCESS',completion_elapsed=5.,closed=True,delivery_blender='SUCCESS',delivery_elapsed=5.,strict_closed=True,comparison_blender='SUCCESS',comparison_status='STRICT_COMPARISON_DELIVERED',comparison_error=False,comparison_reason=None):
    calls=[];clock=[0.]
    monkeypatch.setattr(cli,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    import integrations.mvs_surface.full_depth_inputs as full
    import integrations.mvs_surface.depth_fusion_finalize as final
    def prepare(parent,source,spec,check):
        calls.append(('prepare',parent,source,spec));clock[0]+=10;check()
        spec.parent.mkdir(parents=True);spec.write_text('{}')
    def raw(source,out,**kwargs):
        calls.append(('raw',source,out,kwargs));clock[0]+=20
        out.mkdir(parents=True,exist_ok=True)
        return dict(status='RAW_DIAGNOSTIC',blender=dict(status='SUCCESS'))
    def finalize(raw_path,out,**kwargs):
        calls.append(('final',raw_path,out,kwargs));clock[0]+=final_elapsed
        out.mkdir(parents=True,exist_ok=True)
        return dict(status='AUDITED_NOT_GEOMETRY_ACCEPTED',blender=dict(status=final_blender))
    def complete(spec,raw_path,out,**kwargs):
        calls.append(('completion',spec,raw_path,out,kwargs));clock[0]+=completion_elapsed
        out.mkdir(parents=True,exist_ok=True)
        return dict(status='INFERRED_NOT_GEOMETRY_ACCEPTED',blender=dict(status='SUCCESS'))
    module=ModuleType('integrations.mvs_surface.object_completion_pipeline');module.run=complete
    monkeypatch.setitem(sys.modules,module.__name__,module)
    def deliver(inferred,out,**kwargs):
        calls.append(('delivery',inferred,out,kwargs));clock[0]+=delivery_elapsed
        out.mkdir(parents=True,exist_ok=True)
        return dict(status='CLOSED_ESTIMATED_OUTER_MODEL_DELIVERED',closed_estimated_outer_model=closed,blender=dict(status=delivery_blender))
    delivery=ModuleType('integrations.mvs_surface.completion_delivery');delivery.finalize_completion=deliver
    monkeypatch.setitem(sys.modules,delivery.__name__,delivery)
    def compare(inferred,out,**kwargs):
        calls.append(('comparison',inferred,out,kwargs));clock[0]+=5
        if comparison_error:raise RuntimeError('comparison synthetic failure')
        out.mkdir(parents=True,exist_ok=True)
        blender=dict(status=comparison_blender)
        if comparison_reason is not None:blender['reason']=comparison_reason
        return dict(status=comparison_status,blender=blender,strict_closed=strict_closed,improvement=False)
    comparison=ModuleType('integrations.mvs_surface.strict_completion_comparison');comparison.run=compare
    monkeypatch.setitem(sys.modules,comparison.__name__,comparison)
    monkeypatch.setattr(full,'prepare_full_inputs',prepare)
    monkeypatch.setattr(cli,'run',raw)
    monkeypatch.setattr(final,'finalize',finalize)
    return calls,tmp_path/'output'


def test_completion_only_explicit_full_source_and_shared_budget(tmp_path,monkeypatch,capsys):
    calls,out=stages(tmp_path,monkeypatch)
    rc=cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--blender','--seconds','100'])
    assert rc==0
    assert [x[0] for x in calls]==['prepare','raw','final','completion','delivery','comparison']
    raw,final,completion,delivery,comparison=calls[1:]
    assert raw[3]['seconds']==90 and raw[3]['blender'] is False
    assert raw[3]['full_source']==out/'FULL_INPUTS.json'
    assert final[3]['seconds']==70 and final[3]['blender'] is True
    assert completion[1:4]==(out/'FULL_INPUTS.json',out/'raw',out/'completion')
    assert completion[4]==dict(blender=False,seconds=55)
    assert delivery[1:3]==(out/'completion',out/'completed_model')
    assert delivery[3]==dict(blender=True,seconds=50)
    assert comparison[1:3]==(out/'completion',out/'strict_comparison')
    assert comparison[3]==dict(baseline_delivery=out/'completed_model',blender=True,seconds=45)
    text=(out/'index.html').read_text(encoding='utf-8')
    assert 'completion/index.html' in text and 'final/index.html' in text
    assert text.index('completed_model/index.html')<text.index('final/index.html')
    assert text.index('strict_comparison/index.html')<text.index('completed_model/index.html')
    assert 'raw/index.html' in text
    assert '推算不是新增观测' in text
    result=json.loads(capsys.readouterr().out)
    assert result['observed_status']=='AUDITED_NOT_GEOMETRY_ACCEPTED'
    assert result['status']=='CLOSED_ESTIMATED_OUTER_MODEL_DELIVERED'
    assert result['inferred_candidate_status']=='INFERRED_NOT_GEOMETRY_ACCEPTED'
    assert result['closed_estimated_outer_model'] is True
    assert result['strict_comparison_status']=='STRICT_COMPARISON_DELIVERED'
    assert result['strict_closed'] is True


@pytest.mark.parametrize('full',[False,True])
def test_default_behavior_never_silently_enables_completion(tmp_path,monkeypatch,capsys,full):
    calls,out=stages(tmp_path,monkeypatch)
    args=['--source-model','source','--output',str(out),'--seconds','100']
    if full:args.extend(['--full-depth-parent','parent'])
    assert cli.main(args)==0
    assert not {'completion','comparison','delivery'}&{x[0] for x in calls}
    assert 'completion/index.html' not in (out/'index.html').read_text(encoding='utf-8')
    result=json.loads(capsys.readouterr().out)
    assert set(result)=={'status','wall_seconds'}


@pytest.mark.parametrize('extra',[[],['--raw-only'],['--full-depth-parent','parent','--raw-only']])
def test_inferred_flag_rejects_missing_parent_or_raw_only_before_work(tmp_path,monkeypatch,extra):
    calls,out=stages(tmp_path,monkeypatch)
    with pytest.raises(SystemExit) as caught:
        cli.main(['--source-model','source','--output',str(out),'--allow-inferred-completion',*extra])
    assert caught.value.code==2 and not calls
    assert not out.exists()


def test_no_fresh_budget_when_deadline_exhausted_before_completion(tmp_path,monkeypatch):
    calls,out=stages(tmp_path,monkeypatch,final_elapsed=80.)
    with pytest.raises(TimeoutError,match='before inferred completion'):
        cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--seconds','100'])
    assert [x[0] for x in calls]==['prepare','raw','final']
    assert not (out/'completion').exists()


def test_successful_completion_does_not_hide_failed_observed_blender(tmp_path,monkeypatch,capsys):
    calls,out=stages(tmp_path,monkeypatch,final_blender='FAILED')
    assert cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--blender','--seconds','100'])==1
    assert calls[-1][0]=='comparison'
    capsys.readouterr()


@pytest.mark.parametrize('closed,blend',[(False,'SUCCESS'),(True,'PARTIAL'),(False,'PARTIAL')])
def test_delivery_requires_closed_flag_and_actual_blender_success(tmp_path,monkeypatch,capsys,closed,blend):
    calls,out=stages(tmp_path,monkeypatch,closed=closed,delivery_blender=blend)
    assert cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--blender','--seconds','100'])==1
    assert calls[-1][0]=='comparison'
    summary=json.loads(capsys.readouterr().out)
    assert summary['closed_estimated_outer_model']==closed


def test_no_fresh_budget_after_inference_before_delivery(tmp_path,monkeypatch):
    calls,out=stages(tmp_path,monkeypatch,completion_elapsed=60.)
    with pytest.raises(TimeoutError,match='before closed model delivery'):
        cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--seconds','100'])
    assert [x[0] for x in calls]==['prepare','raw','final','completion']
    assert not (out/'completed_model').exists()


def test_closed_false_fails_even_without_blender_request(tmp_path,monkeypatch,capsys):
    calls,out=stages(tmp_path,monkeypatch,closed=False)
    assert cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--seconds','100'])==1
    assert calls[-1][3]['blender'] is False
    capsys.readouterr()


def test_raw_only_default_route_kept(tmp_path,monkeypatch,capsys):
    calls,out=stages(tmp_path,monkeypatch)
    assert cli.main(['--source-model','source','--output',str(out),'--raw-only','--seconds','100'])==0
    assert [x[0] for x in calls]==['raw']
    assert calls[0][2]==out
    capsys.readouterr()


def test_strict_not_closed_or_improved_is_a_useful_comparison_not_failure(tmp_path,monkeypatch,capsys):
    calls,out=stages(tmp_path,monkeypatch,strict_closed=False)
    assert cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--seconds','100'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['strict_closed'] is False
    assert result['closed_estimated_outer_model'] is True
    assert result['strict_comparison_status']=='STRICT_COMPARISON_DELIVERED'
    assert sum(x[0]=='comparison' for x in calls)==1


@pytest.mark.parametrize('options',[{'comparison_blender':'PARTIAL'},{'comparison_status':'FAILED'}])
def test_comparison_delivery_failure_is_not_hidden(tmp_path,monkeypatch,capsys,options):
    calls,out=stages(tmp_path,monkeypatch,**options)
    assert cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--blender','--seconds','100'])==1
    assert calls[-1][0]=='comparison'
    capsys.readouterr()


def test_comparison_exception_propagates(tmp_path,monkeypatch):
    calls,out=stages(tmp_path,monkeypatch,comparison_error=True)
    with pytest.raises(RuntimeError,match='comparison synthetic failure'):
        cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--seconds','100'])
    assert calls[-1][0]=='comparison'
    assert not (out/'index.html').exists()


def test_shared_deadline_exhausted_before_comparison_does_not_reset(tmp_path,monkeypatch):
    calls,out=stages(tmp_path,monkeypatch,delivery_elapsed=60.)
    with pytest.raises(TimeoutError,match='before strict completion comparison'):
        cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--seconds','100'])
    assert [x[0] for x in calls]==['prepare','raw','final','completion','delivery']
    assert not (out/'strict_comparison').exists()


def test_absent_strict_geometry_is_explicit_not_applicable_not_fake_blender_success(tmp_path,monkeypatch,capsys):
    calls,out=stages(tmp_path,monkeypatch,strict_closed=False,comparison_blender='NOT_APPLICABLE',comparison_reason='strict_envelope_absent')
    assert cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--blender','--seconds','100'])==0
    assert calls[-1][3]['blender'] is True
    summary=json.loads(capsys.readouterr().out)
    assert summary['strict_closed'] is False
    assert summary['strict_comparison_status']=='STRICT_COMPARISON_DELIVERED'
    assert summary['closed_estimated_outer_model'] is True


@pytest.mark.parametrize('status,reason,strict_closed',[
    ('NOT_RUN',None,False),('NOT_RUN','strict_envelope_absent',False),
    ('NOT_APPLICABLE',None,False),('NOT_APPLICABLE','other_reason',False),
    ('NOT_APPLICABLE','strict_envelope_absent',True)])
def test_only_consistent_explicit_absence_exception_is_accepted(tmp_path,monkeypatch,capsys,status,reason,strict_closed):
    calls,out=stages(tmp_path,monkeypatch,comparison_blender=status,comparison_reason=reason,strict_closed=strict_closed)
    assert cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--blender','--seconds','100'])==1
    capsys.readouterr()


def test_absence_exception_does_not_mask_failed_baseline_blender(tmp_path,monkeypatch,capsys):
    calls,out=stages(tmp_path,monkeypatch,final_blender='FAILED',strict_closed=False,comparison_blender='NOT_APPLICABLE',comparison_reason='strict_envelope_absent')
    assert cli.main(['--source-model','source','--full-depth-parent','parent','--output',str(out),'--allow-inferred-completion','--blender','--seconds','100'])==1
    capsys.readouterr()
