import pytest
from integrations.video_masks import region_refinement as ref


@pytest.mark.parametrize('failure,added,expected',[(False,[],['reverse_prepare','reverse_run']),
    (False,[7],['reverse_prepare','reverse_run','bridge_prepare','bridge_run']),
    (True,[7],['reverse_prepare','reverse_run'])])
def test_automatic_selection_and_no_failed_stage_retry(tmp_path,monkeypatch,failure,added,expected):
    source=tmp_path/'source';source.mkdir();ref.save(source/'PLAN.json',{'schema':'glove-region-plan-v1','timeline':[]})
    calls=[]
    def prep(source,out,geometry):
        calls.append('reverse_prepare');out.mkdir();(out/'assessment').mkdir()
        ref.save(out/'PLAN.json',{'source_bindings':{}})
        ref.save(out/'assessment/ASSESSMENT.json',{'models':[]})
        return {'source_bindings':{}}
    def execute(out):
        calls.append('reverse_run')
        return {'status':'PARTIAL_DIAGNOSTIC' if failure else 'EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED'}
    monkeypatch.setattr(ref.reverse,'prepare',prep);monkeypatch.setattr(ref.reverse,'execute',execute)
    monkeypatch.setattr(ref,'select_intermediates',lambda *args:(added,{'endpoint_count':2}))
    monkeypatch.setattr(ref,'resolve_source_geometry',lambda *args:'geometry')
    monkeypatch.setattr(ref,'_code_bindings',lambda :{})
    def bridge_prepare(source,out,policy):
        calls.append('bridge_prepare');(out/'input').mkdir(parents=True)
        ref.save(out/'input/PLAN.json',{'added_decode_indices':added})
    monkeypatch.setattr(ref.bridge,'prepare',bridge_prepare)
    monkeypatch.setattr(ref.bridge,'execute',lambda *args:(calls.append('bridge_run') or {'status':'BRIDGE_EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED'}))
    result=ref.run(source,tmp_path/'new')
    assert calls==expected
    assert result['status']==('PARTIAL_DIAGNOSTIC' if failure else 'REFINEMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED')
    with pytest.raises(FileExistsError):ref.run(source,tmp_path/'new')
    with pytest.raises(ValueError,match='unattempted'):ref.execute(tmp_path/'new')


def test_wrong_input_rejected_without_new_directory(tmp_path):
    source=tmp_path/'raw';source.mkdir();ref.save(source/'PLAN.json',{'schema':'raw-photos'})
    with pytest.raises(ValueError,match='completed region input'):
        ref.prepare(source,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_source_drift_prevents_launch(tmp_path,monkeypatch):
    bound=tmp_path/'data';bound.write_bytes(b'old');out=tmp_path/'new';out.mkdir()
    ref.save(out/'REFINEMENT.json',dict(schema='object-region-refinement-v1',status='PREPARED',
        policy=ref.POLICY,source_bindings={str(bound):ref.sha(bound)},code_bindings={}))
    bound.write_bytes(b'changed')
    monkeypatch.setattr(ref.reverse,'execute',lambda *args:pytest.fail('must not execute on drift'))
    with pytest.raises(ValueError,match='prelaunch'):ref.execute(out)
