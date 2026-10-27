import json
from pathlib import Path
import runpy
import pytest
from integrations.demo_pipeline import pipeline

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('manifest_exists',[True,False])
def test_calibration_entry_requires_native_record_and_fresh_output(tmp_path,monkeypatch,manifest_exists):
    main=runpy.run_path(str(ROOT/'scripts/probe_calibration_control.py'))['main']
    source=tmp_path/'source';source.mkdir();(source/'logs').mkdir()
    run=source/'run.json'
    for p in (run,source/'database.db',source/'frames_map.jsonl',source/'logs/incremental_mapping.result.json'):
        p.write_bytes(b'{}')
    calls=[]
    def invoke(root,out,stage,argv,**budgets):
        plan=json.loads((out/'PLAN.json').read_text())
        assert plan['input_pair']==[20,21] and plan['target_input_index']==19
        assert budgets=={'seconds':120,'memory':1<<30}
        calls.append(argv)
        if manifest_exists:
            (out/'native').mkdir()
            (out/'native/CONTROL.json').write_text(json.dumps(dict(status='CONTROL_COMPLETED_NOT_GEOMETRY_ACCEPTED',source_unchanged=True)))
        return dict(reclaim_confirmed=True,status='completed',root_exit_code=0)
    monkeypatch.setattr(pipeline,'_invoke',invoke)
    out=tmp_path/'out';args=['--sfm-run',str(run),'--pair','20','21','--target','19','--output',str(out)]
    assert main(args)==(0 if manifest_exists else 2)
    assert len(calls)==1
    with pytest.raises(ValueError,match='fresh'):main(args)
    assert len(calls)==1
