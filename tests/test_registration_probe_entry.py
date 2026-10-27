import json
from pathlib import Path
import runpy
import pytest
from integrations.demo_pipeline import pipeline

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('manifest_exists',[True,False])
def test_supervised_entry_binds_source_and_rejects_rc0_without_manifest(tmp_path,monkeypatch,manifest_exists):
    main=runpy.run_path(str(ROOT/'scripts/probe_registration_gate.py'))['main']
    source=tmp_path/'source';source.mkdir();binary=source/'model';binary.mkdir();(source/'logs').mkdir()
    run=source/'run.json';run.write_text(json.dumps({'models':[{'model_id':'1','binary_path':'model'}]}))
    for p in [source/'database.db',source/'frames_map.jsonl',source/'logs/incremental_mapping.result.json',binary/'cameras.bin']:
        p.write_bytes(b'input')
    calls=[]
    def invoke(root,out,stage,argv,**budgets):
        assert (out/'PLAN.json').is_file()
        assert argv[1:4]==['-B','-X','utf8']
        assert budgets=={'seconds':90,'memory':1<<30}
        calls.append(argv)
        if manifest_exists:
            (out/'native').mkdir()
            (out/'native/MANIFEST.json').write_text(json.dumps(dict(status='ONE_NATIVE_CALL_COMPLETED_NOT_GEOMETRY_ACCEPTED',registration_call_count=1,source_unchanged=True)))
        return dict(reclaim_confirmed=True,status='completed',root_exit_code=0)
    monkeypatch.setattr(pipeline,'_invoke',invoke)
    out=tmp_path/'out';args=['--sfm-run',str(run),'--input-index','19','--model-id','1','--output',str(out)]
    assert main(args)==(0 if manifest_exists else 2)
    assert len(calls)==1
    assert json.loads((out/'RESULT.json').read_text())['native_completion_confirmed']==manifest_exists
    with pytest.raises(ValueError,match='fresh'):main(args)
    assert len(calls)==1
