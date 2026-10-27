"""Synthetic UI continuation wiring. No native/GPU work is permitted."""
import importlib.util
import json
import hashlib
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'app/pages/11_Photo_to_3D.py'
spec = importlib.util.spec_from_file_location('photo_resume_ui_test', PAGE)
page = importlib.util.module_from_spec(spec); spec.loader.exec_module(page)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def bind(path):
    from integrations.mvs_surface.photo_inputs import _binding
    return _binding(path)


def source(base, checkpoint=True, status='PARTIAL'):
    run=base/'source';run.mkdir(parents=True)
    save(run/'pipeline.json', {'status':status,'exit_code':None if status=='RUNNING' else 1,
                              'stages':{},'target':'report','errors':[]})
    save(run/'PLAN.json', {'mvs_tool_root':'synthetic_tools','blender_exe':'synthetic_blender'})
    if checkpoint:save(run/'SURFACE_RESUME_SOURCE.json', {'synthetic_fixture':True})
    return run


def continuation(output, source_run, root):
    output.mkdir(parents=True)
    surface=output/'geometry/mesh.ply';surface.parent.mkdir()
    surface.write_bytes(b'synthetic surface')
    blend=output/'geometry/model.blend';blend.write_bytes(b'synthetic blend')
    surf_ref=bind(surface);blend_ref=bind(blend)
    evidence=output/'geometry/result.json'
    save(evidence, {'status':'SURFACE_GENERATED_NOT_ACCEPTED','exit_code':0})
    data={'status':'PASS','payload_sha256':'synthetic-payload','scenes':[{'source_sha256':surf_ref['sha256']}]}
    supervision={'status':'completed','root_exit_code':0,'reclaim_confirmed':True,'errors':[],'log_errors':[]}
    blend_result=output/'geometry/BLENDER_RESULT.json'
    save(blend_result, {'status':'SUCCESS','blend':blend_ref,'blend_sha256_before_readback':blend_ref['sha256'],
        'payload_sha256':'synthetic-payload','stages':[{'mode':mode,'data':data,'supervision':supervision} for mode in ('create','verify')]})
    bindings=[bind(source_run/'SURFACE_RESUME_SOURCE.json')]
    save(output/'PLAN.json', {'schema':'photo-surface-continuation-v1','output':str(output),'source_run':str(source_run),
                              'root':str(root),'bindings':bindings})
    stages={}
    for name in ('native_subset','mvs_bridge','mvs_masks','surface_plan','surface_reconstruction','blender_export'):
        entry=output/(name+'_ENTRY.json')
        save(entry, {**supervision,'pid':123,'cwd':str(root),'argv':['synthetic',name]})
        stages[name]={'status':'SUCCESS','exit_code':0,'reclaim_confirmed':True,'entry_path':str(entry)}
    state={'schema':'photo-surface-continuation-v1','status':'PARTIAL','exit_code':1,'deliverable_ready':True,
        'source_run':str(source_run),'source_bindings':bindings,'stages':stages,
        'surface':{'status':'SUCCESS','artifacts':[surf_ref],'evidence_paths':[str(evidence)]},
        'blender':{'status':'SUCCESS',**blend_ref,'reopen_verified':True,'result_path':str(blend_result)},
        'cad_status':'NOT_RUN','research_status':'NOT_RUN'}
    save(output/'CONTINUATION.json',state)
    return state


def test_no_checkpoint_disabled_reason_and_no_native(tmp_path, monkeypatch):
    root=tmp_path; (root/'outputs').mkdir();run=source(root/'outputs/task',False)
    monkeypatch.setattr(page,'ROOT',root)
    with pytest.raises(ValueError,match='历史任务不能补造'):
        page.surface_resume_plan(run,root/'outputs/new')


def test_running_source_refused(tmp_path, monkeypatch):
    root=tmp_path;(root/'outputs').mkdir();run=source(root/'outputs/task',status='RUNNING')
    monkeypatch.setattr(page,'ROOT',root)
    with pytest.raises(ValueError,match='尚未结束'):
        page.surface_resume_plan(run,root/'outputs/new')


def test_source_reclaim_rejection_from_existing_plan(tmp_path, monkeypatch):
    from integrations.demo_pipeline import resume
    root=tmp_path;(root/'outputs').mkdir();run=source(root/'outputs/task')
    monkeypatch.setattr(page,'ROOT',root)
    def reject(*a,**k):raise ValueError('source native reclamation unconfirmed')
    monkeypatch.setattr(resume,'plan',reject)
    with pytest.raises(ValueError,match='reclamation'):
        page.surface_resume_plan(run,root/'outputs/new')


@pytest.mark.parametrize('changed',['none','blend','surface','reopen','stage','source','escape'])
def test_continuation_download_verifies_bytes_stages_source_and_reopen(tmp_path, monkeypatch, changed):
    root=tmp_path;(root/'outputs').mkdir();run=source(root/'outputs/task')
    out=root/'outputs/continued';state=continuation(out,run,root)
    monkeypatch.setattr(page,'ROOT',root)
    if changed=='blend':Path(state['blender']['path']).write_bytes(b'changed')
    if changed=='surface':Path(state['surface']['artifacts'][0]['path']).write_bytes(b'changed')
    if changed=='source':(run/'SURFACE_RESUME_SOURCE.json').write_bytes(b'changed')
    if changed=='reopen':state['blender']['reopen_verified']=False
    if changed=='stage':
        entry=out/'native_subset_ENTRY.json';value=json.loads(entry.read_text());value['reclaim_confirmed']=False;save(entry,value)
    if changed=='escape':state['blender']['path']=str(run/'outside.blend')
    save(out/'CONTINUATION.json',state)
    if changed=='none':
        checked,refs=page.continuation_artifacts(out)
        assert checked['cad_status']=='NOT_RUN' and len(refs)==2
        assert [page.cad_download_payload(ref) for ref in refs]==[b'synthetic surface',b'synthetic blend']
        assert page.load_existing(str(out))['continuation'] is True
    else:
        with pytest.raises((ValueError,RuntimeError,OSError)):page.continuation_artifacts(out)


def test_explicit_resume_invokes_cli_once_in_new_task(tmp_path, monkeypatch):
    from integrations.demo_pipeline import resume
    from integrations.mvs_surface import supervision
    root=tmp_path;(root/'outputs').mkdir();run=source(root/'outputs/task')
    monkeypatch.setattr(page,'ROOT',root);calls=[]
    monkeypatch.setattr(resume,'plan',lambda *a,**k:{'python':'synthetic_python','tool_root':'synthetic_tools',
        'blender_exe':'synthetic_blender','requested_model_id':'0'})
    def fake(argv,**kw):
        calls.append(argv)
        assert 'resume' in argv and 'run' not in argv and '--model-id' in argv
        out=Path(argv[argv.index('--output')+1]);assert not out.exists()
        assert not out.is_relative_to(run)
        return {'status':'completed','root_exit_code':1,'reclaim_confirmed':True,'argv':argv}
    monkeypatch.setattr(supervision,'run_supervised',fake)
    saved=page.execute_surface_resume(run)
    assert saved['continuation'] and len(calls)==1 and not Path(saved['run']).exists()


@pytest.mark.parametrize('eligible',[False,True])
def test_apptest_load_click_and_rerun_are_bounded(tmp_path,monkeypatch,eligible):
    from streamlit.testing.v1 import AppTest
    from integrations.demo_pipeline import resume
    from integrations.mvs_surface import supervision
    # AppTest imports the real page afresh; use this fresh test's evidence
    # scratch subtree under real outputs for the page's path policy.
    # Each pytest temp identity gets a fresh output scope, rather than writing
    # into the previous acceptance run and colliding on repeated verification.
    identity=hashlib.sha256(str(tmp_path.resolve()).encode()).hexdigest()[:20]
    base=ROOT/'outputs/testing/photo_resume_ui'/identity
    run=source(base,checkpoint=eligible)
    calls=[]
    monkeypatch.setattr(resume,'plan',lambda *a,**k:{'python':str(page.LIGHT_PYTHON),
        'tool_root':'synthetic_tools','blender_exe':'synthetic_blender','requested_model_id':None})
    def fake(argv,**kw):
        calls.append(argv)
        out=Path(argv[argv.index('--output')+1]);continuation(out,run,ROOT)
        return {'status':'completed','root_exit_code':1,'reclaim_confirmed':True,'errors':[],'log_errors':[],
                'argv':argv,'cwd':kw['cwd'],'pid':123}
    monkeypatch.setattr(supervision,'run_supervised',fake)
    app=AppTest.from_file(str(PAGE),default_timeout=15).run()
    app.text_input[-1].set_value(str(run)).run()
    next(b for b in app.button if b.label=='载入已有任务').click().run()
    button=next(b for b in app.button if b.label=='继续表面 / Blender（复用上游）')
    assert button.disabled is (not eligible) and not calls and not app.exception
    if eligible:
        button.click().run()
        assert len(calls)==1 and len(app.get('download_button'))==2 and not app.exception
        app.run();assert len(calls)==1 and not app.exception
    else:
        assert any('没有生产时正式续跑' in c.value for c in app.caption)
        app.run();assert not calls and not app.exception
