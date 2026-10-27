"""UI-only synthetic pipeline records; no reconstruction or native process."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAGE = ROOT / "app/pages/11_Photo_to_3D.py"
spec = importlib.util.spec_from_file_location("photo_ui", PAGE)
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)


def test_bounded_execution_new_task_each_click(tmp_path, monkeypatch):
    import integrations.mvs_surface.supervision as supervision
    monkeypatch.setattr(page, "ROOT", tmp_path)
    calls = []
    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs["wall_seconds"] == 1800
        assert kwargs["memory_bytes"] == 12 * 1024**3
        assert kwargs["output_bytes"] == 3 * 1024**3
        assert Path(kwargs["output"]).is_relative_to(kwargs["cwd"])
        assert not Path(argv[argv.index("--output") + 1]).exists()
        return {"status": "completed", "root_exit_code": 1, "reclaim_confirmed": True}
    monkeypatch.setattr(supervision, "run_supervised", fake)
    first = page.execute("photos", "object", 24, "")
    second = page.execute("video.mp4", "other", 12, "3")
    assert first["run"] != second["run"]
    assert len(calls) == 2
    assert calls[0][0][0] == str(page.LIGHT_PYTHON)
    assert "--model-id" in calls[1][0]


def test_download_requires_all_evidence(tmp_path):
    blend = tmp_path / "model.blend"
    blend.write_bytes(b"synthetic UI fixture")
    blender = {"status": "SUCCESS", "reopen_verified": True, "path": str(blend),
               "sha256": hashlib.sha256(blend.read_bytes()).hexdigest()}
    state = {"deliverable_ready": True, "stages": {"blender": blender}}
    assert page.verified_blend(state, tmp_path) == blend
    state["deliverable_ready"] = False
    assert page.verified_blend(state, tmp_path) is None
    state["deliverable_ready"] = True
    blender["reopen_verified"] = False
    assert page.verified_blend(state, tmp_path) is None
    blender["reopen_verified"] = True
    blend.write_bytes(b"changed")
    assert page.verified_blend(state, tmp_path) is None


def test_cad_button_is_a_bounded_new_continuation(tmp_path, monkeypatch):
    from integrations.demo_pipeline import cad
    from integrations.mvs_surface import supervision
    monkeypatch.setattr(page, "ROOT", tmp_path)
    run = tmp_path / "outputs/source"
    run.mkdir(parents=True)
    (run / "pipeline.json").write_text(json.dumps({"stages": {}}))
    plans, calls = [], []
    monkeypatch.setattr(cad, "plan", lambda *a, **kw: plans.append((a, kw)))
    def fake(argv, **kw):
        calls.append((argv, kw))
        assert "continue_photo_cad.py" in argv[4]
        assert "--run" in argv and str(run) in argv
        assert "run_photo_demo.py" not in " ".join(argv)
        assert kw["wall_seconds"] == 240 and kw["memory_bytes"] == 3 * 1024**3
        assert kw["output_bytes"] == 512 * 1024**2
        assert not Path(argv[argv.index("--output") + 1]).exists()
        return {"status": "completed", "root_exit_code": 2, "reclaim_confirmed": True}
    monkeypatch.setattr(supervision, "run_supervised", fake)
    one, two = page.execute_cad(run), page.execute_cad(run)
    assert len(plans) == len(calls) == 2 and one["output"] != two["output"]
    assert one["supervision"]["root_exit_code"] == 2


def test_cad_download_needs_clean_outer_and_same_source(tmp_path, monkeypatch):
    from integrations.demo_pipeline import cad
    source = tmp_path / "source"
    calls = []
    monkeypatch.setattr(cad, "verify_cad_output", lambda *a: calls.append(a) or {"artifacts": []})
    saved = {"source_run": str(source), "output": str(tmp_path / "cad"), "supervision": {
        "status": "completed", "root_exit_code": 0, "reclaim_confirmed": True}}
    assert page.verified_cad(saved, source) == {"artifacts": []}
    assert len(calls) == 1
    for key, value in (("root_exit_code", False), ("root_exit_code", 1), ("reclaim_confirmed", False),
                       ("errors", ["failed"]), ("log_errors", ["write failed"])):
        original = saved["supervision"].copy()
        saved["supervision"][key] = value
        assert page.verified_cad(saved, source) is None
        saved["supervision"] = original
    assert page.verified_cad(saved, tmp_path / "other") is None
    assert len(calls) == 1


def test_cad_payload_verifies_bytes_actually_sent_after_prior_validation(tmp_path):
    path = tmp_path / "fixture.step"
    good = b"SYNTHETIC CAD PAYLOAD"
    path.write_bytes(good)
    ref = {"path": str(path), "bytes": len(good), "sha256": hashlib.sha256(good).hexdigest()}
    assert page.cad_download_payload(ref) == good
    path.write_bytes(b"X" * len(good))
    with pytest.raises(ValueError, match="发生变化"):
        page.cad_download_payload(ref)
    path.write_bytes(good + b"unbound")
    with pytest.raises(ValueError, match="发生变化"):
        page.cad_download_payload(ref)


def test_unified_cad_selection_uses_main_cli_and_rejects_unsupported_target(tmp_path, monkeypatch):
    from integrations.mvs_surface import supervision
    monkeypatch.setattr(page, 'ROOT', tmp_path)
    calls=[]
    monkeypatch.setattr(supervision,'run_supervised',lambda argv,**kw:calls.append(argv) or {'status':'completed'})
    page.execute('photos','dice',24,'','cad')
    assert len(calls)==1 and calls[0][calls[0].index('--target')+1]=='cad'
    assert 'run_photo_demo.py' in calls[0][4]
    with pytest.raises(ValueError):page.execute('photos','dice',24,'','unknown')
    assert len(calls)==1


def test_cad_failure_preserves_verified_geometry_but_never_invents_cad(tmp_path,monkeypatch):
    blend=tmp_path/'model.blend';blend.write_bytes(b'synthetic fixture')
    state={'target':'cad','status':'PARTIAL','deliverable_ready':False,'geometry_deliverable_ready':True,
           'stages':{'blender':{'status':'SUCCESS','reopen_verified':True,'path':str(blend),
                               'sha256':hashlib.sha256(blend.read_bytes()).hexdigest()},'cad':{'status':'FAILED'}}}
    assert page.verified_blend(state,tmp_path)==blend
    assert page.unified_cad(state,tmp_path) is None
    blend.write_bytes(b'changed')
    assert page.verified_blend(state,tmp_path) is None


def test_unified_cad_reads_only_same_run_verified_delivery(tmp_path,monkeypatch):
    from integrations.demo_pipeline import cad
    calls=[]
    monkeypatch.setattr(cad,'verify_cad_output',lambda *args:calls.append(args) or {'artifacts':[]})
    state={'target':'cad','stages':{'cad':{'status':'SUCCESS','output':str(tmp_path/'cad')}}}
    assert page.unified_cad(state,tmp_path)=={'artifacts':[]}
    assert calls==[(tmp_path/'cad',tmp_path)]
    state['stages']['cad']['output']=str(tmp_path/'other')
    with pytest.raises(ValueError):page.unified_cad(state,tmp_path)
    assert len(calls)==1


def test_apptest_rerun_failure_never_downloads(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import integrations.mvs_surface.supervision as supervision
    calls = []
    def fake(argv, **kwargs):
        calls.append(kwargs)
        return {"status": "completed", "root_exit_code": 1, "reclaim_confirmed": True}
    monkeypatch.setattr(supervision, "run_supervised", fake)
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    app.text_input[0].set_value("synthetic-unread-photos").run()
    app.button[1].click().run()
    assert not app.exception
    assert len(calls) == 1
    app.run()
    assert len(calls) == 1
    assert len(app.get("download_button")) == 0
    assert any("尚无通过全部验证" in item.value for item in app.warning)


def test_apptest_plan_is_read_only_and_each_execution_has_own_file(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import integrations.demo_pipeline.pipeline as pipeline
    import integrations.mvs_surface.supervision as supervision
    plans, runs = [], []
    def fake_plan(options):
        plans.append(options)
        return {"target": "blender", "dependencies": {"synthetic_test": True}}
    def fake(argv, **kwargs):
        run = Path(argv[argv.index("--output") + 1])
        run.mkdir()
        blend = run / "synthetic_ui_fixture.blend"
        blend.write_bytes(b"UI fixture only, not a valid Blender document")
        state = {"status": "SUCCESS", "exit_code": 0, "deliverable_ready": True, "stages": {"blender": {
            "status": "SUCCESS", "path": str(blend), "reopen_verified": True,
            "sha256": hashlib.sha256(blend.read_bytes()).hexdigest()}}}
        (run / "pipeline.json").write_text(json.dumps(state), encoding="utf-8")
        runs.append(run)
        return {"status": "completed", "root_exit_code": 0, "reclaim_confirmed": True}
    monkeypatch.setattr(pipeline, "plan", fake_plan)
    monkeypatch.setattr(supervision, "run_supervised", fake)
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    app.text_input[0].set_value("synthetic-unread-photos").run()
    app.button[0].click().run()
    assert len(plans) == 1 and not runs
    app.button[1].click().run()
    assert not app.exception
    assert len(runs) == 1 and len(app.get("download_button")) == 1
    app.run()
    assert len(runs) == 1
    app.button[1].click().run()
    assert len(runs) == 2 and runs[0] != runs[1]
    assert len(app.get("download_button")) == 1
    assert str(runs[-1].parent) in " ".join(item.value for item in app.caption)


def test_apptest_cad_checkbox_routes_plan_and_command(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from integrations.demo_pipeline import pipeline
    from integrations.mvs_surface import supervision
    plans,calls=[],[]
    monkeypatch.setattr(pipeline,'plan',lambda options: plans.append(options) or {'target':options.target})
    monkeypatch.setattr(supervision,'run_supervised',lambda argv,**kw:calls.append(argv) or
                       {'status':'completed','root_exit_code':4,'reclaim_confirmed':True})
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.text_input[0].set_value('synthetic-unread-photos').run()
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert plans[-1].target=='cad' and not calls
    app.button[1].click().run()
    assert len(calls)==1 and calls[0][calls[0].index('--target')+1]=='cad'
    app.run()
    assert len(calls)==1 and not app.exception


@pytest.mark.parametrize('failure',[KeyError('mesh'),RuntimeError('broken native record')])
def test_apptest_broken_cad_record_keeps_blender_download(tmp_path,monkeypatch,failure):
    from streamlit.testing.v1 import AppTest
    from integrations.demo_pipeline import cad
    from integrations.mvs_surface import supervision
    base=ROOT/'outputs/delegation/GENERAL_PIPELINE/20261011_cad_unified_main' / ('ui_broken_'+uuid.uuid4().hex[:8])
    run=existing_fixture(base)
    state=json.loads((run/'pipeline.json').read_text())
    state.update(target='cad',geometry_deliverable_ready=True)
    state['stages']['cad']={'status':'SUCCESS','output':str(run/'cad')}
    (run/'pipeline.json').write_text(json.dumps(state))
    def fail(*args,**kw):raise failure
    def forbidden(*args,**kw):raise AssertionError('read-only loading cannot execute')
    monkeypatch.setattr(cad,'verify_cad_output',fail)
    monkeypatch.setattr(supervision,'run_supervised',forbidden)
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.text_input[-1].set_value(str(run)).run()
    app.button[-1].click().run()
    assert not app.exception
    assert len(app.get('download_button'))==2  # Existing Blender plus report, no CAD.
    assert any('CAD 记录未通过核验' in x.value for x in app.error)
    assert next(button for button in app.button if button.label=='生成 FCStd / STEP').disabled


@pytest.mark.parametrize("outer_code,inner_code,ready,expected", [
    (1, 1, True, True), (1, 1, False, False), (4, 4, True, False),
    (1, 0, True, False),
])
def test_apptest_partial_delivery_exit_contract(monkeypatch, outer_code, inner_code, ready, expected):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import integrations.mvs_surface.supervision as supervision
    def fake(argv, **kwargs):
        run = Path(argv[argv.index("--output") + 1])
        run.mkdir()
        blend = run / "synthetic_ui_fixture.blend"
        blend.write_bytes(b"UI fixture only, not a valid Blender document")
        state = {"status": "PARTIAL", "exit_code": inner_code, "deliverable_ready": ready,
                 "stages": {"blender": {"status": "SUCCESS", "path": str(blend),
                    "reopen_verified": True, "sha256": hashlib.sha256(blend.read_bytes()).hexdigest()}}}
        (run / "pipeline.json").write_text(json.dumps(state), encoding="utf-8")
        return {"status": "completed", "root_exit_code": outer_code, "reclaim_confirmed": True}
    monkeypatch.setattr(supervision, "run_supervised", fake)
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    app.text_input[0].set_value("synthetic-unread-photos").run()
    app.button[1].click().run()
    assert not app.exception
    assert bool(app.get("download_button")) is expected
    if expected:
        assert any("部分视图未覆盖，已生成可查看文件" in item.value for item in app.warning)


def existing_fixture(base, *, wrapper=True, exit_code=0):
    run = base / "pipeline"
    run.mkdir(parents=True)
    blend = run / "synthetic_ui_fixture.blend"
    blend.write_bytes(b"UI fixture only")
    mesh = run / "synthetic_mesh.ply"
    mesh.write_bytes(b"UI fixture only, not a reconstructed mesh")
    state = {"status": "SUCCESS", "exit_code": 0, "deliverable_ready": True,
             "stages": {"surface": {"status": "SUCCESS", "artifacts": [{
                "path": str(mesh), "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}]},
                "blender": {"status": "SUCCESS", "reopen_verified": True,
                "path": str(blend), "sha256": hashlib.sha256(blend.read_bytes()).hexdigest()}}}
    (run / "pipeline.json").write_text(json.dumps(state), encoding="utf-8")
    (run / "report").mkdir()
    (run / "report/index.html").write_text("synthetic report fixture", encoding="utf-8")
    if wrapper:
        (base / "UI_SUPERVISION.json").write_text(json.dumps({"status": "completed",
            "root_exit_code": exit_code, "reclaim_confirmed": True, "argv": ["cli", "--output", str(run)]}), encoding="utf-8")
    return run


def test_existing_rejects_outside_and_mismatched_wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(page, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()
    outside = existing_fixture(tmp_path / "outside")
    with pytest.raises(ValueError):
        page.load_existing(str(outside))
    run = existing_fixture(tmp_path / "outputs/task")
    wrapper = run.parent / "UI_SUPERVISION.json"
    candidate = json.loads(wrapper.read_text())
    candidate["argv"][-1] = str(outside)
    wrapper.write_text(json.dumps(candidate))
    assert page.load_existing(str(run))["supervision"] == {}


@pytest.mark.parametrize("wrapper,exit_code,download_count,changed", [(True, 0, 2, False), (True, 4, 1, False), (False, 0, 1, False), (True, 0, 1, True)])
def test_apptest_load_existing_is_readonly(wrapper, exit_code, download_count, changed, tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import integrations.mvs_surface.supervision as supervision
    # Stay under actual project outputs because AppTest imports a fresh page module.
    base = ROOT / "outputs/delegation/GENERAL_PIPELINE/20261011_ui_load_existing_03" / (tmp_path.name + "_" + uuid.uuid4().hex[:8])
    run = existing_fixture(base, wrapper=wrapper, exit_code=exit_code)
    if changed:
        (run / "synthetic_mesh.ply").write_bytes(b"changed after successful task")
    def forbidden(*args, **kwargs):
        raise AssertionError("Loading an existing task must not launch a process")
    monkeypatch.setattr(supervision, "run_supervised", forbidden)
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    app.text_input[-1].set_value(str(run)).run()
    app.button[-1].click().run()
    assert not app.exception
    assert len(app.get("download_button")) == download_count


def cli_stage_fixture(base):
    run = existing_fixture(base, wrapper=False)
    state = json.loads((run / "pipeline.json").read_text())
    state["errors"] = []
    for name in ("input", "sfm", "segmentation", "native_subset", "mvs_bridge", "mvs_masks",
                 "surface_plan", "surface_reconstruction", "blender_export"):
        entry_path = run / "logs" / name / "ENTRY.json"
        entry_path.parent.mkdir(parents=True)
        argv = ["synthetic_test_cli", name, str(run)]
        code = 1 if name == "sfm" else 0
        entry = {"status": "completed", "reclaim_confirmed": True, "errors": [],
                 "log_errors": [], "argv": argv, "root_exit_code": code}
        entry_path.write_text(json.dumps(entry))
        state["stages"][name] = {"argv": argv, "exit_code": code, "entry_path": str(entry_path)}
    (run / "pipeline.json").write_text(json.dumps(state))
    return run


@pytest.mark.parametrize("change", ["none", "missing", "rc", "artifact", "escape"])
def test_apptest_existing_cli_stage_evidence(change, tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import integrations.mvs_surface.supervision as supervision
    base = ROOT / "outputs/delegation/GENERAL_PIPELINE/20261011_ui_load_existing_03" / ("cli_" + uuid.uuid4().hex[:12])
    run = cli_stage_fixture(base)
    if change == "missing":
        (run / "logs/mvs_masks/ENTRY.json").unlink()
    elif change == "rc":
        entry = run / "logs/input/ENTRY.json"
        value = json.loads(entry.read_text())
        value["root_exit_code"] = 1
        entry.write_text(json.dumps(value))
    elif change == "artifact":
        (run / "synthetic_ui_fixture.blend").write_bytes(b"changed")
    elif change == "escape":
        state_path = run / "pipeline.json"
        value = json.loads(state_path.read_text())
        value["stages"]["input"]["entry_path"] = str(base / "ENTRY.json")
        state_path.write_text(json.dumps(value))
    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only CLI load must not launch a process")
    monkeypatch.setattr(supervision, "run_supervised", forbidden)
    assert page.load_existing(str(run))["stage_evidence_verified"] is (change in ("none", "artifact"))
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    app.text_input[-1].set_value(str(run)).run()
    app.button[-1].click().run()
    assert not app.exception
    assert len(app.get("download_button")) == (2 if change == "none" else 1)
    if change == "none":
        assert any("各阶段执行记录已核对" in item.value for item in app.caption)
    app.run()
    assert len(app.get("download_button")) == (2 if change == "none" else 1)
