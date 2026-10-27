"""Synthetic graph proves wiring/report identity gates, never real reconstruction."""
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_photo_to_blender_pipeline import rig, native_fixture, renderer, good_surface
from integrations.demo_pipeline import pipeline as p
from integrations.demo_pipeline.report import render_report
from integrations.demo_pipeline.research_code6 import binding


def surface_selected(**kwargs):
    result = good_surface(**kwargs)
    result["surface"]["selected_model_id"] = "selected-actual"
    return result


def research_fixture(**kwargs):
    output = kwargs["output"]
    output.mkdir()
    for name in ("SOURCE_POINTS.json", "INDEXED_CODE3D.json"):
        (output / name).write_text('{"fixture":"UI/wiring only"}')
    result = {"status": "SUCCESS", "complete": True, "source_model_id": kwargs["model_id"],
        "algorithm_called": "kozlov_lab.stereo.official.indexed_code3d.indexed_body_code",
        "source_sfm_status": p.read(kwargs["sfm_run"] / "run.json")["status"],
        "source_bindings": [binding(kwargs["sfm_run"] / "run.json")],
        "artifacts": [binding(output / name) for name in ("SOURCE_POINTS.json", "INDEXED_CODE3D.json")],
        "native_point_ids": [1, 2, 3, 4, 5, 6], "counts": {"points": 6, "entries": 210}}
    p.save(output / "RESULT.json", result)
    return result


@pytest.mark.parametrize("sfm", ["SUCCESS", "PARTIAL"])
def test_selected_model_actually_passed_and_source_partial_preserved(rig, sfm):
    root, options = rig
    execute, _ = native_fixture(sfm=sfm)
    seen = []
    def build(**kwargs):
        seen.append(kwargs)
        return research_fixture(**kwargs)
    run, code = p.run_pipeline(options, root=root, executor=execute, renderer=renderer,
                              surface_builder=surface_selected, research_builder=build)
    state = p.read(run / "pipeline.json")
    assert len(seen) == 1 and seen[0]["model_id"] == "selected-actual"
    assert seen[0]["sfm_run"] / "run.json" == Path(state["stages"]["sfm"]["run_json"])
    assert state["stages"]["research"]["status"] == "SUCCESS"
    assert state["stages"]["research"]["source_sfm_status"] == sfm
    assert state["deliverable_ready"] is True


def test_research_failure_retains_successful_blend(rig):
    root, options = rig
    execute, _ = native_fixture()
    def fail(**kwargs):
        raise ValueError("honest research refusal")
    run, code = p.run_pipeline(options, root=root, executor=execute, renderer=renderer,
                              surface_builder=surface_selected, research_builder=fail)
    state = p.read(run / "pipeline.json")
    assert code == 0 and state["deliverable_ready"] is True
    assert state["stages"]["research"]["status"] == "FAILED"
    assert Path(state["stages"]["blender"]["path"]).is_file()


def test_disabled_does_not_call_research(rig):
    root, options = rig
    options.research_code6 = False
    execute, _ = native_fixture()
    p.run_pipeline(options, root=root, executor=execute, renderer=renderer,
                   surface_builder=surface_selected, research_builder=lambda **k: pytest.fail("disabled"))


def test_default_research_uses_bounded_cli_with_selected_model(rig):
    root, options = rig
    execute, calls = native_fixture()
    def wrapper(root, run, name, argv, **kwargs):
        if name == "research":
            assert kwargs["seconds"] == 30 and kwargs["memory"] == 1 << 30
            assert argv[argv.index("--model-id") + 1] == "selected-actual"
            research_fixture(sfm_run=Path(argv[argv.index("--sfm-run") + 1]),
                model_id="selected-actual", output=Path(argv[argv.index("--output") + 1]))
            return {"status": "completed", "root_exit_code": 0, "reclaim_confirmed": True, "errors": [], "log_errors": []}
        return execute(root, run, name, argv, **kwargs)
    run, code = p.run_pipeline(options, root=root, executor=wrapper, renderer=renderer,
                              surface_builder=surface_selected)
    assert p.read(run / "pipeline.json")["stages"]["research"]["status"] == "SUCCESS"


@pytest.mark.parametrize("flag,expected", [(None, True), ("--research-code6", True), ("--no-research-code6", False)])
def test_cli_research_flags(monkeypatch, flag, expected):
    import importlib.util
    spec = importlib.util.spec_from_file_location("photo_cli_test", ROOT / "scripts/run_photo_demo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = []
    monkeypatch.setattr(cli, "plan", lambda options: seen.append(options.research_code6) or {})
    args = ["plan", "--source", "synthetic", "--prompt", "object"] + ([flag] if flag else [])
    assert cli.main(args) == 0 and seen == [expected]


@pytest.mark.parametrize("changed", ["none", "result", "artifact", "model", "escape"])
def test_report_links_only_current_complete_bound_research(tmp_path, changed):
    sfm = tmp_path / "source"
    sfm.mkdir()
    p.save(sfm / "run.json", {"status": "SUCCESS"})
    result = research_fixture(sfm_run=sfm, model_id="actual", output=tmp_path / "research")
    metadata = {"research": {"status": "SUCCESS", "selected_model_id": "actual",
        "result": binding(tmp_path / "research/RESULT.json"), "artifacts": result["artifacts"]}}
    if changed == "result":
        (tmp_path / "research/RESULT.json").write_text('{}')
    elif changed == "artifact":
        (tmp_path / "research/INDEXED_CODE3D.json").write_text('changed')
    elif changed == "model":
        metadata["research"]["selected_model_id"] = "other"
    elif changed == "escape":
        metadata["research"]["result"] = binding(sfm / "run.json")
    p.save(tmp_path / "REPORT_DATA.json", metadata)
    output = render_report(tmp_path)
    page = output.read_text(encoding="utf-8")
    assert ("下载研究运行记录" in page) is (changed == "none")
    validation = p.read(output.parent / "REPORT_VALIDATION.json")
    assert validation["research"]["available"] is (changed == "none")
