"""Small pure/mock checks for the scripted segmentation entry; no real worker."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from integrations.demo_pipeline import runtime as rt


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "project"
    output = root / "outputs/demo/segmentation"
    io_run = root / "outputs/prepared"
    package = root / rt.PACKAGE_RELATIVE
    python = root / "integrations/sam3/env/Scripts/python.exe"
    checkpoint = tmp_path / "sam3.pt"
    for directory in (output, io_run, package, python.parent):
        directory.mkdir(parents=True)
    python.write_bytes(b"fixture interpreter; never executed")
    checkpoint.write_bytes(b"fixture checkpoint; never loaded")
    put(io_run / "run.json", {"status": "SUCCESS"})
    (io_run / "frames.jsonl").write_text("{}\n", encoding="utf-8")
    dependency = root / "integrations/sam3/sam3_adapter/runner.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("# fixture; never imported\n", encoding="utf-8")
    row = {"path": str(dependency.relative_to(root)).replace("\\", "/"), "sha256": rt.sha256(dependency)}
    old = {"path": "outputs/old_photos/missing.png", "sha256": "0" * 64}
    base = {"files": [row, old], "checkpoint": {"path": str(checkpoint), "sha256": rt.sha256(checkpoint)},
            "package_inventory": [], "checkout_head": "fixture-head"}
    authority = {"files": [row], "project_root": str(root),
                 "checkpoint_stat_only": {"bytes": checkpoint.stat().st_size,
                                           "mtime_ns": checkpoint.stat().st_mtime_ns}}
    put(package / "START_BASELINE.json", base)
    put(package / "AUTHORITY_BINDINGS.json", authority)
    worker_file = package / "worker_entry/mmap_review_worker.py"
    worker_file.parent.mkdir()
    worker_file.write_text("# fixture; never executed\n", encoding="utf-8")
    bundle = {"files": [{"path": name, "sha256": rt.sha256(package / name)} for name in
              ("START_BASELINE.json", "AUTHORITY_BINDINGS.json", "worker_entry/mmap_review_worker.py")]}
    put(package / "BUNDLE.json", bundle)
    monkeypatch.setattr(rt, "BUNDLE_SHA256", rt.sha256(package / "BUNDLE.json"))
    monkeypatch.setattr(rt, "PROJECT_ROOT", root)
    config = {"root": str(root), "io_run": str(io_run), "output_dir": str(output),
        "checkpoint": str(checkpoint), "checkpoint_sha256": rt.sha256(checkpoint),
        "worker_package": str(package), "sam3_python": str(python),
        "text_prompt": "beverage can", "object_id": "can-01", "indices": [2, 0, 9, 4]}
    config_path = output / "config.json"
    put(config_path, config)
    return SimpleNamespace(root=root, output=output, io_run=io_run, package=package, python=python,
        dependency=dependency, config=config, path=config_path, checkpoint=checkpoint)


def allow_preflight(setup, monkeypatch):
    monkeypatch.setattr(rt.sys, "executable", str(setup.python))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(rt, "_heavy_modules", lambda: [])
    monkeypatch.setattr(rt.metadata, "distributions", lambda **kwargs: [])
    monkeypatch.setattr(rt.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 << 30))
    git_exe=setup.root/'git.exe';git_exe.write_bytes(b'fixture Git; not executed')
    monkeypatch.setattr(rt,'discover_git',lambda explicit=None:git_exe)
    def git(argv, **kwargs):
        assert argv[0] == str(git_exe)  # mock only, never launches a process
        return SimpleNamespace(stdout="fixture-head\n" if "rev-parse" in argv else "")
    monkeypatch.setattr(rt.subprocess, "run", git)


def test_explicit_git_absolute_path_and_tool_binding(setup,monkeypatch):
    allow_preflight(setup,monkeypatch)
    config=rt.load_config(setup.path)
    result=rt.preflight(config)
    assert result['status']=='PASS'
    assert Path(result['git_tool_binding']['path']).is_absolute()
    assert result['git_tool_binding']['sha256']==rt.sha256(config.git_exe)
    assert all(argv[0]==str(config.git_exe) for argv in result['git_commands'])
    assert [argv[3:] for argv in result['git_commands']]==[['rev-parse','HEAD'],['status','--porcelain']]


def test_missing_git_blocks_before_runtime_or_checkpoint_rehash(setup,monkeypatch):
    allow_preflight(setup,monkeypatch);config=rt.load_config(setup.path)
    monkeypatch.setattr(rt,'discover_git',lambda explicit=None:None)
    monkeypatch.setattr(rt,'verify_bindings',lambda *args:pytest.fail('bindings hashed before missing Git rejection'))
    result=rt.preflight(config)
    assert result['status']=='PRECONDITION_BLOCKED' and not result['checkpoint_rehashed']
    assert 'Git executable missing' in result['errors'][0]


def test_old_config_git_discovery_and_missing_or_relative_explicit_refused(setup,monkeypatch):
    git=setup.root/'git.exe';git.write_bytes(b'TEST_GIT')
    seen=[]
    monkeypatch.setattr(rt,'discover_git',lambda explicit=None:seen.append(explicit) or git)
    assert rt.load_config(setup.path).git_exe==git and seen==[None]
    payload=rt.read_json(setup.path);payload['git_exe']='relative/git.exe';put(setup.path,payload)
    with pytest.raises(ValueError,match='absolute path'):rt.load_config(setup.path)
    payload['git_exe']=str(git);put(setup.path,payload)
    monkeypatch.setattr(rt,'discover_git',lambda explicit=None:None)
    with pytest.raises(ValueError,match='Git executable missing'):rt.load_config(setup.path)


@pytest.mark.parametrize('fault',['wrong_head','dirty_checkout'])
def test_absolute_git_still_enforces_head_and_clean_checkout(setup,monkeypatch,fault):
    allow_preflight(setup,monkeypatch);config=rt.load_config(setup.path)
    def git(argv,**kwargs):
        assert argv[0]==str(config.git_exe)
        return SimpleNamespace(stdout=('wrong-head' if fault=='wrong_head' else 'fixture-head')
                               if 'rev-parse' in argv else ' M tracked.py' if fault=='dirty_checkout' else '')
    monkeypatch.setattr(rt.subprocess,'run',git)
    report=rt.preflight(config)
    assert report['status']=='PRECONDITION_BLOCKED' and 'revision or clean-state differs' in report['errors'][0]


def fake_adapter(setup, monkeypatch, *, fail=False, evidence=True):
    calls = []
    process = SimpleNamespace(pid=321, args=["fixture worker, not executed"], returncode=0)
    process.poll = lambda: process.returncode
    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return process
    worker = SimpleNamespace(launch_worker=launch)
    def kill(proc, deadline):
        proc.returncode = -9
        return {"requested": True, "exit_code": -9, "test_fixture": True}
    worker.kill_tree = kill
    capture = {}
    def execute(request):
        capture["request"] = request
        worker.launch_worker(str(setup.python), setup.output / "job.json", setup.root / "integrations/sam3",
            setup.output / "worker.stdout", setup.output / "worker.stderr", request.worker_extra_env,
            str(setup.root), request.worker_module)
        run_dir = request.output_root / "fixture"
        run_dir.mkdir(parents=True)
        put(run_dir / "manifest.json", {"fixture": True, "preserved": True})
        if fail:
            process.returncode = None
            raise RuntimeError("fixture failure after launch")
        if evidence:
            (setup.output / "mmap_forward.jsonl").write_text('{"fixture":true}\n', encoding="utf-8")
            put(setup.output / "mmap_status.json", {"complete": True, "business_rc": 0,
                "write_failures": 0, "dropped_events": 0, "restoration_errors": 0,
                "diagnostic_errors": 0, "active_calls": 0,
                "events_sha256": rt.sha256(setup.output / "mmap_forward.jsonl")})
            (setup.output / "gate_events.jsonl").write_text('{"section":"gate_a","status":"PASS","fixture":true}\n', encoding="utf-8")
        return SimpleNamespace(status="SUCCESS", exit_code=0, run_dir=run_dir,
            manifest={"counts": {"requested": 4, "success": 4}, "worker": {"returncode": 0,
                      "ownership_identity": {"fixture": True}}})
    runner = SimpleNamespace(RunRequest=lambda **kw: SimpleNamespace(**kw), execute=execute)
    api = SimpleNamespace(runner=runner, worker=worker, Budgets=lambda **kw: kw, ResourcePolicy=lambda **kw: kw)
    monkeypatch.setattr(rt, "_adapter", lambda config: api)
    return capture, calls, worker, launch


def test_runtime_import_is_lightweight():
    assert not {"torch", "sam3"}.intersection(sys.modules)


@pytest.mark.parametrize("change", [{"indices": [0, 0]}, {"indices": [True]},
                                  {"indices": []}, {"text_prompt": ""}, {"unknown": 1}])
def test_config_rejects_ambiguous_or_silently_ignored_options(setup, change):
    put(setup.path, {**setup.config, **change})
    with pytest.raises(ValueError):
        rt.load_config(setup.path)


def test_root_and_overlap_cannot_redirect_execution(setup):
    put(setup.path, {**setup.config, "root": str(setup.root.parent)})
    with pytest.raises(ValueError, match="root must equal"):
        rt.load_config(setup.path)
    put(setup.path, {**setup.config, "output_dir": str(setup.io_run)})
    with pytest.raises(ValueError, match="disjoint"):
        rt.load_config(setup.path)


def test_missing_historical_photos_are_explicitly_excluded_not_new_inputs(setup):
    report = rt.verify_bindings(rt.load_config(setup.path))
    assert report["match"]
    assert [row["path"] for row in report["excluded"]] == ["outputs/old_photos/missing.png"]
    assert any(row["path"].endswith("sam3_adapter/runner.py") for row in report["checked"])


@pytest.mark.parametrize("target", ["runtime", "worker", "manifest", "bundle"])
def test_protected_runtime_or_package_drift_is_not_bypassed(setup, target):
    path = {"runtime": setup.dependency, "worker": setup.package / "worker_entry/mmap_review_worker.py",
            "manifest": setup.package / "START_BASELINE.json", "bundle": setup.package / "BUNDLE.json"}[target]
    path.write_bytes(path.read_bytes() + b" ")
    config = rt.load_config(setup.path)
    if target in ("manifest", "bundle"):
        with pytest.raises(ValueError, match="digest differs|manifest drift"):
            rt.verify_bindings(config)
    else:
        report = rt.verify_bindings(config)
        assert not report["match"] and report["errors"]


def test_preflight_hash_and_environment_checks_are_actual_read_only_facts(setup, monkeypatch):
    allow_preflight(setup, monkeypatch)
    result = rt.preflight(rt.load_config(setup.path))
    assert result["status"] == "PASS"
    assert result["checkpoint_rehashed"] is True
    assert result["checkpoint_actual_sha256"] == rt.sha256(setup.checkpoint)
    assert result["package_inventory_match"] is True


def test_preflight_block_writes_results_without_loading_adapter(setup, monkeypatch):
    monkeypatch.setattr(rt, "_adapter", lambda config: pytest.fail("adapter must not load"))
    assert rt.execute_config(setup.path) == 3
    assert rt.read_json(setup.output / "PRECONDITIONS.json")["status"] == "PRECONDITION_BLOCKED"
    assert rt.read_json(setup.output / "PRODUCT_LAUNCH.json")["launched"] is False
    assert rt.read_json(setup.output / "PRODUCT_OUTCOME.json")["exit_code"] == 3


def test_success_generalizes_request_and_preserves_original_factory(setup, monkeypatch):
    allow_preflight(setup, monkeypatch)
    capture, calls, worker, original = fake_adapter(setup, monkeypatch)
    before = (setup.io_run / "run.json").read_bytes()
    assert rt.execute_config(setup.path, argv=[str(setup.python), "-B", "demo_segment_worker.py", "--config", str(setup.path)]) == 0
    request = capture["request"]
    assert request.indices == [2, 0, 9, 4] and request.text_prompt == "beverage can"
    assert request.object_id == "can-01" and request.worker_module == "mmap_review_worker"
    assert request.use_worker is True and request.allow_hf_download is False
    assert request.budgets == dict(load_seconds=300, frame_seconds=300, batch_seconds=900, reclaim_seconds=10)
    assert request.resource_policy == dict(ram_commit_limit_bytes=12 << 30, vram_limit_bytes=7 << 30, cpu_threads=2)
    assert len(calls) == 1 and worker.launch_worker is original
    assert (setup.io_run / "run.json").read_bytes() == before
    assert rt.read_json(setup.output / "PRODUCT_LAUNCH.json")["returncode"] == 0
    assert rt.read_json(setup.output / "PRODUCT_OUTCOME.json")["evidence"]["independent_full_assessment"] == "NOT_RUN"
    with pytest.raises(ValueError, match="no retry"):
        rt.execute_config(setup.path)
    assert len(calls) == 1


def test_runtime_failure_retains_product_manifest_and_actual_launch_cleanup(setup, monkeypatch):
    allow_preflight(setup, monkeypatch)
    capture, calls, worker, original = fake_adapter(setup, monkeypatch, fail=True)
    assert rt.execute_config(setup.path) == 4
    assert rt.read_json(setup.output / "product_run/fixture/manifest.json")["preserved"]
    record = rt.read_json(setup.output / "PRODUCT_LAUNCH.json")
    assert record["pid"] == 321 and record["returncode"] == -9
    assert record["fallback_cleanup"]["requested"] is True
    assert worker.launch_worker is original and len(calls) == 1
    assert "fixture failure" in rt.read_json(setup.output / "PRODUCT_OUTCOME.json")["error"]


def test_product_success_without_worker_evidence_is_not_overall_success(setup, monkeypatch):
    allow_preflight(setup, monkeypatch)
    fake_adapter(setup, monkeypatch, evidence=False)
    assert rt.execute_config(setup.path) == 4
    outcome = rt.read_json(setup.output / "PRODUCT_OUTCOME.json")
    assert outcome["product_exit_code"] == 0 and outcome["status"] == "BACKEND_ERROR"


@pytest.mark.parametrize("name", ["gate_events.jsonl", "PRECONDITIONS.json.pending", "tmp"])
def test_existing_stage_content_cannot_mix_or_redirect_new_evidence(setup, name):
    (setup.output / name).write_text("old content", encoding="utf-8")
    with pytest.raises(ValueError, match="only its new config"):
        rt.execute_config(setup.path)
    assert not (setup.output / "SEGMENT_ENTRY.json").exists()


def test_launch_record_failure_preserves_product_result_path(setup, monkeypatch):
    allow_preflight(setup, monkeypatch)
    fake_adapter(setup, monkeypatch)
    original = rt.write_json
    def fail_final_launch(path, value):
        if path.name == "PRODUCT_LAUNCH.json" and "end" in value:
            raise OSError("fixture record failure")
        original(path, value)
    monkeypatch.setattr(rt, "write_json", fail_final_launch)
    assert rt.execute_config(setup.path) == 4
    outcome = rt.read_json(setup.output / "PRODUCT_OUTCOME.json")
    assert outcome["product_exit_code"] == 0
    assert Path(outcome["run_dir"]).name == "fixture"
    assert "launch evidence write failed" in outcome["error"]
