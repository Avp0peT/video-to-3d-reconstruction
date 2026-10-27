"""Diagnostic dispatch/cleanup tests; no native process or model imports."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from integrations.demo_pipeline import runtime as rt, sam_score_worker as sw
from integrations.demo_pipeline import sam_score_diagnostic as sd
from test_demo_runtime import setup, allow_preflight, fake_adapter


@pytest.mark.parametrize("diagnostic_status,expected", [("COMPLETE", 0), ("INCOMPLETE", 4)])
def test_explicit_profile_uses_same_supervision_and_rejects_missing_evidence(setup, monkeypatch, diagnostic_status, expected):
    allow_preflight(setup, monkeypatch)
    capture, calls, worker, original = fake_adapter(setup, monkeypatch)
    def configure(config, env):
        env["SAM3_SCORE_BINDING"] = "fake-bound-record"
        return {"fixture": True}
    monkeypatch.setattr(sw, "configure", configure)
    monkeypatch.setattr(sw, "assess", lambda *args: {"status": diagnostic_status})
    assert rt.execute_config(setup.path, score_diagnostics=True) == expected
    request = capture["request"]
    assert request.worker_module == "integrations.demo_pipeline.sam_score_worker"
    assert request.confidence_threshold == 0.5 and request.resolution == 1008
    assert request.resource_policy["ram_commit_limit_bytes"] == 12 << 30
    assert request.budgets["batch_seconds"] == 900
    assert len(calls) == 1 and worker.launch_worker is original
    result = rt.read_json(setup.output / "PRODUCT_OUTCOME.json")
    assert result["product_exit_code"] == 0
    assert result["score_diagnostics"]["status"] == diagnostic_status


@pytest.mark.parametrize("fail", [False, True])
def test_worker_preserves_return_and_business_exception_and_restores(tmp_path, monkeypatch, fail):
    sealed = tmp_path / "mmap_review_worker.py"
    sealed.write_text("# fake entry only\n")
    binding = {"policy": sw.POLICY, "files": [], "expected_indices": [10, 15],
               "processor_sha256": "0" * 64, "output": str(tmp_path),
               "sealed_worker": str(sealed), "sealed_worker_sha256": rt.sha256(sealed),
               "max_queries": 256, "max_record_bytes": 65536, "max_total_bytes": 4 << 20}
    binding_path = tmp_path / "SCORE_DIAGNOSTIC_BINDING.json"
    rt.write_json(binding_path, binding)
    monkeypatch.setenv("SAM3_SCORE_BINDING", str(binding_path))
    sentinel = object()
    class Backend:
        def __init__(self):
            self.inner = SimpleNamespace(_processor=object())
        def segment(self, *args, **kwargs):
            if fail:
                raise RuntimeError("original business error")
            return sentinel
    original = Backend.segment
    base = SimpleNamespace(AuditedBackend=Backend)
    def business(argv):
        backend = Backend()
        for index in binding["expected_indices"]:
            assert backend.segment(None, None, None, 1, 1, index) is sentinel
        return 0
    entry = SimpleNamespace(__file__=str(sealed), base=base, main=business)
    monkeypatch.setattr(sw.importlib, "import_module", lambda name: entry if name == "mmap_review_worker" else base)
    class FakeDiagnostic:
        def __init__(self, sink, **kwargs):
            self.sink = sink
        @contextmanager
        def observe(self, processor, index):
            yield
            self.sink(json.dumps({"input_index": index, "threshold": 0.5,
                "retained_scores_match": True, "processor_source_sha256": "0" * 64}) + "\n")
    monkeypatch.setattr(sd, "ScoreDiagnostic", FakeDiagnostic)
    if fail:
        with pytest.raises(RuntimeError, match="original business error"):
            sw.main([])
    else:
        assert sw.main([]) == 0
    assert Backend.segment is original
    status = rt.read_json(tmp_path / "score_status.json")
    assert status["restored"] is True and status["trace_restored"] is True
    assert status["completed_indices"] == ([] if fail else [10, 15])
    if not fail:
        config = SimpleNamespace(output_dir=tmp_path, indices=[10, 15])
        assert sw.assess(config, binding)["status"] == "COMPLETE"
        (tmp_path / "prefilter_scores.jsonl").write_text("")
        assert sw.assess(config, binding)["status"] == "INCOMPLETE"
