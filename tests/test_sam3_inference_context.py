"""Pure Python checks of SAM3 forward contexts and worker error propagation.

No torch/SAM3 import, CUDA operation, model load, or OS child is needed. The
worker protocol is exercised by calling child_main in this test process.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = PROJECT_ROOT / "integrations" / "sam3"
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from sam3_adapter import backends, binding, deadline, runner, worker  # noqa: E402


class FakeOOM(RuntimeError):
    pass


class DerivedOOM(FakeOOM):
    pass


class WorkerMemoryStop(backends.ResourceLimit):
    """Name deliberately omits ResourceLimit to distinguish type checks."""


class FakeTorch:
    bfloat16 = object()

    def __init__(self):
        self.active = {"inference": [], "autocast": []}
        self.trace = []
        self.autocast_calls = []
        self.cuda = SimpleNamespace(OutOfMemoryError=FakeOOM, empty_cache=lambda: None)

    @contextmanager
    def _scope(self, kind):
        token = object()
        self.trace.append(("enter", kind, token))
        self.active[kind].append(token)
        try:
            yield token
        finally:
            assert self.active[kind].pop() is token
            self.trace.append(("exit", kind, token))

    def inference_mode(self):
        return self._scope("inference")

    def autocast(self, *, device_type, dtype):
        assert device_type == "cuda"
        assert dtype is self.bfloat16
        self.autocast_calls.append((device_type, dtype))
        return self._scope("autocast")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (2, 3), color=(4, 8, 12)).save(image_path)
    backend = backends.Sam3Backend(
        repo_dir=tmp_path / "repo", checkpoint_path=None, checkpoint_sha256=None,
        device="cuda", confidence_threshold=0.5)
    torch = FakeTorch()
    callbacks = []
    failure = {"stage": None, "exception": None}
    state = {"masks": None, "scores": None}

    def observe(stage):
        assert len(torch.active["inference"]) == 1
        expect_autocast = backend.device.split(":", 1)[0] == "cuda"
        assert len(torch.active["autocast"]) == int(expect_autocast)
        current = (torch.active["inference"][0],
                   torch.active["autocast"][0] if expect_autocast else None)
        if callbacks:
            assert current == callbacks[0][1]
        callbacks.append((stage, current))
        if failure["stage"] == stage:
            raise failure["exception"]

    class Processor:
        def set_image(self, image):
            assert image.size == (2, 3) and image.mode == "RGB"
            observe("image")
            return state

        def set_text_prompt(self, *, prompt, state):
            assert prompt == "skull"
            assert state is expected_state
            observe("text")
            return state

    expected_state = state
    backend._torch = torch
    backend._processor = Processor()
    monkeypatch.setattr(backends, "candidates_from_processor_output", lambda *a, **k: [])
    return SimpleNamespace(backend=backend, torch=torch, callbacks=callbacks,
                           failure=failure, image=image_path)


def segment(rig):
    return rig.backend.segment(rig.image, "skull", "skull-01", 2, 3, 17)


def assert_contexts_restored(rig, cuda=True):
    assert rig.torch.active == {"inference": [], "autocast": []}
    expected = [("enter", "inference")]
    if cuda:
        expected.append(("enter", "autocast"))
        expected.append(("exit", "autocast"))
    expected.append(("exit", "inference"))
    assert [(action, kind) for action, kind, token in rig.torch.trace] == expected
    for kind in ("inference", "autocast"):
        tokens = [token for action, entry_kind, token in rig.torch.trace if entry_kind == kind]
        if tokens:
            assert tokens[0] is tokens[-1]


@pytest.mark.parametrize("stage", [None, "image", "text"], ids=["success", "image_error", "text_error"])
def test_same_inference_and_autocast_contexts_cover_both_callbacks(rig, stage):
    error = RuntimeError("fixture callback error")
    rig.failure.update(stage=stage, exception=error)
    if stage:
        with pytest.raises(backends.BackendError) as caught:
            segment(rig)
        assert caught.value.__cause__ is error
    else:
        assert segment(rig) == []
    assert [name for name, scopes in rig.callbacks] == (["image"] if stage == "image" else ["image", "text"])
    assert len(rig.torch.autocast_calls) == 1
    assert_contexts_restored(rig)


def test_cpu_inference_never_requests_cuda_autocast(rig):
    rig.backend.device = "cpu"
    assert segment(rig) == []
    assert [name for name, scopes in rig.callbacks] == ["image", "text"]
    assert rig.torch.autocast_calls == []
    assert_contexts_restored(rig, cuda=False)


def test_cuda_index_uses_cuda_autocast_device_type(rig):
    rig.backend.device = "cuda:1"
    assert segment(rig) == []
    assert rig.torch.autocast_calls == [("cuda", rig.torch.bfloat16)]
    assert_contexts_restored(rig)


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, OSError])
def test_ordinary_forward_errors_are_not_misclassified_as_oom(rig, error_type):
    original = error_type("mat1 and mat2 must have the same dtype")
    rig.failure.update(stage="text", exception=original)
    with pytest.raises(backends.BackendError) as caught:
        segment(rig)
    assert not isinstance(caught.value, backends.ResourceLimit)
    assert caught.value.__cause__ is original
    assert_contexts_restored(rig)


@pytest.mark.parametrize("error_type", [FakeOOM, DerivedOOM])
def test_only_oom_type_and_its_subclass_become_resource_limit(rig, error_type):
    original = error_type("fixture allocation failed")
    rig.failure.update(stage="image", exception=original)
    with pytest.raises(backends.ResourceLimit) as caught:
        segment(rig)
    assert caught.value.__cause__ is original
    assert [name for name, scopes in rig.callbacks] == ["image"]
    assert_contexts_restored(rig)


@pytest.mark.parametrize("invalid_oom_type", [None, (RuntimeError,)])
def test_absent_or_nonclass_oom_type_cannot_classify_runtime_error(rig, invalid_oom_type):
    if invalid_oom_type is None:
        del rig.torch.cuda.OutOfMemoryError
    else:
        rig.torch.cuda.OutOfMemoryError = invalid_oom_type
    original = RuntimeError("CUDA out of memory is only text here")
    rig.failure.update(stage="text", exception=original)
    with pytest.raises(backends.BackendError) as caught:
        segment(rig)
    assert caught.value.__cause__ is original
    assert_contexts_restored(rig)


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "cpu"])
def test_effective_options_bind_actual_inference_policy(rig, device):
    rig.backend.device = device
    options = rig.backend.effective_options()
    cuda = device.split(":", 1)[0] == "cuda"
    assert options["inference_mode"] is True
    assert options["autocast_enabled"] is cuda
    assert options["autocast_device_type"] == ("cuda" if cuda else None)
    assert options["autocast_dtype"] == ("bfloat16" if cuda else None)
    assert options["autocast_scope"] == "set_image+set_text_prompt"


def test_options_digest_changes_with_cuda_cpu_precision_policy(rig):
    cuda_options = rig.backend.effective_options()
    rig.backend.device = "cpu"
    cpu_options = rig.backend.effective_options()
    assert binding.effective_options_digest(cuda_options) != binding.effective_options_digest(cpu_options)
    # The precision fields themselves are bound, independently of the device name.
    changed = dict(cuda_options, autocast_enabled=False)
    assert binding.effective_options_digest(cuda_options) != binding.effective_options_digest(changed)


@pytest.mark.parametrize("error_type,kind,rc", [
    (backends.ResourceLimit, "RESOURCE_LIMIT", 5),
    (WorkerMemoryStop, "RESOURCE_LIMIT", 5),
    (backends.BackendError, "BACKEND_ERROR", 4),
    (RuntimeError, "BACKEND_ERROR", 4),
    (ValueError, "BACKEND_ERROR", 4),
    (OSError, "BACKEND_ERROR", 4),
])
def test_worker_preserves_first_frame_failure_and_parent_stop_latch(tmp_path, monkeypatch, error_type, kind, rc):
    original = error_type("fixture forward failure mentions ResourceLimit without changing its type")
    calls = {"resolve": 0, "load": 0, "indices": [], "close": 0}

    class Backend:
        def resolve_weights(self):
            calls["resolve"] += 1

        def load(self):
            calls["load"] += 1

        def segment(self, source, prompt, object_id, width, height, index):
            calls["indices"].append(index)
            raise original

        def close(self):
            calls["close"] += 1

    monkeypatch.setattr(runner, "_build_backend", lambda request: Backend())
    event_path = tmp_path / "events.jsonl"
    job_path = tmp_path / "job.json"
    worker.write_job(job_path, {
        "events_path": str(event_path), "run_dir": str(tmp_path), "monitor_device": None,
        "request": {"io_run": str(tmp_path), "indices": [17, 34, 59],
                    "object_id": "skull-01", "text_prompt": "skull", "output_root": str(tmp_path),
                    "repo_dir": str(tmp_path), "device": "cpu"},
        "frames": [{"input_index": index, "source_image": str(tmp_path / f"{index}.png"),
                    "width": 2, "height": 3} for index in (17, 34, 59)],
    })
    assert worker.child_main(job_path) == rc
    assert calls == {"resolve": 1, "load": 1, "indices": [17], "close": 1}
    events, parse_errors = worker.read_events_strict(event_path, allow_trailing_partial=False)
    assert parse_errors == []
    begins = [event["input_index"] for event in events if event["type"] == worker.EVENT_FRAME_BEGIN]
    assert begins == [17]
    frames = [event for event in events if event["type"] == worker.EVENT_FRAME]
    assert len(frames) == 1 and frames[0]["input_index"] == 17
    assert frames[0]["status"] == kind
    assert frames[0]["errors"][0]["code"] == kind
    stops = [event for event in events if event["type"] == worker.EVENT_STOP]
    assert len(stops) == 1 and stops[0]["kind"] == kind and stops[0]["input_index"] == 17
    assert str(original) in stops[0]["reason"]
    assert events[-1]["type"] == worker.EVENT_DONE and events[-1]["stopped"] is True

    stop = deadline.StopState()
    frame_records, errors, finalisation_errors = {}, [], []
    ledger = runner._EventLedger(stop, [], frame_records, errors, finalisation_errors,
                                 deadline.Deadline(clock=lambda: 0.0))
    assert ledger.apply_all(events) == len(events)
    assert stop.kind == kind and stop.source == "worker"
    assert stop.latched is True
    assert sorted(frame_records) == [17] and frame_records[17]["status"] == kind
    assert ledger.state["done"]["stopped"] is True
    assert errors == [] and finalisation_errors == []
    assert stop.latch("BACKEND_ERROR" if kind == "RESOURCE_LIMIT" else "RESOURCE_LIMIT",
                      "later observation cannot replace original", "test") is False
    assert stop.kind == kind
