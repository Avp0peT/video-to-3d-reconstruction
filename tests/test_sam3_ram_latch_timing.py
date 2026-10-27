"""Pure-data RAM adjudication tests; run via the bounded guarded RUN entry.

No OS ownership, RAM, driver, model or native-library call is permitted.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
# Windows ctypes initializes its standard-library scalar/structure types here;
# native library loads and API lookup are forbidden after this bootstrap.
import ctypes
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace


_BLOCKED_IMPORTS = {"torch", "torchvision", "sam3", "triton", "numpy", "PIL", "cv2"}
_BLOCKED_EVENTS = {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.exec",
                   "os.fork", "ctypes.dlopen", "ctypes.dlsym"}


class _NoHeavyImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _BLOCKED_IMPORTS:
            raise AssertionError(f"forbidden heavy import: {fullname}")
        return None


def _no_external_calls(event, args):
    if event in _BLOCKED_EVENTS:
        raise AssertionError(f"forbidden real process/native call: {event}")


assert not (_BLOCKED_IMPORTS & {name.split(".")[0] for name in sys.modules})
sys.meta_path.insert(0, _NoHeavyImport())
sys.addaudithook(_no_external_calls)

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/delegation/SAM3_RAM_LATCH/20261008_main"
sys.path.insert(0, str(ROOT / "integrations/sam3"))
from sam3_adapter import deadline, observer, resources  # noqa: E402

LIMIT = 12 << 30


class Owner:
    def __init__(self, *, alive=True, identity=True):
        self.members_registry = {
            991: SimpleNamespace(identity_still_matches=lambda: identity, alive=lambda: alive),
        }

    def observation_targets(self):
        return [{"pid": 991, "role": "worker_interpreter", "is_actual_worker": True,
                 "identified_by": "TEST_DATA"}]

    def members(self):
        return {"top_level_exited": False, "enumeration_available": True, "unresolved_members": []}


def good_device():
    return {"device_index": 0, "device_used_bytes": 0, "device_total_bytes": 8 << 30,
            "fresh": True, "age_seconds": 0.1, "error": None,
            "query_started_monotonic": 20.0, "query_finished_monotonic": 20.1}


def make_observer(monkeypatch, *, module=observer, commit=LIMIT + 1, parent_commit=100,
                  cpu=False, alive=True, identity=True, sampler=None):
    monkeypatch.setattr(module, "read_process_memory", lambda: {
        "commit_bytes": parent_commit, "rss_bytes": 20})
    monkeypatch.setattr(resources, "read_process_memory_for_pid", lambda pid: {
        "pid": pid, "commit_bytes": commit, "rss_bytes": 10, "commit_channel": "TEST_DATA"})
    policy = {"device_channel_applies": not cpu, "required": not cpu,
              "identity_provable": True, "physical_index": 0, "freshness_seconds": 5.0}
    obj = module.WorkerObserver(Owner(alive=alive, identity=identity), 0.25, [], [],
        deadline.StopState(), policy=resources.ResourcePolicy(),
        device_sampler=sampler, device_policy=policy)
    return obj


def test_ram_is_latched_while_device_still_blocked_and_device_failure_is_kept(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls, failures = [], []
    obj = make_observer(monkeypatch)

    def fake_device():
        calls.append("device")
        assert obj.stop_state.latched
        assert obj.stop_state.source == "observer"
        assert "RAM commit" in obj.stop_state.reason
        entered.set()
        assert release.wait(2.0), "bounded fake-device release was not delivered"
        return {"error": "TEST_DATA driver timeout", "device_used_bytes": None}

    def run():
        try:
            obj.run()
        except BaseException as exc:
            failures.append(exc)

    obj.device_sampler = fake_device
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert entered.wait(2.0)
        assert thread.is_alive() and obj.stop_state.latched
        assert obj.samples == []  # full observation is still waiting on device
        reason = obj.stop_state.reason
    finally:
        release.set()
        thread.join(2.0)
    assert not thread.is_alive() and not failures
    assert calls == ["device"] and obj.observations == len(obj.samples) == 1
    assert obj.stop_state.reason == reason and obj.stop_state.source == "observer"
    assert len(obj.stop_state.history) == 1
    assert any(e.get("channel") == "device" and "TEST_DATA driver timeout" in e["message"]
               for e in obj.errors)
    sample = obj.samples[0]
    assert sample["ram_phase_interval"]["finished_monotonic"] <= sample["ram_stop_decision_monotonic"]
    assert sample["ram_stop_decision_monotonic"] <= sample["device_sample_interval"]["started_monotonic"]


def test_frozen_pilot06_defect_only_latches_after_device_returns(monkeypatch):
    name = "sam3_adapter._ram_latch_frozen"
    spec = importlib.util.spec_from_file_location(name, RUN / "before/observer.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    obj = make_observer(monkeypatch, module=module)
    seen = []

    def fake_device():
        seen.append(obj.stop_state.latched)
        return good_device()

    obj.device_sampler = fake_device
    obj.run()
    assert seen == [False] and obj.stop_state.latched
    assert "RAM commit" in obj.stop_state.reason


@pytest.mark.parametrize("commit", [0, LIMIT - 1, LIMIT])
def test_ram_at_or_below_limit_still_checks_device_failure(monkeypatch, commit):
    obj = make_observer(monkeypatch, commit=commit)
    calls = []

    def fake_device():
        assert not obj.stop_state.latched
        calls.append(1)
        raise OSError("TEST_DATA device unavailable")

    obj.device_sampler = fake_device
    obj.run()
    assert calls == [1] and len(obj.samples) == 1
    assert obj.stop_state.kind == "RESOURCE_LIMIT" and obj.stop_state.source == "observer"
    assert "device query failed" in obj.stop_state.reason
    assert "TEST_DATA device unavailable" in obj.stop_state.reason
    assert "ram_stop_decision_monotonic" not in obj.samples[0]


@pytest.mark.parametrize("commit,parent_commit,alive,identity,reason", [
    (LIMIT + 1, 0, True, True, "RAM commit"),
    (0, LIMIT + 1, True, True, "parent RAM commit"),
    (None, 0, True, True, "unmeasurable"),
    (True, 0, True, True, "non-boolean integer"),
    (0, None, True, True, "parent commit_bytes"),
    (0, 0, True, False, "UNKNOWN"),
])
def test_all_ram_failures_use_shared_early_rule(monkeypatch, commit, parent_commit, alive, identity, reason):
    obj = make_observer(monkeypatch, commit=commit, parent_commit=parent_commit,
                        alive=alive, identity=identity)
    calls = []

    def fake_device():
        assert obj.stop_state.latched and reason in obj.stop_state.reason
        calls.append(1)
        return good_device()

    obj.device_sampler = fake_device
    obj.run()
    assert calls == [1] and len(obj.stop_state.history) == 1
    assert obj.stop_state.reason == obj.ram_violations(obj.samples[0])[0]
    assert obj.stop_state.reason == obj.violations(obj.samples[0])[0]


def test_prior_stop_cause_is_preserved(monkeypatch):
    obj = make_observer(monkeypatch)
    obj.stop_state.latch("BACKEND_ERROR", "TEST_DATA earlier cause", "worker")
    obj.device_sampler = lambda: {"error": "TEST_DATA later device error"}
    obj.run()
    assert obj.stop_state.reason == "TEST_DATA earlier cause" and obj.stop_state.source == "worker"
    assert len(obj.stop_state.history) == 2
    assert any(e.get("channel") == "ram" for e in obj.errors)
    assert any(e.get("channel") == "device" for e in obj.errors)


@pytest.mark.parametrize("cpu,alive", [(False, True), (True, True), (True, False)])
def test_intervals_bound_each_call_and_cpu_has_no_device_interval(monkeypatch, cpu, alive):
    clock = SimpleNamespace(now=10.0)
    spans = {}

    def tick():
        clock.now += 0.1
        return clock.now

    obj = make_observer(monkeypatch, commit=LIMIT, cpu=cpu, alive=alive)
    monkeypatch.setattr(observer, "time", SimpleNamespace(monotonic=tick,
                        strftime=time.strftime, gmtime=time.gmtime))

    def parent_read():
        spans["parent"] = (tick(), tick())
        return {"commit_bytes": 0, "rss_bytes": 0}

    def target_read(pid):
        spans["target"] = (tick(), tick())
        return {"commit_bytes": LIMIT, "rss_bytes": 1}

    def device():
        spans["device"] = (tick(), tick())
        obj._stop_event.set()
        result = good_device()
        result.update(query_started_monotonic=spans["device"][0],
                      query_finished_monotonic=spans["device"][1])
        return result

    monkeypatch.setattr(observer, "read_process_memory", parent_read)
    monkeypatch.setattr(resources, "read_process_memory_for_pid", target_read)
    obj.device_sampler = None if cpu else device
    waits = []
    if cpu:
        obj._stop_event = SimpleNamespace(is_set=lambda: bool(waits),
                                         wait=lambda seconds: waits.append(seconds))
    obj.run()
    assert not obj.stop_state.latched and obj.observations == len(obj.samples) == 1
    sample = obj.samples[0]
    for key, interval in (("parent", sample["parent_ram_sample_interval"]),
                          ("target", sample["targets"][0]["ram_sample_interval"]),
                          ("device", sample["device_sample_interval"])):
        if key not in spans:
            assert interval is None
        else:
            assert interval["started_monotonic"] < spans[key][0] < spans[key][1] < interval["finished_monotonic"]
    ram = sample["ram_phase_interval"]
    assert ram["started_monotonic"] <= sample["parent_ram_sample_interval"]["started_monotonic"]
    assert sample["parent_ram_sample_interval"]["finished_monotonic"] <= ram["finished_monotonic"]
    if alive:
        assert sample["targets"][0]["ram_sample_interval"]["finished_monotonic"] <= ram["finished_monotonic"]
    if cpu:
        assert sample["device"] is None and waits == [0.25]
    else:
        assert ram["finished_monotonic"] <= sample["device_sample_interval"]["started_monotonic"]
        assert sample["device_sample_interval"]["finished_monotonic"] <= sample["t"] + 0.000001
        assert sample["device_sample_interval"]["started_monotonic"] <= sample["device"]["query_started_monotonic"]
        assert sample["device"]["query_finished_monotonic"] <= sample["device_sample_interval"]["finished_monotonic"]
    assert "observation completion" in sample["sampling_time_semantics"]


def test_direct_observe_remains_observation_only(monkeypatch):
    obj = make_observer(monkeypatch, sampler=good_device)
    sample = obj.observe()
    assert not obj.stop_state.latched and not obj.samples and not obj.errors
    assert obj.ram_violations(sample)
