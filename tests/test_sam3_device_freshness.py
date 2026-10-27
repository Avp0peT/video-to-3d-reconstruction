"""Offline device-query wiring/timing regression; no driver, process or heavy imports.

Run with --noconftest and the guarded entry under
outputs/delegation/SAM3_DEVICE_FRESHNESS/20261008_main/run_tests.py.
The frozen-source case records the old defect in the same final test batch.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations" / "sam3"))
from sam3_adapter import observer  # noqa: E402


def policy_for(monkeypatch, timeout=5.0):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return observer.resolve_device_policy(SimpleNamespace(
        device="cuda", device_observe="auto", device_observe_timeout=timeout))


def fake_driver(monkeypatch, module, *, elapsed=3.969, stdout="0, 8151\n", rc=0, error=None):
    """Replace module bindings, never shared time/subprocess module functions."""
    clock = SimpleNamespace(now=100.0)
    calls = []
    shared_clock, shared_run = time.monotonic, subprocess.run

    def run(argv, **kwargs):
        assert argv == ["nvidia-smi", "-i", "0", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"]
        assert kwargs == dict(capture_output=True, text=True, timeout=5.0, check=False,
                              encoding="utf-8", errors="replace")
        calls.append({"argv": argv, "timeout": kwargs["timeout"]})
        clock.now += elapsed
        if error == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if error == "os":
            raise OSError("offline missing command")
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr="offline failure" if rc else "")

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(
        run=run, SubprocessError=subprocess.SubprocessError))
    assert time.monotonic is shared_clock and subprocess.run is shared_run
    return calls


def test_frozen_old_source_reproduces_policy_five_query_default_two(monkeypatch):
    source = Path(os.environ["SAM3_FROZEN_OBSERVER"])
    expected_hash = os.environ["SAM3_FROZEN_OBSERVER_SHA256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
    name = "sam3_adapter._frozen_freshness_counterexample"
    spec = importlib.util.spec_from_file_location(name, source)
    old = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, old)
    spec.loader.exec_module(old)
    policy = policy_for(monkeypatch)
    calls = fake_driver(monkeypatch, old)
    result = old.make_device_sampler("driver", "cuda", timeout=5.0, device_policy=policy)()
    assert len(calls) == 1 and policy["freshness_seconds"] == 5.0
    assert result["age_seconds"] == 3.969 and result["fresh"] is False
    problems = old.device_violations(result, policy, 7 << 30)
    assert problems == ["device reading is stale (age 3.969s)"]
    evidence = Path(os.environ["SAM3_FRESHNESS_EVIDENCE_DIR"]) / "OLD_COUNTEREXAMPLE.json"
    evidence.write_text(json.dumps({"source_sha256": expected_hash, "policy": policy,
                                    "reading": result, "violations": problems,
                                    "execution": "fake clock and fake subprocess only"}, indent=2) + "\n",
                        encoding="utf-8")


@pytest.mark.parametrize("elapsed", [0.0, 1.999999, 2.001, 3.969, 4.9999996, 5.0, 5.00000001])
def test_declared_freshness_reaches_query_without_rounding(monkeypatch, elapsed):
    policy = policy_for(monkeypatch)
    calls = fake_driver(monkeypatch, observer, elapsed=elapsed)
    result = observer.make_device_sampler("driver", "cuda", 5.0, device_policy=policy)()
    assert len(calls) == 1
    assert result["query_rc"] == 0 and result["error"] is None
    assert result["freshness_seconds"] == result["query_timeout_seconds"] == 5.0
    assert result["query_started_monotonic"] == 100.0
    assert result["query_finished_monotonic"] == 100.0 + elapsed
    assert result["age_seconds"] == result["query_elapsed_seconds"]
    assert result["age_seconds"] == result["query_finished_monotonic"] - 100.0
    assert result["fresh"] is (elapsed <= 5.0)
    assert bool(observer.device_violations(result, policy, 7 << 30)) is (elapsed > 5.0)
    assert result["device_used_bytes"] == 0  # a valid zero is not a missing reading
    assert "not a device measurement timestamp" in result["age_semantics"]
    if elapsed > 5.0:
        assert result["age_seconds"] > 5.0, "rounding must not wash out a slight overrun"


@pytest.mark.parametrize("provided_policy", [False, True])
def test_sampler_without_complete_policy_explicitly_uses_timeout(monkeypatch, provided_policy):
    policy_for(monkeypatch)
    policy = observer.resolve_device_identity("cuda") if provided_policy else None
    fake_driver(monkeypatch, observer)
    result = observer.make_device_sampler("driver", "cuda", 5.0, device_policy=policy)()
    assert result["freshness_seconds"] == 5.0 and result["fresh"] is True


def test_explicit_policy_freshness_is_distinct_from_query_timeout(monkeypatch):
    policy = policy_for(monkeypatch)
    policy["freshness_seconds"] = 3.0
    fake_driver(monkeypatch, observer)
    result = observer.make_device_sampler("driver", "cuda", 5.0, device_policy=policy)()
    assert result["query_timeout_seconds"] == 5.0 and result["freshness_seconds"] == 3.0
    assert result["fresh"] is False and observer.device_violations(result, policy, 7 << 30)


@pytest.mark.parametrize("kwargs,expected_rc", [
    ({"error": "timeout", "elapsed": 5.25}, None),
    ({"error": "os"}, None),
    ({"rc": 7}, 7),
    ({"stdout": ""}, 0),
    ({"stdout": "N/A, 8151\n"}, 0),
    ({"stdout": "1, N/A\n"}, 0),
    ({"stdout": "1, 2, 3\n"}, 0),
])
def test_query_failures_keep_timing_rc_and_unknown_usage(monkeypatch, kwargs, expected_rc):
    policy = policy_for(monkeypatch)
    fake_driver(monkeypatch, observer, **kwargs)
    result = observer.make_device_sampler("driver", "cuda", 5.0, device_policy=policy)()
    assert result["query_rc"] == expected_rc and result["error"]
    assert result["fresh"] is False
    assert result["device_used_bytes"] is None and result["device_total_bytes"] is None
    assert result["query_started_monotonic"] == 100.0
    assert result["query_finished_monotonic"] > 100.0
    assert result["age_seconds"] == result["query_elapsed_seconds"] > 0
    assert result["freshness_seconds"] == result["query_timeout_seconds"] == 5.0
    assert observer.device_violations(result, policy, 7 << 30)[0].startswith("device query failed:")


@pytest.mark.parametrize("changes", [
    {"fresh": False}, {"fresh": None}, {"fresh": 1}, {"age_seconds": None},
    {"age_seconds": float("nan")}, {"age_seconds": float("inf")}, {"age_seconds": -1},
    {"age_seconds": True}, {"age_seconds": 5.00000001}, {"device_used_bytes": True},
    {"device_used_bytes": None}, {"device_used_bytes": -1}, {"device_used_bytes": 8 << 30},
    {"device_used_bytes": 2, "device_total_bytes": 1}, {"device_index": False},
    {"device_index": 3}, {"error": "query failed"},
])
def test_validator_retains_fail_closed_rules(monkeypatch, changes):
    policy = policy_for(monkeypatch)
    reading = dict(device_used_bytes=0, device_total_bytes=8 << 30, device_index=0,
                   fresh=True, age_seconds=0.01)
    assert observer.device_violations(reading, policy, 7 << 30) == []
    reading.update(changes)
    assert observer.device_violations(reading, policy, 7 << 30)


def test_cpu_sampler_does_not_query_and_limits_are_unchanged(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU must not query a device")
    monkeypatch.setattr(observer, "driver_device_query", forbidden)
    policy = observer.resolve_device_policy(SimpleNamespace(device="cpu"))
    assert not policy["device_channel_applies"]
    assert observer.make_device_sampler("none", "cpu", device_policy=policy) is None
    assert observer.device_violations(None, policy, 7 << 30) == []
    assert observer.RAM_COMMIT_LIMIT_BYTES == 12 << 30
    assert observer.VRAM_LIMIT_BYTES == 7 << 30
