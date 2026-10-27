"""Offline TEST_DATA checks; no torch import and no model execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import pytest


spec = importlib.util.spec_from_file_location("video_mask_worker", Path(__file__).parents[1] / "integrations/video_masks/worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_proxy_mmap_local_and_restored(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"TEST_DATA")
    calls = []
    original = SimpleNamespace(load=lambda *args, **kwargs: calls.append((args, kwargs)))
    builder = SimpleNamespace(torch=original)
    with worker.builder_mmap(builder, checkpoint) as audit:
        with checkpoint.open("rb") as stream:
            builder.torch.load(stream, map_location="cpu", weights_only=True)
        assert original.load is not builder.torch.load
    assert builder.torch is original
    assert audit == {"mmap_load_count": 1, "builder_torch_restored": True}
    assert calls == [((str(checkpoint),), {"map_location": "cpu", "weights_only": True, "mmap": True})]


def test_proxy_restored_after_failure(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"TEST_DATA")
    original = SimpleNamespace(load=lambda *a, **k: None)
    builder = SimpleNamespace(torch=original)
    with pytest.raises(RuntimeError, match="unbound"):
        with worker.builder_mmap(builder, checkpoint) as audit:
            builder.torch.load("wrong", map_location="cpu", weights_only=True)
    assert builder.torch is original
    assert audit["builder_torch_restored"]


def config(tmp_path):
    repo = tmp_path / "repo"
    (repo / "sam3").mkdir(parents=True)
    (repo / "sam3/model_builder.py").write_text("# TEST_DATA")
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "000000.jpg").write_bytes(b"TEST_DATA")
    seed = tmp_path / "seed.png"
    seed.write_bytes(b"TEST_DATA")
    return dict(repo_root=str(repo), checkpoint_path=str(seed), bpe_path=str(seed),
                frames_dir=str(frames), output_dir=str(tmp_path / "out"), frame_count=1,
                max_frames=1, direction="forward", seeds=[dict(frame_index=0,
                mask_path=str(seed), mask_sha256=worker.sha256(seed))])


def test_config_bounds_and_seed_hash(tmp_path):
    raw = config(tmp_path)
    assert worker.validate_config(raw)["threads"] == 2
    for mutation in ({"max_frames": 97}, {"frame_count": True}, {"threads": 4}, {"seeds": []}):
        with pytest.raises(ValueError):
            worker.validate_config({**raw, **mutation})
    raw["seeds"][0]["mask_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        worker.validate_config(raw)


def test_failed_load_events_and_no_commit(tmp_path, monkeypatch):
    raw = config(tmp_path)
    def fail(name):
        raise RuntimeError("TEST_DATA load failure")
    monkeypatch.setattr(worker.importlib, "import_module", fail)
    assert worker.run(raw) == 1
    events = [json.loads(line) for line in (tmp_path / "out/events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["ready", "load_begin", "fatal", "done"]
    assert json.loads((tmp_path / "out/result.json").read_text())["status"] == "FAILED"


def test_commit_readback(tmp_path):
    import numpy as np
    from PIL import Image
    target = tmp_path / "mask.png"
    metadata = worker.commit_mask(np.array([[True, False], [False, True]]), target)
    assert metadata["foreground_pixels"] == 2
    assert metadata["mask_sha256"] == worker.sha256(target)
    with Image.open(target) as image:
        assert list(image.getdata()) == [255, 0, 0, 255]
    assert not target.with_suffix(".png.partial").exists()


def test_native_call_contract_source():
    source = Path(worker.__file__).read_text()
    for required in ("strict_state_dict_loading=True", "load_from_HF=False, compile=False",
                     "tracker.backbone = model.detector.backbone", "offload_video_to_cpu=True",
                     "offload_state_to_cpu=True, async_loading_frames=False", "torch.inference_mode()",
                     'dtype=torch.bfloat16', 'emit("frame_commit", **item)'):
        assert required in source
    assert source.index('emit("frame_begin", frame_id=expected)') < source.index("next(iterator)")
    assert source.index("metadata = commit_mask") < source.index('emit("frame_commit", **item)')


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, 2, 255])
def test_commit_rejects_invalid_without_cast(tmp_path, invalid):
    import numpy as np
    with pytest.raises(ValueError, match="finite boolean or 0/1"):
        worker.commit_mask(np.array([[invalid]]), tmp_path / "bad.png")
    assert not list(tmp_path.iterdir())


def test_seed_strict_mode_and_labels(tmp_path):
    from PIL import Image
    path = tmp_path / "seed.png"
    Image.new("RGB", (2, 2)).save(path)
    with pytest.raises(ValueError, match="mode L"):
        worker.read_seed(path)
    Image.new("L", (2, 2), 1).save(path)
    with pytest.raises(ValueError, match="strictly 0 or 255"):
        worker.read_seed(path)


def test_both_one_load_independent_states_and_protocol(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    raw = config(tmp_path)
    (Path(raw["frames_dir"]) / "000001.jpg").write_bytes(b"TEST_DATA")
    Image.new("L", (2, 2), 255).save(raw["seeds"][0]["mask_path"])
    raw["seeds"][0]["mask_sha256"] = worker.sha256(raw["seeds"][0]["mask_path"])
    raw.update(direction="both", frame_count=2, max_frames=2)
    raw["seeds"].append(dict(raw["seeds"][0], frame_index=1))
    calls = {"loads": 0, "states": [], "directions": [], "eval": 0}

    class Tensor:
        shape = (1, 1, 2, 2)
        def __getitem__(self, key):
            return self
        def detach(self):
            return self
        def float(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return np.ones((2, 2))

    class Tracker:
        def eval(self):
            return self
        def init_state(self, **kwargs):
            assert kwargs["offload_video_to_cpu"] and kwargs["offload_state_to_cpu"]
            assert kwargs["async_loading_frames"] is False
            state = dict(num_frames=2, video_height=2, video_width=2)
            calls["states"].append(state)
            return state
        def add_new_mask(self, state, **kwargs):
            state["seed"] = kwargs["frame_idx"]
        def propagate_in_video(self, state, **kwargs):
            calls["directions"].append(kwargs["reverse"])
            assert kwargs["max_frame_num_to_track"] == 1
            order = [1, 0] if kwargs["reverse"] else [0, 1]
            assert state["seed"] == order[0]
            for index in order:
                yield index, [1], None, Tensor(), None

    class Model:
        training = True
        tracker = Tracker()
        detector = SimpleNamespace(backbone=object())
        def eval(self):
            self.training = False
            calls["eval"] += 1
            return self
        def named_parameters(self):
            return [("test", SimpleNamespace(device="cuda:0", requires_grad=False))]
        def named_modules(self):
            return [("", self)]

    torch = SimpleNamespace(set_num_threads=lambda count: None, load=lambda *a, **k: {},
                            inference_mode=nullcontext, autocast=lambda **kwargs: nullcontext(),
                            bfloat16="TEST_DATA", from_numpy=lambda array: array)
    builder = SimpleNamespace(torch=torch, __file__=str(Path(raw["repo_root"]) / "sam3/model_builder.py"))
    def build(**kwargs):
        calls["loads"] += 1
        assert kwargs["strict_state_dict_loading"] is True
        assert kwargs["load_from_HF"] is False and kwargs["compile"] is False
        with open(kwargs["checkpoint_path"], "rb") as stream:
            builder.torch.load(stream, map_location="cpu", weights_only=True)
        return Model()
    builder.build_sam3_video_model = build
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: {"torch": torch, "sam3.model_builder": builder}[name])
    assert worker.run(raw) == 0
    assert calls["loads"] == 1 and calls["eval"] == 1
    assert len(calls["states"]) == 2 and calls["states"][0] is not calls["states"][1]
    assert calls["directions"] == [False, True]
    output = Path(raw["output_dir"])
    result = json.loads((output / "result.json").read_text())
    assert len(result["frames"]) == 4
    assert set(result["directions"]) == {"forward", "reverse"}
    for direction in ("forward", "reverse"):
        child = json.loads((output / direction / "result.json").read_text())
        assert len(child["frames"]) == 2
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [event["seq"] for event in events] == list(range(len(events)))
    assert all(event["pid"] > 0 and event["monotonic_ns"] > 0 for event in events)
    assert sum(e["event"] == "load_begin" for e in events) == 1
    assert sum(e["event"] == "frame_commit" for e in events) == 4


def test_partial_commits_survive_failure(tmp_path, monkeypatch):
    raw = config(tmp_path)
    monkeypatch.setattr(worker, "model_audit", lambda *a: {})
    torch = SimpleNamespace(set_num_threads=lambda count: None, load=lambda *a, **k: {})
    builder = SimpleNamespace(torch=torch, __file__=str(Path(raw["repo_root"]) / "sam3/model_builder.py"))
    def build(**kwargs):
        with open(kwargs["checkpoint_path"], "rb") as stream:
            builder.torch.load(stream, map_location="cpu", weights_only=True)
        return object()
    builder.build_sam3_video_model = build
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: {"torch": torch, "sam3.model_builder": builder}[name])
    def partial(cfg, torch, model, emit):
        emit("frame_commit", frame_id=0, status="propagated", mask_sha256="TEST_DATA")
        raise RuntimeError("TEST_DATA after commit")
    monkeypatch.setattr(worker, "propagate", partial)
    assert worker.run(raw) == 1
    result = json.loads((Path(raw["output_dir"]) / "result.json").read_text())
    assert result["frames"] == [dict(frame_id=0, status="propagated", mask_sha256="TEST_DATA", direction="forward")]
