"""Bounded offline SAM3 mask propagation. Importing this module never imports torch."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_config(raw):
    cfg = dict(raw)
    for name in ("repo_root", "checkpoint_path", "bpe_path", "frames_dir", "output_dir"):
        cfg[name] = str(Path(cfg[name]).resolve(strict=name != "output_dir"))
    for name in ("frame_count", "max_frames"):
        if type(cfg.get(name)) is not int or not 1 <= cfg[name] <= 96:
            raise ValueError(f"{name} must be an integer in [1,96]")
    if cfg["frame_count"] > cfg["max_frames"]:
        raise ValueError("frame_count exceeds max_frames")
    if cfg.get("threads", 2) != 2:
        raise ValueError("threads must be 2")
    cfg["threads"] = 2
    cfg.setdefault("device", "cuda:0")
    if not re.fullmatch(r"cuda:\d+", cfg["device"]):
        raise ValueError("device must be explicit cuda:N")
    if cfg.get("direction") not in ("forward", "reverse", "both"):
        raise ValueError("direction must be forward, reverse or both")
    seeds = cfg.get("seeds")
    count = 2 if cfg["direction"] == "both" else 1
    if not isinstance(seeds, list) or len(seeds) != count:
        raise ValueError(f"direction {cfg['direction']} requires exactly {count} seeds")
    cfg["seeds"] = []
    for value in seeds:
        seed = dict(value)
        if type(seed.get("frame_index")) is not int or not 0 <= seed["frame_index"] < cfg["frame_count"]:
            raise ValueError("seed frame_index outside local frame range")
        seed["mask_path"] = str(Path(seed["mask_path"]).resolve(strict=True))
        if sha256(seed["mask_path"]) != seed.get("mask_sha256"):
            raise ValueError("seed mask hash mismatch")
        cfg["seeds"].append(seed)
    if cfg["direction"] == "both" and (cfg["frame_count"] < 2 or
            sorted(s["frame_index"] for s in cfg["seeds"]) != [0, cfg["frame_count"] - 1]):
        raise ValueError("both requires distinct endpoint seeds at 0 and frame_count-1")
    frames = [p for p in Path(cfg["frames_dir"]).iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    if any(not p.stem.isdecimal() for p in frames):
        raise ValueError("JPEG filenames must have integer local indexes")
    if sorted(int(p.stem) for p in frames) != list(range(cfg["frame_count"])):
        raise ValueError("JPEG frame indexes must be unique contiguous 0..frame_count-1")
    if not (Path(cfg["repo_root"]) / "sam3" / "model_builder.py").is_file():
        raise ValueError("repo_root must be the local SAM3 checkout")
    return cfg


@contextmanager
def builder_mmap(builder, checkpoint_path):
    """Replace only the builder's torch binding; preserve strict loader behavior."""
    original = builder.torch
    audit = {"mmap_load_count": 0, "builder_torch_restored": False}
    bound = Path(checkpoint_path).resolve(strict=True)

    class Proxy:
        def __getattr__(self, name):
            return getattr(original, name)

        def load(self, source, *args, **kwargs):
            source_name = getattr(source, "name", None)
            if args or kwargs != {"map_location": "cpu", "weights_only": True}:
                raise RuntimeError("unexpected builder torch.load arguments")
            if source_name is None or Path(source_name).resolve() != bound:
                raise RuntimeError("builder attempted an unbound checkpoint load")
            if audit["mmap_load_count"]:
                raise RuntimeError("checkpoint load repeated")
            audit["mmap_load_count"] += 1
            return original.load(str(bound), map_location="cpu", weights_only=True, mmap=True)

    builder.torch = Proxy()
    try:
        yield audit
        if audit["mmap_load_count"] != 1:
            raise RuntimeError("expected exactly one checkpoint mmap load")
    finally:
        builder.torch = original
        audit["builder_torch_restored"] = builder.torch is original


def commit_mask(mask, target):
    """Publish only after lossless PNG write and full pixel readback."""
    import numpy as np
    from PIL import Image
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("expected a 2D mask")
    if values.dtype.kind not in "biuf" or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("mask must contain only finite boolean or 0/1 values")
    pixels = values.astype(np.uint8) * 255
    temporary = target.with_suffix(".png.partial")
    with temporary.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())
    with Image.open(temporary) as image:
        readback = np.asarray(image).copy()
    if not np.array_equal(pixels, readback):
        raise RuntimeError("mask readback mismatch")
    os.replace(temporary, target)
    with Image.open(target) as image:
        if not np.array_equal(pixels, np.asarray(image)):
            raise RuntimeError("published mask readback mismatch")
    return {"mask_path": str(target), "mask_sha256": sha256(target),
            "width": pixels.shape[1], "height": pixels.shape[0],
            "foreground_pixels": int((pixels != 0).sum())}


def read_seed(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError("seed mask must be mode L")
        labels = np.asarray(image).copy()
    if not np.isin(labels, [0, 255]).all():
        raise ValueError("seed mask labels must be strictly 0 or 255")
    return labels == 255


def model_audit(model, expected_device):
    """Audit every named parameter and every module after explicit eval()."""
    model.eval()
    parameters = [{"name": name, "device": str(parameter.device),
                   "requires_grad": bool(parameter.requires_grad)}
                  for name, parameter in model.named_parameters()]
    training_modules = [name for name, module in model.named_modules() if module.training]
    audit = {"parameters": parameters, "parameter_count": len(parameters),
             "training": bool(model.training), "training_modules": training_modules}
    if not parameters or any(p["device"] != expected_device for p in parameters) or training_modules or model.training:
        raise RuntimeError("model device/eval audit failed: " + json.dumps(audit))
    return audit


def propagate(cfg, torch, model, emit):
    import numpy as np
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    tracker.eval()
    seed = cfg["seeds"][0]
    start = seed["frame_index"]
    order = list(range(start, cfg["frame_count"])) if cfg["direction"] == "forward" else list(range(start, -1, -1))
    commits = []
    state = None
    iterator = None
    try:
      with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = tracker.init_state(video_path=cfg["frames_dir"], offload_video_to_cpu=True,
                                   offload_state_to_cpu=True, async_loading_frames=False)
        if state["num_frames"] != cfg["frame_count"]:
            raise RuntimeError("tracker frame_count mismatch")
        mask = read_seed(seed["mask_path"])
        if mask.shape != (state["video_height"], state["video_width"]):
            raise ValueError("seed mask dimensions differ from video")
        tracker.add_new_mask(state, frame_idx=start, obj_id=1, mask=torch.from_numpy(mask))
        iterator = iter(tracker.propagate_in_video(state, start_frame_idx=start,
                            max_frame_num_to_track=len(order) - 1, reverse=cfg["direction"] == "reverse",
                            propagate_preflight=True, tqdm_disable=True))
        for expected in order:
            emit("frame_begin", frame_id=expected)
            frame_id, obj_ids, _, masks, _ = next(iterator)
            if frame_id != expected or list(obj_ids) != [1] or tuple(masks.shape[:2]) != (1, 1):
                raise RuntimeError("tracker frame/object output mismatch")
            logits = masks[0, 0].detach().float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError("tracker mask logits contain nonfinite values")
            metadata = commit_mask(logits > 0,
                                   Path(cfg["output_dir"]) / f"{frame_id:06d}.png")
            item = {"frame_id": frame_id, "status": "propagated", **metadata}
            commits.append(item)
            emit("frame_commit", **item)
        if next(iterator, None) is not None:
            raise RuntimeError("tracker exceeded bounded processing order")
    finally:
        if iterator is not None and hasattr(iterator, "close"):
            iterator.close()
        del iterator
        del state
    return commits


def write_result(path, result):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def propagate_directions(cfg, torch, model, emit):
    """Sequential independent tracker states sharing exactly one loaded model."""
    directions = ("forward", "reverse") if cfg["direction"] == "both" else (cfg["direction"],)
    results = {}
    for direction in directions:
        child = dict(cfg, direction=direction)
        if cfg["direction"] == "both":
            child["seeds"] = [next(s for s in cfg["seeds"] if s["frame_index"] ==
                                    (0 if direction == "forward" else cfg["frame_count"] - 1))]
            directory = Path(cfg["output_dir"]) / direction
            directory.mkdir(exist_ok=False)
            child["output_dir"] = str(directory)
        committed = []
        def directional_emit(event, **values):
            emit(event, direction=direction, **values)
            if event == "frame_commit":
                committed.append(dict(values, direction=direction))
        try:
            propagate(child, torch, model, directional_emit)
            result = {"status": "PROPAGATED_NOT_SEMANTICALLY_EVALUATED",
                      "direction": direction, "frame_count": cfg["frame_count"], "frames": committed}
        except Exception as exc:
            result = {"status": "FAILED", "direction": direction, "frame_count": cfg["frame_count"],
                      "frames": committed, "error": str(exc)}
            if cfg["direction"] == "both":
                write_result(Path(child["output_dir"]) / "result.json", result)
            raise
        results[direction] = result
        if cfg["direction"] == "both":
            write_result(Path(child["output_dir"]) / "result.json", result)
    return results


def run(raw):
    cfg = validate_config(raw)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    audit = {}
    commits = []
    runtime_audit = {}
    with (output / "events.jsonl").open("x", encoding="utf-8") as events:
        sequence = 0
        def emit(event, **values):
            nonlocal sequence
            record = {"event": event, "time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                      "pid": os.getpid(), "seq": sequence, **values}
            events.write(json.dumps(record) + "\n")
            events.flush()
            os.fsync(events.fileno())
            sequence += 1
            if event == "frame_commit":
                commits.append(dict(values))
        try:
            emit("ready", frame_count=cfg["frame_count"], direction=cfg["direction"])
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                os.environ[key] = "1"
            os.environ["OMP_NUM_THREADS"] = "2"
            os.environ["MKL_NUM_THREADS"] = "2"
            sys.path.insert(0, cfg["repo_root"])
            emit("load_begin")
            torch = importlib.import_module("torch")
            torch.set_num_threads(2)
            builder = importlib.import_module("sam3.model_builder")
            if Path(builder.__file__).resolve() != Path(cfg["repo_root"]) / "sam3" / "model_builder.py":
                raise RuntimeError("SAM3 builder import provenance mismatch")
            with builder_mmap(builder, cfg["checkpoint_path"]) as audit:
                model = builder.build_sam3_video_model(checkpoint_path=cfg["checkpoint_path"],
                    bpe_path=cfg["bpe_path"], load_from_HF=False, compile=False,
                    strict_state_dict_loading=True, device=cfg["device"])
            runtime_audit = model_audit(model, cfg["device"])
            emit("load_end", **audit, strict_state_dict_loading=True, model_audit=runtime_audit)
            directions = propagate_directions(cfg, torch, model, emit)
            result = {"status": "PROPAGATED_NOT_SEMANTICALLY_EVALUATED", "frame_count": cfg["frame_count"],
                      "frames": commits, "directions": directions, "load_audit": audit, "model_audit": runtime_audit}
            code = 0
        except Exception as exc:
            emit("fatal", error_type=type(exc).__name__, error=str(exc), load_audit=audit)
            result = {"status": "FAILED", "frame_count": cfg["frame_count"], "frames": commits,
                      "error": str(exc), "load_audit": audit, "model_audit": runtime_audit}
            code = 1
        write_result(output / "result.json", result)
        emit("done", status=result["status"], exit_code=code)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8-sig") as stream:
        return run(json.load(stream))


if __name__ == "__main__":
    raise SystemExit(main())
