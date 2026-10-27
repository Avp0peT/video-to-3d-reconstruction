"""Bounded local video -> timestamp-bound PNGs -> unchanged photo-io-v1."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from integrations.video_io.timeline import parse_probe, select_uniform, select_all, parse_showinfo, validate_export
from kozlov_lab.io.paths import is_valid_dataset_id

LIMITS = dict(input_bytes=512 * 1024**2, decoded_frames=20000, span_seconds=120,
              input_pixels=16_777_216, max_side=1024, process_seconds=120,
              cleanup_seconds=5, output_bytes=512 * 1024**2, log_bytes=32 * 1024**2,
              threads=2)

class LimitError(RuntimeError):
    pass

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")

def bytes_under(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0

def bounded_run(argv, run, name, deadline, dataset, env_overrides=None):
    """File-backed logs; shared deadline; terminate only the owned Popen handle."""
    out, err = run / (name + ".stdout.txt"), run / (name + ".stderr.txt")
    record_path = run / (name + ".process.json")
    record = dict(argv=[str(x) for x in argv], cwd=str(ROOT), started_utc=utc(),
                  env_overrides=env_overrides or {}, pid=None, returncode=None,
                  stdout=str(out), stderr=str(err), cleanup=None)
    save(record_path, record)
    started = time.monotonic()
    process = None
    try:
        if started >= deadline:
            raise LimitError("shared process deadline exhausted before " + name)
        environment = os.environ.copy()
        environment.update(env_overrides or {})
        with out.open("xb") as stdout, err.open("xb") as stderr:
            process = subprocess.Popen(record["argv"], cwd=ROOT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            record["pid"] = process.pid
            save(record_path, record)
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise LimitError("shared process deadline exceeded at " + name)
                if out.stat().st_size + err.stat().st_size > LIMITS["log_bytes"]:
                    raise LimitError("process logs exceeded bound at " + name)
                if bytes_under(run) + bytes_under(dataset) > LIMITS["output_bytes"]:
                    raise LimitError("output size exceeded bound at " + name)
                time.sleep(0.05)
            record["returncode"] = process.returncode
        if out.stat().st_size + err.stat().st_size > LIMITS["log_bytes"]:
            raise LimitError("process logs exceeded bound at exit: " + name)
        if bytes_under(run) + bytes_under(dataset) > LIMITS["output_bytes"]:
            raise LimitError("output size exceeded bound at exit: " + name)
        if process.returncode:
            raise RuntimeError(name + " exited " + str(process.returncode) + "; see " + str(err))
        return out, err
    except BaseException as error:
        record["error"] = type(error).__name__ + ": " + str(error)
        if process is not None and process.poll() is None:
            cleanup = dict(attempted=True, exit_confirmed=False, error=None)
            record["cleanup"] = cleanup
            cleanup_deadline = time.monotonic() + LIMITS["cleanup_seconds"]
            try:
                process.kill()
                process.wait(timeout=max(0, cleanup_deadline - time.monotonic()))
                cleanup["exit_confirmed"] = True
            except Exception as failure:
                cleanup["error"] = type(failure).__name__ + ": " + str(failure)
        raise
    finally:
        if process is not None:
            record["returncode"] = process.poll()
        record.update(ended_utc=utc(), wall_seconds=time.monotonic()-started)
        save(record_path, record)

def inspect_image(path):
    """Exact decoded RGB identity; Laplacian variance is a quality proxy only."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        if im.format != "PNG" or im.mode != "RGB" or max(im.size) > LIMITS["max_side"]:
            raise ValueError("export must be RGB8 PNG within max_side")
        width, height = im.size
        pixels = im.tobytes()
        data = np.asarray(im, dtype=np.float64).mean(axis=2)
    if min(width, height) < 3:
        raise ValueError("video frame is too small")
    lap = (data[1:-1, :-2] + data[1:-1, 2:] + data[:-2, 1:-1] +
           data[2:, 1:-1] - 4 * data[1:-1, 1:-1])
    return dict(width=width, height=height, sha256=digest(path),
                decoded_rgb_sha256=hashlib.sha256(
                    f"{width}x{height}:RGB:".encode("ascii") + pixels).hexdigest(),
                sharpness_laplacian_variance=float(lap.var()))

def expected_orientation(timeline, width, height):
    source_w, source_h = timeline["width"], timeline["height"]
    if timeline["rotation_degrees"] % 180:
        source_w, source_h = source_h, source_w
    ratio = min(1, LIMITS["max_side"] / source_w, LIMITS["max_side"] / source_h)
    # Swscale can round an output side by one pixel.
    return abs(width - source_w * ratio) <= 1 and abs(height - source_h * ratio) <= 1

def resolve_tool(value, name):
    raw = value or shutil.which(name)
    if not raw:
        raise ValueError(name + " is not available")
    path = Path(raw).resolve(strict=True)
    if not path.is_file():
        raise ValueError(name + " is not a file")
    return path

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--data-kind", required=True, choices=("synthetic", "real"))
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=48)
    p.add_argument("--all-frames", action="store_true", help="whole-video-192-v1: retain all 2..192 decoded frames")
    p.add_argument("--stream-index", type=int)
    p.add_argument("--min-sharpness", type=float, default=0,
                   help="Laplacian variance threshold; 0 disables heuristic rejection")
    p.add_argument("--ffmpeg")
    p.add_argument("--ffprobe")
    return p

def execute(args):
    if not is_valid_dataset_id(args.dataset_id) or len(args.dataset_id) > 64:
        raise ValueError("dataset-id must match photo-io-v1 and be at most 64 characters")
    all_frames = getattr(args, "all_frames", False)
    if type(all_frames) is not bool:
        raise ValueError("all-frames must be boolean")
    if not all_frames and (type(args.count) is not int or not 2 <= args.count <= 96):
        raise ValueError("count must be 2..96")
    if args.stream_index is not None and args.stream_index < 0:
        raise ValueError("stream-index must be nonnegative")
    if not math.isfinite(args.min_sharpness) or args.min_sharpness < 0:
        raise ValueError("min-sharpness must be finite and nonnegative")
    video = Path(args.video).resolve(strict=True)
    if not video.is_file() or not 0 < video.stat().st_size <= LIMITS["input_bytes"]:
        raise ValueError("video must be a local nonempty file <=512 MiB")
    ffmpeg, ffprobe = resolve_tool(args.ffmpeg, "ffmpeg"), resolve_tool(args.ffprobe, "ffprobe")
    run = Path(args.output).resolve()
    if (ROOT / "outputs").resolve() not in run.parents or run.exists():
        raise ValueError("output must be a new directory strictly under project outputs")
    prefix = "data/real" if args.data_kind == "real" else "outputs/delegation/T01A/fixtures"
    dataset = (ROOT / prefix / args.dataset_id).resolve()
    if dataset.parent != (ROOT / prefix).resolve() or dataset.exists():
        raise ValueError("dataset root must be new and contained in the contract prefix")
    if dataset == run or dataset in run.parents or run in dataset.parents:
        raise ValueError("dataset root and evidence output must be disjoint")
    run.mkdir(parents=True, exist_ok=False)
    result = dict(contract_version="video-photo-v1", status="RUNNING", exit_code=None,
                  started_utc=utc(), source_video=str(video), data_kind=args.data_kind,
                  dataset_root=str(dataset.relative_to(ROOT)).replace("\\", "/"),
                  selection=dict(method="all_decoded_frames" if all_frames else "uniform_presentation_time_nearest", requested=None if all_frames else args.count,
                                 min_sharpness=args.min_sharpness, exact_duplicate_policy="reject"),
                  limits=LIMITS, frames=[], photo_io=None, error=None,
                  coverage_status="NOT_EVALUATED", scale_status="unknown",
                  reconstruction_status="NOT_RUN")
    deadline = time.monotonic() + LIMITS["process_seconds"]
    try:
        result["source_sha256"] = digest(video)
        result["implementation"] = {
            str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
            for p in (Path(__file__), ROOT / "integrations/video_io/timeline.py")}
        result["tools"] = {str(p): digest(p) for p in (ffmpeg, ffprobe)}
        result["interpreter"] = dict(path=sys.executable, version=sys.version)
        save(run / "RESULT.json", result)
        overrides = {n: os.environ[n] for n in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")}
        overrides["PYTHONIOENCODING"] = "utf-8"
        for name, tool in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
            bounded_run([tool, "-version"], run, name+"_version", deadline, dataset, overrides)
        stream = str(args.stream_index) if args.stream_index is not None else "V:0"
        entries = ("stream=index,codec_type,time_base,start_pts,start_time,width,height,pix_fmt,"
                   "sample_aspect_ratio,color_range,color_space,color_transfer,color_primaries:"
                   "stream_disposition=attached_pic:stream_tags=rotate:stream_side_data:"
                   "frame=media_type,stream_index,best_effort_timestamp,best_effort_timestamp_time,"
                   "pts,pkt_dts,duration,width,height,pix_fmt,sample_aspect_ratio,"
                   "color_range,color_space,color_transfer,color_primaries:frame_side_data")
        out, _ = bounded_run([ffprobe, "-v", "error", "-threads", "2",
                              "-protocol_whitelist", "file", "-format_whitelist", "mov,matroska,avi",
                              "-err_detect", "explode",
                              "-select_streams", stream, "-show_streams", "-show_frames",
                              "-show_entries", entries, "-of", "json", video],
                             run, "probe", deadline, dataset, overrides)
        timeline = parse_probe(json.loads(out.read_text(encoding="utf-8")))
        if len(timeline["frames"]) > LIMITS["decoded_frames"]:
            raise LimitError("too many decoded frames")
        if timeline["width"] * timeline["height"] > LIMITS["input_pixels"]:
            raise LimitError("input pixel count exceeds profile")
        base = Fraction(*timeline["time_base"])
        span = (timeline["frames"][-1]["pts"] - timeline["frames"][0]["pts"]) * base
        if span > LIMITS["span_seconds"]:
            raise LimitError("video timestamp span exceeds 120 seconds")
        if len(timeline["frames"]) < 2:
            raise ValueError("at least two decoded video frames are required")
        save(run / "timeline.json", timeline)
        selected = select_all(timeline) if all_frames else select_uniform(timeline, args.count)
        if all_frames:
            result["profile"] = "whole-video-192-v1"
            result["selection"].update(profile="whole-video-192-v1", requested=len(selected))
        result["timeline"] = {k: v for k, v in timeline.items() if k != "frames"}
        result["decoded_count"] = len(timeline["frames"])
        result["frames"] = [dict(f, output_index=i, status="NOT_RUN", photo_input_index=None)
                            for i, f in enumerate(selected)]
        save(run / "RESULT.json", result)
        (dataset / "frames").mkdir(parents=True, exist_ok=False)
        expression = "1" if all_frames else "+".join(f"eq(n,{f['decode_index']})" for f in selected)
        filters = ("select='" + expression + "',"
                   r"scale=w=min(1024\,iw):h=min(1024\,ih):force_original_aspect_ratio=decrease,"
                   "format=rgb24,showinfo")
        _, log = bounded_run(
            [ffmpeg, "-hide_banner", "-nostdin", "-n", "-loglevel", "info", "-copyts", "-xerror",
             "-threads", "2", "-protocol_whitelist", "file", "-format_whitelist", "mov,matroska,avi",
             "-err_detect", "explode", "-i", video, "-map", f"0:{timeline['stream_index']}",
             "-an", "-sn", "-dn", "-filter_threads", "2", "-vf", filters,
             "-fps_mode", "passthrough", "-enc_time_base", "filter",
             "-avoid_negative_ts", "disabled", "-c:v", "png", "-threads", "2",
             "-start_number", "0", "-f", "image2", dataset / "frames/frame_%06d.png"],
            run, "extract", deadline, dataset, overrides)
        observed = parse_showinfo(log.read_text(encoding="utf-8", errors="replace"))
        validate_export(timeline, selected, observed)
        paths = [dataset / "frames" / f"frame_{i:06d}.png" for i in range(len(selected))]
        if sorted((dataset / "frames").iterdir()) != paths:
            raise ValueError("PNG set differs from the selected frame count")
        seen = {}
        accepted = []
        for row, path, shown in zip(result["frames"], paths, observed):
            info = inspect_image(path)
            row.update(info, image=path.relative_to(dataset).as_posix())
            if (info["width"], info["height"]) != (shown["width"], shown["height"]):
                raise ValueError("PNG dimensions differ from showinfo")
            if not expected_orientation(timeline, info["width"], info["height"]):
                raise ValueError("PNG dimensions conflict with static autorotation and scale")
            row["export_pts_verified"] = True
            identity = info["decoded_rgb_sha256"]
            if identity in seen:
                row.update(status="REJECTED_DUPLICATE", duplicate_of_output_index=seen[identity])
            elif info["sharpness_laplacian_variance"] < args.min_sharpness:
                row["status"] = "REJECTED_SHARPNESS"
                seen[identity] = row["output_index"]
            else:
                row.update(status="EXTRACTED", photo_input_index=len(accepted))
                seen[identity] = row["output_index"]
                accepted.append(dict(image=row["image"], mask=None))
            save(run / "RESULT.json", result)
        result["counts"] = dict(selected=len(selected), accepted=len(accepted),
                                rejected=len(selected)-len(accepted))
        if len(accepted) < 2:
            raise ValueError("fewer than two distinct frames survive quality checks")
        if digest(video) != result["source_sha256"]:
            raise ValueError("source video changed during extraction")
        manifest = dict(contract_version="photo-io-v1", dataset_id=args.dataset_id,
                        data_kind=args.data_kind, dataset_root=result["dataset_root"],
                        frames=accepted)
        save(run / "photo_manifest.json", manifest)
        overrides["PYTHONPATH"] = str(ROOT / "src")
        bounded_run([sys.executable, "-B", "-m", "kozlov_lab.io",
                     "--manifest", run / "photo_manifest.json", "--output", run / "photo_io",
                     "--data-kind", args.data_kind],
                    run, "photo_io", deadline, dataset, overrides)
        runs = list((run / "photo_io").glob("*/run.json"))
        if len(runs) != 1:
            raise ValueError("photo-io produced an unexpected run set")
        photo_run = json.loads(runs[0].read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (runs[0].parent / "frames.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        if photo_run.get("status") != "SUCCESS" or len(rows) != len(accepted):
            raise ValueError("photo-io did not accept every surviving frame")
        by_index = {r["input_index"]: r for r in rows}
        if set(by_index) != set(range(len(accepted))) or len(by_index) != len(rows):
            raise ValueError("photo-io input identity mismatch")
        for row in result["frames"]:
            if row["photo_input_index"] is not None:
                photo = by_index[row["photo_input_index"]]
                if photo.get("status") != "OK" or not photo.get("frame_id"):
                    raise ValueError("photo-io row lacks successful frame identity")
                row.update(status="PHOTO_IO_OK", photo_frame_id=photo["frame_id"])
        result["photo_io"] = dict(run=str(runs[0]), sha256=digest(runs[0]),
                                 frames_sha256=digest(runs[0].parent / "frames.jsonl"))
        result["source_sha256_after"] = digest(video)
        if result["source_sha256_after"] != result["source_sha256"]:
            raise ValueError("source video changed before completion")
        for filename, before in result["tools"].items():
            if digest(filename) != before:
                raise ValueError("native tool identity changed")
        for filename, before in result["implementation"].items():
            if digest(ROOT / filename) != before:
                raise ValueError("video implementation changed during execution")
        for row in result["frames"]:
            if digest(dataset / row["image"]) != row["sha256"]:
                raise ValueError("extracted PNG changed during photo preparation")
        result.update(status="SUCCESS", exit_code=0)
    except Exception as error:
        code = 5 if isinstance(error, LimitError) else 2 if isinstance(error, ValueError) else 4
        result.update(status={2:"INPUT_ERROR", 4:"BACKEND_ERROR", 5:"RESOURCE_LIMIT"}[code],
                      exit_code=code, error=type(error).__name__ + ": " + str(error))
    finally:
        result.update(ended_utc=utc(), output_bytes=bytes_under(run)+bytes_under(dataset))
        save(run / "RESULT.json", result)
    print(json.dumps(dict(status=result["status"], exit_code=result["exit_code"],
                          result=str(run / "RESULT.json")), ensure_ascii=False))
    return result["exit_code"]

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (OSError, ValueError) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
