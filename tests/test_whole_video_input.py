"""Synthetic whole-video contract tests; never launch ffmpeg or SAM3."""
import pytest
from integrations.video_io.timeline import select_all, select_uniform
from scripts.prepare_video import parser
from test_object_input import project, make_test_video_receipt, write_json
from integrations.object_input import session


def timeline(count):
    return {"time_base": [1, 24], "frames": [
        {"decode_index": i, "pts": i, "time_base": [1, 24]} for i in range(count)]}


def test_all_192_preserves_every_original_frame_and_pts():
    value = timeline(192)
    selected = select_all(value)
    assert selected == value["frames"]
    assert selected is not value["frames"]
    assert selected[0]["pts"] == 0 and selected[-1]["pts"] == 191
    assert len(select_uniform(value, 96)) == 96


@pytest.mark.parametrize("count", [0, 1, 193])
def test_all_frames_rejects_outside_profile_instead_of_truncating(count):
    with pytest.raises(ValueError, match="2..192"):
        select_all(timeline(count))


@pytest.mark.parametrize("change", ["index", "pts"])
def test_all_frames_rejects_corrupted_timeline(change):
    value = timeline(192)
    value["frames"][100]["decode_index" if change == "index" else "pts"] = 99
    with pytest.raises(ValueError):
        select_all(value)


def test_old_uniform_limit_is_unchanged():
    with pytest.raises(ValueError, match="2 and 96"):
        select_uniform(timeline(192), 192)


def test_cli_all_frames_is_explicit():
    args = ["--video", "TEST_DATA", "--dataset-id", "TEST_DATA", "--data-kind", "synthetic", "--output", "TEST_DATA"]
    assert parser().parse_args(args).all_frames is False
    assert parser().parse_args(args + ["--all-frames"]).all_frames is True


def test_whole_video_receipt_requires_profile_and_complete_selection(project, tmp_path):
    directory, video = make_test_video_receipt(project, tmp_path)
    video["selection"].update(method="all_decoded_frames", profile="whole-video-192-v1")
    video["profile"] = "whole-video-192-v1"
    write_json(directory / "RESULT.json", video)
    checked, _, _ = session._validated_video(directory, project)
    assert checked["selection"]["method"] == "all_decoded_frames"
    video["frames"] = video["frames"][:-1]
    write_json(directory / "RESULT.json", video)
    with pytest.raises(ValueError, match="selected count"):
        session._validated_video(directory, project)


def test_whole_video_rejects_missing_profile(project, tmp_path):
    directory, video = make_test_video_receipt(project, tmp_path)
    video["selection"]["method"] = "all_decoded_frames"
    write_json(directory / "RESULT.json", video)
    with pytest.raises(ValueError, match="profile/count"):
        session._validated_video(directory, project)


@pytest.mark.parametrize("whole", [False, True])
def test_execute_extract_argv_uses_constant_selector_for_whole_video(tmp_path, monkeypatch, whole):
    # Stub only process execution: capture the production argv without decoding.
    from scripts import prepare_video as entry
    from argparse import Namespace
    root = tmp_path / "TEST_DATA_root"
    (root / "outputs").mkdir(parents=True)
    for relative in ("scripts/prepare_video.py", "integrations/video_io/timeline.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# TEST_DATA binding marker", encoding="utf8")
    source = root / "TEST_DATA_video.bin"
    source.write_bytes(b"TEST_DATA non-video; native execution prohibited")
    monkeypatch.setattr(entry, "ROOT", root)
    # __file__ participates in implementation binding under ROOT.
    monkeypatch.setattr(entry, "__file__", str(root / "scripts/prepare_video.py"))
    monkeypatch.setattr(entry, "resolve_tool", lambda value, name: source)
    value = timeline(192)
    value.update(width=720, height=1280, stream_index=0)
    monkeypatch.setattr(entry, "parse_probe", lambda probe: value)
    captured = []
    def process(argv, run, name, deadline, dataset, env_overrides=None):
        captured.append((name, argv))
        if name == "extract":
            raise ValueError("TEST_DATA stop before native extraction")
        out = run / (name + ".stdout")
        out.write_text("{}", encoding="utf8")
        return out, out
    monkeypatch.setattr(entry, "bounded_run", process)
    args = Namespace(video=str(source), dataset_id="TEST_DATA_all", data_kind="synthetic",
                     output=str(root / "outputs/run"), count=96, all_frames=whole,
                     ffmpeg=None, ffprobe=None, stream_index=None, min_sharpness=0)
    assert entry.execute(args) != 0
    argv = next(argv for name, argv in captured if name == "extract")
    filters = argv[argv.index("-vf") + 1]
    if whole:
        assert filters.startswith("select='1',scale=")
        assert "eq(n," not in filters
    else:
        assert "eq(n,0)" in filters and "eq(n,191)" in filters
        assert filters.count("eq(n,") == 96
    assert argv[argv.index("-fps_mode") + 1] == "passthrough"
    assert filters.endswith("format=rgb24,showinfo")
