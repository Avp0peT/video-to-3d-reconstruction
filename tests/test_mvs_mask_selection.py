"""TEST ONLY: synthetic masks/models through the actual consumer CLI function."""
import copy
import json

import pytest
import test_mvs_mask_bridge as base

cli, mio = base.cli, base.mio


def save(path, doc):
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def make_case(tmp_path, count=4):
    world = base.build_world(tmp_path / "fixture_world", model_images=count, extra_frames=0, n_ok=count-2)
    primary_doc = json.loads(world["manifest"].read_text(encoding="utf-8"))
    primary_doc["status"] = "PARTIAL"
    primary_doc["counts"].update(no_match=2, not_run=0)
    for frame in primary_doc["frames"][-2:]:
        frame["status"] = "NO_MATCH"
        frame.pop("not_run_reason", None)
    save(world["manifest"], primary_doc)
    supplement = base.world_manifest(world["root"], n_ok=count, directory=world["root"] / "supplement")
    doc = json.loads(supplement.read_text(encoding="utf-8"))
    doc["frames"] = doc["frames"][-2:]
    doc["counts"].update(requested=2, success=2)
    save(supplement, doc)
    ledger = tmp_path / "selection.json"
    primary = world["manifest"]
    parsed = [mio.parse_object_manifest(p) for p in (primary, supplement)]
    selections = []
    for m in parsed:
        for frame in m["frames"]:
            if frame["status"] == "OK":
                selections.append({"input_index": frame["input_index"], "frame_id": frame["frame_id"],
                    "manifest_path": m["manifest_path"], "selected_candidate_id": frame["selected_candidate_id"],
                    "selected_mask_sha256": frame["selected_mask"]["sha256"]})
    save(ledger, {"schema_version": "object-mask-selection-v1",
        "manifests": [{"path": m["manifest_path"], "sha256": m["manifest_sha256"]} for m in parsed],
        "selections": selections})
    return world, supplement, ledger


def arguments(case, output, ledger=True):
    world, supplement, path = case
    extra = ["--mask-manifest", str(supplement)]
    if ledger:
        extra += ["--selection-ledger", str(path)]
    return base.cli_args(world, world["manifest"], output, *extra)


def refresh_source_binding(ledger, source):
    doc = json.loads(ledger.read_text(encoding="utf-8"))
    for item in doc["manifests"]:
        if item["path"] == str(source.resolve()):
            item["sha256"] = mio.sha256_file(source)
    save(ledger, doc)


@pytest.mark.parametrize("count", [4, 44])
def test_complete_cli_keeps_original_partial_and_no_match(tmp_path, count):
    case = make_case(tmp_path, count)
    before = [p.read_bytes() for p in (case[0]["manifest"], case[1])]
    output = tmp_path / "converted"
    assert cli.main(arguments(case, output)) == 0
    result = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert result["counts"] == {"expected": count, "ok": count, "missing": 0, "error": 0, "not_run": 0}
    assert len({(f["input_index"], f["frame_id"]) for f in result["frames"]}) == count
    assert all(f["readback_verified"] for f in result["frames"])
    assert result["evidence_level"] == "synthetic_interface_fixture"
    assert result["object_mask_manifests"][0]["status"] == "PARTIAL"
    assert result["mask_selection"]["source_runs"][0]["counts"]["no_match"] == 2
    excluded = result["mask_selection"]["unselected_source_rows"]
    assert len(excluded) == 2 and all(f["original_status"] == "NO_MATCH" for f in excluded)
    assert {f["input_index"] for f in excluded} == {count-2, count-1}
    assert all(f["source_manifest_sha256"] and f["replacement_manifest_sha256"] for f in excluded)
    assert result["unbound_object_mask_frames"] == []
    assert result["quality_status"] == result["cross_view_status"] == "NOT_EVALUATED"
    assert result["mvs_mask_arguments"]["actual_mvs_invocations"] == 0
    assert result["inputs"]["input_bindings_unchanged"] is True
    assert str(case[2]) in {f["path"] for f in result["inputs"]["files"]}
    assert before == [p.read_bytes() for p in (case[0]["manifest"], case[1])]


def test_default_cli_still_rejects_duplicate_no_match(tmp_path):
    case = make_case(tmp_path)
    output = tmp_path / "rejected"
    assert cli.main(arguments(case, output, ledger=False)) == 2
    assert not output.exists()


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "candidate", "mask_hash", "source_hash",
    "unknown_source", "relative_source", "frame_id", "bool_index", "missing_source", "duplicate_source", "schema", "select_no_match"])
def test_bad_ledger_refused_before_output(tmp_path, change):
    case = make_case(tmp_path)
    path = case[2]
    doc = json.loads(path.read_text(encoding="utf-8"))
    if change == "missing": doc["selections"].pop()
    elif change == "extra":
        item = copy.deepcopy(doc["selections"][0]); item["input_index"] = 500; doc["selections"].append(item)
    elif change == "duplicate": doc["selections"].append(copy.deepcopy(doc["selections"][0]))
    elif change == "candidate": doc["selections"][0]["selected_candidate_id"] = "absent"
    elif change == "mask_hash": doc["selections"][0]["selected_mask_sha256"] = "0" * 64
    elif change == "source_hash": doc["manifests"][0]["sha256"] = "0" * 64
    elif change == "unknown_source": doc["selections"][0]["manifest_path"] = str(tmp_path / "missing.json")
    elif change == "relative_source": doc["manifests"][0]["path"] = "relative.json"
    elif change == "frame_id": doc["selections"][0]["frame_id"] = "wrong-photo"
    elif change == "bool_index": doc["selections"][0]["input_index"] = False
    elif change == "missing_source": doc["manifests"].pop()
    elif change == "duplicate_source": doc["manifests"][1] = copy.deepcopy(doc["manifests"][0])
    elif change == "schema": doc["schema_version"] = "unknown"
    elif change == "select_no_match": doc["selections"][-1]["manifest_path"] = str(case[0]["manifest"])
    save(path, doc)
    output = tmp_path / "rejected"
    assert cli.main(arguments(case, output)) == 2
    assert not output.exists()


@pytest.mark.parametrize("change", ["object", "t01", "frame_id", "input_index", "canonical_hash", "canonical_path",
    "photo_hash", "size", "counts", "duplicate_row", "source_bytes", "unselected_mask_bytes", "no_source_image"])
def test_all_sources_validated_even_unselected_rows(tmp_path, change):
    case = make_case(tmp_path)
    primary = case[0]["manifest"]
    doc = json.loads(primary.read_text(encoding="utf-8"))
    frame = doc["frames"][-1]  # the ledger excludes this original NO_MATCH
    if change == "object": doc["object"]["object_id"] = "other-object"
    elif change == "t01": doc["source_io"]["run_id"] = "other-run"
    elif change == "frame_id": frame["frame_id"] = "wrong-image"
    elif change == "input_index": frame["input_index"] = 99
    elif change == "canonical_hash": frame["source_image_sha256"] = "0" * 64
    elif change == "canonical_path": frame["source_image"] = doc["frames"][0]["source_image"]
    elif change == "photo_hash": frame["photo_io_image_sha256"] = "2" * 64
    elif change == "size": frame["width"] += 1
    elif change == "no_source_image": frame["source_image"] = None
    elif change == "counts": doc["counts"]["no_match"] = 1
    elif change == "duplicate_row":
        doc["frames"].append(copy.deepcopy(frame)); doc["counts"]["requested"] += 1; doc["counts"]["no_match"] += 1
    elif change == "source_bytes":
        save(primary, doc)  # alter actual source bytes after the frozen ledger
        primary.write_text(primary.read_text(encoding="utf-8") + " ", encoding="utf-8")
        assert cli.main(arguments(case, tmp_path / "rejected")) == 2
        return
    elif change == "unselected_mask_bytes":
        # Make supplement contain OK duplicates of all four frames. Select only
        # its last two; corrupt one of its unchosen OK masks without changing hash.
        supplement = base.world_manifest(case[0]["root"], n_ok=4, directory=case[1].parent)
        refresh_source_binding(case[2], supplement)
        mask = supplement.parent / "masks/000000_c00.png"
        mask.write_bytes(b"broken")
        assert cli.main(arguments(case, tmp_path / "rejected")) == 2
        return
    save(primary, doc)
    refresh_source_binding(case[2], primary)
    output = tmp_path / "rejected"
    assert cli.main(arguments(case, output)) == 2
    assert not output.exists()


@pytest.mark.parametrize("target", ["ledger", "manifest", "unselected_mask"])
def test_input_drift_after_warp_blocks_publication(tmp_path, monkeypatch, target):
    case = make_case(tmp_path)
    if target == "unselected_mask":
        base.world_manifest(case[0]["root"], n_ok=4, directory=case[1].parent)
        refresh_source_binding(case[2], case[1])
    real = cli.warp_binary_mask
    done = []
    def drifting(*args, **kwargs):
        result = real(*args, **kwargs)
        if not done:
            done.append(True)
            path = case[2] if target == "ledger" else case[0]["manifest"] if target == "manifest" else case[1].parent / "masks/000000_c00.png"
            path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(cli, "warp_binary_mask", drifting)
    output = tmp_path / "drifted"
    assert cli.main(arguments(case, output)) == 2
    result = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert result["publishable_arguments"] is False
    assert result["inputs"]["input_bindings_unchanged"] is False
    assert result["inputs"]["drift"]
