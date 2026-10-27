"""Independent capacity/runtime boundaries for the new object profile."""
import hashlib,json
from pathlib import Path
import pytest
from PIL import Image
from kozlov_lab.reconstruction import colmap_adapter as adapter, profiles, cli

def upstream(tmp_path, count):
    root=tmp_path/"input";root.mkdir()
    rows=[]
    for i in range(count):
        path=root/f"{i}.png"
        Image.new("RGB",(8,6),(i % 256,i//256,17)).save(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(adapter.UpstreamRow(input_index=i,frame_id=f"frame{i}",
            image_output=path.name,mask_output=None,upstream_image_sha256=digest,
            canonical_width=8,canonical_height=6))
    return adapter.UpstreamRun(root,"fixture","synthetic","photo-io-v1","SUCCESS",rows)

@pytest.mark.parametrize("profile", [profiles.PROFILE_T02A_V1,profiles.PROFILE_CAD02A_SHARED_V1])
def test_legacy_limit_still_rejects_thirteen(tmp_path,profile):
    with pytest.raises(adapter.InputError,match="limit of 12"):
        adapter.stage_inputs(upstream(tmp_path,13),tmp_path/"stage",profile)

@pytest.mark.parametrize("count",[13,66,96])
def test_object_accepts_distinct_views_beyond_pilot_without_resampling(tmp_path,count):
    source=upstream(tmp_path,count)
    staged=adapter.stage_inputs(source,tmp_path/"stage",profiles.PROFILE_CAD_OBJECT_V1)
    assert len(staged.distinct)==count
    assert all(hashlib.sha256((staged.images_dir/f"{r.input_index:06d}.png").read_bytes()).hexdigest()
               ==r.upstream_image_sha256 for r in source.rows)

def test_object_rejects_ninety_seven_before_copy(tmp_path):
    with pytest.raises(adapter.InputError,match="limit of 96"):
        adapter.stage_inputs(upstream(tmp_path,97),tmp_path/"stage",profiles.PROFILE_CAD_OBJECT_V1)
    assert list((tmp_path/"stage/images").iterdir())==[]

def test_dedup_does_not_spend_additional_view_budget(tmp_path):
    u=upstream(tmp_path,97)
    (u.run_dir/u.rows[-1].image_output).write_bytes((u.run_dir/u.rows[0].image_output).read_bytes())
    u.rows[-1].upstream_image_sha256=u.rows[0].upstream_image_sha256
    s=adapter.stage_inputs(u,tmp_path/"stage",profiles.PROFILE_CAD_OBJECT_V1)
    assert len(s.distinct)==96
    assert s.rows[-1].representative_input_index==0

def test_legacy_options_do_not_gain_new_capacity_or_runtime_fields():
    a=adapter.effective_options()
    b=adapter.effective_options(profiles.PROFILE_CAD02A_SHARED_V1)
    c=adapter.effective_options(profiles.PROFILE_CAD_OBJECT_V1)
    assert a["max_distinct_images"]==b["max_distinct_images"]==12
    assert "runtime_thread_environment" not in a and "runtime_thread_environment" not in b
    assert "profile_id" not in a
    assert c["max_distinct_images"]==96 and set(c["runtime_thread_environment"].values())=={"2"}
    assert profiles.contract_version(profiles.PROFILE_CAD_OBJECT_V1)=="sfm-native-object-v1"

def test_object_requires_startup_thread_policy_before_input_or_backend(tmp_path,monkeypatch):
    for key in profiles.RUNTIME_THREAD_ENV:monkeypatch.delenv(key,raising=False)
    def forbid(*a,**k):raise AssertionError("Native/input was called before environment rejection")
    monkeypatch.setattr(cli,"load_upstream_run",forbid)
    monkeypatch.setattr(cli,"backend_info",forbid)
    out=tmp_path/"result";out.mkdir()
    exit_code=cli.run_reconstruction(io_run=str(tmp_path/"absent"),run_dir=out,project_root=tmp_path,profile=profiles.PROFILE_CAD_OBJECT_V1)
    rec=json.loads((out/"run.json").read_text(encoding="utf-8"))
    assert exit_code==2 and rec["status"]=="INPUT_ERROR"
    assert "thread environment" in rec["errors"][0]["message"] and rec["stages"]==[]
