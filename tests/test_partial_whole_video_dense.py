"""Offline dense partial profile isolation; no native launches."""
import importlib.util
import json
from pathlib import Path
import pytest
from integrations.demo_pipeline import partial_reconstruction as p
from test_partial_reconstruction_inputs import saved
from test_partial_whole_video import expand


def test_dense_capacity_changes_only_id_and_point_budget():
    old=p.resolve_profile(p.WHOLE_VIDEO_PROFILE_ID)
    dense=p.resolve_profile(p.WHOLE_VIDEO_DENSE_PROFILE_ID)
    assert old['max_points']==p.PROFILE['max_points']==1000
    assert dense['max_points']==4096 and dense['max_images']==192
    assert {k:v for k,v in dense.items() if k not in ('id','max_points')} == {k:v for k,v in old.items() if k not in ('id','max_points')}
    assert dense['max_groups']==8 and dense['mapping_seconds_per_group']==45
    assert dense['final_ba_seconds']==15
    # Returned profiles cannot mutate any registered profile.
    dense['max_points']=1
    assert p.resolve_profile(p.WHOLE_VIDEO_DENSE_PROFILE_ID)['max_points']==4096


def test_dense_plan_accepts_video_contract_and_retains_exclusion(saved):
    expand(saved,3)
    plan,_=p.prepare(saved,profile_id=p.WHOLE_VIDEO_DENSE_PROFILE_ID)
    assert plan['profile']['max_points']==4096
    assert plan['excluded_source_rows']==[dict(input_index=999,status='NO_MATCH',reason='saved no match')]
    assert plan['hole_filling'] is False and plan['components_merged'] is False


def test_dense_cli_plan_preserves_existing_supervision(saved,tmp_path,monkeypatch):
    expand(saved,3)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('densepartialentry',root/'scripts/reconstruct_partial.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    from integrations.demo_pipeline import pipeline
    monkeypatch.setattr(pipeline,'_invoke',lambda *a,**kw:pytest.fail('native launched'))
    output=tmp_path/'denseplan'
    assert entry.main(['--sfm-run',str(saved),'--output',str(output),'--profile',p.WHOLE_VIDEO_DENSE_PROFILE_ID])==0
    plan=json.loads((output/'PLAN.json').read_text())
    entry_plan=json.loads((output/'ENTRY_PLAN.json').read_text())
    assert plan['profile']['max_points']==4096
    assert entry_plan['wall_seconds']==480 and entry_plan['memory_bytes']==2<<30
    assert not (output/'local').exists()
