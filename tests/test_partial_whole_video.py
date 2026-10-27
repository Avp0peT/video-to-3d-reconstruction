"""Offline whole-video partial planning, preserving local geometry and holes."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import pytest
from integrations.demo_pipeline import partial_reconstruction as p
from test_partial_reconstruction_inputs import saved


def expand(saved,count,contract='sfm-native-object-video-v1',excluded=True):
    run=json.loads((saved/'run.json').read_text());run['contract_version']=contract
    rows=[json.loads(x) for x in (saved/'frames_map.jsonl').read_text().splitlines()]
    first=copy.deepcopy(rows[0]);first_mask=copy.deepcopy(run['effective_options']['foreground']['mask_bindings'][0])
    rows=rows[:3]
    with sqlite3.connect(saved/'database.db') as con:
        for iid in range(4,count+1):
            image=saved/f'view_{iid}.png';image.write_bytes((saved/'view_1.png').read_bytes())
            mask=saved/f'mask_{iid}.png';mask.write_bytes((saved/'mask_1.png').read_bytes())
            row=dict(first,input_index=1000+iid,database_image_id=iid,staged_image_name=image.name)
            rows.append(row)
            run['effective_options']['foreground']['mask_bindings'].append(dict(first_mask,input_index=row['input_index'],image_name=image.name,staged_path=str(mask)))
            con.execute('INSERT INTO images VALUES(?,?,1)',(iid,image.name))
            con.execute('INSERT INTO keypoints VALUES(?,0,2,?)',(iid,b''))
    if excluded:rows.append(dict(input_index=999,database_image_id=None,status='NO_MATCH',reason='saved no match'))
    (saved/'run.json').write_text(json.dumps(run))
    (saved/'frames_map.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
    return saved


def test_profiles_only_change_id_and_capacity():
    new=p.resolve_profile(p.WHOLE_VIDEO_PROFILE_ID)
    assert new['max_images']==192 and p.PROFILE['max_images']==32
    assert {k:v for k,v in new.items() if k not in ('id','max_images')} == {k:v for k,v in p.PROFILE.items() if k not in ('id','max_images')}
    assert new['max_groups']==8 and new['max_points']==1000
    assert new['mapping_seconds_per_group']==45 and new['final_ba_seconds']==15
    with pytest.raises(ValueError,match='unknown'):p.resolve_profile('unknown')


def test_full_192_selected_views_plan_without_truncation(saved):
    expand(saved,192,excluded=False)
    plan,_=p.prepare(saved,profile_id=p.WHOLE_VIDEO_PROFILE_ID)
    assert len(plan['frames'])==192
    assert sum(len(g['image_ids']) for g in plan['graph']['groups'])==192
    assert plan['profile']==p.WHOLE_VIDEO_PROFILE
    assert plan['hole_filling'] is False and plan['components_merged'] is False
    assert plan['graph']['groups'][0]['scheduled']
    assert len(plan['graph']['tracks'])==15
    with pytest.raises(ValueError,match='source required'):p.prepare(saved)


def test_192_original_rows_keep_no_match(saved):
    expand(saved,191)
    plan,_=p.prepare(saved,profile_id=p.WHOLE_VIDEO_PROFILE_ID)
    assert len(plan['frames'])==191
    assert plan['excluded_source_rows']==[dict(input_index=999,status='NO_MATCH',reason='saved no match')]


def test_193_original_rows_rejected(saved):
    expand(saved,192)
    with pytest.raises(ValueError,match='original input row limit'):
        p.prepare(saved,profile_id=p.WHOLE_VIDEO_PROFILE_ID)


def test_default_still_rejects_33_selected_views(saved):
    expand(saved,33,contract='sfm-native-object-v1',excluded=False)
    with pytest.raises(ValueError,match='bounded unique'):
        p.prepare(saved)


def test_graph_192_and_193_bounds_and_eight_groups_unchanged():
    assert len(p.graph_plan(list(range(1,193)),[],profile_id=p.WHOLE_VIDEO_PROFILE_ID)['groups'])==192
    with pytest.raises(ValueError,match='bounded unique'):
        p.graph_plan(list(range(1,194)),[],profile_id=p.WHOLE_VIDEO_PROFILE_ID)
    pairs=[]
    for base in range(1,28,3):
        for a,b in ((base,base+1),(base+1,base+2)):
            pairs.append((a,b,[[i,i] for i in range(15)]))
    with pytest.raises(ValueError,match='group budget'):
        p.graph_plan(list(range(1,28)),pairs,profile_id=p.WHOLE_VIDEO_PROFILE_ID)


def test_new_cli_plan_only_binds_profile_without_native(saved,tmp_path,monkeypatch):
    expand(saved,3)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('partial192entry',root/'scripts/reconstruct_partial.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    from integrations.demo_pipeline import pipeline
    monkeypatch.setattr(pipeline,'_invoke',lambda *a,**kw:pytest.fail('native launched'))
    output=tmp_path/'output'
    assert entry.main(['--sfm-run',str(saved),'--output',str(output),'--profile',p.WHOLE_VIDEO_PROFILE_ID])==0
    plan=json.loads((output/'PLAN.json').read_text())
    assert plan['profile']['max_images']==192
    assert not (output/'local').exists()
    frozen=json.loads((output/'ENTRY_PLAN.json').read_text())
    assert frozen['wall_seconds']==480 and frozen['memory_bytes']==2<<30
