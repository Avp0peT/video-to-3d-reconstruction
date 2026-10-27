"""Offline bounds for the explicit whole-video SfM profile; no native SfM."""
import copy
import json
import pytest
from kozlov_lab.reconstruction import colmap_adapter as adapter, profiles, point_selection
from integrations.mask_review import review
from test_object_profile import upstream
from test_mask_review import _make_input, _write_json

VIDEO = profiles.PROFILE_CAD_OBJECT_VIDEO_V1

@pytest.mark.parametrize('count', [97, 192])
def test_video_capacity_preserves_bytes(tmp_path, count):
    source = upstream(tmp_path, count)
    result = adapter.stage_inputs(source, tmp_path/'stage', VIDEO)
    assert len(result.distinct) == count
    assert all((result.images_dir/f'{i:06d}.png').read_bytes() == (source.run_dir/f'{i}.png').read_bytes() for i in range(count))

def test_video_193_refused_before_copy(tmp_path):
    with pytest.raises(adapter.InputError, match='limit of 192'):
        adapter.stage_inputs(upstream(tmp_path,193),tmp_path/'stage',VIDEO)
    assert not list((tmp_path/'stage/images').iterdir())

def test_separate_profile_options_and_fingerprint(tmp_path):
    source=upstream(tmp_path,2)
    old=adapter.effective_options(profiles.PROFILE_CAD_OBJECT_V1)
    new=adapter.effective_options(VIDEO)
    assert old['max_distinct_images']==96 and new['max_distinct_images']==192
    assert new['camera_sharing']=='single'
    assert set(new['runtime_thread_environment'].values())=={'2'}
    assert profiles.contract_version(VIDEO)=='sfm-native-object-video-v1'
    assert profiles.camera_mode(VIDEO)=='SINGLE'
    assert point_selection.selection_policy('rcover-v1')['max_images']==96
    assert point_selection.selection_policy('rcover-v1',profile=VIDEO)['max_images']==192
    from kozlov_lab.reconstruction import cli
    assert cli.TOTAL_BUDGET_SECONDS == 600
    assert old != new
    # The fingerprint directly binds both contract and effective options.
    fp=lambda opts,contract:adapter.input_fingerprint(source,source.rows,opts,{},'fixture',contract_version=contract)
    assert fp(old,profiles.contract_version(profiles.PROFILE_CAD_OBJECT_V1)) != fp(new,profiles.contract_version(VIDEO))

def many_masks(tmp_path,count):
    fixture=_make_input(tmp_path)
    original=copy.deepcopy(fixture.upstream[0])
    fixture.upstream=[]
    for i in range(count):
        row=copy.deepcopy(original);row['input_index']=i;row['frame_id']='frame'+str(i)
        fixture.upstream.append(row)
    run=json.loads((fixture.io/'run.json').read_text());run.update(input_count=count,ok_count=count)
    _write_json(fixture.io/'run.json',run)
    (fixture.io/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fixture.upstream))
    fixture.document['frames']=[fixture.frame(i) for i in range(count)]
    fixture.bind_source();fixture.recount();fixture.save()
    return fixture

def test_review_192_requires_explicit_limits_and_193_refused(tmp_path):
    fixture=many_masks(tmp_path,192)
    with pytest.raises(ValueError,match='manifest frames'):
        review.load_inputs([fixture.manifest],review.Bindings())
    frames,_,_=review.load_inputs([fixture.manifest],review.Bindings(),limits_profile='whole-video-192-v1')
    assert len(frames)==192
    fixture.document['frames'].append(copy.deepcopy(fixture.document['frames'][0]));fixture.recount();fixture.save()
    with pytest.raises(ValueError,match='manifest frames'):
        review.load_inputs([fixture.manifest],review.Bindings(),limits_profile='whole-video-192-v1')

def test_review_default_pixel_bounds_unchanged(tmp_path,monkeypatch):
    fixture=_make_input(tmp_path)
    monkeypatch.setattr(review,'MAX_TOTAL_SOURCE_PIXELS',24)
    with pytest.raises(ValueError,match='source pixel total'):
        review.load_inputs([fixture.manifest],review.Bindings())
    assert len(review.load_inputs([fixture.manifest],review.Bindings(),limits_profile='whole-video-192-v1')[0])==1
    assert review.MAX_TOTAL_MASK_PIXELS==64*1024**2 and review.MAX_TOTAL_BYTES==256*1024**2

def test_rcover_executes_192_bound_images_in_sqlite(tmp_path):
    import sqlite3
    from test_reconstruction_point_selection import fixture
    db,bindings,evidence=fixture(tmp_path)
    original=bindings[0]
    with sqlite3.connect(db) as c:
        for i in range(2,193):
            name=f'image{i}.png'
            c.execute('INSERT INTO images VALUES(?,?,1)',(i,name))
            c.execute('INSERT INTO keypoints SELECT ?,rows,cols,data FROM keypoints WHERE image_id=1',(i,))
            c.execute('INSERT INTO descriptors SELECT ?,rows,cols,data,type FROM descriptors WHERE image_id=1',(i,))
            bindings.append(dict(original,image_name=name))
    with pytest.raises(ValueError,match='capacity'):
        point_selection.apply_point_selection(db,bindings,evidence)
    # Count rejection creates no evidence; the same immutable DB is then valid for the new profile.
    result=point_selection.apply_point_selection(db,bindings,evidence,profile=VIDEO)
    assert result['status']=='VERIFIED' and len(result['per_image'])==192
    assert result['before_total']==576 and result['after_total']==384

def test_cli_new_profile_and_rcover_pass_explicit_profile_to_foreground(tmp_path,monkeypatch):
    from kozlov_lab.reconstruction import cli,foreground
    from test_colmap_adapter import make_upstream_run
    source=make_upstream_run(tmp_path/'input')
    for key in profiles.RUNTIME_THREAD_ENV:monkeypatch.setenv(key,'2')
    called=[]
    def select(upstream,path,out,*,profile):
        called.append(profile)
        raise ValueError('intentional foreground stop before native')
    monkeypatch.setattr(foreground,'select_foreground',select)
    monkeypatch.setattr(cli,'backend_info',lambda *a:pytest.fail('native backend must not be used'))
    out=tmp_path/'out';out.mkdir()
    assert cli.run_reconstruction(io_run=str(source),run_dir=out,project_root=tmp_path,
        profile=VIDEO,point_selection='rcover-v1',object_mask_manifest=tmp_path/'unused.json')==2
    assert called==[VIDEO]
    rec=json.loads((out/'run.json').read_text())
    assert rec['point_selection']['requested']=='rcover-v1'
    assert rec['effective_options']['max_distinct_images']==192
    assert 'intentional foreground stop' in rec['errors'][0]['message']
