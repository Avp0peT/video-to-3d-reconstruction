"""Unstaged object-first exclusions remain identities, never native views."""
import importlib.util
import json
from pathlib import Path
import pytest

from test_mvs_mask_bridge import world,cli,mio,cli_args,ROOT


def excluded_world(world):
    path=world['sfm']/'run.json';run=json.loads(path.read_text())
    run['foreground']=dict(contract_version='sfm-foreground-v1',requested=True,mask_consumption_verified=True)
    path.write_text(json.dumps(run))
    path=world['sfm']/'frames_map.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
    for row in rows[len(world['names']):]:
        row.update(staged_image_name=None,status='EXCLUDED_FOREGROUND',foreground_selected=False,
            database_image_id=None,representative_input_index=None,registrations=[],reason='SAM3 NO_MATCH')
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    return rows


def index(world,names=None):
    return cli.frame_index(cli.bridge_artifacts(world['bridge']),world['sfm'],world['t01'],names or world['names'])


def rewrite_rows(world,rows):
    (world['sfm']/'frames_map.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')


def test_old_consumer_refuses_null_new_consumer_preserves_identity(world):
    rows=excluded_world(world)
    frozen=ROOT/'outputs/delegation/PPP_VIDEO/20261014_object_first01/compatibility/original/prepare_mvs_masks.py'
    spec=importlib.util.spec_from_file_location('old_prepare_mvs_masks_exclusion_fixture',frozen)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    with pytest.raises(mio.InputError,match='staged_image_name'):
        old.frame_index(old.bridge_artifacts(world['bridge']),world['sfm'],world['t01'],world['names'])
    result=index(world)
    assert result['frames_map_rows']==4 and result['selected_rows']==2
    assert [r['input_index'] for r in result['excluded_foreground_rows']]==[2,3]
    assert len(result['map_by_key'])==len(rows)
    assert set(result['by_staged'])==set(world['names'])


def test_complete_object_first_consumer_without_native_mvs(tmp_path,world):
    excluded_world(world);out=tmp_path/'consumer'
    assert cli.main(cli_args(world,world['manifest'],out))==0
    result=json.loads((out/'manifest.json').read_text())
    assert result['selection_summary']['frames_map_rows_unselected']==2
    assert len(result['selection_summary']['excluded_foreground_rows'])==2


@pytest.mark.parametrize('field,value',[
    ('status','UNREGISTERED'),('foreground_selected',True),('database_image_id',8),
    ('representative_input_index',2),('registrations',[{'model_id':'0','image_id':3}]),
    ('staged_image_name','images/000002.png'),('frame_id','wrong'),('upstream_io_run_id','wrong'),
    ('upstream_image_output','images/000000.png'),('canonical_image_sha256','f'*64)])
def test_malformed_exclusion_never_silently_skipped(world,field,value):
    rows=excluded_world(world);rows[2][field]=value;rewrite_rows(world,rows)
    with pytest.raises(mio.InputError):index(world)


@pytest.mark.parametrize('field,value',[
    ('contract_version','unknown'),('requested',False),('mask_consumption_verified',False)])
def test_no_verified_foreground_contract_no_null_name_exception(world,field,value):
    excluded_world(world);path=world['sfm']/'run.json';run=json.loads(path.read_text())
    run['foreground'][field]=value;path.write_text(json.dumps(run))
    with pytest.raises(mio.InputError,match='invalid unstaged'):index(world)


def test_excluded_identity_cannot_be_selected_native_scene_view(world):
    excluded_world(world)
    with pytest.raises(mio.InputError,match='missing from frames_map'):
        index(world,world['names']+['images/000002.png'])


def test_missing_original_row_coverage_refused(world):
    rows=excluded_world(world);rewrite_rows(world,rows[:-1])
    with pytest.raises(mio.InputError,match='identity coverage'):index(world)


@pytest.mark.parametrize('field',['staged_image_name','foreground_selected','database_image_id','representative_input_index','registrations'])
def test_missing_exclusion_identity_field_is_not_none(world,field):
    rows=excluded_world(world);del rows[2][field];rewrite_rows(world,rows)
    with pytest.raises(mio.InputError,match='invalid unstaged'):index(world)
