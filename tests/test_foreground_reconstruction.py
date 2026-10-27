"""Object-first staging/actual reader and SQLite keypoint-consumption gates."""
import json
import sqlite3
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from kozlov_lab.reconstruction import colmap_adapter as adapter,cli,foreground
from test_colmap_adapter import make_upstream_run


def selected_masks(root,indices):
    masks={}
    for i in indices:
        path=root/f'mask{i}.png'; image=Image.new('L',(16,12),0)
        image.putpixel((2,3),255);image.save(path)
        masks[i]=path
    return masks


def staged_fixture(tmp_path):
    source=make_upstream_run(tmp_path/'input',entries=[dict(value=v,mask=None) for v in [10,20,30]])
    upstream=adapter.load_upstream_run(source)
    masks=selected_masks(tmp_path,[0,2])
    staged=adapter.stage_inputs(upstream,tmp_path/'native',masks_by_index=masks)
    return upstream,staged


def database(path,bindings,xy=(2.9,3.8)):
    connection=sqlite3.connect(path)
    connection.execute('CREATE TABLE images(image_id INTEGER,name TEXT)')
    connection.execute('CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB)')
    for i,b in enumerate(bindings,1):
        connection.execute('INSERT INTO images VALUES (?,?)',(i,b['image_name']))
        connection.execute('INSERT INTO keypoints VALUES (?,1,4,?)',(i,struct.pack('<ffff',*xy,1.,0.)))
    connection.commit();connection.close()


def test_selected_only_native_mask_names_and_full_identity(tmp_path):
    upstream,staged=staged_fixture(tmp_path)
    assert len(staged.rows)==3 and [r.input_index for r in staged.distinct]==[0,2]
    for b in staged.mask_bindings:
        assert Path(b['staged_path']).relative_to(staged.masks_dir).as_posix()==b['image_name']+'.png'
    for row in staged.distinct:
        assert (staged.images_dir/Path(row.staged_image_name).name).read_bytes()==(upstream.run_dir/row.image_output).read_bytes()
    frames=[];cli._init_frames_map(upstream,staged,frames)
    cli._fill_registrations(upstream,staged,{},frames)
    assert [r['input_index'] for r in frames]==[0,1,2]
    assert frames[1]['status']=='EXCLUDED_FOREGROUND'
    assert frames[1]['representative_input_index'] is None and frames[1]['database_image_id'] is None


def test_mask_db_rule_truncates_not_rounds_and_is_read_only(tmp_path):
    _,staged=staged_fixture(tmp_path);db=tmp_path/'database.db'
    database(db,staged.mask_bindings)
    before=db.read_bytes()
    audit=adapter.audit_foreground_keypoints(db,staged.mask_bindings)
    assert audit['checked_images']==2 and audit['checked_keypoints']==2 and audit['violations']==0
    assert db.read_bytes()==before


def test_mask_db_bad_point_is_backend_error(tmp_path):
    _,staged=staged_fixture(tmp_path);db=tmp_path/'database.db'
    database(db,staged.mask_bindings,xy=(3.1,3.0))
    with pytest.raises(adapter.BackendError,match='out-of-mask'):
        adapter.audit_foreground_keypoints(db,staged.mask_bindings)


def test_mask_drift_fails_before_feature_call(tmp_path):
    _,staged=staged_fixture(tmp_path)
    Path(staged.mask_bindings[0]['staged_path']).write_bytes(b'changed')
    with pytest.raises(adapter.BackendError,match='binding drift'):
        adapter.verify_foreground_masks(staged.mask_bindings)


def test_actual_reader_received_mask_root(tmp_path):
    _,staged=staged_fixture(tmp_path);captured={}
    fake=SimpleNamespace(FeatureExtractionOptions=lambda:SimpleNamespace(sift=SimpleNamespace()),
        ImageReaderOptions=lambda:SimpleNamespace(),CameraMode=SimpleNamespace(AUTO='auto'),Device=SimpleNamespace(cpu='cpu'))
    def extract(*args,**kwargs):
        captured.update(kwargs);raise RuntimeError('captured without native execution')
    fake.extract_features=extract
    spec=dict(database_path=str(tmp_path/'db'),image_path=str(tmp_path),image_names=[b['image_name'] for b in staged.mask_bindings],
        num_threads=2,max_long_side=1024,max_sift_features=4096,camera_model='SIMPLE_RADIAL',camera_mode='AUTO',focal_length_factor=1.2,
        foreground_mask_path=str(staged.masks_dir),foreground_mask_bindings=staged.mask_bindings)
    with pytest.raises(RuntimeError,match='captured'):adapter._stage_extract(fake,spec)
    assert captured['reader_options'].mask_path==staged.masks_dir


def test_insufficient_selection_keeps_original_rows_and_no_backend(tmp_path,monkeypatch):
    source=make_upstream_run(tmp_path/'input');masks=selected_masks(tmp_path,[0]);manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    def selection(upstream,path,out):
        out.mkdir();ledger=out/'ledger.json';ledger.write_text('{}')
        return dict(selected_indices=[0],masks_by_index=masks,ledger_path=ledger,manifest_sha256=adapter.sha256_file(manifest),coverage={})
    monkeypatch.setattr(foreground,'select_foreground',selection)
    monkeypatch.setattr(cli,'backend_info',lambda *a:pytest.fail('backend must not start'))
    out=tmp_path/'out';out.mkdir()
    assert cli.run_reconstruction(io_run=str(source),run_dir=out,project_root=tmp_path,object_mask_manifest=manifest)==2
    report=json.loads((out/'run.json').read_text());frames=[json.loads(x) for x in (out/'frames_map.jsonl').read_text().splitlines()]
    assert report['status']=='INPUT_ERROR' and report['counts']['input_rows']==3
    assert len(frames)==3 and frames[1]['status']=='EXCLUDED_FOREGROUND' and frames[2]['status']=='EXCLUDED_FOREGROUND'
    assert not report['foreground']['mask_consumption_verified']


def test_explicit_mask_source_hash_changes_fingerprint(tmp_path):
    upstream,staged=staged_fixture(tmp_path);options=adapter.effective_options()
    options['foreground']=dict(requested=True,manifest_sha256='a'*64,mask_bindings=staged.mask_bindings)
    original=adapter.input_fingerprint(upstream,staged.rows,options,{},'code')
    options['foreground']['manifest_sha256']='b'*64
    assert adapter.input_fingerprint(upstream,staged.rows,options,{},'code')!=original


def test_empty_selected_mask_is_input_error(tmp_path):
    source=make_upstream_run(tmp_path/'input');mask=tmp_path/'empty.png';Image.new('L',(16,12),0).save(mask)
    with pytest.raises(adapter.InputError,match='empty'):
        adapter.stage_inputs(adapter.load_upstream_run(source),tmp_path/'out',masks_by_index={0:mask})


def test_cli_accepts_object_mask_manifest():
    args=cli.build_parser().parse_args(['--io-run','source','--output','out','--object-mask-manifest','m.json'])
    assert args.object_mask_manifest=='m.json'


@pytest.mark.parametrize('consumption_verified',[False,True])
def test_object_spec_fingerprint_and_missing_consumption_refused(tmp_path,monkeypatch,consumption_verified):
    source=make_upstream_run(tmp_path/'input',entries=[dict(value=v,mask=None) for v in [10,20,30]])
    masks=selected_masks(tmp_path,[0,2]);manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    def selection(upstream,path,out):
        out.mkdir();ledger=out/'ledger.json';ledger.write_text('{}')
        return dict(selected_indices=[0,2],masks_by_index=masks,ledger_path=ledger,
            manifest_sha256=adapter.sha256_file(manifest),coverage={'selected':2})
    monkeypatch.setattr(foreground,'select_foreground',selection)
    monkeypatch.setattr(cli,'backend_info',lambda *a:dict(name='pycolmap',version=adapter.BACKEND_PIN,runtime='cpu'))
    captured={}
    def stage(name,spec,**kwargs):
        if name!='extract_features':
            return adapter.NativeStageResult(stage=name,started_utc='T',finished_utc='T',returncode=4,timed_out=False,
                stdout_path='',stderr_path='',result=None,error='intentional mock match stop; no native SfM')
        assert name=='extract_features';captured.update(spec)
        evidence={}
        if consumption_verified:
            database(Path(spec['database_path']),spec['foreground_mask_bindings'])
            evidence=dict(num_keypoints_total=2,foreground_mask_audit=adapter.audit_foreground_keypoints(spec['database_path'],spec['foreground_mask_bindings']))
        return adapter.NativeStageResult(stage=name,started_utc='T',finished_utc='T',returncode=0,timed_out=False,
            stdout_path='',stderr_path='',result=dict(num_images=2,cameras={'1':dict(width=16,height=12)},
            image_names_to_ids={'images/000000.png':1,'images/000002.png':2},
            actual_options={'reader':{'mask_path':spec['foreground_mask_path']}},**evidence),error=None)
    monkeypatch.setattr(cli,'run_native_stage',stage)
    out=tmp_path/'out';out.mkdir()
    assert cli.run_reconstruction(io_run=str(source),run_dir=out,project_root=tmp_path,object_mask_manifest=manifest)==4
    report=json.loads((out/'run.json').read_text())
    assert captured['image_names']==['images/000000.png','images/000002.png']
    assert captured['foreground_manifest_sha256']==adapter.sha256_file(manifest)
    assert len(captured['foreground_mask_bindings'])==2
    assert report['input_fingerprint_before_native'] and report['effective_options']['foreground']['requested']
    assert report['foreground']['mask_consumption_verified']==consumption_verified
    assert report['mask_usage']==('foreground_verified' if consumption_verified else 'foreground_requested_not_verified')
    assert report['reconstruction_scope']==('foreground_feature_centres' if consumption_verified else 'foreground_requested_not_verified')
    assert len((out/'frames_map.jsonl').read_text().splitlines())==3


def test_identical_rgb_conflicting_object_masks_refused(tmp_path):
    source=make_upstream_run(tmp_path/'input',entries=[dict(value=v,mask=None) for v in [10,10,20]])
    masks=selected_masks(tmp_path,[0,1,2]);image=Image.new('L',(16,12),0);image.putpixel((3,3),255);image.save(masks[1])
    with pytest.raises(adapter.InputError,match='conflicting foreground masks'):
        adapter.stage_inputs(adapter.load_upstream_run(source),tmp_path/'out',masks_by_index=masks)


def test_bad_foreground_manifest_is_input_error_not_internal(tmp_path,monkeypatch):
    source=make_upstream_run(tmp_path/'input')
    def invalid(*args):raise ValueError('identity mismatch')
    monkeypatch.setattr(foreground,'select_foreground',invalid)
    out=tmp_path/'out';out.mkdir()
    assert cli.run_reconstruction(io_run=str(source),run_dir=out,project_root=tmp_path,object_mask_manifest='bad.json')==2
    report=json.loads((out/'run.json').read_text())
    assert report['errors'][0]['code']=='INPUT_ERROR'
    assert report['foreground']['contract_version']=='sfm-foreground-v1'
    assert len((out/'frames_map.jsonl').read_text().splitlines())==3
