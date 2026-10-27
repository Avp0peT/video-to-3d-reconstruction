"""Pinned reusable text profile; no model/device/native or weight-read operations."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.demo_pipeline import pipeline, runtime as rt

EXPECTED_PROFILE='outputs/delegation/DEMO_RUNTIME/20261011_text_v2'
EXPECTED_SHA='6704d9a2331bb6c82500bd0fa58d276e21188cc407c84296539b60c76d36f843'
OLD_SHA='16f263fc809fc29d823a08744c0432f934582dc082a6d1e633da20e7399e3fd6'
CHANGED={'integrations/sam3/sam3_adapter/'+n for n in ('runner.py','worker.py','backends.py')}
NEW='integrations/sam3/sam3_adapter/visual_prompt.py'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def metadata_config(tmp_path):
    # The config is assembled from actual pinned metadata. No real preflight is called.
    profile=ROOT/EXPECTED_PROFILE
    return rt.SegmentConfig(root=ROOT,io_run=tmp_path,output_dir=tmp_path,
        checkpoint=Path(r'C:\sam3\sam3.pt'),checkpoint_sha256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',
        worker_package=profile,sam3_python=ROOT/'integrations/sam3/env/Scripts/python.exe',
        text_prompt='a new photographed object',object_id='new-object',indices=[1,9,3])

def test_real_default_plan_selects_accepted_profile_without_creating_or_launching(tmp_path,monkeypatch):
    photos=tmp_path/'photos';photos.mkdir()
    for n in ('0.jpg','1.png'):(photos/n).write_bytes(b'PLAN_FIXTURE_NOT_A_PHOTO')
    result=pipeline.plan(pipeline.Options(source=photos,prompt='new object',output=ROOT/'outputs/NEVER_CREATED_BY_PLAN'))
    assert pipeline.PACKAGE_REL==rt.PACKAGE_RELATIVE==EXPECTED_PROFILE
    assert result['sam3_package']==str((ROOT/EXPECTED_PROFILE).resolve())
    assert result['dependencies']['sam3_package'] is True
    assert result['stages']==['input','sfm','segmentation','viewer','surface','blender','research','report']
    assert result['pointcloud_scope']=='scene_sparse; segmentation is displayed separately; no object-only filtering'
    assert not (ROOT/'outputs/NEVER_CREATED_BY_PLAN').exists()
    assert rt.BUNDLE_SHA256==EXPECTED_SHA

def test_accepted_profile_current_dependencies_all_match_without_weights(tmp_path,monkeypatch):
    config=metadata_config(tmp_path);original=rt.sha256
    def safe_sha(path):
        assert Path(path).resolve()!=config.checkpoint.resolve(),'must not rehash weights'
        return original(path)
    monkeypatch.setattr(rt,'sha256',safe_sha)
    report=rt.verify_bindings(config)
    assert report['match'] and report['errors']==[] and report['bundle_sha256']==EXPECTED_SHA
    for name in CHANGED|{NEW}:
        rows=[r for r in report['checked'] if r['path'].replace('\\','/')==name]
        assert len(rows)==2 and all(r['match'] for r in rows)
    assert any(r['path'].endswith('memory_diag_worker.py') for r in report['checked'])
    assert not {'torch','sam3'}.intersection(sys.modules)

def test_only_reviewed_hash_differences_and_four_original_workers(tmp_path):
    profile=ROOT/EXPECTED_PROFILE
    origin=read(profile/'PROFILE.json')
    assert origin['source_bundle_sha256']==OLD_SHA and origin['mode']=='text_only'
    assert {r['path'] for r in origin['approved_rebindings'] if r['previous'] is not None}==CHANGED
    assert {r['path'] for r in origin['approved_rebindings'] if r['previous'] is None}=={NEW}
    assert len(origin['approved_rebindings'])==8
    for row in origin['workers']:
        assert (profile/row['path']).read_bytes()==(ROOT/'outputs/delegation/SAM3_MMAP/20261008_execute_v1'/row['path']).read_bytes()
    assert len(origin['workers'])==4 and origin['text_config_sha256']=='43c793b8de82a93d145d27bd5619431937c727de748b84b7a77aebcbf8df3e90'
    assert not any((profile/n).exists() for n in ('REAL_ENTRY.json','DRIVER_CLAIM.json','run_once.py','execute_pilot.ps1'))

@pytest.mark.parametrize('name', ['runner.py','backends.py','worker.py','visual_prompt.py'])
def test_current_product_drift_still_refused_without_mutation(tmp_path,monkeypatch,name):
    config=metadata_config(tmp_path);target=(ROOT/'integrations/sam3/sam3_adapter'/name).resolve();original=rt.sha256
    monkeypatch.setattr(rt,'sha256',lambda p:'0'*64 if Path(p).resolve()==target else original(p))
    report=rt.verify_bindings(config)
    assert not report['match']
    assert any(name in error for error in report['errors'])

def test_wrong_profile_digest_and_wrong_origin_manifest_rejected(tmp_path,monkeypatch):
    config=metadata_config(tmp_path);original=rt.sha256
    monkeypatch.setattr(rt,'sha256',lambda p:OLD_SHA if Path(p).name=='BUNDLE.json' else original(p))
    with pytest.raises(ValueError,match='digest differs'):rt.verify_bindings(config)
    monkeypatch.setattr(rt,'sha256',lambda p:'0'*64 if Path(p).name=='PROFILE.json' else original(p))
    result=rt.verify_bindings(config)
    assert not result['match'] and any('PROFILE.json' in e for e in result['errors'])

def test_arbitrary_new_or_old_package_and_box_config_rejected(tmp_path):
    base=read(ROOT/EXPECTED_PROFILE/'START_BASELINE.json')
    output=ROOT/'outputs/delegation/DEMO_RUNTIME/20261011_text_v2' # path exists, used only for config validation
    original=dict(root=str(ROOT),io_run=str(ROOT/'outputs'),output_dir=str(ROOT/'outputs/delegation/PROJECT_FINISH'),
        checkpoint=base['checkpoint']['path'],checkpoint_sha256=base['checkpoint']['sha256'],
        worker_package=str(ROOT/EXPECTED_PROFILE),sam3_python=str(ROOT/'integrations/sam3/env/Scripts/python.exe'),
        text_prompt='new object',object_id='new-object',indices=[7])
    # Use independent existing IO/output directories so package rejection is tested first.
    for package in ('outputs/delegation/SAM3_MMAP/20261008_execute_v1','outputs/delegation/PROJECT_FINISH/20261011_box_execute_v1'):
        path=tmp_path/'config.json';path.write_text(json.dumps({**original,'worker_package':str(ROOT/package)}),encoding='utf-8')
        with pytest.raises(ValueError,match='accepted text_v2'):rt.load_config(path)
    path=tmp_path/'box.json';path.write_text(json.dumps({**original,'visual_prompt':{},'text_prompt':None}),encoding='utf-8')
    with pytest.raises(ValueError,match='exactly'):rt.load_config(path)

def test_mock_preflight_can_pass_current_profile_without_checkpoint_read(tmp_path,monkeypatch):
    config=metadata_config(tmp_path)
    base=read(config.worker_package/'START_BASELINE.json');authority=read(config.worker_package/'AUTHORITY_BINDINGS.json')
    expected=authority['checkpoint_stat_only'];original=rt.sha256;reads=[]
    def metadata_sha(p):
        if Path(p).resolve()==config.checkpoint.resolve():
            reads.append('frozen mock checkpoint SHA; no bytes read');return config.checkpoint_sha256
        if Path(p).name in ('run.json','frames.jsonl') and Path(p).parent==tmp_path:return 'a'*64
        return original(p)
    original_stat=Path.stat
    def stat(path,*args,**kwargs):
        if path==config.checkpoint:return SimpleNamespace(st_size=expected['bytes'],st_mtime_ns=expected['mtime_ns'])
        return original_stat(path,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',stat)
    monkeypatch.setattr(rt,'sha256',metadata_sha)
    monkeypatch.setattr(rt.sys,'executable',str(config.sam3_python))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(rt.metadata,'distributions',lambda **kw:[SimpleNamespace(metadata={'Name':p['name']},version=p['version']) for p in base['package_inventory']])
    monkeypatch.setattr(rt.shutil,'disk_usage',lambda p:SimpleNamespace(free=20<<30))
    def fake_git(argv,**kwargs):
        assert argv[0]==str(rt.discover_git(config.git_exe));return SimpleNamespace(stdout=base['checkout_head']+'\n' if 'rev-parse' in argv else '')
    monkeypatch.setattr(rt.subprocess,'run',fake_git)
    result=rt.preflight(config)
    assert result['status']=='PASS' and result['bindings']['match'] and len(reads)==1
    assert result['checkpoint_rehashed'] is True # runtime flag; this test explicitly supplied frozen metadata, not real hash evidence
