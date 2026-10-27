"""Recovery-specific continuation gates using only synthetic local receipts."""
import json
from pathlib import Path
import pytest
from test_demo_resume import source, commit
from integrations.demo_pipeline import resume as r
from integrations.demo_pipeline.surface import save


def camera_receipt(source):
    root, run, out, tools, meta = source
    sfm_file=Path(meta['stages']['sfm']['run_json'])
    sfm=json.loads(sfm_file.read_text());sfm['contract_version']='sfm-native-object-pinhole-v1';save(sfm_file,sfm)
    binary=sfm_file.parent/'models/0/binary'
    refs={p.name:{k:r._binding(p)[k] for k in ('bytes','sha256')} for p in binary.iterdir()}
    audit=dict(status='CAMERA_DOMAIN_VALID',source_model=str(binary),source_unchanged=True,
               source_bindings_before=refs,source_bindings_after=refs)
    audit_path=run/'camera_recovery/original.json';audit_path.parent.mkdir();save(audit_path,audit)
    stage=dict(meta['stages']['sfm'])
    entry_path=run/'logs/camera_domain_original/ENTRY.json';entry_path.parent.mkdir()
    entry=json.loads(Path(stage['entry_path']).read_text());entry['argv']=['fixture','audit'];save(entry_path,entry)
    meta['stages']['camera_domain_original']=dict(status='EXECUTED',argv=entry['argv'],entry_path=str(entry_path),exit_code=0,reclaim_confirmed=True)
    detail=dict(status='ORIGINAL_DOMAIN_VALID',surface_ready=True,original_sfm=str(sfm_file),original_sfm_sha256=r._binding(sfm_file)['sha256'],
        selected_sfm=str(sfm_file),selected_sfm_sha256=r._binding(sfm_file)['sha256'],
        audits=[dict(label='original',path=str(audit_path),sha256=r._binding(audit_path)['sha256'],exit_code=0,result=audit)])
    meta['automatic_camera_recovery']=detail;save(run/'camera_recovery/RESULT.json',detail);save(run/'pipeline.json',meta)
    return detail, audit_path


def test_pinhole_checkpoint_and_domain_receipts_are_bound(source):
    camera_receipt(source)
    result=commit(source)
    paths={row['path'] for row in result['bindings']}
    assert str(source[1]/'camera_recovery/RESULT.json') in paths
    assert str(source[1]/'camera_recovery/original.json') in paths


@pytest.mark.parametrize('mutation',['invalid','wrong_component','drift','missing_native_entry'])
def test_invalid_or_unbound_camera_refused(source,mutation):
    detail,path=camera_receipt(source)
    root,run,out,tools,meta=source
    if mutation=='invalid':detail['surface_ready']=False
    elif mutation=='wrong_component':detail['audits'][0]['result']['source_model']=str(run)
    elif mutation=='drift':
        binary=Path(detail['audits'][0]['result']['source_model']);(binary/'cameras.bin').write_bytes(b'changed')
    else:meta['stages']['camera_domain_original'].pop('entry_path')
    if mutation=='wrong_component':
        save(path,detail['audits'][0]['result']);detail['audits'][0]['sha256']=r._binding(path)['sha256']
    save(run/'camera_recovery/RESULT.json',detail);save(run/'pipeline.json',meta)
    with pytest.raises(ValueError):r.write_resume_checkpoint(run,root=root)


def test_aggregation_resume_explicitly_refused(source):
    root,run,out,tools,meta=source
    meta['stages']['segmentation']=dict(status='SUCCESS',exit_code=0,run_dir='derived',counts={},
        execution_stages=['segmentation_primary','segmentation_recovery'],composition='derived')
    save(run/'pipeline.json',meta)
    with pytest.raises(ValueError,match='aggregation segmentation resume is unsupported'):
        r.write_resume_checkpoint(run,root=root)

def test_constrained_selected_sfm_requires_matching_execution_and_receipt(source):
    detail,path=camera_receipt(source)
    root,run,out,tools,meta=source
    sfm_file=Path(detail['selected_sfm'])
    primary=run/'camera_recovery/primary/run.json';primary.parent.mkdir();save(primary,{'retained':'primary'})
    detail.update(status='CONSTRAINED_DOMAIN_VALID',original_sfm=str(primary),original_sfm_sha256=r._binding(primary)['sha256'],
        candidate_sfm=str(sfm_file),candidate_sfm_sha256=r._binding(sfm_file)['sha256'])
    item=detail['audits'][0];item['label']='candidate'
    candidate_audit=path.with_name('candidate.json');path.rename(candidate_audit);item['path']=str(candidate_audit)
    meta['stages']['camera_domain_candidate']=meta['stages'].pop('camera_domain_original')
    meta['stages']['sfm']['selected_from']='sfm_constrained'
    save(run/'camera_recovery/RESULT.json',detail);save(run/'pipeline.json',meta)
    assert commit(source)['status']=='READY'
