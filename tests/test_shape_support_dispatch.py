"""TEST_DATA: shape-support controller gates; every native launch is mocked."""
import json
import sys
from types import SimpleNamespace

import pytest

from integrations.video_masks import region_surface as surface
from test_region_surface_pipeline import controller


STATUS = 'NO_FLAGGED_REGRESSION_NOT_QUALITY_ACCEPTANCE'


@pytest.fixture
def ready(controller, monkeypatch):
    controller.options.update(fail=None)
    from integrations.viewer import export
    def viewer(spec, destination, profile):
        destination.write_text('TEST_DATA raw viewer')
        return dict(TEST_DATA=True)
    monkeypatch.setattr(export, 'export_html', viewer)
    return controller


def record(source, candidate=None, **changes):
    value=dict(schema='shape-support-audit-v1',status=STATUS,source_unchanged=True,
               geometry_changed=False,candidate_promotable=candidate is not None,
               source_surface_run=str(source.resolve()),
               candidate_run=str(candidate.resolve()) if candidate is not None else None)
    value.update(changes)
    return value


def test_audit_is_mandatory_and_only_gate_can_promote(ready, monkeypatch):
    invoke=surface._invoke
    snapshots=[]
    def guarded(*args, **kwargs):
        if args[2] in ('face_local','boundary_geometry','shape_support'):
            snapshots.append(surface.read(ready.out/'RESULT.json'))
        return invoke(*args, **kwargs)
    monkeypatch.setattr(surface,'_invoke',guarded)
    result=surface.run(ready.source,ready.out)
    assert len([c for c in ready.calls if c['name']=='shape_support'])==1
    audit=ready.calls[-1]
    assert audit['name']=='shape_support'
    assert audit['seconds']==600 and audit['memory']==4<<30
    assert audit['argv'][4:]==[str(ready.root/'scripts/build_region_surface.py'),'shape-support-worker',
        '--source',str(ready.out),'--output',str(ready.out/'shape_support'),
        '--candidate-run',str(ready.out/'face_local')]
    assert all(s['preferred_viewer']==str(ready.out/'mesh_viewer.html') for s in snapshots)
    assert all(s['candidate_promoted'] is False for s in snapshots)
    assert result['preferred_viewer']==str(ready.out/'face_local/retained_viewer.html')
    assert result['candidate_promoted'] is True and result['geometry_accepted'] is False
    assert result['shape_support_status']==STATUS
    plan=surface.read(ready.out/'PLAN.json')
    assert plan['automatic_shape_support'] is True
    assert plan['shape_support_seconds']==600 and plan['shape_support_memory_bytes']==4<<30
    bindings=json.loads((ready.out/'STAGE_BINDINGS.json').read_text(encoding='utf8'))
    assert any(x['path']==str(ready.out/'shape_support/RESULT.json') for x in bindings)
    html=(ready.out/'index.html').read_text(encoding='utf8')
    assert 'shape_support/index.html' in html and STATUS in html


def test_review_required_retains_raw_and_all_candidate_artifacts(ready):
    ready.options['shape_record']=dict(status='REVIEW_REQUIRED',candidate_promotable=False)
    result=surface.run(ready.source,ready.out)
    assert result['shape_support_status']=='REVIEW_REQUIRED'
    assert result['preferred_viewer']==str(ready.out/'mesh_viewer.html')
    assert result['candidate_promoted'] is False and result['warnings']
    assert result['geometry_accepted'] is False
    assert (ready.out/'surface/work/__mvs_surface_raw.ply').read_bytes()==b'TEST_DATA never parsed'
    for name in ('retained.ply','quarantined.ply','LINEAGE.npz'):
        assert (ready.out/'face_local'/name).read_text()=='TEST_DATA'
    assert result['artifacts']['单独修正版候选（未验收）']==str(ready.out/'face_local/retained_viewer.html')


@pytest.mark.parametrize('changes',[
    {'schema':'wrong'},{'status':'PASS'},{'status':'UNKNOWN'},
    {'source_unchanged':False},{'geometry_changed':True},
    {'candidate_promotable':1},{'candidate_promotable':None},
    {'source_surface_run':'unrelated'},{'candidate_run':None},{'candidate_run':'unrelated'},
    {'status':'REVIEW_REQUIRED','candidate_promotable':True},
    {'error':'producer failed','candidate_promotable':True},
    {'drift_error':'bound source changed','candidate_promotable':True},
])
def test_missing_or_invalid_evidence_never_promotes(ready,changes):
    ready.options['shape_record']=changes
    result=surface.run(ready.source,ready.out)
    assert result['status']=='PARTIAL_OR_FAILED'
    assert result['shape_support_status']=='FAILED'
    assert result['preferred_viewer']==str(ready.out/'mesh_viewer.html')
    assert result['candidate_promoted'] is False
    assert result['stages'][-1]['status']=='BUSINESS_GATE_FAILED'
    assert result['surface_result']['status']=='SURFACE_GENERATED_NOT_ACCEPTED'


@pytest.mark.parametrize('failure',['process','business','report','missing_result'])
def test_audit_failure_keeps_raw_without_claiming_completion(ready,monkeypatch,failure):
    if failure=='report':ready.options['shape_missing_report']=True
    elif failure=='missing_result':
        invoke=surface._invoke
        def without_result(*args,**kwargs):
            if args[2]=='shape_support':
                ready.calls.append(dict(name='shape_support'))
                return dict(status='completed',root_exit_code=0,reclaim_confirmed=True,errors=[],log_errors=[])
            return invoke(*args,**kwargs)
        monkeypatch.setattr(surface,'_invoke',without_result)
    else:ready.options.update(fail='shape_support',kind=failure)
    result=surface.run(ready.source,ready.out)
    assert result['status']=='PARTIAL_OR_FAILED'
    assert result['shape_support_status']=='FAILED'
    assert result['preferred_viewer']==str(ready.out/'mesh_viewer.html')
    assert result['candidate_promoted'] is False and result['warnings']


@pytest.mark.parametrize('failed_stage',['surface_support','face_mask_gate','face_local','boundary_geometry'])
@pytest.mark.parametrize('audit_fails',[False,True])
def test_secondary_audit_preserves_first_failure_and_never_promotes(ready,failed_stage,audit_fails):
    ready.options.update(fail=failed_stage,kind='process')
    if audit_fails:ready.options['shape_record']={'status':'UNKNOWN'}
    result=surface.run(ready.source,ready.out)
    assert result['error']=='RuntimeError: '+failed_stage+' execution failed; see preserved logs'
    assert result['status']=='PARTIAL_OR_FAILED'
    assert result['preferred_viewer']==str(ready.out/'mesh_viewer.html')
    assert result['candidate_promoted'] is False
    assert next(s for s in result['stages'] if s['name']==failed_stage)['status']=='FAILED'
    assert ready.calls[-1]['name']=='shape_support'
    assert ('--candidate-run' in ready.calls[-1]['argv'])==(failed_stage=='boundary_geometry')
    if failed_stage!='boundary_geometry':
        assert surface.read(ready.out/'shape_support/RESULT.json')['candidate_run'] is None
    if audit_fails:assert result['shape_support_error']


def test_failure_before_surface_does_not_start_audit(ready):
    ready.options.update(fail='surface',kind='process')
    result=surface.run(ready.source,ready.out)
    assert not any(c['name']=='shape_support' for c in ready.calls)
    assert result['shape_support_status']=='NOT_EVALUATED'
    assert result['candidate_promoted'] is False


@pytest.mark.parametrize('has_candidate',[False,True])
@pytest.mark.parametrize('status',['REVIEW_REQUIRED',STATUS])
def test_worker_dispatch_is_readonly_and_accepts_review_result(tmp_path,monkeypatch,has_candidate,status):
    source,out=tmp_path/'source',tmp_path/'audit'
    candidate=tmp_path/'candidate' if has_candidate else None
    out.mkdir();(out/'index.html').write_text('TEST_DATA')
    calls=[]
    def run(*args,**kwargs):
        calls.append((args,kwargs))
        return record(source,candidate,status=status,candidate_promotable=False)
    monkeypatch.setitem(sys.modules,'integrations.demo_pipeline.shape_support_audit',SimpleNamespace(run=run))
    monkeypatch.setattr(surface,'run',lambda *a:pytest.fail('native pipeline must not run'))
    args=['shape-support-worker','--source',str(source),'--output',str(out)]
    if candidate is not None:args+=['--candidate-run',str(candidate)]
    assert surface.main(args)==0
    assert calls==[((source,out),{'candidate_run':candidate})]


def test_worker_requires_explicit_null_candidate_and_cannot_promote_none(tmp_path,monkeypatch):
    out=tmp_path/'audit';out.mkdir();(out/'index.html').write_text('TEST_DATA')
    source=tmp_path/'source'
    invalid=record(source,candidate_promotable=True)
    monkeypatch.setitem(sys.modules,'integrations.demo_pipeline.shape_support_audit',SimpleNamespace(run=lambda *a,**k:invalid))
    assert surface.main(['shape-support-worker','--source',str(source),'--output',str(out)])==1
    invalid['candidate_promotable']=False
    del invalid['candidate_run']
    assert surface.main(['shape-support-worker','--source',str(source),'--output',str(out)])==1


def test_kernel_is_bound_by_frozen_plan(tmp_path,monkeypatch):
    monkeypatch.setattr(surface,'ROOT',tmp_path)
    monkeypatch.setattr(surface,'binding',lambda p:{'path':str(p)})
    paths={r['path'] for r in surface.code_bindings()}
    for name in ('shape_support_audit.py','silhouette_review.py'):
        assert str(tmp_path/'integrations/demo_pipeline'/name) in paths
