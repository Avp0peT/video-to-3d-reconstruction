"""TEST_DATA: controller/CLI dispatch; every native reader and launch is mocked."""
import json
import sys
from types import SimpleNamespace

import pytest

from integrations.video_masks import region_surface as surface
from test_region_surface_pipeline import controller


STATUS='BOUNDARY_GEOMETRY_DIAGNOSTIC_NOT_ACCEPTED'


@pytest.fixture
def ready(controller,monkeypatch):
    controller.options.update(fail=None)
    from integrations.viewer import export
    def viewer(spec,destination,profile):
        destination.write_text('TEST_DATA');return dict(TEST_DATA=True)
    monkeypatch.setattr(export,'export_html',viewer)
    return controller


def assert_preferred_unchanged(result,out):
    expected=str(out/'face_local/retained_viewer.html' if result['candidate_promoted'] else out/'mesh_viewer.html')
    assert result['preferred_viewer']==expected
    assert result['artifacts']['单独修正版候选（未验收）']==str(out/'face_local/retained_viewer.html')
    assert result['artifacts']['局部保留面候选 PLY（未验收）']==str(out/'face_local/retained.ply')
    assert result['geometry_accepted'] is False


def test_boundary_geometry_runs_after_local_validation_with_frozen_limits(ready):
    ready.options['boundary_record']={'preferred_viewer':'must-not-be-promoted'}
    result=surface.run(ready.source,ready.out)
    assert [x['name'] for x in ready.calls][-3:]==['face_local','boundary_geometry','shape_support']
    launch=ready.calls[-2]
    assert launch['seconds']==600 and launch['memory']==4<<30
    assert launch['argv'][4:]==[str(ready.root/'scripts/build_region_surface.py'),'boundary-geometry-worker',
        '--source',str(ready.out),'--gate',str(ready.out/'face_local'),'--output',str(ready.out/'boundary_geometry')]
    assert result['status']=='SURFACE_GENERATED_NOT_ACCEPTED'
    assert result['boundary_geometry_status']==STATUS
    assert result['candidate_promoted'] is True
    assert result['shape_support_status']=='NO_FLAGGED_REGRESSION_NOT_QUALITY_ACCEPTANCE'
    assert_preferred_unchanged(result,ready.out)
    plan=surface.read(ready.out/'PLAN.json')
    assert plan['automatic_boundary_geometry'] is True
    assert plan['boundary_geometry_max_frames']==16
    assert plan['boundary_geometry_seconds']==600 and plan['boundary_geometry_memory_bytes']==4<<30
    bindings=json.loads((ready.out/'STAGE_BINDINGS.json').read_text(encoding='utf8'))
    assert any(x['path']==str(ready.out/'boundary_geometry/RESULT.json') for x in bindings)
    page=(ready.out/'index.html').read_text(encoding='utf8')
    assert 'boundary_geometry/index.html' in page and 'boundary_geometry/RESULT.json' in page
    assert STATUS in page


def test_not_applicable_is_explicit_without_promoting_a_model(ready):
    ready.options['boundary_record']={'status':'NOT_APPLICABLE','reason':'No bound literal-union source'}
    result=surface.run(ready.source,ready.out)
    assert result['status']=='SURFACE_GENERATED_NOT_ACCEPTED'
    assert result['boundary_geometry_status']=='NOT_APPLICABLE'
    assert result['boundary_geometry_reason']=='No bound literal-union source'
    assert result['stages'][-2]['status']=='NOT_APPLICABLE'
    assert result['stages'][-1]['name']=='shape_support'
    assert_preferred_unchanged(result,ready.out)
    assert 'No bound literal-union source' in (ready.out/'index.html').read_text(encoding='utf8')


@pytest.mark.parametrize('record',[
    {'status':'FAILED'},{'status':'UNKNOWN'},{'source_unchanged':False},
    {'geometry_accepted':True},{'geometry_changed':True},{'geometry_changed':None},
    {'status':'NOT_APPLICABLE','reason':''},{'status':'NOT_APPLICABLE','reason':None},
])
def test_invalid_diagnostic_contract_fails_overall_and_keeps_preferred(ready,record):
    ready.options['boundary_record']=record
    result=surface.run(ready.source,ready.out)
    assert result['status']=='PARTIAL_OR_FAILED'
    assert result['boundary_geometry_status']=='FAILED'
    assert result['stages'][-2]['status']=='BUSINESS_GATE_FAILED'
    assert result['candidate_promoted'] is False
    assert 'boundary geometry' in result['error']
    assert result['surface_result']['status']=='SURFACE_GENERATED_NOT_ACCEPTED'
    assert_preferred_unchanged(result,ready.out)


def test_missing_report_is_not_accepted_as_finished_diagnostic(ready):
    ready.options['boundary_missing_report']=True
    result=surface.run(ready.source,ready.out)
    assert result['status']=='PARTIAL_OR_FAILED'
    assert result['boundary_geometry_status']=='FAILED'
    assert 'report missing' in result['error']
    assert result['candidate_promoted'] is False
    assert_preferred_unchanged(result,ready.out)


def test_boundary_worker_process_failure_cannot_report_main_success(ready):
    ready.options.update(fail='boundary_geometry',kind='process')
    result=surface.run(ready.source,ready.out)
    assert result['status']=='PARTIAL_OR_FAILED'
    assert result['boundary_geometry_status']=='FAILED'
    assert result['stages'][-2]['status']=='FAILED'
    assert result['candidate_promoted'] is False
    assert_preferred_unchanged(result,ready.out)


@pytest.mark.parametrize('failure',['process','business','missing_artifact'])
def test_failed_local_stage_never_launches_boundary_geometry(ready,failure):
    if failure=='missing_artifact':ready.options['local_missing']='LINEAGE.npz'
    else:ready.options.update(fail='face_local',kind=failure)
    result=surface.run(ready.source,ready.out)
    assert ready.calls[-2]['name']=='face_local'
    assert ready.calls[-1]['name']=='shape_support'
    assert '--candidate-run' not in ready.calls[-1]['argv']
    assert result['candidate_promoted'] is False
    assert not any(x['name']=='boundary_geometry' for x in ready.calls)
    assert result['boundary_geometry_status']=='NOT_EVALUATED'


@pytest.mark.parametrize('status',[STATUS,'NOT_APPLICABLE'])
def test_worker_routes_only_to_diagnostic_kernel_with_max_16(tmp_path,monkeypatch,status):
    calls=[]
    record=dict(status=status,source_unchanged=True,geometry_accepted=False,geometry_changed=False,
                reason='No literal-union source' if status=='NOT_APPLICABLE' else None)
    def run(*args,**kwargs):calls.append((args,kwargs));return record
    monkeypatch.setitem(sys.modules,'integrations.video_masks.boundary_geometry',SimpleNamespace(run=run))
    monkeypatch.setattr(surface,'run',lambda *a:pytest.fail('surface/native pipeline must not run'))
    source,gate,out=tmp_path/'source',tmp_path/'local',tmp_path/'diagnostic'
    assert surface.main(['boundary-geometry-worker','--source',str(source),'--gate',str(gate),'--output',str(out)])==0
    assert calls==[((source,gate,out),{'max_frames':16})]


def test_worker_requires_explicit_local_gate_before_dispatch(tmp_path):
    with pytest.raises(SystemExit) as exc:
        surface.main(['boundary-geometry-worker','--source',str(tmp_path/'source'),'--output',str(tmp_path/'out')])
    assert exc.value.code==2


def test_worker_rejects_false_readonly_claim(tmp_path,monkeypatch,capsys):
    monkeypatch.setitem(sys.modules,'integrations.video_masks.boundary_geometry',SimpleNamespace(run=lambda *a,**k:
        dict(status=STATUS,source_unchanged=True,geometry_accepted=False,geometry_changed=True)))
    assert surface.main(['boundary-geometry-worker','--source',str(tmp_path/'source'),
                         '--gate',str(tmp_path/'local'),'--output',str(tmp_path/'out')])==1
    assert 'diagnostic contract failed' in capsys.readouterr().err


def test_new_diagnostic_modules_are_in_code_binding_plan(tmp_path,monkeypatch):
    monkeypatch.setattr(surface,'ROOT',tmp_path)
    monkeypatch.setattr(surface,'binding',lambda p:{'path':str(p)})
    paths={x['path'] for x in surface.code_bindings()}
    for name in ('boundary_geometry.py','boundary_geometry_binding.py','boundary_geometry_evidence.py',
                 'auto_boundary.py','native_gap_evidence.py','narrow_gap.py','narrow_gap_review.py'):
        assert str(tmp_path/'integrations/video_masks'/name) in paths
