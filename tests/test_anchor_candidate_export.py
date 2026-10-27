"""Candidate export through the real gate, PLY reader and offline viewer."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from integrations.demo_pipeline import anchor_candidate_export as export
from test_anchor_candidate_gate import diagnostics, seal_batches


def source_file(tmp_path, source):
    path = tmp_path / 'diagnostics.json'
    path.write_text(json.dumps(source, allow_nan=False), encoding='utf-8')
    return export.binding(path)


@pytest.mark.parametrize('n,k,tier,branch', [
    (2,0.,'TWO_VIEW_PROVISIONAL','MONOTONIC_RADIAL_CONDITIONAL'),
    (4,.2,'STRICT_CONDITIONAL','MONOTONIC_RADIAL_CONDITIONAL'),
    (4,-.2,'STRICT_CONDITIONAL','CENTRAL_BRANCH_CONDITIONAL'),
])
def test_real_gate_export_readback_and_binding(tmp_path,n,k,tier,branch):
    source=diagnostics(n,k);before=deepcopy(source);bound=source_file(tmp_path,source)
    out=tmp_path/'clouds';result=export.export_candidate_clouds(source,out,bound)
    manifest=json.loads((out/'MANIFEST.json').read_text(encoding='utf-8'))
    assert result['status']=='CANDIDATE_CLOUDS_EXPORTED' and manifest['counts']['exportable']==1
    a=manifest['artifacts'][0]
    assert a['tier']==tier and a['branch_scope']==branch
    assert a['vertices']==1 and a['faces']==0 and a['readback'] is True
    assert a['binding']==export.binding(out/a['path'])
    assert a['vertex_ledger'][0]['group_id']=='0:7:0'
    parsed=export.read_ply(out/a['path'])
    assert parsed['vertices']==[source['groups'][0]['fit']['xyz']]
    assert parsed['colors']==[list(export.COLORS[tier])]
    assert manifest['viewers'][0]['binding']==export.binding(out/'viewer_000.html')
    assert len(manifest['viewers'][0]['scenes'])==1
    for r in (result,manifest,a):
        assert all(r[key] is False for key in ('geometry_accepted','native_injected','identity_verified','surface_generation_allowed'))
    assert source==before and export.binding(bound['path'])==bound


def test_all_rejected_no_fabricated_cloud(tmp_path):
    source=diagnostics();source['groups'][0]['fit']['converged']=False
    out=tmp_path/'clouds';result=export.export_candidate_clouds(source,out,source_file(tmp_path,source))
    manifest=json.loads((out/'MANIFEST.json').read_text())
    assert result['status']=='NO_EXPORTABLE_CANDIDATES'
    assert manifest['counts']['tier_counts']=={'REJECTED':1}
    assert not list(out.rglob('*.ply')) and not manifest['viewers']
    assert 'FIT_NOT_CONVERGED' in (out/'CLASSIFICATION.json').read_text()


def test_models_stay_separate_and_viewers_are_bounded(tmp_path):
    source=diagnostics(2)
    base=source['groups'][0];models=[f'model{i:02d}' for i in range(9)]+['../not_a_path']
    source['groups']=[]
    for model in sorted(models):
        group=deepcopy(base);group.update(model_key=model,group_id=f'{model}:7:0');source['groups'].append(group)
    source['selected_groups']=[g['group_id'] for g in source['groups']]
    source['counts'].update(groups=10,selected_groups=10,statuses={'CONDITIONAL_DIAGNOSTIC_EVALUATED':10})
    seal_batches(source)
    out=tmp_path/'clouds';export.export_candidate_clouds(source,out,source_file(tmp_path,source))
    manifest=json.loads((out/'MANIFEST.json').read_text())
    assert len(manifest['artifacts'])==10
    assert {a['model_key'] for a in manifest['artifacts']}==set(models)
    assert [len(v['scenes']) for v in manifest['viewers']]==[8,2]
    assert not (tmp_path/'not_a_path').exists()
    assert all(Path(a['binding']['path']).is_relative_to(out) for a in manifest['artifacts'])
    assert all(a['vertices']==1 for a in manifest['artifacts'])


def test_same_model_batches_export_more_than_32_vertices_without_truncation(tmp_path):
    source=diagnostics(3);base=source['groups'][0];source['groups']=[]
    for i in range(70):
        group=deepcopy(base);group.update(track_id=i,group_id=f'0:{i}:0')
        for observation in group['observations']:observation['track_id']=i
        source['groups'].append(group)
    source['selected_groups']=[g['group_id'] for g in source['groups']]
    source['counts'].update(groups=70,selected_groups=70,statuses={'CONDITIONAL_DIAGNOSTIC_EVALUATED':70})
    seal_batches(source)
    out=tmp_path/'clouds';export.export_candidate_clouds(source,out,source_file(tmp_path,source))
    manifest=json.loads((out/'MANIFEST.json').read_text())
    assert len(manifest['artifacts'])==1 and manifest['artifacts'][0]['vertices']==70
    assert len(manifest['artifacts'][0]['vertex_ledger'])==70
    assert len(export.read_ply(out/manifest['artifacts'][0]['path'])['vertices'])==70
    assert {r['group_id'] for r in manifest['artifacts'][0]['vertex_ledger']}==set(source['selected_groups'])


@pytest.mark.parametrize('kind',['drift','memory_drift','already_exists','bad_partition'])
def test_refusal_before_output_creation(tmp_path,kind):
    source=diagnostics();bound=source_file(tmp_path,source);out=tmp_path/'clouds'
    if kind=='drift':Path(bound['path']).write_text('{}')
    elif kind=='memory_drift':source['counts']['groups']=2
    elif kind=='already_exists':out.mkdir();(out/'user.txt').write_text('keep')
    else:
        source['selected_groups']=[];Path(bound['path']).write_text(json.dumps(source));bound=export.binding(bound['path'])
    with pytest.raises(ValueError):export.export_candidate_clouds(source,out,bound)
    if kind=='already_exists':assert (out/'user.txt').read_text()=='keep'
    else:assert not out.exists()


@pytest.mark.parametrize('kind',['readback','viewer_timeout','source_drift'])
def test_late_failure_keeps_evidence_without_publication(tmp_path,monkeypatch,kind):
    source=diagnostics();bound=source_file(tmp_path,source);out=tmp_path/'clouds'
    if kind=='readback':
        original=export.read_ply
        def broken(*a,**k):
            result=original(*a,**k);result['vertices'][0][0]+=1;return result
        monkeypatch.setattr(export,'read_ply',broken)
    elif kind=='viewer_timeout':
        def stop(*a,**k):raise TimeoutError('TEST_DATA elapsed')
        monkeypatch.setattr(export,'export_html',stop)
    else:
        original=export.export_html
        def drift(*a,**k):
            result=original(*a,**k);Path(bound['path']).write_text('{}');return result
        monkeypatch.setattr(export,'export_html',drift)
    exception,message={
        'readback':(ValueError,'PLY readback differs'),
        'viewer_timeout':(TimeoutError,'TEST_DATA elapsed'),
        'source_drift':(ValueError,'source or classification drift'),
    }[kind]
    with pytest.raises(exception,match=message):export.export_candidate_clouds(source,out,bound)
    assert (out/'FAILURE.json').is_file() and not (out/'MANIFEST.json').exists()
    failure=json.loads((out/'FAILURE.json').read_text())
    assert failure['partial_files_not_publishable'] is True and failure['geometry_accepted'] is False


def test_budget_checked_before_export_side_effect(tmp_path):
    source=diagnostics();bound=source_file(tmp_path,source);out=tmp_path/'clouds'
    def budget():raise TimeoutError('TEST_DATA elapsed')
    with pytest.raises(TimeoutError):export.export_candidate_clouds(source,out,bound,budget)
    assert not out.exists()


def test_upstream_closing_failure_never_publishes_child_success(tmp_path):
    source=diagnostics();bound=source_file(tmp_path,source);out=tmp_path/'clouds'
    def verify_sources():raise ValueError('TEST_DATA upstream image changed during export')
    with pytest.raises(ValueError,match='upstream image changed'):
        export.export_candidate_clouds(source,out,bound,verify_sources=verify_sources)
    assert (out/'FAILURE.json').is_file()
    assert not (out/'MANIFEST.json').exists() and not (out/'index.html').exists()
    assert list(out.rglob('*.ply'))  # Partial files remain inspectable, never published as success.
