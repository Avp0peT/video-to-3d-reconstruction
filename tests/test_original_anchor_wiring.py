"""Offline production wiring, without native cameras, tracking or a neural model."""
import json
from pathlib import Path
import pytest
from integrations.demo_pipeline import automatic_point_review as auto
from integrations.demo_pipeline import original_anchor_review as original
from integrations.demo_pipeline import point_review_cameras as cameras
from integrations.demo_pipeline import anchor_candidate_export as exporter
from integrations.demo_pipeline import seed_pair_validation as seed_module
from integrations.demo_pipeline import view_extension as extension_module


@pytest.mark.parametrize('timeout', [False, True])
def test_default_review_calls_original_branch_even_without_provisional(tmp_path, monkeypatch, timeout):
    run=tmp_path/'run';run.mkdir()
    (run/'pipeline.json').write_text(json.dumps(dict(source_kind='video',stages=dict(segmentation=dict(run_dir='TEST_DATA')))),encoding='utf-8')
    frames=[dict(input_index=i,frame_id=f'TEST_DATA_{i}',original_mask_status='OK') for i in (17,91)]
    tracking=dict(tracks=[dict(track_id=3,observations=[dict(frame_index=i,xy=[1.,2.]) for i in range(2)])])
    calls=[]
    class CV:
        class OCL:
            def setUseOpenCL(self,*a):pass
        ocl=OCL()
        def setNumThreads(self,*a):pass
        def setRNGSeed(self,*a):pass
    monkeypatch.setattr(auto.inputs,'load_cv',lambda refs:CV())
    monkeypatch.setattr(auto.inputs,'load_inputs',lambda *a,**k:(frames,[None,None],[None,None],{}))
    monkeypatch.setattr(auto,'candidate_bindings',lambda *a:{17:[],91:[]})
    monkeypatch.setattr(auto,'track_coverage',lambda *a,**k:tracking)
    camera_result=dict(status='UNSUPPORTED',models=[],unsupported_models=[dict(model_key='test')],native_reads=0)
    monkeypatch.setattr(cameras,'load_registered_cameras',lambda *a:camera_result)
    def analyse(*a,**k):
        calls.append('candidate')
        return dict(decisions=[],geometry_groups=[],counts=dict(provisional_frames=0))
    monkeypatch.setattr(auto,'analyse_sequence',analyse)
    def original_review(actual_frames,masks,mapped,actual_cameras,budget,*,evaluate=True,seed_validation=None,progress=lambda e:None):
        calls.append('original' if evaluate else 'admission');budget()
        assert actual_frames is frames and actual_cameras is camera_result
        assert [o['frame_index'] for o in mapped['tracks'][0]['observations']]==[17,91]
        assert [o['frame_index'] for o in tracking['tracks'][0]['observations']]==[0,1]
        if timeout:raise TimeoutError('TEST_DATA original review budget')
        if not evaluate:return dict(status='ORIGINAL_ANCHOR_ADMISSION_ONLY',groups=[])
        assert seed_validation['status']=='SEED_PAIRS_EVALUATED'
        progress(dict(kind='PLAN',profile=original.PROFILE,batches=[],selected_groups=[],omitted_groups=[]))
        return dict(status='ORIGINAL_ANCHOR_DIAGNOSTICS_COMPLETE',counts=dict(groups=0),camera_count=0)
    monkeypatch.setattr(original,'review_original_anchors',original_review)
    def seed_review(actual_frames,grays,masks,mapped,admission,cv,budget,*,camera_result,progress):
        calls.append('seed');budget()
        assert actual_frames is frames and admission['status']=='ORIGINAL_ANCHOR_ADMISSION_ONLY'
        progress(dict(kind='PLAN',profile=seed_module.PROFILE,batches=[],selected_groups=[],omitted_groups=[]))
        return dict(status='SEED_PAIRS_EVALUATED',counts=dict(verified=0),flow_calls=0,unique_pair_measurements=0)
    monkeypatch.setattr(seed_module,'validate_seed_pairs',seed_review)
    def export_review(actual, output, diagnostic_binding, budget, verify_sources):
        calls.append('export');budget()
        verify_sources()
        assert actual['status']=='ORIGINAL_ANCHOR_DIAGNOSTICS_COMPLETE'
        assert Path(diagnostic_binding['path']).is_file()
        output.mkdir()
        (output/'CLASSIFICATION.json').write_text(json.dumps(dict(rows=[])),encoding='utf-8')
        return dict(status='NO_EXPORTABLE_CANDIDATES',counts=dict(exportable=0))
    monkeypatch.setattr(exporter,'export_candidate_clouds',export_review)
    def extension_review(actual_frames, grays, masks, classification, actual_cameras, cv, budget, progress):
        calls.append('extension');budget()
        assert actual_frames is frames and actual_cameras is camera_result
        assert classification == dict(rows=[])
        progress(dict(event='PLAN', plan=[], profile=extension_module.PROFILE))
        return dict(status='VIEW_EXTENSION_LEDGER_COMPLETE',plan=[],profile=extension_module.PROFILE,query_rows=[],counts=dict(accepted_correspondences=0),flow_calls=0)
    monkeypatch.setattr(extension_module,'extend_views',extension_review)
    out=run/'point_review';result=auto.review(run,out)
    assert calls==(['candidate','admission'] if timeout else ['candidate','admission','seed','original','export','extension'])
    assert result['geometry_accepted'] is False
    if timeout:
        assert result['status']=='RESOURCE_LIMIT'
    else:
        assert result['status']=='AUTO_POINT_REVIEW_COMPLETE'
        assert result['original_anchor_review']['native_injected'] is False
        assert (out/'ORIGINAL_ANCHOR_DIAGNOSTICS.json').is_file()
        assert 'ORIGINAL_ANCHOR_DIAGNOSTICS.json' in (out/'index.html').read_text(encoding='utf-8')
        plan=json.loads((out/'PLAN.json').read_text(encoding='utf-8'))
        assert plan['original_anchor_profile']['max_groups']==96
        assert plan['candidate_profile']['strict']['max_raw_error_px']==2.
        assert plan['seed_pair_profile']['max_groups']==96
        assert plan['seed_pair_profile']['batch_size']==32
        assert result['batch_execution']['seed']['status']==result['batch_execution']['geometry']['status']=='COMPLETE'
        assert (out/'batches/seed/PLAN.json').is_file() and (out/'batches/geometry/PLAN.json').is_file()
        assert (out/'SEED_PAIR_VALIDATION.json').is_file()
        assert result['candidate_clouds']['status']=='NO_EXPORTABLE_CANDIDATES'
        assert 'candidate_clouds/index.html' in (out/'index.html').read_text(encoding='utf-8')
        assert result['view_extension']['pose_fits']==0 and result['view_extension']['new_cameras']==0
        assert result['view_extension_progress']['status']=='COMPLETE'
        assert 'view_extension/index.html' in (out/'index.html').read_text(encoding='utf-8')
        assert plan['view_extension_profile']['max_total_points']==256
