"""No native work: technical files cannot auto-accept object shape."""
import hashlib
import json
from integrations.demo_pipeline import quality


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def ref(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def state():
    return {'run_id':'TEST_DATA', 'target':'cad', 'deliverable_ready':True,
            'target_achieved':True, 'status':'SUCCESS', 'stages':{
                'sfm':{'counts':{'input_rows':36,'registered_unique_images':27,'model_count':4}},
                'segmentation':{'counts':{'no_match':8,'needs_selection':1,'error':0}},
                'surface':{'status':'SUCCESS','selected_model_id':'0','selected_indices':list(range(15)),
                           'missing_indices':list(range(15,36))},
                'blender':{'status':'SUCCESS','reopen_verified':True},'cad':{'status':'SUCCESS'}}}


def test_success_full_registration_low_error_closed_cad_never_auto_accepts(tmp_path):
    s=state();s['stages']['sfm']['counts'].update(registered_unique_images=36,model_count=1)
    s['reprojection_error']=0.0;s['closed']=True
    q=quality.build_quality(tmp_path,s)
    assert q['object_shape_status']=='NOT_EVALUATED'
    assert q['object_goal_achieved'] is q['full_object_accepted'] is False
    assert s['target_achieved'] is True


def test_existing_video_facts_and_unknowns_not_inferred(tmp_path):
    q=quality.build_quality(tmp_path,state())
    assert q['facts']['requested_views']==36 and q['facts']['registered_views']==27
    assert q['facts']['component_count']==4 and q['facts']['selected_view_count']==15
    assert q['facts']['segmentation_no_match']==8
    assert q['facts']['object_supported_observations'] is None
    assert q['facts']['scale_known'] is None
    assert quality.build_quality(tmp_path,{})['facts']['requested_views'] is None


def test_video_total_uses_prepared_frame_identities_not_one_video_file(tmp_path):
    (tmp_path/'input').mkdir()
    save(tmp_path/'input/RESULT.json',{'input_kind':'video','photo_io':{
        'identities':[{'input_index':i} for i in range(36)]}})
    assert quality.build_quality(tmp_path,{})['facts']['requested_views']==36


def test_explicit_bound_result_counts_and_drift_are_distinguished(tmp_path):
    record=tmp_path/'SFM_RESULT.json'
    save(record,{'counts':{'input_rows':28,'registered_unique_images':21,'model_count':3},'scale_status':'unknown'})
    s=state();s['stages']['sfm']['result_reference']=ref(record)
    q=quality.build_quality(tmp_path,s)
    assert q['facts']['registered_views']==21 and q['facts']['requested_views']==28
    assert q['facts']['fact_sources']['sfm']=='HASH_BOUND_RESULT'
    assert q['facts']['scale_known'] is False
    record.write_text('{}')
    q=quality.build_quality(tmp_path,s)
    assert q['facts']['registered_views'] is None


def test_partial_review_binds_current_surface_and_cannot_become_full(tmp_path):
    mesh=tmp_path/'mesh.ply';mesh.write_bytes(b'TEST_DATA')
    s=state();s['stages']['surface']['artifacts']=[ref(mesh)]
    review=tmp_path/'REVIEW.json'
    save(review,{'schema_version':'object-shape-review-v1','status':'PARTIAL_SHAPE',
                 'reviewer':'human TEST_DATA','notes':'Only a can strip is represented.',
                 'surface_artifacts':[ref(mesh)]})
    s['shape_review']=ref(review)
    q=quality.build_quality(tmp_path,s)
    assert q['object_shape_status']=='PARTIAL_SHAPE' and q['object_goal_achieved'] is False
    save(tmp_path/'QUALITY.json',q)
    s['quality_reference']=ref(tmp_path/'QUALITY.json')
    assert quality.read_for_display(tmp_path,s)['object_shape_status']=='PARTIAL_SHAPE'
    # Final pipeline status, ready and report fields do not invalidate a core-facts certificate.
    s.update(status='PARTIAL',finished_utc='later',deliverable_ready=False,target_achieved=False)
    assert quality.read_for_display(tmp_path,s)['object_shape_status']=='PARTIAL_SHAPE'
    mesh.write_bytes(b'changed')
    assert quality.read_for_display(tmp_path,s)['object_shape_status']=='NOT_EVALUATED'


def test_forged_or_stale_quality_never_elevates_shape(tmp_path):
    s=state();q=quality.build_quality(tmp_path,s)
    q.update(object_shape_status='PASS',full_object_accepted=True,object_goal_achieved=True)
    save(tmp_path/'QUALITY.json',q)
    got=quality.read_for_display(tmp_path,s)
    assert got['object_goal_achieved'] is False and got['object_shape_status']=='NOT_EVALUATED'
    save(tmp_path/'QUALITY.json',quality.build_quality(tmp_path,s))
    s['stages']['sfm']['counts']['input_rows']=28
    assert 'quality_warning' in quality.read_for_display(tmp_path,s)


def test_alias_duplicate_indices_are_unknown_not_counted_twice(tmp_path):
    s=state();s['stages']['surface']['selected_indices']=[1,1]
    s['stages']['sfm']['counts']['input_rows']=True
    q=quality.build_quality(tmp_path,s)
    assert q['facts']['selected_view_count'] is None and q['facts']['requested_views'] is None


def test_old_report_and_partial_files_stay_downloadable_with_quality_warning(tmp_path):
    from test_demo_report_surface import fixture_run
    page, validation=fixture_run(tmp_path)
    assert '下载 .blend' in page and '文件已就绪' in page
    assert '物体外形未评估' in page
    assert validation['object_quality']['object_goal_achieved'] is False


def test_quality_app_loaded_old_verified_file_has_warning_and_download(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    from test_photo_3d_ui import existing_fixture, PAGE
    from integrations.mvs_surface import supervision
    run=existing_fixture(tmp_path/'quality_ui')
    def forbidden(*a, **k): raise AssertionError('loading quality must not execute native work')
    monkeypatch.setattr(supervision,'run_supervised',forbidden)
    app=AppTest.from_file(str(PAGE),default_timeout=15).run()
    app.text_input[-1].set_value(str(run)).run()
    app.button[-1].click().run()
    assert not app.exception
    assert any('物体外形未评估' in item.value for item in app.warning)
    assert app.get('download_button')
