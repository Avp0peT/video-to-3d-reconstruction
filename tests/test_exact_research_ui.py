import json
from pathlib import Path
from fractions import Fraction
import runpy
import subprocess
import socket
import sys
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'app/pages/12_Exact_Research.py'


def module(): return runpy.run_path(str(PAGE), run_name='exact_research_test')


def test_family_lossless_algebraic_complete_input():
    m = module(); result = m['compute_family'](m['FAMILY_EXAMPLE'])
    restored = m['exact_decode'](json.loads(result['payload']))
    assert result['independent_check'] is True and result['status'] == 'COMPLETE'
    assert result['family_count'] == 14 and result['unknown_covers'] == 0
    assert set(restored['input']) == {0, 1, 2, 3}
    assert restored['result']['families'][0]['blocks'][0]['kind'] == 'contact4'
    assert restored['result']['families'][0]['centers'][0]['expression'] == 'algebraic_vector'
    exact = {17: (Fraction(1, 3), 2)}
    assert m['exact_decode'](json.loads(json.dumps(m['exact_encode'](exact)))) == exact
    assert isinstance(m['exact_decode'](m['exact_encode'](exact))[17], tuple)


def test_futlyar_cli_schema_serialized_independent_check_and_witness():
    m = module(); result = m['compute_futlyar'](m['FUTLYAR_EXAMPLE'])
    assert result['status'] == result['independent_check'] == 'TRUE'
    assert json.loads(result['payload'])['schema'] == 'generalized-futlyar-result-v1'
    data = json.loads(m['FUTLYAR_EXAMPLE']); data['radius'] = 4
    result = m['compute_futlyar'](json.dumps(data))
    assert result['status'] != 'TRUE'
    assert 'witness' in json.loads(result['witness'])
    assert m['compute_futlyar'](m['FUTLYAR_EXAMPLE'], max_work=0)['status'] == 'UNKNOWN'


def test_invalid_full_input_no_sampling_or_float_coercion():
    m = module(); data = json.loads(m['FAMILY_EXAMPLE'])
    data['points'] = [{'label': i, 'xy': [i, 0]} for i in range(8)]
    with pytest.raises(ValueError, match='不会抽样'): m['compute_family'](json.dumps(data))
    data['points'] = [{'label': 1, 'xy': [0.1, 0]}]
    with pytest.raises(ValueError): m['compute_family'](json.dumps(data))
    with pytest.raises(ValueError, match='duplicate'): m['parse']('{"x":1,"x":2}')


def test_pure_compute_no_process_network_heavy_import(monkeypatch):
    m = module()
    def forbidden(*a, **k): raise AssertionError('unexpected side effect')
    monkeypatch.setattr(subprocess, 'run', forbidden); monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    before = set(sys.modules)
    assert m['compute_futlyar'](m['FUTLYAR_EXAMPLE'])['status'] == 'TRUE'
    assert m['compute_family'](m['FAMILY_EXAMPLE'])['independent_check'] is True
    assert not any(n.split('.')[0] in {'torch', 'sam3'} for n in set(sys.modules) - before)


def test_apptest_click_only_rerun_and_stale_downloads(monkeypatch):
    from kozlov_lab.kozlov2025 import cover_center_family as family
    original = family.build_cover_center_family; calls = []
    def tracked(*a, **k): calls.append(1); return original(*a, **k)
    monkeypatch.setattr(family, 'build_cover_center_family', tracked)
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not app.exception and not calls
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls) == 1
    assert app.session_state['exact_result']['result']['unknown_covers'] == 0
    assert len(app.get('download_button')) == 2
    app.run(); assert len(calls) == 1
    app.text_area(key='exact_family_input').set_value('{}').run()
    assert len(calls) == 1 and not app.get('download_button')
    assert any('旧结果失效' in w.value for w in app.warning)
    app.button(key='exact_compute').click().run()
    assert app.error and 'exact_result' not in app.session_state


def test_apptest_second_mode_exact_result_and_budget_change():
    app = AppTest.from_file(str(PAGE), default_timeout=30).run()
    app.selectbox(key='exact_mode').select('给定中心与半径 Futlyar 验证').run()
    assert not app.exception
    app.button(key='exact_compute').click().run()
    assert not app.exception
    assert app.session_state['exact_result']['result']['independent_check'] == 'TRUE'
    assert len(app.get('download_button')) == 2
    app.number_input(key='futlyar_work').set_value(0).run()
    assert not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert app.session_state['exact_result']['result']['status'] == 'UNKNOWN'


@pytest.mark.parametrize('function,example,status,check',[
    ('compute_center','CENTER_EXAMPLE','EXACT_OPTIMUM_COMPLETE_INPUT','VERIFIED'),
    ('compute_affine','AFFINE_EXAMPLE','TRUE','TRUE'),
    ('compute_environment','ENVIRONMENT_EXAMPLE','COMPLETE','TRUE'),
    ('compute_primary','PRIMARY_EXAMPLE','TRUE','VERIFIED'),
    ('compute_fit_search','FIT_SEARCH_EXAMPLE','TRUE','VERIFIED'),
    ('compute_collinear_fits','COLLINEAR_FITS_EXAMPLE','COLLINEAR_FITS_SUFFICIENT_WITNESS','VERIFIED')])
def test_new_modes_lossless_serialized_independent_checks(function,example,status,check):
    m=module(); result=m[function](m[example])
    assert result['status']==status and result['independent_check']==check
    restored=m['decode_exact'](json.loads(result['payload']))
    assert restored['schema']=='exact-research-tagged-v1' and restored['encoding']=='exact-center-tags-v1'
    assert restored['physical_scale']=='unknown' and restored['photo_quality']=='not evaluated'
    if function=='compute_center':
        assert restored['result']['profile']=='contact4'
        assert set(restored['result']['certificate']['all_input_quadratic_values'])=={0,1,2,3}
        assert len(restored['result']['number_context']['algebraic_root']['poly'])==4
    if function=='compute_environment':
        assert result['parts_count']==7 and result['group_count']==14
        assert result['star_cardinality']==128 and result['parts_check'] is True
    if function=='compute_fit_search':
        assert result['center_rank']==2 and result['candidates_tried']>1
        assert result['general_fits_decided'] is False
    if function=='compute_collinear_fits':
        assert result['error_squared']==Fraction(1,16)
        assert result['general_fits_decided'] is False
        assert set(restored['input']['source'])=={10,11,12}
        assert len(restored['result']['certificate']['point_checks'])==3


@pytest.mark.parametrize('mode,key,status,check',[
    ('单个完整点集精确中心','center','EXACT_OPTIMUM_COMPLETE_INPUT','VERIFIED'),
    ('精确仿射子像','affine','TRUE','TRUE'),
    ('有限环境与标签互斥 S+','environment','COMPLETE','TRUE'),
    ('给定覆盖 primary 验证','primary','TRUE','VERIFIED'),
    ('零中心误差 fits 搜索','fit_search','TRUE','VERIFIED'),
    ('共线中心严格 fits','collinear_fits','COLLINEAR_FITS_SUFFICIENT_WITNESS','VERIFIED')])
def test_apptest_new_modes_click_download_stale_and_unknown(mode,key,status,check):
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.selectbox(key='exact_mode').select(mode).run()
    assert not app.exception and not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert not app.exception
    result=app.session_state['exact_result']['result']
    assert result['status']==status and result['independent_check']==check
    assert len(app.get('download_button'))==2
    app.run();assert not app.exception and app.session_state['exact_result']['result']==result
    app.number_input(key=key+'_work').set_value(0).run()
    assert not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert not app.exception
    result=app.session_state['exact_result']['result']
    assert result['status'] in ('PARTIAL_UNKNOWN','UNKNOWN') and result['independent_check']=='NOT_AVAILABLE'
    app.text_area(key='exact_'+key+'_input').set_value('{}').run()
    assert not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert app.error and 'exact_result' not in app.session_state


def test_new_modes_no_process_network_or_model_calls(monkeypatch):
    m=module()
    def forbidden(*a,**k): raise AssertionError('unexpected external execution')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(socket,'socket',forbidden)
    before=set(sys.modules)
    for function,example in [('compute_center','CENTER_EXAMPLE'),('compute_affine','AFFINE_EXAMPLE'),
                             ('compute_environment','ENVIRONMENT_EXAMPLE'),('compute_primary','PRIMARY_EXAMPLE')]:
        assert m[function](m[example])['complete'] is True
    assert not any(n.split('.')[0] in {'torch','sam3'} for n in set(sys.modules)-before)


def test_new_schema_type_and_empty_policy_refusal():
    m=module();data=json.loads(m['CENTER_EXAMPLE']);data['points'][0]['xy'][0]=True
    with pytest.raises(ValueError):m['compute_center'](json.dumps(data))
    data=json.loads(m['ENVIRONMENT_EXAMPLE']);data['include_empty']=0
    with pytest.raises(ValueError):m['compute_environment'](json.dumps(data))
    data['include_empty']=True
    result=m['compute_environment'](json.dumps(data))
    assert result['group_count']==15 and result['independent_check']=='TRUE'


def test_seven_point_environment_lossless_chunk_transport():
    m=module();text=json.dumps(dict(schema='finite-environment-input-v1',
        points=[dict(label=i,xy=[i,0]) for i in range(7)],include_empty=False))
    answer=m['compute_environment'](text)
    assert answer['status']=='COMPLETE' and answer['independent_check']=='TRUE'
    assert answer['group_count']==4139 and answer['parts_count']==127
    encoded=json.loads(answer['payload'])
    assert encoded['schema']=='exact-research-chunked-v1'
    restored=m['decode_new_payload'](encoded)
    assert len(restored['result']['groups'])==4139
    assert restored['result']['groups']==restored['result']['certificate']['groups']
    encoded['group_chunks'][0]['offset']=1
    with pytest.raises(ValueError):m['decode_new_payload'](encoded)


def test_apptest_genuine_cubic_primary_and_false_is_not_invalid(monkeypatch):
    m=module()
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.selectbox(key='exact_mode').select('给定覆盖 primary 验证').run()
    app.text_area(key='exact_primary_input').set_value(m['PRIMARY_CUBIC_EXAMPLE']).run()
    app.button(key='exact_compute').click().run()
    assert not app.exception
    result=app.session_state['exact_result']['result']
    assert result['status']=='TRUE' and result['independent_check']=='VERIFIED'
    restored=m['decode_new_payload'](json.loads(result['payload']))
    assert len(restored['result']['certificate']['number_context']['algebraic_root']['poly'])==4
    def forbidden(*a,**kw): raise AssertionError('TRUE-only verifier must not see FALSE')
    monkeypatch.setitem(m['compute_primary'].__globals__,'assess_algebraic_primary_certificate',forbidden)
    negative=dict(schema='primary-futlyar-input-v1',points=[dict(label=i,xy=p) for i,p in enumerate([[-1,0],[1,0],[1,1]])],parts=[[0,1],[2]])
    result=m['compute_primary'](json.dumps(negative))
    assert result['status']=='FALSE' and result['independent_check']=='NOT_AVAILABLE'
    assert '反例' in result['check_note']


def test_fit_search_click_only_and_rank1_scope(monkeypatch):
    from kozlov_lab.kozlov2025 import zero_error_fit_search as search
    original=search.search_zero_error_fits;calls=[]
    def tracked(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(search,'search_zero_error_fits',tracked)
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.selectbox(key='exact_mode').select('零中心误差 fits 搜索').run()
    assert not calls
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==1
    assert app.session_state['exact_result']['result']['center_rank']==2
    app.run();assert len(calls)==1
    reference=[dict(label=0,xy=[0,0]),dict(label=1,xy=[10,0])]
    source=[dict(label=i,xy=p) for i,p in enumerate([[-1,-20],[1,-20],[0,40],[9,-20],[11,-20],[10,40]])]
    text=json.dumps(dict(schema='zero-error-fit-search-input-v1',reference=reference,reference_parts=[[0],[1]],
                         source=source,source_parts=[[0,1,2],[3,4,5]]))
    app.text_area(key='exact_fit_search_input').set_value(text).run()
    assert len(calls)==1 and not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==2
    result=app.session_state['exact_result']['result']
    assert result['status']=='NO_SUPPORTED_WITNESS' and result['independent_check']=='NOT_AVAILABLE'
    assert result['center_rank']==1 and result['general_fits_decided'] is False
    assert any('不证明通用 fits 不存在' in info.value for info in app.info)


def test_collinear_apptest_click_only_nonzero_and_no_general_false(monkeypatch):
    import importlib
    algorithm=importlib.import_module('kozlov_lab.kozlov2025.collinear_fits')
    original=algorithm.collinear_fits;calls=[]
    def tracked(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(algorithm,'collinear_fits',tracked)
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.selectbox(key='exact_mode').select('共线中心严格 fits').run()
    assert not calls
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==1
    result=app.session_state['exact_result']['result']
    assert result['error_squared']==Fraction(1,16) and result['independent_check']=='VERIFIED'
    assert any('1/16' in text.value for text in app.markdown)
    app.run();assert len(calls)==1
    m=module();data=json.loads(m['COLLINEAR_FITS_EXAMPLE'])
    data['source']=[dict(label=i+10,xy=p) for i,p in enumerate([[-10,-10],[10,-10],[0,20],[1,0],[2,0]])]
    data['source_parts']=[[10,11,12],[13],[14]]
    app.text_area(key='exact_collinear_fits_input').set_value(json.dumps(data)).run()
    assert not app.get('download_button') and len(calls)==1
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==2
    result=app.session_state['exact_result']['result']
    assert result['status']=='NO_SUPPORTED_WITNESS' and result['independent_check']=='NOT_AVAILABLE'
    assert result['general_fits_decided'] is False
    assert any('不证明通用 fits 不存在' in info.value for info in app.info)

@pytest.mark.parametrize('function,example,status',[
    ('compute_four_center_fits','FOUR_CENTER_FITS_EXAMPLE','FOUR_CENTER_FITS_SUFFICIENT_WITNESS'),
    ('compute_compatibility','COMPATIBILITY_EXAMPLE','EXACT_GRANULARITY_CERTIFIED')])
def test_algorithm_completion_helpers_full_exact_readback(function,example,status):
    m=module();result=m[function](m[example],seconds=30)
    assert result['status']==status and result['independent_check']=='VERIFIED'
    restored=m['decode_new_payload'](json.loads(result['payload']))
    if function=='compute_four_center_fits':
        from kozlov_lab.kozlov2025.four_center_fits import assess_four_center_fits_certificate
        assert result['error_squared']==Fraction(1,16) and result['general_fits_decided'] is False
        assert set(restored['input']['source'])=={10,11,12,13}
        assert len(restored['result']['certificate']['alignment_certificate']['ledger'])==24
        assert assess_four_center_fits_certificate(restored['input']['reference'],restored['input']['source'],restored['result']['certificate'])['status']=='VERIFIED'
    else:
        from kozlov_lab.kozlov2025.finite_compatibility_search import assess_finite_compatibility_certificate
        from kozlov_lab.kozlov2025.full_group_certificate import assess_full_group_certificate
        assert result['granularity']==3 and result['full_group_status']=='FULL_GROUP_CERTIFIED'
        assert result['full_group_check']=='VERIFIED' and result['general_compatibility_decided'] is False
        assert set(restored['input']['environment'])==set(range(7))
        assert len(restored['result']['visited_candidates'])==result['candidates_tried']>0
        assert assess_finite_compatibility_certificate(restored['input']['environment'],restored['input']['group'],restored['result']['certificate'])['status']=='VERIFIED'
        assert assess_full_group_certificate(restored['input']['environment'],restored['input']['group'],restored['result']['full_group_result']['certificate'])['status']=='VERIFIED'
    unknown=m[function](m[example],max_work=0)
    assert unknown['status']=='UNKNOWN' and unknown['independent_check']=='NOT_AVAILABLE'

@pytest.mark.parametrize('mode,key,module_name,function,status',[
    ('四中心非共线严格 fits','four_center_fits','four_center_fits','four_center_fits','FOUR_CENTER_FITS_SUFFICIENT_WITNESS'),
    ('自动群兼容、粒度与完整群','compatibility','finite_compatibility_search','search_finite_compatibility','EXACT_GRANULARITY_CERTIFIED')])
def test_algorithm_completion_apptest_actual_button_cache_stale_budget(monkeypatch,mode,key,module_name,function,status):
    import importlib
    algorithm=importlib.import_module('kozlov_lab.kozlov2025.'+module_name)
    original=getattr(algorithm,function);calls=[]
    def tracked(*args,**kwargs): calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(algorithm,function,tracked)
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.selectbox(key='exact_mode').select(mode).run()
    assert not app.exception and not calls and not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==1
    result=app.session_state['exact_result']['result']
    assert result['status']==status and result['independent_check']=='VERIFIED'
    assert len(app.get('download_button'))==2
    if key=='four_center_fits': assert any('1/16' in text.value for text in app.markdown)
    else: assert result['full_group_status']=='FULL_GROUP_CERTIFIED' and result['full_group_check']=='VERIFIED'
    app.run();assert not app.exception and len(calls)==1 and app.session_state['exact_result']['result']==result
    app.number_input(key=key+'_work').set_value(0).run()
    assert not app.get('download_button') and len(calls)==1
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==2
    unknown=app.session_state['exact_result']['result']
    assert unknown['status']=='UNKNOWN' and unknown['independent_check']=='NOT_AVAILABLE'
    assert any('不表示一般不兼容' in i.value or '不证明通用 fits 不存在' in i.value for i in app.info)
    app.text_area(key='exact_'+key+'_input').set_value('{}').run()
    assert not app.get('download_button') and len(calls)==2
    app.button(key='exact_compute').click().run()
    assert app.error and 'exact_result' not in app.session_state and len(calls)==2


def test_algorithm_completion_large_candidate_ledger_explicit_pieces():
    from kozlov_lab.kozlov2025.exact_center import decode_exact
    m=module()
    ledger=[dict(id=i,attempts=[dict(matrix=((Fraction(1,3),Fraction(2)),(Fraction(0),Fraction(1))),mapping=tuple((j,j+10) for j in range(20)),status='UNKNOWN')]) for i in range(1000)]
    result=dict(status='UNKNOWN',visited_candidates=ledger,certificate=None,general_compatibility_decided=False)
    payload,restored=m['_new_payload']('test',dict(environment={9:(Fraction(1,7),Fraction(2))},group=((9,),)),result,'test proof')
    encoded=json.loads(payload)
    assert encoded['schema']=='exact-research-proof-chunked-v1'
    assert restored['result']==result and restored['input']['environment']=={9:(Fraction(1,7),Fraction(2))}
    assert m['decode_new_payload'](encoded)==restored
    with pytest.raises(ValueError): decode_exact(m['encode_exact'](restored))
    # Reject malformed transport without relaxing the exact per-piece decoder.
    encoded['document']['kind']='unexpected'
    with pytest.raises(ValueError):m['decode_new_payload'](encoded)


def test_algorithm_completion_search_partial_ledger_unknown():
    m=module();answer=m['compute_compatibility'](m['COMPATIBILITY_EXAMPLE'],max_work=1000)
    assert answer['status']=='UNKNOWN' and answer['general_compatibility_decided'] is False
    restored=m['decode_new_payload'](json.loads(answer['payload']))
    assert len(restored['result']['visited_candidates'])==answer['candidates_tried']
    assert restored['result']['certificate'] is None and answer['full_group_status']=='NOT_AVAILABLE'


def test_four_center_ui_tied_minima_are_not_general_negative():
    m=module();data=json.loads(m['FOUR_CENTER_FITS_EXAMPLE'])
    data['reference']=[dict(label=i,xy=p) for i,p in enumerate(((0,0),(2,0),(0,2),(-1,-1)))]
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.selectbox(key='exact_mode').select('四中心非共线严格 fits').run()
    app.text_area(key='exact_four_center_fits_input').set_value(json.dumps(data)).run()
    app.button(key='exact_compute').click().run()
    assert not app.exception
    result=app.session_state['exact_result']['result']
    assert result['status']=='NO_SUPPORTED_WITNESS' and result['independent_check']=='NOT_AVAILABLE'
    assert result['general_fits_decided'] is False
    assert any('不证明通用 fits 不存在' in i.value for i in app.info)
    restored=m['decode_new_payload'](json.loads(result['payload']))
    assert restored['result']['alignment_status']=='GLOBAL_OPTIMAL_STABILITY_UNRESOLVED'


def test_algorithm_completion_no_native_network_models_and_shared_budget(monkeypatch):
    m=module()
    def forbidden(*args,**kwargs): raise AssertionError('unexpected external execution')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(socket,'socket',forbidden)
    before=set(sys.modules)
    assert m['compute_four_center_fits'](m['FOUR_CENTER_FITS_EXAMPLE'])['independent_check']=='VERIFIED'
    result=m['compute_compatibility'](m['COMPATIBILITY_EXAMPLE'])
    assert result['full_group_check']=='VERIFIED'
    restored=m['decode_new_payload'](json.loads(result['payload']))
    assert restored['result']['work_used']<restored['result']['total_work_used']<=2000000
    assert not any(n.split('.')[0] in {'torch','sam3'} for n in set(sys.modules)-before)


def test_circuit_fits_complete_nonzero_five_points_readback():
    from kozlov_lab.kozlov2025.circuit_fits import assess_circuit_fits_certificate
    m=module();result=m['compute_circuit_fits'](m['CIRCUIT_FITS_EXAMPLE'],seconds=30)
    assert result['status']=='CIRCUIT_FITS_SUFFICIENT_WITNESS' and result['independent_check']=='VERIFIED'
    assert result['error_squared']==Fraction(1,100) and result['general_fits_decided'] is False
    restored=m['decode_new_payload'](json.loads(result['payload']))
    assert set(restored['input']['source'])==set(range(10,15))
    assert len(restored['result']['certificate']['alignment_certificate']['ledger'])==120
    assert len(restored['result']['certificate']['point_checks'])==5
    assert assess_circuit_fits_certificate(restored['input']['reference'],restored['input']['source'],restored['result']['certificate'])['status']=='VERIFIED'


def test_circuit_fits_apptest_button_download_rerun_stale_unknown(monkeypatch):
    import importlib
    algorithm=importlib.import_module('kozlov_lab.kozlov2025.circuit_fits')
    original=algorithm.circuit_fits;calls=[]
    def tracked(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(algorithm,'circuit_fits',tracked)
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    assert len(app.selectbox(key='exact_mode').options)==18
    app.selectbox(key='exact_mode').select('多中心下界证书 fits').run()
    assert not calls and not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==1
    r=app.session_state['exact_result']['result']
    assert r['status']=='CIRCUIT_FITS_SUFFICIENT_WITNESS' and r['independent_check']=='VERIFIED'
    assert len(app.get('download_button'))==2
    assert any('1/100' in row.value for row in app.markdown)
    m=module();readback=m['decode_new_payload'](json.loads(r['payload']))
    assert len(readback['result']['certificate']['alignment_certificate']['ledger'])==120
    app.run();assert len(calls)==1 and app.session_state['exact_result']['result']==r
    app.number_input(key='circuit_fits_work').set_value(0).run()
    assert not app.get('download_button') and len(calls)==1
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==2
    unknown=app.session_state['exact_result']['result']
    assert unknown['status']=='UNKNOWN' and unknown['independent_check']=='NOT_AVAILABLE'
    assert unknown['general_fits_decided'] is False
    assert any('不证明通用 fits 不存在' in row.value for row in app.info)
    app.text_area(key='exact_circuit_fits_input').set_value('{}').run()
    assert not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert app.error and 'exact_result' not in app.session_state and len(calls)==2


def test_circuit_six_point_apptest_real_ledger_chunked_readback():
    from kozlov_lab.kozlov2025.circuit_fits import assess_circuit_fits_certificate
    m=module();data=json.loads(m['CIRCUIT_FITS_EXAMPLE'])
    data['reference'].append(dict(label=5,xy=[3,5]));data['source'].append(dict(label=15,xy=[3,5]))
    data['reference_parts'].append([5]);data['source_parts'].append([15])
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.selectbox(key='exact_mode').select('多中心下界证书 fits').run()
    app.text_area(key='exact_circuit_fits_input').set_value(json.dumps(data)).run()
    app.number_input(key='circuit_fits_seconds').set_value(30).run()
    app.button(key='exact_compute').click().run()
    assert not app.exception
    r=app.session_state['exact_result']['result']
    assert r['status']=='CIRCUIT_FITS_SUFFICIENT_WITNESS' and r['independent_check']=='VERIFIED'
    encoded=json.loads(r['payload'])
    assert encoded['schema']=='exact-research-proof-chunked-v1'
    restored=m['decode_new_payload'](encoded);c=restored['result']['certificate']
    assert len(c['alignment_certificate']['ledger'])==720 and len(c['point_checks'])==6
    assert set(restored['input']['source'])==set(range(10,16))
    assert assess_circuit_fits_certificate(restored['input']['reference'],restored['input']['source'],c)['status']=='VERIFIED'
    assert len(app.get('download_button'))==2


def test_circuit_unknown_keeps_complete_inputs_and_no_general_negative():
    m=module();data=json.loads(m['CIRCUIT_FITS_EXAMPLE'])
    data['reference'][4]['xy']=[31,48]
    r=m['compute_circuit_fits'](json.dumps(data),seconds=30)
    assert r['status']=='UNKNOWN' and r['independent_check']=='NOT_AVAILABLE'
    assert r['general_fits_decided'] is False and '不证明通用 fits 不存在' in r['check_note']
    restored=m['decode_new_payload'](json.loads(r['payload']))
    assert len(restored['input']['source'])==len(restored['input']['reference'])==5


@pytest.mark.parametrize('function,example,status',[
    ('compute_translation','TRANSLATION_EXAMPLE','EXACT_OPTIMUM'),
    ('compute_candidate_acceptance','CANDIDATE_ACCEPTANCE_EXAMPLE','EXACT_PLACED_CANDIDATE'),
    ('compute_placed_bottleneck','PLACED_BOTTLENECK_EXAMPLE','EXACT_PLACED_BOTTLENECK')])
def test_legacy_registration_lossless_readback(function,example,status):
    m=module();text=m[example];out=m[function](text)
    assert out['status']==status and out['independent_check']=='VERIFIED'
    restored=m['decode_new_payload'](json.loads(out['payload']))
    assert set(restored['input']['source'])=={0,1,2}
    assert set(restored['input']['target'])=={10,11,12}
    assert out['total_work_used']<=2000000 and restored['physical_scale']=='unknown'
    assert out['payload']==m[function](text)['payload']  # insertion order remains deterministic
    if function=='compute_translation':
        assert out['distance_squared']==Fraction(169,36)
        assert out['translation']==(-2,-Fraction(5,6))
        assert len(restored['result']['certificate']['residuals_squared'])==3
    if function=='compute_candidate_acceptance':
        assert out['acceptable'] is True and out['narrow_acceptable'] is False
        assert out['obstruction']['kind']=='hall_deficiency'
    if function=='compute_placed_bottleneck': assert out['distance_squared']==64
    unknown=m[function](text,max_work=0)
    assert unknown['status']=='UNKNOWN' and unknown['independent_check']=='NOT_AVAILABLE'
    assert m['decode_new_payload'](json.loads(unknown['payload']))['input']==restored['input']


@pytest.mark.parametrize('mode,key,status',[
    ('固定朝向与给定对应最优平移','translation','EXACT_OPTIMUM'),
    ('给定位置接受性','candidate_acceptance','EXACT_PLACED_CANDIDATE'),
    ('给定位置瓶颈匹配','placed_bottleneck','EXACT_PLACED_BOTTLENECK')])
def test_apptest_legacy_registration_modes(mode,key,status):
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    assert len(app.selectbox(key='exact_mode').options)==18
    app.selectbox(key='exact_mode').select(mode).run()
    assert not app.exception and not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert not app.exception
    result=app.session_state['exact_result']['result']
    assert result['status']==status and result['independent_check']=='VERIFIED'
    assert len(app.get('download_button'))==2
    app.run();assert app.session_state['exact_result']['result']==result
    app.number_input(key=key+'_work').set_value(0).run()
    assert not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert app.session_state['exact_result']['result']['status']=='UNKNOWN'
    app.text_area(key='exact_'+key+'_input').set_value('{}').run()
    assert not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert app.error and 'exact_result' not in app.session_state


def test_registration_ui_input_boundary_and_no_external_execution(monkeypatch):
    m=module()
    def forbidden(*a,**k): raise AssertionError('unexpected external execution')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(socket,'socket',forbidden)
    before=set(sys.modules)
    for function,example in [('compute_translation','TRANSLATION_EXAMPLE'),
                             ('compute_candidate_acceptance','CANDIDATE_ACCEPTANCE_EXAMPLE'),
                             ('compute_placed_bottleneck','PLACED_BOTTLENECK_EXAMPLE')]:
        assert m[function](m[example])['independent_check']=='VERIFIED'
    assert not any(n.split('.')[0] in {'torch','sam3'} for n in set(sys.modules)-before)
    data=json.loads(m['TRANSLATION_EXAMPLE']);data['mapping'][1]['source']=0
    with pytest.raises(ValueError):m['compute_translation'](json.dumps(data))
    data=json.loads(m['TRANSLATION_EXAMPLE']);data['orientation'][0][0]=True
    with pytest.raises(ValueError):m['compute_translation'](json.dumps(data))
    data=json.loads(m['PLACED_BOTTLENECK_EXAMPLE'])
    data['source']=[dict(label=i,xy=[i,0]) for i in range(33)]
    with pytest.raises(ValueError,match='不会抽样'):m['compute_placed_bottleneck'](json.dumps(data))


@pytest.mark.parametrize('function,example,status',[
    ('compute_environment_code','ENVIRONMENT_CODE_EXAMPLE','EXACT_RESTRICTED_CODE'),
    ('compute_ellipse_code2019','ELLIPSE_CODE2019_EXAMPLE','EXACT_ELLIPSE_CODE')])
def test_environment_and_ellipse_codes_lossless_readback(function,example,status):
    m=module();r=m[function](m[example])
    assert r['status']==status and r['independent_check']=='VERIFIED'
    assert r['total_work_used']<=2000000
    doc=m['decode_new_payload'](json.loads(r['payload']))
    assert doc['result']['affine_equivalence_decided'] is False
    if function=='compute_environment_code':
        from kozlov_lab.coding.environment_code import assess_restricted_code
        assert set(doc['input']['environment'])=={11,22,33,44,55}
        assert doc['input']['subset']==(11,22,33,44)
        assert assess_restricted_code(doc['input']['environment'],doc['input']['subset'],doc['result']['certificate'])['status']=='VERIFIED'
        assert r['global_labels']==(11,22,33,44,55) and r['subset_labels']==(11,22,33,44)
    else:
        from kozlov_lab.coding.ellipse_code2019 import assess_ellipse_code2019
        assert set(doc['input']['points'])=={11,22,33,44}
        assert r['ratio_count']==12 and r['ellipse_count']==4
        assert assess_ellipse_code2019(doc['input']['points'],doc['result']['certificate'])['status']=='VERIFIED'
    unknown=m[function](m[example],max_work=0)
    assert unknown['status']=='UNKNOWN' and unknown['independent_check']=='NOT_AVAILABLE'
    assert m['decode_new_payload'](json.loads(unknown['payload']))['input']==doc['input']


@pytest.mark.parametrize('mode,key,modname,function,status',[
    ('环境代码与子图像限制','environment_code','environment_code','restrict_environment_code','EXACT_RESTRICTED_CODE'),
    ('2019椭圆代码','ellipse_code2019','ellipse_code2019','build_ellipse_code2019','EXACT_ELLIPSE_CODE')])
def test_codes_apptest_actual_button_cache_stale_unknown(monkeypatch,mode,key,modname,function,status):
    import importlib
    algorithm=importlib.import_module('kozlov_lab.coding.'+modname)
    original=getattr(algorithm,function);calls=[]
    def tracked(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(algorithm,function,tracked)
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    assert len(app.selectbox(key='exact_mode').options)==18
    app.selectbox(key='exact_mode').select(mode).run()
    assert not app.exception and not calls and not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==1
    r=app.session_state['exact_result']['result']
    assert r['status']==status and r['independent_check']=='VERIFIED'
    assert len(app.get('download_button'))==2
    assert any('不判仿射等价' in x.value for x in app.markdown)
    app.run();assert len(calls)==1 and app.session_state['exact_result']['result']==r
    app.number_input(key=key+'_work').set_value(0).run()
    assert not app.get('download_button') and len(calls)==1
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==2
    assert app.session_state['exact_result']['result']['status']=='UNKNOWN'
    app.text_area(key='exact_'+key+'_input').set_value('{}').run()
    assert not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert app.error and 'exact_result' not in app.session_state and len(calls)==2


@pytest.mark.parametrize('function,schema',[
    ('compute_environment_code','environment-code-input-v1'),
    ('compute_ellipse_code2019','ellipse-code2019-input-v1')])
def test_complete_n8_codes_download_and_independent_readback(function,schema):
    m=module();rows=[dict(label=100+i,xy=[i,i*i]) for i in range(8)]
    data=dict(schema=schema)
    if function=='compute_environment_code':data.update(environment=rows,subset=list(range(100,108)))
    else:data.update(points=rows)
    r=m[function](json.dumps(data),seconds=30)
    assert r['independent_check']=='VERIFIED' and r['ratio_count']==3080
    restored=m['decode_new_payload'](json.loads(r['payload']))
    assert len(restored['result']['certificate'].get('ellipse_ratios',restored['result']['certificate'].get('restricted',{}).get('ratios',{})))==3080
    if function=='compute_ellipse_code2019':assert r['ellipse_count']==56
    assert set(restored['input'].get('environment',restored['input'].get('points')))==set(range(100,108))


def test_codes_strict_inputs_degenerate_and_outside_plane_domain():
    m=module();data=json.loads(m['ENVIRONMENT_CODE_EXAMPLE'])
    data['subset']=[True]
    with pytest.raises(ValueError):m['compute_environment_code'](json.dumps(data))
    data=json.loads(m['ELLIPSE_CODE2019_EXAMPLE']);data['points'][2]['xy']=[4,0]
    r=m['compute_ellipse_code2019'](json.dumps(data))
    assert r['status']=='UNSUPPORTED' and r['independent_check']=='NOT_AVAILABLE'
    assert len(m['decode_new_payload'](json.loads(r['payload']))['input']['points'])==4
    data=dict(schema='environment-code-input-v1',environment=[dict(label=9,xy=[0,0]),dict(label=42,xy=[1,0]),dict(label=87,xy=[2,0])],subset=[9,42,87])
    r=m['compute_environment_code'](json.dumps(data))
    assert r['status']=='EXACT_RESTRICTED_CODE' and r['independent_check']=='VERIFIED'
    assert r['code_defined'] is False and r['plane_domain'] is False
    assert r['ratio_count']==0 and r['zero_denominator_count']==1
    data['environment']=[dict(label=i,xy=[i,0]) for i in range(9)]
    with pytest.raises(ValueError,match='不会抽样'):m['compute_environment_code'](json.dumps(data))


def test_codes_ui_no_external_execution(monkeypatch):
    m=module()
    def forbidden(*a,**kw):raise AssertionError('external execution forbidden')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(socket,'socket',forbidden)
    before=set(sys.modules)
    assert m['compute_environment_code'](m['ENVIRONMENT_CODE_EXAMPLE'])['independent_check']=='VERIFIED'
    assert m['compute_ellipse_code2019'](m['ELLIPSE_CODE2019_EXAMPLE'])['independent_check']=='VERIFIED'
    assert not any(n.split('.')[0] in {'torch','sam3'} for n in set(sys.modules)-before)



def test_proportionality_ui_separates_one_way_definition_from_all_areas():
    from kozlov_lab.coding.environment_proportionality import assess_proportionality_certificate
    m=module();r=m['compute_proportionality'](m['PROPORTIONALITY_EXAMPLE'])
    assert r['status']=='PROPORTIONAL_GIVEN_MAPPING' and r['complete'] is True
    assert r['independent_check']=='VERIFIED' and r['total_work_used']<=2000000
    assert r['paper_proportionality_given_mapping'] is True
    assert r['degeneracy_preserved'] is False and r['full_area_proportionality'] is False
    assert r['theorem_eligible'] is False and r['ledger_count']==4
    doc=m['decode_new_payload'](json.loads(r['payload']));i=doc['input'];c=doc['result']['certificate']
    assert set(i['env'])=={0,1,2,3,4} and i['source_labels']==(0,1,2,3)
    assert assess_proportionality_certificate(i['env'],i['source_labels'],i['target_labels'],i['mapping'],c)['status']=='VERIFIED'
    assert c['ledger'][0]['source_double_area']==0 and c['ledger'][0]['ratio']=='NOT_QUANTIFIED'
    assert doc['result']['unknown_mapping_decided'] is False and doc['result']['affine_matrix_constructed'] is False
    assert r['self_pair_count']==1
    for row in c['ledger']:
        if row['environment_ratio_reference']=='SELF_PAIR_EXCLUDED':
            assert row['ratio']==1 and row['pair'] not in c['environment']['ratios']
    unknown=m['compute_proportionality'](m['PROPORTIONALITY_EXAMPLE'],max_work=0)
    assert unknown['status']=='UNKNOWN' and unknown['independent_check']=='NOT_AVAILABLE' and unknown['complete'] is False
    assert m['decode_new_payload'](json.loads(unknown['payload']))['input']==i


def test_proportionality_negative_is_complete_only_for_given_mapping():
    m=module();data=dict(schema='environment-proportionality-input-v1',
        env=[dict(label=i,xy=p) for i,p in enumerate(((0,0),(1,0),(0,1),(10,0),(11,0),(12,0)))],
        source_labels=[0,1,2],target_labels=[3,4,5],mapping=[dict(source=i,target=i+3) for i in range(3)])
    r=m['compute_proportionality'](json.dumps(data))
    assert r['status']=='NOT_PROPORTIONAL_GIVEN_MAPPING' and r['complete'] is True
    assert r['independent_check']=='VERIFIED'
    assert r['paper_proportionality_given_mapping'] is False and r['full_area_proportionality'] is False
    doc=m['decode_new_payload'](json.loads(r['payload']))
    assert doc['result']['unknown_mapping_decided'] is False
    assert doc['result']['certificate']['ledger'][0]['ratio']=='UNDEFINED'
    assert '不判未知对应或非仿射' in r['check_note']


def test_proportionality_apptest_actual_click_rerun_budget_and_input_stale(monkeypatch):
    import importlib
    algorithm=importlib.import_module('kozlov_lab.coding.environment_proportionality')
    original=algorithm.paper_proportionality_given_mapping;calls=[]
    def tracked(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(algorithm,'paper_proportionality_given_mapping',tracked)
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    assert len(app.selectbox(key='exact_mode').options)==18
    app.selectbox(key='exact_mode').select('环境内给定对应比例性').run()
    assert not calls and not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(calls)==1
    r=app.session_state['exact_result']['result']
    assert r['independent_check']=='VERIFIED' and r['complete'] is True
    assert len(app.get('download_button'))==2
    assert any('原文给定对应比例性：True' in x.value for x in app.markdown)
    assert any('退化保持：False' in x.value for x in app.markdown)
    assert any('全部面积比例：False' in x.value for x in app.markdown)
    assert any('平面定理适用：False' in x.value for x in app.markdown)
    app.run();assert len(calls)==1 and app.session_state['exact_result']['result']==r
    app.number_input(key='proportionality_work').set_value(0).run()
    assert not app.get('download_button') and len(calls)==1
    app.button(key='exact_compute').click().run()
    assert len(calls)==2 and app.session_state['exact_result']['result']['status']=='UNKNOWN'
    app.text_area(key='exact_proportionality_input').set_value('{}').run()
    assert not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert app.error and 'exact_result' not in app.session_state and len(calls)==2


@pytest.mark.parametrize('field', ['source_labels','target_labels','mapping_source','mapping_target','mapping_duplicate'])
def test_proportionality_ui_rejects_boolean_or_duplicate_correspondence(field):
    m=module();data=json.loads(m['PROPORTIONALITY_EXAMPLE'])
    if field in ('source_labels','target_labels'):data[field][0]=False
    elif field=='mapping_source':data['mapping'][0]['source']=False
    elif field=='mapping_target':data['mapping'][0]['target']=False
    else:data['mapping'][1]['source']=data['mapping'][0]['source']
    with pytest.raises(ValueError):m['compute_proportionality'](json.dumps(data))


def test_proportionality_n8_complete_identity_without_fabricated_self_ratio():
    m=module();data=dict(schema='environment-proportionality-input-v1',
        env=[dict(label=i+100,xy=[i,i*i]) for i in range(8)],source_labels=list(range(100,108)),target_labels=list(range(100,108)),
        mapping=[dict(source=i,target=i) for i in range(100,108)])
    r=m['compute_proportionality'](json.dumps(data),seconds=30)
    assert r['status']=='PROPORTIONAL_GIVEN_MAPPING' and r['independent_check']=='VERIFIED'
    assert r['self_pair_count']==56 and r['ledger_count']==56
    doc=m['decode_new_payload'](json.loads(r['payload']));c=doc['result']['certificate']
    assert len(c['environment']['ratios'])==3080
    assert all(row['pair'] not in c['environment']['ratios'] and row['ratio']==1 for row in c['ledger'])
    assert set(doc['input']['env'])==set(range(100,108))


def test_proportionality_ui_no_external_execution(monkeypatch):
    m=module()
    def forbidden(*a,**kw):raise AssertionError('external execution forbidden')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(socket,'socket',forbidden)
    before=set(sys.modules)
    assert m['compute_proportionality'](m['PROPORTIONALITY_EXAMPLE'])['independent_check']=='VERIFIED'
    assert not any(n.split('.')[0] in {'torch','sam3'} for n in set(sys.modules)-before)


def _ellipse_equivalence_input(source,target):
    return json.dumps(dict(schema='ellipse-equivalence2019-input-v1',
        source=[dict(label=i,xy=list(p)) for i,p in source.items()],
        target=[dict(label=i,xy=list(p)) for i,p in target.items()]))


def test_ellipse_equivalence_ui_complete_n8_special_codec_and_separate_verify():
    m=module()
    source={i:(i,i*i) for i in range(8)}
    target={100+i:(2*x+y+7,-x+3*y-4) for i,(x,y) in source.items()}
    text=_ellipse_equivalence_input(source,target)
    r=m['compute_ellipse_equivalence2019'](text,seconds=30)
    assert r['status']=='ELLIPSE_EQUIVALENT' and r['complete'] is True
    assert r['independent_check']=='NOT_RUN' and r['has_certificate']
    transport=json.loads(r['payload'])
    assert transport['schema']=='exact-research-ellipse-equivalence2019-v1'
    assert len(transport['certificate']['ratio_chunks'])==13
    restored=m['decode_new_payload'](transport)
    assert restored['input']['source']==source and restored['input']['target']==target
    assert len(restored['result']['certificate']['ratio_ledger'])==3080
    assert len(restored['result']['certificate']['source_code']['ellipses'])==56
    checked=m['verify_ellipse_equivalence2019_payload'](r['payload'],seconds=30)
    assert checked['status']=='VERIFIED' and checked['code_equivalent'] is True
    assert checked['theorem_domain'] is True and checked['affine_mapping_status']=='WITNESS_VERIFIED'
    assert m['verify_ellipse_equivalence2019_payload'](r['payload'],max_work=0)['status']=='UNKNOWN'


def test_ellipse_equivalence_small_negative_and_unknown_download():
    m=module()
    source={0:(0,0),1:(1,0),2:(0,1),3:(1,1)}
    target={10:(0,0),11:(3,0),12:(0,2),13:(4,4)}
    text=_ellipse_equivalence_input(source,target)
    r=m['compute_ellipse_equivalence2019'](text)
    assert r['status']=='NOT_ELLIPSE_EQUIVALENT' and r['complete'] is True
    assert r['independent_check']=='NOT_RUN' and r['rejected_correspondences']==24
    checked=m['verify_ellipse_equivalence2019_payload'](r['payload'])
    assert checked['status']=='VERIFIED' and checked['code_equivalent'] is False
    assert checked['theorem_conclusion']=='NOT_APPLICABLE'
    unknown=m['compute_ellipse_equivalence2019'](text,max_work=0)
    assert unknown['status']=='UNKNOWN' and unknown['complete'] is False
    assert unknown['independent_check']=='NOT_AVAILABLE' and not unknown['has_certificate']
    doc=m['decode_new_payload'](json.loads(unknown['payload']))
    assert doc['input']['source']==source and doc['input']['target']==target
    assert doc['result']['certificate'] is None
    assert m['verify_ellipse_equivalence2019_payload'](unknown['payload'])['status']=='NOT_AVAILABLE'


def test_ellipse_equivalence_outside_theorem_domain_preserves_code_predicate():
    m=module()
    source={0:(0,0),1:(1,0),2:(0,1),3:(1,1)}
    target={10:(0,0),11:(1,0),12:(1,1),13:(0,1)}
    r=m['compute_ellipse_equivalence2019'](_ellipse_equivalence_input(source,target))
    assert r['code_equivalent'] is True and r['theorem_domain'] is False
    assert r['theorem_conclusion']=='NOT_APPLICABLE'
    assert r['affine_mapping_status']=='NOT_ESTABLISHED'
    assert m['verify_ellipse_equivalence2019_payload'](r['payload'])['status']=='VERIFIED'


@pytest.mark.parametrize('mutation',['schema','extra','float','bool','duplicate','too_many','too_few'])
def test_ellipse_equivalence_ui_invalid_json_is_not_sampled(mutation):
    m=module();data=json.loads(m['ELLIPSE_EQUIVALENCE2019_EXAMPLE'])
    if mutation=='schema':data['schema']='ellipse-code2019-input-v1'
    elif mutation=='extra':data['extra']=0
    elif mutation=='float':data['source'][0]['xy'][0]=0.1
    elif mutation=='bool':data['source'][0]['label']=False
    elif mutation=='duplicate':data['source'][1]['label']=data['source'][0]['label']
    elif mutation=='too_many':data['source']=[dict(label=i,xy=[i,i*i]) for i in range(9)]
    else:data['source']=data['source'][:2]
    with pytest.raises(ValueError):m['compute_ellipse_equivalence2019'](json.dumps(data))


def test_ellipse_equivalence_degenerate_is_unsupported_with_input_download():
    m=module();data=json.loads(m['ELLIPSE_EQUIVALENCE2019_EXAMPLE'])
    data['source'][2]['xy']=[4,0]
    r=m['compute_ellipse_equivalence2019'](json.dumps(data))
    assert r['status']=='UNSUPPORTED' and not r['has_certificate']
    assert len(m['decode_new_payload'](json.loads(r['payload']))['input']['source'])==4
    assert m['verify_ellipse_equivalence2019_payload'](r['payload'])['status']=='NOT_AVAILABLE'


def test_ellipse_equivalence_apptest_two_buttons_cache_and_stale(monkeypatch):
    from kozlov_lab.coding import ellipse_equivalence2019 as algorithm
    search=algorithm.search_ellipse_equivalence2019;assess=algorithm.assess_ellipse_equivalence2019
    searches=[];checks=[]
    def tracked_search(*a,**kw):searches.append(1);return search(*a,**kw)
    def tracked_check(*a,**kw):checks.append(1);return assess(*a,**kw)
    monkeypatch.setattr(algorithm,'search_ellipse_equivalence2019',tracked_search)
    monkeypatch.setattr(algorithm,'assess_ellipse_equivalence2019',tracked_check)
    app=AppTest.from_file(str(PAGE),default_timeout=30).run()
    app.selectbox(key='exact_mode').select('2019椭圆代码未知对应等价').run()
    assert not app.exception and not searches and not checks
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(searches)==1 and not checks
    r=app.session_state['exact_result']['result']
    assert r['status']=='ELLIPSE_EQUIVALENT' and r['independent_check']=='NOT_RUN'
    assert len(app.get('download_button'))==2
    app.run();assert len(searches)==1 and not checks
    app.button(key='exact_ellipse_equivalence2019_verify').click().run()
    assert not app.exception and len(searches)==1 and len(checks)==1
    assert app.session_state['exact_result']['result']['independent_check']=='VERIFIED'
    assert any('作者平面定理域：True' in x.value for x in app.markdown)
    app.run();assert len(searches)==1 and len(checks)==1
    app.number_input(key='ellipse_equivalence2019_work').set_value(0).run()
    assert not app.get('download_button') and len(searches)==1
    app.button(key='exact_compute').click().run()
    assert not app.exception and len(searches)==2 and len(checks)==1
    assert app.session_state['exact_result']['result']['status']=='UNKNOWN'
    assert app.button(key='exact_ellipse_equivalence2019_verify').disabled
    app.text_area(key='exact_ellipse_equivalence2019_input').set_value('{}').run()
    assert not app.get('download_button')
    app.button(key='exact_compute').click().run()
    assert app.error and 'exact_result' not in app.session_state


def test_ellipse_equivalence_download_metadata_tamper_cannot_verify():
    m=module();r=m['compute_ellipse_equivalence2019'](m['ELLIPSE_EQUIVALENCE2019_EXAMPLE'])
    transport=json.loads(r['payload']);doc=m['decode_exact'](transport['document'])
    doc['result']['code_equivalent']=1
    transport['document']=m['encode_exact'](doc)
    assert m['verify_ellipse_equivalence2019_payload'](json.dumps(transport))['status']=='INVALID'


def test_ellipse_equivalence_serialization_budget_keeps_unknown_input(monkeypatch):
    from kozlov_lab.coding import ellipse_equivalence2019 as algorithm
    m=module()
    def exhausted(*a,**kw):raise algorithm._Exhausted('codec budget')
    monkeypatch.setattr(algorithm,'encode_ellipse_equivalence2019',exhausted)
    r=m['compute_ellipse_equivalence2019'](m['ELLIPSE_EQUIVALENCE2019_EXAMPLE'])
    assert r['status']=='UNKNOWN' and r['code_equivalent'] is None
    assert not r['has_certificate'] and r['independent_check']=='NOT_AVAILABLE'
    restored=m['decode_new_payload'](json.loads(r['payload']))
    assert len(restored['input']['source'])==4 and restored['result']['certificate'] is None


def test_ellipse_equivalence_ui_no_external_execution(monkeypatch):
    m=module()
    def forbidden(*a,**kw):raise AssertionError('external execution forbidden')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(socket,'socket',forbidden)
    before=set(sys.modules)
    r=m['compute_ellipse_equivalence2019'](m['ELLIPSE_EQUIVALENCE2019_EXAMPLE'])
    assert r['independent_check']=='NOT_RUN'
    assert m['verify_ellipse_equivalence2019_payload'](r['payload'])['status']=='VERIFIED'
    assert not any(n.split('.')[0] in {'torch','sam3'} for n in set(sys.modules)-before)
