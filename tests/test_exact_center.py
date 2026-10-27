import copy
from fractions import Fraction as F
import json
from pathlib import Path
import runpy

import pytest
from kozlov_lab.kozlov2025 import exact_center as mod

GENERAL={0:(0,0),1:(1,0),2:(0,1),3:(2,F(3,2))}
FIVE={0:(1,0),1:(F(-3,5),F(4,5)),2:(F(-3,5),F(-4,5)),3:(0,1),4:(0,-1)}
CLI=Path(__file__).resolve().parents[1]/'scripts'/'solve_exact_center.py'


@pytest.mark.parametrize('points,profile',[
    ({17:(F(1,2),2)},'point'),({0:(0,0),1:(1,2),2:(2,4)},'segment'),
    ({0:(0,0),1:(3,0),2:(0,3),3:(1,1)},'triangle'),
    ({0:(0,0),1:(3,0),2:(0,3),3:(2,2)},'contact3'),(GENERAL,'contact4'),(FIVE,'contact5')])
def test_exact_profiles_and_tagged_identity(points,profile):
    answer=mod.solve_exact_center(points)
    assert answer['exact_optimum'],answer
    assert answer['profile']==profile
    tagged=mod.encode_exact(answer)
    restored=mod.decode_exact(json.loads(json.dumps(tagged)))
    assert restored==answer
    assert mod.verify_exact_center_result(points,restored)
    assert restored['declared_labels']==sorted(points)
    if profile=='contact4':
        assert answer['number_context']['domain']=='real-algebraic'
        assert len(answer['number_context']['algebraic_root']['poly'])==4
        assert all(type(i) is int for i in restored['certificate']['all_input_quadratic_values'])


def test_independent_verifier_forbids_all_constructors(monkeypatch):
    answers=[(p,mod.solve_exact_center(p)) for p in (GENERAL,FIVE,{0:(0,0),1:(2,2)})]
    def forbidden(*a,**k): raise AssertionError('constructor forbidden')
    monkeypatch.setattr(mod,'solve_exact_center',forbidden)
    for module,name in ((mod.triangle,'exact_triangle_enclosing_ellipse'),(mod.contact3,'complete_input_contact3_ellipse'),
                        (mod.contact5,'complete_input_contact5_ellipse'),(mod.contact4,'complete_input_contact4_ellipse')):
        monkeypatch.setattr(module,name,forbidden)
    for points,answer in answers: assert mod.verify_exact_center_result(points,answer)


def test_pollution_and_source_binding():
    answer=mod.solve_exact_center(GENERAL)
    for field in ('source_coordinate_sha256','number_context','center','major_axis_length'):
        bad=copy.deepcopy(answer)
        bad[field]='polluted'
        assert not mod.verify_exact_center_result(GENERAL,bad)
    changed=dict(GENERAL);changed[4]=(0,0)
    assert not mod.verify_exact_center_result(changed,answer)
    segment=mod.solve_exact_center({0:(0,0),1:(1,1),2:(2,2)})
    bad=copy.deepcopy(segment);bad['certificate']['endpoint_labels']=[0,1]
    assert not mod.verify_exact_center_result({0:(0,0),1:(1,1),2:(2,2)},bad)


@pytest.mark.parametrize('points',[
    {},{0:(0.0,0)},{False:(0,0)},{0:(True,0)},
    {0:(0,0),1:(0,0)},{i:(i,0) for i in range(8)}, {0:(F(1,2**2049),0)}])
def test_strict_input_refusal(points):
    answer=mod.solve_exact_center(points)
    assert answer['status']=='UNSUPPORTED_INPUT' and not answer['exact_optimum']


def test_one_shared_budget_and_verify_budget():
    for kw in ({'max_work':0},{'max_work':400},{'seconds':0}):
        answer=mod.solve_exact_center(GENERAL,**kw)
        assert answer['status']=='PARTIAL_UNKNOWN' and not answer['exact_optimum']
    answer=mod.solve_exact_center(GENERAL)
    assert mod.assess_exact_center_result(GENERAL,answer,max_work=0)['status']=='UNKNOWN'
    assert not mod.verify_exact_center_result(GENERAL,answer,max_work=0)


def _input_json():
    return dict(schema='exact-center-input-v1',points=[dict(label=i,xy=[
        v if type(v) is int else dict(numerator=str(v.numerator),denominator=str(v.denominator)) for v in p]) for i,p in GENERAL.items()])


def test_cli_positive_readback_and_refuse_overwrite(tmp_path):
    cli=runpy.run_path(str(CLI)); source=tmp_path/'input.json'; output=tmp_path/'result.json'; checked=tmp_path/'checked.json'
    source.write_text(json.dumps(_input_json()),encoding='utf-8')
    argv=['--input',str(source),'--output',str(output)]
    assert cli['main'](argv)==0
    saved=json.loads(output.read_text(encoding='utf-8'))
    assert saved['schema']=='exact-center-cli-result-v1' and saved['argv']==argv
    result=mod.decode_exact(saved['result'])
    assert mod.verify_exact_center_result(GENERAL,result)
    assert saved['serialized_independent_check']['status']=='VERIFIED'
    assert cli['main'](['--input',str(source),'--verify',str(output),'--output',str(checked)])==0
    original=output.read_bytes()
    assert cli['main'](argv)==2 and output.read_bytes()==original


@pytest.mark.parametrize('text',[
    '{"schema":"exact-center-input-v1","schema":"exact-center-input-v1","points":[]}',
    '{"schema":"exact-center-input-v1","points":[{"label":0,"xy":[true,0]}]}',
    '{"schema":"exact-center-input-v1","points":[{"label":0,"xy":[0.0,0]}]}',
    '{"schema":"exact-center-input-v1","points":[{"label":0,"xy":[NaN,0]}]}'])
def test_cli_rejects_duplicate_keys_and_numeric_drift(tmp_path,text):
    source=tmp_path/'bad.json';source.write_text(text,encoding='utf-8');output=tmp_path/'out.json'
    assert runpy.run_path(str(CLI))['main'](['--input',str(source),'--output',str(output)])==2
    assert not output.exists()


def test_decoder_refuses_duplicate_integer_identity_and_float():
    with pytest.raises(ValueError): mod.decode_exact({'$exact':'integer-map','items':[[1,0],[1,1]]})
    with pytest.raises(ValueError): mod.decode_exact(0.0)
