"""Pure bounded theoretical CLI; complete differently labelled rational views."""
import json
from fractions import Fraction as F
from pathlib import Path
import runpy

import pytest

CLI=Path(__file__).resolve().parents[1]/'scripts/search_parallel_correspondence.py'


def module(): return runpy.run_path(str(CLI))


def fixture(n=5):
    # Independent rational 3D truth projected to (x,y), ((x+z)/2,(y+2z)/3).
    # The algorithm receives only the two differently labelled projections.
    truth=[(0,0,0),(1,0,0),(0,1,0),(1,1,1),(2,3,2)][:n]
    first=[42,-3,7,11,19][:n];second=[108,77,-10,55,92][:n]
    source=[dict(label=i,xy=[x,y]) for i,(x,y,z) in zip(first,truth)]
    def q(n,d):
        value=F(n,d)
        return value.numerator if value.denominator==1 else dict(numerator=str(value.numerator),denominator=str(value.denominator))
    target=[dict(label=i,xy=[q(x+z,2),q(y+2*z,3)]) for i,(x,y,z) in zip(second,truth)]
    return dict(schema='bounded-parallel-correspondence-input-v1',model='exact-parallel-theory',
                source_xy=list(reversed(source)),target_xy=target[2:]+target[:2])


@pytest.mark.parametrize('n',[4,5])
def test_theoretical_four_five_view_search_readback_verify_and_no_overwrite(tmp_path,n):
    m=module();source=tmp_path/'views.json';output=tmp_path/'result.json';verified=tmp_path/'verified.json'
    source.write_text(json.dumps(fixture(n)),encoding='utf-8')
    argv=['--input',str(source),'--output',str(output)]
    assert m['main'](argv)==0
    saved=json.loads(output.read_text(encoding='utf-8'))
    assert saved['search_status']=='SEARCH_COMPLETE' and saved['search_complete'] is True
    assert saved['solutions_count']>0 and saved['unsupported_count']==0
    assert saved['independent_check']['status']=='VERIFIED' and saved['argv']==argv
    restored=m['decode_result_transport'](saved['result'])
    assert len(restored['ledger'])==(24 if n==4 else 120)
    assert len(restored['accepted'])==saved['solutions_count']
    assert m['decode_result_transport'](json.loads(json.dumps(m['encode_result_transport'](restored))))==restored
    assert m['main'](['--input',str(source),'--verify',str(output),'--output',str(verified)])==0
    assert json.loads(verified.read_text(encoding='utf-8'))['independent_check']['status']=='VERIFIED'
    original=output.read_bytes()
    assert m['main'](argv)==2 and output.read_bytes()==original


def test_budget_partial_never_means_complete_reconstruction(tmp_path):
    m=module();source=tmp_path/'input.json';source.write_text(json.dumps(fixture(4)),encoding='utf-8')
    for name,option in [('zero',['--max-work','0']),('one_candidate',['--max-candidates','1'])]:
        output=tmp_path/(name+'.json')
        assert m['main'](['--input',str(source),'--output',str(output)]+option)==1
        data=json.loads(output.read_text(encoding='utf-8'))
        assert data['search_status']=='PARTIAL_UNKNOWN' and data['search_complete'] is False
        assert data['physical_scale']=='unknown'
        if name=='zero': assert data['solutions_count']==0


@pytest.mark.parametrize('kind',['float','bool','duplicate_label','duplicate_coordinate','size','model','unequal','oversize_rational'])
def test_strict_complete_input_rejection(tmp_path,kind):
    m=module();data=fixture(4)
    if kind=='float':data['source_xy'][0]['xy'][0]=0.0
    elif kind=='bool':data['target_xy'][0]['label']=False
    elif kind=='duplicate_label':data['source_xy'][0]['label']=data['source_xy'][1]['label']
    elif kind=='duplicate_coordinate':data['target_xy'][0]['xy']=data['target_xy'][1]['xy']
    elif kind=='size':data['source_xy']=data['source_xy'][:3]
    elif kind=='model':data['model']='photographs'
    elif kind=='unequal':data=fixture(5);data['source_xy']=data['source_xy'][:4]
    elif kind=='oversize_rational':data['source_xy'][0]['xy'][0]={'numerator':'1000000000001','denominator':'1'}
    source=tmp_path/'invalid.json';output=tmp_path/'out.json'
    source.write_text(json.dumps(data),encoding='utf-8')
    assert m['main'](['--input',str(source),'--output',str(output)])==2
    assert not output.exists()


def test_duplicate_json_keys_nonfinite_and_codec_identity(tmp_path):
    m=module();source=tmp_path/'bad.json';output=tmp_path/'out.json'
    source.write_text('{"schema":"a","schema":"b"}',encoding='utf-8')
    assert m['main'](['--input',str(source),'--output',str(output)])==2
    source.write_text('{"value":NaN}',encoding='utf-8')
    assert m['main'](['--input',str(source),'--output',str(output)])==2
    result=dict(status='SEARCH_COMPLETE',ledger=tuple(dict(status='ACCEPTED',mapping=((i,i+100),)) for i in range(30)))
    result['accepted']=result['ledger']
    encoded=m['encode_result_transport'](result)
    assert m['decode_result_transport'](json.loads(json.dumps(encoded)))==result
    encoded['accepted_indices'][0]=False
    with pytest.raises(ValueError):m['decode_result_transport'](encoded)


def test_verification_source_byte_binding_and_no_search(tmp_path,monkeypatch):
    import kozlov_lab.stereo.official.bounded_parallel_correspondence as backend
    m=module();source=tmp_path/'source.json';output=tmp_path/'saved.json'
    source.write_text(json.dumps(fixture(4)),encoding='utf-8')
    assert m['main'](['--input',str(source),'--output',str(output)])==0
    def forbidden(*a,**kw): raise AssertionError('search forbidden during independent verification')
    monkeypatch.setattr(backend,'search_bounded_parallel_correspondence',forbidden)
    assert m['main'](['--input',str(source),'--verify',str(output),'--output',str(tmp_path/'checked.json')])==0
    source.write_text(json.dumps(fixture(4),indent=2),encoding='utf-8')
    assert m['main'](['--input',str(source),'--verify',str(output),'--output',str(tmp_path/'mismatch.json')])==2


def test_cli_no_process_network_photo_or_model_import(tmp_path,monkeypatch):
    import subprocess
    import socket
    import sys
    m=module();source=tmp_path/'input.json';output=tmp_path/'output.json'
    source.write_text(json.dumps(fixture(4)),encoding='utf-8')
    def forbidden(*a,**kw):raise AssertionError('external execution forbidden')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(socket,'socket',forbidden)
    before=set(sys.modules)
    assert m['main'](['--input',str(source),'--output',str(output)])==0
    assert not any(name.split('.')[0] in ('PIL','torch','sam3') for name in set(sys.modules)-before)
