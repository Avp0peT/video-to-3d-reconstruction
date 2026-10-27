import json
from PIL import Image
from integrations.demo_pipeline.short_tracklet_report import render_tracklets


def _data(tmp_path):
    decoded={}
    for i in range(4):
        path=tmp_path/f'raw{i}.png';Image.new('RGB',(200,150),'black').save(path)
        decoded[i]=dict(path=str(path),width=200,height=150,original_mask_status='OK' if i==0 else 'UNKNOWN_NOT_SEGMENTED',mask=object() if i==0 else None)
    observations=[dict(decode_index=i,raw_xy=[10.5+i,20.5],width=200,height=150,path=decoded[i]['path'],time_seconds=i/30,original_mask_status=decoded[i]['original_mask_status']) for i in range(3)]
    record=dict(tracklet_id='p:edge:0',point_id='p',edge_ordinal=0,source_point=dict(source_decode_index=0,source_raw_xy=[10.5,20.5]),observations=observations,verified_legs=2,status='CANDIDATE_2D_TRACKLET',termination_reason='LK_STATUS_FAILED',wholecycle_status='NOT_COMPLETED',wholecycle_reasons=['later failure'])
    return decoded,dict(status='COMPLETE',records=[record],counts=dict(CANDIDATE_2D_TRACKLET=1))


def test_prefix_retained_without_future_prediction(tmp_path):
    decoded,ledger=_data(tmp_path)
    geometry=dict(records=[dict(tracklet_id='p:edge:0',status='BLOCKED',model_assessments=[dict(model_key='m',status='MISSING_POSES',missing_camera_decode_indices=[1,2],unknown_mask_decode_indices=[1,2])])],counts=dict(BLOCKED=1))
    paths=render_tracklets(tmp_path/'report',ledger,geometry,decoded)
    assert {'index.html','evidence.json','prefix_000.png','source_000.png'} <= {p.name for p in paths}
    evidence=json.loads((tmp_path/'report'/'evidence.json').read_text(encoding='utf-8'))
    record=evidence['records'][0]
    assert record['verified_legs']==2 and record['termination_reason']=='LK_STATUS_FAILED'
    assert [o['decode_index'] for o in record['observations']]==[0,1,2]
    assert record['geometry']['model_assessments'][0]['missing_camera_decode_indices']==[1,2]
    assert all('raw3' not in o['url'] for o in record['observations'])
    page=(tmp_path/'report'/'index.html').read_text(encoding='utf-8')
    assert 'NOT_OBSERVED' in page and '不是新增三维点' in page and '后续失败不抹去已有前缀' in page
    with Image.open(tmp_path/'report'/'prefix_000.png') as im:
        colors=set(im.getdata())
    assert (80,168,255) in colors and (68,237,121) in colors


def test_missing_measurements_remain_excluded(tmp_path):
    decoded,ledger=_data(tmp_path)
    ledger['records'][0].update(observations=[],verified_legs=0,status='EXCLUDED',termination_reason='NOT_MEASURED_OR_INTERRUPTED')
    render_tracklets(tmp_path/'report',ledger,{},decoded)
    evidence=json.loads((tmp_path/'report'/'evidence.json').read_text(encoding='utf-8'))['records'][0]
    assert evidence['observations']==[] and evidence['geometry']['status']=='NOT_EVALUATED'
    assert not (tmp_path/'report'/'prefix_000.png').exists()


def test_embedded_script_escaped_and_empty(tmp_path):
    decoded,ledger=_data(tmp_path)
    ledger['records'][0]['point_id']='</script><script>attack()</script>'
    render_tracklets(tmp_path/'report',ledger,{},decoded)
    page=(tmp_path/'report'/'index.html').read_text(encoding='utf-8')
    assert ledger['records'][0]['point_id'] not in page
    evidence=json.loads((tmp_path/'report'/'evidence.json').read_text(encoding='utf-8'))
    assert evidence['records'][0]['point_id']==ledger['records'][0]['point_id']
    render_tracklets(tmp_path/'empty',dict(records=[]),{}, {})
    assert json.loads((tmp_path/'empty'/'evidence.json').read_text(encoding='utf-8'))['records']==[]
