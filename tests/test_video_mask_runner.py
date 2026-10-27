"""CPU fixtures for committed-result review; never loads a model."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest
from integrations.video_masks.runner import review, ResourceWatch, complete_records
from integrations.demo_pipeline.pipeline import sha


def fixture(tmp):
    frames=tmp/'frames';frames.mkdir()
    worker=tmp/'worker';worker.mkdir()
    image=frames/'000000.jpg';Image.new('RGB',(4,6),'gray').save(image)
    commits=[]
    for direction in ('forward','reverse'):
        folder=worker/direction;folder.mkdir()
        path=folder/'000000.png';Image.fromarray(np.full((6,4),255,dtype=np.uint8)).save(path)
        commits.append(dict(direction=direction,frame_id=0,mask_path=str(path),mask_sha256=sha(path)))
    (worker/'result.json').write_text(json.dumps(dict(status='PROPAGATED_NOT_SEMANTICALLY_EVALUATED',frames=commits)))
    return dict(frame_count=1,start_decode_index=363,end_decode_index=363,width=4,height=6,
                targets=[dict(local_frame_index=0,input_index=22,original_status='NO_MATCH')])


def test_review_missing_mask_stays_separate(tmp_path):
    plan=fixture(tmp_path)
    result=review(tmp_path,plan)
    assert result['candidate_count']==1
    assert result['rows'][0]['original_status']=='NO_MATCH'
    assert result['published_for_reconstruction'] is False
    assert (tmp_path/'motion.gif').is_file()
    assert (tmp_path/'index.html').is_file()


def test_review_missing_direction_is_not_accepted(tmp_path):
    plan=fixture(tmp_path)
    path=tmp_path/'worker/result.json'; value=json.loads(path.read_text())
    value['frames']=value['frames'][:1];path.write_text(json.dumps(value))
    assert review(tmp_path,plan)['candidate_count']==0


def test_review_committed_hash_is_required(tmp_path):
    plan=fixture(tmp_path)
    (tmp_path/'worker/forward/000000.png').write_bytes(b'bad')
    with pytest.raises(ValueError,match='identity drift'):
        review(tmp_path,plan)


def test_device_query_failure_latches(tmp_path,monkeypatch):
    from integrations.sam3.sam3_adapter import observer
    monkeypatch.setattr(observer,'resolve_device_identity',lambda _:dict(identity_provable=True,physical_index=0))
    monkeypatch.setattr(observer,'driver_device_query',lambda *a,**k:dict(fresh=False,device_used_bytes=None,error='TEST_DATA'))
    monitor=ResourceWatch(tmp_path,'cuda:0');monitor.start()
    assert monitor.reason and monitor.threads==[]
    assert monitor.close()['joined']


def test_incremental_jsonl_tail_is_not_monitor_failure(tmp_path):
    path=tmp_path/'events.jsonl'
    path.write_text('{"pid": 42}\n{"pid":',encoding='utf-8')
    assert complete_records(path)==[{'pid':42}]
    path.write_text('{bad}\n',encoding='utf-8')
    with pytest.raises(ValueError): complete_records(path)
