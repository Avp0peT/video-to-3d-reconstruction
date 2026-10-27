"""Synthetic CPU evidence only; no real-video execution."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.video_masks import region_tracks as rt, region_reverse as reverse


@pytest.fixture(scope='module')
def cv():
    value=rt.load_cv([])
    if value is None:
        pytest.skip('bound OpenCV unavailable')
    return value


def prepare(tmp_path,cv):
    gray=cv.GaussianBlur(np.random.default_rng(20).integers(30,180,(96,96),dtype=np.uint8),(3,3),.7)
    timeline=[]
    for n in range(3):
        image=cv.warpAffine(gray,np.float32([[1,0,n],[0,1,n]]),(96,96))
        row=dict(decode_index=n,input_index=n,width=96,height=96)
        for kind,array in (('image',image),('mask',np.full_like(gray,255))):
            path=tmp_path/f'{kind}{n}.png'; Image.fromarray(array).save(path)
            row[kind+'_path']=str(path); row[kind+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        timeline.append(row)
    path=tmp_path/'plan.json'
    path.write_text(json.dumps(dict(schema='glove-region-plan-v1',profile=rt.PROFILE,timeline=timeline)))
    forward=rt.run(path,tmp_path/'forward')
    return path,forward,tmp_path/'forward'/'REGION_TRACKS.json'


def test_reverse_translation_adds_isolated_ids_and_preserves_forward(tmp_path,cv):
    plan,old,forward=prepare(tmp_path,cv)
    result=reverse.run(plan,forward,tmp_path/'reverse')
    assert result['reverse_track_ids']
    assert min(result['reverse_track_ids'])>max(t['track_id'] for t in old['tracks'])
    assert result['tracks'][:len(old['tracks'])]==old['tracks']
    for before,after in zip(old['frames'],result['frames']):
        assert {k:v for k,v in before.items() if k!='observations'}=={k:v for k,v in after.items() if k!='observations'}
        assert after['observations'][:len(before['observations'])]==before['observations']
    assert all(len(t['observations'])>=2 and t['origin']=='reverse_anchor_validated_candidate' for t in result['tracks'][len(old['tracks']):])
    assert Path(result['source_bindings'][0]['path'])==plan
    bound={Path(r['path']).name for r in result['source_bindings']}
    assert {'region_reverse.py','region_anchor.py','REGION_TRACKS.json'} <= bound
    progress=[json.loads(s) for s in (tmp_path/'reverse'/'progress.jsonl').read_text().splitlines()]
    assert [s['decode_index'] for s in progress if s['event']=='REVERSE_FRAME']==[2,1,0]
    assert progress[-1]['event']=='FINAL'


def test_unknown_splits_no_interpolation_or_merge():
    observations=[dict(decode_index=n,xy=[n,0]) for n in range(5)]
    segments=reverse.split_segments(observations,{(0,1),(2,3)})
    assert [[o['decode_index'] for o in segment] for segment in segments]==[[0,1],[2,3]]
    assert reverse.split_segments(observations,set())==[]


def test_unknown_anchor_adds_nothing_preserves_original(tmp_path,cv,monkeypatch):
    plan,old,forward=prepare(tmp_path,cv)
    original=reverse.anchor.audit_pair
    def unknown(*args):
        result=original(*args)
        for row in result['candidates']:
            row['status']='unknown'; row['reasons']=['TEST_DATA_unobservable']
        return result
    monkeypatch.setattr(reverse.anchor,'audit_pair',unknown)
    result=reverse.run(plan,forward,tmp_path/'reverse')
    assert result['frames']==old['frames'] and result['tracks']==old['tracks']
    assert result['reverse_track_ids']==[]
    audit=json.loads((tmp_path/'reverse'/'REVERSE_AUDIT.json').read_text())
    assert any(row['reasons']==['TEST_DATA_unobservable'] for pair in audit['endpoint_pair_audits'] for row in pair['candidates'])


def test_input_sha_drift_rejected(tmp_path,cv):
    plan,_,forward=prepare(tmp_path,cv)
    document=json.loads(plan.read_text())
    path=Path(document['timeline'][-1]['image_path'])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        reverse.run(plan,forward,tmp_path/'reverse')
