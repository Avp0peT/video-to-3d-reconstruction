import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from integrations.video_masks import narrow_gap_review as review
from integrations.video_masks.narrow_gap import detect_narrow_gaps,evaluate_gap_preservation


def shape():
    base=np.zeros((64,64),bool);base[5:59,8:56]=True;base[5:34,30:34]=False
    return base


def test_guard_blocks_union_fill_without_removing_base():
    base=shape();valid=np.ones_like(base);diag=detect_narrow_gaps(base,valid)
    union=base.copy();union[25:28,30:34]=True
    candidate,guards=review.protect_union(base,union,valid,diag)
    assert not guards['before']['preserved'] and guards['after']['preserved']
    assert not np.any(base&~candidate) and not np.any(candidate&~union)
    assert not np.any(candidate&diag['protected_background'])


def test_input_identity_mismatch_is_refused():
    base=shape();valid=np.ones_like(base);diag=detect_narrow_gaps(base,valid)
    other=base.copy();other[40,40]=False
    with pytest.raises(ValueError):review.protect_union(other,base,valid,diag)


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'ROOT',tmp_path)
    # prepare() binds fixed product entries relative to ROOT.
    for name in ('scripts/protect_narrow_gaps.py','integrations/demo_pipeline/temporal_review.py'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
    source=tmp_path/'source';source.mkdir();(source/'frames').mkdir()
    originals=tmp_path/'originals';originals.mkdir()
    base=shape();support=base.copy();support[25:28,30:34]=True
    def png(name,a):
        p=source/name;Image.fromarray(a.astype(np.uint8)*255).save(p);return p
    b=png('base.png',base);s=png('support.png',support)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    image=originals/'frame.jpg';Image.new('RGB',(64,64),(50,80,90)).save(image)
    (source/'frames/000000.jpg').write_bytes(image.read_bytes())
    f=dict(decode_index=0,mode='union',base_mask_path=str(b),base_mask_sha256=sha(b),
        support_evidence=[dict(mask_path=str(s),mask_sha256=sha(s),root_seed_decode_index=2)])
    a=dict(decode_index=0,mask_path=str(s),mask_sha256=sha(s),fusion=f,status='FUSED_UNION_CANDIDATE',
        original_status='OK',selected_root_seed_decode_index=0)
    def save(p,value):p.write_text(json.dumps(value),encoding='utf8')
    save(source/'RESULT.json',dict(status='FULL_TIMELINE_FUSED_EXPORTED',mode='union',source_run=str(originals)))
    save(source/'ASSEMBLY.json',dict(rows=[a]));save(source/'FUSION.json',dict(rows=[f]))
    save(source/'PLAN.json',dict(frame_count=1,width=64,height=64))
    save(originals/'DECODE.json',dict(frames=[dict(decode_index=0,path=str(image),sha256=sha(image))]))
    return source,tmp_path/'outputs/run',b,s,image


def test_real_entry_byte_readback_defaults_and_no_relaunch(tmp_path,monkeypatch):
    source,out,b,s,image=fixture(tmp_path,monkeypatch)
    original={str(p):p.read_bytes() for p in (b,s,image)}
    review.prepare(source,out,[0]);result=review.execute(out)
    assert result['status']=='NARROW_GAP_CANDIDATES_NOT_ACCEPTED',result.get('error')
    assert result['summary']['candidate_guard_failures']==0
    assert result['summary']['blocked_union_additions']>0
    assert (out/'default_masks/000000.png').read_bytes()==s.read_bytes()
    assert result['published_for_reconstruction'] is False
    assert result['source_unchanged'] is True
    assert all(Path(p).read_bytes()==blob for p,blob in original.items())
    with pytest.raises(ValueError,match='claimed'):review.execute(out)


@pytest.mark.parametrize('bad',['duplicate','bool','unknown'])
def test_frame_selection_refuses_unbound_ids(tmp_path,monkeypatch,bad):
    source,out,*_=fixture(tmp_path,monkeypatch)
    indices={'duplicate':[0,0],'bool':[False],'unknown':[1]}[bad]
    with pytest.raises(ValueError):review.prepare(source,out,indices)
    assert not out.exists()


def test_rgb_drift_rejected_before_start(tmp_path,monkeypatch):
    source,out,*_=fixture(tmp_path,monkeypatch)
    review.prepare(source,out,[0]);(source/'frames/000000.jpg').write_bytes(b'bad')
    with pytest.raises(ValueError,match='drift'):review.execute(out)
    assert not (out/'START.json').exists()


def test_union_mismatch_cannot_publish_proposal(tmp_path,monkeypatch):
    source,out,b,s,image=fixture(tmp_path,monkeypatch)
    # A bound false producer claims extra pixels with no support: prepare accepts
    # declared bytes; execute must independently reproduce the OR and refuse.
    extra=source/'false_union.png';a=np.asarray(Image.open(s)).copy();a[60,60]=255;Image.fromarray(a).save(extra)
    doc=json.loads((source/'ASSEMBLY.json').read_text());doc['rows'][0].update(mask_path=str(extra),mask_sha256=hashlib.sha256(extra.read_bytes()).hexdigest())
    (source/'ASSEMBLY.json').write_text(json.dumps(doc))
    review.prepare(source,out,[0]);result=review.execute(out)
    assert result['status']=='FAILED' and 'reproduce' in result['error']
    assert not list((out/'proposals').iterdir())
    assert (out/'default_masks/000000.png').read_bytes()==extra.read_bytes()


def test_completed_inline_union_uses_own_decode(tmp_path,monkeypatch):
    source,out,b,s,image=fixture(tmp_path,monkeypatch)
    original=json.loads((source/'RESULT.json').read_text())['source_run']
    (source/'DECODE.json').write_bytes((Path(original)/'DECODE.json').read_bytes())
    (source/'RESULT.json').write_text(json.dumps(dict(status='FULL_TIMELINE_EXPORTED',fusion={'mode':'union'})))
    p=json.loads((source/'PLAN.json').read_text());p['mask_fusion_mode']='union'
    (source/'PLAN.json').write_text(json.dumps(p))
    prepared=review.prepare(source,out,[0])
    assert prepared['frames'][0]['original_image']==str(image)


def test_inline_nonunion_is_not_silently_accepted(tmp_path,monkeypatch):
    source,out,*_=fixture(tmp_path,monkeypatch)
    (source/'RESULT.json').write_text(json.dumps(dict(status='FULL_TIMELINE_EXPORTED',fusion={'mode':'supported-holes'})))
    p=json.loads((source/'PLAN.json').read_text());p['mask_fusion_mode']='supported-holes'
    (source/'PLAN.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='literal-union'):review.prepare(source,out,[0])
