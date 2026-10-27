"""Test the actual automatic pipeline's coherence wiring with labelled test data."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from integrations.video_masks import motion_part_pipeline as pipeline


def raw_fixture(images,masks,**kwargs):
    n=len(images); shape=masks[0].shape
    parts=[]
    for _ in images:
        m=np.zeros(shape,bool);m[5:15,5:15]=True;m[28:38,35:45]=True;parts.append(m)
    xy=[[7,7],[10,7],[7,10],[10,10],[37,30],[40,30],[37,33],[40,33]]
    pairs=[dict(pair=[i,i+1],track_ids=list(range(8)),source_points=xy,destination_points=xy) for i in range(n-1)]
    return dict(status='SYNTHETIC_TEST_DATA',global_masks=masks,part_masks={'raw':parts},
        fallback_masks=[m&~p for m,p in zip(masks,parts)],
        unknown_masks=[np.zeros(shape,bool) for _ in images],evidence_domain_masks=masks,
        ledger=dict(parts=[dict(id='raw',track_ids=list(range(8)))],pairs=pairs),
        metrics=[dict(global_pixels=int(m.sum()),part_pixels=200,unknown_pixels=0) for m in masks])


def prepare(tmp_path, count=3):
    source=tmp_path/'source';source.mkdir()
    rows=[]
    for i in range(count):
        image=source/f'{i}.jpg'; mask=source/f'{i}.png'
        Image.new('RGB',(60,45),(80,80,80)).save(image)
        Image.new('L',(60,45),255).save(mask)
        rows.append(dict(frame_id=str(i),time_seconds=i,
            image=dict(path=str(image),sha256=pipeline.sha(image)),
            mask=dict(path=str(mask),sha256=pipeline.sha(mask))))
    manifest=source/'input.json'
    pipeline.save(manifest,dict(schema='motion-part-input-v1',object_id='TEST_DATA',frames=rows))
    return manifest


def test_default_automatic_entry_splits_remote_components_and_writes_raw_control(tmp_path,monkeypatch):
    source=prepare(tmp_path)
    monkeypatch.setattr(pipeline,'propose_motion_parts',raw_fixture)
    result=pipeline.run(source,tmp_path/'run')
    assert result['coherence_refinement'] is True
    window=result['windows'][0]
    assert len(window['raw_part_ids'])==1 and len(window['part_ids'])==2
    for row in window['frames']:
        assert row['raw_candidate_pixels']==row['refined_candidate_pixels']==200
        assert row['candidate_returned_to_global_pixels']==0
        assert len(row['parts'])==2 and len(row['raw_parts'])==1
    assert (tmp_path/'run/window_000/COHERENCE_EVIDENCE.json').exists()
    assert '原运动候选' in (tmp_path/'run/index.html').read_text(encoding='utf8')


def test_explicit_raw_control_does_not_silently_apply_refinement(tmp_path,monkeypatch):
    source=prepare(tmp_path)
    monkeypatch.setattr(pipeline,'propose_motion_parts',raw_fixture)
    result=pipeline.run(source,tmp_path/'run',coherence=False)
    assert result['coherence_refinement'] is False
    assert result['windows'][0]['part_ids']==['raw']
    assert not (tmp_path/'run/window_000/COHERENCE_EVIDENCE.json').exists()
    assert result['cross_window_association']['status']=='NOT_RUN_RAW_CONTROL'
    assert not (tmp_path/'run/window_links').exists()


def test_automatic_linking_runs_through_real_files_without_changing_masks(tmp_path,monkeypatch):
    source=prepare(tmp_path,7)
    monkeypatch.setattr(pipeline,'propose_motion_parts',raw_fixture)
    result=pipeline.run(source,tmp_path/'run')
    assert result['cross_window_association']['accepted_link_count']==2
    assert result['cross_window_identity']=='CANDIDATE_LINKS_ONLY_NOT_PHYSICAL_IDENTITY'
    links=pipeline.strict_json(tmp_path/'run/window_links/RESULT.json')
    assert links['physical_identity']=='NOT_ESTABLISHED'
    assert len(links['groups'])==2
    assert all(len(g['members'])==2 for g in links['groups'])
    assert links['previews']
    inputs=pipeline.strict_json(source)['frames']
    for window in result['windows']:
        for frame in window['frames']:
            assert pipeline.sha(frame['whole']['path'])==inputs[frame['input_position']]['mask']['sha256']
            for ref in frame['parts'].values():
                assert pipeline.sha(ref['path'])==ref['sha256']
    assert not links['masks_modified'] and not links['geometry_accepted']


def test_saved_candidate_hash_drift_refuses_link_publication(tmp_path,monkeypatch):
    from integrations.video_masks.motion_window_review import build_review
    source=prepare(tmp_path,7)
    monkeypatch.setattr(pipeline,'propose_motion_parts',raw_fixture)
    result=pipeline.run(source,tmp_path/'run')
    frame=result['windows'][0]['frames'][0]
    ref=next(iter(frame['parts'].values()))
    Path(ref['path']).write_bytes(b'TAMPERED_SYNTHETIC_TEST_DATA')
    _,rows,_=pipeline.validate_manifest(source)
    with pytest.raises(ValueError,match='hash drift'):
        build_review(result,rows,tmp_path/'run',tmp_path/'run/tampered_links')
    assert not (tmp_path/'run/tampered_links/RESULT.json').exists()


def test_second_ledger_read_cannot_replace_initial_binding(tmp_path,monkeypatch):
    from integrations.video_masks.motion_window_review import _load_window
    source=prepare(tmp_path,7)
    monkeypatch.setattr(pipeline,'propose_motion_parts',raw_fixture)
    result=pipeline.run(source,tmp_path/'run')
    _,rows,_=pipeline.validate_manifest(source)
    refs={}; record=result['windows'][0]; root=tmp_path/'run'
    _load_window(record,rows,root,refs,lambda:None)
    ledger=root/'window_000/MOTION_EVIDENCE.json'
    original=refs[str(ledger)]
    data=pipeline.strict_json(ledger);data['interpretation']='changed after initial read'
    pipeline.save(ledger,data)
    with pytest.raises(ValueError,match='drift between reads'):
        _load_window(record,rows,root,refs,lambda:None)
    assert refs[str(ledger)]==original
