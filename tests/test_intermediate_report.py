"""Report counterexamples: rejected arms must remain visibly rejected."""
from pathlib import Path

from PIL import Image
import pytest

from integrations.demo_pipeline.intermediate_report import render_report


def fixture(tmp_path):
    source = tmp_path / 'raw images'
    source.mkdir()
    decoded = {}
    for i in range(3):
        path = source / f'{i}.png'
        Image.new('RGB', (40, 80), (30 + i * 20, 50, 90)).save(path)
        decoded[i] = dict(path=str(path), width=40, height=80, pts=i*10,
                          time_base='1/100', mask=None if i == 1 else True,
                          original_mask_status='UNKNOWN_NOT_SEGMENTED' if i == 1 else 'OK')
    def arm(passed):
        return dict(status='PASSED' if passed else 'REJECTED', reasons=[] if passed else ['LK_ERROR_THRESHOLD'],
                    observed_raw_coordinates=[dict(decode_index=0, raw_xy=[10.5,20.5]),dict(decode_index=2,raw_xy=[12.5,22.5])],
                    steps=[dict(forward=dict(to_decode_index=1,observed_raw_xy=[11.5,21.5]))],
                    fullcycle=dict(status='PASSED' if passed else 'NOT_COMPLETED',error_px=0 if passed else None))
    rows = [dict(group_id='<group&one>',direct=arm(False),dense=arm(True),outcome='RECOVERED'),
            dict(group_id='failed',direct=arm(False),dense=arm(False),outcome='BOTH_FAIL')]
    edge = dict(model_key='model',source_input_index=4,target_input_index=8,source_decode_index=0,target_decode_index=2,decode_path=[0,1,2],points=[])
    result = dict(status='COMPARED',profile={},plan=dict(edges=[edge]),edges=[dict(rows=rows,counts={'BOTH_PASS':999},flow_calls=8)],extraction={'scaled':True})
    output = tmp_path/'report'
    output.mkdir()
    return output,result,decoded


def test_report_raw_links_status_counts_and_aspect_ratio(tmp_path):
    output,result,decoded = fixture(tmp_path)
    artifacts = render_report(output,result,decoded)
    assert {p.name for p in artifacts} == {'index.html','sequence_000.png','tracking_000.png','crop_000_001.png','crop_000_002.png'}
    assert all(p.is_file() for p in artifacts)
    page = (output/'index.html').read_text(encoding='utf-8')
    assert '../raw%20images/1.png' in page
    assert 'UNKNOWN / NOT SEGMENTED' in page and 'DIAGNOSIS ONLY' in page
    assert 'REJECTED' in page and 'LK_ERROR_THRESHOLD' in page and 'NOT_COMPLETED' in page
    assert '&lt;group&amp;one&gt;' in page and '<group&one>' not in page
    assert "&#x27;RECOVERED&#x27;: 1" in page and "&#x27;BOTH_PASS&#x27;: 0" in page
    assert '999' not in page  # never present stale supplied counts as observations
    with Image.open(output/'sequence_000.png') as image:
        assert image.size == (1616,570)
        # The complete 40x80 portrait is centered, with no stretching or crop.
        assert image.getpixel((182,48)) == (30,50,90)
        assert image.getpixel((182,127)) == (30,50,90)
        assert image.getpixel((181,48)) == (21,26,34)
    with Image.open(output/'tracking_000.png') as tracking, Image.open(output/'sequence_000.png') as sequence:
        assert tracking.tobytes() != sequence.tobytes()


def test_crop_missing_coordinate_is_gray_and_selection_is_bounded(tmp_path):
    output,result,decoded = fixture(tmp_path)
    first = result['edges'][0]['rows'][0]
    first['dense']['steps'] = []
    first['direct']['steps'] = []
    result['edges'][0]['rows'].extend([dict(first,group_id=f'extra{i}') for i in range(4)])
    artifacts = render_report(output,result,decoded)
    crops = [p for p in artifacts if p.name.startswith('crop_')]
    assert len(crops) == 3  # first two recovered and first one both-failed
    assert {p.name for p in crops} == {'crop_000_001.png','crop_000_002.png','crop_000_003.png'}
    with Image.open(output/'crop_000_001.png') as image:
        assert image.size == (336,292)
        # No intermediate measurement: the crop interior remains gray rather
        # than showing a copied/predicted source or target patch.
        assert image.getpixel((112+50,60)) == (85,85,85)
        assert image.getpixel((112+50,146+60)) == (85,85,85)
        # A measured source has real raw grayscale pixels away from the cross.
        assert image.getpixel((55,90)) != (85,85,85)


def test_budget_propagates_and_unrun_edge_is_explicit(tmp_path):
    output,result,decoded = fixture(tmp_path)
    result['edges'] = []
    render_report(output,result,decoded)
    assert 'NOT_RUN' in (output/'index.html').read_text(encoding='utf-8')
    def stop():
        raise RuntimeError('budget exhausted')
    with pytest.raises(RuntimeError,match='budget exhausted'):
        render_report(output,result,decoded,budget=stop)
