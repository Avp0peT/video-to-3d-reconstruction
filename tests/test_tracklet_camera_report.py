import copy

import pytest
from PIL import Image

from integrations.demo_pipeline.tracklet_camera_report import render_camera_review


def _fixture(tmp_path):
    frame = tmp_path / 'actual.png'
    Image.new('RGB', (120, 90), (17, 29, 41)).save(frame)
    decoded = {3: dict(path=str(frame), width=120, height=90, time_seconds=.125)}
    points = [dict(point_id='p', tracklet_id='t', xyz=[1, 2, 3],
                   raw_xy=[20.5, 30.5], tier='STRICT')]
    errors = dict(rows=[dict(point_id='p', predicted_raw_xy=[45.5, 30.5],
                             error_px=25, denominator=1)], median_px=25, max_px=25)
    assessment = dict(status='PROJECTIVE_FIT_DIAGNOSTIC', count=1,
                      train_ids=[], holdout_ids=['p'], all_fit=None,
                      heldout_fit=dict(holdout_reprojection=errors),
                      negative_control=dict(status='BLOCKED_DEGENERATE'), reasons=[])
    record = dict(model_key='m', query_decode_index=3, cohort='STRICT', points=points,
                  assessment=assessment, original_mask_status='UNKNOWN', geometry_accepted=False)
    return decoded, dict(records=[record])


def test_empty_is_valid_without_placeholder(tmp_path):
    output = tmp_path / 'review'
    output.mkdir()
    files = render_camera_review(output, dict(records=[]), dict(pairs=[]), {})
    assert files == [output / 'index.html']
    page = files[0].read_text(encoding='utf-8')
    assert '没有投影记录' in page and 'geometry_accepted=false' in page
    assert 'EPIPOLAR.json' in page and 'PROJECTION.json' in page
    assert not list(output.glob('*.png'))


def test_actual_frame_overlay_and_input_preserved(tmp_path):
    decoded, projection = _fixture(tmp_path)
    original = copy.deepcopy(projection)
    output = tmp_path / 'review'
    output.mkdir()
    files = render_camera_review(output, projection, dict(pairs=[]), decoded)
    assert {p.name for p in files} == {'strict.png', 'index.html'}
    with Image.open(output / 'strict.png') as image:
        colors = set(image.getdata())
        assert (17, 29, 41) in colors  # Actual pixels remain in the frame copy.
        assert (255, 159, 67) in colors
        assert (0, 229, 255) in colors
    page = (output / 'index.html').read_text(encoding='utf-8')
    assert 'UNKNOWN' in page and '0.125' in page and '25' in page
    assert '共享原始数据' in page and '未知畸变' in page
    assert projection == original


def test_no_holdout_does_not_draw_predictions(tmp_path):
    decoded, projection = _fixture(tmp_path)
    projection['records'][0]['assessment']['heldout_fit'] = None
    output = tmp_path / 'review'
    output.mkdir()
    render_camera_review(output, projection, {}, decoded)
    with Image.open(output / 'strict.png') as image:
        assert (0, 229, 255) not in set(image.getdata())


def test_html_escaping(tmp_path):
    decoded, projection = _fixture(tmp_path)
    attack = '</pre><script>bad()</script>'
    projection['records'][0]['model_key'] = attack
    output = tmp_path / 'review'
    output.mkdir()
    render_camera_review(output, projection, dict(pairs=[dict(status=attack, reasons=[attack])]), decoded)
    page = (output / 'index.html').read_text(encoding='utf-8')
    assert attack not in page
    assert '&lt;script&gt;' in page


def test_summaries_exclude_matrices_and_point_payloads(tmp_path):
    decoded, projection = _fixture(tmp_path)
    projection['records'][0]['assessment']['negative_control'] = dict(
        status='EVALUATED', P='MATRIX_MUST_NOT_APPEAR',
        holdout_reprojection=dict(median_px=31, max_px=42, rows=['ROWS_MUST_NOT_APPEAR']))
    epipolar = dict(pairs=[dict(source_decode_index=3, target_decode_index=4,
        counts=dict(points=16, train=12, holdout=4), status='DIAGNOSTIC', reasons=['reason'],
        F=dict(status='EVALUATED', model='F_MATRIX_MUST_NOT_APPEAR', holdout=dict(median=1.25, max=2.5)),
        H=dict(status='EVALUATED', holdout=dict(median=3.75, max=4.5)),
        negative_control=dict(F=dict(median=15), H=dict(median=25)))])
    output = tmp_path / 'review'
    output.mkdir()
    render_camera_review(output, projection, epipolar, decoded)
    page = (output / 'index.html').read_text(encoding='utf-8')
    assert 'MUST_NOT_APPEAR' not in page
    assert '1.25' in page and '3.75' in page and '31' in page and '42' in page
    assert '3 → 4' in page and 'reason' in page


def test_out_of_frame_predictions_are_omitted(tmp_path):
    decoded, projection = _fixture(tmp_path)
    projection['records'][0]['assessment']['heldout_fit']['holdout_reprojection']['rows'][0]['predicted_raw_xy'] = [10000, 10000]
    output = tmp_path / 'review'
    output.mkdir()
    render_camera_review(output, projection, {}, decoded)
    with Image.open(output / 'strict.png') as image:
        assert (0, 229, 255) not in set(image.getdata())
    assert '原图外的预测省略' in (output / 'index.html').read_text(encoding='utf-8')


def test_decoded_dimensions_must_match_actual_image(tmp_path):
    decoded, projection = _fixture(tmp_path)
    decoded[3]['width'] = 121
    output = tmp_path / 'review'
    output.mkdir()
    with pytest.raises(ValueError, match='DECODED_IMAGE_SIZE_MISMATCH'):
        render_camera_review(output, projection, {}, decoded)
    assert not list(output.iterdir())


def test_budget_exception_propagates_without_outputs(tmp_path):
    output = tmp_path / 'review'
    output.mkdir()
    def budget():
        raise RuntimeError('BUDGET_EXHAUSTED')
    with pytest.raises(RuntimeError, match='BUDGET_EXHAUSTED'):
        render_camera_review(output, {}, {}, {}, budget=budget)
    assert not list(output.iterdir())


def test_existing_artifacts_are_never_overwritten(tmp_path):
    output = tmp_path / 'review'
    output.mkdir()
    target = output / 'index.html'
    target.write_text('sealed', encoding='utf-8')
    with pytest.raises(FileExistsError):
        render_camera_review(output, {}, {}, {})
    assert target.read_text(encoding='utf-8') == 'sealed'


def test_image_limit_does_not_remove_table_records(tmp_path):
    decoded, projection = _fixture(tmp_path)
    projection['records'] = [dict(copy.deepcopy(projection['records'][0]), model_key=f'model_{i}')
                             for i in range(13)]
    output = tmp_path / 'review'
    output.mkdir()
    render_camera_review(output, projection, {}, decoded)
    with Image.open(output / 'strict.png') as image:
        assert image.size == (1212, 2280)
    assert 'model_12' in (output / 'index.html').read_text(encoding='utf-8')
