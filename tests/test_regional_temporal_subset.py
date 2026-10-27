import copy
import pytest
import numpy as np
from pathlib import Path
from integrations.mvs_surface.regional_temporal_subset import select_views, scene_intrinsic_from_native, copied_relative_path


def fixture():
    frames = []; rows = []
    for i, seconds in enumerate((-3, -2, -1, 0, 1, 2, 3)):
        frames.append(dict(decode_index=i, export_image_id=i+10, exported_name=str(i)+'.png', canonical_mask_path=str(i)+'.mask', canonical_mask_sha256=str(i)))
        rows.append(dict(decode_index=i, pts=seconds+3, time_base=[1, 1], seconds_fraction=[seconds+3, 1], canonical_mask_path=str(i)+'.mask', canonical_mask_sha256=str(i)))
    analysis = dict(status='TEMPORAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ADOPTED', source_unchanged=True, target_decode_index=3, rows=[dict(window_seconds=dict(numerator=2, denominator=1), target_view_id=3, near_view_ids=[1, 2, 4, 5])])
    return dict(frames=frames), dict(rows=rows), analysis


def test_fixed_closed_middle_window_and_export_identity():
    selected = select_views(*fixture(), 2)
    assert [x['decode_index'] for x in selected] == [1, 2, 3, 4, 5]
    assert [x['export_image_id'] for x in selected] == [11, 12, 13, 14, 15]
    assert sum(x['is_target'] for x in selected) == 1


@pytest.mark.parametrize('bad', ['bestwindow', 'duplicate', 'pts', 'mask', 'analysis', 'nativeid'])
def test_selection_fail_closed(bad):
    f, t, a = copy.deepcopy(fixture()); w = 2
    if bad == 'bestwindow': w = 4
    if bad == 'duplicate': t['rows'].append(t['rows'][0])
    if bad == 'pts': t['rows'][0]['seconds_fraction'] = [1, 2]
    if bad == 'mask': t['rows'][0]['canonical_mask_sha256'] = 'wrong'
    if bad == 'analysis': a['rows'][0]['near_view_ids'] = [1, 2]
    if bad == 'nativeid': f['frames'][0]['export_image_id'] = 11
    with pytest.raises(ValueError): select_views(f, t, a, w)


def test_interface_principal_point_convention_only_comparison():
    params = np.array([750., 751., 282., 512.]); before = params.copy()
    k = scene_intrinsic_from_native(params)
    assert np.array_equal(k, [[750, 0, 281.5], [0, 751, 511.5], [0, 0, 1]])
    assert np.array_equal(params, before)
    with pytest.raises(ValueError): scene_intrinsic_from_native([750, 751, np.nan, 512])


def test_copy_paths_match_runner_contract():
    assert copied_relative_path('image_00000080.png', 'image') == Path('undistorted/images/image_00000080.png')
    assert copied_relative_path('image_00000080.png', 'mask') == Path('masks/image_00000080.mask.png')
    assert copied_relative_path('image_00000080.png', 'validity') == Path('validity/image_00000080.validity.png')
    with pytest.raises(ValueError): copied_relative_path('../image.png', 'mask')
    with pytest.raises(ValueError): copied_relative_path('image.png', 'other')
