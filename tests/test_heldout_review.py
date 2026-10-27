"""Independent regressions for refusal accounting and minimal heldout coverage."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline import heldout_multiview as review


def diagnostic(value):
    return dict(status='EVALUATED', residual_norms=[value],
                residual_vectors=[[value, 0.]], rms=value, max=value)


def test_refused_pair_is_unknown_even_when_heldout_within_threshold():
    method = dict(status='WORLD_CONVERTED_NOT_ACCEPTED', xyz=[0., 0., 5.],
                  training=diagnostic(.5), heldout=diagnostic(.5),
                  pair=dict(status='PROJECTION_REFUSED', max=None))
    sample = dict(case_kind='negative_control', point_id=1, source_group='g0',
                  split=dict(status='SPLIT_READY'), status='EVALUATED_WITH_STAGE_STATUSES',
                  methods={name: deepcopy(method) for name in ('A', 'C', 'J')},
                  fit_details=dict(converged=False, reason='iteration bound reached'))
    summary = review.summarize([sample])['negative_control']
    assert summary['joint_converged'] == 0
    assert summary['joint_fit_reasons']['iteration bound reached'] == 1
    for threshold in ('1.0', '2.0', '4.0'):
        counters = summary['methods']['C']['threshold_diagnostics'][threshold]
        assert counters['eligible_slots'] == counters['heldout_max_within'] == 1
        assert counters['pair_projection_refused'] == 1
        assert counters['pair_and_heldout_evaluable'] == 0
        assert counters['pair_and_heldout_max_within'] == 0
        assert counters['pair_or_heldout_max_exceeds'] == 0
        assert (counters['pair_and_heldout_max_within']
                + counters['pair_or_heldout_max_exceeds']
                + counters['pair_projection_refused']) == counters['eligible_slots']


def test_exactly_two_extra_views_evaluate_one_heldout_view():
    xyz = np.array([.1, -.2, 5.])
    observations = []
    for image_id, center in enumerate((0., .3, .6, .9), 1):
        t = np.array([-center, 0., 0.])
        camera_xyz = xyz + t
        pixels = (800. * camera_xyz[:2] / camera_xyz[2]
                  + np.array([320., 240.]))
        observations.append(dict(image_id=image_id, raw_pixels=pixels.tolist(),
            params=[800., 320., 240., 0.], R=np.eye(3), t=t,
            max_ideal_radius=2.))
    split = review.split_views([1, 2, 3, 4], [1, 2])
    assert split['train_extra'] == [3]
    assert split['heldout'] == [4]
    sample = dict(split=split,
        observations=dict(pair=observations[:2], train_extra=observations[2:3],
                          heldout=observations[3:]),
        candidates={m: dict(status='WORLD_CONVERTED_NOT_ACCEPTED', xyz=xyz.copy())
                    for m in ('A', 'C')})
    result = review.evaluate_sample(sample)
    assert result['fit_details']['train_image_ids'] == [1, 2, 3]
    for name in ('A', 'C', 'J'):
        heldout = result['methods'][name]['heldout']
        assert heldout['status'] == 'EVALUATED'
        assert len(heldout['residual_norms']) == 1
        assert heldout['max'] < 1e-7
