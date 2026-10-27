"""Synthetic contract checks only; no learned tracker or measurement validation."""

import numpy as np
import pytest
from unittest import mock

from integrations.demo_pipeline import learned_point_contract as contract

from integrations.demo_pipeline.learned_point_contract import (
    classify_point, classify_points, cv_to_model_queries, model_to_cv,
)


def test_half_pixel_and_axis_roundtrip_non_square():
    points = np.array([[410, 830], [0, 0], [575, 1023]], dtype=float)
    query = cv_to_model_queries(points, 576, 1024)
    np.testing.assert_allclose(query[0], [0, 207.625, 182.44444444444443])
    np.testing.assert_allclose(model_to_cv(query[:, 1:], 576, 1024), points, atol=1e-12)
    assert not np.allclose(model_to_cv(query[:, 1:][:, ::-1], 576, 1024), points)
    assert not np.allclose(query[0, 1:], [830 * .25, 410 * 256 / 576])
    assert cv_to_model_queries(np.empty((0, 2)), 576, 1024).shape == (0, 3)


def test_visibility_and_missing_mask_are_separate_gates():
    assert classify_point([1, 1], 1, .5, None, 4, 3) == "OCCLUDED_OR_UNCERTAIN"
    assert classify_point([1, 1], .8, .8, None, 4, 3) == "MASK_UNAVAILABLE"
    assert classify_point([np.nan, 1], 1, 1, None, 4, 3) == "OUTSIDE_IMAGE"
    assert classify_point([4, 1], 1, 1, None, 4, 3) == "OUTSIDE_IMAGE"


def test_mask_lookup_rounds_half_up_without_clamping():
    mask = np.zeros((3, 4), np.uint8)
    mask[1, 2] = 255
    assert classify_point([1.5, 1], 1, 1, mask, 4, 3) == "VISIBLE_CANDIDATE"
    assert classify_point([1.49, 1], 1, 1, mask, 4, 3) == "OUTSIDE_MASK"
    assert classify_point([3.9, 1], 1, 1, mask, 4, 3) == "OUTSIDE_MASK"
    assert classify_point([-.01, 1], 1, 1, mask, 4, 3) == "OUTSIDE_IMAGE"


@pytest.mark.parametrize("bad", [True, 0, -1, 3.5, np.nan])
def test_reject_invalid_dimensions(bad):
    with pytest.raises(ValueError):
        cv_to_model_queries([[0, 0]], bad, 3)
    with pytest.raises(ValueError):
        model_to_cv([[.5, .5]], 4, 3, bad)
    with pytest.raises(ValueError):
        classify_point([1, 1], 1, 1, None, 4, bad)


@pytest.mark.parametrize("bad", [True, -.1, 1.1, np.nan, np.inf, "0.8"])
def test_reject_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        classify_point([1, 1], bad, 1, None, 4, 3)
    with pytest.raises(ValueError):
        classify_point([1, 1], 1, bad, None, 4, 3)


@pytest.mark.parametrize("bad", [[[True, False]], [[np.nan, 0]], [1, 2], [["1", "2"]]])
def test_reject_invalid_coordinate_arrays(bad):
    with pytest.raises(ValueError):
        cv_to_model_queries(bad, 4, 3)
    with pytest.raises(ValueError):
        model_to_cv(bad, 4, 3)


@pytest.mark.parametrize("bad", [np.zeros((4, 3)), np.ones((3, 4)), np.zeros((3, 4), bool), np.full((3, 4), np.nan)])
def test_reject_invalid_mask_even_for_occluded_point(bad):
    with pytest.raises(ValueError):
        classify_point([1, 1], 0, 0, bad, 4, 3)


def test_batch_mixed_states_match_single_point_rules_and_validate_mask_once():
    points = np.array([[1.5, 1], [1.49, 1], [3.9, 1], [4, 1], [np.nan, 1], [1, 1]])
    probabilities = np.ones(6)
    certainties = np.array([1, 1, 1, 1, 1, .5])
    mask = np.zeros((3, 4), np.uint8)
    mask[1, 2] = 255
    with mock.patch.object(contract, "_validate_mask", wraps=contract._validate_mask) as validate:
        statuses = classify_points(points, probabilities, certainties, mask, 4, 3)
        assert validate.call_count == 1
    assert statuses == ["VISIBLE_CANDIDATE", "OUTSIDE_MASK", "OUTSIDE_MASK",
                        "OUTSIDE_IMAGE", "OUTSIDE_IMAGE", "OCCLUDED_OR_UNCERTAIN"]
    assert statuses == [classify_point(p, pv, c, mask, 4, 3)
                        for p, pv, c in zip(points, probabilities, certainties)]
    assert classify_points(points[:2], probabilities[:2], certainties[:2], None, 4, 3) == [
        "MASK_UNAVAILABLE", "MASK_UNAVAILABLE"]
    assert classify_points(np.empty((0, 2)), np.empty(0), np.empty(0), mask, 4, 3) == []


@pytest.mark.parametrize("bad", [np.ones((2, 1)), np.ones(1), np.array([True, False]),
                                  np.array([1, np.nan]), np.array([1, 1.1])])
def test_batch_rejects_invalid_probability_vectors(bad):
    with pytest.raises(ValueError):
        classify_points(np.zeros((2, 2)), bad, np.ones(2), None, 4, 3)
    with pytest.raises(ValueError):
        classify_points(np.zeros((2, 2)), np.ones(2), bad, None, 4, 3)
