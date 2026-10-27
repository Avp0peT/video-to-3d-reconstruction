"""Equivalence tests for the SAM 3 mask array-conversion layer (offline only).

Scope: prove that the array-level fast conversion in
``integrations/sam3/sam3_adapter/backends.py`` is *equivalent* to the original
nested-list conversion, and that it never materialises a Python float tree for
a whole plane.

Nothing here loads SAM 3, CUDA, or weights. Real tensors, when available, are
small CPU tensors only. This is interface/equivalence evidence, **not**
evidence that SAM 3 works.

Run only this file:
    python -m pytest tests/test_sam3_array_conversion.py -p no:cacheprovider
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "integrations" / "sam3"
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from sam3_adapter import backends  # noqa: E402

try:  # real torch is optional only in the declared lightweight environment
    import torch
except ImportError:  # pragma: no cover - exercised by the light environment
    if os.environ.get("SAM3_ARRAY_REQUIRE_TORCH") == "1":
        raise
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch not installed in this environment")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
# Two independent, hand-written planes. The expected bytes are written out
# literally so the expectation does not come from the code under test.
MASK_A = [[True, False, False],
          [False, True, True]]
MASK_A_BYTES = bytes([255, 0, 0, 0, 255, 255])

MASK_B = [[False, True, False],
          [True, False, True]]
MASK_B_BYTES = bytes([0, 255, 0, 255, 0, 255])

WIDTH = 3
HEIGHT = 2


def _shape_of(data):
    shape = []
    node = data
    while isinstance(node, list):
        shape.append(len(node))
        if not node:
            break
        node = node[0]
    return tuple(shape)


class ListTensor:
    """List-backed stand-in with the same surface as the interface-test stub.

    It deliberately exposes no ``numpy()`` and no ``__array__``, so it must
    keep taking the original element-by-element path.
    """

    def __init__(self, data, shape=None):
        self._data = data
        self._shape = tuple(shape) if shape is not None else _shape_of(data)

    @property
    def shape(self):
        return self._shape

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def tolist(self):
        return self._data


class SpyArray:
    """A real array handle that records whether ``tolist()`` was used.

    ``tolist()`` raises, so an armed spy is a behavioural discriminator: if the
    converter ever built a Python tree, it would blow up instead of passing.
    """

    def __init__(self, array):
        self._array = np.asarray(array)
        self.tolist_calls = 0
        self.numpy_calls = 0

    @property
    def shape(self):
        return self._array.shape

    def numpy(self):
        self.numpy_calls += 1
        return self._array

    def tolist(self):
        self.tolist_calls += 1
        raise AssertionError("tolist() must not be called on the array path")


def _sha(flat: bytes) -> str:
    return hashlib.sha256(flat).hexdigest()


def _fingerprint(candidates):
    return [(c.candidate_id, c.flat, c.width, c.height, c.score, c.score_semantics)
            for c in candidates]


# =========================================================================== #
# 1. the list-backed compatibility path is unchanged
# =========================================================================== #
def test_list_backed_path_still_produces_the_literal_expected_bytes():
    masks = ListTensor([[MASK_A]])
    scores = ListTensor([0.75])
    candidates = backends.candidates_from_processor_output(masks, scores, WIDTH, HEIGHT)

    assert [c.candidate_id for c in candidates] == ["c00"]
    assert candidates[0].flat == MASK_A_BYTES
    assert candidates[0].flat == bytes(candidates[0].flat)
    assert (candidates[0].width, candidates[0].height) == (WIDTH, HEIGHT)
    assert candidates[0].score == 0.75
    assert candidates[0].score_semantics == backends.SAM3_SCORE_SEMANTICS


def test_array_path_is_not_used_for_a_list_backed_object():
    assert backends._array_like(ListTensor([[MASK_A]])) is None


# =========================================================================== #
# 2. the array path matches the literal expectation and the old path
# =========================================================================== #
@pytest.mark.parametrize(
    ("dtype", "plane_a"),
    [
        (np.bool_, MASK_A),
        (np.uint8, [[1, 0, 0], [0, 1, 1]]),
        (np.int64, [[1, 0, 0], [0, 1, 1]]),
        (np.float32, [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
        (np.float64, [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
    ],
)
def test_array_path_matches_literal_expected_for_every_dtype(dtype, plane_a):
    masks = np.array([[plane_a]], dtype=dtype)
    candidates = backends.candidates_from_processor_output(masks, np.array([0.5]), WIDTH, HEIGHT)
    assert candidates[0].flat == MASK_A_BYTES


def test_array_path_and_list_path_agree_on_two_candidates():
    array_masks = np.array([[MASK_A], [MASK_B]], dtype=bool)
    list_masks = ListTensor([[MASK_A], [MASK_B]])
    scores_array = np.array([0.9, 0.4])
    scores_list = ListTensor([0.9, 0.4])

    fast = backends.candidates_from_processor_output(array_masks, scores_array, WIDTH, HEIGHT)
    slow = backends.candidates_from_processor_output(list_masks, scores_list, WIDTH, HEIGHT)

    assert _fingerprint(fast) == _fingerprint(slow)
    assert [c.candidate_id for c in fast] == ["c00", "c01"]
    assert [c.flat for c in fast] == [MASK_A_BYTES, MASK_B_BYTES]
    assert [c.score for c in fast] == [0.9, 0.4]
    assert _sha(fast[0].flat) == _sha(slow[0].flat)
    assert _sha(fast[1].flat) == _sha(slow[1].flat)


def test_array_path_does_not_mutate_its_input():
    array_masks = np.array([[MASK_A]], dtype=np.float32)
    before = array_masks.copy()
    backends.candidates_from_processor_output(array_masks, np.array([0.5]), WIDTH, HEIGHT)
    assert np.array_equal(array_masks, before)


# =========================================================================== #
# 3. the array path never builds a Python float tree
# =========================================================================== #
def test_the_spy_is_armed_and_really_raises():
    """Negative control: a passing test must mean something."""
    spy = SpyArray(np.zeros((1, 1, HEIGHT, WIDTH), dtype=bool))
    with pytest.raises(AssertionError):
        spy.tolist()
    assert spy.tolist_calls == 1


def test_mask_plane_is_never_converted_with_tolist():
    spy = SpyArray(np.array([[MASK_A]], dtype=bool))
    candidates = backends.candidates_from_processor_output(spy, ListTensor([0.5]), WIDTH, HEIGHT)
    assert spy.tolist_calls == 0
    assert spy.numpy_calls >= 1
    assert candidates[0].flat == MASK_A_BYTES


def test_logits_are_never_converted_with_tolist():
    spy = SpyArray(np.array([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=np.float32))
    masks = np.array([[[[True, False], [False, True]]]], dtype=bool)
    candidates = backends.candidates_from_processor_output(
        masks, ListTensor([0.5]), 2, 2, mask_logits=spy)
    assert spy.tolist_calls == 0
    assert spy.numpy_calls >= 1
    assert candidates[0].flat == bytes([255, 0, 0, 255])


def test_array_like_dispatch_selects_the_array_path_only_for_real_handles():
    assert backends._array_like(ListTensor([[[[True]]]])) is None
    assert isinstance(backends._array_like(np.zeros((1, 1, 1, 1), dtype=bool)), np.ndarray)


# =========================================================================== #
# 4. contract violations keep the same exception type on both paths
# =========================================================================== #
@pytest.mark.parametrize(
    ("masks", "scores", "needle"),
    [
        (np.zeros((1, 1, 2), dtype=bool), np.array([0.5]), "rank 4"),
        (np.zeros((1, 2, HEIGHT, WIDTH), dtype=bool), np.array([0.5]), "channel dimension"),
        (np.zeros((1, 1, WIDTH, HEIGHT), dtype=bool), np.array([0.5]), "spatial size"),
        (np.array([[[[1, 2, 0], [0, 1, 1]]]], dtype=np.int64), np.array([0.5]), "non-binary"),
        (np.array([[[[0.0, np.nan, 1.0], [0, 1, 1]]]], dtype=np.float32), np.array([0.5]), "non-finite"),
        (np.array([[[[0.0, np.inf, 1.0], [0, 1, 1]]]], dtype=np.float32), np.array([0.5]), "non-finite"),
        (np.array([[MASK_A]], dtype=bool), np.array([0.5, 0.6]), "scores length"),
    ],
)
def test_array_path_contract_violations_are_explicit(masks, scores, needle):
    with pytest.raises(backends.TensorContractError) as excinfo:
        backends.candidates_from_processor_output(masks, scores, WIDTH, HEIGHT)
    assert needle in str(excinfo.value)


def test_non_numeric_array_plane_is_rejected():
    plane = [["ab", "cd", "ef"], ["gh", "ij", "kl"]]
    masks = np.array([[plane]], dtype="<U2")
    assert masks.shape == (1, 1, HEIGHT, WIDTH)
    with pytest.raises(backends.TensorContractError) as excinfo:
        backends.candidates_from_processor_output(masks, np.array([0.5]), WIDTH, HEIGHT)
    assert "not numeric" in str(excinfo.value)


def test_non_finite_logits_are_rejected_on_the_array_path():
    masks = np.array([[MASK_A]], dtype=bool)
    logits = np.array([[[[0.0, 1.0], [1.0, np.inf]]]], dtype=np.float32)
    with pytest.raises(backends.TensorContractError) as excinfo:
        backends.candidates_from_processor_output(masks, np.array([0.5]), WIDTH, HEIGHT,
                                                  mask_logits=logits)
    assert "non-finite" in str(excinfo.value)


def test_bad_score_keeps_the_score_contract_error_type():
    masks = np.array([[MASK_A]], dtype=bool)
    with pytest.raises(backends.ScoreContractError):
        backends.candidates_from_processor_output(masks, np.array([np.nan]), WIDTH, HEIGHT)
    with pytest.raises(backends.ScoreContractError):
        backends.candidates_from_processor_output(masks, np.array([np.inf]), WIDTH, HEIGHT)


def test_empty_candidate_set_returns_empty_list_on_the_array_path():
    masks = np.zeros((0, 1, HEIGHT, WIDTH), dtype=bool)
    assert backends.candidates_from_processor_output(masks, np.array([]), WIDTH, HEIGHT) == []


# =========================================================================== #
# 5. non-contiguous inputs
# =========================================================================== #
def test_non_contiguous_numpy_input_is_supported():
    staging = np.zeros((1, 1, HEIGHT, WIDTH * 2), dtype=bool)
    staging[..., ::2] = np.array(MASK_A, dtype=bool)
    view = staging[..., ::2]
    assert not view.flags["C_CONTIGUOUS"]

    candidates = backends.candidates_from_processor_output(view, np.array([0.5]), WIDTH, HEIGHT)
    assert candidates[0].flat == MASK_A_BYTES


# =========================================================================== #
# 6. real CPU torch tensors (small, never CUDA, never a model)
# =========================================================================== #
@requires_torch
def test_torch_dispatch_selects_the_array_path():
    tensor = torch.zeros((1, 1, 1, 1), dtype=torch.bool)
    assert isinstance(backends._array_like(tensor), np.ndarray)


@requires_torch
def test_real_torch_tensor_never_reaches_tolist_on_the_array_path():
    """Behavioural proof on a *real* tensor, not only on a mock.

    A real ``torch.Tensor`` subclass is used whose ``tolist`` raises. The test
    first proves the guard is armed, then proves it survives the converter's
    own detach/cpu/to normalisation, and only then runs the conversion. If the
    converter built a Python tree the conversion would raise instead of
    returning the expected bytes.
    """

    class NoToListTensor(torch.Tensor):
        def tolist(self, *args, **kwargs):  # noqa: D102
            raise AssertionError("tolist() must not be called on the array path")

    guarded = torch.tensor([[MASK_A]], dtype=torch.bool).as_subclass(NoToListTensor)

    # negative control: the guard really is armed
    with pytest.raises(AssertionError):
        guarded.tolist()

    probe = guarded.detach().cpu().to("cpu")
    if not isinstance(probe, NoToListTensor):
        pytest.skip("this torch build does not preserve the tensor subclass through detach/cpu/to")

    assert isinstance(backends._array_like(guarded), np.ndarray)
    candidates = backends.candidates_from_processor_output(guarded, torch.tensor([0.5]), WIDTH, HEIGHT)
    assert candidates[0].flat == MASK_A_BYTES


@requires_torch
@pytest.mark.parametrize("dtype", ["bool", "uint8", "int64", "float32", "float64"])
def test_torch_dtypes_match_the_literal_expected_bytes(dtype):
    torch_dtype = getattr(torch, dtype)
    masks = torch.tensor([[MASK_A]], dtype=torch_dtype)
    candidates = backends.candidates_from_processor_output(masks, torch.tensor([0.5]), WIDTH, HEIGHT)
    assert candidates[0].flat == MASK_A_BYTES


@requires_torch
def test_torch_bfloat16_uses_a_lossless_widening_route():
    logits = torch.tensor([[[[0.5, -0.5], [1.5, 0.0]]]], dtype=torch.bfloat16)
    masks = torch.tensor([[MASK_A]], dtype=torch.bool)
    array = backends._array_like(logits)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.dtype("float32"), "bfloat16 must widen to float32, not to a lossy type"
    candidates = backends.candidates_from_processor_output(
        masks, torch.tensor([0.5]), WIDTH, HEIGHT, mask_logits=logits)
    assert candidates[0].flat == MASK_A_BYTES

    bf16_masks = torch.tensor([[MASK_A]], dtype=torch.bfloat16)
    widened = backends.candidates_from_processor_output(bf16_masks, torch.tensor([0.5]), WIDTH, HEIGHT)
    assert widened[0].flat == MASK_A_BYTES


@requires_torch
def test_torch_bfloat16_non_finite_logits_are_still_rejected():
    logits = torch.tensor([[[[0.0, float("inf")]]]], dtype=torch.bfloat16)
    masks = torch.tensor([[[[True, False]]]], dtype=torch.bool)
    with pytest.raises(backends.TensorContractError) as excinfo:
        backends.candidates_from_processor_output(masks, torch.tensor([0.5]), 2, 1,
                                                  mask_logits=logits)
    assert "non-finite" in str(excinfo.value)


@requires_torch
def test_torch_non_contiguous_input_is_supported():
    staging = torch.zeros((1, 1, HEIGHT, WIDTH * 2), dtype=torch.bool)
    staging[..., ::2] = torch.tensor(MASK_A, dtype=torch.bool)
    view = staging[..., ::2]
    assert not view.is_contiguous()

    candidates = backends.candidates_from_processor_output(view, torch.tensor([0.5]), WIDTH, HEIGHT)
    assert candidates[0].flat == MASK_A_BYTES


@requires_torch
def test_torch_input_is_not_mutated():
    masks = torch.tensor([[MASK_A]], dtype=torch.bool)
    before = masks.clone()
    backends.candidates_from_processor_output(masks, torch.tensor([0.5]), WIDTH, HEIGHT)
    assert torch.equal(masks, before)


@requires_torch
@pytest.mark.parametrize(
    ("masks", "scores", "needle"),
    [
        (lambda: torch.zeros((1, 1, 2), dtype=torch.bool), lambda: torch.tensor([0.5]), "rank 4"),
        (lambda: torch.zeros((1, 2, HEIGHT, WIDTH), dtype=torch.bool), lambda: torch.tensor([0.5]), "channel dimension"),
        (lambda: torch.zeros((1, 1, WIDTH, HEIGHT), dtype=torch.bool), lambda: torch.tensor([0.5]), "spatial size"),
        (lambda: torch.tensor([[[[1, 2, 0], [0, 1, 1]]]], dtype=torch.int64), lambda: torch.tensor([0.5]), "non-binary"),
        (lambda: torch.tensor([[[[0.0, float("nan"), 1.0], [0, 1, 1]]]], dtype=torch.float32), lambda: torch.tensor([0.5]), "non-finite"),
        (lambda: torch.tensor([[MASK_A]], dtype=torch.bool), lambda: torch.tensor([0.5, 0.6]), "scores length"),
    ],
)
def test_torch_contract_violations_are_explicit(masks, scores, needle):
    with pytest.raises(backends.TensorContractError) as excinfo:
        backends.candidates_from_processor_output(masks(), scores(), WIDTH, HEIGHT)
    assert needle in str(excinfo.value)


@requires_torch
def test_torch_empty_candidate_set_returns_empty_list():
    masks = torch.zeros((0, 1, HEIGHT, WIDTH), dtype=torch.bool)
    assert backends.candidates_from_processor_output(masks, torch.tensor([]), WIDTH, HEIGHT) == []


@requires_torch
def test_torch_two_candidates_agree_with_the_list_path():
    torch_masks = torch.tensor([[MASK_A], [MASK_B]], dtype=torch.bool)
    list_masks = ListTensor([[MASK_A], [MASK_B]])
    # float64 on purpose: a float32 tensor legitimately yields the float32
    # rounded score, which would differ from the float64 literal for reasons
    # unrelated to the mask conversion under test.
    torch_scores = torch.tensor([0.9, 0.4], dtype=torch.float64)
    fast = backends.candidates_from_processor_output(torch_masks, torch_scores, WIDTH, HEIGHT)
    slow = backends.candidates_from_processor_output(list_masks, ListTensor([0.9, 0.4]), WIDTH, HEIGHT)
    assert _fingerprint(fast) == _fingerprint(slow)
    assert [c.flat for c in fast] == [MASK_A_BYTES, MASK_B_BYTES]


@requires_torch
def test_float32_score_is_passed_through_unchanged():
    """The array path must not alter score handling.

    Both paths take the score from ``tolist()`` of the same tensor, so a
    float32 score keeps its float32 value; this is the pre-existing behaviour,
    not something the array path introduced.
    """
    masks = torch.tensor([[MASK_A]], dtype=torch.bool)
    float32_scores = torch.tensor([0.9])
    expected = float32_scores.tolist()[0]
    candidates = backends.candidates_from_processor_output(masks, float32_scores, WIDTH, HEIGHT)
    assert candidates[0].score == expected
    assert candidates[0].score_semantics == backends.SAM3_SCORE_SEMANTICS


# =========================================================================== #
# 7. one synthetic full-size case (no memory stress, one only)
# =========================================================================== #
def test_single_full_size_case_is_equivalent():
    rng = np.random.default_rng(20261008)
    big = rng.integers(0, 2, size=(2, 1, 683, 1024), dtype=np.uint8).astype(bool)

    fast = backends.candidates_from_processor_output(big, np.array([0.8, 0.2]), 1024, 683)
    slow = backends.candidates_from_processor_output(
        ListTensor(big.tolist()), ListTensor([0.8, 0.2]), 1024, 683)

    assert _fingerprint(fast) == _fingerprint(slow)
    for candidate in fast:
        assert len(candidate.flat) == 1024 * 683
        assert set(candidate.flat) <= {0, 255}
    assert [c.candidate_id for c in fast] == ["c00", "c01"]
    assert _sha(fast[0].flat) == _sha(slow[0].flat)
    assert _sha(fast[1].flat) == _sha(slow[1].flat)


# =========================================================================== #
# 8. nothing here may touch the GPU
# =========================================================================== #
@requires_torch
def test_no_cuda_context_was_created():
    """Offline evidence must not include a CUDA context.

    ``is_initialized()`` is False unless something actually drove a CUDA API.
    When the caller declares that CUDA was hidden (as the delegate run does),
    the stronger ``is_available()`` check is also enforced.
    """
    assert torch.cuda.is_initialized() is False, "no CUDA context may be created by this suite"
    if os.environ.get("SAM3_ARRAY_EXPECT_CUDA_HIDDEN") == "1":
        assert torch.cuda.is_available() is False, "CUDA must stay hidden for this offline task"


# Main-line closure counterexamples. These exercise the prior failure modes;
# neither dtype acceptance nor error precedence is inferred from passing masks.
@pytest.mark.parametrize("empty", [False, True])
def test_boolean_logits_match_the_list_validator(empty):
    logits = np.empty((0,), dtype=bool) if empty else np.array([True, False])
    masks = np.array([[MASK_A]], dtype=bool)
    calls = (
        lambda: backends.candidates_from_processor_output(
            masks, np.array([0.5]), WIDTH, HEIGHT, mask_logits=logits),
        lambda: backends.candidates_from_processor_output(
            ListTensor([[MASK_A]]), ListTensor([0.5]), WIDTH, HEIGHT,
            mask_logits=ListTensor(logits.tolist())),
    )
    if empty:
        assert _fingerprint(calls[0]()) == _fingerprint(calls[1]())
    else:
        for call in calls:
            with pytest.raises(backends.TensorContractError, match="non-numeric value bool"):
                call()


def test_object_numbers_and_none_keep_the_old_fallback_semantics():
    masks = np.array([[[[True, 0, 0.0], [False, 1, 1.0]]]], dtype=object)
    logits = np.array([0.25, None, -2], dtype=object)
    scores = np.array([None], dtype=object)
    before = masks.copy(), logits.copy(), scores.copy()
    assert backends._array_like(masks) is None
    assert backends._array_like(logits) is None
    fast = backends.candidates_from_processor_output(
        masks, scores, WIDTH, HEIGHT, mask_logits=logits)
    slow = backends.candidates_from_processor_output(
        ListTensor(masks.tolist()), ListTensor(scores.tolist()), WIDTH, HEIGHT,
        mask_logits=ListTensor(logits.tolist()))
    assert _fingerprint(fast) == _fingerprint(slow)
    assert fast[0].flat == MASK_A_BYTES and fast[0].score is None
    for actual, original in zip((masks, logits, scores), before):
        assert np.array_equal(actual, original)


@pytest.mark.parametrize("bad", [None, "1", 2, 0.5, float("nan"), float("inf"), 1j])
def test_invalid_object_mask_values_match_the_old_error_type(bad):
    masks = np.array([[[[bad]]]], dtype=object)
    for source in (masks, ListTensor(masks.tolist())):
        with pytest.raises(backends.TensorContractError):
            backends.candidates_from_processor_output(source, ListTensor([0.5]), 1, 1)


@pytest.mark.parametrize("bad", [True, "1", float("nan"), float("inf"), 1j])
def test_invalid_object_logits_match_the_old_error_type(bad):
    logits = np.array([bad], dtype=object)
    for source in (logits, ListTensor(logits.tolist())):
        with pytest.raises(backends.TensorContractError):
            backends.candidates_from_processor_output(
                ListTensor([[[[True]]]]), ListTensor([0.5]), 1, 1, mask_logits=source)


def test_unrelated_numpy_typeerror_cannot_trigger_float_coercion():
    class RefusesNumpy(ListTensor):
        dtype = "torch.float64"
        float_calls = 0

        def numpy(self):
            raise TypeError("not a BF16 conversion failure")

        def float(self):
            self.float_calls += 1
            return SpyArray(np.ones((1, 1, 1, 1), dtype=np.float32))

    masks = RefusesNumpy([[[[0.5]]]])
    with pytest.raises(backends.TensorContractError, match="non-binary"):
        backends.candidates_from_processor_output(masks, ListTensor([0.5]), 1, 1)
    assert masks.float_calls == 0


@pytest.mark.parametrize("dtype", [np.complex64, "U1"])
def test_unsupported_numpy_dtypes_keep_the_list_error(dtype):
    masks = np.ones((1, 1, 1, 1), dtype=dtype)
    for source in (masks, ListTensor(masks.tolist())):
        with pytest.raises(backends.TensorContractError):
            backends.candidates_from_processor_output(source, ListTensor([0.5]), 1, 1)


def test_logits_validation_precedes_missing_scores_on_both_paths():
    for logits in (np.array([True]), ListTensor([True])):
        with pytest.raises(backends.TensorContractError, match="non-numeric value bool"):
            backends.candidates_from_processor_output(
                np.ones((1, 1, 1, 1), dtype=bool), None, 1, 1, mask_logits=logits)


def test_module_import_does_not_request_heavy_dependencies(monkeypatch):
    import builtins
    import importlib.util

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "torch", "sam3"}:
            raise AssertionError(f"eager heavy import: {name}")
        return original_import(name, *args, **kwargs)

    name = "sam3_adapter._array_closure_import_check"
    spec = importlib.util.spec_from_file_location(name, backends.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec.loader.exec_module(module)
    assert module._array_like(ListTensor([[[[True]]]])) is None


@requires_torch
@pytest.mark.parametrize("empty", [False, True])
def test_real_torch_boolean_logits_match_the_list_validator(empty):
    logits = torch.empty((0,), dtype=torch.bool) if empty else torch.tensor([True])
    masks = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    if empty:
        assert backends.candidates_from_processor_output(
            masks, torch.tensor([0.5]), 1, 1, mask_logits=logits)[0].flat == b"\xff"
    else:
        for source in (logits, ListTensor([True])):
            with pytest.raises(backends.TensorContractError, match="non-numeric value bool"):
                backends.candidates_from_processor_output(
                    masks, torch.tensor([0.5]), 1, 1, mask_logits=source)


@requires_torch
@pytest.mark.parametrize("bad", [0.5, float("nan"), float("inf")])
def test_bfloat16_invalid_masks_are_rejected_before_integer_conversion(bad):
    masks = torch.tensor([[[[bad]]]], dtype=torch.bfloat16)
    before = masks.float().numpy().copy()
    with pytest.raises(backends.TensorContractError):
        backends.candidates_from_processor_output(masks, torch.tensor([0.5]), 1, 1)
    assert np.array_equal(masks.float().numpy(), before, equal_nan=True)


@requires_torch
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_torch_logits_do_not_call_tolist_and_are_not_mutated(dtype):
    class NoToListTensor(torch.Tensor):
        def tolist(self, *args, **kwargs):
            raise AssertionError("logits tolist is forbidden")

    logits = torch.tensor([[[[0.25, -0.5]]]], dtype=getattr(torch, dtype)).as_subclass(NoToListTensor)
    masks = torch.tensor([[[[True, False]]]])
    scores = torch.tensor([0.75])
    before_logits, before_masks, before_scores = logits.clone(), masks.clone(), scores.clone()
    with pytest.raises(AssertionError, match="logits tolist"):
        logits.tolist()
    result = backends.candidates_from_processor_output(masks, scores, 2, 1, mask_logits=logits)
    assert result[0].flat == b"\xff\x00"
    assert torch.equal(logits, before_logits)
    assert torch.equal(masks, before_masks)
    assert torch.equal(scores, before_scores)
