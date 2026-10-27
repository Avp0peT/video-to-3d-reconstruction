import hashlib
import functools
import json
from pathlib import Path
import sys

import pytest

from integrations.demo_pipeline.sam_score_diagnostic import DiagnosticError, ScoreDiagnostic


class Tensor:
    dtype = "fake_float"
    device = "cpu"

    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def __gt__(self, threshold):
        return Tensor([v > threshold for v in self.values])

    def __getitem__(self, keep):
        return Tensor([v for v, k in zip(self.values, keep.values) if k])


class Model:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def forward_grounding(self):
        self.calls += 1
        return {"pred_logits": Tensor(self.values), "presence_logit_dec": Tensor([1.0]),
                "pred_masks": object()}


def inference_wrapper(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        return function(*args, **kwargs)
    return wrapped


class Processor:
    confidence_threshold = 0.5

    def __init__(self, values):
        self.model = Model(values)

    @inference_wrapper
    def _forward_grounding(self, state):
        outputs = self.model.forward_grounding()
        presence_score = Tensor([1.0])
        out_probs = outputs["pred_logits"]
        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        state["scores"] = out_probs
        return state


def recorder(sink, **kwargs):
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return ScoreDiagnostic(sink, expected_source_sha256=digest, **kwargs)


@pytest.mark.parametrize("values,expected", [([], []), ([0.5], []),
                          ([0.49, 0.5, 0.50001, 0.9], [0.50001, 0.9])])
def test_actual_filter_identity(values, expected):
    lines = []
    diagnostic = recorder(lines.append)
    processor = Processor(values)
    state = {}
    with diagnostic.observe(processor, 15):
        result = processor._forward_grounding(state)
    assert result is state
    assert processor.model.calls == 1
    row = json.loads(lines[0])
    assert row["retained_scores"]["values"] == expected
    assert row["combined_scores"]["values"] == values
    assert row["retained_scores_match"] is True
    assert sys.gettrace() is None


def test_logging_error_restores():
    def sink(line):
        raise OSError("disk full")
    diagnostic = recorder(sink)
    with pytest.raises(OSError, match="disk full"):
        with diagnostic.observe(Processor([0.8]), 1):
            # Use the same processor instance, as production does.
            processor = None
            raise OSError("disk full")
    assert sys.gettrace() is None
    processor = Processor([0.8])
    with pytest.raises(OSError, match="disk full"):
        with diagnostic.observe(processor, 2):
            processor._forward_grounding({})
    assert sys.gettrace() is None


def test_limits_and_nonfinite():
    for values, kwargs, error in [([0.9, 0.8], {"max_queries": 1}, "query count"),
                                 ([float("nan")], {}, "non-finite"),
                                 ([float("inf")], {}, "non-finite"),
                                 ([0.9], {"max_record_bytes": 1}, "byte budget")]:
        diagnostic = recorder(lambda line: None, **kwargs)
        processor = Processor(values)
        with pytest.raises(DiagnosticError, match=error):
            with diagnostic.observe(processor, 1):
                processor._forward_grounding({})
        assert sys.gettrace() is None
    diagnostic = recorder(lambda line: None, max_frames=1)
    processor = Processor([])
    with diagnostic.observe(processor, 0):
        processor._forward_grounding({})
    with pytest.raises(DiagnosticError, match="frame count"):
        with diagnostic.observe(processor, 1):
            pass


def test_trace_and_source_refusal():
    diagnostic = recorder(lambda line: None)
    existing = lambda *args: None
    sys.settrace(existing)
    try:
        with pytest.raises(DiagnosticError, match="existing trace"):
            with diagnostic.observe(Processor([]), 0):
                pass
        assert sys.gettrace() is existing
    finally:
        sys.settrace(None)
    diagnostic.expected_source_sha256 = "0" * 64
    with pytest.raises(DiagnosticError, match="binding differs"):
        with diagnostic.observe(Processor([]), 0):
            pass
