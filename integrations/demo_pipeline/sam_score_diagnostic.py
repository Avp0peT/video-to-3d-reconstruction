"""Bounded thread-local observation of actual SAM processor filter locals.

No torch/SAM imports or recomputation. CPU copies of small score tensors can
synchronize CUDA. Python tracing adds CPU overhead; this is not a timing run.
The caller retains the sealed worker, ownership, budget and resource gates.
"""
from contextlib import contextmanager
import inspect
import hashlib
from pathlib import Path
import json
import math
import marshal
import sys
import threading
from types import CodeType


class DiagnosticError(RuntimeError):
    pass


def _code_identity(code):
    """Compare compiled code including nested constants, ignoring only filename."""
    constants = tuple(_code_identity(v) if isinstance(v, CodeType) else v
                      for v in code.co_consts)
    return code.replace(co_filename="<bound-source>", co_consts=constants)


def _compiled_function(source, filename, target):
    pending = [compile(source, filename, "exec", dont_inherit=True)]
    matches = []
    while pending:
        code = pending.pop()
        if code.co_qualname == target.co_qualname and code.co_firstlineno == target.co_firstlineno:
            matches.append(code)
        pending.extend(v for v in code.co_consts if isinstance(v, CodeType))
    if len(matches) != 1 or marshal.dumps(_code_identity(matches[0])) != marshal.dumps(_code_identity(target)):
        raise DiagnosticError("processor executable code differs from bound source")


def _flatten(value):
    if isinstance(value, list):
        return [item for part in value for item in _flatten(part)]
    if type(value) not in (float, int, bool) or not math.isfinite(value):
        raise DiagnosticError("non-finite or nonnumeric score evidence")
    return [value]


class ScoreDiagnostic:
    """One recorder per worker; ``observe(processor, index)`` per segment call.

    sink receives a bounded UTF-8 JSONL string and must write synchronously.
    Any evidence failure raises, letting existing worker stop handling apply.
    max_queries is an admission limit, never truncation or silent omission.
    """
    def __init__(self, sink, *, expected_source_sha256, max_frames=32, max_queries=256,
                 max_record_bytes=65536, max_total_bytes=2097152):
        for value, ceiling in ((max_frames, 96), (max_queries, 1024),
                               (max_record_bytes, 131072), (max_total_bytes, 8388608)):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid diagnostic bound")
        self.sink = sink
        if (not isinstance(expected_source_sha256, str) or len(expected_source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_source_sha256)):
            raise ValueError("source SHA-256 required")
        self.expected_source_sha256 = expected_source_sha256
        self.max_frames = max_frames
        self.max_queries = max_queries
        self.max_record_bytes = max_record_bytes
        self.max_total_bytes = max_total_bytes
        self.frames = self.total_bytes = 0
        self.owner = threading.get_ident()

    def _tensor(self, tensor, bound):
        if tensor.numel() > bound:
            raise DiagnosticError("score tensor exceeds admitted query count")
        shape = list(tensor.shape)
        if len(shape) > 4:
            raise DiagnosticError("unexpected score rank")
        values = _flatten(tensor.detach().cpu().tolist())
        if len(values) != tensor.numel():
            raise DiagnosticError("score tensor size mismatch")
        return {"shape": shape, "dtype": str(tensor.dtype),
                "device": str(tensor.device), "values": values}

    @contextmanager
    def observe(self, processor, input_index):
        if threading.get_ident() != self.owner or sys.gettrace() is not None:
            raise DiagnosticError("wrong thread or existing trace hook")
        if type(input_index) is not int or input_index < 0:
            raise DiagnosticError("invalid input index")
        if self.frames >= self.max_frames:
            raise DiagnosticError("frame count exceeded")
        method = processor._forward_grounding
        function = inspect.unwrap(getattr(method, "__func__", method))
        source_path = Path(inspect.getsourcefile(function))
        source_bytes = source_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != self.expected_source_sha256:
            raise DiagnosticError("processor source binding differs")
        _compiled_function(source_bytes, str(source_path), function.__code__)
        lines, start = inspect.getsourcelines(function)
        points = [start + i for i, line in enumerate(lines)
                  if line.strip() == "out_probs = out_probs[keep]"]
        if len(points) != 1 or not any(line.strip() == "keep = out_probs > self.confidence_threshold"
                                        for line in lines):
            raise DiagnosticError("processor filter source contract differs")
        capture = None
        returned = False
        code = function.__code__
        threshold = processor.confidence_threshold
        if type(threshold) not in (float, int) or not math.isfinite(threshold):
            raise DiagnosticError("invalid processor threshold")

        def trace(frame, event, arg):
            nonlocal capture, returned
            if frame.f_code is not code:
                return None
            if frame.f_locals.get("self") is not processor:
                raise DiagnosticError("unexpected processor instance")
            if event == "line" and frame.f_lineno == points[0]:
                if capture is not None:
                    raise DiagnosticError("multiple grounding calls in one frame")
                local = frame.f_locals
                outputs = local["outputs"]
                capture = {"pred_logits": self._tensor(outputs["pred_logits"], self.max_queries),
                           "presence_logit_dec": self._tensor(outputs["presence_logit_dec"], self.max_queries),
                           "presence_score": self._tensor(local["presence_score"], self.max_queries),
                           "combined_scores": self._tensor(local["out_probs"], self.max_queries),
                           "keep": self._tensor(local["keep"], self.max_queries)}
                scores = capture["combined_scores"]["values"]
                keep_values = capture["keep"]["values"]
                if (processor.confidence_threshold != threshold or len(scores) != len(keep_values)
                        or any(type(k) is not bool or k != (s > threshold)
                               for s, k in zip(scores, keep_values))):
                    raise DiagnosticError("actual keep differs from strict threshold expression")
            elif event == "return" and capture is not None and isinstance(arg, dict):
                scores = self._tensor(arg["scores"], self.max_queries)
                combined = capture["combined_scores"]["values"]
                keep = capture["keep"]["values"]
                if len(combined) != len(keep) or any(type(k) is not bool for k in keep):
                    raise DiagnosticError("keep contract differs")
                expected = [s for s, k in zip(combined, keep) if k]
                if scores["values"] != expected:
                    raise DiagnosticError("retained scores differ from production filter")
                capture["retained_scores"] = scores
                returned = True
            return trace

        self.frames += 1
        sys.settrace(trace)
        try:
            yield
            if capture is None or not returned:
                raise DiagnosticError("missing completed production filter observation")
            if processor.confidence_threshold != threshold:
                raise DiagnosticError("processor threshold changed during observation")
            record = {"input_index": input_index, "threshold": threshold,
                      "processor_source_sha256": self.expected_source_sha256,
                      "retained_scores_match": True,
                      "score_semantics": "presence_calibrated_sigmoid_product_not_probability",
                      "capture": "actual_processor_locals_before_filter_no_recompute",
                      "overhead": "Python trace; bounded detach/cpu/tolist score copies may synchronize CUDA; no mask copy or extra forward",
                      **capture}
            line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
            size = len(line.encode("utf-8"))
            if size > self.max_record_bytes or self.total_bytes + size > self.max_total_bytes:
                raise DiagnosticError("diagnostic byte budget exceeded")
            self.total_bytes += size
            self.sink(line)
        finally:
            sys.settrace(None)
