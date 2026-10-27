"""Error and warning codes for ``photo-io-v1`` (contract clause 5).

The contract fixes the vocabulary; this module is the single definition so that
the CLI, the row processor and the tests cannot disagree about a spelling.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ERROR_CODES",
    "WARNING_CODES",
    "FrameIssue",
    "Issue",
    "as_json",
]

#: Minimum error vocabulary required by the contract.
ERROR_CODES = (
    "INVALID_MANIFEST",
    "INVALID_PATH",
    "MISSING_FILE",
    "DECODE_ERROR",
    "UNSUPPORTED_IMAGE",
    "INVALID_ORIENTATION",
    "INVALID_MASK",
    "MASK_SIZE_MISMATCH",
    "INPUT_CHANGED",
    "OUTPUT_ERROR",
    "INTERNAL_ERROR",
)

#: Warning vocabulary required by the contract.
WARNING_CODES = ("EMPTY_MASK",)


@dataclass(frozen=True)
class Issue:
    """A single ``{"code": ..., "message": ...}`` entry.

    ``message`` stays short: the contract forbids embedding whole logs or
    unnecessary machine paths in it.
    """

    code: str
    message: str

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES and self.code not in WARNING_CODES:
            raise ValueError(f"unknown issue code: {self.code!r}")


def as_json(issue: Issue) -> dict[str, str]:
    """Plain-JSON form of an :class:`Issue`."""
    return {"code": issue.code, "message": issue.message}


class FrameIssue(Exception):
    """A row-level failure carrying the contract's error code.

    Raised inside row processing and converted to an error record; never used to
    abort the whole run, because the contract requires one output row per input
    line.
    """

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {code!r}")
        super().__init__(f"{code}: {message}")
        self.issue = Issue(code, message)
