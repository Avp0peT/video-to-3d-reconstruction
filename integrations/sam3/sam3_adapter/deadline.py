"""Shared wall-clock budgets plus the irrevocable stop latch.

Two rules matter here:

* **one origin.** Load, per-frame and batch budgets are all derived once from a
  single monotonic origin; a later phase never resets the total, and the
  effective frame deadline is the earlier of its own budget and the remaining
  batch budget.
* **first latch wins.** A stop cause recorded by *any* observer -- the parent
  supervisor, a child-worker event, the monitor, a deadline -- is irrevocable.
  A later sample falling back below the threshold must not un-latch it, and a
  later cleanup error must never overwrite the real stop cause.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .contract import BATCH_BUDGET_SECONDS, FRAME_BUDGET_SECONDS, LOAD_BUDGET_SECONDS


class DeadlineExceeded(RuntimeError):
    """Raised when a phase overran its budget (mapped to RESOURCE_LIMIT)."""


#: stop causes, ordered by the contract run-status they map to
STOP_KINDS = ("RESOURCE_LIMIT", "BACKEND_ERROR", "AUTH_REQUIRED", "BACKEND_UNAVAILABLE", "INPUT_ERROR")


@dataclass
class StopState:
    """Thread-safe, irrevocable stop latch shared by every observer."""

    kind: str | None = None
    reason: str | None = None
    source: str | None = None
    at_utc: str | None = None
    history: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def latched(self) -> bool:
        return self.kind is not None

    def latch(self, kind: str, reason: str, source: str) -> bool:
        """Record a stop cause. Returns True if this call latched it first."""
        with self._lock:
            entry = {
                "kind": kind,
                "reason": reason,
                "source": source,
                "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "first": self.kind is None,
            }
            self.history.append(entry)
            if self.kind is None:
                self.kind = kind
                self.reason = reason
                self.source = source
                self.at_utc = entry["at_utc"]
                return True
            return False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "latched": self.latched,
                "kind": self.kind,
                "reason": self.reason,
                "source": self.source,
                "at_utc": self.at_utc,
                "all_observations": list(self.history),
                "note": "the first latch is irrevocable; later observations are recorded but cannot clear it",
            }


@dataclass
class Budgets:
    load_seconds: float = LOAD_BUDGET_SECONDS
    frame_seconds: float = FRAME_BUDGET_SECONDS
    batch_seconds: float = BATCH_BUDGET_SECONDS
    reclaim_seconds: float = 5.0

    def as_dict(self) -> dict:
        return {
            "load_seconds": self.load_seconds,
            "frame_seconds": self.frame_seconds,
            "batch_seconds": self.batch_seconds,
            "reclaim_seconds": self.reclaim_seconds,
            "origin": "one monotonic origin for the whole run; phases never reset the total",
            "reclaim": "kill, pipe close and wait share this single reclaim deadline",
        }


@dataclass
class Deadline:
    """Absolute deadlines derived from one origin."""

    budgets: Budgets = field(default_factory=Budgets)
    clock: object = time.monotonic
    origin: float | None = None
    load_deadline: float = 0.0
    batch_deadline: float = 0.0
    _frame_index: int | None = None
    _frame_deadline: float = 0.0
    expired: list[str] = field(default_factory=list)
    _reclaim_deadline: float | None = field(default=None, init=False, repr=False)
    _reclaim_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.origin is None:
            self.origin = float(self.clock())
        self.load_deadline = self.origin + self.budgets.load_seconds
        self.batch_deadline = self.origin + self.budgets.batch_seconds

    # ------------------------------------------------------------------ phases
    def begin_load(self) -> None:
        self.check("load", self.load_deadline)

    def end_load(self) -> None:
        self.check("load", self.load_deadline)

    def begin_frame(self, input_index: int) -> None:
        remaining_batch = self.batch_deadline - float(self.clock())
        own = min(self.budgets.frame_seconds, max(remaining_batch, 0.0))
        self._frame_index = input_index
        self._frame_deadline = float(self.clock()) + own

    def frame_deadline_for(self, input_index: int) -> float:
        """Absolute deadline for a frame, clamped by the remaining batch budget."""
        remaining_batch = self.batch_deadline - float(self.clock())
        return float(self.clock()) + min(self.budgets.frame_seconds, max(remaining_batch, 0.0))

    def current_frame_deadline(self) -> float | None:
        if self._frame_index is None:
            return None
        return min(self._frame_deadline, self.batch_deadline)

    def end_frame(self) -> None:
        if self._frame_index is None:
            return
        self.check(f"frame[{self._frame_index}]", self._frame_deadline)

    def check_batch(self) -> None:
        self.check("batch", self.batch_deadline)

    def check(self, label: str, deadline: float) -> None:
        if float(self.clock()) > deadline:
            reason = (
                f"{label} deadline exceeded (budget origin={self.origin:.3f}, "
                f"deadline={deadline:.3f}, now={float(self.clock()):.3f})"
            )
            self.expired.append(reason)
            raise DeadlineExceeded(reason)

    def reclaim_deadline_from_now(self) -> float:
        with self._reclaim_lock:
            if self._reclaim_deadline is None:
                self._reclaim_deadline = float(self.clock()) + self.budgets.reclaim_seconds
            return self._reclaim_deadline

    # ---------------------------------------------------------------- reporting
    def snapshot(self) -> dict:
        now = float(self.clock())
        return {
            "budgets": self.budgets.as_dict(),
            "origin_monotonic": round(self.origin or 0.0, 6),
            "load_deadline_epoch_monotonic": round(self.load_deadline, 6),
            "batch_deadline_epoch_monotonic": round(self.batch_deadline, 6),
            "load_remaining_seconds": round(max(self.load_deadline - now, 0.0), 3),
            "batch_remaining_seconds": round(max(self.batch_deadline - now, 0.0), 3),
            "current_frame": self._frame_index,
            "current_frame_remaining_seconds": (
                round(max(self._frame_deadline - now, 0.0), 3) if self._frame_index is not None else None
            ),
            "expired": list(self.expired),
        }
