"""The compressor contract and result type (G5.2).

Every compressor must declare *what it guarantees*, because "we compressed 300 points to 82"
is not a mathematical statement.  The contract fields are the ones the specification requires:

``surjective_on``
    Whether the sketch is a subset of the input.  This matters because the source's эскиз is
    defined as a fragment ($A_r\\subseteq A$), so a compressor that returns points *not in*
    the input is not producing an эскиз however good its ratio.
``cover_guarantee``
    ``"guaranteed"`` when a proof accompanies the declared radius, ``"measured"`` when the
    radius is merely observed.  A measured radius must never be reported as a guarantee.
``cardinality_guarantee``
    ``"minimum"``, ``"upper_bound"`` or ``"fixed"`` — and ``"minimum"`` may only be claimed
    when the solver proved it (the exact branch-and-bound path), never when a heuristic was
    used.
``transform_equivariance``
    **Not a claim to be trusted**: the specification requires it to be *measured* (5D), and
    for Euclidean covers under general affine maps the expectation is that it fails.  The field
    records what has been established, defaulting to ``"unmeasured"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import numpy as np

__all__ = [
    "CompressorContract",
    "CompressionResult",
    "Compressor",
    "COMPRESSORS",
    "get_compressor",
    "list_compressors",
    "run_compressor",
]


@dataclass(frozen=True)
class CompressorContract:
    """What a compressor promises, in fields a test can check."""

    name: str
    family: str  # "official" | "teaching" | "experimental" | "control"
    #: Result is a subset of the input (required for the source's эскиз).
    surjective_on: bool
    #: "guaranteed" (with proof in the docstring) or "measured".
    cover_guarantee: str
    #: "minimum" (proved), "upper_bound", or "fixed".
    cardinality_guarantee: str
    deterministic: bool
    complexity: str
    source: str
    #: Measured, never assumed.  ``None``/``"unmeasured"`` until 5D has run.
    transform_equivariance: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "surjective_on": self.surjective_on,
            "cover_guarantee": self.cover_guarantee,
            "cardinality_guarantee": self.cardinality_guarantee,
            "deterministic": self.deterministic,
            "complexity": self.complexity,
            "source": self.source,
            "transform_equivariance": dict(self.transform_equivariance),
        }


@dataclass(frozen=True)
class CompressionResult:
    """A sketch plus everything needed to judge it."""

    points: np.ndarray
    indices: tuple[int, ...]
    contract: CompressorContract
    #: Radius the compressor *aimed* at (``None`` for cardinality-parameterised methods).
    requested_radius: float | None
    #: Radius actually achieved; for a guaranteed cover this is <= requested.
    achieved_radius: float
    build_ms: float
    peak_bytes: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_before(self) -> int:
        return int(self.diagnostics.get("n_before", 0))

    @property
    def n_after(self) -> int:
        return int(self.points.shape[0])

    @property
    def compression_ratio(self) -> float:
        before = self.n_before
        return 0.0 if before == 0 else 1.0 - self.n_after / before

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_before": self.n_before,
            "n_after": self.n_after,
            "compression_ratio": self.compression_ratio,
            "requested_radius": self.requested_radius,
            "achieved_radius": self.achieved_radius,
            "build_ms": self.build_ms,
            "peak_bytes": self.peak_bytes,
            "contract": self.contract.as_dict(),
            "diagnostics": dict(self.diagnostics),
        }


class Compressor(ABC):
    """Base class: a compressor is a declared contract plus a ``fit``."""

    contract: ClassVar[CompressorContract]

    @abstractmethod
    def fit(self, points: Any, **parameters: Any) -> CompressionResult:  # pragma: no cover
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------- #
    @staticmethod
    def _matrix(points: Any) -> np.ndarray:
        """The image **preserving its arithmetic**.

        No ``dtype=float`` here, and that is a correctness requirement rather than a detail.  A
        first version coerced to ``float64``, which meant the identity compressor turned exact
        rationals into floats — so ``C_0(X) != X``, and the exact recogniser then refused the
        result.  The audit's Control A and Control D both failed on that, and the same coercion
        is why the earlier compression experiment reported universal failure.

        Compressors that need to measure distances take a float view via
        :meth:`_float_view`; the points they *return* keep the type they were given.
        """
        array = np.asarray(points)
        if array.ndim == 1 and array.shape == (2,):
            array = array.reshape(1, 2)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"expected shape (n, 2), got {array.shape}")
        if array.shape[0] == 0:
            raise ValueError("an image must be non-empty")
        return array

    @staticmethod
    def _float_view(matrix: np.ndarray) -> np.ndarray:
        """A float64 view for *measurement* only — never for the returned points."""
        return np.asarray(matrix, dtype=float)

    @staticmethod
    def _result(
        matrix: np.ndarray,
        indices: list[int],
        contract: CompressorContract,
        *,
        requested_radius: float | None,
        achieved_radius: float,
        build_ms: float,
        peak_bytes: int,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> CompressionResult:
        payload = {"n_before": int(matrix.shape[0])}
        if diagnostics:
            payload.update(diagnostics)
        return CompressionResult(
            points=matrix[indices] if indices else np.zeros((0, 2)),
            indices=tuple(indices),
            contract=contract,
            requested_radius=requested_radius,
            achieved_radius=achieved_radius,
            build_ms=build_ms,
            peak_bytes=peak_bytes,
            diagnostics=payload,
        )


def _registry() -> dict[str, type[Compressor]]:
    """Import lazily so that importing this module does not pull in every compressor."""
    from .control.identity import IdentityCompressor
    from .experimental.affine_aware import AffineAwareSampler
    from .official.kozlov_sketch import KozlovSketch, KozlovSkeleton
    from .teaching.epsilon_cover import TeachingEpsilonCover
    from .teaching.farthest_point import TeachingFarthestPoint
    from .teaching.grid_reduce import TeachingGridReduce

    return {
        "KozlovSketch": KozlovSketch,
        "KozlovSkeleton": KozlovSkeleton,
        "TeachingEpsilonCover": TeachingEpsilonCover,
        "TeachingFarthestPoint": TeachingFarthestPoint,
        "TeachingGridReduce": TeachingGridReduce,
        "AffineAwareSampler": AffineAwareSampler,
        "IdentityCompressor": IdentityCompressor,
    }


class _LazyRegistry(dict):
    """A dict that materialises on first use, so ``COMPRESSORS`` stays import-light."""

    def _ensure(self) -> None:
        if not dict.__len__(self):
            dict.update(self, _registry())

    def __getitem__(self, key: str) -> type[Compressor]:
        self._ensure()
        return dict.__getitem__(self, key)

    def __iter__(self):
        self._ensure()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._ensure()
        return dict.__len__(self)

    def keys(self):  # type: ignore[override]
        self._ensure()
        return dict.keys(self)

    def values(self):  # type: ignore[override]
        self._ensure()
        return dict.values(self)

    def items(self):  # type: ignore[override]
        self._ensure()
        return dict.items(self)

    def __contains__(self, key: object) -> bool:
        self._ensure()
        return dict.__contains__(self, key)


#: Name → compressor class.  The names carry the family, so a call site cannot accidentally
#: attribute a baseline to the source (G5.9).
COMPRESSORS: dict[str, type[Compressor]] = _LazyRegistry()


def get_compressor(name: str) -> type[Compressor]:
    if name not in COMPRESSORS:
        raise KeyError(f"unknown compressor {name!r}; available: {', '.join(sorted(COMPRESSORS))}")
    return COMPRESSORS[name]


def list_compressors() -> list[dict[str, Any]]:
    """Every compressor with its contract — the table a report should quote."""
    return [COMPRESSORS[name].contract.as_dict() for name in sorted(COMPRESSORS)]


def run_compressor(name: str, points: Any, **parameters: Any) -> CompressionResult:
    """Convenience: look up, run, and check the family-appropriate sanity condition."""
    result = get_compressor(name)().fit(points, **parameters)
    if result.contract.surjective_on:
        matrix = Compressor._matrix(points)
        available = {tuple(point) for point in matrix}
        for point in result.points:
            if tuple(point) not in available:
                raise AssertionError(
                    f"{name} declares surjective_on=True but returned a point outside the input; "
                    "the source's эскиз is a subset by definition"
                )
    return result
