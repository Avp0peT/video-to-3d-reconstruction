"""Finite point-set model for two-dimensional images.

Mathematical object
-------------------
Following V. N. Kozlov, a two-dimensional image is a **finite non-empty set**
of points of the plane::

    X = {x_1, x_2, ..., x_n},   x_i in R^2,   n >= 1

Set semantics are a hard requirement of the project (README section 4:
"点的排列顺序改变时，作为数学集合应视作同一输入对象").  Everything in this
module is therefore explicitly *order-insensitive*:

* :meth:`PointSet2D.__eq__` compares sets, never sequences;
* :meth:`PointSet2D.signature` returns an order-independent canonical
  fingerprint;
* :meth:`PointSet2D.sorted_points` gives a deterministic lexicographic
  ordering, which is the canonical starting point for the image-code layer.

Implementation note — set vs. multiset
--------------------------------------
Internally the points are stored exactly as given, so ``PointSet2D`` behaves
like a *multiset*.  Repeated points are legal input (the stability test matrix
in README section 19 lists "重复点" explicitly) and are reported by
:attr:`PointSet2D.has_duplicates`.  Call :meth:`PointSet2D.deduplicated` to
collapse them before feeding a strict-set algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from ..config import FLOAT_TOL, SIGNATURE_DECIMALS

__all__ = [
    "Point2D",
    "PointSet2D",
    "as_point_array",
    "as_point_matrix",
    "quantise_point",
]


# --------------------------------------------------------------------------- #
# coercion helpers
# --------------------------------------------------------------------------- #
def as_point_array(value: Any, *, name: str = "point") -> np.ndarray:
    """Coerce a single point to a ``float64`` array of shape ``(2,)``.

    Accepts :class:`Point2D`, a sequence of two numbers, or an array-like of
    shape ``(2,)``, ``(1, 2)`` or ``(2, 1)``.

    Raises
    ------
    TypeError
        If the value is not point-like at all.
    ValueError
        If the value has the wrong shape or contains non-finite entries.
    """
    if isinstance(value, Point2D):
        arr = np.array([value.x, value.y], dtype=np.float64)
    else:
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError(f"{name}: cannot interpret {value!r} as numbers") from exc

        if arr.shape == (1, 2):
            arr = arr[0]
        elif arr.shape == (2, 1):
            arr = arr.reshape(2)
        elif arr.shape != (2,):
            raise ValueError(f"{name}: expected shape (2,), got {arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: coordinates must be finite, got {arr.tolist()}")
    return arr


def as_point_matrix(value: Any, *, name: str = "points", min_points: int = 1) -> np.ndarray:
    """Coerce any point-container to a ``(n, 2)`` ``float64`` array.

    Accepted inputs: :class:`PointSet2D`, ``(n, 2)`` array-like, sequence of
    :class:`Point2D` or of 2-element sequences, or a mapping with a
    ``"points"`` key (the on-disk JSON format).
    """
    if isinstance(value, PointSet2D):
        arr = value.as_array()
    elif isinstance(value, Mapping):
        if "points" not in value:
            raise ValueError(f"{name}: mapping input needs a 'points' key")
        arr = as_point_matrix(value["points"], name=f"{name}['points']", min_points=min_points)
        return arr
    elif isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1 and arr.shape == (2,):
            arr = arr.reshape(1, 2)
    else:
        try:
            items = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"{name}: cannot iterate over {type(value).__name__}") from exc
        if len(items) == 0:
            arr = np.zeros((0, 2), dtype=np.float64)
        elif isinstance(items[0], Point2D):
            arr = np.array([[p.x, p.y] for p in items], dtype=np.float64)
        else:
            try:
                arr = np.asarray(items, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name}: cannot interpret sequence as points") from exc

    if arr.ndim == 1 and arr.shape == (2,):
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name}: expected shape (n, 2), got {arr.shape}")
    if arr.shape[0] < min_points:
        raise ValueError(f"{name}: expected at least {min_points} point(s), got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"{name}: non-finite coordinate(s) at {bad.tolist()}")
    return np.ascontiguousarray(arr, dtype=np.float64)


def quantise_point(point: Any, decimals: int = SIGNATURE_DECIMALS) -> tuple[int, int]:
    """Map a point onto an integer lattice (used for canonical signatures)."""
    arr = as_point_array(point)
    scale = 10**decimals
    return (int(round(float(arr[0]) * scale)), int(round(float(arr[1]) * scale)))


# --------------------------------------------------------------------------- #
# Point2D
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Point2D:
    """An immutable point of ``R^2``."""

    x: float
    y: float

    def __post_init__(self) -> None:
        fx, fy = float(self.x), float(self.y)
        if not (math.isfinite(fx) and math.isfinite(fy)):
            raise ValueError(f"Point2D({self.x!r}, {self.y!r}) is not finite")
        object.__setattr__(self, "x", fx)
        object.__setattr__(self, "y", fy)

    # -- conversions ------------------------------------------------------ #
    def to_array(self) -> np.ndarray:
        """Return the point as a fresh ``(2,)`` ``float64`` array."""
        return np.array([self.x, self.y], dtype=np.float64)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    # -- geometry --------------------------------------------------------- #
    def norm(self) -> float:
        """Euclidean norm ``|P|`` (distance to the origin)."""
        return math.hypot(self.x, self.y)

    def distance_to(self, other: Any) -> float:
        """Euclidean distance ``|P - Q|``."""
        q = as_point_array(other, name="other")
        return float(math.hypot(self.x - q[0], self.y - q[1]))

    def dot(self, other: Any) -> float:
        q = as_point_array(other, name="other")
        return float(self.x * q[0] + self.y * q[1])

    def cross(self, other: Any) -> float:
        """Scalar cross product ``P x Q`` (the signed area of the parallelogram)."""
        q = as_point_array(other, name="other")
        return float(self.x * q[1] - self.y * q[0])

    def scaled(self, factor: float) -> "Point2D":
        return Point2D(self.x * factor, self.y * factor)

    # -- operators -------------------------------------------------------- #
    def __add__(self, other: Any) -> "Point2D":
        q = as_point_array(other, name="other")
        return Point2D(self.x + q[0], self.y + q[1])

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Point2D":
        q = as_point_array(other, name="other")
        return Point2D(self.x - q[0], self.y - q[1])

    def __neg__(self) -> "Point2D":
        return Point2D(-self.x, -self.y)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Point2D({self.x:g}, {self.y:g})"


# --------------------------------------------------------------------------- #
# PointSet2D
# --------------------------------------------------------------------------- #
class PointSet2D:
    """A finite non-empty point set in ``R^2`` with *set* semantics.

    Parameters
    ----------
    points:
        Anything accepted by :func:`as_point_matrix` — an ``(n, 2)`` array, a
        list of ``[x, y]`` pairs, a list of :class:`Point2D`, another
        :class:`PointSet2D`, or a mapping with a ``"points"`` key.
    name:
        Optional human-readable label (kept in JSON dumps).
    meta:
        Optional free-form metadata.

    Notes
    -----
    The class is deliberately immutable from the outside: every "mutating"
    operation returns a new instance.  The GUI layer (README section 4) is
    expected to hold one such value and swap it out, which keeps the display
    and the mathematical object in sync by construction.
    """

    __slots__ = ("_pts", "name", "meta")

    def __init__(
        self,
        points: Any,
        *,
        name: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self._pts: np.ndarray = as_point_matrix(points)
        self.name: str | None = name
        self.meta: dict[str, Any] = dict(meta) if meta else {}

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, payload: Mapping[str, Any] | str) -> "PointSet2D":
        """Build a point set from the project's JSON format.

        ``{"name": ..., "points": [[x, y], ...], "meta": {...}}``

        ``payload`` may be an already-parsed mapping *or* the JSON text.  Use
        :meth:`load` to read a file — a bare string is never interpreted as a
        path here, so ``from_json("{...}")`` and ``load("x.json")`` cannot be
        confused with one another.
        """
        if isinstance(payload, str):
            text = payload.strip()
            if not text.startswith("{"):
                raise ValueError(
                    "from_json() expects JSON text or a mapping; "
                    "use PointSet2D.load(path) to read a file"
                )
            raw: Any = json.loads(text)
        else:
            raw = dict(payload)
        if not isinstance(raw, Mapping):
            raise ValueError("JSON payload must be an object")
        return cls(
            raw.get("points", []),
            name=raw.get("name"),
            meta=raw.get("meta"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PointSet2D":
        """Read a point set from a ``.json`` file written by :meth:`save`."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(raw)

    def to_json(self) -> dict[str, Any]:
        """Serialise to the project's JSON format (id-stable round trip)."""
        payload: dict[str, Any] = {
            "name": self.name,
            "points": [[float(x), float(y)] for x, y in self._pts],
        }
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload

    def to_json_text(self) -> str:
        """Pretty-print the JSON format, one point per line.

        ``json.dumps(indent=2)`` would spread every ``[x, y]`` over five lines and
        make the example files unreadable; this format is meant to be opened in a
        text editor and edited by hand (README section 4).  ``repr`` of a finite
        float is always a valid JSON number, so the output stays machine-readable.
        """
        name = "null" if self.name is None else json.dumps(self.name, ensure_ascii=False)
        points = "[\n" + ",\n".join(
            f"    [{float(x)!r}, {float(y)!r}]" for x, y in self._pts
        ) + "\n  ]"
        parts = [f'  "name": {name}', f'  "points": {points}']
        if self.meta:
            parts.append('  "meta": ' + json.dumps(self.meta, ensure_ascii=False))
        return "{\n" + ",\n".join(parts) + "\n}"

    def save(self, path: str | Path) -> Path:
        """Write the JSON representation, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json_text() + "\n", encoding="utf-8", newline="\n")
        return target

    # ------------------------------------------------------------------ #
    # basic access
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self._pts.shape[0])

    @property
    def n(self) -> int:
        """Number of points (multiset cardinality)."""
        return len(self)

    def __iter__(self) -> Iterator[Point2D]:
        for x, y in self._pts:
            yield Point2D(float(x), float(y))

    def __getitem__(self, index: int | slice) -> Point2D | "PointSet2D":
        if isinstance(index, slice):
            return PointSet2D(self._pts[index], name=self.name, meta=self.meta)
        x, y = self._pts[index]
        return Point2D(float(x), float(y))

    def point(self, index: int) -> Point2D:
        """Return point ``index`` as a :class:`Point2D`."""
        return self[index]  # type: ignore[return-value]

    def as_array(self) -> np.ndarray:
        """A fresh ``(n, 2)`` copy of the coordinates."""
        return self._pts.copy()

    def to_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self._pts]

    def with_points(self, points: Any, *, name: str | None = None) -> "PointSet2D":
        """Return a new set with the same identity but new coordinates."""
        return PointSet2D(
            points,
            name=self.name if name is None else name,
            meta=self.meta,
        )

    # ------------------------------------------------------------------ #
    # set semantics
    # ------------------------------------------------------------------ #
    @property
    def has_duplicates(self) -> bool:
        """True when two points coincide (within :data:`FLOAT_TOL`)."""
        return self.duplicate_count() > 0

    def duplicate_count(self) -> int:
        """Number of points that are repeats of an earlier point.

        ``|multiset| - |set|``: the number of entries that vanish when the
        collection is read as a mathematical set.
        """
        return self.n - len(self.signature())

    def deduplicated(self, tol: float = FLOAT_TOL) -> "PointSet2D":
        """Collapse coincident points, preserving first-occurrence order.

        ``tol`` defaults to :data:`FLOAT_TOL`, i.e. "bit-identical" only.  Use
        a larger value to merge near-identical points (e.g. after a
        compression step).
        """
        if tol <= 0:
            raise ValueError("tol must be > 0")
        seen: list[tuple[float, float]] = []
        kept: list[list[float]] = []
        for x, y in self._pts:
            hit = False
            for sx, sy in seen:
                if math.hypot(x - sx, y - sy) <= tol:
                    hit = True
                    break
            if not hit:
                seen.append((float(x), float(y)))
                kept.append([float(x), float(y)])
        if not kept:  # pragma: no cover - empty input is rejected upstream
            raise ValueError("deduplicated() produced an empty set")
        return PointSet2D(kept, name=self.name, meta=self.meta)

    def signature(self, decimals: int = SIGNATURE_DECIMALS) -> frozenset[tuple[int, int]]:
        """Order-independent fingerprint of the point set — **set** semantics.

        Two collections are equal *as sets* when their signatures agree, so
        ``[[0,0], [1,1]]`` and ``[[0,0], [1,1], [1,1]]`` have the same
        signature.  That is deliberate (the project reads an image as a
        mathematical set) and it is what :meth:`duplicate_count` exploits:
        ``|multiset| - |signature|`` is the number of repeats.

        Use :meth:`multiset_signature` when multiplicity must be part of the
        identity.  Because the coordinates are snapped to an integer lattice of
        ``10**-decimals`` resolution, equality is tolerance-aware but can miss
        points straddling a lattice cell boundary; for an exact geometric
        comparison use :meth:`hausdorff_distance`.
        """
        return frozenset(quantise_point(p, decimals) for p in self._pts)

    def multiset_signature(
        self, decimals: int = SIGNATURE_DECIMALS
    ) -> frozenset[tuple[tuple[int, int], int]]:
        """Fingerprint that **includes multiplicity** (a multiset fingerprint).

        ``((point, count), ...)``.  Deliberately separate from
        :meth:`signature`: recognition (Phase 4) needs to distinguish "the same
        shape" from "the same shape with a duplicated point", and conflating the
        two would make the missing/extra-point experiments of README section 13
        unmeasurable.
        """
        counts: dict[tuple[int, int], int] = {}
        for point in self._pts:
            key = quantise_point(point, decimals)
            counts[key] = counts.get(key, 0) + 1
        return frozenset(counts.items())

    def is_same_multiset(self, other: object) -> bool:
        """``True`` when both the points and their multiplicities agree."""
        if not isinstance(other, PointSet2D):
            return NotImplemented  # type: ignore[return-value]
        return self.multiset_signature() == other.multiset_signature()

    def sorted_points(self) -> np.ndarray:
        """Points in a deterministic order: lexicographic by ``(x, y)``.

        This is the canonical entry point for order-dependent downstream
        algorithms (image codes in Phase 3).
        """
        order = np.lexsort((self._pts[:, 1], self._pts[:, 0]))
        return self._pts[order].copy()

    def sorted_set(self) -> "PointSet2D":
        return PointSet2D(self.sorted_points(), name=self.name, meta=self.meta)

    def as_point_set(self) -> "PointSet2D":
        """Alias for :meth:`deduplicated` — "as a mathematical set"."""
        return self.deduplicated()

    def hausdorff_distance(self, other: Any) -> float:
        """Symmetric Hausdorff distance to ``other`` (order-independent).

        For two sets of equal size whose minimum separation exceeds the
        distance, a small Hausdorff distance *is* a bijection witness.
        """
        b = as_point_matrix(other, name="other")
        if self.n == 0 or b.shape[0] == 0:  # pragma: no cover - guarded upstream
            return math.inf

        def _directed(src: np.ndarray, dst: np.ndarray) -> float:
            diff = src[:, None, :] - dst[None, :, :]
            return float(np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).min(axis=1).max())

        return max(_directed(self._pts, b), _directed(b, self._pts))

    def __eq__(self, other: object) -> bool:
        """Equality **as mathematical sets**: order- and multiplicity-insensitive.

        ``[[0,0], [1,1]] == [[1,1], [0,0], [0,0]]`` is ``True``: both describe
        the same two-element subset of the plane.  This is the semantics the
        project README asks for ("点的排列顺序改变时，作为数学集合应视作同一输入对象").
        Use :meth:`is_same_multiset` when the point *count* is part of the
        identity — e.g. when comparing an image with one that has extra points.
        """
        if not isinstance(other, PointSet2D):
            return NotImplemented
        return self.signature() == other.signature()

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return hash(self.signature())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        label = f" {self.name!r}" if self.name else ""
        return f"PointSet2D{label}(n={self.n})"

    # ------------------------------------------------------------------ #
    # measurements
    # ------------------------------------------------------------------ #
    def centroid(self) -> Point2D:
        c = self._pts.mean(axis=0)
        return Point2D(float(c[0]), float(c[1]))

    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)``."""
        lo = self._pts.min(axis=0)
        hi = self._pts.max(axis=0)
        return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

    def bounding_box_size(self) -> tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bounds()
        return (xmax - xmin, ymax - ymin)

    def diameter(self) -> float:
        """Largest pairwise distance."""
        if self.n < 2:
            return 0.0
        d = self.distance_matrix()
        return float(d.max())

    def characteristic_scale(self) -> float:
        """A robust length scale used to normalise other quantities.

        Falls back to ``1.0`` for degenerate (single-point / coincident)
        inputs so that callers never divide by zero.
        """
        scale = self.diameter()
        return scale if scale > FLOAT_TOL else 1.0

    def distance_matrix(self) -> np.ndarray:
        """``(n, n)`` matrix of pairwise Euclidean distances."""
        diff = self._pts[:, None, :] - self._pts[None, :, :]
        return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    def pairwise_distances(self) -> np.ndarray:
        """Upper-triangular part of :meth:`distance_matrix` as a flat vector."""
        iu = np.triu_indices(self.n, k=1)
        return self.distance_matrix()[iu]

    def is_degenerate(self) -> bool:
        """All points coincide (i.e. the set spans no direction)."""
        return self.diameter() <= FLOAT_TOL

    def is_one_dimensional(self) -> bool:
        """Every point lies on a single line (possibly degenerate).

        Uses the singular values of the centred coordinate matrix: the second
        singular value is exactly ``sqrt(n)`` times the RMS distance to the
        best-fit line.
        """
        if self.n <= 2:
            return True
        centred = self._pts - self._pts.mean(axis=0)
        singular = np.linalg.svd(centred, compute_uv=False)
        scale = singular[0] if singular.size else 0.0
        if scale <= FLOAT_TOL:
            return True
        return bool(singular[1] / scale <= 1e-9)

    def distance(self, i: int, j: int) -> float:
        """Euclidean distance between points ``i`` and ``j``."""
        delta = self._pts[i] - self._pts[j]
        return float(math.hypot(float(delta[0]), float(delta[1])))
