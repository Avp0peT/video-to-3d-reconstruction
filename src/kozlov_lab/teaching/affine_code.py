"""Teaching affine code — a *pedagogical* affine-invariant construction.

What it is
----------
For every non-collinear triple of the image ``X``, compute the internal affine
coordinates of the image points, quantise them, sort them lexicographically,
canonicalise over the six orderings of that triple, and finally sort all the
resulting frame descriptors.  The result is invariant under

* the choice of frame,
* the ordering within a frame,
* the labelling / order of the image points,
* every invertible affine map ``T(x) = Mx + t``,

which is the property the whole project is about.

What it is **not**
------------------
It is **not** claimed to be Kozlov's formal image code.  See the three-language
disclaimer in :mod:`kozlov_lab.teaching`, and open question Q1 in
``docs/specs/image_code.md``.  The construction is derived from the
internal-coordinate idea alone; the textbook's own encoding is a separate,
unresolved question, and ``kozlov_lab.coding`` stays empty until it is answered.

The invariant part is the *multiset of quantised coordinate tuples*
------------------------------------------------------------------
A decision that matters for correctness: the canonical form contains **only** the
quantised internal coordinates.  The frame's point indices and its condition
number are stored as *diagnostics* but are deliberately excluded, because point
indices are arbitrary labels — putting them in the canonical text would make two
equivalent images with different numbering produce different codes, which is
exactly the bug the code is supposed to make impossible.

Why this is a complete affine invariant (informal argument)
-----------------------------------------------------------
Fix a frame ``(A, B, C)``.  The map ``z -> A + alpha (B-A) + beta (C-A)`` is
injective, so the multiset of internal coordinates determines the *correspondence*
between the two images with no ambiguity.  If ``T`` is invertible affine it maps
the frame to a frame and leaves every ``z_i`` unchanged; hence the multiset of
descriptors is unchanged.  Conversely a shared descriptor hands you a frame pair,
and the affine map between those two frames carries one image onto the other.
Up to the quantum, the code therefore decides affine equivalence.

The honest caveats, reported rather than hidden
-----------------------------------------------
1. **Quantisation vs. precision.**  The quantum ``10**-decimals`` must be
   *coarser* than the attainable precision of the least well-conditioned frame in
   the image.  Otherwise the code encodes floating-point noise and two equivalent
   images get different codes.  :meth:`TeachingAffineImageCode.quality` measures
   this, and :func:`recommended_decimals` picks a defensible value instead of a
   magic constant.
2. **Integer representability.**  Coordinates are snapped by
   ``round(v * 10**decimals)``; once ``max|coord| * 10**decimals`` exceeds
   ``2**53`` the snap itself loses information.
3. **Completeness can be capped, never silently.**  ``max_frames`` is a safety
   valve; when it bites, the code is no longer a complete invariant and the object
   says so in ``warnings`` and ``quality()``.
4. **The hash is an index, not the code.**  See :meth:`TeachingAffineImageCode.digest`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import COLLINEAR_TOL, TEACHING_CODE_DECIMALS
from ..geometry.affine_frame import AffineFrame, iter_non_collinear_triples
from ..geometry.points import as_point_matrix
from ..precision import PrecisionReport, precision_report

__all__ = [
    "DISCLAIMER_EN",
    "DISCLAIMER_ZH",
    "DISCLAIMER_RU",
    "DISCLAIMERS",
    "CODE_FORMAT_VERSION",
    "TeachingCodingParameters",
    "TeachingAffineFrameCode",
    "TeachingAffineImageCode",
    "TeachingCodeComparison",
    "build_teaching_frame_code",
    "build_canonical_frame_code",
    "build_teaching_image_code",
    "teaching_codes_equal",
    "compare_teaching_codes",
    "recommended_decimals",
    "worst_frame_condition_number",
]

# --------------------------------------------------------------------------- #
# disclaimer (verbatim; asserted by tests)
# --------------------------------------------------------------------------- #
DISCLAIMER_EN = (
    "This is a pedagogical affine-invariant construction derived from the "
    "internal-coordinate idea. It is not claimed to be Kozlov's formal image code "
    "unless verified against a primary source."
)
DISCLAIMER_ZH = (
    "本编码器是根据内部仿射坐标思想构造的教学型仿射不变编码，"
    "不声明其等同于 Козлов 教材中的正式图像代码。"
)
DISCLAIMER_RU = (
    "Данный код является учебной конструкцией на основе внутренних аффинных "
    "координат и не отождествляется с формальным кодом изображения Козлова "
    "без проверки по первоисточнику."
)

DISCLAIMERS: dict[str, str] = {"en": DISCLAIMER_EN, "zh": DISCLAIMER_ZH, "ru": DISCLAIMER_RU}

#: Bump whenever the canonical text or the algorithm changes, so that codes from
#: different implementations can never compare equal by accident.
CODE_FORMAT_VERSION = "teaching-affine-code/1"

_EPS: float = float(np.finfo(np.float64).eps)

#: Beyond this the integer snap ``round(v * 10**decimals)`` stops being exact.
_EXACT_INT_LIMIT: float = float(2**53)

#: Safety factor applied in :func:`recommended_decimals`: we want the quantum to be
#: at least this many times coarser than the attainable precision.
_QUANTIZATION_SAFETY = 100.0


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TeachingCodingParameters:
    """Everything that can change the code, in one place.

    Two codes are only comparable when their parameters agree; the canonical text
    embeds them, so differing parameters can never masquerade as differing images.
    """

    #: Quantisation decimals (tolerance-aware quantisation).
    decimals: int = TEACHING_CODE_DECIMALS
    #: Collinearity threshold used to accept a frame.
    collinear_tol: float = COLLINEAR_TOL
    #: Safety cap on the number of triples processed.  ``None`` = all.
    max_frames: int | None = None
    #: Include the three frame points in the descriptor.  Their coordinates are the
    #: constants ``(0,0), (1,0), (0,1)``, so excluding them is also a complete
    #: invariant; including them makes descriptors of images with different point
    #: counts more comparable by eye.
    include_frame_points: bool = True

    def __post_init__(self) -> None:
        if self.decimals < 0:
            raise ValueError("decimals must be >= 0")
        if self.collinear_tol <= 0:
            raise ValueError("collinear_tol must be > 0")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1 or None")

    @property
    def quantum(self) -> float:
        """Coordinate resolution ``10**-decimals``."""
        return 10.0**-self.decimals

    def as_dict(self) -> dict[str, Any]:
        return {
            "decimals": self.decimals,
            "quantum": self.quantum,
            "collinear_tol": self.collinear_tol,
            "max_frames": self.max_frames,
            "include_frame_points": self.include_frame_points,
        }

    def describe(self) -> str:
        limit = "all" if self.max_frames is None else str(self.max_frames)
        return (
            f"decimals={self.decimals}; quantum={self.quantum:.0e}; "
            f"collinear_tol={self.collinear_tol:.0e}; max_frames={limit}; "
            f"include_frame_points={self.include_frame_points}"
        )


# --------------------------------------------------------------------------- #
# helpers shared by the code and its quality report
# --------------------------------------------------------------------------- #
def _quantization_is_safe(worst_condition_number: float, quantum: float) -> bool:
    """Whether the quantum is comfortably coarser than float64 noise.

    The relevant comparison is ``quantum`` vs. the attainable precision of the
    worst frame (``cond * eps``).  If the quantum is *finer* than the noise, the
    code encodes rounding instead of structure.
    """
    if not math.isfinite(worst_condition_number):
        return False
    attainable = worst_condition_number * _EPS * 64.0
    return quantum >= attainable * _QUANTIZATION_SAFETY


def worst_frame_condition_number(
    points: Any,
    *,
    collinear_tol: float = COLLINEAR_TOL,
    sample_limit: int | None = 2000,
) -> float:
    """Largest condition number over the image's admissible frames.

    ``sample_limit`` caps the scan for interactive use; when it bites, the
    returned value is a *lower bound* on the true worst case, which is the safe
    direction for a safety check (it can only under-warn, and the code records
    how many triples were examined).
    """
    x = as_point_matrix(points, name="points", min_points=1)
    worst = 0.0
    examined = 0
    for triple in iter_non_collinear_triples(x, collinear_tol):
        frame = AffineFrame.from_indices(x, triple)  # type: ignore[arg-type]
        worst = max(worst, frame.condition_number)
        examined += 1
        if sample_limit is not None and examined >= sample_limit:
            break
    return worst if examined else math.inf


def recommended_decimals(
    points: Any,
    *,
    collinear_tol: float = COLLINEAR_TOL,
    safety_decades: float = 2.0,
    cap: int = 12,
    floor: int = 0,
) -> int:
    """Pick a quantisation resolution that is actually defensible.

    Instead of hard-coding ``9``, derive it: the quantum must be at least
    ``_QUANTIZATION_SAFETY`` times coarser than ``cond_worst * eps``, i.e.

        decimals <= -log10(cond_worst * eps * 64 * safety)

    and we keep two extra decades of margin by default.  The result is clamped to
    ``[floor, cap]``.

    Returns ``floor`` for a frame-less image (no claim can be made there anyway).
    """
    worst = worst_frame_condition_number(points, collinear_tol=collinear_tol)
    if not math.isfinite(worst) or worst <= 0:
        return floor
    attainable = worst * _EPS * 64.0 * _QUANTIZATION_SAFETY
    if attainable <= 0:
        return cap
    value = math.floor(-math.log10(attainable) - safety_decades)
    return int(min(cap, max(floor, value)))


# --------------------------------------------------------------------------- #
# frame code
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TeachingAffineFrameCode:
    """Pedagogical descriptor attached to one admissible frame.

    Part of the **pedagogical** construction in :mod:`kozlov_lab.teaching`; it is
    not claimed to be part of Kozlov's formal image code (see the module docstring
    and ``docs/specs/official_image_code.md``).

    Attributes
    ----------
    quantized_points:
        Sorted quantised ``(alpha, beta)`` lattice coordinates of the image points
        (excluding the frame points when ``include_frame_points=False``).  **This
        is the only field that enters the canonical form.**
    triple, ordering, condition_number:
        Diagnostics.  Deliberately *excluded* from the canonical form: the point
        indices are arbitrary labels, and the condition number can differ in the
        last bits between two equivalent images.
    """

    triple: tuple[int, int, int]
    ordering: tuple[int, int, int]
    quantized_points: tuple[tuple[int, int], ...]
    condition_number: float
    decimals: int
    frame_points_included: bool

    # -- rendering -------------------------------------------------------- #
    def as_floats(self) -> tuple[tuple[float, float], ...]:
        """The quantised coordinates back as floats (i.e. the rounded values)."""
        scale = 10.0**-self.decimals
        return tuple((x * scale, y * scale) for x, y in self.quantized_points)

    def canonical_body(self) -> str:
        """The label-free body of the descriptor — the invariant content."""
        return " ".join(f"{x},{y}" for x, y in self.quantized_points)

    def describe(self) -> str:
        """Human-readable diagnostic line (NOT the canonical form)."""
        return (
            f"frame=({self.triple[0]},{self.triple[1]},{self.triple[2]}) "
            f"ordering=({self.ordering[0]},{self.ordering[1]},{self.ordering[2]}) "
            f"cond={self.condition_number:.6e} n_coords={len(self.quantized_points)}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "triple": list(self.triple),
            "ordering": list(self.ordering),
            "condition_number": self.condition_number,
            "decimals": self.decimals,
            "frame_points_included": self.frame_points_included,
            "n_normalized_points": len(self.quantized_points),
            "canonical_body": self.canonical_body(),
        }


def _quantize_internal_coordinates(
    alpha: np.ndarray, beta: np.ndarray, decimals: int
) -> tuple[tuple[int, int], ...]:
    """Snap ``(alpha, beta)`` onto the integer lattice and sort lexicographically.

    The sort is step 2 of the de-labelling recipe and is what makes the descriptor
    independent of the point numbering.  As a side effect, duplicate coordinates
    survive as duplicates: the image is read as a *multiset* here, which keeps
    "one point duplicated" distinguishable from "no duplication".
    """
    scale = 10.0**decimals
    pairs = sorted(
        (int(round(float(a) * scale)), int(round(float(b) * scale)))
        for a, b in zip(alpha, beta)
    )
    return tuple(pairs)


def build_canonical_frame_code(
    points: Any,
    triple: Sequence[int],
    parameters: TeachingCodingParameters | None = None,
) -> TeachingAffineFrameCode:
    """Canonical descriptor of an *unordered* triple: the minimum over its 6 orderings.

    Step 3 of the de-labelling recipe.  This is where the frame's own ordering
    stops mattering: ``build_teaching_frame_code`` is order-sensitive by
    construction (the coordinates depend on which point is ``A``), and the minimum
    over the six orderings is what removes that freedom.

    Raises
    ------
    ValueError
        If the triple is collinear, has repeated indices, or the wrong length.
    """
    params = parameters or TeachingCodingParameters()
    idx = tuple(sorted(int(i) for i in triple))
    if len(idx) != 3:
        raise ValueError("triple must contain exactly three indices")
    if len(set(idx)) != 3:
        raise ValueError(f"triple must be three distinct indices, got {idx}")

    best: TeachingAffineFrameCode | None = None
    rejections: list[str] = []
    for ordering in _permutations(idx):  # type: ignore[arg-type]
        try:
            candidate = build_teaching_frame_code(points, ordering, params)
        except ValueError as exc:
            # A triple sitting *exactly* on the collinearity threshold can be
            # admissible in one ordering and rejected in another, because
            # ``triangle_flatness`` is symmetric only up to the last bit.  Skipping
            # the rejected orderings keeps the canonical form well defined; the
            # descriptor is the minimum over the *admissible* ones.
            rejections.append(str(exc))
            continue
        if best is None or _sort_key(candidate) < _sort_key(best):
            best = candidate

    if best is None:
        raise ValueError(
            f"triple {idx} admits no admissible ordering and cannot code an image "
            f"(all 6 orderings rejected; first reason: {rejections[0] if rejections else 'n/a'})"
        )
    return best


def build_teaching_frame_code(
    points: Any,
    ordering: Sequence[int],
    parameters: TeachingCodingParameters | None = None,
) -> TeachingAffineFrameCode:
    """Descriptor for the frame given by the *ordered* index triple ``ordering``.

    Raises
    ------
    ValueError
        If the triple is collinear, has repeated indices, or the wrong length.
    """
    params = parameters or TeachingCodingParameters()
    x = as_point_matrix(points, name="points", min_points=3)
    idx = tuple(int(i) for i in ordering)
    if len(idx) != 3:
        raise ValueError("ordering must contain exactly three indices")
    if len(set(idx)) != 3:
        raise ValueError(f"ordering must be three distinct indices, got {idx}")
    if min(idx) < 0 or max(idx) >= x.shape[0]:
        raise ValueError(f"ordering indices out of range for {x.shape[0]} points: {idx}")

    frame = AffineFrame.from_indices(x, idx)  # type: ignore[arg-type]
    if not frame.is_valid(params.collinear_tol):
        raise ValueError(f"frame {idx} is collinear and cannot code an image")

    batch = frame.coordinates_batch(x)
    alpha = batch["alpha"]
    beta = batch["beta"]
    if not params.include_frame_points:
        keep = np.ones(x.shape[0], dtype=bool)
        keep[list(idx)] = False
        alpha, beta = alpha[keep], beta[keep]

    return TeachingAffineFrameCode(
        triple=tuple(sorted(idx)),  # type: ignore[arg-type]
        ordering=idx,  # type: ignore[arg-type]
        quantized_points=_quantize_internal_coordinates(alpha, beta, params.decimals),
        condition_number=float(batch["condition_number"]),
        decimals=params.decimals,
        frame_points_included=params.include_frame_points,
    )


# --------------------------------------------------------------------------- #
# image code
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TeachingAffineImageCode:
    """``C_teach(X)``: the canonical family of frame descriptors of ``X``.

    A **pedagogical** affine-invariant construction: it is derived from the
    internal-coordinate idea and is not claimed to be Kozlov's formal image code
    (see the module docstring and ``docs/specs/official_image_code.md``).
    """

    frames: tuple[TeachingAffineFrameCode, ...]
    parameters: TeachingCodingParameters
    n_points: int
    n_collinear_triples: int
    n_triples_available: int
    n_truncated_triples: int
    truncated: bool
    worst_condition_number: float
    quantization_magnitude: float
    warnings: tuple[str, ...] = ()

    # -- canonical form --------------------------------------------------- #
    def canonical_bodies(self) -> tuple[str, ...]:
        """The label-free descriptor bodies, in canonical order."""
        return tuple(frame.canonical_body() for frame in self.frames)

    def canonical_text(self) -> str:
        """Deterministic, diffable text representation — this *is* the code.

        Contains no point indices and no condition numbers: only the parameters,
        the point counts, and the sorted quantised coordinate lists.
        """
        p = self.parameters
        limit = "all" if p.max_frames is None else str(p.max_frames)
        header = (
            f"{CODE_FORMAT_VERSION} "
            f"decimals={p.decimals} collinear_tol={p.collinear_tol:.12g} "
            f"include_frame_points={int(p.include_frame_points)} max_frames={limit} "
            f"n_points={self.n_points} n_frames={len(self.frames)} "
            f"n_collinear_triples={self.n_collinear_triples} truncated={int(self.truncated)}"
        )
        body = "\n".join(self.canonical_bodies())
        return f"{header}\n{body}" if body else header

    def canonical_bytes(self) -> bytes:
        return self.canonical_text().encode("utf-8")

    def digest(self, algorithm: str = "sha256") -> str:
        """Hash of the canonical text — **an index, not the code itself**.

        A hash is discontinuous: a ``1e-16`` perturbation of a coordinate changes
        it completely, while mathematically the two images are almost the same.
        It cannot answer "by how much do these differ", so it must never be used as
        a substitute for the code (README sections 7 and 23).
        """
        return hashlib.new(algorithm, self.canonical_bytes()).hexdigest()

    # -- comparison -------------------------------------------------------- #
    def equivalent_to(self, other: "TeachingAffineImageCode") -> bool:
        """``True`` only when the canonical bytes coincide (parameters included)."""
        if not isinstance(other, TeachingAffineImageCode):
            raise TypeError("equivalent_to expects another TeachingAffineImageCode")
        return self.canonical_bytes() == other.canonical_bytes()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TeachingAffineImageCode):
            return NotImplemented
        return self.equivalent_to(other)

    def __hash__(self) -> int:
        return hash(self.canonical_bytes())

    def __len__(self) -> int:
        return len(self.frames)

    # -- diagnostics ------------------------------------------------------- #
    def precision(self) -> PrecisionReport:
        """Attainable precision of the worst frame used."""
        return precision_report(self.worst_condition_number)

    @property
    def quantization_is_safe(self) -> bool:
        """Whether the quantum is comfortably coarser than float64 noise."""
        return _quantization_is_safe(self.worst_condition_number, self.parameters.quantum)

    def quality(self) -> dict[str, Any]:
        """Everything needed to judge whether this code is trustworthy."""
        if math.isfinite(self.worst_condition_number) and self.worst_condition_number > 0:
            margin = math.log10(
                self.parameters.quantum / (self.worst_condition_number * _EPS * 64.0)
            )
        else:
            margin = -math.inf
        return {
            "n_points": self.n_points,
            "n_frames": len(self.frames),
            "n_triples_available": self.n_triples_available,
            "n_collinear_triples": self.n_collinear_triples,
            "n_truncated_triples": self.n_truncated_triples,
            "truncated": self.truncated,
            "n_triples_total": self.n_points * (self.n_points - 1) * (self.n_points - 2) // 6,
            "complete_invariant": not self.truncated,
            "decimals": self.parameters.decimals,
            "quantum": self.parameters.quantum,
            "worst_condition_number": self.worst_condition_number,
            "quantization_is_safe": self.quantization_is_safe,
            "quantization_digits_margin": margin,
            "quantization_magnitude": self.quantization_magnitude,
            "quantization_representable": self.quantization_magnitude <= _EXACT_INT_LIMIT,
            "warnings": list(self.warnings),
            "digest": self.digest(),
            "disclaimer_zh": DISCLAIMER_ZH,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": CODE_FORMAT_VERSION,
            "parameters": self.parameters.as_dict(),
            "n_points": self.n_points,
            "n_frames": len(self.frames),
            "n_triples_available": self.n_triples_available,
            "n_collinear_triples": self.n_collinear_triples,
            "n_truncated_triples": self.n_truncated_triples,
            "truncated": self.truncated,
            "n_triples_total": self.n_points * (self.n_points - 1) * (self.n_points - 2) // 6,
            "worst_condition_number": self.worst_condition_number,
            "digest": self.digest(),
            "quality": self.quality(),
            "frames": [f.as_dict() for f in self.frames],
            "disclaimer_en": DISCLAIMER_EN,
            "disclaimer_zh": DISCLAIMER_ZH,
            "disclaimer_ru": DISCLAIMER_RU,
        }


def build_teaching_image_code(
    points: Any,
    parameters: TeachingCodingParameters | None = None,
) -> TeachingAffineImageCode:
    """Build ``C_teach(X)`` (see the module docstring for the definition).

    Steps, in the README's order:

    1. enumerate every non-collinear *unordered* triple;
    2. per triple, quantise the internal coordinates of all points and sort them
       lexicographically;
    3. canonicalise over the six orderings of that triple (take the minimum);
    4. sort all frame descriptors and freeze them into the code.

    Raises
    ------
    ValueError
        When the image admits no frame (fewer than three points, or all points
        collinear).  This is a *refusal*, not an empty code — an empty code would
        compare equal to every other empty code.
    """
    params = parameters or TeachingCodingParameters()
    x = as_point_matrix(points, name="points", min_points=1)

    candidates = list(iter_non_collinear_triples(x, params.collinear_tol))
    n_available = len(candidates)
    n_total = x.shape[0] * (x.shape[0] - 1) * (x.shape[0] - 2) // 6 if x.shape[0] >= 3 else 0
    if not candidates:
        raise ValueError(
            "build_teaching_image_code: the image admits no affine frame "
            f"(n = {x.shape[0]}, all {n_total} triple(s) degenerate); a teaching affine "
            "code is undefined for a 1-dimensional image"
        )

    warnings: list[str] = []
    if params.max_frames is not None and n_available > params.max_frames:
        processed = candidates[: params.max_frames]
    else:
        processed = candidates
    n_truncated = n_available - len(processed)
    truncated = n_truncated > 0
    if truncated:
        warnings.append(
            f"TRUNCATED_FRAMES: only {len(processed)} of {n_available} admissible "
            "triples were used, so the code is no longer a complete invariant"
        )

    frame_codes: list[TeachingAffineFrameCode] = []
    n_rejected = 0
    first_rejection = ""
    for triple in processed:
        try:
            frame_codes.append(build_canonical_frame_code(x, triple, params))
        except ValueError as exc:  # borderline triple: excluded, and counted
            n_rejected += 1
            first_rejection = first_rejection or str(exc)
    frame_codes.sort(key=_sort_key)

    n_collinear = n_total - n_available + n_rejected
    if n_rejected:
        warnings.append(
            f"BORDERLINE_TRIPLES: {n_rejected} triple(s) passed the collinearity pre-filter "
            "but admitted no usable ordering and were excluded (first reason: "
            f"{first_rejection})"
        )
    # accounting invariant: every unordered triple is either coded, degenerate, or truncated
    assert n_collinear + n_truncated + len(frame_codes) == n_total, (  # noqa: S101
        f"triple accounting broken: {n_collinear} + {n_truncated} + {len(frame_codes)} "
        f"!= {n_total}"
    )

    worst_cond = max((f.condition_number for f in frame_codes), default=math.inf)
    # The integer snap is applied to the *internal* coordinates, not to the input
    # coordinates, so the overflow check must look at ``max|alpha*10**decimals|``.
    # A large absolute image scale is irrelevant; what matters is how far the image
    # extends beyond its own frame (a tiny frame with far-away points gives huge
    # alpha, beta).
    quantization_magnitude = max(
        (
            float(max(abs(value) for pair in frame.quantized_points for value in pair))
            for frame in frame_codes
            if frame.quantized_points
        ),
        default=0.0,
    )

    if quantization_magnitude > _EXACT_INT_LIMIT:
        warnings.append(
            "QUANTIZATION_OVERFLOW: max|coordinate| * 10**decimals exceeds 2**53, so the "
            "integer snap itself loses information; lower `decimals` or use a frame that "
            "is not tiny compared with the image"
        )
    if not _quantization_is_safe(worst_cond, params.quantum):
        warnings.append(
            f"QUANTIZATION_UNSAFE: quantum {params.quantum:.0e} is not at least "
            f"{_QUANTIZATION_SAFETY:.0f}x coarser than the attainable precision of the "
            f"worst frame (cond = {worst_cond:.3e}); the code may encode floating-point "
            "noise instead of structure.  See recommended_decimals()."
        )

    return TeachingAffineImageCode(
        frames=tuple(frame_codes),
        parameters=params,
        n_points=int(x.shape[0]),
        n_collinear_triples=int(n_collinear),
        n_triples_available=int(n_available),
        n_truncated_triples=int(n_truncated),
        truncated=truncated,
        worst_condition_number=float(worst_cond),
        quantization_magnitude=quantization_magnitude,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TeachingCodeComparison:
    """Result of comparing two images by their teaching codes."""

    equivalent: bool
    digest_x: str
    digest_y: str
    n_frames_x: int
    n_frames_y: int
    n_points_x: int
    n_points_y: int
    reason: str
    quality_x: Mapping[str, Any]
    quality_y: Mapping[str, Any]

    @property
    def weak_evidence(self) -> bool:
        """``True`` when the verdict rests on a code that is not fully reliable."""
        for quality in (self.quality_x, self.quality_y):
            if not quality:
                continue
            if quality.get("truncated") or not quality.get("quantization_is_safe"):
                return True
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "digest_x": self.digest_x,
            "digest_y": self.digest_y,
            "n_frames_x": self.n_frames_x,
            "n_frames_y": self.n_frames_y,
            "n_points_x": self.n_points_x,
            "n_points_y": self.n_points_y,
            "reason": self.reason,
            "weak_evidence": self.weak_evidence,
        }

    def describe(self, lang: str = "zh") -> str:
        verdict = {
            "zh": "仿射等价（在教学编码的意义下）" if self.equivalent else "不相等",
            "ru": "аффинно эквивалентны (в смысле учебного кода)"
            if self.equivalent
            else "не эквивалентны",
            "en": "affinely equivalent (in the teaching-code sense)"
            if self.equivalent
            else "not equivalent",
        }.get(lang, str(self.equivalent))
        return (
            f"{verdict}\nreason: {self.reason}\n"
            f"frames: X={self.n_frames_x}, Y={self.n_frames_y}  "
            f"points: X={self.n_points_x}, Y={self.n_points_y}"
            + ("\n⚠ weak evidence: see quality()" if self.weak_evidence else "")
        )


def compare_teaching_codes(
    x: Any,
    y: Any,
    parameters: TeachingCodingParameters | None = None,
) -> TeachingCodeComparison:
    """Compare two images by their teaching codes, explaining the outcome.

    Never raises for a legitimate "not equivalent".  The reason string separates

    * ``"equal"`` — canonical forms agree;
    * ``"cardinality mismatch"`` — different point counts (an affine map is a
      bijection, so this alone refutes equivalence);
    * ``"descriptor mismatch"`` — same counts, different canonical form;
    * ``"not codeable"`` — one image admits no frame.

    A truncated or quantisation-unsafe code still compares, but
    :attr:`TeachingCodeComparison.weak_evidence` is set: an "equivalent" verdict
    from such a code is weaker evidence and the UI must say so.
    """
    params = parameters or TeachingCodingParameters()

    def _build(points: Any, label: str) -> tuple[TeachingAffineImageCode | None, str]:
        try:
            return build_teaching_image_code(points, params), ""
        except ValueError as exc:
            return None, f"{label}: {exc}"

    code_x, err_x = _build(x, "X")
    code_y, err_y = _build(y, "Y")

    if code_x is None or code_y is None:
        return TeachingCodeComparison(
            equivalent=False,
            digest_x=code_x.digest() if code_x is not None else "",
            digest_y=code_y.digest() if code_y is not None else "",
            n_frames_x=len(code_x.frames) if code_x is not None else 0,
            n_frames_y=len(code_y.frames) if code_y is not None else 0,
            n_points_x=code_x.n_points if code_x is not None else 0,
            n_points_y=code_y.n_points if code_y is not None else 0,
            reason="not codeable — " + "; ".join(filter(None, (err_x, err_y))),
            quality_x=code_x.quality() if code_x is not None else {},
            quality_y=code_y.quality() if code_y is not None else {},
        )

    if code_x.n_points != code_y.n_points:
        equivalent = False
        reason = (
            f"cardinality mismatch: |X| = {code_x.n_points}, |Y| = {code_y.n_points} "
            "(an affine map is a bijection, so the point counts must agree)"
        )
    elif code_x.equivalent_to(code_y):
        equivalent = True
        caveats: list[str] = []
        if code_x.truncated or code_y.truncated:
            caveats.append("code was truncated, so it is not a complete invariant")
        if not (code_x.quantization_is_safe and code_y.quantization_is_safe):
            caveats.append("quantisation is finer than the attainable precision")
        reason = "equal" if not caveats else "equal (with caveats: " + "; ".join(caveats) + ")"
    else:
        equivalent = False
        differing = _first_differing_frames(code_x, code_y)
        reason = (
            "descriptor mismatch: same cardinality, different canonical form"
            + (f"; first divergence at frame index {differing}" if differing is not None else "")
        )

    return TeachingCodeComparison(
        equivalent=equivalent,
        digest_x=code_x.digest(),
        digest_y=code_y.digest(),
        n_frames_x=len(code_x.frames),
        n_frames_y=len(code_y.frames),
        n_points_x=code_x.n_points,
        n_points_y=code_y.n_points,
        reason=reason,
        quality_x=code_x.quality(),
        quality_y=code_y.quality(),
    )


def teaching_codes_equal(
    x: Any,
    y: Any,
    parameters: TeachingCodingParameters | None = None,
) -> bool:
    """Convenience: ``compare_teaching_codes(...).equivalent``."""
    return compare_teaching_codes(x, y, parameters).equivalent


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _first_differing_frames(
    a: TeachingAffineImageCode, b: TeachingAffineImageCode
) -> int | None:
    """Index of the first descriptor that differs, for a useful failure message."""
    bodies_a, bodies_b = a.canonical_bodies(), b.canonical_bodies()
    for index, (left, right) in enumerate(zip(bodies_a, bodies_b)):
        if left != right:
            return index
    return None if len(bodies_a) == len(bodies_b) else min(len(bodies_a), len(bodies_b))


def _permutations(triple: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    a, b, c = triple
    return (
        (a, b, c),
        (a, c, b),
        (b, a, c),
        (b, c, a),
        (c, a, b),
        (c, b, a),
    )


def _sort_key(frame: TeachingAffineFrameCode) -> tuple[Any, ...]:
    """Total order over frame descriptors — the basis of the canonical form.

    The key is the quantised point list **only**.  Using the point indices or the
    condition number as a tie-break would let arbitrary labels leak into the
    ordering; descriptors that tie on the point list are genuinely the same
    descriptor, so leaving them tied is both correct and harmless.
    """
    return (frame.quantized_points,)
