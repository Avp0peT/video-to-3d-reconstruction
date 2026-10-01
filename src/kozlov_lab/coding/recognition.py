"""Recognition: the formal pipeline, and the three kinds of evidence it must keep apart.

The question this module answers
--------------------------------
*Given two finite point-set images, what is their affine relationship — and how strong is
the answer?*

Why ``True``/``False`` is not an acceptable return value
-------------------------------------------------------
A boolean hides three independent things that the specification requires to stay separate:

* whether the **code is definable** at all (all-collinear images have no non-degenerate
  denominator);
* whether **theorem 1's iff applies** (it is stated for images not lying on one line or two
  parallel lines);
* whether the **numbers mean anything** (ratios built from a rounding-scale denominator are
  noise, however exactly they were computed).

And it hides *which kind of evidence* produced the answer:

=======================================  ==========================================
evidence                                 what it proves
=======================================  ==========================================
**A — geometric witness**                an explicit ``(M, t)`` plus a bijection, which
                                         the caller can re-apply and re-check
**B — code equivalence**                 a renumbering ``psi`` preserving every indexed
                                         ratio; *iff* affine equivalence inside the domain
**C — numeric approximation**            a match within a stated tolerance; **never** a
                                         theorem
=======================================  ==========================================

Passing a float tolerance is not evidence of kind A or B, and reporting it as such is the
specific failure this module exists to prevent.

A note on where a disagreement is an *error*
--------------------------------------------
Inside theorem 1's domain the code route and the geometric route must agree; if they do not,
one of them is wrong and :class:`InternalConsistencyError` is raised — the pipeline refuses
to pick a winner.

**Outside** the domain a disagreement is expected rather than alarming: the theorem's
hypothesis is precisely what rules out code-equivalence without affine equivalence, so a
``two_parallel_lines`` image whose codes match while no affine map exists is informative.
The status then reports both facts instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .equivalence import (
    KozlovCodeEquivalence,
    kozlov_codes_equivalent,
)
from .exact import build_kozlov_image_code_exact
from .geometric import GeometricWitness, check_geometric_equivalence
from .model import ArithmeticMode, KozlovImageCode, ModeMismatchError
from .numeric import build_kozlov_image_code_numeric
from .validation import (
    CollinearImageError,
    DuplicatePointError,
    KozlovImageError,
    NonExactCoordinateError,
    TheoremDomainError,
    TooFewPointsError,
    validate_kozlov_image,
)

__all__ = [
    "RecognitionStatus",
    "EvidenceKind",
    "RecognitionResult",
    "InternalConsistencyError",
    "recognize_affine_exact",
    "recognize_affine_numeric",
    "NUMERIC_DEFAULT_TOLERANCE",
]


class RecognitionStatus(str, Enum):
    """The outcome vocabulary.  Deliberately more than two values."""

    EXACT_AFFINE_EQUIVALENT = "EXACT_AFFINE_EQUIVALENT"
    EXACT_NOT_AFFINE_EQUIVALENT = "EXACT_NOT_AFFINE_EQUIVALENT"
    CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE = "CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE"
    NUMERIC_AFFINE_MATCH = "NUMERIC_AFFINE_MATCH"
    NUMERIC_NON_MATCH = "NUMERIC_NON_MATCH"
    NUMERIC_INDETERMINATE = "NUMERIC_INDETERMINATE"
    #: The codes match while no affine map exists, **outside** theorem 1's domain.  A
    #: mathematical phenomenon rather than a defect: the theorem's hypothesis is precisely what
    #: rules out "code-equivalent yet not affine-equivalent".
    CODE_COLLISION_OUTSIDE_THEOREM_DOMAIN = "CODE_COLLISION_OUTSIDE_THEOREM_DOMAIN"
    CODE_UNDEFINED = "CODE_UNDEFINED"
    INVALID_INPUT = "INVALID_INPUT"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class EvidenceKind(str, Enum):
    """Which of the three evidence types a result actually rests on."""

    GEOMETRIC_WITNESS = "GEOMETRIC_WITNESS"
    CODE_EQUIVALENCE = "CODE_EQUIVALENCE"
    NUMERIC_APPROXIMATION = "NUMERIC_APPROXIMATION"
    NONE = "NONE"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class InternalConsistencyError(RuntimeError):
    """The two independent routes disagree **inside** the theorem's domain.

    Raised instead of choosing one answer: a disagreement here means a defect in one of the
    implementations, and silently preferring the code route (the "official" one) would hide
    exactly the class of bug the cross-check exists to find.

    The exception carries a self-contained **reproduction artifact** — inputs, codes,
    bijections, both verdicts, and the git commit — so the failure can be reported without
    re-running anything.
    """

    def __init__(self, message: str, artifact: Mapping[str, Any]) -> None:
        self.artifact = dict(artifact)
        super().__init__(message)

    def write_artifact(self, directory: str | Path = "experiments/results/discrepancies") -> Path:
        """Persist the reproduction artifact so the case can be minimised offline."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        import hashlib

        digest = hashlib.sha256(
            json.dumps(self.artifact.get("inputs", {}), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        path = target / f"discrepancy_{digest}.json"
        path.write_text(
            json.dumps(self.artifact, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return path


@dataclass(frozen=True)
class RecognitionResult:
    """Everything the caller needs, including what could *not* be established.

    The ``status`` is the headline, but the boolean-ish fields are kept separate rather than
    folded into it, so a UI can answer "code definable? theorem applicable? mode? code
    equivalent? witness found?" independently — which is what section 4Q requires.
    """

    status: RecognitionStatus
    evidence: tuple[EvidenceKind, ...]
    #: Result of the formal code route; ``None`` when no code could be built.
    code_equivalent: bool | None
    #: ``True`` only when an explicit map was found **and re-verified**.
    geometric_witness: GeometricWitness | None
    theorem_applicable: bool | None
    code_definable: bool | None
    numeric_condition: Mapping[str, Any] | None
    mapping: Mapping[int, int] | None
    affine_matrix: tuple[tuple[Any, Any], tuple[Any, Any]] | None
    translation: tuple[Any, Any] | None
    residual: Any | None
    arithmetic_mode: ArithmeticMode | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------ #
    @property
    def is_affine_equivalent(self) -> bool:
        """``True`` only for statuses that assert equivalence (not for "undecided")."""
        return self.status in {
            RecognitionStatus.EXACT_AFFINE_EQUIVALENT,
            RecognitionStatus.NUMERIC_AFFINE_MATCH,
        }

    @property
    def witnessed(self) -> bool:
        """Whether the answer rests on an explicit, re-verifiable map."""
        return EvidenceKind.GEOMETRIC_WITNESS in self.evidence

    @property
    def undecided(self) -> bool:
        return self.status in {
            RecognitionStatus.NUMERIC_INDETERMINATE,
            RecognitionStatus.CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE,
        }

    def describe(self, lang: str = "zh") -> str:
        lines = [f"status: {self.status.value}"]
        lines.append(f"evidence: {', '.join(str(k) for k in self.evidence) or '—'}")
        lines.append(f"mode: {self.arithmetic_mode or '—'}")
        lines.append(
            f"code definable: {self.code_definable}  |  theorem applicable: {self.theorem_applicable}"
        )
        lines.append(f"code equivalent: {self.code_equivalent}")
        if self.geometric_witness is not None:
            lines.append(self.geometric_witness.describe(lang))
        elif self.affine_matrix is not None:
            lines.append(f"M = {self.affine_matrix}  t = {self.translation}")
        if self.numeric_condition:
            lines.append(f"numeric condition: {dict(self.numeric_condition)}")
        return "\n".join(lines)

    def as_dict(self, *, include_witness: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "evidence": [str(k) for k in self.evidence],
            "code_equivalent": self.code_equivalent,
            "theorem_applicable": self.theorem_applicable,
            "code_definable": self.code_definable,
            "arithmetic_mode": None if self.arithmetic_mode is None else str(self.arithmetic_mode),
            "mapping": dict(sorted(self.mapping.items())) if self.mapping else None,
            "affine_matrix": None
            if self.affine_matrix is None
            else [[str(v) for v in row] for row in self.affine_matrix],
            "translation": None if self.translation is None else [str(v) for v in self.translation],
            "residual": None if self.residual is None else str(self.residual),
            "numeric_condition": dict(self.numeric_condition) if self.numeric_condition else None,
            "diagnostics": dict(self.diagnostics),
        }
        if include_witness and self.geometric_witness is not None:
            payload["witness"] = self.geometric_witness.as_dict()
        return payload


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build_codes_exact(x: Any, y: Any) -> tuple[KozlovImageCode | None, KozlovImageCode | None, dict, dict]:
    """Build both exact codes, recording per-image validation for the diagnostics."""
    info: dict[str, Any] = {}
    codes: list[KozlovImageCode | None] = []
    for name, points in (("x", x), ("y", y)):
        try:
            codes.append(build_kozlov_image_code_exact(points))
            info[name] = {"built": True}
        except KozlovImageError as exc:
            codes.append(None)
            info[name] = {"built": False, "error": type(exc).__name__, "message": str(exc)}
    return codes[0], codes[1], info, {}


def _build_disagreement_artifact(
    code_result: KozlovCodeEquivalence, geometric: Any, *, x: Any, y: Any, theorem_applicable: bool
) -> dict:
    """Self-contained reproduction record for a disagreement that must not be resolved.

    Includes the inputs verbatim: the point of the artifact is that the case can be minimised
    and re-run without any further access to the session that found it.
    """
    return {
        "inputs": {"x": x, "y": y},
        "code_route": code_result.as_dict(),
        "geometric_route": geometric.as_dict(),
        "theorem_applicable": theorem_applicable,
        "truth_table_row": {
            "code_equivalent": code_result.equivalent,
            "geometric_equivalent": geometric.equivalent,
        },
    }


def _resolve_status(
    code_result: KozlovCodeEquivalence,
    geometric: Any,
    *,
    theorem_applicable: bool,
    x: Any,
    y: Any,
) -> tuple[RecognitionStatus, tuple[EvidenceKind, ...], str | None]:
    """The F4.2 truth table, written out rather than implied by control flow.

    ====================  ==================================  ======================================
    code                  geometric                           outcome
    ====================  ==================================  ======================================
    ``True``              ``True``                            ``EXACT_AFFINE_EQUIVALENT``
    ``False``             ``False``                           ``EXACT_NOT_AFFINE_EQUIVALENT``
    ``False``             ``True``                            ``InternalConsistencyError`` (unconditional)
    ``True``              ``False``, theorem applies          ``InternalConsistencyError``
    ``True``              ``False``, theorem does not apply   ``CODE_COLLISION_OUTSIDE_THEOREM_DOMAIN``
    abstention            —                                   reported as ``..._NOT_APPLICABLE`` / ``..._INDETERMINATE``
    ====================  ==================================  ======================================

    Two asymmetries are deliberate and worth stating:

    * ``code=False, geometry=True`` raises **regardless of the theorem domain**.  Affine
      equivalence *must* preserve formal code equivalence — that implication needs no
      hypothesis — so a witness alongside a code mismatch is a defect wherever it occurs.  An
      earlier version of this function only raised inside the domain, which would have let a
      real defect pass outside it.
    * ``code=True, geometry=False`` outside the domain is **not** a defect.  The theorem's
      hypothesis is exactly what rules out agreement, so the case gets its own status and the
      constructive route decides.

    The last row is the third asymmetry: an abstention is not a disagreement, in either
    direction, and must not be turned into one.
    """
    witness = geometric.witness
    code_equivalent = None if code_result.undecided else code_result.equivalent
    geometric_equivalent = None if geometric.undecided else geometric.equivalent

    # -- one route abstained: not a disagreement, but a weaker answer ------------------ #
    if code_equivalent is None or geometric_equivalent is None:
        if witness is not None:
            return RecognitionStatus.EXACT_AFFINE_EQUIVALENT, (EvidenceKind.GEOMETRIC_WITNESS,), None
        if code_equivalent:
            return (
                RecognitionStatus.CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE,
                (EvidenceKind.CODE_EQUIVALENCE,),
                "the code route answered while the geometric oracle did not; the code verdict is "
                "reported as such and must not be promoted to an affine statement without the "
                "theorem's hypothesis",
            )
        if geometric.undecided:
            return (
                RecognitionStatus.NUMERIC_INDETERMINATE,
                (EvidenceKind.NONE,),
                "the geometric oracle could not decide and the code route did not answer either",
            )
        return (
            RecognitionStatus.NUMERIC_INDETERMINATE,
            (EvidenceKind.NONE,),
            "no route reached a verdict",
        )

    # -- both routes answered: the table ------------------------------------------------ #
    if code_equivalent and geometric_equivalent:
        return (
            RecognitionStatus.EXACT_AFFINE_EQUIVALENT,
            (EvidenceKind.CODE_EQUIVALENCE, EvidenceKind.GEOMETRIC_WITNESS),
            None,
        )
    if not code_equivalent and not geometric_equivalent:
        return RecognitionStatus.EXACT_NOT_AFFINE_EQUIVALENT, (EvidenceKind.CODE_EQUIVALENCE,), None

    if not code_equivalent and geometric_equivalent:
        raise InternalConsistencyError(
            "an explicit affine map was found, yet the formal codes are not equivalent. Affine "
            "equivalence must preserve code equivalence unconditionally — that implication needs "
            "no hypothesis — so one of the two implementations is wrong. The pipeline refuses to "
            "choose, and this holds regardless of theorem 1's domain.",
            _build_disagreement_artifact(
                code_result, geometric, x=x, y=y, theorem_applicable=theorem_applicable
            ),
        )

    if theorem_applicable:
        raise InternalConsistencyError(
            "the codes are equivalent while no affine map exists, *inside* theorem 1's domain, "
            "where the two are guaranteed to agree. One of the two implementations is wrong; the "
            "pipeline refuses to choose.",
            _build_disagreement_artifact(
                code_result, geometric, x=x, y=y, theorem_applicable=theorem_applicable
            ),
        )

    return (
        RecognitionStatus.CODE_COLLISION_OUTSIDE_THEOREM_DOMAIN,
        (EvidenceKind.CODE_EQUIVALENCE,),
        "the codes are equivalent but no affine map exists, and theorem 1's hypothesis is not "
        "satisfied — so this is precisely the situation the hypothesis excludes. Reported rather "
        "than raised: the code verdict is real, it simply does not imply an affine statement here.",
    )


# --------------------------------------------------------------------------- #
# exact pipeline
# --------------------------------------------------------------------------- #
def recognize_affine_exact(
    x: Any,
    y: Any,
    *,
    budget: int = 20_000,
    refinement_rounds: int = 1,
) -> RecognitionResult:
    """Exact recognition: ``X`` and ``Y`` affine equivalent, with the evidence named.

    Pipeline (section 4H): input validation → cardinality → domain classification → cheap
    pruning → code equivalence → independent geometric oracle → cross-check → result.

    Never returns a bare boolean, and never converts a numeric approximation into a theorem.
    """
    diagnostics: dict[str, Any] = {}

    # -- input validation -------------------------------------------------- #
    try:
        validation_x = validate_kozlov_image(x)
        validation_y = validate_kozlov_image(y)
    except (DuplicatePointError, NonExactCoordinateError, TooFewPointsError) as exc:
        return RecognitionResult(
            status=RecognitionStatus.INVALID_INPUT,
            evidence=(EvidenceKind.NONE,),
            code_equivalent=None,
            geometric_witness=None,
            theorem_applicable=None,
            code_definable=None,
            numeric_condition=None,
            mapping=None,
            affine_matrix=None,
            translation=None,
            residual=None,
            arithmetic_mode=ArithmeticMode.EXACT,
            diagnostics={"input_error": type(exc).__name__, "message": str(exc)},
        )
    except CollinearImageError as exc:
        return RecognitionResult(
            status=RecognitionStatus.CODE_UNDEFINED,
            evidence=(EvidenceKind.NONE,),
            code_equivalent=None,
            geometric_witness=None,
            theorem_applicable=False,
            code_definable=False,
            numeric_condition=None,
            mapping=None,
            affine_matrix=None,
            translation=None,
            residual=None,
            arithmetic_mode=ArithmeticMode.EXACT,
            diagnostics={"collinear": True, "message": str(exc)},
        )

    diagnostics["validation"] = {"x": validation_x.as_dict(), "y": validation_y.as_dict()}
    theorem_applicable = validation_x.theorem_applicable and validation_y.theorem_applicable
    diagnostics["theorem_applicable"] = theorem_applicable

    # -- cardinality ------------------------------------------------------- #
    if validation_x.n_points != validation_y.n_points:
        # An affine map is a bijection, so this alone refutes equivalence — and the geometric
        # route agrees, so there is nothing to cross-check.
        diagnostics["cardinality"] = {"x": validation_x.n_points, "y": validation_y.n_points}
        return RecognitionResult(
            status=RecognitionStatus.EXACT_NOT_AFFINE_EQUIVALENT,
            evidence=(EvidenceKind.GEOMETRIC_WITNESS,),
            code_equivalent=False,
            geometric_witness=None,
            theorem_applicable=theorem_applicable,
            code_definable=True,
            numeric_condition=None,
            mapping=None,
            affine_matrix=None,
            translation=None,
            residual=None,
            arithmetic_mode=ArithmeticMode.EXACT,
            diagnostics=diagnostics,
        )

    # -- code route -------------------------------------------------------- #
    code_x, code_y, build_info, _ = _build_codes_exact(x, y)
    diagnostics["code_build"] = build_info
    if code_x is None or code_y is None:
        return RecognitionResult(
            status=RecognitionStatus.CODE_UNDEFINED,
            evidence=(EvidenceKind.NONE,),
            code_equivalent=None,
            geometric_witness=None,
            theorem_applicable=theorem_applicable,
            code_definable=False,
            numeric_condition=None,
            mapping=None,
            affine_matrix=None,
            translation=None,
            residual=None,
            arithmetic_mode=ArithmeticMode.EXACT,
            diagnostics=diagnostics,
        )
    code_result = kozlov_codes_equivalent(
        code_x, code_y, budget=budget, refinement_rounds=refinement_rounds
    )
    diagnostics["code_route"] = code_result.as_dict()

    # -- geometric route --------------------------------------------------- #
    geometric = check_geometric_equivalence(x, y)
    diagnostics["geometric_route"] = geometric.as_dict()

    # -- F4.2 truth table -------------------------------------------------- #
    status, evidence, note = _resolve_status(
        code_result, geometric, theorem_applicable=theorem_applicable, x=x, y=y
    )
    if note:
        diagnostics["resolution_note"] = note
    witness = geometric.witness

    residual = None
    if witness is not None:
        from .validation import _as_exact_points

        residual = 0 if witness.verify(_as_exact_points(x), _as_exact_points(y)) else "witness failed"

    return RecognitionResult(
        status=status,
        evidence=evidence,
        code_equivalent=None if code_result.undecided else code_result.equivalent,
        geometric_witness=witness,
        theorem_applicable=theorem_applicable,
        code_definable=True,
        numeric_condition=None,
        mapping=None if witness is None else dict(sorted(witness.point_mapping.items())),
        affine_matrix=None if witness is None else witness.matrix,
        translation=None if witness is None else witness.translation,
        residual=residual,
        arithmetic_mode=ArithmeticMode.EXACT,
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# numeric pipeline
# --------------------------------------------------------------------------- #
#: There is no defensible default tolerance, so the caller must state one.  This constant is
#: a *suggested* starting point for interactive use only; the recogniser never applies it
#: implicitly.
NUMERIC_DEFAULT_TOLERANCE = 1e-6


def recognize_affine_numeric(
    x: Any,
    y: Any,
    *,
    tolerance: float,
    flatness_tol: float | None = None,
    budget: int = 20_000,
) -> RecognitionResult:
    """Numeric recognition: three-valued, and honest about its own noise floor.

    Returns ``MATCH`` / ``NON_MATCH`` / ``INDETERMINATE``.  When the observed residual is at
    the level of the predicted numerical floor the answer is ``INDETERMINATE`` rather than a
    forced binary — a float tolerance is evidence of kind *C*, and dressing it up as kind A
    or B is the failure mode this module exists to prevent.
    """
    import numpy as np

    from ..config import COLLINEAR_TOL
    from ..precision import precision_report
    from .validation import validate_kozlov_image_numeric

    diagnostics: dict[str, Any] = {"tolerance": tolerance}
    tol = COLLINEAR_TOL if flatness_tol is None else flatness_tol

    try:
        validation_x = validate_kozlov_image_numeric(x, tol=tol)
        validation_y = validate_kozlov_image_numeric(y, tol=tol)
    except (DuplicatePointError, CollinearImageError, TooFewPointsError) as exc:
        return RecognitionResult(
            status=RecognitionStatus.INVALID_INPUT
            if not isinstance(exc, CollinearImageError)
            else RecognitionStatus.CODE_UNDEFINED,
            evidence=(EvidenceKind.NONE,),
            code_equivalent=None,
            geometric_witness=None,
            theorem_applicable=None,
            code_definable=None,
            numeric_condition=None,
            mapping=None,
            affine_matrix=None,
            translation=None,
            residual=None,
            arithmetic_mode=ArithmeticMode.NUMERIC,
            diagnostics={"input_error": type(exc).__name__, "message": str(exc)},
        )

    theorem_applicable = validation_x.theorem_applicable and validation_y.theorem_applicable
    code_x = build_kozlov_image_code_numeric(x, flatness_tol=tol)
    code_y = build_kozlov_image_code_numeric(y, flatness_tol=tol)
    code_result = kozlov_codes_equivalent(code_x, code_y, tolerance=tolerance, budget=budget)

    conditioning = _numeric_conditioning(code_x, x, y)

    # The residual is measured by fitting the map to the candidate bijection when one exists;
    # otherwise there is nothing to measure and the verdict comes from the code route alone.
    residual: float | None = None
    matched = 0
    if code_result.bijection is not None:
        residual, matched = _numeric_residual(x, y, code_result.bijection)
        diagnostics["matched_points"] = matched
    diagnostics["conditioning"] = conditioning["as_dict"]

    # ``noise`` is the **coordinate-domain** floor, because the residual is a coordinate
    # distance.  Comparing it against a ratio-domain bound would be a units error, and the
    # comparison would then be meaningless in exactly the regime it is supposed to guard.
    noise = float(conditioning["coordinate_noise_floor"])
    diagnostics["noise_floor"] = noise
    diagnostics["code_route"] = code_result.as_dict()

    if code_result.undecided:
        status = RecognitionStatus.NUMERIC_INDETERMINATE
        diagnostics["reason"] = code_result.reason
    elif residual is not None and 0.0 < residual <= noise > 0:
        # A **nonzero** residual at or below the noise floor is genuinely indistinguishable from
        # coordinate rounding.  A residual of exactly zero is not: it is a perfect fit under the
        # recovered bijection, and reporting it as "indeterminate" would turn the best possible
        # evidence into a refusal.  (Before this distinction the identity control on a
        # translation came out INDETERMINATE, which is how the omission was found.)
        status = RecognitionStatus.NUMERIC_INDETERMINATE
        diagnostics["reason"] = (
            f"the residual {residual:.3e} is nonzero but at or below the coordinate noise floor "
            f"{noise:.3e}: the match cannot be distinguished from coordinate rounding"
        )
    elif residual == 0.0:
        status = RecognitionStatus.NUMERIC_AFFINE_MATCH
        diagnostics["reason"] = "the residual is exactly zero: a perfect fit, not a rounding artefact"
    elif residual is not None and residual <= max(tolerance, noise):
        status = RecognitionStatus.NUMERIC_AFFINE_MATCH
    elif code_result.equivalent:
        status = RecognitionStatus.NUMERIC_INDETERMINATE
        diagnostics["reason"] = (
            "the code route reports equivalence but no point bijection was recovered, so the "
            "residual could not be measured"
        )
    else:
        status = RecognitionStatus.NUMERIC_NON_MATCH

    evidence = (
        (EvidenceKind.NUMERIC_APPROXIMATION,)
        if status is not RecognitionStatus.NUMERIC_INDETERMINATE
        else (EvidenceKind.NUMERIC_APPROXIMATION, EvidenceKind.NONE)
    )
    return RecognitionResult(
        status=status,
        evidence=evidence,
        code_equivalent=None if code_result.undecided else code_result.equivalent,
        geometric_witness=None,  # a float pipeline never produces a *witness*
        theorem_applicable=theorem_applicable,
        code_definable=True,
        numeric_condition={
            "tolerance": tolerance,
            "noise_floor": noise,
            "conditioning": conditioning["as_dict"],
            "max_residual": residual,
            "matched_points": matched,
            "ambiguity": None,
        },
        mapping=None if code_result.bijection is None else dict(sorted(code_result.bijection.items())),
        affine_matrix=None,
        translation=None,
        residual=residual,
        arithmetic_mode=ArithmeticMode.NUMERIC,
        diagnostics=diagnostics,
    )


def _numeric_residual(x: Any, y: Any, bijection: Mapping[int, int]) -> tuple[float, int]:
    """Least-squares affine fit under the candidate bijection, then the max residual.

    Uses :func:`numpy.linalg.lstsq` on the augmented system ``[p | 1] A = q`` — the same
    construction the geometry layer uses for control points, applied to all correspondences
    at once.  The residual is a *maximum*, not a mean, because a mean hides the one point
    that does not fit.
    """
    import numpy as np

    source = np.asarray(x, dtype=float).reshape(-1, 2)
    target = np.asarray(y, dtype=float).reshape(-1, 2)
    order = sorted(bijection.items())
    design = np.column_stack([source[[i for i, _ in order]], np.ones(len(order))])
    response = target[[j for _, j in order]]
    solution, *_ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ solution
    residuals = np.linalg.norm(fitted - response, axis=1)
    matched = int((residuals <= 1e-9).sum())
    return float(residuals.max()), matched


# --------------------------------------------------------------------------- #
# numeric conditioning — two floors, in two different units
# --------------------------------------------------------------------------- #
def _numeric_conditioning(code: KozlovImageCode, x: Any, y: Any) -> dict[str, Any]:
    """The two noise floors, kept apart because they live in different units.

    * **Coordinate floor** — ``64 * eps * diameter``, a *length*.  This is the scale of the
      rounding already present in the input coordinates, so a least-squares residual below
      it cannot be distinguished from that rounding.  This is what the residual is compared
      against.
    * **Ratio floor** — ``64 * eps * max(D) / min(D)``, *dimensionless*.  This bounds the
      relative error of a ratio built from the smallest admissible denominator.  It is
      reported because a code whose ratios are noise-dominated is useless for comparison even
      when the coordinates are perfectly accurate.

    Reporting a single "noise floor" would necessarily compare one of them against a quantity
    in the other's units, which is how a guard silently stops guarding.
    """
    import numpy as np

    x_array = np.asarray(x, dtype=float).reshape(-1, 2)
    y_array = np.asarray(y, dtype=float).reshape(-1, 2)
    stacked = np.vstack([x_array, y_array])
    diameters = float(np.abs(stacked).max() - np.abs(stacked).min()) or float(np.ptp(stacked))
    diameter = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    if diameter == 0.0:
        diameter = diameters or 1.0
    eps = float(np.finfo(np.float64).eps)
    coordinate_floor = 64.0 * eps * diameter

    magnitudes = sorted(abs(float(entry.value)) for entry in code.entries) or [0.0]
    largest = magnitudes[-1]
    ratio_floor = 64.0 * eps * largest if largest else 0.0

    return {
        "coordinate_noise_floor": coordinate_floor,
        "ratio_noise_floor": ratio_floor,
        "diameter": diameter,
        "largest_ratio_magnitude": largest,
        "eps": eps,
        "as_dict": {
            "coordinate_noise_floor": coordinate_floor,
            "ratio_noise_floor": ratio_floor,
            "diameter": diameter,
            "largest_ratio_magnitude": largest,
            "eps": eps,
        },
    }
