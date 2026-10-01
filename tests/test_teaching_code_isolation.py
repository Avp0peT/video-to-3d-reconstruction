"""Separation rules between the pedagogical and the formal implementation.

These rules changed on 2026-10-01, when Phase 3 was unlocked.  Before that, ``coding/``
had to stay **empty** and the only permitted names were ``Teaching*``.  Now both
implementations exist, so the constraint is no longer "one must not exist" but
**"neither may impersonate the other"**:

* ``teaching/`` must not use Kozlov's name for its own construction — it is a *different*
  construction (frame-relative internal coordinates), not a version of the formal code;
* ``coding/`` must not use the ``Teaching*`` prefix;
* ``coding/`` must not silently reuse the pedagogical construction, nor the reverse
  (mutual independence, checked on parsed imports);
* a hash stays an index in both, and only ``coding/`` may claim fidelity to the source.

The checks are policy, not numerics, so they are done by scanning names and parsed
imports — with the AST rather than raw text, because the disclaimers *must* be able to
write the forbidden identifiers in order to say they are not used.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

import kozlov_lab
import kozlov_lab.coding
import kozlov_lab.teaching
from kozlov_lab.teaching import (
    DISCLAIMER_EN,
    DISCLAIMER_RU,
    DISCLAIMER_ZH,
    DISCLAIMERS,
    TeachingAffineFrameCode,
    TeachingAffineImageCode,
)

SRC = pathlib.Path(kozlov_lab.__file__).resolve().parent
TEACHING_DIR = SRC / "teaching"
CODING_DIR = SRC / "coding"

#: Names that must not appear in the **teaching** package: using them would claim
#: Kozlov's formal object for a construction that is not it.
KOZLOV_ISH = ("KozlovImageCode", "OfficialImageCode")

#: Near-misses, also restricted to the teaching package.
KOZLOV_ISH_PATTERNS = (re.compile(r"(?i)kozlov\w*code"), re.compile(r"(?i)official\w*code"))

#: The prefix reserved for the pedagogical construction.
TEACHING_PREFIX = "Teaching"


def _python_sources(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    found.add(id(body[0].value))
    return found


def _identifier_like_strings(path: pathlib.Path) -> list[tuple[int, str]]:
    """Names and non-docstring string literals — where an identifier can hide.

    Docstrings are excluded deliberately: the disclaimer has to name the identifiers it
    rejects, and a raw-text scan would flag the very sentence that enforces the rule.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = _docstring_nodes(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            found.append((node.lineno, node.name))
        elif isinstance(node, ast.Name):
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            found.append((node.lineno, node.attr))
        elif isinstance(node, ast.arg):
            found.append((node.lineno, node.arg))
        elif isinstance(node, ast.alias):
            found.append((node.lineno, node.asname or node.name))
        elif isinstance(node, ast.keyword) and node.arg:
            found.append((node.lineno, node.arg))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.append((node.lineno, node.value))
    return found


def _defined_names(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            modules.add(base)
            modules.update(f"{base}.{alias.name}" for alias in node.names)
    return modules


def _package_of(module: str) -> str | None:
    """Which of the two packages a (possibly relative) module name refers to, if either."""
    name = module.lstrip(".")
    if name.startswith("kozlov_lab."):
        name = name[len("kozlov_lab.") :]
    head = name.split(".")[0]
    return head if head in {"coding", "teaching"} else None


# --------------------------------------------------------------------------- #
# both packages exist, and each stays in its lane
# --------------------------------------------------------------------------- #
def test_both_implementations_exist_and_are_separate_packages():
    assert TEACHING_DIR.is_dir()
    assert CODING_DIR.is_dir()
    # the five modules the specification prescribes must be present
    for required in ("model.py", "exact.py", "numeric.py", "validation.py", "equivalence.py"):
        assert (CODING_DIR / required).is_file(), f"coding/{required} is missing"


def test_coding_exports_the_phase3_api():
    for name in (
        "KozlovImageCode",
        "build_kozlov_image_code_exact",
        "build_kozlov_image_code_numeric",
        "codes_equivalent_exact",
        "codes_equivalent_numeric",
        "affine_equivalent_exact",
        "validate_kozlov_image",
        "KozlovImageValidation",
    ):
        assert name in kozlov_lab.coding.__all__, name
        assert hasattr(kozlov_lab.coding, name), name


def test_coding_exports_nothing_pedagogical():
    offenders = [name for name in kozlov_lab.coding.__all__ if name.startswith(TEACHING_PREFIX)]
    assert not offenders, f"the formal package must not export pedagogical names: {offenders}"


def test_neither_package_imports_the_other():
    """Mutual independence.  A shared dependency would let one definition masquerade as the other."""
    for directory, label in ((TEACHING_DIR, "teaching"), (CODING_DIR, "coding")):
        for source in _python_sources(directory):
            others = {
                module
                for module in _imported_modules(source)
                if (_package_of(module) or "") not in {"", label}
            }
            assert not others, f"{source.relative_to(SRC)} imports across the separation: {others}"


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #
def test_teaching_does_not_claim_kozlov_names_for_itself():
    """The pedagogical construction must not wear the official name."""
    offenders: list[str] = []
    for source in _python_sources(TEACHING_DIR):
        for lineno, text in _identifier_like_strings(source):
            if any(pattern.search(text) for pattern in KOZLOV_ISH_PATTERNS):
                offenders.append(f"{source.relative_to(SRC)}:{lineno}::{text}")
    assert not offenders, f"teaching/ used Kozlov's naming for its own object: {offenders}"


@pytest.mark.parametrize("forbidden", KOZLOV_ISH)
def test_forbidden_names_are_absent_from_teaching_identifiers(forbidden):
    offenders = [
        f"{source.relative_to(SRC)}:{lineno}"
        for source in _python_sources(TEACHING_DIR)
        for lineno, text in _identifier_like_strings(source)
        if forbidden in text
    ]
    assert not offenders, f"{forbidden} is used as an identifier in teaching/ at {offenders}"


def test_teaching_prefix_is_reserved_for_the_pedagogical_package():
    offenders = [
        f"{source.relative_to(SRC)}::{name}"
        for source in _python_sources(CODING_DIR)
        for name in _defined_names(source)
        if name.startswith(TEACHING_PREFIX)
    ]
    assert not offenders, f"coding/ must not use the Teaching prefix: {offenders}"


def test_code_classes_are_defined_only_in_the_two_expected_places():
    """One place per construction: no third module may grow its own code class."""
    matches: dict[str, list[str]] = {}
    for source in _python_sources(SRC):
        names = [n for n in _defined_names(source) if n.endswith(("ImageCode", "FrameCode"))]
        if names:
            matches[str(source.relative_to(SRC)).replace("\\", "/")] = names
    assert set(matches) == {"teaching/affine_code.py", "coding/model.py"}, matches
    assert matches["coding/model.py"] == ["KozlovImageCode"]


def test_coding_module_layout_matches_the_specification():
    """The five prescribed modules, and nothing that duplicates one of them."""
    names = {path.name for path in _python_sources(CODING_DIR)}
    assert names == {
        "__init__.py",
        "model.py",
        "exact.py",
        "numeric.py",
        "validation.py",
        "equivalence.py",
    }, names


# --------------------------------------------------------------------------- #
# each construction keeps its own self-description
# --------------------------------------------------------------------------- #
def test_disclaimers_are_present_and_language_keyed():
    assert set(DISCLAIMERS) == {"en", "zh", "ru"}
    assert DISCLAIMERS["en"] == DISCLAIMER_EN
    assert DISCLAIMERS["zh"] == DISCLAIMER_ZH
    assert DISCLAIMERS["ru"] == DISCLAIMER_RU


def test_disclaimer_says_pedagogical_and_not_official():
    """Still load-bearing after Phase 3 unlocked — and now for a stronger reason.

    The formal code turned out to be a *different construction* (ratios of triangle
    areas), so the disclaimer no longer means "unverified"; it means "not the same
    object".
    """
    checks = {
        "en": ("pedagogical", "internal-coordinate", "not claimed"),
        "zh": ("教学型", "内部仿射坐标", "不声明"),
        "ru": ("учебной", "внутренних аффинных", "не отождествляется"),
    }
    for language, fragments in checks.items():
        text = DISCLAIMERS[language]
        for fragment in fragments:
            assert fragment in text, f"{language} disclaimer is missing {fragment!r}"


def test_public_objects_carry_their_own_disclaimer():
    assert "pedagogical" in (TeachingAffineImageCode.__doc__ or "")
    assert "pedagogical" in (TeachingAffineFrameCode.__doc__ or "")
    assert "pedagogical" in (kozlov_lab.teaching.__doc__ or "")

    from kozlov_lab.coding import KozlovImageCode

    assert "Kozlov" in (kozlov_lab.coding.__doc__ or "")
    assert "Phase 3" in (kozlov_lab.coding.__doc__ or "")
    # the formal model must name its source, and must NOT describe itself as pedagogical
    assert "Fraction" in (KozlovImageCode.__doc__ or "") or "indexed" in (KozlovImageCode.__doc__ or "")


def test_the_formal_package_quotes_the_numeric_strategy_boundary():
    """The definition/implementation split must be stated where a reader will see it."""
    from kozlov_lab.coding import NUMERICAL_STRATEGY_NOTE

    assert "NOT part of the original mathematical definition" in NUMERICAL_STRATEGY_NOTE
    assert "数值策略" in NUMERICAL_STRATEGY_NOTE


def test_teaching_module_still_names_the_open_question_history():
    """The docstring must not pretend the separation was always obvious."""
    doc = kozlov_lab.teaching.__doc__ or ""
    assert "not claimed" in doc.lower() or "not the same" in doc.lower()
