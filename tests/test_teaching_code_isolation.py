"""Enforces the separation rules of Sprint 3C.

The instruction was explicit:

* the pedagogical encoding lives in ``src/kozlov_lab/teaching/``;
* it must **not** live in ``src/kozlov_lab/coding/`` until Q1 is resolved;
* the class names are ``TeachingAffineFrameCode`` / ``TeachingAffineImageCode``;
* ``KozlovImageCode`` and ``OfficialImageCode`` must not be used;
* every public surface carries the three-language disclaimer.

These are *policy* constraints, not numerical ones, so they are checked by
scanning names and source text rather than by exercising the algorithm.  Without
this file the constraints would decay the first time someone refactors.
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

#: Names that may not appear anywhere in the source tree.
FORBIDDEN_NAMES = ("KozlovImageCode", "OfficialImageCode")

#: Regexes that catch near-misses (e.g. ``KozlovCode``, ``official_image_code``).
FORBIDDEN_PATTERNS = (
    re.compile(r"(?i)kozlov\w*code"),
    re.compile(r"(?i)official\w*code"),
)


def _python_sources(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _defined_names(path: pathlib.Path) -> list[str]:
    """Class and function names defined in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return names


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """``id()`` of every docstring ``Constant`` node (module, class, function)."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    found.add(id(body[0].value))
    return found


def _identifier_like_strings(path: pathlib.Path) -> list[tuple[int, str]]:
    """Names and non-docstring string literals — the places a *name* can hide.

    Docstrings are excluded on purpose: the disclaimer legitimately writes the
    forbidden names in order to say they are **not** used, and a naive raw-text
    scan would flag the very sentence that enforces the rule.
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


def _imported_modules(path: pathlib.Path) -> list[str]:
    """Every module name an ``import`` statement refers to, including ``from x import y``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            modules.append(base)
            modules.extend(f"{base}.{alias.name}" for alias in node.names)
    return modules


# --------------------------------------------------------------------------- #
# location
# --------------------------------------------------------------------------- #
def test_pedagogical_code_lives_in_teaching_not_coding():
    """The module of record must be ``teaching``, and ``coding`` must stay empty."""
    assert TEACHING_DIR.is_dir(), "src/kozlov_lab/teaching/ must exist"
    assert CODING_DIR.is_dir(), "src/kozlov_lab/coding/ must exist (as an empty placeholder)"

    coding_definitions = [
        name for source in _python_sources(CODING_DIR) for name in _defined_names(source)
    ]
    assert not coding_definitions, (
        "src/kozlov_lab/coding/ must define nothing until Q1 is resolved, found: "
        f"{coding_definitions}"
    )

    teaching_definitions = [
        name for source in _python_sources(TEACHING_DIR) for name in _defined_names(source)
    ]
    assert "TeachingAffineFrameCode" in teaching_definitions
    assert "TeachingAffineImageCode" in teaching_definitions


def test_coding_public_api_is_empty():
    """Importing the placeholder must not hand out an implementation."""
    public = [name for name in dir(kozlov_lab.coding) if not name.startswith("_")]
    public = [name for name in public if name not in {"annotations"}]
    assert public == [], f"kozlov_lab.coding unexpectedly exports {public}"

    with pytest.raises(ImportError):
        from kozlov_lab.coding import build_teaching_image_code  # noqa: F401


def test_coding_docstring_states_the_block():
    """The placeholder must explain *why* it is empty, not just be empty."""
    doc = kozlov_lab.coding.__doc__ or ""
    assert "Q1" in doc
    assert "not implemented" in doc.lower() or "未实现" in doc
    assert "gate" in doc.lower() or "阻塞" in doc


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
def test_forbidden_names_are_absent_from_every_identifier(forbidden):
    """The forbidden names must not be used as names or in non-docstring strings.

    Scoped to identifiers and live string literals rather than raw text, because
    the disclaimer has to be able to say "this is **not** ``KozlovImageCode``".
    """
    offenders = [
        f"{source.relative_to(SRC)}:{lineno}"
        for source in _python_sources(SRC)
        for lineno, text in _identifier_like_strings(source)
        if forbidden in text
    ]
    assert not offenders, f"{forbidden} is used at {offenders}"


def test_no_near_miss_names_outside_the_disclaimer():
    """``KozlovCode``-style names must not be used as identifiers."""
    offenders: list[str] = []
    for source in _python_sources(SRC):
        for lineno, text in _identifier_like_strings(source):
            if any(pattern.search(text) for pattern in FORBIDDEN_PATTERNS):
                offenders.append(f"{source.relative_to(SRC)}:{lineno}::{text}")
    assert not offenders, f"forbidden identifier style: {offenders}"


def test_only_teaching_defines_an_image_code_class():
    """No other package may grow its own image-code class."""
    matches: dict[str, list[str]] = {}
    for source in _python_sources(SRC):
        names = [n for n in _defined_names(source) if n.endswith(("ImageCode", "FrameCode"))]
        if names:
            matches[str(source.relative_to(SRC)).replace("\\", "/")] = names
    assert list(matches) == ["teaching/affine_code.py"], (
        f"ImageCode/FrameCode classes must live only in teaching/affine_code.py: {matches}"
    )


# --------------------------------------------------------------------------- #
# disclaimer
# --------------------------------------------------------------------------- #
def test_disclaimers_are_present_and_language_keyed():
    assert set(DISCLAIMERS) == {"en", "zh", "ru"}
    assert DISCLAIMERS["en"] == DISCLAIMER_EN
    assert DISCLAIMERS["zh"] == DISCLAIMER_ZH
    assert DISCLAIMERS["ru"] == DISCLAIMER_RU


def test_disclaimer_says_pedagogical_and_not_official():
    """Each language must carry the two load-bearing clauses.

    Not an exact-string match (so wording may be improved later), but the two
    claims that protect the project from misrepresentation must survive any edit:
    (a) the construction is pedagogical; (b) it is **not** claimed to be the
    textbook's formal code unless verified against a primary source.
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


def test_public_objects_carry_the_disclaimer():
    """A caller must not be able to get the code without the disclaimer nearby."""
    assert "pedagogical" in (TeachingAffineImageCode.__doc__ or "")
    assert "pedagogical" in (TeachingAffineFrameCode.__doc__ or "")
    assert "pedagogical" in (kozlov_lab.teaching.__doc__ or "")

    from kozlov_lab.datasets.corpus import DATASETS
    from kozlov_lab.teaching import build_teaching_image_code

    payload = build_teaching_image_code(DATASETS["A"].points).as_dict()
    for key in ("disclaimer_en", "disclaimer_zh", "disclaimer_ru"):
        assert payload[key] == DISCLAIMERS[key.split("_")[1]]
    assert payload["quality"]["disclaimer_zh"] == DISCLAIMER_ZH


def test_teaching_module_docstring_forbids_the_misreading():
    """The module docstring must state what the code is *not*."""
    doc = kozlov_lab.teaching.__doc__ or ""
    assert "not" in doc.lower()
    assert "Kozlov" in doc  # named explicitly; e.g. "not claimed to be Kozlov's formal image code"
    assert "Q1" in doc, "the open question that blocks the official implementation must be named"


def test_teaching_does_not_import_the_coding_placeholder():
    """A dependency the other way round would defeat the isolation.

    Checked on the parsed import statements, not on raw text: the module docstring
    mentions ``kozlov_lab.coding`` precisely in order to say it is *not* this
    package.
    """
    for source in _python_sources(TEACHING_DIR):
        modules = _imported_modules(source)
        offenders = [
            module
            for module in modules
            if "coding" in module.split(".") or module.startswith("..coding")
        ]
        assert not offenders, f"{source.relative_to(SRC)} imports {offenders}"


def test_teaching_imports_are_within_the_math_layer():
    """The pedagogical code must depend on the math layer, not on the UI or the app."""
    forbidden_prefixes = ("streamlit", "plotly", "app", "experiments")
    for source in _python_sources(TEACHING_DIR):
        for module in _imported_modules(source):
            root = module.lstrip(".").split(".")[0]
            assert root not in forbidden_prefixes, (
                f"{source.relative_to(SRC)} must not import {module!r}: the math layer has to "
                "stay independent of the visualisation layer"
            )
