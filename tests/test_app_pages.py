"""Headless smoke tests for the Streamlit app (Sprint 3A).

Uses Streamlit's own ``AppTest`` harness, which executes each page in-process and
surfaces any exception as ``at.exception``.  That turns "the page is written" into
"the page runs", which is the only claim worth making about UI code without a
browser.

Beyond "does it render", the assertions here check the presentation rules that the
task spec fixed explicitly:

* no page may show a bare PASS / FAIL where a measurement is required
  (M3 must expose the observed deviation *and* the attainable precision);
* the pedagogical code page must render its disclaimer;
* the invariant page must render the discrepancy machinery and the control section.

These are smoke tests: they do not check layout, and they make no claim about
visual quality.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"
SRC_DIR = APP_DIR.parent / "src"

# Streamlit's AppTest imports the script by path; the script itself bootstraps
# sys.path, but the *test* process needs the package importable too.
for _path in (SRC_DIR, APP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = {
    "home": APP_DIR / "Home.py",
    "m1_point_set": APP_DIR / "pages" / "01_Point_Set.py",
    "m2_transformations": APP_DIR / "pages" / "02_Transformations.py",
    "m3_affine_coordinates": APP_DIR / "pages" / "03_Affine_Coordinates.py",
    "m4_teaching_code": APP_DIR / "pages" / "04_Teaching_Code.py",
    "m6_invariants": APP_DIR / "pages" / "06_Invariants.py",
}

TIMEOUT = 60


def _run(key: str) -> AppTest:
    app = AppTest.from_file(str(PAGES[key]), default_timeout=TIMEOUT)
    app.run()
    return app


def _all_text(app: AppTest) -> str:
    """Everything the page rendered, flattened into one searchable string.

    Dataframe *content* is included on purpose.  Most of the substantive numbers in
    these pages live in ``st.dataframe`` tables, so an extractor that only looked at
    markdown would make the content assertions below pass vacuously.
    """
    chunks: list[str] = []

    for name in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "code",
        "text",
        "latex",
        "info",
        "warning",
        "error",
        "success",
        "exception",
    ):
        for element in app.get(name):
            value = getattr(element, "value", None)
            if isinstance(value, str):
                chunks.append(value)

    for element in app.get("metric"):
        for attribute in ("label", "value", "delta"):
            value = getattr(element, attribute, None)
            if value is not None:
                chunks.append(str(value))

    for element in app.get("dataframe"):
        value = getattr(element, "value", None)
        if value is None:
            continue
        columns = getattr(value, "columns", None)
        if columns is not None:
            chunks.append(" ".join(str(column) for column in columns))
        try:
            chunks.append(value.to_string())
        except (AttributeError, TypeError):  # pragma: no cover - non-pandas payload
            chunks.append(str(value))

    # widget labels/options: the controls are part of what the page "says"
    for name in (
        "button",
        "selectbox",
        "radio",
        "slider",
        "number_input",
        "checkbox",
        "multiselect",
        "download_button",
        "expander",
        "file_uploader",
    ):
        for element in app.get(name):
            for attribute in ("label", "help"):
                value = getattr(element, attribute, None)
                if isinstance(value, str):
                    chunks.append(value)
            options = getattr(element, "options", None)
            if options is not None and not isinstance(options, str):
                try:
                    chunks.append(" ".join(str(option) for option in options))
                except TypeError:  # pragma: no cover - defensive
                    pass

    return "\n".join(chunks)


# --------------------------------------------------------------------------- #
# every page renders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(PAGES))
def test_page_renders_without_exception(key):
    app = _run(key)
    assert not app.exception, f"{PAGES[key].name} raised: {app.exception}"
    assert len(app.get("markdown")) + len(app.get("code")) > 0, "page rendered nothing"


@pytest.mark.parametrize("key", sorted(PAGES))
def test_page_has_a_title_and_a_footer(key):
    app = _run(key)
    text = _all_text(app)
    assert "kozlov" in text.lower(), f"{PAGES[key].name} does not name the project"
    assert "README" in text or "第 " in text, f"{PAGES[key].name} lacks a specification reference"


# --------------------------------------------------------------------------- #
# M1 — the set-semantics demonstration must actually be on the page
# --------------------------------------------------------------------------- #
def test_m1_states_that_storage_order_is_irrelevant():
    text = _all_text(_run("m1_point_set"))
    assert "顺序" in text
    assert "集合" in text
    # the mandatory sentence from the task specification
    assert "点列表的存储顺序不应该改变点集本身的数学意义" in text


def test_m1_offers_the_required_shapes():
    """M1 must offer triangle, quadrilateral, pentagon, star and a random set."""
    from kozlov_lab.datasets import shapes

    available = set(shapes.shape_names())
    required = {"triangle", "pentagon", "star5", "random_8"}
    missing = required - available
    assert not missing, f"the shape registry is missing {missing}"
    # a quadrilateral is present under a different name
    assert any("quad" in name or "parallelogram" in name or "square" in name for name in available)
    # and M1 renders the registry selector
    assert "内置图形" in _all_text(_run("m1_point_set"))


def test_m1_reports_the_measured_set_invariants():
    text = _all_text(_run("m1_point_set"))
    assert "signature" in text
    assert "multiset_signature" in text
    assert "sorted_points" in text


# --------------------------------------------------------------------------- #
# M2 — real measured values, not a preserved/not-preserved label
# --------------------------------------------------------------------------- #
def test_m2_shows_determinant_condition_and_flatness():
    text = _all_text(_run("m2_transformations"))
    for token in ("det A", "cond", "flatness", "理论预测", "一致"):
        assert token in text, f"M2 does not show {token!r}"


def test_m2_shows_the_measured_error_columns():
    text = _all_text(_run("m2_transformations"))
    for token in ("max|Δ|", "容差", "见证"):
        assert token in text, f"M2 is missing the {token!r} column"


def test_m2_shows_distance_and_angle_error_not_only_a_verdict():
    """The spec: 不要只显示「保持 / 不保持」.  Real numbers are required."""
    text = _all_text(_run("m2_transformations"))
    assert "真实测量值" in text
    assert "距离" in text and "角度" in text
    # and the numbers are formatted as floats, not as words
    assert "e-" in text or "e+" in text


# --------------------------------------------------------------------------- #
# M3 — the required five-part table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "token",
    [
        "外部 (x, y)",
        "外部 T(x, y)",
        "内部 (α, β)",
        "内部 (α′, β′)",
        "Δα",
        "Δβ",
    ],
)
def test_m3_shows_the_five_required_columns(token):
    text = _all_text(_run("m3_affine_coordinates"))
    assert token in text, f"M3 is missing the column {token!r}"


def test_m3_shows_observed_error_alongside_predicted_precision():
    """The spec forbids a bare PASS / FAIL: both numbers must appear."""
    text = _all_text(_run("m3_affine_coordinates"))
    assert "实测偏差" in text or "实测 max|" in text
    assert "Predicted attainable precision" in text
    assert "可达精度" in text
    assert "cond 项" in text
    assert "量级项" in text


def test_m3_does_not_claim_an_unconditional_1e9():
    """The page must qualify the 1e-9 requirement, not assert it blindly."""
    text = _all_text(_run("m3_affine_coordinates"))
    assert "1e-9" in text
    assert "良态" in text, "M3 must state that 1e-9 holds for well-conditioned frames"


# --------------------------------------------------------------------------- #
# M4 — the pedagogical extension and its disclaimer
# --------------------------------------------------------------------------- #
def test_m4_renders_the_three_language_disclaimer():
    text = _all_text(_run("m4_teaching_code"))
    assert "不声明" in text, "the Chinese disclaimer must be rendered"
    assert "pedagogical" in text.lower()
    assert "первоисточник" in text, "the Russian disclaimer must be rendered"
    assert "不是 Козлов 教材的正式图像代码" in text


def test_m4_states_that_the_hash_is_only_an_index():
    text = _all_text(_run("m4_teaching_code"))
    assert "哈希不是数学图像代码本身" in text


def test_m4_shows_quality_before_the_conclusion():
    text = _all_text(_run("m4_teaching_code"))
    assert "编码质量" in text
    assert "量化是否安全" in text
    assert "推荐精度" in text, "the page must derive the quantisation, not hard-code it"


# --------------------------------------------------------------------------- #
# M6 — theory, measurement, error and control, plus the discrepancy banner
# --------------------------------------------------------------------------- #
def test_m6_shows_theory_measurement_and_error():
    text = _all_text(_run("m6_invariants"))
    assert "理论预测" in text
    assert "实测" in text
    assert "max|Δ|" in text
    assert "容差" in text


def test_m6_has_the_control_experiment_section():
    text = _all_text(_run("m6_invariants"))
    assert "对照实验" in text
    assert "无关图像" in text or "无关点集" in text


def test_m6_explains_the_vacuous_state():
    text = _all_text(_run("m6_invariants"))
    assert "vacuous" in text
    assert "无见证" in text


def test_the_discrepancy_banner_actually_renders_when_there_is_one():
    """Render the banner, do not merely assert its text exists somewhere.

    The M6 page only shows the banner when theory and measurement disagree — which
    is the correct behaviour, and it means a test that looks for the literal string
    in the *rendered page* would be asserting that a bug exists.  Instead the shared
    helper is rendered directly, on a deliberately inconsistent report, through a
    throwaway Streamlit script.
    """
    from kozlov_lab.datasets import shapes
    from kozlov_lab.geometry import (
        AffineTransform,
        check_against_theory,
        report_for_transform,
    )

    x = shapes.square(2.0, n_per_side=3).as_array()
    _, report = report_for_transform(x, AffineTransform.shear(0.8, 0.0))
    consistent = report.theory_check(AffineTransform.shear(0.8, 0.0))
    assert consistent.banner() is None, "a consistent measurement must not raise a banner"

    # claim a similarity: the table then requires the angles to be preserved
    tampered = report.__class__(
        transform_kind="similarity",
        invariant_group=report.invariant_group,
        n_points=report.n_points,
        entries=report.entries,
    )
    tampered_check = check_against_theory(tampered, transform=AffineTransform.similarity(2.5, 41.0))
    assert tampered_check.banner() is not None, "a real inconsistency must raise the banner"
    assert tampered_check.banner().startswith("NUMERICAL / IMPLEMENTATION DISCREPANCY")

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "import _shared as ui\n"
        "from kozlov_lab.geometry import AffineTransform, check_against_theory, report_for_transform\n"
        "from kozlov_lab.datasets import shapes\n"
        "x = shapes.square(2.0, n_per_side=3).as_array()\n"
        "_, report = report_for_transform(x, AffineTransform.shear(0.8, 0.0))\n"
        "tampered = report.__class__(transform_kind='similarity',\n"
        "    invariant_group=report.invariant_group, n_points=report.n_points,\n"
        "    entries=report.entries)\n"
        "ui.discrepancy_banner(check_against_theory(tampered,\n"
        "    transform=AffineTransform.similarity(2.5, 41.0)))\n"
    )
    app = AppTest.from_string(script, default_timeout=TIMEOUT)
    app.run()
    assert not app.exception, app.exception
    rendered = "\n".join(str(element.value) for element in app.error)
    assert "NUMERICAL / IMPLEMENTATION DISCREPANCY" in rendered
    assert "property = " in rendered and "severity=" in rendered

    # and the consistency path renders a success message instead
    ok_script = (
        "import sys\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "import _shared as ui\n"
        "from kozlov_lab.geometry import AffineTransform, report_for_transform\n"
        "from kozlov_lab.datasets import shapes\n"
        "x = shapes.square(2.0, n_per_side=3).as_array()\n"
        "t = AffineTransform.shear(0.8, 0.0)\n"
        "_, report = report_for_transform(x, t)\n"
        "ui.discrepancy_banner(report.theory_check(t))\n"
    )
    ok_app = AppTest.from_string(ok_script, default_timeout=TIMEOUT)
    ok_app.run()
    assert not ok_app.exception, ok_app.exception
    assert len(ok_app.success) == 1
    assert len(ok_app.error) == 0


def test_m6_page_wires_the_discrepancy_banner():
    """The M6 page must call the banner helper — a source-level wiring check."""
    source = PAGES["m6_invariants"].read_text(encoding="utf-8")
    assert "discrepancy_banner" in source
    assert "theory_check" in source
    shared = (APP_DIR / "_shared.py").read_text(encoding="utf-8")
    assert "banner()" in shared and "st.error" in shared


# --------------------------------------------------------------------------- #
# the UI must not re-implement mathematics
# --------------------------------------------------------------------------- #
def test_pages_do_not_import_numpy_linalg_for_math():
    """The architecture rule: all numerics come from ``kozlov_lab``.

    A page may use numpy for array plumbing, but not for the mathematics the math
    layer already owns.  ``np.linalg`` in a page is the signature of a
    re-implementation, so it is rejected outright.
    """
    offenders = []
    for path in sorted(APP_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "np.linalg" in source or "numpy.linalg" in source:
            offenders.append(path.name)
    assert not offenders, f"pages must not call numpy.linalg: {offenders}"


def test_pages_only_import_the_public_math_api():
    """Pages must not reach into private helpers of the math layer."""
    offenders = []
    for path in sorted(APP_DIR.rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and "kozlov_lab" in stripped:
                if "._" in stripped:
                    offenders.append(f"{path.name}: {stripped}")
    assert not offenders, f"pages touched private math helpers: {offenders}"
