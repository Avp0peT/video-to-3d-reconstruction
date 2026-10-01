"""``kozlov_lab.teaching`` — **pedagogical** affine-invariant coding.

Isolation rule (enforced by ``tests/test_teaching_code_isolation.py``)
---------------------------------------------------------------------
This package is deliberately **not** ``kozlov_lab.coding``.  The project README
(section 16/23/26) forbids implementing a "textbook algorithm" that has not been
verified against a primary source, and open question **Q1** — the exact
definition of Kozlov's formal image code — is still unresolved.

Therefore:

* nothing in this package is named ``KozlovImageCode`` or ``OfficialImageCode``;
* every public object carries the three-language disclaimer below;
* ``kozlov_lab.coding`` stays empty until Q1 is resolved.

Three-language disclaimer (verbatim, must be shown next to any output)
---------------------------------------------------------------------
EN: This is a pedagogical affine-invariant construction derived from the
    internal-coordinate idea. It is not claimed to be Kozlov's formal image code
    unless verified against a primary source.
ZH: 本编码器是根据内部仿射坐标思想构造的教学型仿射不变编码，不声明其等同于
    Козлов 教材中的正式图像代码。
RU: Данный код является учебной конструкцией на основе внутренних аффинных
    координат и не отождествляется с формальным кодом изображения Козлова
    без проверки по первоисточнику.
"""

from __future__ import annotations

from .affine_code import (
    CODE_FORMAT_VERSION,
    DISCLAIMER_EN,
    DISCLAIMER_RU,
    DISCLAIMER_ZH,
    DISCLAIMERS,
    TeachingAffineFrameCode,
    TeachingAffineImageCode,
    TeachingCodeComparison,
    TeachingCodingParameters,
    build_canonical_frame_code,
    build_teaching_frame_code,
    build_teaching_image_code,
    compare_teaching_codes,
    recommended_decimals,
    teaching_codes_equal,
    worst_frame_condition_number,
)

__all__ = [
    "CODE_FORMAT_VERSION",
    "DISCLAIMER_EN",
    "DISCLAIMER_ZH",
    "DISCLAIMER_RU",
    "DISCLAIMERS",
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
