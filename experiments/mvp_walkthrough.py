"""Scripted MVP walkthrough — README section 25, steps 1-7.

This is the terminal-runnable version of the live demo.  It performs the ten
steps of the MVP completion standard and stops honestly at step 7:

| step | what the README asks for | status here |
|---|---|---|
| 1 | create a figure with at least 5 points | done |
| 2 | generate a random invertible affine transform | done |
| 3 | show the changed figure | done (coordinates printed) |
| 4 | show that the external coordinates have all changed | done |
| 5 | pick three non-collinear reference points | done |
| 6 | compute the internal coordinates of all points | done |
| 7 | prove numerically that they are unchanged | done |
| 8-10 | image code, equivalence - X ~aff Y, X !~aff Z | **Phase 3-4, not claimed** |

Steps 8-10 are deliberately absent rather than faked: the image-code
definition is still blocked on open question Q1 (see `docs/specs/README.md`).

Run::

    python experiments/mvp_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kozlov_lab import __version__  # noqa: E402
from kozlov_lab.datasets import shapes  # noqa: E402
from kozlov_lab.geometry import (  # noqa: E402
    AffineFrame,
    AffineTransform,
    invariant_report,
)

RULE = "=" * 72


def section(number: str, title: str) -> None:
    print(f"\n{RULE}\n{number}  {title}\n{RULE}")


def main() -> int:
    print(f"Kozlov Visual Perception Lab — MVP walkthrough (v{__version__})")

    # ---------------------------------------------------------------- step 1
    section("①", "创建一个至少 5 点的二维图形")
    image = shapes.square(2.0, n_per_side=3)
    x = image.as_array()
    print(f"图像: {image.name}  点数 = {image.n}")
    for i, (px, py) in enumerate(x):
        print(f"   #{i:<2d} ({px:+.4f}, {py:+.4f})")

    # ---------------------------------------------------------------- step 2
    section("②", "随机生成一个可逆仿射变换 T(x) = Ax + t")
    t = AffineTransform.random_invertible(20261001, max_condition=20.0)
    print("A =")
    for row in t.matrix:
        print("   " + "  ".join(f"{v:+.6f}" for v in row))
    print(f"t = ({t.translation[0]:+.6f}, {t.translation[1]:+.6f})")
    print(f"det A = {t.det:+.6f}   cond(A) = {t.condition_number:.4f}   类型 = {t.classify()}")

    # ---------------------------------------------------------------- step 3
    section("③", "变换后的图形 T(X)")
    y = t.apply(x)
    for i, ((px, py), (qx, qy)) in enumerate(zip(x, y)):
        print(f"   #{i:<2d} ({px:+.4f}, {py:+.4f})  -->  ({qx:+.4f}, {qy:+.4f})")

    # ---------------------------------------------------------------- step 4
    section("④", "外部坐标已经完全改变")
    shifts = np.linalg.norm(y - x, axis=1)
    print(f"max_i ||x_i - x'_i|| = {shifts.max():.6f}   (min = {shifts.min():.6f})")
    print("结论：绝对坐标不是视觉对象的可靠描述。")

    # ---------------------------------------------------------------- step 5
    section("⑤", "选择三个不共线参考点 A, B, C")
    frame = AffineFrame.auto(x)
    assert frame.indices is not None
    i, j, k = frame.indices
    print(f"自动选择（优化 cond/flatness）：指数 ({i}, {j}, {k})")
    print(f"   A = {frame.a.tolist()}")
    print(f"   B = {frame.b.tolist()}")
    print(f"   C = {frame.c.tolist()}")
    print(f"flatness = {frame.collinearity_sine:.6f}   cond([B-A, C-A]) = {frame.condition_number:.4f}")
    print(f"det[B-A, C-A] = {np.linalg.det(frame.matrix):+.6f}  != 0  ->  合法标架")
    print("（auto 选择 cond/flatness 最小的三元组，因此这个标架是手算最方便的那种）")

    # ---------------------------------------------------------------- step 6
    section("⑥", "计算所有点的内部坐标 (alpha, beta)")
    before = frame.coordinates_batch(x)
    moved_frame = frame.transformed(t)
    after = moved_frame.coordinates_batch(y)

    print(f"{'#':>3} {'外部坐标':>26} {'内部坐标 (原图)':>24} {'内部坐标 (变换后)':>26}")
    for idx in range(image.n):
        ext = f"({x[idx, 0]:+.4f}, {x[idx, 1]:+.4f})"
        a = f"({before['alpha'][idx]:+.6f}, {before['beta'][idx]:+.6f})"
        b = f"({after['alpha'][idx]:+.6f}, {after['beta'][idx]:+.6f})"
        print(f"{idx:>3} {ext:>26} {a:>24} {b:>26}")

    sample = frame.coordinates(x[3])
    print(f"\n示例（点 #3）公式：{sample.formula(3, 'P')}")
    print("中文：点 P 的位置可以完全由它相对于 A、B、C 的关系描述。")
    print("俄文：Положение точки P задаётся её внутренними аффинными координатами")
    print("       относительно A, B и C.")

    # ---------------------------------------------------------------- step 7
    section("⑦", "数值上证明内部坐标保持不变")
    d_alpha = np.abs(after["alpha"] - before["alpha"])
    d_beta = np.abs(after["beta"] - before["beta"])
    print(f"max |alpha - alpha'| = {d_alpha.max():.3e}")
    print(f"max |beta  - beta' | = {d_beta.max():.3e}")
    print(f"要求 < 1e-9            : {'通过' if max(d_alpha.max(), d_beta.max()) < 1e-9 else '未通过'}")
    print(f"可达精度界 (cond*eps*64) = {max(before['condition_number'], after['condition_number']) * np.finfo(float).eps * 64:.3e}")
    print(f"最大残差 ||P - (A + a(B-A) + b(C-A))|| = {max(before['residual'].max(), after['residual'].max()):.3e}")

    report = invariant_report(
        x, y, kind=t.classify(), invariant_group=t.invariant_group(), frame_indices=frame.indices
    )
    print("\n不变量表（实测）：")
    print(report.markdown_table())

    # ---------------------------------------------------------------- 8-10
    section("⑧–⑩", "图像代码与等价判定 —— 本阶段不做")
    print("README 第 25 节的第 8-10 步（构造代码、判定 X ~aff Y 与 X !~aff Z）")
    print("属于 Phase 3-4，被以下条件阻塞：")
    print("  * Phase 3 Gate：Phase 2 性质测试必须通过 —— 已通过（本页第 7 步即其内容）")
    print("  * 未决问题 Q1：教材正式编码定义尚未由一手资料确认")
    print("详见 docs/specs/image_code.md。此处不提供占位实现。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
