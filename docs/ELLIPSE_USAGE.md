# 二维最小面积外接椭圆：数值模块用法

`kozlov_lab.kozlov2025.ellipse.minimum_enclosing_ellipse` 返回带面积证书的数值近似，或明确的退化/失败状态。当前证据为 `TESTED_REAL_PARTIAL`（数值有限点域及真实 SfM 特征像素单例已测试）；RC04 与整个项目均未宣布完成。

定义来源为冻结的 S-2025 俄文论文 PDF p.4 / 期刊 149：中心是包含图像凸包的唯一最小面积椭圆的中心，size 是长轴**全长**；共线时退化成线段，中点为中心。原文文件为 [kozlov2025_three_approximations.pdf](../.sources/kozlov2025_three_approximations.pdf)，SHA-256 为 `82fc78d0c0f8fa3de99ede3fd227adaeddc8f8328a1ef99d06755f1ec231b8f9`；来源登记见 [SOURCES.md 的 S-2025](sources/SOURCES.md)。本项目的单点退化约定、实现和完整证书推导以[契约 `ellipse-area-v1`](contracts_ellipse_v1.md) 为准。

数值算法采用契约引用的 Todd–Yıldırım 作者版 §2 / Algorithm 3.1 的 Khachiyan 对偶更新。NumPy float64 负责求候选，Python 标准库 `Fraction` 负责最终精确包含性和面积比检查；本模块没有新增 SciPy/OpenCV 依赖。

## API 示例

以下代码在能够导入本项目包的既有 Python 环境运行；它是合成矩形示例，不是实物测量。

```python
from kozlov_lab.kozlov2025.ellipse import (
    EllipseInputError,
    minimum_enclosing_ellipse,
)

points = [[-3.0, -2.0], [-3.0, 2.0], [3.0, -2.0], [3.0, 2.0]]

try:
    result = minimum_enclosing_ellipse(
        points,
        area_tolerance=1e-3,
        max_iterations=10_000,
        max_work=10_000_000,
    )
except EllipseInputError as exc:
    print("输入或参数无效：", exc)
else:
    if result.status == "APPROXIMATE_CERTIFIED":
        print("中心显示值：", result.center)
        print("长轴全长显示值：", result.major_axis_length)
        print("面积比上界：", result.certificate["area_ratio_upper_bound"])
        print("全部输入点包含：", bool(result.contains(points).all()))
    elif result.status in ("DEGENERATE_POINT", "DEGENERATE_SEGMENT"):
        print("退化几何：", result.status)
        print("精确中心：", result.certificate["center_exact"])
    else:
        print("面积要求未获证明：", result.status)
        print(result.certificate)

    record = result.to_dict()  # 可 JSON 序列化的独立副本，保留状态和证书
```

`converged` 对 `APPROXIMATE_CERTIFIED`、`DEGENERATE_POINT`、`DEGENERATE_SEGMENT` 为真。应用应先检查 `status`，因为退化成功没有非退化椭圆 factor，也没有正面积的面积比保证。

## 输入域与资源边界

| 参数 | 约束与含义 |
|---|---|
| `points` | 1–4096 个有限二维 binary64 兼容数值坐标，形状为 `(n, 2)`；上限在去重前检查 |
| `area_tolerance` | 默认 `1e-3`，范围 `[1e-8, 0.25]`；只约束面积相对超额 |
| `max_iterations` | 默认 `10000`，整数范围 `[0, 100000]` |
| `max_work` | 默认 `10000000`，整数范围 `[1, 100000000]`；更新次数 × 唯一点数 |

拒绝 bool、complex、string、空输入、不规则维数、NaN/Inf，以及不能无损表示为 binary64 的整型坐标。float32 等输入按其实际表示值提升；转换前已经丢失的测量或坐标精度不会恢复。只去除坐标完全相同的点，`unique_points` 为排序后的集合；没有容差聚类、极点筛除或默认子采样。

实际更新上限为 `min(max_iterations, max_work // n_unique)`。零次更新也可能让对称初值通过证书。work 不包含排序、初始化/终止检查、预处理和末端 `Fraction` 证书，因此不是墙钟或内存硬上限；批量或真实数据调用应另设外层预算。预算耗尽后不能把未证明的结果当作成功，也不能静默放宽 tolerance 或扩预算。

## 状态与下游处理

| 状态 | 含义与使用边界 |
|---|---|
| `APPROXIMATE_CERTIFIED` | 全部实际输入点被精确包含，且面积比满足指定上限；仍为数值近似 |
| `DEGENERATE_POINT` | 唯一点，面积/尺寸为 0，中心是该点；`factor=None` |
| `DEGENERATE_SEGMENT` | 存储坐标精确共线，保留真正极端端点；精确有理中点及长度平方在证书中，`factor=None` |
| `MAX_ITERATIONS` | 迭代或 work 上限达到，指定面积界未获证明；可返回精确包含的候选，但 `converged=False` |
| `NUMERICAL_FAILURE` | 白化、候选、证书或显示范围失败，原因在证书中；可能无几何，不能当面积通过 |
| `EllipseInputError` | 非法输入/参数抛出异常，不返回成功结果 |

真共线由原始存储 float 的精确有理数秩判断。近共线但精确秩 2 的数据不会被压成线段；不能稳定白化时允许 `NUMERICAL_FAILURE`，不会人为加厚以制造成功。

退化线段的 `center` 是 binary64 显示值；`center_exact` 是精确中点，`squared_length_exact` 是精确长度平方。长度溢出或非零半轴无法以 binary64 显示时，相应字段为 `None`（JSON 中为 `null`），并附原因；不能将缺失显示值解释为零长度。

## 几何字段与证书

非退化椭圆的权威表示为：

```text
|| factor @ (x - center) ||² <= 1
```

`factor` 是 2×2 矩阵 B。避免为判包含而先把它乘成病态的 `B.T @ B`；`result.contains(query_points)` 会直接对保存的 binary64 值执行 `Fraction` 精确判定，不用任意 epsilon 把外点变成内点。查询点也受同样的输入域限制；失败结果可能没有可供查询的几何，此时 `contains()` 抛 `ValueError`。

`semiaxes` 按长/短半轴排列，`major_axis_length` 是长轴全长，`area` 是面积显示值。这些数值是浮点近似，不是额外的中心或尺寸精度证书。结果数组由不可写 bytes 支持；`certificate` 和 `to_dict()` 返回独立副本，正常调用不能修改几何却继续携带旧证书。

最终证书从**实际存储的输入坐标、中心 c、factor B、权重和 tolerance** 直接转为有理数。权重按其有理总和精确归一，计算加权协方差 `C_u`。对任何包住输入点的椭圆，最优面积有下界 `L = 2π√det(C_u)`；已精确验证包含的输出面积为 `U = π/|det(B)|`。因此：

```text
R² = U² / L² = 1 / (4 * det(B)² * det(C_u))
1 <= area(output) / area(optimum) <= R
```

只有精确有理数比较 `R² <= (1 + tolerance)²` 成立才返回 `APPROXIMATE_CERTIFIED`；浮点迭代停止条件本身不足以成功。末端若需向外膨胀候选，写回 float 后仍重新精确验证，次数有界。

| 证书字段 | 解读 |
|---|---|
| `proof_scope` | 固定为 `stored_binary64_inputs`，不覆盖采样、传感器或真实物体误差 |
| `containment_exact`、`max_quadratic_exact` | 对当前保存的 c/B 精确验证所有输入点；后者是最大二次值 |
| `det_factor_exact`、`dual_covariance_det_exact` | 用于独立重算面积上下界的精确有理行列式 |
| `area_ratio_squared` | 精确有理数 `R²`，分子/分母以字符串保存 |
| `area_ratio_upper_bound` | 经精确平方检查、向上舍入的 binary64 `R` 上界 |
| `area_approximation_certified` | 指定面积比要求是否成立；不能只看候选存在或包含为真 |
| `exact_optimum` | 非退化候选固定为 `False` |
| `center_coordinate_error_bound`、`axis_error_bound` | 非退化候选为 `None`，没有相应误差保证 |
| `stop_reason`、`iterations`、`solver_point_update_work` | 停止原因及实际更新计数；`iterations` 在结果顶层 |

上述非退化字段不要求出现在所有失败或退化结果中；应先看状态，再读取对应字段。面积比里的 π 相消；`area` 的浮点显示与严格面积比证书是两层证据。

`area_tolerance=1e-3` 表示通过证书时，输出面积相对该存储点集的最优面积超额不超过 0.1%。它**不表示**中心误差不超过 0.001、轴长误差不超过 0.1%，也不证明物体毫米尺寸、分割或图像测量准确。

## 已有证据与剩余范围

运行目录为 [`outputs/delegation/ELLIPSE_2025/20261007T105441Z_main/`](../outputs/delegation/ELLIPSE_2025/20261007T105441Z_main/)。保存的 [pure02.xml](../outputs/delegation/ELLIPSE_2025/20261007T105441Z_main/pure02.xml) 显示 45 例、0 failure、0 error、0 skipped，时间为 `2026-10-07T19:06:36.552746+08:00`。这是已有纯数据测试证据，本次文档编辑未重跑测试。

用例覆盖已知三角形/矩形/圆对称解、非对称云及独立重算面积证书、仿射中心/形状/面积协变、包含性独立 oracle、点/线/重复/近线退化、预算耗尽与零更新成功、非法输入，以及防止几何/证书可变造成陈旧证明。有限测试不替代所有输入的一般证明或完整 Kozlov 理论验收。

真实 SfM 特征像素 demo 已在同一 RUN 内仅执行一次，记录见 [DEMO_RESULT.json](../outputs/delegation/ELLIPSE_2025/20261007T105441Z_main/DEMO_RESULT.json)、[demo01.json](../outputs/delegation/ELLIPSE_2025/20261007T105441Z_main/demo01.json) 与[完整椭圆输出](../outputs/delegation/ELLIPSE_2025/20261007T105441Z_main/real_features_ellipse.json)。

| 项目 | 实际记录 |
|---|---|
| 输入来源 | 既有 T03 `native01/real44_scene/research_scene.json`；SHA-256 `c523b22c7e0dc1495f9d10b87690e8cb0fb1730eb57a31d0b16bda82434d1f27`，运行后未变 |
| 输入图像 | native image ID `4`，`images/000003.png`，camera ID `1`、frame ID `4` |
| 点集 | 全部 3571 个特征像素，仅精确去重为 3179 点，无子采样 |
| 参数 | `area_tolerance=0.01`、`max_iterations=2000`、`max_work=2000000` |
| 实际输出 | `APPROXIMATE_CERTIFIED`，395 次更新，面积比上界 `1.0049380785009892 ≤ 1.01` |
| 独立检查 | 从输出 JSON 与原输入使用 Fraction 和原始二阶矩重算，3571 个原始点均精确包含，面积比检查通过 |
| 调用记录 | 唯一一次 demo，退出码 0，外层耗时约 0.945 秒；绑定文件未变 |
| 未评估项 | 真实物体中心/尺寸精度为 `NOT_EVALUATED`；未作物体 mask、物体轮廓或背景移除 |

这个单例只支持 `TESTED_REAL_PARTIAL`。SfM 特征像素集合可能含背景且不构成物体轮廓；对该集合的包含性和面积比不代表物体 mask、图像边界、背景移除、完整重建或研究收益通过。

当前模块不将数值近似自动传播为严格 fits、独立覆盖、ostov、完整群或高层形象结论。其余未决项见 [open_questions.md](open_questions.md)，整体完成状态见 [PROJECT_COMPLETION_CHECKLIST.md](PROJECT_COMPLETION_CHECKLIST.md#rc04--2025-最小面积外接椭圆的完整数值契约)。
