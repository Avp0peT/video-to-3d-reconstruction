# 中俄术语对照 / Русско-китайский словарь терминов

本项目以 В. Н. Козлов 的俄文教材为理论来源，界面与文档使用中文。
本表是两者之间的**唯一对照表**：代码中的注释、界面的「为什么？」面板、
报告中的术语都必须引用此表，不得另起一套译法。

| 俄文 | 中文 | 英文 | 代码标识 |
|---|---|---|---|
| изображение | 图像 | image | — |
| конечное множество точек | 有限点集 | finite point set | `PointSet2D` |
| точка | 点 | point | `Point2D` |
| зрительное восприятие | 视觉感知 | visual perception | — |
| распознавание изображений | 图像识别 | image recognition | Phase 4 |
| преобразование | 变换 | transformation | `AffineTransform` |
| параллельный перенос / сдвиг | 平移 | translation | `classify() == "translation"` |
| поворот | 旋转 | rotation | `"rotation"` |
| отражение | 反射 | reflection | `"reflection"` |
| изометрия | 等距变换 | isometry | `invariant_group() == "isometry"` |
| преобразование подобия | 相似变换 | similarity | `"similarity"` |
| аффинное преобразование | 仿射变换 | affine transformation | `"affine"` |
| вырожденное преобразование | 退化（奇异）变换 | singular transformation | `"singular"` |
| аффинные координаты | 仿射坐标 | affine coordinates | `AffineCoordinates` |
| внутренние координаты | 内部坐标 | internal coordinates | $(\alpha,\beta)$ |
| базисный треугольник / репер | 基准三角形 / 标架 | reference frame | `AffineFrame` |
| невырожденный | 非退化的 | non-degenerate | `is_invertible` |
| коллинеарность | 共线性 | collinearity | `triangle_flatness`、`is_non_collinear` |
| параллельность | 平行性 | parallelism | `parallelism_relation_change` |
| инвариант | 不变量 | invariant | `invariants.py` |
| код изображения | 图像代码 | image code | Phase 3（未实现） |
| канонический код | 规范代码 | canonical code | Phase 3（未实现） |
| эквивалентность | 等价性 | equivalence | Phase 4 |
| $k$-эквивалентность | $k$-等价 | $k$-equivalence | Phase 10 E1（未实现） |
| сжатие информации | 信息压缩 | information compression | Phase 5（未实现） |
| $\varepsilon$-покрытие | $\varepsilon$-覆盖 | epsilon-cover | Phase 5（未实现） |
| стереозрение / стерео vision | 立体视觉 | stereo vision | Phase 6（未实现） |
| проекция | 投影 | projection | Phase 6（未实现） |
| триангуляция | 三角测量 | triangulation | Phase 6（未实现） |
| соответствующая точка | 对应点 | corresponding point | Phase 6（未实现） |
| дискриминантная функция | 判别函数 | discriminant function | Phase 10 E2（未实现） |
| линейный классификатор | 线性分类器 | linear classifier | Phase 10 E2（未实现） |
| перцептрон | 感知机 | perceptron | Phase 10 E3（未实现） |
| структурно-лингвистическое распознавание | 结构—语言学识别 | structural-linguistic recognition | Phase 10 E4（未实现） |
| грамматика | 文法 | grammar | Phase 10 E4（未实现） |
| голосование | 投票 | voting | Phase 10 E5（未实现） |
| неприводимый тест | 不可约检验 | irreducible test | Phase 10 E5（未实现） |
| обучающая выборка | 训练样本集 | training set | Phase 10 E3 |
| линейная разделимость | 线性可分性 | linear separability | Phase 10 E3 |

---

## 数值与工程术语（本项目自定义，非教材术语）

| 中文 | 英文 | 代码标识 | 定义 |
|---|---|---|---|
| 平坦度 | flatness | `triangle_flatness` | $2\,\mathrm{Area}/\max(\text{边})^2$，置换不变的共线性度量 |
| 顶点正弦 | vertex sine | `collinearity_sine` | $\lvert\sin\angle BAC\rvert$，**依赖顶点命名**，仅用于诊断条件数 |
| 条件数 | condition number | `condition_number` | $\sigma_{\max}/\sigma_{\min}$ |
| 可达精度界 | accuracy bound | `tolerance_for_condition` | $\max(\mathrm{FLOAT\_TOL},\ 64\,\mathrm{cond}\,\varepsilon)$ |
| 无见证 | vacuous | `verdict == "vacuous"` | 关系为空，结论不含信息（**不等于**"保持"） |
| 集合指纹 | set signature | `signature()` | 顺序无关、忽略重数的量化指纹 |
| 重数指纹 | multiset signature | `multiset_signature()` | 顺序无关、保留重数的量化指纹 |

> ⚠️ **译名注意**：`невырожденный`（非退化）在本项目里对应两个不同的检查：
> ① 变换的 $\det A\ne0$（`is_invertible`）；
> ② 三点的共线性判据（`is_non_collinear`）。
> 二者**不可互换**：一个由共线点构成的图像可以有完全可逆的变换。
> 界面文案必须说清是哪一个。

## 界面文案模板（README 第 20 节要求：图形 + 公式 + 数值 + 解释）

仿射坐标页面：

```text
外部坐标：
  P = (5.82, 7.31)

内部坐标：
  P = (0.50, 0.25)ABC

公式：
  P - A = 0.500 (B - A) + 0.250 (C - A)

中文：
  点 P 的位置可以完全由它相对于 A、B、C 的关系描述。

俄文：
  Положение точки P задаётся её внутренними аффинными координатами
  относительно A, B и C.

规范度：
  |sin(angle BAC)| = 0.555   cond([B-A, C-A]) = 3.37   → 良态
  可达精度 ≈ 1e-12
```
