# 用法与证据边界：`labelled_stereo_v1`

产品：`src/kozlov_lab/stereo/official/parallel_construction.py`
契约：`docs/contracts_labelled_stereo_v1.md`
测试：`tests/test_parallel_construction.py`
来源：S-1996（Козлов 1996, pp.331–338）· 步骤抽取 `docs/specs/stereo_algorithms_1996.md`
R1/R2 历史证据：`outputs/delegation/LABELLED_STEREO_C/<run_tag>/`
主线补充验收：`outputs/delegation/LABELLED_STEREO_C_MAIN/20261008_r2_closure/HANDOFF.md`

模块**未**加入 `src/kozlov_lab/stereo/official/__init__.py`（该文件按任务只读），
因此请按完整路径导入。

## 1. 最小用法

```python
from fractions import Fraction as F
from kozlov_lab.stereo.official import parallel_construction as pc

first = {"a": (F(0), F(0)), "b": (F(1), F(0)), "c": (F(0), F(1)),
         "e": (F(1, 4), F(1, 4)), "d": (F(5, 4), F(1, 4)),
         "g": (F(2), F(3)), "h": (F(1, 4), F(9, 4))}
second = {"a": (F(0), F(0)), "b": (F(1), F(0)), "c": (F(0), F(1)),
          "e": (F(5, 4), F(9, 4)), "d": (F(9, 4), F(9, 4)),
          "g": (F(4), F(7)), "h": (F(13, 4), F(33, 4))}

prep = pc.prepare_views(first, second, "a", "b", "c", "e")   # 校验 + 固定规范化

rep = pc.construct_representative(prep)                       # AlgT'
print(rep.status, rep.points["d"], rep.witnesses["d"].branch)

img = pc.construct_output_image(prep, ((F(0), F(0)), (F(1), F(0)),
                                       (F(0), F(1)), (F(2), F(1))))   # AlgS0
print(img.status, img.points["d"], img.points["g"] == img.points["h"])
```

`prepare_views` 抛 `ValidationError`（非法值、标签集合不一致、`abc` 退化、**对应不相容**）
或 `BudgetExceeded`（预算耗尽）——两者都是**明确拒绝**，不会静默量化或补齐。

## 2. 用 `base_parameters` 复核 U1

```python
a = pc.construct_representative(prep).points
b = pc.construct_representative(prep, base_parameters=(F(2), F(5))).points
assert a == b          # 换交线基点/参数，结果不变
```

看点：`witnesses["d"].detail["line"]["detail"]["second_point_method"]` 记录引理 6 情形 4 用的是
`de_intersection` / `de_coincident_fixed_point` / `aux_edge_fixed_point(ab)` 等哪一种；
辅助边随局部角色而变，也可为 `bc` 或 `ac`。

## 3. 怎么读见证

```python
w = rep.witnesses["d"]
w.branch          # 来源分支名（见契约 §5 的映射表）
w.route           # 走到的路径标记，例如 ('STEP2B_REDUCE_FACE_AED',)
w.source_view     # 步骤 2 用的是哪一幅图作为映射源（'second' / 'first'）
w.completeness    # 'CONSTRUCTED' 或 'UNSUPPORTED'（后者附 w.note）
w.detail          # 基点、参数、交线、比值等可核验中间量
```

**分支名 + 坐标都要看。** 只有名字没有正确的精确坐标，不算覆盖（契约 §7）。

## 4. 证据边界（必须随结果一起转述）

* 精确性只针对**输入的有理坐标**；`scale_status = "unknown"`，**不**给真实单位/精度。
* `OK` 只表示所给对应按来源步骤构造成功；**不**表示对应是对的、也不表示照片可用。
* 未知对应搜索、噪声/透视、遮挡、错误匹配：**不在**本模块范围。
* 定理 6 的 `объемное тело` 定义域**另说**；构造成功不自动等于定理 6 通过。
* 本轮未运行真实照片、SAM3、SfM、OpenMVS、FreeCAD 或 GUI。

## 5. 复现

```bash
PY=outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe
"$PY" -B -m pytest tests/test_parallel_construction.py
```

测试自带的独立真值来自**固定有理三维体 + 两个方向投影**，并用本文件内的重心坐标 / 高斯消元
求解标记载荷仿射映射（与产品的 Cramer 风格求解器**不共用**）。规格第 8 节的代数对照只作**次级**
交叉检查，且只存在于测试中；真值断言在前，互相一致不取代真值。

## 6. R2 修订要点（务必一起转述）

* **失败要分三类**，不要混为一谈：
  - 错标签/取值/结构 → `prepare_views` 抛 `ValidationError(kind="INVALID_INPUT")`；
  - 格式合法但不相容 → `ValidationError(kind="INCOMPATIBLE_CORRESPONDENCE")`；
  - 输入**合法**但某分支实现不了 → 该标签 `UNSUPPORTED`（整体 `UNSUPPORTED_BRANCH`，gate 不通过）。
    内部构造失败**不会**被写成"输入非法"。
* **换名后的角色必须代入**：`_lemma6_line` 的辅助边由面的局部已知顶点推出
  （`c`→`ab`，`a`→`bc`，`b`→`ac`）。可用
  `witnesses[label].detail["line"]["detail"]["second_point_method"]` 核对：
  `aux_edge_fixed_point(bc)` 之类表示走了换名后的辅助边。
* **`STEP2B_REDUCE_FACE_BED` 是可达的成功分支**：三维面不保证投影成三角形，`aed` 一条路不可用
  （三维共线或投影退化）时按步骤 2б 继续走 `bed`。
* **`U1` 基线退化在 `STEP2` 分支中不可达**，前提是 `abc` 两图皆三角形、且源视图里
  `(面已知顶点, d, e)` 成三角形（证明见契约 §6.7）。该守卫不计入分支覆盖。

主线补充用例令 `e=(1/4,3/2,1), d=(1,3,2)`，两个投影为 `(x,y)` 和 `(x+z,y+2z)`。
它验证换到 `aed` 面后由局部共同点 `b` 确定交线，代表体 `d=(1,3,2)`；上述输出锚点下
`d0=(9/2,2)`。这补充了 R2 明示未覆盖的 `COMMON_POINT_RENAMED_FACE`，实际测试结果
见主线补充验收入口。已知分支在声明域内的有限精确用例通过，不等于所有输入的形式证明。
U1 用例总数和真正检查到的非退化基线数分别写入本次 JUnit 属性，不用总数代替守卫检查数。
R1/R2 的丢失首轮记录、外层起止或退出码证据仍属历史缺口，本次记录不能补造这些证据。
