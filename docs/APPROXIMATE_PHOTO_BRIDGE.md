# 真实照片到 Kozlov 六点构造的工程适配

本模块复用已有 COLMAP 对应、相机内参与姿态，把已冻结的六个场景点送入实际的 S-1996 AlgT′。它是**依赖标定的工程扩展**，不是作者原文中已给出的透视重建算法，也不是新的未知对应求解器。

入口：`scripts/experiment_photo_parallel.py`。输入是 `audit_photo_parallel.py` 生成的 `INPUT.json`。输出必须是新目录；原照片、SfM 和此前诊断均只读。

```powershell
python -B -X utf8 scripts/experiment_photo_parallel.py --prior-input '<旧诊断目录>\INPUT.json' --output '<新实验目录>'
```

## 固定输入与两条实验

重新核验源文件大小、SHA-256、二进制双向 tracks 和原六点 binary64。每个视图有多个观测的 track 保留排除记录，不从中任选一个。不按残差重选点。

1. **近似平行诊断**：先排除原六点，其余 ID 升序交替划入训练和验证。raw 与去畸变像素使用同一划分。四维 TLS 仅在训练集拟合，验证不重拟合。固定 18bit 网格以共同有理仿射关系生成六点代理，随后调用实际 AlgT′。代理证书只覆盖改造后的输入。
2. **标定透视桥**：SIMPLE_RADIAL 有界逆解 → 共旋转立体校正 → 两侧 y 均值 → 固定 18bit 有理量化 → 实际 AlgT′（两视图）→ 独立证书核验 → 从构造的 body 及两投影矩阵恢复像素 → 用现有相机基线和视差映回旧世界坐标系。

TLS 的归一尺度是两相机最大图像边长；输出残差恢复为像素单位。校正代理的坐标则是 `(pixel-principal)/f`，`scale=1`，像素量化间距 `f/2^18`；两种尺度不能混用。输入、划分、这些常数在数值拟合前写入 `PLAN.json`。

径向逆解的工程域固定为“图像四角最大归一半径的两倍”；负畸变系数若在此域不单调，直接拒绝，不通过缩域选根。最多 128 步，每点正向重畸变误差 ≤1e-8 像素。它不是额外标定，也不证明真实镜头参数准确。

## 数学含义

共旋转校正后，理想两图有相同 y。它们可表示为三维**逆深度代表** `(X/Z, Y/Z, 1/Z)` 的两次仿射投影。此时 AlgT′ 的任意仿射坐标**不能直接称为实际物体坐标**。

工程层先精确核对 AlgT′ 证书将全部六点重投影回所声明的有理输入，再恢复校正像素。令 `d=x1-x2`、相机基线为 `B`，则 `Z=fB/d`、`X=(x1-cx)B/d`、`Y=(y1-cy)B/d`，最后用校正旋转和第一相机中心映回原世界系。非正/接近零视差、地平线与非正深度拒绝。

还有独立数值闭环：世界点经**原**相机 `R/t/K` 正投影，再经校正单应，须与证书恢复的像素在 1e-7 像素内一致。这是变换一致性门，**不是照片重建质量阈值**。原始畸变像素的回投影误差单独报告，不截断、不按误差删点。

## 产物与结论边界

- `PLAN.json`：拟合前冻结选择、来源和标定。
- `OBSERVATIONS.json`：完整 raw→ideal→rectified 坐标、逆解误差、Sampson 和纵坐标差。
- `TLS_DIAGNOSTICS.json` 与两个 TLS 代理：训练/留出证据与逐点修改量。
- `RECTIFIED_SURROGATE.json` / `RECTIFIED_EXACT.json`：修改账本、实际 AlgT′ 结果和证书。
- `CALIBRATED_POINTS.json` / `calibrated_six_points.ply`：六点工程候选、几何回读、原像素重投影；**没有面**。
- `POINT_COMPARISON.json` / `old_colmap_six_points.ply`：同 ID、同旧世界坐标系对照。原 XYZ 仅在新候选形成后读取，未作为拟合或构造输入；它也不是真值。
- `RESULT.json` / `report.html`：分层结论。

旧 COLMAP 点由多视图估计，新候选来自两视图且依赖旧相机。距离是未知单位下的方法差异，不是精度提升。Sampson、四维 TLS、两图纵坐标差的定义不同，不能直接排名。留出划分只约束新 TLS 拟合；旧相机标定已经使用这些 tracks，所以不是独立相机标定验证。

原始六点可能仍精确 rank4；改造后输入的 rank3 是工程处理施加的条件，不能冒充发现真实照片天然满足论文假设。六点场景子集不证明前景表面、网格、CAD、真实尺度或未知对应能力。

公式/格式参考：COLMAP [输出格式](https://colmap.github.io/format.html)、[SIMPLE_RADIAL 源码](https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/sensor/models/radial.h)、[essential matrix 源码](https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/geometry/essential_matrix.cc)。有界逆解、校正桥、TLS 和量化均为本项目实现；论文构造仍使用现有 `parallel_construction.py`。
