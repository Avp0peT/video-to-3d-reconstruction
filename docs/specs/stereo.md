---
spec: stereo
module: M8 立体视觉
status: planned
source_tier: S0（结构层）/ S1（相机模型与对应点求解，待确认）
phase: 6
implemented_in: null
blocked_by:
  - 未决问题 Q4
---

# 规格 M8：立体视觉（**尚未实现**）

## 1. 目标（README 第 11 节）

建立三维点集 $P_i=(X_i,Y_i,Z_i)$，两个虚拟相机，投影

$$p_L=P_LX,\qquad p_R=P_RX,$$

再由两个二维投影三角测量得到 $\hat P$，比较 $P$ 与 $\hat P$：

$$E=\frac1n\sum_i\|P_i-\hat P_i\|.$$

## 2. 两版实现

| 版本 | 对应关系 | 说明 |
|---|---|---|
| 第一版 | **已知** | 左图第 $i$ 点对应右图第 $i$ 点 |
| 第二版 | **未知** | corresponding point problem |

README 明确把「由平面投影恢复三维图像」与「确定投影中的对应点」
列为其立体视觉模型的重要内容。

## 3. ⚠️ 概念边界：不要与 Phase 4 混淆

立体视觉的对应点问题与 Phase 4 的仿射等价**不是同一个数学问题**：

| 问题 | 变换族 | 关键约束 |
|---|---|---|
| Phase 4 仿射等价 | 仿射群 $GL_2\ltimes\mathbb R^2$ | 内部仿射坐标不变 |
| Stereo 对应点 | **投影**变换 | 极线约束（epipolar） |

投影变换比仿射更广（仿射是投影的子群），因此**内部仿射坐标在投影下不再保持不变**，
Phase 2/3 的不变量机制不能直接搬过来。
报告中必须写明这一区别，否则会被读成"不变量方法可以解决立体匹配"。

## 4. 接口草案

```python
camera_matrix(camera) -> (3, 4) ndarray
project_points(points3d, camera) -> (n, 2) ndarray
triangulate(left, right, cameras) -> {
    "points3d": (n, 3),
    "reprojection_error_left": float,
    "reprojection_error_right": float,
}
```

## 5. 计划实验（Experiment E）

生成已知三维点 → 投影到两个相机 → 重建 → 比较 $P$ 与 $\hat P$。

必须额外报告：

- **重投影误差**（与三维误差分开）：三维误差小但重投影误差大，意味着重建落在
  相机之间的"盲区"，这种情况下的低 $E$ 不可信；
- 相机基线长度对 $E$ 的影响（三角测量的经典病态区：基线与深度方向接近平行）。
