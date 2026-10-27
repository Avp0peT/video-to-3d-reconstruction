# 有限精确研究入口

这些功能是独立研究侧支，不修改照片、SAM3掩膜、SfM相机或生成网格。输入必须明确声明完整点集；小点集证书不等于完整照片物体的数学重建。

## 覆盖中心族

从 `kozlov_lab.kozlov2025.cover_center_family` 导入 `build_cover_center_family` 和 `verify_cover_center_family`。输入为1–7个完整二维点的标签字典，接口和共享工作/时间上限见 [合同](contracts_cover_center_family_v1.md)。结果为每个合法覆盖保存块中心、尺寸和独立证书；不能处理的覆盖保留UNKNOWN。调用者必须同时查看 `complete`，不能把核验器返回true当作全族完成。

已有完整真实7点实例实跑877个分划、433个合法覆盖，407精确成功、26未知，整体PARTIAL_UNKNOWN。证据与可复核精确JSON见 [交付](../outputs/delegation/GENERAL_PIPELINE/RUN20261013_cover_center_family/HANDOFF.md)。这不是一般4/5接触域的完整椭圆求解。

## 给定互位的 futlyar 判定

命令入口 `scripts/verify_generalized_futlyar.py` 接受完整给定点集、中心和半径。只用整数或精确有理数；不接受浮点近似，也不搜索其他仿射互位。以下是**人工小例**，不是照片或测量数据：

```json
{
  "schema": "generalized-futlyar-input-v1",
  "points": [
    {"label": 7, "xy": [0, 0]},
    {"label": 9, "xy": [{"numerator": "1", "denominator": "2"}, 0]},
    {"label": 11, "xy": [3, 0]}
  ],
  "centers": [{"label": 20, "xy": [0, 0]}, {"label": 30, "xy": [3, 0]}],
  "radius": 1
}
```

```powershell
& '<项目Python>' -B -X utf8 scripts/verify_generalized_futlyar.py --input '<完整输入.json>' --output '<新的结果.json>'
& '<项目Python>' -B -X utf8 scripts/verify_generalized_futlyar.py --input '<同一输入.json>' --verify '<结果.json>' --output '<新的核验.json>'
```

输出不覆盖。TRUE结果保存完整分块、精确距离和输入字节SHA，并在写出前独立核验序列化后的证书。第二条命令独立重算证书并核对同一完整输入。FALSE只说明给定互位不成立；UNKNOWN表示工作预算不足；单中心等未定域为UNSUPPORTED。退出0仅TRUE，退出1保留其他数学状态，退出2为输入或证据错误。默认距离计算预算262144；可以用 `--max-work` 减小。

本CLI拒重复JSON键、重复/布尔标签、浮点坐标、未约分有理数和已存在输出。输入及待核验JSON限制4MiB，适合有限研究实例；更大的证书使用Python API，不能截断点集以适应文件限制。严格定义、支持域和仍未实现的最优fits/primary futlyar见 [合同](contracts_generalized_futlyar_v1.md)。真实尺度仍unknown。
