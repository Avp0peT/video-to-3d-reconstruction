# 原生深度文件读取与排除结果检查

模块：`integrations/openmvs_depth/`。使用项目既有 NumPy/Pillow 环境；不需要导入 SAM3、torch、OpenCV 或 pycolmap，不启动 DensifyPointCloud。完整规则见 [契约](contracts_openmvs_depth_v1.md)。

从项目根目录导入，只读查看单份文件：

```python
from integrations.openmvs_depth import read_dmap

depth = read_dmap(r"C:\your-new-run\masked\depth0004.dmap")
print(depth.sha256, depth.view_ids, depth.depth_size)
print(depth.depth)  # [row,column]；0 表示无有效深度
```

对照核验 API：

```python
from integrations.openmvs_depth.verify import verify_depth_pair

report = verify_depth_pair(
    masked_dmap, control_dmap,
    input_root=new_comparison_root,
    masked_image_path=staged_masked_image,
    control_image_path=staged_control_image,
    image_sha256=frozen_image_sha256,
    mask_path=undistorted_mask, mask_sha256=frozen_mask_sha256,
    validity_path=undistorted_validity, validity_sha256=frozen_validity_sha256,
    reference_id=native_global_image_id,
    expected_K=depth_resolution_K, expected_R=world_to_camera_R,
    expected_C=camera_center_C,
    masked_ignore_label=0, control_ignore_label=-1, postprocess_dmaps=0,
)
```

变量须来自冻结输入与实际运行，不能从待验证文件自身抄一遍作为独立预期。K 是实际深度分辨率的校准；mask/validity 应是 B 消费模块输出的去畸变域结果，不能直接放原 SAM3 canonical mask。两次 stage 的图片路径可不同，实际 bytes/hash 必须相同；相对 `.dmap` 图像路径经解析后分别匹配这些文件。

读取错误抛出 `DMapError`，比较身份/输入错误抛出 `DepthVerificationError` 或底层 I/O 异常。调用者负责将异常记录为失败，而不是输出默认成功报告。函数自身不写文件；保存时请使用新文件并以 `json.dump(..., allow_nan=False)` 写完整报告。

`SUPPORTED` 只支持这一对保存深度的效果；`VIOLATION` 是声明忽略位置仍有深度；`INCONCLUSIVE` 是缺少对照信号、开启后处理或缩放映射未经验证。任何返回都不认证原生程序、缓存新鲜度、mask 实际读取、形状正确或 CAD 完成。尺寸变化的参考采样仅作诊断，保持 INCONCLUSIVE，待真实绑定 OpenCV 后另扩展。

当前验证级别为 **TESTED_SYNTHETIC**：测试自行构造 DR bytes 与小 PNG 标签。尚无实际 Densify `.dmap` 回读、原生 masked/control 对照、真实 mask 质量或稠密/网格/CAD 证据。本轮没有执行外包 A pilot03 或 B consumer。
