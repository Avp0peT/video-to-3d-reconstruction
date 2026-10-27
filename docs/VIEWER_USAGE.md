# 离线三维成果查看器

入口：`scripts/export_3d_viewer.py`。Python 标准库生成一个自包含 HTML，浏览器打开时不需要 Python 服务、联网、CDN 或额外安装。浏览器须支持 WebGL。源码位于 `integrations/viewer/`。

目前交付的文件：

`outputs/delegation/VIEWER/20261007T094322Z_main/preview02/viewer.html`

在文件管理器中用本地浏览器打开即可。默认场景是现有 44 图、2393 点的真实稀疏成果；另有 400 面合成 CAD 输入网格。两者的来源、单位与限制分别显示。浏览器自动化工具拒绝 `file://`，因此本轮没有界面截图或真实交互验收。状态为 `EXPORTED_NOT_UI_VERIFIED`；需要人工确认页面能显示、旋转、缩放、切换场景和查询顶点。

## 操作

- 拖动视图、方向键或旋转按钮改变视角；滚轮或缩放按钮改变大小；重置回到初始视角。
- 点云保留逐点 RGB，只能选择“点”。三角网格支持表面、线框及点显示；无 RGB 时使用统一预览色。
- “查看顶点行索引”显示 PLY 中从 0 开始的原始坐标，黄色标记始终置前。这个行号不是 COLMAP `point3D_id`。
- 三轴等比例正交显示。红绿蓝轴为包围盒中心的方向指示，不代表原始世界原点。渲染会居中并转换成归一化 float32；原始 double 坐标仍保留在 HTML 数据与顶点读数里。预览不能替代测量工具。

## 导出其他成果

在项目根目录执行（不需要 SAM3、PyTorch、FreeCAD）：

```powershell
python -B .\scripts\export_3d_viewer.py --spec <场景JSON路径> --output <项目outputs内的新目录>
```

场景 JSON 示例：

```json
{
  "schema": "ply-viewer-scenes-v1",
  "scenes": [{
    "ply": "relative/to/spec/model.ply",
    "sha256": "填写该PLY文件的64位小写SHA256",
    "title": "我的成果",
    "data_kind": "unspecified",
    "declared_unit": null,
    "note": "来源和限制说明",
    "evidence": ["relative/to/spec/report.json"]
  }]
}
```

`data_kind` 为 `real_photos`、`synthetic` 或 `unspecified`；`declared_unit` 为 null、mm、cm、m、in。来源与单位是导入者声明；报告文件只绑定哈希，查看器不重新裁定报告中的 PASS。默认未知尺度应填 null。现有 `scenes.json` 可作实际范例。

接受 PLY 1.0 ASCII / binary little endian / binary big endian。x/y/z 必须存在；RGB 必须同时为 uchar/uint8；face 可缺省，存在时必须为索引有效的三角面。额外标量字段经检查后忽略并登记；额外元素、多边形、额外列表、非有限值、尾随数据及输入哈希漂移拒绝。文件、点、面超限时拒绝，不悄悄抽样、补面或去点。

上限：1–8 场景；各 PLY ≤16 MiB；总顶点 ≤100000，总面 ≤200000；HTML ≤32 MiB。输出目录须位于项目 outputs 下且不存在。退出 0 表示导出完成，2 输入拒绝，4 内部错误。GUI 状态始终由独立人工/浏览器验收决定，导出不自动声称 GUI PASS。

## 证据边界

本轮真实点云来源仍是已有稀疏重建，含背景、尺度未知、无稠密表面；未重跑 SfM。合成网格虽然是先前通过 FreeCAD/STEP 往返的输入，本页并不导入 STEP 或 FCStd。不增加真实照片网格、CAD、物理精度、跨视分割或端到端通路的完成结论。

现有验证：最终 64 项 Python 读取/导出测试；JS 源码用 DOM/WebGL 测试替身检查控制与提交绘制的数据。后者不是浏览器执行、shader 编译或 GPU 像素验证。真实显示与交互仍待验。
