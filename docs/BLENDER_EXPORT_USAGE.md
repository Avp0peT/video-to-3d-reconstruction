# 系统生成 Blender 文件

已有本地 PLY 后，在实验室侧栏打开 **生成 Blender 文件**。填入 PLY 路径，点击 **生成 Blender 文件**。系统在后台创建 `.blend` 并独立回读检查，通过后提供下载；用户可直接在 Blender 打开 `model.blend`，无需手工导入 PLY。

PLY 路径默认空。**使用当前已验收样例**填入本次开放修复曲面；其他输入不继承该样例的 FreeCAD 或原生几何验收。Blender 程序路径可留空，让后端寻找已有安装；找不到时填写 `blender.exe` 的完整路径。系统不会自动安装 Blender 或依赖。

标准范围最多 100000 顶点、200000 面；显式选择 `surface400k` 后最多 200000 顶点、400000 面。两者的单个 PLY 上限均为 16 MiB。不会自动提限、抽面或采样。无面 PLY 按点云生成，三角面按输入保留，孔洞不补齐。

物理尺度未知；CAD 文件的 mm 约定不建立物理尺寸。当前样例来自 component0 经 QEM 简化的照片重建，删除 4 面产生的缺口是 `repair_created_excision`，不代表原物体真实孔洞。Blender 导出检查不证明照片保真、物理精度或参数化工程 CAD。

## 命令行入口

从项目根目录运行。以下沿用已有、绑定本次真实 PLY 摘要的场景说明；输出目录必须不存在：

```powershell
python -B -X utf8 scripts/export_blender.py --spec outputs/delegation/PROJECT_FINISH/20261011_open_surface_viewer_C/SCENES.json --output outputs/blender/manual_open_surface/export
```

指定已有 Blender 安装：

```powershell
python -B -X utf8 scripts/export_blender.py --spec outputs/delegation/PROJECT_FINISH/20261011_open_surface_viewer_C/SCENES.json --output outputs/blender/manual_open_surface_custom/export --blender-exe "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --profile standard
```

其他 PLY 可通过页面自动生成场景说明，或自行准备 `ply-viewer-scenes-v1` JSON：

```json
{
  "schema": "ply-viewer-scenes-v1",
  "scenes": [{
    "ply": "C:/data/input.ply",
    "sha256": "替换为该文件实际的64位小写SHA256",
    "title": "我的模型",
    "data_kind": "unspecified",
    "declared_unit": null,
    "note": "本地PLY；尺度未知，保留输入孔洞。"
  }]
}
```

支持的来源声明为 `real_photos`、`synthetic`、`unspecified`；未知尺度用 `declared_unit: null`。来源声明不会自动成为验证结论。场景 spec、输入 hash 和 profile 由已有 PLY 数据读取层核验。

## 结果与失败

页面每次建立独立任务目录 `outputs/blender/<UTC时间_uuid>/`，其中保存 `SCENES.json`，导出目录为 `export/`。后端只有创建与独立回读通过才返回 `status=SUCCESS`；成功的 `blend` 字段包含实际文件路径、SHA256、字节数，`scenes` 包含顶点与面数。页面另核对下载文件摘要和字节数，不能仅凭文件存在提供成功下载。

失败时显示错误，任务及报告保留供排查；页面异常另保留 `UI_ERROR.json`。未通过时不提供 `.blend` 下载按钮。后台导出只接收已绑定的 PLY 场景数据，不运行用户提供的脚本；不自动执行 SAM3、MVS、CAD 或照片/视频流水线。

此入口新增独立 PLY → Blender 输出步骤，不代表既有照片/视频主流水线已经串接 Blender。页面或 CLI 的真实导出验收与静态检查应分别记录。

## 当前实测

2026-10-11：本次开放修复PLY经系统API导出成功。Blender 5.1.2后台创建与独立回读均通过，源文件摘要保持。输出 `outputs/delegation/PROJECT_FINISH/20261011_blender_A/export01/model.blend`（331587字节），5000顶点、9996面、12边界边，本样例float32坐标误差为0；不代表其他双精度输入均无损。报告在同目录 `RESULT.json`。

Streamlit侧栏的 `10_Blender_Export`（生成 Blender 文件）可自行选择本地PLY生成新文件；默认输入空，当前样例按钮只填路径。页面AppTest验证了输入与下载门禁、rerun保持；真实后台导出由API验收，原生GUI交互未验证。这是独立PLY导出入口，照片至mesh主流程未自动串接。
