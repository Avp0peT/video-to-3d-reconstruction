# 照片 / 视频三维环境预检（PC14）

此入口只读检查部署位置，不执行照片处理、抽帧、SAM3、SfM、OpenMVS、Blender 或 FreeCAD，也不安装、下载、联网或查询设备。它独立于 `pipeline.plan`：现有 plan 还负责素材与执行计划；本预检补充独立机器 JSON、目标必要项分类、静态身份来源及已知迁移绑定阻塞。

```powershell
& '<已有Python路径>' -B -X utf8 '<代码目录>\scripts\check_photo3d_environment.py' --project-root 'D:\work\project with spaces' --input-kind photos --target blender --python 'D:\envs\light\Scripts\python.exe' --sam3-python 'D:\work\project with spaces\integrations\sam3\env\Scripts\python.exe' --checkpoint 'D:\models\sam3.pt' --blender-exe 'D:\tools\Blender\blender.exe' --mvs-tool-root 'D:\tools\OpenMVS'
```

`--project-root` 必填；相对参数以该根目录解析，与终端 cwd 无关。可选路径参数还有 `--sam3-package`、`--git-exe`、`--ffmpeg`、`--ffprobe`、`--freecad-python`。`--input-kind` 为 `photos` 或 `video`；`--target` 为 `report`、`blender`、`cad`。默认 stdout 为 JSON；`--format text` 输出状态、路径和下一步。脚本不写结果文件；用户可通过 shell 重定向保存。入口禁止写 bytecode。

默认的项目 Python、SAM3 和运行包、OpenMVS、FreeCAD 路径来自当前产品已有配置，并以选定根目录拼接。模型 `C:\sam3\sam3.pt`、Blender 本机路径和 PATH 发现标记为 **NONPORTABLE**；这是本机缺省，不是跨机器部署承诺。请在目标机器显式提供路径。

| 必要组件 | 照片 report | 视频 report | blender | cad |
|---|---|---|---|---|
| 项目标记与基本入口、项目 Python、SAM3 Python、模型、Git、三个运行包元数据文件 | 必需 | 必需 | 必需 | 必需 |
| FFmpeg / FFprobe | 可选 | 必需 | 仅视频必需 | 仅视频必需 |
| OpenMVS 三个可执行文件及表面入口、Blender | 可选 | 可选 | 必需 | 必需 |
| FreeCAD Python 与 CAD 入口 | 可选 | 可选 | 可选 | 必需 |

`MISSING`、`WRONG_TYPE`、`EMPTY_FILE`、`UNREADABLE` 是必要文件的静态阻塞；可选项使用 `OPTIONAL_` 前缀，不阻塞所选目标。JSON 的 `missing_required` 收集阻塞名称，每条提供下一步。程序不检查素材或输出目录容量，因为本预检不准备执行任务。

证据解释：

- `PRESENT` 只表示当前文件存在且非空；大小与 mtime 是 stat 快照，不是内容摘要，也不证明可执行、可加载或版本兼容。
- Python 的 `pyvenv.cfg` 版本与指定环境内 dist-info 包版本为 `STATIC_METADATA`，不运行解释器、不导入包。配置版本不是当前实际运行的二进制版本；不存在配置时版本为 null。原生工具版本为 null，未运行 `--version`。
- 运行包只读小于等于 1 MiB 的 BUNDLE、AUTHORITY、BASELINE JSON，展示 schema/profile 和已记录根/模型绑定。未核验全部文件哈希、完整运行包清单、SAM3 checkout、checkpoint digest 或环境包清单与 authority 的一致性。这些留给已有运行门；本预检不可替代它。
- 当前 worker 强制固定的 SAM3 解释器/运行包相对路径。显式覆盖到其他位置产生 `RUNTIME_PATH_CONTRACT`；旧元数据绑定另一个项目根产生 `BOUND_TO_OTHER_PROJECT_ROOT`，模型路径改变产生 `CHECKPOINT_PATH_BINDING`。先重新建立并审计目标机器绑定，不能复制历史 PASS。
- `current_runtime_evidence=[]`、`runtime_compatibility=UNVERIFIED`、`historical_pass_reused=false` 永远如实保留。预检没有导入历史原生执行日志，也不声称新环境已运行。

退出码 `0` 表示选定的静态项目齐全且没有检测到已知绑定阻塞，状态为 `STATIC_ITEMS_PRESENT_RUNTIME_UNVERIFIED`；`2` 表示 `STATIC_BLOCKED` 或 argparse 参数错误。零退出码不表示照片三维流程成功，也不表示干净机器部署通过。

迁移步骤是：准备批准的既有依赖和路径；运行预检并解决缺失与绑定阻塞；再按 [PHOTO_DEMO_USAGE.md](PHOTO_DEMO_USAGE.md) 检查真实素材计划，在目标机器完成有界原生整链和产物独立回读。需要另行验证 CUDA/驱动、内存、模型加载、重建、照片/视频外形和尺度等。本机旧成功案例不能替代该步骤。

本次证据：[只读本机 video/cad 预检](../outputs/delegation/GENERAL_PIPELINE/20261012_deployment_preflight/PREFLIGHT.json)、[执行及测试记录](../outputs/delegation/GENERAL_PIPELINE/20261012_deployment_preflight/EXECUTION.json)。7 项纯数据测试一批通过，未执行原生工具。真实预检只运行一次，本机全部列出的必要文件存在、未检测到已知根/模型路径绑定阻塞；运行兼容性未验证，**干净机器安装验收尚未完成**。
