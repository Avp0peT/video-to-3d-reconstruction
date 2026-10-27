# 显式部署配置创建与复审（PC14）

`scripts/configure_photo3d.py` 是真实配置入口：创建独立 JSON 记录，流式计算已有工具与 checkpoint SHA-256，复用现有封印包校验，保留 PROFILE 来源和批准过的修改。它不启动解释器、Git、模型或原生工具，不安装、下载或联网。大权重哈希可能耗时；存在性检查不能代替内容校验。

用户先在自己的 spec JSON 提供这些字段：`project_root`、`project_python`、`sam3_python`、`sam3_package`、`sam3_source`、`checkpoint`、`checkpoint_sha256`、`git`、`input_kind`、`target`。所有路径必须显式绝对路径，权重摘要必须是已有批准来源的预期 SHA-256。`input_kind` 为 `photos/video`，`target` 为 `report/blender/cad`。video 还必须显式提供 `ffmpeg/ffprobe`；blender 或 cad 必须提供 `blender/mvs_tool_root`；cad 另需 `freecad_python`。源码固定为选定根下 `integrations/sam3/src/sam3-repo`，外部源码不能静默代换。

```powershell
python -B -X utf8 scripts/configure_photo3d.py create --spec D:\spec.json --output D:\project\outputs\deployment-new\CONFIG.json
python -B -X utf8 scripts/configure_photo3d.py audit --config D:\project\outputs\deployment-new\CONFIG.json --config-sha256 <create返回的sha256>
```

配置只能新建，不能覆盖；输出必须在选定项目 outputs 下且与源码、包及 OpenMVS 目录分离。即使依赖缺失或旧绑定阻塞，也保存当前实际阻塞记录，退出码 2；参数非法时 stdout 返回 INPUT_ERROR。静态绑定满足时返回 `STATIC_BOUND_RUNTIME_UNVERIFIED`，退出码 0。配置和依赖漂移均拒绝消费，需要新建记录。API `create_deployment(spec,destination)` 返回记录路径和摘要；`audit_deployment(path,expected_sha256)` 给出当前审核结果；`options_overrides(path,expected_sha256)` 仅在当前静态审核通过时返回 `pipeline.Options` 所需显式参数。调用方继续提供 source、prompt、output 等业务参数，核对素材种类与配置 input_kind 一致，并拒绝与配置冲突的命令行覆盖。该 API 不执行 plan/run。

运行包 BUNDLE 摘要仍必须等于现有 runtime 接受的封印；START_BASELINE/AUTHORITY 的源码哈希、checkpoint 路径/摘要/stat 与环境包清单仍保留约束。checkpoint 字节相同但 stat 或路径不同也不会伪造通过。解释器文件、工具文件、核心产品入口、pyvenv.cfg 与 dist-info METADATA 均在复审快照中。真实 Git HEAD/clean 状态、原生 DLL、CUDA/驱动、模型加载及实际重建仍留给原运行门和整链验收；没有当前运行证据，没有执行授权，没有历史 PASS 继承。

**新项目根可创建配置，但目前不能直接运行：PC14 可搬运执行尚未完成。** 当前 loader 固定代码根、SAM3 环境及运行包位置，authority 另绑定旧根和权重。重绑定必须走现有 PROFILE 的正规审查方式：保留原 source_package、source_bundle_sha256 与逐条 previous/accepted 内容摘要；逐项审查仅部署路径/authority/checkpoint stat 的修改范围，并核验新目录实际内容；生成新封印后单独审查 runtime 接受的 profile/path/BUNDLE digest 更新及 worker 来源绑定；再执行目标机器有界原生整链。此入口会列出 `REBINDING_REQUIRES_REVIEWED_RUNTIME_PROFILE`、根/权重/源码等具体原因，绝不修改旧 AUTHORITY、产生伪授权 BUNDLE 或发布被阻塞的执行参数。若源码内容也变了，必须另行源码审查，不能把修改掩饰为路径迁移。

本次测试：第二批 13 项纯数据测试通过，包括创建/不可覆盖、配置与权重篡改、显式参数、目标条件、保护目录、无 native/network/heavy import、阻塞不可消费及 API 参数映射。参数映射正例为人工静态审核桩，只验证映射，不宣称真实新部署通过。第一批因 pytest basetemp 父目录不存在产生 12 个 setup errors（1 项通过），失败事实保留于独立运行记录。未运行真实 SAM3 或原生工具，未验证干净机器部署。

主线统一入口已消费配置：`run_photo_demo.py run/plan --deployment-config <CONFIG.json> --deployment-config-sha256 <sha256>`，每次先复审完整依赖、素材种类与冲突覆盖项，审核失败不执行。主线24项相关测试通过。本机实际配置位于`outputs/delegation/GENERAL_PIPELINE/20261013_feature_integration/local_deployment/`：实际创建/哈希绑定完成，随后公共`plan`返回0、未创建输出、原生调用0。这是当前根静态路径闭环，不替代新根迁移或运行验收。配置后若受绑定代码变化，须重新创建，不能复用旧审核状态。
