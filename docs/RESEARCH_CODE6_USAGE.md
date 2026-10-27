# 真实 SfM 的固定六点研究表示

本入口只读取已完成 SfM，不重建、不加载 GPU 模型。调用既有 `indexed_body_code`，按原生 point_id 升序固定取前六点；没有足够点、共面、重复点或预算拒绝都如实保留，不挑选另一组有利点。

依据：Kozlov 1996 §2，PDF p.6–7（期刊325–326）。有限三维点集 A 具有标签集 M_A；所有不同无序四元组之间，非零分母的四面体体积比组成带索引 T_A，K(A)=(M_A,T_A)。零分子保留，不按比值去重。全体共面代码未定义。现有算法以实际 binary64 存储值逐个转 Fraction，算术精确不代表照片恢复精度。

```powershell
& '<项目轻量Python>' -B -X utf8 scripts/export_research_code6.py --sfm-run '<成功原生SfM run.json所在目录>' --model-id '0' --output '<项目outputs内全新研究目录>'
```

API：`integrations.demo_pipeline.research_code6.export_native_code6(sfm_run, model_id, output, max_entries=10000)` 返回 RESULT 字典。读前/后绑定 run.json、points3D.bin 和描述文件；二进制扫描上限32 MiB/100000点，描述JSON64 MiB，只建六点代码，记录上限10000。原始point_id、坐标hex及有理值保存至 SOURCE_POINTS.json；完整代码在 INDEXED_CODE3D.json；RESULT.json含状态、来源/hash、计数、定理域与限制；index.html是独立离线研究卡片。

输出目录拒绝覆盖。退出0仅表示六点代码构建完整，退出1为拒绝、未定义或不足点；退出2为调用/目录错误。JSON hash绑定文件和编号呈现，不是重编号不变量。当前通过检查的来源模型及原生二进制哈希必须保持不变。

统一 `run_photo_demo.py run` 入口现在默认开启侧支，在成功表面与 Blender 后使用表面登记的 selected_model_id 和原始 SfM run 自动执行。参数 `--research-code6` 显式开启，`--no-research-code6` 关闭；report 目标没有成功表面与选定模型时登记 NOT_RUN。研究子进程由现有监督器限制30秒/1 GiB，不加载模型。源 SfM 整体为 PARTIAL 时，只要明确选定模型 usable 与 roundtrip_verified 真实，通过检查即可建码，并保留源 PARTIAL。

研究阶段及报告卡片与几何状态分开；失败、退化或预算拒绝保持真实状态，不销毁成功 Blender 文件。研究下载必须核对当前任务 research/RESULT.json 及其全部产物路径和哈希。旧独立骰子成功运行保留在 `outputs/delegation/GENERAL_PIPELINE/20261011_research_code6_run01/dice_real/`；该旧记录本身不能证明新版本统一入口已实跑。

只有实际统一调用记录才能称该运行使用了研究模块；mock接线验证不替代真实调用。它不实现 AlgT′/AlgS0，不表示整个物体，不解决2025 ostov/futlyar与三类形象，不证明研究重建收益或真实尺度。原算法保持只读。

## 默认自动生成的已知仿射正/负对照

独立旧实验 `CODE3D_INDEXED/20261007T112500Z_main` 已证明另一真实来源六点上的受控正/负例。本轮补齐统一侧支的自动对照，而不重复将历史证据写成新执行。

成功建码后默认生成 `AFFINE_CODE3D.json`、`NEGATIVE_CODE3D.json`、`CONTROLS.json`，全部加入 RESULT 的产物哈希。对照按预先固定的有理上三角矩阵（det=-3）、有理平移、目标标签和重编号变换。用同一已知全局双射逐条验证所有索引体积比；固定第六原生点变换后的 z 精确加1/7作为负对照，不换点或换扰动直到成功。

非等价结论只在完整6!=720双射检查且独立oracle均否定时报告。独立oracle从序列化有理坐标用4×4 Leibniz行列式重算全部四面体体积和完整索引表；不复用算法内部3×3体积计算。预算不足保留 UNKNOWN，意外等价或对照未建立明确记录。表示构建 SUCCESS 与 controls.status 分开，不能把前者冒称后者通过。

API增加可选 `controls=True, max_control_permutations=720`；CLI `--no-controls` 仅建表示，`--control-permutations 1..720` 显式限制负对照搜索，禁止自动提高。现有统一入口默认调用因此自动得到对照文件，无需新增模型阶段。生产外监督仍为30秒/1 GiB。

理论依据仅为1996 §2的索引码定义和期刊326 Lemma3（仿射等价推出代码等价）。矩阵/平移/扰动/有限枚举oracle是项目实验设计，不是作者命名算法；正例对应本来已知，负例只限固定六点，未恢复未知照片对应，未评价重建收益、整个物体或真实精度。
