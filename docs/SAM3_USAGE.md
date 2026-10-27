# SAM 3 目标分割接入与使用（object-mask-v1）

## 2026-10-08 pilot06 后的状态与采样语义

本节覆盖本文历史轮次的“当前/本轮”状态措辞；历史安装失败和旧测试保留为当时记录。真实环境已能加载SAM3及生成掩膜。pilot06的后端segment返回2次，但只有idx17提交完整产品frame事件；idx34是开始后被资源看守中断的未提交帧，idx59未开始。不得把磁盘中间mask或审核wrapper的segment_complete当作已提交的selected_mask，旧RUN原样保存。

当前父层保存独立的帧开始证据。最终drain接受的完整frame事件优先；仍无结果的已开始帧记RESOURCE_LIMIT（首因为资源）或BACKEND_ERROR，并记FRAME_RESULT_UNCOMMITTED、空候选/空选中身份、not_run_reason=null。只有真正未开始的余项记NOT_RUN。不会从孤立文件补造选择身份，已完成帧仍保留真实终态。

当前RAM判断在设备查询之前进行：本轮RAM读取完成后发现越限/不可测即锁存，触发既有监督器回收；随后仍保留原有一次有界设备查询及其失败证据，不能覆盖首因。不再等待该查询完成才裁决RAM。这不保证跨进程原子停机或任意短峰值捕获，不能据此承诺锁存与进程退出之间没有调度延迟。

资源记录新增parent_ram_sample_interval、各target.ram_sample_interval、ram_phase_interval、device_sample_interval，均为主机读/查询调用的单调起止区间；未调用的区间为null。ram_stop_decision_monotonic是本次RAM裁决记录时刻，公共t仍是完整观察结束时刻。设备自身query区间继续保留，age不是设备测量时间戳。0.25秒是循环等待间隔，不是实际采样周期；同步查询耗时也计入实际周期。父层设备freshness/timeout均5秒，gate A独立审核查询按其冻结配置单列。

这两项为状态正确性和RAM停止时序修复，RAM12GiB/整卡used7GiB及线程、模型、输入配置未放宽。仅离线验证本轮修复；不声称降低内存、已完成三帧或已定位分配根因。当前下一项是分阶段诊断包准备，详见`docs/delegation/20261008_PILOT06_REVIEW/`，不自动重跑模型。

日期：2026-10-07（Asia/Shanghai）。pilot02已完成真实本地权重加载及GPU后置检查，首帧进入segment后发生BF16/Float错误，0完成帧、0 mask。随后主线补入CUDA BF16 autocast及错误分类修正；新版本真实前向仍待验证，见 `outputs/delegation/SAM3_REAL3_DTYPE_REPAIR/20261007T072707Z_main/HANDOFF.md`。以下A-R1至A-R6记录保留各自历史验证边界，不代表当前版本整体验收。
本文件不构成 SAM 3 效果、mask 质量、跨视一致性或下游重建的结论。

当前设备规则：CPU的显存通道为NOT_APPLICABLE，RAM与进程身份仍检查；CUDA默认auto观察，显式none在创建worker前拒绝。job保留cuda:N，驱动采样使用已解析的设备索引；缺失、非有限、负值或过期的必需读数不视为健康。pilot02的cuda:0和驱动UUID已有真实证据，其他映射不得由此类推。

设备新鲜度接线修正（2026-10-08）：sampler 现在显式把已解析的 `freshness_seconds` 传给驱动查询；没有完整 policy 时明确使用 query timeout。默认声明与 query timeout 都仍为 5 秒，但旧 sampler 漏传参数，使实际查询按函数隐式 2 秒裁决。此次将 sampler 的**实际有效阈值从 2 秒对齐到既有声明的 5 秒**；不是“阈值行为完全不变”。直接调用 `driver_device_query` 的历史默认仍是 2 秒，调用者可显式指定。RAM 12 GiB、VRAM 7 GiB 和 query timeout 5 秒均未提高。

每次驱动查询记录实际 `freshness_seconds`、`query_timeout_seconds`、`query_rc`、单调起止时间与 `query_elapsed_seconds`；超时、命令失败、解析失败也保留时间，无法取得退出码时为 null。兼容字段 `age_seconds` 等于未四舍五入的查询耗时，表示保守查询窗口，**不是驱动提供的显存测量时间戳**；不以舍入把略超界读数变成通过。错误、`fresh=false`、无效 age、用量越限或设备身份不符仍拒绝；合法的使用量 0 不等于查询失败。`observe()` 的汇总 t/utc 仍为整次观察完成时间，各 RAM/设备通道未分别加时间戳，不能由它证明通道同时采样。

pilot05 的 `3.969s / freshness_seconds=5 / fresh=false` 可由上述旧接线确定复现；查询慢或使用量为 0 的机器原因尚未证实。本次仅以假时钟和假 subprocess 验证新接线、时间记录和拒绝规则，证据入口 `outputs/delegation/SAM3_DEVICE_FRESHNESS/20261008_main/HANDOFF.md`；不代表真实 GPU 再验证、真实数组转换收益或后续模型运行成功。

当前推理上下文：CUDA在同一有限作用域内以 `torch.inference_mode()` 和 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 覆盖 `set_image` 及 `set_text_prompt`；正常或异常退出均恢复上下文，不全局启用autocast、不改权重dtype。CPU仅进入inference_mode，未据此主张CPU真实推理可用。有效选项与配置摘要同时绑定构造和推理策略。只有真实CUDA OOM异常类及其子类转为RESOURCE_LIMIT；普通dtype/IO/参数错误保留BACKEND_ERROR及原异常原因。worker逐帧资源异常保留RESOURCE_LIMIT/退出5，后续帧停止。

当前收尾规则：进程创建后的初始化、等待、事件读取均有异常清理；回收与线程join共享一个固定截止。所有已登记成员参与退出裁决。终止调用结果与实际退出分别记录，无法确认的身份/成员/枚举结果阻止普通成功；只使用本任务持有的身份句柄，线程停止后才释放。超时、失权或未登记的短命后代仍可能只能报告UNKNOWN，这不是硬资源限额或完整进程树覆盖的实测承诺。

A-R6只执行冻结的离线TARGETS，测试驱动阻止真实子进程。不要为了引用这次结果直接运行整个测试文件：其中包含历史真实进程场景，这些本轮NOT_RUN。

> **A-R1 更正（2026-10-07）**：本文件经离线返工修订。旧措辞把权重访问写成"当前阻塞项"、暗示"登录后即可复跑"，
> 该说法**不成立**——权重访问、torch 安装/CUDA 探针、资源看守与代码验收是**四件互相独立的事**，任何一件未完成都不能开始真实推理。
> 旧日志与旧报告未被改写；更正与新增说明见 `outputs/delegation/SAM3_A_R1/20261006T171557Z/ERRATA.md`。
> 真实父目录是 `outputs/delegation/SAM3_A/20261006T163505Z/`，`RUN/HANDOFF.md`、`RESULT.json`、`BINDINGS.json`、`tests/` 都在该目录下。

> **A-R2 更正（2026-10-07）**：A-R1 的"五类缺陷全部关闭"**不成立**：指定反例有离线修复证据，但**运行控制未接入产品路径**。
> A-R2 已接合并见 §0a；旧措辞更正逐条列于 `outputs/delegation/SAM3_A_R2/20261006T174308Z/ERRATA.md`。

> **A-R3 更正（2026-10-07）**：§0a 关于运行控制的四项陈述当时**只做到一半**，以下三条为更正后的正确行为，逐条更正见
> `outputs/delegation/SAM3_A_R3/20261006T182953Z/ERRATA.md`：
> ① **停止回调拿到的是真实进程**。所有权（`WorkerOwnership`）在进程一存在时创建并立即发布，等待循环可被唤醒打断，
>   因此 load/frame 截止、越限、监控失败都会**当场**停止并回收，而不是等整批超时；回收幂等、截止只定一次。
>   身份绑定 OS 创建信息，子进程自报 pid 另记且**不假定相等**；顶层退出**不等于**整树无残留，存活后代单独回收，无法确认者记 `UNKNOWN`。
> ② **worker 工作期间持续观察**。父层 `WorkerObserver` 在 load/segment 阻塞时从 OS 读取**worker 自身**的 commit，
>   与父进程/设备维度分列；不可测/查询失败/已观察越限一律锁存 `RESOURCE_LIMIT`/退出 5，后续帧不再启动。
>   父层监控改为在权重解析之后构造且 `import_torch=False`，**父层不再隐式导入 torch**。
> ③ **最终事件与清理事实进入裁决**。实时读取与最终 drain 共用同一个 `_EventLedger`（按写入方单调 `seq` 去重），
>   处理全部事件类型；坏记录在写入期可容忍、退出后**明确报错**；`close` 错误使"看似正常完成"降级为非 SUCCESS。

## 0a. 运行控制（A-R2）：产品路径、锁存、回收

- **产品执行路径**：CLI 默认 `use_worker=True`。父层（`runner`）持有预算、停止锁存、阶段记录与 manifest；
  **重型模型加载与每一帧调用都在本任务的有界 worker 子进程内**，且**每批只启动一个** worker —— 单次加载服务整批，不为每帧重复加载。
  权重解析仍在父层、且在任何重型导入**之前**完成。`--in-process` 只用于确定性接口测试，manifest 里以 `execution_mode` 区分。
- **不可撤销停止锁存**：采样结果**先检查、再更新** `stop_state`；父层监督线程、子进程、采样线程、截止时间任一先观察到原因即锁存，
  **首个原因为准**。13 GiB 后回落到正常值**不会**解除锁存；锁存后**不再启动任何帧**；已完成帧保留真实状态，未启动帧 `NOT_RUN` + 原因。
  run 为 `RESOURCE_LIMIT`/退出 5。辅助清理报错**不会**覆盖真实停止原因。
- **截止与看守**：加载/单帧/批次预算**共享同一个单调起点**，后一阶段不重置总截止；父层监督线程以短周期读取子进程事件文件，
  在加载或某帧尚未结束而预算已过时**主动终止**该 worker 自己的进程树（`taskkill /F /T` 只针对本任务启动的 PID）。
  `--load-budget-seconds` / `--frame-budget-seconds` / `--batch-budget-seconds`（默认 300/300/900）。
- **有界回收**：不使用无界 `communicate()`；子进程 stdout/stderr 落文件；等待、管道、kill 共用一个回收截止（`Budgets.reclaim_seconds`）。
  **未证实退出**的成员记为 `residual_unknown: true`（UNKNOWN/残留），**不因 kill 调用成功就当作已退出**。
- **显式探针**：CLI **绝不**隐式执行环境/CUDA 探针（隐式执行等于绕过监督）。需要时用 `--env-probe`，
  它作为**显式有界前置步骤**在独立 worker 内运行（60s 预算），结果记入 `host_probe`；默认该字段为 `NOT_RUN`。
- **RAM 覆盖范围**：worker 模式下令牌同时覆盖父进程与 worker 自身（子进程每帧上报自身提交量），
  不会只测"空闲的父进程"；容量/使用量/torch 分配仍按 §5 分开记录。
- **收尾可靠**：后台采样线程的停止标志命名为 `_stop_event`，避免覆盖 `Thread._stop`（Python 3.12 会因此 `join()` 抛 `TypeError`）；
  join/close/通道查询失败均结构化记录；末尾的通道查询只做一次并包在 guard 内，**查询抛错也照样写出 manifest**（用最后一次已知快照/未知）。

权威 schema 是 [contracts_object_masks_v1.md](contracts_object_masks_v1.md)；本文件只讲怎么用，不复述字段定义。
选型与分工见 [SAM3_INTEGRATION_PLAN.md](SAM3_INTEGRATION_PLAN.md)。

## 0. 真实推理的四个独立前置条件（缺一不可）

| 条件 | 当前状态 | 说明 |
|---|---|---|
| 本地权重 | **已绑定并加载** | pilot02使用 `C:\sam3\sam3.pt`，完整SHA256已核对；本地路径不依赖HF登录 |
| torch / torchvision | **已安装并验证导入** | 专属环境2.11.0+cu128 / 0.26.0+cu128；不据旧AUTH_REQUIRED记录重装 |
| CUDA及GPU模型加载 | **pilot02 PASS** | 同一worker的CUDA小算子、UUID及1135参数/缓冲设备检查通过；不等于完整SAM3前向通过 |
| 新版推理接线 | **离线验证，真实待验** | 缺失autocast、过宽OOM匹配及worker错误传递已修正；以新RUN交付为准，真实mask/质量仍无通过证据 |

四项全部具备之前，真实推理（`--device cuda` + 真实权重）都不会启动，也不应启动。

## 1. 目录与环境

| 内容 | 位置 |
|---|---|
| Python 3.12 解释器（独立） | `integrations/sam3/python/cpython-3.12.13-windows-x86_64-none/` |
| 虚拟环境 | `integrations/sam3/env/`（解释器哈希见 `integrations/sam3/VERSIONS.json`） |
| SAM 3 源码（固定 commit） | `integrations/sam3/src/sam3-repo/` @ `2345a4ad109ac29c569da749c91d84f10dc08c40` |
| 适配器包 | `integrations/sam3/sam3_adapter/`（`contract` / `imagery` / `sourceio` / `backends` / `binding` / `resources` / `deadline` / `worker` / `runner` / `probe`） |
| CLI | `scripts/segment_sam3.py` |

与既有 SFM 环境（Python 3.13 / NumPy 2.5.3）**完全隔离**：SAM 3 侧要求 `NumPy<2`，且适配器与测试**都不导入 torch / sam3**（有专门测试守护这一点），
因此 `tests/test_sam3_adapter.py` 可在项目环境与隔离环境两处运行。

线程/字节码策略在**进程启动时、任何重依赖导入之前**设置：`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`NUMEXPR_NUM_THREADS`=2、`PYTHONDONTWRITEBYTECODE=1`。

## 2. 条件一：权重访问（HF 门控）

官方权重 `facebook/sam3`（`sam3.pt` + `config.json`）是**门控仓库**。本轮探测结果：无 HF token，`hf_hub_download` 返回 **401**。
如需自行完成这一步（可选，且必须由账号本人操作）：

1. 在 <https://huggingface.co/facebook/sam3> 提交访问申请并接受条款；
2. 在能访问该账号的环境里登录，例如 `hf auth login`（token 只存本机，不要写进仓库或交付物）。文档：<https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication>。

注意口径：**401 只说明匿名请求被拒，不能推断该账号一定未获批**；本项目不使用第三方镜像绕过访问限制。
鉴权失败时 CLI 会写出 `status=AUTH_REQUIRED`、`counts.not_run=requested`、退出码 3 的 manifest，这是正确行为，不是缺陷。

即使权重到手，**仍需**满足 §0 的另外三项（torch 安装、CUDA 探针、代码验收），才能开始真实推理。
建议优先走**本地权重**路线并显式锁定哈希：`--checkpoint <ABS>/sam3.pt --checkpoint-sha256 <sha256>`。
HF 路线固定 revision 与缓存目录：`--hf-revision <commit> --hf-cache-dir <ABS>`（默认即冻结 commit，不用滚动 `main`）；同一轮只解析一次，不重复下载探针。

关于配置：`build_sam3_image_model` **不读取** `config.json`（架构由代码常量决定），因此 manifest 里的 `config_sha256` 记录的是**实际生效的构造选项**
（代码 commit、BPE 资产哈希、device、resolution、confidence threshold、load 标志等）的规范 JSON 哈希；
`--config` 传入的文件会在 `backend_bindings.config.declared_config_file` 中单独登记并明确标注 `declared_config_file_consumed: false`，
**不会**被冒充成"已验证配置"。

## 3. 用法

```bash
# 1) 环境与设备探针（不含权重）
integrations/sam3/env/Scripts/python.exe -m sam3_adapter.probe --mode all \
    --repo-dir integrations/sam3/src/sam3-repo \
    --allow-hf-download
# 需在 integrations/sam3 位于 PYTHONPATH 时运行

# 2) 真实分割（三帧批次，batch_size=1，顺序执行）
integrations/sam3/env/Scripts/python.exe scripts/segment_sam3.py \
    --io-run  outputs/delegation/T01A/CAD_FULL66/20261006T153820Z_full66/20261006T153857Z_5c3795b195b6 \
    --indices 17,34,59 \
    --object-id skull-01 --text-prompt skull \
    --checkpoint <ABS>/sam3.pt --checkpoint-sha256 <sha256> \
    --output outputs/delegation/SAM3_A/<RUN>/mask_runs \
    --device cuda
```

关键参数：

- `--checkpoint` + `--checkpoint-sha256`：**推荐**。给出本地权重并校验哈希，模型/配置成组锁定，不依赖滚动 `main`。
- `--allow-hf-download`：未给 `--checkpoint` 时，尝试用本机既有 HF 登录解析 `facebook/sam3`。鉴权失败即 `AUTH_REQUIRED`（退出 3），**不循环重试**。
- `--select 17:c01`：多候选时显式指定候选，记为 `explicit_candidate` / `source=cli_argument`。不提供时多候选一律 `NEEDS_SELECTION`。
- `--backend fake --fake-spec <json>`：**仅**用于接口测试，manifest 标 `evidence_level=fake_backend_interface_test`，不得当作 SAM 3 证据。
- `--no-host-probe`：跳过设备探针（真实运行前不建议跳过）。
- `--device cpu|cuda|cuda:N`：设备索引被显式解析并**实际**用于查询（不再固定查 0）；无法解析即拒绝。
- `--load-budget-seconds` / `--frame-budget-seconds` / `--batch-budget-seconds`（默认 300/300/900）：从**同一个**单调起点派生，后一阶段不会重置总截止时间；超时计 `RESOURCE_LIMIT`（退出 5）。
- `--visual-prompt` 等**未实现**选项会被**显式拒绝**（退出 2），不会"记录为已使用却实际忽略"。
- 参数越界、重复 `--indices`、`--select` 中不属于 `--indices` 的索引或非法候选号，一律按输入错误拒绝（退出 2），不会落成未捕获异常。

退出码：`0` SUCCESS、`1` PARTIAL、`2` INPUT_ERROR、`3` BACKEND_UNAVAILABLE / AUTH_REQUIRED、`4` BACKEND_ERROR、`5` RESOURCE_LIMIT。
输出目录已存在或不可写时，直接向 stderr 输出 `REFUSED: ...` 并退出 2，**不写任何 manifest**。
任何**异常**路径都会尽力留下 manifest（含未启动帧的 NOT_RUN 与原因），不会只留一个空目录。

## 4. 输出与语义

`<output>/<run_id>/`：`manifest.json`、`masks/<index>_c<NN>.png`、`overlays/<index>.png`（仅目视，非权威 mask）。

必须知道的语义，避免把预测读成真值：

- **分数**：SAM 3 返回的是 `sigmoid(pred_logits) × sigmoid(presence_logit)`，是排序用的置信代理，**不是概率、不是 mask IoU**；manifest 里以 `score_semantics` 原文保留。
- **内部 resize**：`Sam3Processor` 先把输入缩放到 **1008×1008 的正方形网格（不保持长宽比）**再前向；mask 由模型自身双线性重采样回 canonical 尺寸，并以 **`logits > 0.5`** 二值化。适配器**不做任何额外的几何变换**，因此不存在"外部变换用最近邻"的问题；记录在 `backend_notes`。
- **上游张量契约（R1 修复）**：官方处理器返回的是 **N×1×H×W** 布尔张量。适配器会**显式校验四维**（秩、`channels==1`、空间尺寸等于 canonical、分数长度等于候选数），**去掉唯一通道维**后再逐像素生成二值 mask；标签必须是布尔或有限 `{0,1}`，`masks_logits` 中的 NaN/Inf 一律拒绝。**绝不**对嵌套结构整体取布尔值。违反契约按 `BACKEND_ERROR` 记录，不降级成 `NO_MATCH`。
- **分数契约**：`score` 只能是 `null` 或**有限**原始数值；NaN/Inf 明确拒绝（该帧 `BACKEND_ERROR`），manifest 使用严格有限 JSON（`allow_nan=False`），**不会**静默改 0 或丢弃候选。
- **空前景候选**：零前景候选会被导出供复核，但**排除在选择之外**，所以被选中的 mask 永远非空（`candidates_dropped_empty` 记录数量）。
- **不取最大**：多候选绝不以"面积最大/分数最高"隐式选定。
- **叠图不是 mask**；`quality_status` / `cross_view_status` 恒为 `NOT_EVALUATED`。
- 本契约沿用 `photo_io_canonical` 图像域；去畸变消费者必须另行登记变换，不能把本 mask 直接标为去畸变域。

字段补充（在契约必需字段之外新增，均为附加信息）：`frames[].candidates_dropped_empty`、`evidence_level`、`backend_bindings`、
`backend_notes`、`resource_policy`、`resource_observation`、`deadlines`、`upstream_recheck`、`host_probe`、`execution`、
`backend.backend_unavailable_reason`（契约要求）、`backend.evidence_caveat`（仅 fake）。

### 4a. 转换层：数组路径及列表兼容路径（A-ARRAY 主线收口）

`candidates_from_processor_output()` 按输入句柄及 dtype 选择转换路径；离线验证只覆盖列出的行为，不能当作全部输入的等价性形式证明：

- **数组路径（快路径）**：支持的 bool/整数/常规浮点数组（CPU torch 张量、`ndarray`），契约检查与字节生成**在数组层面完成**，
  **不会**为整张 plane 或整份 `masks_logits` 构造 Python 浮点嵌套列表。
- **嵌套列表路径（兼容路径）**：列表型替身、object 数组及其他不在快路径内的 dtype 沿用原逐元素检查。object 中的合法数字以及 logits 中的 None 保留旧语义；非法值仍被拒绝。

两条路径共同保证：

- **有限性先于任何整型转换**：先判有限性、再判精确 `{0,1}` 归属，最后才生成连续 `uint8`。**绝不**先 `astype(uint8)`——
  那样会把 NaN 静默变成 0，等于把非有限标签洗成背景像素。`masks_logits` 的有限性检查**不被绕过**。
- **字节序一致**：数组路径输出 C 序（行优先）连续字节，与嵌套路径逐行逐列 append 的顺序相同；0→0、否则→255。
- **BF16 的安全路线**：仅明确 `torch.bfloat16` 的 `.numpy()` 抛 `TypeError` 时才允许**无损**加宽到 float32
  （bfloat16 是截断的 float32，加宽是精确的）后再做检查。若该句柄无法加宽，则回退嵌套路径，**不改变分割阈值或图像尺寸**。
- **非连续输入**：支持（数组层用 `ascontiguousarray` 归一化后再取字节）。
- **logits 布尔值**：非空布尔 logits 与旧列表路径一样拒绝；空数组不含非法叶子，保留旧接受行为。
- **不变量**：`N×1×H×W`、通道维、空间尺寸、候选/分数长度、候选顺序（`c00, c01, …`）、`score_semantics`、空候选返回 `[]`、
  异常类型（`TensorContractError` / `ScoreContractError`）、以及**输入不被修改**，全部保持不变。
- **导入纪律**：NumPy **仅惰性导入**，且只在句柄确实能给出数组时才导入；CLI 帮助/无后端路径仍不引入重后端。

旧 `SAM3_ARRAY/20261008T055350Z_arrayconv` 留有失败测试、未执行的最终测试修改，并把额外 3 个导入探针排除于入口计数；其“残余风险仅一次分数比较”与“全部等价”结论不采纳。新收口证据见 `outputs/delegation/SAM3_ARRAY_MAIN/20261008_closure/HANDOFF.md`，不能覆盖或消除上述历史缺口。内存占用、真实前向、mask 质量及跨视效果均不由转换测试证明；本节不作“修复泄漏/必降峰”声明。

## 5. 资源、峰值与停止阈值

预注册阈值：**RAM 提交 12 GiB、VRAM 7 GiB、CPU 线程 2**（`resource_policy`）。监控是**软观测**，**不是**硬限额，
manifest 里 `enforcement=soft_observed_not_hard_capped`，并记录采样间隔与覆盖范围（加载期后台线程 + 每帧一次显式采样）。

口径必须分清（`resource_observation.channels.coverage` 逐项写明）：

| 通道 | 覆盖范围 |
|---|---|
| `process_commit_bytes` | **仅本任务进程**；Windows 提交量（`PagefileUsage`），不是 RSS/USS |
| `process_rss_bytes` | 仅本任务进程；次要指标，**绝不**拿来充当提交量 |
| `device_total_bytes` | 整卡**容量**；仅供参考，**永不与预算比较** |
| `device_used_bytes` / `device_free_bytes` | 整卡使用/空闲（含其他进程），因此是本任务的上界 |
| `torch_allocated_bytes` / `torch_reserved_bytes` | 仅本进程 torch 分配/缓存分配器预留 |

- **容量大于预算不是违规**；只有**实测使用量**越过预算才算。
- 峰值取自**整个采样序列**（`peaks`），不用"最后一次采样"代表峰值。
- 采样缺失或采样失败**明确停止**，不得解释为"未超限"。
- 无法证明监控通道或余量时（`can_prove_headroom` 为假，含设备容量之外的任何不可测情形），真实推理**不启动**，报 `RESOURCE_LIMIT`。
- **任何时刻**观察到越限（最后一帧之后、首帧之后、或加载阶段）都会使 run 变为 `RESOURCE_LIMIT`/退出 5，**已完成帧保留真实状态**，未启动帧置 `NOT_RUN` 并写明原因。
- 截止时间与看守：`deadline.py` 从一个单调起点派生存活预算；`worker.py` 是对**子进程**的硬性看守（超时后只终止**本任务自己启动的那棵进程树**，不做全量 python 终止）。

## 6. 测试与证据层级

```bash
# 项目环境（3.13.14 + pytest 9.1.1）
<project venv>/Scripts/python.exe -m pytest tests/test_sam3_adapter.py -p no:cacheprovider -rf --basetemp=<RUN>/tests/pytest_tmp
# 隔离环境（3.12.13，不加载项目 conftest）
integrations/sam3/env/Scripts/python.exe -m pytest tests/test_sam3_adapter.py --noconftest -p no:cacheprovider -rf --basetemp=<RUN>/tests/pytest_tmp_iso
```

R1 结果：两环境各 **66 passed / 0 failed / 0 error / 0 skipped**（8.52s + 9.75s，合计 ≤180s）；
另有 5 项**变异探针**全部被检出且源文件按字节还原（证明这些测试**能失败**，不只是能通过）。

定向覆盖（原 27 项 + R1 新增）：处理器四维契约（2×3 非全前景、多候选、零候选、错误通道、错误空间尺寸、分数长度不符、非二值/非有限/非数值标签）、
真实 `Sam3Backend.segment()` 路径的逐像素与 PNG 文件哈希、容量 vs 使用量、设备索引实际生效、峰值取自全序列、采样不可用即拒、
越限后的最终状态（首帧后/末帧后）、加载/帧/批次截止、子进程看守、本地权重缺哈希即拒、申明配置标为未消费、HF 仓库/revision/文件/缓存固定且单轮只解析一次、
非对象/损坏/缺失上游输入、非法分数、上游 hash 漂移、CLI 未实现选项与参数范围拒绝、线程策略为**赋值**而非继承。

A-ARRAY-OFFLINE 新增离线等价测试 `tests/test_sam3_array_conversion.py`（不加载 SAM 3 / CUDA / 权重）：

```bash
# 轻量环境：无 torch，覆盖列表兼容路径、NumPy 数组、数组路径不调用 tolist
<light venv>/Scripts/python.exe -B -m pytest tests/test_sam3_array_conversion.py -p no:cacheprovider --basetemp=<RUN>/tmp/pytest_basetemp
# 隔离环境：真实小 CPU torch 张量，必须隐藏 CUDA
CUDA_VISIBLE_DEVICES=-1 SAM3_ARRAY_EXPECT_CUDA_HIDDEN=1 SAM3_ARRAY_REQUIRE_TORCH=1 integrations/sam3/env/Scripts/python.exe -B -m pytest \
  tests/test_sam3_array_conversion.py -p no:cacheprovider --basetemp=<RUN>/tmp/pytest_basetemp
```

覆盖：独立写死的期望字节、与旧转换路径的逐候选对照（指纹含 id/bytes/尺寸/分数/语义）、
bool/int/float/BF16 各 dtype、BF16 无损加宽路线、非连续输入、空候选、多候选顺序、
错秩/错通道/错空间尺寸/分数长度不符、NaN/Inf/非二值/非数值、输入不被修改、
数组路径**不调用** `tolist()`（带"守卫确已武装"的反向对照），以及"本进程未创建 CUDA 上下文"。

证据层级必须分开写，不可互相替代：

1. **源码审阅**（固定上游 `sam3_image_processor.py` / `model_builder.py`）；
2. **mock processor**（StubTensor/StubProcessor 驱动真实 `segment()` 路径）—— 只证明控制流与格式；
3. **假 worker**（3 个 ≤4 秒的有限寿命子进程）—— 只证明看守逻辑；
4. **真实 CUDA / 真实 SAM 3 推理** —— 本轮 **NOT_RUN**（未安装 torch、未联网鉴权、0 次真实前向）；
5. **人工外形检查**、**跨视一致性**、**下游重建** —— 本轮 **NOT_EVALUATED**，绝不写成 CAD 完成。

## 7. 本轮已知缺口

- torch / torchvision **未安装**：cu128 轮子（2.6 GiB）下载完成后，在 uv 缓存归档重命名处被 Windows 拒绝访问（`os error 5`）。该失败根因**未定位**，不得直接归因于杀毒软件；按预算本轮**未消耗**那次有记录的重试，也**未**新增任何安装或下载。
  因此 CUDA 探针的 torch 部分为 NOT_RUN；`nvidia-smi` 部分正常（RTX 5060 Laptop GPU / 驱动 582.05 / 8151 MiB / CUDA 13.0）。
- 权重访问仍为 AUTH_REQUIRED（无本机 token，匿名 401）。**401 不等于账号被拒**。
- SAM 3 运行所需的 `timm`、`iopath`、`ftfy` 等推理依赖因此也未安装；它们不影响接口层与测试。
- 真实推理、mask 质量、跨视一致性、下游重建：**未评价**。
- **A-ARRAY 转换收口**：旧离线交付尚有行为差异与验证缺口，主线修复及实际定向验证结果另记于新 HANDOFF。真实模型上的**峰值是否变化未测**——
  本次越限根因尚未定位，转换层开销**不是**已证明的越限根因；不得据此声称"修复泄漏"或"峰值必降"。
  真机复验由主线在接收本改动后另派。


## Identity-bound AI-assisted positive boxes (2026-10-11)

The CLI now accepts `--visual-prompt <JSON file>` instead of `--text-prompt` for
box-only input. These arguments are mutually exclusive. Visual JSON is bounded
to 1 MiB, rejects duplicate keys/nonfinite values, and is accepted only for the
SAM3 backend. It has this exact structure (replace frame identity/hash with
verified canonical values; the frame ID and SHA below are placeholders):

```json
{
  "kind": "positive_boxes",
  "image_domain": "photo_io_canonical",
  "annotation_source": "ai_assisted",
  "frames": [{
    "input_index": 59,
    "frame_id": "ACTUAL_T01_FRAME_ID",
    "image_sha256": "ACTUAL_CANONICAL_FILE_SHA256",
    "width": 1024,
    "height": 683,
    "box_xyxy": [230, 70, 705, 637]
  }]
}
```

Use `--indices 59 --object-id skull-01 --visual-prompt box59.json` with the
existing checkpoint, output, worker and resource options; do not pass a text
prompt. API users pass `visual_prompt=<the dict>, text_prompt=None` on
RunRequest. Every requested frame needs exactly one positive integer-pixel box
and matching canonical frame ID, dimensions and actual file SHA256 (not the
raw-input T01 hash). No negative boxes/points/mixed prompting/fallback exist.

The box is normalized to `[cx,cy,w,h]` against canonical dimensions, then
`set_image -> add_geometric_prompt(label=True)` runs once with unchanged CUDA
BF16/inference scope and instance interactivity disabled. Masks/overlays are
real model predictions, never truth labels. The box mode has a distinct
configuration fingerprint and a complete manifest visual field. Existing text
mode configuration options stay unchanged. Frozen text-only diagnostic packages
need separately reviewed geometric hooks and source bindings; this product
extension does not silently authorize or revalidate those packages.
