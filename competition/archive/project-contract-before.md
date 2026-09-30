---
title: flagos-s2 项目入口卡
tags: [project, flagos-s2]
created: 2026-09-17
updated: 2026-10-01
current_node: M1
status: 用户已选择 Task 103 recompute_w_u；competition/task103 目录已初始化，本轮范围仅创建目录
code_head: ab800a8
next_step: 按用户后续指令推进 Task 103；实现前刷新 Chrome inventory，补齐本题 profile、语义验证器、成绩账本与目标 runner 证据
last_verified_device: macOS arm64 / Apple M3（本机无比赛加速卡）
publish: true
---

# FlagOS S2

> FlagOS 开放计算全球挑战赛第二季，赛道一：SGLang 框架算子在多款芯片的性能优化。

## 目标

围绕 FlagOS S2 建立可复现的跨芯片优化项目。所有当前开放及后续解锁的题目复用同一套 KernelGen 迭代、Chrome 上传、检查点恢复与官方成绩回填流程；只有用户明确指定的 task ID 才进入迭代队列。Task 60 `clamp_position` 和 Task 78 `concat_and_cast_mha_k` 是当前已有独立题目适配器的任务。新题需先根据官方资料补齐独立 profile、语义验证器和成绩账本，再进入所选任务的流程。

比赛入口：[FlagOS 赛道一](https://flagos.io/race-detail-season2?id=782kzq4m&lang=cn)。

## 技术栈与代码仓

- 代码仓：[hypiasd/FlagGems-sglang](https://github.com/hypiasd/FlagGems-sglang)，本地目录 `project/flagos-s2/`
- `origin`：用户 fork `https://github.com/hypiasd/FlagGems-sglang.git`
- `upstream`：官方 `https://github.com/flagos-ai/FlagGems-sglang.git`
- 当前 Task 103 分支：`codex/task103-recompute-w-u`，代码 HEAD：`ab800a8`；从包含共享工作流与非目标 GPU smoke 的 `e2af813` 创建。Task 78 保留在 `codex/task78-concat-cast-mha-k`，Task 60 保留在 `codex/task60-clamp-position`。
- 主要技术：Python、PyTorch、Triton、FlagOS 多芯片后端

## 项目约束

- 提交入口必须与题目算子同名：Task 60 为 `clamp_position`，Task 78 为 `concat_and_cast_mha_k(k, k_nope, k_rope)`。
- 所有浏览器交互统一使用 Chrome。会话授权范围是当前登录团队、FlagOS S2 当前可见每日额度；不包括其他账号或比赛。
- 所有 S2 任务复用 `competition/flagos_s2_workflow.py` 状态机，但题目契约、测试集、目标芯片、成绩汇总和 release hurdle 必须来自各自 profile；新任务模板占位符未清理时拒绝候选生成。
- Campaign overview 合并 Chrome 捕获的当前开放题目 inventory、本地全部 task profile 与各题最新 checkpoint。开放题总览不等于迭代授权：`list-tasks` 默认不选任务、`queue_order` 为空；只有显式 `--tasks taskNN ...` 选择的题才进入调度。所选任务按提交恢复、终态回填、已有生成流程续跑、可运行新迭代、MCP 检查、该题建档和证据缺口排序。inventory 不完整时不能声称覆盖全部开放题；未在 Chrome 确认为开放的题不能选入迭代；缺 profile/语义验证器只阻断所选题目，不借用其他题目契约。
- KernelGen 必须出现在当前任务 fresh live tool registry 才能生成代码；配置文件存在或只读服务端握手成功都不算可调用。没有可信目标 runner 时，可在已授权范围内将所选任务的完整 gated package 提交官方评测，但终态结果返回前保持未验证。
- 中断恢复读取 `competition/.autopilot/runs/<task>/<run-id>/` 事件日志。若停在 `upload_armed` 或更后，先用 Chrome 查当前记录再恢复，禁止盲目重传；同一官方记录的账本回填可幂等续跑。
- 结果必须覆盖该题官方列出的全部支持芯片，并保持输出 dtype、shape 和边界语义正确；Task 60/78 的历史矩阵为 8 款，2026-09-30 核实的第 7 批 Task 93–109 为 7 款，不含燧原。不能把某题的目标数量直接用于其他题。
- 评测以 FlagOS 网站提交结果为准；本机 Apple M3 无法替代昇腾、燧原等目标硬件做真实性能验证。
- 团队提交次数有限；每个线上版本必须打包一个有明确收益目标的完整优化假设，不能只提交单个微小参数改动。提交前先做本地静态/语义检查，并用历史线上结果评估预期收益与回退风险。
- 新增硬约束：每个版本的每个目标芯片都必须有独立、可辨认的结构性新优化；不能只保留、恢复旧路径，或只改 `BLOCK`、`num_warps`、阈值等参数。结构性优化至少要改变数据流、tile/映射粒度、访存布局、kernel 拆分/融合、持久化循环或地址计算方式之一。若多个芯片共享同一文件，必须说明该结构变化如何分别作用于这些芯片；提交前建立逐芯片优化矩阵，任何芯片缺少新结构优化则该版本不准提交。
- 线上回归不能只靠下一版恢复旧代码解决：发生回退的芯片，下一版必须同时提出新的结构性优化并单独记录预期收益、兼容风险和验证证据；只有恢复而没有新结构变化的候选视为无效版本。
- 代码候选与实验压缩包不直接混入 grounds；项目代码仓只保留可复现的源文件和必要记录。
- 多个同名提交文件可能发生覆盖，当前优先采用单文件内部按设备分支的结构。
- 非目标 GPU smoke（`competition/task78/kernelgen/gpu_smoke.py` + `gpu_smoke_remote.py`）只提供真实 Triton 编译/执行信号，默认 advisory、`--require-gpu-smoke` 才阻断；它**永远不能**作为 `target_validated` / `measured` 证据，证据类别固定为 `nvidia-smoke-nontarget`。连接与凭据只放在被 `.gitignore` 忽略的 `competition/.autopilot/gpu-smoke.json`，不进 Git、不落日志。sm_75 与各 target 后端不同，其失败可能是假阳性、其通过不能外推。

## 当前现状

- 最新 Task 78 官方记录为 v25：2026-09-24 完成 `8/8`，平均 `1.12×`；全芯片最佳有效聚合仍为 v12 的 `1.27×`。结果账本 `task78_results.py verify` 已检查 25 条记录及包/源码哈希、一致性规则；各芯片冠军与摘要由 `competition/task78/results.md` 和 `task78_results.py summary` 提供。
- 当前代码仓 HEAD 为 `ab800a8`（`project flagos-s2: initialize task103 directory`），已推送到 `origin/codex/task103-recompute-w-u`；在 `e2af813` 的基础上新增 `competition/task103/README.md`，保存题目目录。用户未跟踪的 3 个 ZIP 与 `v20/` 源码保持原样。
- 用户于 2026-10-01 明确选择 Task 103 `recompute_w_u`，本轮仅要求先创建文件夹；目录初始化完成，见 [runbook 节点 40](runbook.md#节点-40选定-task-103-并创建题目目录m12026-10-01)。尚未建立本题 profile/validator/ledger，未启动生成、GPU 实验或上传。
- 2026-09-30 接入租用 GPU 作为**非目标 smoke 层**：远端为 2× Tesla T4（sm_75）、驱动 580.159.04、torch 2.10.0+cu128、triton 3.6.0。全量 201 用例 × 7 文件 = 1407 次设备执行全部通过，负控有效，门控三路径（advisory / 要求生效 / 缺报告）实测符合预期；新增 19 项离线测试与既有 40 项回归全绿。同一轮里静态 compiler-risk 扫描仍报 blocker，证明两层证据互不替代。细节见 [runbook 节点 39](runbook.md#节点-39把租用-gpu-作为非目标-smoke-层接入工作流m12026-09-30)。该机以 root + 弱口令暴露在公网，用户本轮选择暂不处理。
- Task 60 与 Task 78 上次检查点为 `auto-20260924-01` / `tool_unavailable`，Task 60 CPU 语义门因当时环境缺 PyTorch 标记 `inconclusive`。2026-09-30 fresh live registry 已列出 KernelGen 的 generate、optimize、specialize、autotune 四项操作；本轮未调用，目标 runner 和当前 CPU 环境未重新验证。未生成候选、未上传，也未检查当前每日额度。
- 2026-09-30 Chrome 与官方公开接口确认第 7 批 `17` 道任务：Task 93–109，均为竞争中；提交截止 `2026-10-01 19:59:59`（北京时间）。已逐题读取官方计算定义和参考实现，按实现深度优先推荐 `recompute_w_u`、`tiny_n_gemm`、`topk_sigmoid`，完整比较见 [runbook 节点 38](runbook.md#节点-38第-7-批-17-题的实现深度与简历选题评估m12026-09-30)。现有 campaign inventory 仍是上次导入的 Task 76–92，未把本轮只读清单导入调度器；用户选题后须刷新。
- `list-tasks` 默认仅输出总览，显式 `--tasks task103` 才会调度所选题目；当前选择已记录于项目入口与时间线，本轮没有调用调度器。
- 上次核验（`2026-09-24 08:51 UTC`）的 Codex live registry 无 KernelGen 操作，检查点记为 `operation_unavailable`；本轮工具注册的变化见上。配置或 HTTP 握手不能替代当前工具注册和目标验证证据。

- Task 78 `concat_and_cast_mha_k` 已确认题目定义：输入/输出均为 3D，NoPE 按 head，RoPE 为单 head 广播；参考实现是 `expand -> cat -> to(k.dtype)`，正确性为 exact。
- Task 78 v1 已新增 `competition/task78/concat_and_cast_mha_k.py`，用单个 Triton kernel 直接把 NoPE 前缀和 RoPE 后缀写入目标 dtype，避免 expanded RoPE、cat 和额外中间结果；已通过 `py_compile`、AST 入口检查和单文件 zip 检查。
- Task 78 支持 8 款芯片；2026-09-18 本次查看题目详情时，第 1 名显示 `1.90x`，团队最佳 `1.27x`。此前观察的 `1.39x` 已不是最新榜首值；本机无法替代目标芯片做真实性能验证。
- Task 78 v1 通过 7/8，华为失败；v2 仍然华为失败（其余两款芯片当时仍在评测），说明仅改 contiguous 输出布局没有解决问题。`TensorLikePair` 是比较器外层包装，用户提供的文本没有包含内部异常，不能据此断言是 layout 根因。
- Task 78 v3 新增 `concat_and_cast_mha_k_ascend.py`：华为专用实现把 NoPE 前缀和 RoPE 后缀拆为两个纯 Triton copy/cast kernel，避开通用 kernel 的 `tl.where`、混合 masked load 和 RoPE 负指针算术；项目提交 `3629909`，已推送到 `origin/codex/task78-concat-cast-mha-k`。

- 官方 fork 已干净克隆，`origin` 与 `upstream` 均已配置。
- Task 60 `clamp_position` 已有 v5 版本通过 8/8，曾达到平均加速比 2.29×；燧原曾达到 9.40×。
- 用户反馈 v7 的昇腾路径有提升。
- 统一候选已迁入 `competition/task60/clamp_position.py`，并推送到 `codex/task60-clamp-position`，提交 `c652b42`。
- 该候选把通用、燧原和昇腾路径合并到一个公开函数中，避免多个同名提交文件互相覆盖；已通过 `py_compile`、AST 公开入口检查和单文件 zip 检查。
- 对照官方仓库后确认：官方后端大量采用连续输入的直接指针、显式 `empty_like` 输出、低并行度的小型整数/逐元素 kernel，以及按 vendor 配置 tile/warp；没有可直接复用的 `clamp_position` 实现。
- v10 提交 `6ad16dc`：连续输入走无 stride 乘法的通用 Triton kernel，昇腾对减一临时结果执行 `clamp_min_` 以减少一次输出分配，燧原路径保持不变；已通过语法、AST 公开入口和 diff 检查并推送。
- v11 提交 `16c14a1`：依据官方 `pointwise_dynamic`，仅对 `DNN_VENDOR=metax/hygon/tsingmicro` 调整 tile 上限和 warp，其他设备保持 v10 配置；已推送等待在线验证。
- v11 在线结果为 `1.85/1.35/1.45/1.58/1.01/1.18/1.42/1.44`，8/8，平均 `1.41×`；相较 v10 的 `1.39×`，海光和华为上涨，但燧原仍只有 `1.45×`，显著低于历史 v5 的 `9.40×`。
- v12 提交 `4abeb7e` 的 Enflame vendor-only dispatch 因历史 9.40× 结果被用户确认是 bug，不再作为有效优化方向。
- v13 提交 `c716dd4`：撤销仅凭 `DNN_VENDOR=enflame` 强制切换燧原路径，保留真实 `device.type == "gcu"` 判断；后续只比较可复现的正常分数。

- Task 78 v8 线上通过 `8/8`，但平均从 v7 的 `0.95x` 退到 `0.90x`：天数智芯 `1.69x`、沐曦 `0.87x`、燧原 `0.15x`、海光 `1.24x`、昆仑芯 `0.31x`、华为 `0.22x`、国际通用 A `1.30x`、国际通用 B `1.38x`。因此“仅增加 num_warps”被证伪，尤其伤害沐曦和天数智芯。
- v9 项目提交 `50a5022`：撤回后缀 warp-only 调参；Iluvatar/MetaX/Enflame/Hygon 的连续输入路径改为四行合并处理，Kunlunxin 回到 v7 一行路径，Ascend 保留已通过的 persistent row-tile 并将 BM 上限从 16 调到 32、grid 上限设为 48。已通过 Python 编译、AST 入口/无 native cat 检查、批量行映射语义检查和 ZIP 完整性检查；本机无 Triton 与目标芯片，尚未验证线上正确性和性能。
- v9 线上结果已完成 `8/8`：天数智芯 `2.29x`、沐曦 `1.50x`、燧原 `0.18x`、海光 `2.06x`、昆仑芯 `0.31x`、华为 `0.14x`、国际通用 A `1.36x`、国际通用 B `1.38x`，平均 `1.15x`。相较 v8 平均提升 `0.25x`，rows-per-program 对天数、沐曦、海光有效；华为下降 `0.08x`，燧原仍未改善到可接受水平。
- v10 项目提交 `ea5a63d`：这是按提交预算合并的完整候选。Ascend 保持 v5 已验证的 `BM/grid` 参数，同时新增 contiguous 专用 row-tile，去掉运行时 stride 乘法；Enflame 改为按 token/head 的规则 tile，避免连续路径的 row 除法/取模并固定 1 warp；Kunlunxin 加入保守四行连续 copy；Iluvatar/MetaX/Hygon 进一步改为 token/head tile，让同一 token 的 RoPE 在一个 head tile 内广播加载，去掉 flatten row 除法并减少重复 RoPE 访存。已通过语法、AST、行/head 映射语义和 ZIP 检查，等待线上评测。
- v10 线上结果已完成 `8/8`：天数智芯 `2.20x`、沐曦 `1.50x`、燧原 `0.31x`、海光 `1.96x`、昆仑芯 `0.31x`、华为 `0.17x`、国际通用 A `1.32x`、国际通用 B `1.40x`，平均 `1.14x`。相较 v9，Enflame `+0.13`、华为 `+0.03`，但 Iluvatar `-0.09`、Hygon `-0.10`，平均略降 `0.01x`；四 head 并非跨芯片最优。
- v11 项目提交 `32a4c5e`：Iluvatar/Hygon 使用 2-head，MetaX/Enflame 小 tile 自适应到 8-head、较大 tile 降为 4/1-head；通用入口和 Kunlunxin 也改为 token/head RoPE 广播路径；Ascend 新增 bounded persistent token/head-tile kernel，连续路径按 token 循环并在 4-head tile 内复用 RoPE。已通过语法、AST、adaptive head-tile 参考映射和 ZIP 检查。
- v11 线上结果：天数智芯 `2.21x`、沐曦 `1.51x`、燧原 `0.39x`、海光 `1.57x`、昆仑芯 `0.31x`、华为 `0.17x`、国际通用 A `1.53x`、国际通用 B `Failed`，通过 `7/8`，平均值未生成。相较 v10，燧原 `+0.08`、国际通用 A `+0.21`，但海光 `-0.39`，且国际通用 B 从 `1.40x` 回归为失败；v11 不能作为正常性能基线。当前首要问题是定位通用 token/head 路径对国际通用 B 的兼容性回归，再单独处理海光退步。
- v12 项目提交 `36c81c8`：通用、GPU 后缀和 Ascend 连续路径把 RoPE 改为一维加载后显式广播，保留 head-tile 复用但降低二维 masked source load 的后端兼容风险；Hygon 改回 4-head，并与新 RoPE 加载方式组合，针对 v11 的 `1.57x` 回退。已通过语法、AST、入口/无 native op 和 ZIP 检查，等待用户线上提交。
- v12 线上结果：天数智芯 `2.21x`、沐曦 `1.50x`、燧原 `0.40x`、海光 `2.39x`、昆仑芯 `0.31x`、华为 `0.25x`、国际通用 A `1.59x`、国际通用 B `1.51x`，`8/8`，平均 `1.27x`。相较 v11，国际通用 B 从失败恢复，Hygon `+0.82`、华为 `+0.08`，平均成绩首次恢复为有效值；当前离已观察到的 `1.39x` 第一名仍差 `0.12x`。
- v13 项目提交 `e208db8`：以 v12 的兼容结构为底座继续优化所有相关路径；Iluvatar 新增四行 contiguous row-batch，通用/Hygon/MetaX/Enflame/Iluvatar 做实际 head-tail specialization，昆仑芯新增 bounded contiguous row-batch，燧原小 block 试 16-head，华为小 block 试 8-head。已通过语法、AST、入口/无 native op、row mapping 和 ZIP 检查，等待用户线上提交。
- v13 线上结果：天数智芯 `2.23x`、沐曦 `1.50x`、燧原 `0.49x`、海光 `1.97x`、昆仑芯 `0.28x`、华为 `0.33x`、国际通用 A `1.52x`、国际通用 B `1.50x`，`8/8`，平均 `1.23x`。Enflame `+0.09`、Ascend `+0.08`、Iluvatar `+0.02`，但 Hygon `-0.42`、Kunlunxin `-0.03`，通用 A/B 也回退；v14 将按文件和变量隔离这些回归。
- v14 项目提交 `a31552f`：Enflame 将 16-head 小 block 区间扩到 256 元素并对大 block 使用 2 warp；Ascend 的 tiny-block persistent program 每个覆盖 2 个 token；Iluvatar 小 block row-batch 试 8 行。Hygon、Kunlunxin 和通用 A/B 保持回归隔离后的多-head调度，并新增仅针对单 head 输入的窄范围 tile 特化。已通过语法、AST、入口/无 native op、映射和 ZIP 检查，等待用户最后一次线上提交。
- v15 项目提交 `55d77ff`：按新增硬约束为每个目标芯片加入结构性变化。国际 A/B 使用双 token contiguous 映射；Hygon 使用 token-persistent 循环；MetaX 使用完整幂二块无 mask kernel；Kunlunxin、Ascend、Iluvatar 使用 token 级 RoPE 复用；Enflame 将宽行拆成 NoPE/RoPE 两个 kernel。逐芯片 README 矩阵、Python 编译、AST 入口/无 native op、token/head 映射、前后缀覆盖和 ZIP 检查均通过；本机无目标 Triton 后端，尚未证明线上编译、正确性和性能。
- v15 线上结果：天数智芯 `1.37x`、沐曦 `1.51x`、燧原 `0.48x`、海光 `1.45x`、昆仑芯 `Failed`、华为 `0.21x`、国际通用 A `1.55x`、国际通用 B `1.60x`，通过 `7/8`，平均值无效。相较 v13，国际 B 有提升，但 Hygon、Ascend、Iluvatar 明显回退，Kunlunxin 的 token-persistent RoPE 路径触发失败；v15 不能作为有效整体基线。
- v16 项目提交 `ba25d1f`：国际 A/B 新增完整幂二块无 mask kernel；Hygon 改为 flattened two-row kernel；MetaX 新增双 token 完整块 kernel；Kunlunxin 改为 NoPE/RoPE 双 kernel；Ascend 改为 bounded flat row-tile；Enflame 新增 token 级 RoPE 复用；Iluvatar 改为双 token/head-tile。已通过 Python、AST、token/head 映射、前后缀覆盖和 ZIP 检查。
- v16 线上结果：Iluvatar `1.86x`、MetaX `1.47x`、Enflame `0.60x`、Hygon `1.81x`、Kunlunxin `0.30x`、Ascend `0.21x`、国际通用 A `1.62x`、国际通用 B `1.53x`，`8/8`，平均 `1.18x`，状态为已完成。相较 v15，仅昆仑芯从失败恢复通过；Iluvatar、Hygon 性能回升，Ascend 持平。Enflame、Kunlunxin、Ascend 仍是主要性能短板。
- v17 项目提交 `ec5b273`：按逐芯片硬约束为每个目标加入新的 kernel 组织。国际 A/B 将 compact contiguous rows 路由到 NoPE/RoPE 双 kernel；Hygon 加入双 token/head-tile；MetaX 加入 wide-row 双 kernel；Kunlunxin 加入非持久双 token/head-tile；Ascend 加入 bounded-grid 双 token/head-tile；Enflame 加入双 token/head-tile；Iluvatar 加入 wide-row 双 kernel。已通过 Python、AST、尾部映射、前后缀覆盖、diff 和 ZIP 检查；v17 尚未线上提交。

- v17 线上结果（09-18 21:19 提交）：Iluvatar `1.86x`、MetaX `1.49x`、Enflame `0.47x`、Hygon `2.50x`、Kunlunxin `Failed`、Ascend `0.20x`、国际 A `0.85x`、国际 B `1.08x`；已完成 `7/8`，平均无效。通用双 launch 与 A/B 退步相关，但聚合分数不能证明单一根因；昆仑芯也没有可据以确诊的 traceback。
- v18 项目提交 `3749eff`：国际 A/B、Iluvatar、MetaX 使用 dense NoPE + broadcast RoPE 的单次 launch；Enflame、Kunlunxin 使用一维分段任务单次 launch；Ascend 用同类任务加一维 persistent grid，限制总 program 数不超过 32；Hygon 使用顺序双 token 二维 tile 和列分块。7 份文件所有非空路径均执行新 kernel，逐芯片收益假设与风险记录于项目仓 `competition/task78/v18/README.md`。直接执行源码的受限 CPU 模型测试 `1323/1323` 通过，语法、入口和 ZIP 源码一致性通过；尚未进行目标 Triton 编译、设备正确性或性能验证，未在线提交。
- v18 线上结果（09-18 22:38 提交）：Iluvatar `2.18x`、MetaX `1.25x`、Enflame `Failed`、Hygon `2.12x`、Kunlunxin `Failed`、Ascend `0.02x`、国际 A `1.57x`、国际 B `1.54x`；仅 `6/8`，平均值无效。Ascend 从 v17 的 `0.20x` 降至 `0.02x`，Enflame/Kunlunxin 继续失败；Iluvatar 有明显提升，国际 A/B 从 v17 的 `0.85/1.08x` 恢复到 `1.57/1.54x`，但 MetaX 和 Hygon 回退。结果页没有失败 traceback，因此对分段任务、persistent grid、整数除法/取模和 64 位地址计算的判断仍是高可信嫌疑，不是已证明的单一根因。

历史候选 v19（`6d5bf2d`）使用输出列对齐的两源融合 store、静态 head 复用和 Ascend 有界 token/column 分块；其线上结果及后续版本由 Task 78 结果账本维护。不要把 v19 根目录源文件当作下一轮逐芯片基线。
- v19 线上结果（09-18 22:57 提交）：Iluvatar `2.36x`、MetaX `1.45x`、Enflame `0.25x`、Hygon `2.45x`、Kunlunxin `0.27x`、Ascend `0.14x`、国际 A `1.60x`、国际 B `1.51x`；`8/8`，平均 `1.25x`。相较 v18，Iluvatar/MetaX/Hygon/Ascend/国际 A/B 分别变化 `+0.18/+0.20/+0.33/+0.12/+0.03/-0.03`，燧原和昆仑芯从失败变为通过。相较完整 v12 的 `1.27x` 仍低 `0.02x`；燧原、华为、昆仑芯仍是主要性能短板。
- FlagOS Task 78 流水核对（09-18）：共 `19` 条线上记录，覆盖 v1–v13、v15–v19；v14 只有离线候选，没有线上成绩。两条 v7 使用相同文件名但提交时间不同，必须按两次独立提交记录：12:03 为 `0.95x`，12:10 为 `0.97x`。完整流水和各芯片最高观测/可信状态见 runbook 的“Task 78 线上成绩总表（持续维护）”；单次峰值不自动视为稳定最佳。

v18 记录口径修正：Enflame 是从 v17 的 `0.47x` 新回归为 Failed，Kunlunxin 才是连续失败；此前“Enflame/Kunlunxin 继续失败”的表述不准确。没有 traceback 时，也不能将失败认定为已经确认的正确性或编译错误。

## 新增硬约束：逐芯片结构性新优化（2026-09-18）

- **触发**：用户指出不能接受某个版本只在部分芯片增加新优化，其他芯片仅恢复旧版本或做简单参数调节。
- **规则**：从下一版本开始，每个版本、每个目标芯片都必须有可辨认的结构性新优化。允许的结构变化包括数据流、tile/映射粒度、访存布局、kernel 拆分或融合、persistent 循环、地址计算等；单独修改 `BLOCK`、`num_warps`、`num_stages`、阈值或 dispatch 条件不算满足。
- **共享路径口径**：同一文件服务多个芯片时，只有当结构变化确实作用于这些芯片，并在逐芯片矩阵中分别写出收益假设和风险，才算每个芯片完成；不能用“一份通用代码改动”无证据地覆盖所有芯片。
- **回归处理**：恢复旧路径可以作为兼容保护，但不能作为该芯片本轮唯一变化。对线上回退芯片，下一版必须在恢复保护之外增加新的结构性优化；否则不打包、不提交。
- **准入证据**：每个候选提交前必须有“芯片 → 代码位置 → 结构变化 → 预期收益 → 风险 → 静态/线上证据”矩阵。矩阵缺项、只有调参、或只是保留/回滚旧路径，都判为不合格。
- **v14 复盘**：v14 的 Enflame 主要是 head 区间和 warp 调整、Ascend 是 token 覆盖数调整，严格按新规则都不能单独算结构性新优化；v14 尚未线上提交，不能把它当作满足新约束的合格版本。后续候选必须重新设计逐芯片结构变化。

## 跨芯片 Triton 适配约束（比赛决策合同）

本节是后续 FlagOS 多芯片算子优化的默认约束。它记录的是调研得到的稳定事实、已经验证的比赛判断和仍需在线确认的假设；除非新的线上结果推翻，否则后续候选按此执行。

### 总体判断

- Triton 只统一编程模型和算子数学语义，不统一各芯片的编译器、LLVM、设备运行时、内存层次和执行模型。FlagTree 官方为不同芯片维护不同 Triton 分支/版本；FlagGems 的 backend matrix 也显示各 vendor 使用独立插件和 Torch 栈。
- 后续目标不是“一个完全相同的 kernel 跑遍所有芯片”，而是“一个统一数学实现 + 后端可控的 tile、warp、grid、地址计算适配”。任何专用路径都必须保持同一个公开函数签名和同一个输出语义。
- 正确性是硬门槛：必须先达到 8/8，再比较平均性能；本机 Apple M3 只能做语法、AST、打包和有限的 CPU 逻辑检查，不能证明目标芯片的编译、正确性或性能。
- 线上比赛结果是唯一有效的目标芯片证据。疑似 bug、未能复现或只在异常路径得到的高分不得作为基线，也不得据此继续优化；Task 60 的燧原 `9.40x` 已被用户确认无效。
- 已合并算子 PR 只能提供异题结构参考，不能证明作者在每道题排名第一，也不能解释 Task 78 同题榜首的具体实现。对照 [赛道一 PR #87](https://github.com/flagos-ai/FlagGems-sglang/pull/87) 的源码，`moe_fused_mul_sum` 为保住跨芯片 `8/8`，撤回本地略快但在燧原/昆仑芯只达 `6/8` 的二维 `top_k × hidden` tile，保留标量 `k` 循环、固定 hidden tile、受限 grid 和运行时 mode。该例支持“线上兼容性优先于局部速度”的既有规则，但不证明同样的 kernel 组织适合 Task 78；本轮尚无 Task 78 榜首源码和逐 case 耗时可做同题归因。

### 后端适配矩阵

| 后端 | 已知运行栈/设备模型 | 默认约束 |
| --- | --- | --- |
| Iluvatar | CoreX；CUDA 风格接口但有独立 vendor Triton/运行时 | 先复用通用连续 row-tile；单独 sweep tile/warp，不假设等价于 NVIDIA |
| MetaX | MACA；独立 MetaX Triton 和 LLVM 插件 | 允许较大 tile，但 warp 上限/启发式与 NVIDIA 不同；优先试 1024/2048 tile |
| Enflame | GCU；triton-gcu/FlagTree Enflame | 使用规则 `arange` 和连续地址；避免逐元素除法/取模、依赖加载值的动态 mask；大 tile、少 warp 优先 |
| Hygon | DTK/HCU；HIP 风格 Triton | 使用 GPU 通用路径的数学结构，但按 Hygon 的 tile/warp 上限单独调参 |
| Kunlunxin | XPU；Triton XPU | 避免复杂 `tl.dot`/归约、dot 输入 masked load、复杂广播和混合 pointer 算术；大索引考虑 int64 program id；不要假设 `num_warps=1` 一定生效 |
| Ascend | CANN + Triton Ascend；NPU/PrivateUse1 | grid、UB/vector core 与 GPU 不同；优先 bounded grid/persistent row-tile，不能照搬 GPU 大 grid |
| 国际通用 A/B | 比赛公开资料未给出具体芯片映射 | 不猜具体 vendor；先用通用连续 kernel，依据线上分数再决定是否增加专用后缀文件 |

### Kernel 设计规则

1. 统一数学逻辑，优先统一“按行处理、连续读写、直接写目标 dtype”的结构；不要为了表面统一而把所有后端强行塞进同一套 launch 参数。
2. 通用路径优先服务 NVIDIA/AMD 风格以及 Iluvatar、MetaX、Hygon 和未识别的国际芯片；必须保留连续输入/输出的快路径，同时为特殊 stride 保留正确路径。
3. Enflame 路径优先采用 `program_id -> row`、连续 `arange`、静态边界和少量 warp，避免把一维线性 index 拆成 `i // D`、`i % D` 后再参与地址计算。
4. Kunlunxin 路径优先拆成最简单的连续 copy/cast 阶段；不要引入不必要的归约、复杂 `tl.where`、masked source pointer 或依赖中间计算结果的广播。
5. Ascend 路径保持独立 bounded-grid 方案；任何扩大 grid、改变 persistent 粒度或复用 GPU mask 的改动都必须单独提交并观察 8 芯片结果。
6. 每轮只改变一个主要变量（后端 dispatch、tile、warp、grid、地址布局或 kernel 拆分之一），记录提交包、8 芯片通过数、各芯片分数和平均值；不能把多个猜测混进同一候选。
7. 遵守比赛的 Triton/Triton-TLE 核心计算要求：不能用 try/except、设备分支或 native Torch 算子作为 Triton 失败时的隐式 fallback；专用后端也必须保持可审计的 Triton 核心实现。

### 事实来源和未决项

- FlagTree 多后端说明：<https://github.com/flagos-ai/flagtree>
- FlagGems 官方后端环境矩阵：<https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/backends.yaml>
- Enflame、Kunlunxin、Ascend 的官方后端手册：<https://github.com/flagos-ai/flagtree/wiki/User-manual-for-enflame>、<https://github.com/flagos-ai/flagtree/wiki/User-manual-for-xpu>、<https://github.com/flagos-ai/flagtree/wiki/User-manual-for-ascend>
- 国际通用 A/B 的实际硬件映射，当前公开资料未找到，后续只能以比赛 UI、提交日志或用户观察到的后端信息确认，不能凭名称猜测。

## Task 78 KernelGen 工作流约束（2026-09-19）

v23 的 FlagOS 结果证明旧工作流存在真实漏检：Enflame 因 autotune launch 的重复
`BR` 参数失败，Ascend 因 `multibuffer` 配置 ABI 失败，Kunlunxin 出现大范围
数值错误；旧 CPU 模型只跑首个 autotune config，且子代理的 `novel_findings`
没有阻断 promotion。

当前工作流已在代码仓提交 `02fe8d3` 并推送到用户 fork 分支，规则如下：

- 先跑便宜的 deterministic compiler-risk gate；静态硬阻塞时跳过耗时语义矩阵。
- `triton.Config` 默认只接受可移植的 `num_warps`、`num_stages`；其他选项必须有目标编译器证据。
- autotune kernel 不允许在 launch 中再次显式传入 tile constexpr；检查 `BR/NRC/BC/BN/BH/HS/BT` 的绑定冲突。
- 检查 masked load 的潜在负指针、标量 predicate 与向量 mask 的隐式广播，以及 host loop/grid 与 autotune tile 不一致。
- 日常 `--autotune-sweep` 对首个 config 跑 189 个完整用例，其余 config 跑 tile 边界、空段、尾块、stride 和 view 风险集；发布审计使用 `--autotune-sweep-full`。
- 子代理的任何非空 `novel_findings` 都是硬失败，必须 KernelGen 修复或目标 compile/correctness smoke 后才能继续；不能把它当免责声明。
- v23 复盘验证：新静态门报告 21 个阻塞；Iluvatar 的 config sweep 在 `BC=256` 的 config 4/5 复现缺写。该结果是本地语义证据，不替代目标芯片验证。

本机仍只有 Apple M3；新工作流能提高“提交前发现错误”的概率，但不能声称通过了任何真实 vendor 编译器或性能测试。

### Codex MCP 本机注册约束（2026-09-24）

- Codex 从其本机 `config.toml` 的 `mcp_servers` 加载 MCP；项目 `.mcp.json` 的存在不代表 Codex 工具已注册。Codex 远端配置使用 Streamable HTTP，并须以当前会话的工具注册表确认 `generate_kernel`、`optimize_kernel`、`specialize_kernel` 或 `autotune_kernel` 可调用。
- KernelGen bearer 凭据只保存在被忽略的本机配置源中；Codex 使用本机 header helper 读取，不复制到 Git 跟踪文件或打印到日志。MCP 工具列表的远端握手成功仍需在新 Codex 会话确认实际工具已加载。

### Task 78 端到端工作流 v7（2026-09-24）

- 代码仓 `1533723` 整理了完整 run state、逐芯片基线快照、`optimization-manifest.example.json`、预注册成绩预测、Stage A → 静态扫描 → Stage B 审查、目标设备全量 preflight、zip 校验、一次性提交与结果账本回填顺序。
- 每个候选使用唯一 run ID；`prepare_mixed_candidate.py` 从已验证账本快照逐芯片最佳观测源码，shared generic 同时参考 International A/B。生成前先填 8 个 target forecast 并通过 `submission_hurdle.py`：均值至少 `1.50×` 且至少比当前 champion composite 高 5%（取较高门槛），保守均值不回退、每目标 lower bound 不低于 -5%。
- 模板内的 `TODO`、空哈希、低置信度是故意的阻断占位；填入实际 source/baseline 哈希和可核实证据前，forecast / structural gate 不会通过。非 KernelGen 代码只有用户明确批准该次来源后才能生成并标成 `agent-authored`。
- 最新 v25 `8/8` 但均值 `1.12×`，低于最佳逐芯片源 composite 的约 `1.39×`；新候选先以有证据的预测判断是否值得开展，不能把低效历史源当作总包基线。
- 当前 Codex 任务的 live tool registry 中没有 KernelGen 操作；本机配置和服务端握手结果不代表本会话可调用，状态为 `configured_unavailable`。新建/刷新本地任务后仍须重新核对工具注册表。Apple M3 无目标比赛芯片，可信目标 preflight 当前仍缺；未完成时保留 `inconclusive`。

## 入口

- [运行手册](./runbook.md)：时间线、实验证据、决策和下一步。
