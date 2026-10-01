# FlagOS S2 执行手册

算子实验和比赛适配各有一个入口。命令从项目仓根目录运行，Python 3.10+；GPU 执行端还需要 PyTorch、Triton 和 CUDA。当前 T4 验收见 [ACCEPTANCE.md](ACCEPTANCE.md)。本轮未上传比赛包。

## 目录

- `experiments/`：冻结源码、设备执行、正确性、计时、分析和报告。
- `adaptation/`：官方契约、发布检查、独立适配、ZIP 和官方成绩账本。
- `task60/`、`task78/`、`task103/`：本题源码、profile、adapter、开发用例及历史官方记录。
- `archive/`：旧工作流和版本资产，入口见 [归档说明](archive/README.md)。
- `.local/`：被 Git 忽略的连接配置、冻结快照、实验报告、profiler 原始产物及本地资产备份。

上游 `src/`、`tests/`、CI 和包布局继续保留。日常实验直接编写源码，不调用 KernelGen 工具（其专精知识库只覆盖华为且本会话未注册）；复用的是它的证据方法论，见 [证据纪律](#证据纪律复用-kernelgen-方法论)。旧工作流只作历史参考。

## 实验

### 配置设备

保存 `.local/devices/t4.json`，POSIX 权限设为 `0600`，先在本机 SSH 配置中确认服务器主机密钥：

```json
{
  "host": "your-host",
  "user": "your-user",
  "port": 22,
  "auth": {"method": "key", "identity_file": "/absolute/path/to/private-key"},
  "remote_python": "python3",
  "remote_root": "/tmp/flagos-experiments",
  "gpu": 0,
  "env": {}
}
```

密码认证可使用 `{"method":"password","password_env":"FLAGOS_DEVICE_PASSWORD"}`，本机需有 `sshpass`。凭据和真实主机地址只留本机。`env` 可配置设备所需的 PATH/LD_LIBRARY_PATH；不要写入共享文档。执行端需有 SSH、tar、timeout、flock；默认 GPU 0、前台串行运行。同一 remote_root 下的 GPU 锁保证串行，设备忙时返回 busy，稍后重试。可另建其他设备别名，通过 `--device` 选择。

```sh
python -m competition.experiments doctor --device t4
```

doctor 报告 GPU、显存、能力、驱动、PyTorch、Triton、CUDA 和 profiler。工具存在不代表有采集权限；实际 `profile` 成功才证明该次采集有效。T4 不支持原生 BF16 Tensor Core 路径。

### 编译环境与 Triton-TLE 能力

优化前按 [Triton / TLE 指南检查清单](TLE_GUIDE.md) 核实目标编译器版本、原语支持与匹配示例。指南推荐版本不等于赛事实测版本；未列芯片记 unknown，不推断 unsupported。先解决编译兼容性，再根据目标瓶颈选择普通 Triton、TLE-Lite 或显式结构实现；不盲用其他芯片的原语。Failed 单元格须点击查看编译详情，资源包装标签不直接当根因。该清单目前人工执行，不是 CLI 自动能力门禁。

### 每轮操作

1. 写候选，记录能被实验否定的假设。
2. 建立源码快照，记录返回的 run ID。
3. quick bench，查看正确性、耗时、波动和覆盖，再决定下一轮。
4. 稳定候选执行完整 test；小幅收益或大波动时显式 confirm。

```sh
python -m competition.experiments new --task task78 --source competition/task78 --hypothesis '验证已有候选在三个代表形状上的表现'
python -m competition.experiments bench --run RUN_ID
python -m competition.experiments test --run RUN_ID --all-sources
python -m competition.experiments bench --run RUN_ID --mode confirm
python -m competition.experiments report --run RUN_ID
```

下一轮用 `new --parent RUN_ID --source ... --hypothesis ...`。实验冻结源码、执行工具、契约、种子和硬件要求，并保存逐文件 SHA-256；修改任何冻结文件会被拒绝。命令执行保存独立 attempt 和事件，失败/中断可用同一个 run 重试并保留前次记录；修改实现必须新建 run。`--json` 输出完整报告，默认输出摘要。

`--source` 可以指定公开入口源码文件，或含题目声明成员的目录。`test --all-sources` 验证快照中的全部成员；默认只验证通用入口。`--cases CASE_ID ...` 选择明确的受影响用例，未知 ID 被拒绝。完整开发矩阵通过仍不声明官方用例覆盖。

### 无设备时的语义回路

设备不可达时，仍有一条**纯语义**回路：`competition/<task>/validate_cpu.py` 通过共享的 `competition/experiments/cpu_model.py`（Task 60/78/112 同一实现）逐 program 串行执行**真实的 Triton JIT 体**（用 CPU torch），检查真实启动路径、每个输出元素恰好写一次且在返回视图内、输入与底层存储未被改动、以及数值与参考在本题容差内。

- **这是语义证据，永远不是设备或性能证据**；每次运行都打印 `LIMIT` 行，不得把它写成目标芯片正确性或加速比。
- **必须带否证对照**：故意破坏一个语义分支（例如删掉 NaN 净化、把 base-2 分支改成总是 `exp`），确认验证器在**对应那条用例**上失败。没有否证对照的"全过"没有信息量，也不写进任何报告。
- **否证对照必须真的失败过，且要针对新分支**：Task 112 的 `vec` 变体（把 position 轴折进 block）删掉输出行掩码后**照样全过**，因为开发用例的位置数全部可整除，掩码行从未出现——"变体通过"当时并不覆盖它自己的分支。做法固定为：先写下这个变体新增的分支，再造一个只破坏该分支的对照；对照若通过，先补用例（非整除位置数、非 2 幂 `D`、`H=1` 这类边角），再重跑对照直到它按预期失败。补用例会改动 `adapter.py`（在快照内），因此必须新建 run 并把新旧两版证据都留在各自的 `cpu-semantic.txt` 里。
- **结构收益要按可复算的口径记账**：program 数是事实，位置级访存指令数可以按源码算术数出（例如每位置 `3N+1` 条），但都**不是速度**；证据文件里要把"这一轮改的是调度粒度还是指令数"写清楚，否则设备上无法区分两个假设谁对。
- **环境**（本机实测）：共享模型自己桩 `triton`，所以只要 CPU torch，不需要真实 Triton。用 `uv` 建隔离环境到忽略目录：`uv venv --python 3.12 .local/.venv-cpu`，再 `uv pip install --python .local/.venv-cpu/bin/python --index-url https://download.pytorch.org/whl/cpu torch`（PyTorch CPU 索引可达；PyPI 直连此前被挡）。
- **证据要绑到冻结字节**：改了源码必须新建 run，并对**该 run 的 `source/` 文件**运行验证，把输出（命令、torch 版本、逐例结果、否证对照）存进 `.local/runs/<run>/cpu-semantic.txt`。`validate_cpu.py` 本身不在 run 的 harness 快照内，可以随时修工具而不作废 run；`adapter.py` 在快照内，改它就要新建 run。

### 计时与预算

| 模式 | bench 阶段墙钟预算 | 样本组 | 每组目标 | 重复上限 | 远端进程硬超时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| quick（默认） | 30 秒 | 3 | 10 ms | 256 | 120 秒 |
| confirm（显式） | 120 秒 | 5 | 100 ms | 4096 | 210 秒 |

quick 默认最多三个代表性用例。输入准备和候选首次编译单独报告；bench 阶段包含所选用例正确性检查、负控、校准、5 ms 目标预热、候选/参考计时及阶段内报告开销。每组排队前检查剩余预算，根据 GPU 和墙钟校准自适应重复次数，慢参考降低采样量。CUDA Events 记录公开入口的分配和 kernel 执行；编译耗时不计入性能数字。候选和参考复用相同输入，在同一进程交替采样。

预算不足停止新工作，报告缺失用例/执行；不足规定组数时 speedup 为空且标为不充分。单次 CUDA/参考调用不可中断，超出预算的时间单独列为 `budget_overrun_seconds`。远端硬超时保留已写的部分报告并终止本轮进程；不将部分结果误报完整通过。可用 `--budget-seconds` 显式调整阶段预算。

报告包含中位数、组间极差/中位数、加速比、初始化、输入/编译、bench、上传和远端耗时。提升不超过 5% 或组间波动超过 5% 时建议 confirm，程序不自动追加长测试。quick 通过只表示选中用例通过，完整 test 独立运行。

每轮批量上传必要文件，远端一个 worker 完成验证和计时；编译缓存跨轮保留，日常仅回收小型报告。`compare --runs FIRST SECOND` 要求设备、软件、用例、种子和计时协议一致；跨轮结果明确标为历史基线比值，每轮报告中的 reference 才是本轮同进程实测。

### 性能分析

```sh
python -m competition.experiments profile --run RUN_ID --tool nsys --cases grid-loop-65-513
python -m competition.experiments profile --run RUN_ID --tool ncu --cases grid-loop-65-513
```

只允许一个明确用例，采集耗时独立于 bench。nsys 采集 CUDA/NVTX 时间线及 stats；ncu 默认请求 basic 硬件计数器，可显式 `--ncu-mode launch` 请求 launch 资源指标。工具返回码、目标正确性、原始报告和 stats 均验证后才能称为采集成功。当前租用 T4 的 ncu 两种路径均受 `ERR_NVGPUCTRPERM` 阻断；用户接受保留该记录，nsys 2025.1.3 已实测成功。

### 题目适配接口

题目 `profile.json` 提供入口/参数、精度、包成员、目标矩阵、官方来源/核验时间、开发用例覆盖标签和硬件要求。`adapter.py` 提供 `cases()`、`quick_ids()`、`inputs(case, device, seed)`、`reference(*inputs)`、`check(got,want)`、`metadata(case)`；用例 ID 稳定且唯一。新增题目要从该题的官方契约建立接口，不借用其他题的矩阵。

Task 103 当前三个用例是开发假设，原生 BF16 路径默认要求支持 BF16 Tensor Core 的设备。不满足时返回 `unsupported`，不改变 dtype；如果实现确实不需要该硬件，可用 `new --requirements path.json` 明确声明 `{"bf16_tensor_core":false}`，仍须验证原 BF16 输入/输出契约。

## 比赛适配

### 契约、独立快照与包

```sh
python -m competition.adaptation inspect --task task103 --refresh
python -m competition.adaptation prepare --run RUN_ID --baseline BASELINE_SOURCE_DIRECTORY
python -m competition.adaptation check --adaptation ADAPT_ID --package-only
python -m competition.adaptation check --adaptation ADAPT_ID
python -m competition.adaptation report --task task78
```

`inspect --refresh` 只读官方公开 API 并保存来源时间和原始证据，检查目标矩阵漂移。Task 60/78 各八个目标，Task 103 当前七个目标；国际 A/B 的实际卡型尚未确认。

`prepare` 复制冻结实验到 `.local/adaptations/ADAPT_ID/`，后端修改只发生在这里。baseline 只复制本题声明的源码文件；ZIP 确定性生成，根目录成员、源码字节、入口签名、Triton 核心计算规则均检查。`--package-only` 只验证 ZIP，不授权发布。

编辑适配源码后，在 `release.json` 填真实证据：

- `targets` 精确覆盖本题目标；每项有 `source`、`source_sha256`、`baseline_sha256`、`structural_change`、`expected_mechanism`、`bottleneck_evidence`、`falsifier`。
- `reviews` 含至少两次不同 `invocation_id` 的独立审查，未解决的 `novel_findings` 阻断。
- `full_correctness` 有 `status: passed`、`evidence` 和全部适配源码的 `source_hashes`。T4 证据必须标明非官方目标，不能代替目标验证。
- Task 78 的 `performance_forecast` 逐目标填写预期/下界 delta、medium/high confidence、`same_target_measurement` 或 `historical_structural_comparison` 依据及 evidence。沿用门槛 `max(1.50×,1.05×冠军 composite)`，保守聚合不回退，单目标下界不低于 -5%。
- 其他题目按自己的已验证发布策略填写 `frozen_goal`、`projected_aggregate`、`forecast_evidence`；Task 103 策略尚未核实，完整发布检查会阻断。

`check` 重建包，检查结构差异、哈希、兼容性、完整正确性、审查和本题收益门槛，记录 `ready` / `blocked` / `package_checked`。只改常量或注释不算结构变化。后端新变更应建立新的适配快照；不得把 T4 的加速比回填为官方成绩。

### 包成员与每芯片专用文件

一个包是 `<op>.py`（必需）加上任意个 `<op>_<chip>.py`，其中 `chip` 必须是本题 `targets` 里的名字。平台把某颗芯片路由到它的专用文件，没有专用文件时用 generic。

- **这是实测行为，不是文档规定**：Task 103 的昆仑芯在 generic 文件（`482c55`）下是 **0.09×**，只新增 `recompute_w_u_kunlunxin.py`（`05c363`）后变 **1.85×**，而 generic 文件字节完全没变。Task 78 长期用 7 个文件（generic + 6 个厂商）并正常计分。
- **清单容易写错，而且错得无声**：成员表就是 `profile.json` 的 `package_members`，打包只写声明里的文件。声明少了 → 未声明的专用文件被**静默丢掉**，而 ZIP 恰好等于声明集合，所以 `validate_package` 会自洽通过、不报错；声明多了或文件不在 → 才报错。Task 103 就发生过：真正提交并拿到 7/7、4.80× 的包是 3 个文件，而 `package_members` 只声明 1 个。
- **`competition.members` 现在 fail closed**：`new`（冻结实验）、`prepare`（建适配快照）、`package`（`validate_package`）三处都会审计源码目录，出现未声明的 `<op>_<chip>.py`、未知后缀、声明了却不存在的文件、缺少 generic 模块，一律报错而不是丢文件。

```sh
python -m competition.adaptation members --task taskNN
python -m competition.adaptation members --task taskNN --source PATH
python -m competition.adaptation members --task taskNN --adaptation ADAPT_ID
```

- 输出每颗芯片实际由哪个文件服务（`target_sources`）、哪些芯片**共用 generic**（`targets_without_dedicated`，作为 warning —— 那是剩余优化空间所在），以及**契约漂移** `drift`：当前声明 vs 本机最近实际提交过的包成员。Task 103 实测 `drifted: true`（声明 1 个，最近 26 个包里最新的是 3 个）。
- 给每颗芯片都配专用文件不是必须的（共用 generic 是合法状态）；但共用意味着这颗芯片在用为别人写的实现，**Task 103 里 `intl_a` 只有 0.89×，低于 1.0 基线**，就是这种情况。

### 证据纪律（复用 KernelGen 方法论）

KernelGen 的 MCP 工具对本项目不可用（专精知识库只有华为，且本会话未注册），但它的证据契约可复用。方法论出处是 `competition/archive/legacy/.agents/skills/kernelgen-flagos/references/reliability-gates.md` 及其 `target-evidence.md` / `run-record.md` / `reviewer-protocol.md`；当时把它落成代码的 `evidence.py` 已随上游 PR 工具一起归档（见 [归档说明](archive/README.md)），现在这些条目按**规则**执行，不再由 CLI 检查。

对一个目标，只有两个结论：**证据齐备**，或者**不可用**（缺证据、部分覆盖、未绑定契约覆盖）。记录互相矛盾就是失败，不允许用解释把缺证据升格成通过。

1. **能力三态**：`absent` / `configured_unavailable` / `callable`。配置文件、可达主机、有效 token 都只算 `configured_unavailable`；只有当前工具注册表里真的能调用才算 `callable`。本机现状：平台通道 `callable`（`browser/submit.mjs` 在且已有记录），租用设备 `configured_unavailable`（隧道配了不等于设备能用，且这里不探测可达性）。
2. **逐目标一个证据状态**：`generated` / `tool_unavailable` / `reported_only` 无论怎么解释都不能变成 `target_validated` 或 `measured`；零覆盖、部分覆盖、未绑定契约覆盖一律 `inconclusive`，只有真实失败的用例才 `rejected`。
3. **提升要目标证据**：`compile.success`、用例集契约的精确覆盖、每例 `input_signature`、target/compiler/runtime 身份、provenance 的 `invocation_id` 与 raw hash，缺一项就 `inconclusive`。
4. **性能只给标签**：`reported_only` / `performance_reported_complete` / `performance_reported_subset` / `performance_reported_incomplete`；`official_score_computed` 恒为 `false`，官方聚合单独记录，不由本地解析器推断。
5. **平台生命周期**：`prepared → submitted → record_confirmed → evaluating → completed`。`evaluating` 是临时态——逐目标值可被修订、`aggregate_speedup` 必须为空；同一提交的 interim 行与终态修订都保留，interim 标 `superseded`，既不删除也不进聚合。**这是 15.76× / 2.43× 一类误读的结构性防线。**
6. **run record 字段纪律 + 追加不覆盖**：重试或修复必须换新 `run_id` 并用 `parent_run_id` 关联；`service_state != callable` 时不得写 `tool_name`；骨架不得编造任何值。
7. **两阶段独立审查**：A 盲审并保存原始输出，之后 B 由**不同** reviewer 挑刺并逐条处置 A 的结论。本地只能校验收据一致性（源码/基线哈希、两个 agent ID 不同、每条 A 发现都有 B 处置、覆盖记录必须带源引用 trace），**不能**证明审查真的发生；`release.json` 的 `reviews` 为空时报不可用，而不是通过。

### Chrome 提交与恢复

本轮不上传。后续按已获用户授权和 flagos-adaptation skill：Chrome 确认题目/批次、团队、可见额度、最近提交和重复包，复核最新 check 与源码/包哈希；相邻提交至少间隔 120 秒。单次上传前记录 `upload_armed`、包哈希和 preflight 到适配事件。若上传期间被打断，先查 Chrome 官方记录，再决定关联既有记录或重试，避免重复上传。共享文档不保存账号、Cookie 或密钥。

### 官方结果账本

```sh
python -m competition.adaptation record --task task78 --input OFFICIAL_OBSERVATION_JSON
python -m competition.adaptation report --task taskNN --targets
python -m competition.task78.official_history verify
```

观察 JSON 包含 `record_id`、`submitted_at`、`status`、`evidence_class: official-platform`、真实 `evidence`、`local_archive_sha256`、完整 `targets`（status/speedup/已知 source_sha256）、`pass_count` 和官方 `aggregate_speedup`。缺失结果留 pending/evaluating，未知摘要不编造；部分目标通过时聚合为空。异常排除用 `excluded` 和证据说明，原观察保留。

**逐目标账本（`report --targets`）**：聚合最优只回答"哪个包总分最高"，不回答"这颗芯片的最佳值出现在哪个包"。逐目标账本对每颗芯片给出：

- `best_eligible`（只在**全部目标通过**的包中取最优）与 `best_observed`（含未合格包的观测）——两者不同说明某个更好的观测来自一个整体不合格的包；
- `source_versions`：该芯片在该题历史上出现过几个不同的源码；**只有 1 个版本而数值仍在动，就是平台重复测量的抖动，不是版本变差**；
- `delta_vs_best_eligible` 与 `verdict`：`at_or_above_best` / `below_best_same_source`（源码字节相同，属复测）/ `below_best_changed_source`（真的换了版本）；
- `resolution` 与 `repeated_observations`：同一份源码被重复观测到的相对极差，即这颗芯片在平台上的分辨力（Task 103 实测：ascend 13.9%、hygon 2.3%、metax 1.4%）。

`limitation` 必须一并阅读：每个包对每颗芯片只有**一次**观测，且极差由被比较的那批观测自身算出，所以 `below_best_same_source` 的含义是"与复测不可区分"，**不是**"没有回退"。

历史 `results.jsonl` 不改写；新观察追加 `official-records.jsonl`，重复同一内容幂等，评测中的记录可同包追加修订，终态冲突被拒绝。Task 78 历史校验会解析原路径、tracked archive 和本地私有备份，核对 25 条结果的包/源码摘要。完整原始官方账本仍是成绩事实源。

### 结果驱动的调整（`decide`）

**先要知道分数怎么算：官方聚合 = 各目标 speedup 的算术平均**（用 Task 103 全部合格记录逐条核验，最大残差 <0.005）。因此把一颗芯片提高 Δ，总分增量是 **Δ / 目标数**，优先级必须按**绝对增量**排，而不是按相对提升——一颗 0.89× 但有余量的芯片，价值远高于一颗 18.46× 且已到顶的芯片。

```sh
python -m competition.adaptation decide --task taskNN
python -m competition.adaptation decide --task taskNN --next-package
```

每颗芯片给出：`value`（最近合格值）、`anchors`（参照锚）、`reference` + `reference_kind`、`headroom`、`marginal_aggregate_gain_upper_bound`，以及 `action`：

| action | 含义 |
|---|---|
| `blocked_no_eligible_result` | 还没有合格观测 |
| `rework_below_baseline` | **低于 1.0，比参考实现还慢**（缺陷，最高优先） |
| `rework_behind_reference` | 落后参照超过 10% |
| `rework_under_served` | 低于同期中位数的一半 |
| `hold_re_measure` | 源码没变、差异落在复测范围内，不追 |
| `keep` | 已到参照水平 |

**参照取四个锚的最大值**——缺任一个都会失真：

| 锚 | 看得到什么 | 盲区 |
|---|---|---|
| `best_eligible_ever` | 已达成过的最好值 | 一直很差的芯片看起来"没有余量" |
| `provisional_observation` | 评测中的观测（存在性证明） | 会被修订 |
| `peer_median` | 同一包里其它芯片的中位表现 | 只是同期横向对比 |
| `baseline_parity`（1.0） | 低于它就是比参考实现还慢 | 只是地板 |

**硬规则**：低于 1.0 的目标**不因"落在复测波动内"而豁免**——那是缺陷，不是噪声。

`--next-package` 给出可执行的**沿用计划**（carry-forward）：每颗芯片该沿用哪个文件、来自哪个 adaptation、依据哪条官方记录。规则是**除新假设明确指名要改的芯片外，一律沿用产出该芯片最佳值的文件字节**——不再整包手工重挑。

`headroom` 与 `marginal_aggregate_gain_upper_bound` 都是**上界**，不是预测：估算一次只动一颗芯片，而平台一次给整包打分。

Task 103 当前输出：`iluvatar`（0.17，+0.47）→ `intl_a`（0.89，+0.36）→ `kunlunxin`（1.85，+0.23）→ `metax`（4.98，+0.18），上界合计 +1.24；其余 `hold_re_measure` / `keep`。

**强制项：`check` 会对照沿用计划阻断。** `publication_check` 逐目标比较实际源码字节与该芯片"产出最佳值"的字节：

- 一致 → 通过（`carried`）；
- 不一致 → 必须在 `release.json` 里**显式声明**该芯片：`changed_targets: ["<chip>"]` 或该目标条目写 `replaces_carry_forward: true`；**并且** 该目标的 `structural_change` / `expected_mechanism` 必须**指名这颗芯片**（写芯片名或文件名），否则 `check` 报错阻断，记录 `deviated` / `undeclared_deviations`；
- 该芯片没有可用观测（无沿用计划）→ 不受约束，列进 `no_carry_forward_plan`；
- `--package-only` 只给**非阻断预览**（`carry_forward_preview`），不影响 `passed`。

边界必须一起读：芯片名检查是**文本检查**，只保证"声明过且指名了"，**不验证论证是否正确**；`forecast_missing` 会列出没有数值 `expected_speedup` 的目标（当前只提示，不阻断，因为现有 `release.json` 都没有这个字段）。
