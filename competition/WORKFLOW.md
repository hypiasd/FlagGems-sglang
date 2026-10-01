# FlagOS S2 执行手册

算子实验和比赛适配各有一个入口。命令从项目仓根目录运行，Python 3.10+；GPU 执行端还需要 PyTorch、Triton 和 CUDA。当前 T4 验收见 [ACCEPTANCE.md](ACCEPTANCE.md)。本轮未上传比赛包。

## 目录

- `experiments/`：冻结源码、设备执行、正确性、计时、分析和报告。
- `adaptation/`：官方契约、发布检查、独立适配、ZIP 和官方成绩账本。
- `pr/`：第二条出口（上游仓库 PR）的规则、打包、闸门、证据纪律与 PR 描述。
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

### 上游仓库 PR（第二条提交通道）

平台 ZIP 和上游仓库是两套契约。平台要一个含 `<op>.py` 与 `<op>_<chip>.py` 的压缩包，在自己的 harness 上评分；[flagos-ai/FlagGems-sglang](https://github.com/flagos-ai/FlagGems-sglang) 要同样的源码放进 dispatcher 树，带 Apache 2.0 头和 `__all__`，并通过 `basic-ci.yml`。竞赛规则是官方 `docs/CONTRIBUTING.md` 第 9 节：**只交算子文件**，不需要 tests/benchmark/docs/`conf/operators.yaml`（维护方用留出 harness 复验），但 CLA、`pre-commit run --all-files`、以及 `import flaggems_sglang` 后 `all_registered_ops()` 能列出该算子必须成立。

```sh
python -m competition.pr rules
python -m competition.pr plan --task task103 --adaptation ADAPT_ID
python -m competition.pr bundle --task task103 --adaptation ADAPT_ID --rename O=out_ptr
python -m competition.pr materialize --task task103 --adaptation ADAPT_ID
python -m competition.pr check --task task103 --adaptation ADAPT_ID
python -m competition.pr evidence --task task103 --adaptation ADAPT_ID
```

- `bundle` 把获奖源码重排进 `.local/pr/<task>/<adaptation>/`：加 Apache 头、补 `__all__ = ["<op>"]`、用官方同版本 isort+black 格式化、把超长注释/模块 docstring 折进 flake8 的 120 列、按 `--rename` 做 token 级改名（flake8 E741 等）。每一处非源码改动都记进 `bundle.json`，并在 PR 描述里逐条披露；原始获奖文件另存 `awarded/` 供比对。
- `materialize` 用 `git archive upstream/master` 展开官方树并把 bundle 文件复制进去；`check` 依次跑 structure、hygiene、ast_preservation（获奖实现的 AST 除声明过的改名外必须一致）、style（black / isort / flake8）、官方 `tools/ci_checks`（竞赛 PR 不适用的检查显式标为 out of scope）和 import smoke。
- `check` 的结论只是“本地可跑的闸门全过”，**不是**“可以合并”；未跑的闸门必须在上游 CI 上补齐。`import_smoke` 需要 torch+triton，本机通常 `unavailable`，此时会打印在设备上执行的确切命令，不得声称注册已通过。
- 样式工具在 `competition/.local/venv-pr/`（black 24.8.0 / isort 5.12.0 / flake8 7.1.0，与 `basic-ci.yml` 同版本）；可用 `--tools` 指向其他 bin 目录。没有工具时 style 闸门报 `unavailable`，不会假装通过。

### 证据纪律（复用 KernelGen 方法论）

KernelGen 的 MCP 工具对本项目不可用（专精知识库只有华为，且本会话未注册），但它的证据契约可复用：`competition/pr/evidence.py` 把这套方法论落成代码，源码出处是 `competition/archive/legacy/.agents/skills/kernelgen-flagos/references/reliability-gates.md` 及其 `target-evidence.md` / `run-record.md` / `reviewer-protocol.md`。

`competition.pr evidence` 逐目标写出 `evidence.json`（`check` 同时把它折成 `evidence` 闸门）。闸门语义：**记录互相矛盾 → `fail`；证据在这里根本不存在 → `unavailable`；齐备 → `pass`**。因此 `ready_for_pr` 始终只表示“本地能跑的闸门没有失败”。

1. **能力三态**：`absent` / `configured_unavailable` / `callable`。配置文件、可达主机、有效 token 都只算 `configured_unavailable`；只有当前工具注册表里真的能调用才算 `callable`。本机现状：平台通道 `callable`（`browser/submit.mjs` 在且已有记录），租用设备 `configured_unavailable`（隧道配了不等于设备能用，且这里不探测可达性）。
2. **逐目标一个证据状态**：`generated` / `tool_unavailable` / `reported_only` 无论怎么解释都不能变成 `target_validated` 或 `measured`；零覆盖、部分覆盖、未绑定契约覆盖一律 `inconclusive`，只有真实失败的用例才 `rejected`。
3. **提升要目标证据**：`compile.success`、用例集契约的精确覆盖、每例 `input_signature`、target/compiler/runtime 身份、provenance 的 `invocation_id` 与 raw hash，缺一项就 `inconclusive`。
4. **性能只给标签**：`reported_only` / `performance_reported_complete` / `performance_reported_subset` / `performance_reported_incomplete`；`official_score_computed` 恒为 `false`，官方聚合单独记录，不由本地解析器推断。
5. **平台生命周期**：`prepared → submitted → record_confirmed → evaluating → completed`。`evaluating` 是临时态——逐目标值可被修订、`aggregate_speedup` 必须为空；同一提交的 interim 行与终态修订都保留，interim 标 `superseded`，既不删除也不进聚合。**这是 15.76× / 2.43× 一类误读的结构性防线。**
6. **run record 字段纪律 + 追加不覆盖**：重试或修复必须换新 `run_id` 并用 `parent_run_id` 关联；`service_state != callable` 时不得写 `tool_name`。`evidence --run-record FILE` 校验既有记录，否则写出 null 骨架 `run-record.template.json`（骨架不编造任何值）。
7. **两阶段独立审查**：A 盲审并保存原始输出，之后 B 由**不同** reviewer 挑刺并逐条处置 A 的结论。本地只能校验收据一致性（源码/基线哈希、两个 agent ID 不同、每条 A 发现都有 B 处置、覆盖记录必须带源引用 trace），**不能**证明审查真的发生。当前 `release.json` 的 `reviews` 为空，所以 evidence 闸门报 `unavailable`，而不是通过。

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
