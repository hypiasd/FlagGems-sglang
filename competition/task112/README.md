# Task 112 `dcp_lse_combine`

Split-KV attention 的 "combine" 一半：把 `N` 个 decode-context-parallel rank 的部分注意力输出按各自的 LSE 加权合并。

## 契约来源

- 官方只读 API：`https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/dcp_lse_combine`，2026-10-01 抓取（原始响应存于被忽略的 `competition/.local/contracts/task112/`）。
- 签名：`dcp_lse_combine(recv_output[N,B,H,D], recv_lse[N,B,H] fp32, is_lse_base_on_e: bool, return_lse: bool) -> (out[B,H,D], out_lse[B,H] | None)`。
- 语义要点：`m = max_i lse[i]`；**NaN 或 `+inf` 的 shard 替换为 `-inf`，即无贡献**；若全部 shard 都是 `-inf` 则强制 `m = 0`（题面说明其用例不会出现全 dead 的 `(batch, head)`，所以参考实现允许在那里出现 `0/0`）；`w = exp(lse - m)`，`is_lse_base_on_e=False` 时改用 `exp2`/`log2`（FlashInfer 约定）；fp32 累加，输出携带 `recv_output.dtype`；`return_lse=False` 时第二个返回值为 `None`。
- 反作弊：核心计算必须完全基于 Triton / Triton-TLE，禁止 try/except、条件分支或设备判断回退到 PyTorch。
- 七目标与批次：天数/沐曦/海光/昆仑芯/华为/国际 A/B；第 8 批窗口 `2026-10-01T20:00:00`–`2026-10-08T19:59:59`（北京时间），`speedup_threshold=0.1`；抓取时榜一平均 7.845×。

## 上游对照（量级参考，不是官方用例）

- baseline：`sgl-project/sglang` @ `41cbe65d` 的 `python/sglang/kernels/ops/attention/dcp_kernels.py`，`dcp_lse_combine_triton` + `_dcp_lse_combine_kernel`；grid `(B, H)`，`N`/`HEAD_DIM` 为 constexpr + `tl.static_range`，两遍（先 max 后加权），无 BLOCK/autotune 参数。其签名与返回结构与官方题面一致。
- 上游单测：`test/registered/kernels/ops/attention/test_dcp_lse_combine.py`；输入 `recv_output` 为 **bf16**、`recv_lse` 为 **fp32**，形状 `(N,B,H,D)` 覆盖 (2,4,8,64)、(4,8,16,128)、(8,4,8,128)、(8,4,8,512)、(2,64,16,128)、(4,8,8,512)，另有 `N=1`、`return_lse=True` 与 FlashMLA 自然对数修正用例。
- **官方 `cases.py` / `baseline.py` 未公开**（`/cases`、`/download` 实测 404），所以 `case_coverage: development-assumptions`、`official_cases_complete: false`。

## 开发用例矩阵

| id | N | B | H | D | base_e | return_lse | 覆盖 |
|---|---|---|---|---|---|---|---|
| `n2-base-e` ⚡ | 2 | 4 | 8 | 64 | ✓ | | 基本路径、无 LSE 返回 |
| `n4-base-e-lse` ⚡ | 4 | 8 | 16 | 128 | ✓ | ✓ | 合并 LSE 输出 |
| `n8-base2-lse` ⚡ | 8 | 4 | 8 | 128 | | ✓ | `exp2`/`log2` 分支 |
| `n8-d512` | 8 | 4 | 8 | 512 | ✓ | | 大 `HEAD_DIM` |
| `n2-b64` | 2 | 64 | 16 | 128 | ✓ | | 大 batch |
| `n4-d512-base2` | 4 | 8 | 8 | 512 | | ✓ | base-2 + 大 D |
| `n1-single` | 1 | 4 | 8 | 64 | ✓ | ✓ | 单 shard 退化（`w=1`） |
| `n4-dead-shard` | 4 | 4 | 8 | 128 | ✓ | | NaN 与 `+inf` shard 必须消失 |
| `n3-b1h3-d96` | 3 | 1 | 3 | 96 | ✓ | ✓ | 非整除位置数（1 个被掩码行）+ 非 2 幂 `D`（掩码列） |
| `n7-b7h1-base2` | 7 | 7 | 1 | 64 | | ✓ | 非整除位置数 + `H=1` + base-2 |

⚡ = quick。用例表刻意覆盖 `base_e × return_lse` 的三个组合、单 shard、以及 dead-shard 净化——这些正是该题语义上最容易做错、且一次提交就要赔上整包的地方。

后两例是**补上回路自己暴露的覆盖漏洞**后加的：公开形状的 `B*H` 都能被合理的分块整除，于是"把多个位置折进一个 block"的变体永远不跑它的行掩码——删掉掩码的否证对照照样通过。现在这两个用例专门走掩码路径，否证对照立刻失败（见下）。

**容差来源**：题面只写"标准 per-dtype tolerance"。开发检查取 bf16 `atol=rtol=1.5e-2`、合并 LSE（fp32）`atol=rtol=1e-4`，数值取自 Task 103 官方题面里的全赛道表；上游单测用的是 bf16 `atol=rtol=1e-2`。**官方 harness 的真实判定仍未知。**

## 验证状态（必须分清）

### 已验证

**CPU 语义证据（本机，torch 2.14.1 CPU，无 GPU）**

```sh
competition/.local/.venv-cpu/bin/python competition/task112/validate_cpu.py --verbose
```

共享 CPU 语义模型 `competition/experiments/cpu_model.py`（与 Task 60/78 同一实现）逐 program 串行执行真实的 Triton JIT 体，检查：真实 wrapper 与启动路径、每个输出元素恰好写一次且在返回视图内、输入与底层存储未被改动、以及数值与官方参考在本题容差内一致。

- **10/10 通过**，含 `n4-dead-shard`（NaN 与 `+inf` 必须消失）、`n8-base2-lse`（`exp2`/`log2` 分支）、`n1-single`（单 shard 退化）与两种 `return_lse` 返回形态；证据（含命令、torch 版本、逐例结果）保存在忽略目录 `.local/runs/20261001T134448-d77aa81e/cpu-semantic.txt`（8 例版）与 `.local/runs/20261001T135708-0bb18ad8/cpu-semantic.txt`（10 例版）。
- **否证对照**（证明验证器真的会咬，而不是空过）：删掉 NaN/+inf 净化 → 在 `n4-dead-shard.out` 失败；把 base-2 分支改成总是 `exp` → 在 `n8-base2-lse.out` 失败；删掉输出行掩码 → 在 `n3-b1h3-d96` 报 `store: active address outside storage`；把 `position < B*H` 守卫改回 `position < H` → 在 `n7-b7h1-base2` 报写覆盖缺 384 个元素。四条都记录在最新证据文件里。
- **为跑通它给共享模型补了加法项**（不改变既有检查语义）：`Ptr.dtype.element_ty`（`value.to(out_ptr.dtype.element_ty)` 需要）、`tl.exp/exp2/log/log2/zeros/full`，以及 `tl.max` 的单轴归约（`vec` 变体要把 `PAIRS` 个 LSE 一次读成向量再取最大值）。补之前 CPU 模型会直接对未支持算子报 `NotImplementedError`；改动后 `flake8 competition/experiments/cpu_model.py` 的告警数与改动前一致（49 条，全部是既有项）。

**契约与结构**

- `competition.adaptation inspect --task task112 --refresh` → 七目标与官方一致，`contract_drift: false`；
- `competition.adaptation members --task task112 --source competition/task112` → 包布局通过，并给出"7/7 目标共用 generic"这一预期内的 warning；
- `competition.adaptation decide --task task112` → 聚合模型 `unverified`、七目标全部 `blocked_no_eligible_result`（无观测时拒绝给出边际收益）；
- `competition.task112.test_task` 20 项离线检查：profile 与官方契约一致、adapter 暴露 harness 六个接口且 `reference` 签名一致、用例表覆盖全部分支（含两个非整除/非 2 幂探针）、候选模块 `__all__`/公共入口/无 `try-except`/仅使用允许的 torch 调用。

**冻结 run**

| run | 快照 | 源码 sha256 | 假设 |
|---|---|---|---|
| `20261001T133618-4388ba8b` | `64fe04f9…` | 编辑前版本 | baseline 同构两遍实现可复现官方语义 |
| `20261001T134448-d77aa81e`（parent 上一条） | `b8a84e2d…` | `db712e60…` | 去掉 `program_id` 的 int64 提升与 `return_lse=False` 时的无用分配后语义不变 |
| `20261001T135028-3810bb88`（parent 上一条） | `4d6c9d8c…` | `a75ba0ea…` | 每 program 处理 `PAIRS` 个 `(b,h)` 位置可把 program 数降 70%，且逐位一致 |
| `20261001T135708-0bb18ad8`（parent 上一条） | `32a95b0f…` | `349242de…` | 把 position 轴折进 block（`[PAIRS, D_BLOCK]` 宽载入）可按 `PAIRS` 倍减少位置级访存指令数，且逐位一致 |
| `20261001T141254-4ad3ec2b`（parent 上一条） | `404bf4e1…` | `e9f4e870…` | 在线单遍（running max + 重缩放）消掉第一遍 max，访存指令再降到 `2N+1`，代价是每 shard 两个 `exp` 与串行依赖 |

### 结构变体：每 program 多个 `(b,h)`

候选在 `competition/.local/candidates/task112-pairs/`（含假设与可证伪条件）。动因：公开形状下每个位置只有 `N*D` 次读、`D` 次写；小端只有几百个元素，而昆仑芯后端的成本模型是"program 之间不重叠、每条访存指令串行"，program 数直接决定串行长度。

| 用例 | 种子（1 位置/program） | 变体（`PAIRS`=4 或 1） | program 数 |
|---|---:|---:|---:|
| n2-base-e | 32 | 8 | −75% |
| n4-base-e-lse | 128 | 32 | −75% |
| n8-base2-lse | 32 | 8 | −75% |
| n8-d512（D≥256 → `PAIRS`=1） | 32 | 32 | 0 |
| n2-b64 | 1024 | 256 | −75% |
| n4-d512-base2（D≥256） | 64 | 64 | 0 |
| n1-single | 32 | 8 | −75% |
| n4-dead-shard | 32 | 8 | −75% |
| **合计** | **1376** | **416** | **−70%** |

**这是结构事实，不是速度测量**：`PAIRS` 的阈值（D<256 用 4）是待设备扫参的猜测；寄存器压力（`PAIRS*HEAD_DIM` 个 fp32 同时存活）与每位置多出的整除/取模都可能让它变慢。设备上无提速即否证。

### 结构变体二：把 position 轴折进 block

候选在 `competition/.local/candidates/task112-vec/`。第一个变体只改**调度粒度**（program 数少了，位置级访存指令总数不变）；第二个变体改**访存形状**：一个 program 用一条 `[PAIRS, D_BLOCK]` 宽载入吃下 `PAIRS` 个位置，LSE 也读成向量再用 `tl.max` 取最大，于是位置级访存指令数本身按 `PAIRS` 倍下降。两个假设因此在设备上可分离：`pairs` 单独检验调度粒度，`vec` 检验指令数。

（下表按源码算术数出，**不是仪表计数，也不是测量**：`seed`/`pairs` 每位置 `3N+1` 条访存指令（返回 LSE 时 +1），`vec` 每 program `3N+1` 条、每条覆盖 `PAIRS` 个位置。）

| 用例 | positions | seed program | pairs program | vec program | seed 指令 | vec 指令 |
|---|---:|---:|---:|---:|---:|---:|
| n2-base-e | 32 | 32 | 8 | 2 | 224 | 14 |
| n4-base-e-lse | 128 | 128 | 32 | 16 | 1792 | 224 |
| n8-base2-lse | 32 | 32 | 8 | 4 | 832 | 104 |
| n8-d512 | 32 | 32 | 32 | 16 | 800 | 400 |
| n2-b64 | 1024 | 1024 | 256 | 128 | 7168 | 896 |
| n4-d512-base2 | 64 | 64 | 64 | 32 | 896 | 448 |
| n1-single | 32 | 32 | 8 | 2 | 160 | 10 |
| n4-dead-shard | 32 | 32 | 8 | 4 | 416 | 52 |
| n3-b1h3-d96 | 3 | 3 | 1 | 1 | 33 | 11 |
| n7-b7h1-base2 | 7 | 7 | 1 | 1 | 161 | 23 |
| **合计** | **1386** | **1386** | **419** | **206** | **12482** | **2182** |

即 `vec` 只花种子 17.5% 的位置级访存指令，`pairs` 用 30% 的 program 跑同样的指令数。`PAIRS = 1024/D`（上限 16）是为了把 `PAIRS*D_BLOCK` 个 fp32 累加器压到约 4 KiB 的猜测值，同样待设备扫参；2-D 索引算术、向量归约与寄存器压力都可能吃掉收益。**设备上无提速即否证。**

### 结构变体三：在线单遍合并

候选在 `competition/.local/candidates/task112-online/`。前两个变体都在改**访存**；这个变体改**遍数**：用 running max 加标准重缩放

```
m' = max(m, lse_i);  s = s*exp(m-m') + exp(lse_i-m');  a = a*exp(m-m') + exp(lse_i-m')*partial_i
```

把第一遍求 max 整个消掉，每 program 访存指令从 `3N+1` 降到 `2N+1`。代价是**算术与依赖**：每 shard 两个 `exp` 加两个 select，而且第 i 个 shard 的累加器依赖第 i−1 个的 max（前三个变体没有这个循环依赖）。

| | 位置级访存指令（10 例表） | 超越函数次数 | 循环依赖 |
|---|---:|---:|---|
| 种子 / `pairs` | 12482 | 3610 | 无 |
| `vec` | 2182 | 3610 | 无 |
| `online` | **1542（12.4%）** | **7220（×2）** | 有 |

两个 select 专门处理 `m` 还是 `-inf` 的时刻——那里 `exp(-inf - -inf)` 是 NaN，会毒化累加器。**它们确实被用例覆盖**：`n4-dead-shard` 让 shard 0 死亡，于是第一次迭代就在 `m == -inf` 下运行；删掉任一 select 都会在该用例失败（两条否证对照已验证）。唯一**只靠代数论证、未经验证**的是"所有 shard 都死"的位置：harness 的容差比对没有 NaN 相等模式，期望值是 NaN 的用例断言不了，而题面声明用例里不存在这种位置。

### CPU 回路这一轮抓到的四个错误（都已修复或作为否证保留）

1. **NaN/+inf 净化缺失** → `n4-dead-shard.out` 失败（否证 A）。
2. **base-2 分支被忽略** → `n8-base2-lse.out` 失败，最大相对差 480×（否证 B）。
3. **真实 bug：越界守卫写错**。`pairs` 变体最初把守卫写成 `position < H`（应为 `position < B*H`）。原先只能用 `PAIRS=5` 的合成配置强制走非整除路径，**写覆盖检查**才抓住它：`n2-base-e.out: write coverage missing=1536`（输出元素被整块漏写，而数值比对根本不会发现未初始化内存）。补上 `n7-b7h1-base2` 后，用例表用自己的 `PAIRS` 就能复现：`write coverage missing=384`。
4. **覆盖漏洞（元问题）**：`vec` 变体删掉输出行掩码后**照样 10/10 通过**——因为开发用例全是整除的，掩码行从未出现。这说明"变体通过"当时并不覆盖它自己的分支。补 `n3-b1h3-d96`、`n7-b7h1-base2` 两个非整除/非 2 幂用例后，该否证对照才按预期失败。教训写进工作流：**每个结构性变体都要配一个能杀掉它的否证对照，且对照必须真的失败过。**

## 未验证（当前无法验证）

- **目标芯片正确性、设备行为与任何性能数据**：CPU 语义模型明确不提供这些（`LIMIT` 行每次都打印），租用设备仍不可达（`doctor --device t4` → `SSH scratch creation failed`）。
- **官方用例集与 baseline 包**：平台不公开；本表的形状/容差来源是其"参考"而非官方判定。
- **性能**：全部 program 数/指令数都是**结构性计数**，不是速度。当前候选 `dcp_lse_combine.py` 仍是 baseline 同构起点（没有换成任何变体），三个变体都只有 CPU 语义证据——`online` 尤其分不清：它同时减少访存、增加超越函数、引入循环依赖，方向相反的三种效应只能在设备上分辨。

## 下一步

1. 设备扫参（一旦有设备）：`pairs`/`vec`/`online` × `PAIRS ∈ {1,2,4,8,16}` × 公开形状表 × 两个 `is_lse_base_on_e` 值，判定变体是否真的更快；`vec` 的关键观测量是寄存器溢出/占用率拐点，`pairs` 的关键观测量是 program 粒度带来的调度收益，`online` 的关键观测量是访存减少能否盖过翻倍的超越函数与循环依赖；
2. 无设备时继续做语义安全的结构改动，每次改动新建 run 并重跑 `validate_cpu`（含**会失败的**否证对照），结构收益按上表的算术口径记账；
3. 真正的性能判定仍需要设备或平台评测：租用设备需要新的隧道端点，平台评测需要用户在批次/团队范围内的显式授权；提交 Task 112 仍需用户显式授权。

