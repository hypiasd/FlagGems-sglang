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

⚡ = quick。用例表刻意覆盖 `base_e × return_lse` 的三个组合、单 shard、以及 dead-shard 净化——这些正是该题语义上最容易做错、且一次提交就要赔上整包的地方。

**容差来源**：题面只写"标准 per-dtype tolerance"。开发检查取 bf16 `atol=rtol=1.5e-2`、合并 LSE（fp32）`atol=rtol=1e-4`，数值取自 Task 103 官方题面里的全赛道表；上游单测用的是 bf16 `atol=rtol=1e-2`。**官方 harness 的真实判定仍未知。**

## 验证状态（必须分清）

已验证（本机、无 torch）：

- `competition.adaptation inspect --task task112 --refresh` → 七目标与官方一致，`contract_drift: false`；
- `competition.adaptation members --task task112 --source competition/task112` → 包布局通过，并给出"7/7 目标共用 generic"这一预期内的 warning；
- `competition.adaptation decide --task task112` → 聚合模型 `unverified`、七目标全部 `blocked_no_eligible_result`（无观测时拒绝给出边际收益）；
- `competition.task112.test_task` 19 项离线检查：profile 与官方契约一致、adapter 暴露 harness 六个接口且 `reference` 签名一致、用例表覆盖全部分支、候选模块 `__all__`/公共入口/无 `try-except`/仅使用允许的 torch 调用；
- 首轮 run 已冻结：`20261001T133618-4388ba8b`，快照 `64fe04f9…`，`source_method: agent-authored`。

未验证（当前无法验证）：

- **任何数值正确性**：本机没有 torch/triton；`competition/task112/dcp_lse_combine.py` 从未执行过。
- **任何设备行为与性能**：租用设备不可达（见 runbook 节点 61/62 的设备约束）。
- **官方用例集与 baseline 包**：平台不公开。

## 下一步

1. 需要有 torch+triton 的环境（租用设备或平台机）跑 `test`/`bench`，才能把这份候选从"未执行"变成"CPU/设备语义通过"；
2. 提交需要用户在具体批次/团队范围上的显式授权。
