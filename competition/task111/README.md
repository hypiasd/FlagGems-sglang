# Task 111 `conv_window_scatter_with_mask`

第 8 批（窗口 2026-10-01 20:00 – 2026-10-08 19:59，七目标、`speedup_threshold=0.1`）。

## 题目

`dst: [layers, cache, dim, K-1]`、`src: [layers, requests, draft, dim, K-1]`、
`dst_indices_raw`/`step_indices_raw`: `[requests] int32`。语义是 `out = dst.clone()`，然后对每个
`step_indices_raw[i] >= 0` 的请求做 `out[:, dst_indices_raw[i]] = src[:, i, step_indices_raw[i]]`。

**关键在于 `src` 是重叠的 `as_strided` 视图**：元素 `(l, i, s, d, k)` 位于共享缓冲区
`buffer[l, i, d, s + k]`，因此 step 轴与 window 轴共用同一条连续轴（stride 均为 1），相邻 step 重叠，
而相邻 `dim` 行相隔 `draft + K - 1` 而不是 `K - 1`。把它当成扁平连续行读会静默读错值。

## 本目录

| 文件 | 作用 |
|---|---|
| `profile.json` | 官方契约（来源 URL + 核验日期）、七目标、`core_computation: triton-only` |
| `adapter.py` | 官方参考实现的逐字转写、开发用例表、`check` |
| `conv_window_scatter_with_mask.py` | Triton 候选：单次 launch 完成，内核内反查 `slot -> request` |
| `validate_cpu.py` | 无设备的**语义**回路（不是设备/性能证据） |
| `test_task.py` | 免 torch 的离线结构检查（契约、adapter 接口、分支覆盖、无 fallback） |

## 候选的结构改动

官方参考实现是 `clone` + `nonzero` + 高级索引（含 device→host 同步）。候选改成**一次 launch**：
每个 program 拥有 `(layer, slot, row-chunk)`，读取必须保留的 `dst` 值，在内核里扫描请求表把 slot 反查成
`(request, step)`，再按 `src` 的真实 stride 读出源值；命中则写源值，否则写回 `dst` 值。没有 clone、
没有主机同步、没有假设源是连续的。

## 两条口径（必须一起读）

1. **重复 slot 不可评分**：官方参考用 `index_put_` 解析重复写者，而 torch 2.14.1 CPU 上赢家**按 layer 不同**
   （slot 31 被请求 1 和 5 写时，layer 0 保留后者、layer 1/2 保留前者）。这来自写操作的并行分块，稳定但任意。
   因此开发用例只用 `randperm` 生成**唯一** slot；候选实现"最后一个有效请求胜出"并写进文档，
   仓库不断言那个任意行为。
2. **CPU 回路只证明语义**：它执行真实 JIT 体（串行 CPU torch），检查每个输出元素恰好写一次、输入与底层
   存储未被改动、数值与参考一致；它**不**编译任何厂商编译器，也不测速度。每次运行都打印 `LIMIT` 行。

## 当前证据

- 冻结 run `20261001T163531-2a1f7c91`（快照 `9fbfbf7c…`，父 run `20261001T163405-465a540e` 是格式化前的那次冻结），候选源码 sha `d71e0d02…`。
- 9/9 语义用例通过（含非 2 幂 extents、`requests=1`、`layers=1`、fp16、全无效请求、2048 program 的
  512-slot 形状）。
- 五条否证对照全部按预期失败：删有效性判断 → 越界地址；把重叠视图当连续行 → 数值不符；
  永不命中 → 数值不符；去掉行掩码 → 重复写；不复制未命中值 → 数值不符。
- **尚无**任何目标芯片的编译或性能证据；七目标闸门未验证。
