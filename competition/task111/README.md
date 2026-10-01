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

## 三方案结构对比（A / B / C）

同一语义的三种并行分解，全部通过 9/9 语义用例（`--allow-rewrite` 用于两趟方案）：

| 方案 | 分解 | launches | programs(9 例) | 重写元素 | 解析访存元素 |
|---|---|---:|---:|---:|---:|
| **A** 融合单趟 | `(layer, slot, row-chunk)`，核内扫描请求表反查 | **1** | 2862 | 0 | 10,809,092 |
| **B** 两趟 | 拷贝全表 + 按请求散射 | 2 | 3153 | 220,832 | 11,063,794 |
| **C** 索引内核 | 建 `slot->request` 索引 + 融合应用 | 2 | 2874 | 0 | 10,669,792 |

- **B 被严格支配**：访存更多（+2.4%）、额外重写 220,832 个元素、还要多一次 launch。它的唯一优势是"按请求并行"看起来更直观。
- **A 与 C 的差是 1.3%**：A 每 program 付 `2R` 次标量加载，C 用一次额外 launch + `cache` 个 int32 换到 O(1) 查表。**平台对同一份字节的复测极差约 16%，所以 A/C 的差异单次不可分辨**——该选择必须靠结构账和推理，而不是刷分。
- 代价模型第一版把**被掩码的通道**也算成访存，结论自相矛盾（B 反而"更省"）；是实测的重写数把它暴露出来，改成只计真实发生的访问后才自洽。这条比结论更值得记。

证据：A/B/C 分别冻结在 run `20261001T164454-eb62ff42` / `20261001T164454-e940efb2` / `20261001T164454-ee3de703`，各自 `cpu-semantic.txt`（语义 + 该方案专属否证对照）与 `structural-comparison.txt`。新增否证对照：`b_drop_valid`、`b_no_copy`（缺覆盖 528）、`c_no_clamp`、`c_ignore_validity` 全部按预期失败。

**口径**：这些是结构计数，不是速度，也不是任何目标芯片的证据；`--allow-rewrite` 是显式的多趟声明（默认仍是"每元素恰好写一次"的严格模式）。

## 优化记录（E 版为当前候选，T4 实测）

`cuda_timer` 用 events 夹住 N 次调用取平均，所以在 kernel 只有几 µs 时**测到的就是 Python 侧启动开销**。设备微基准（T4，同口径）：

| 项 | 耗时 |
|---|---:|
| 空 Triton kernel（1 参数 1 program） | **12–14 µs** ← 派发地板 |
| 本 kernel 的裸启动（参数预先构造） | 22.3 µs |
| 方案 A 的公开入口 | 43.2 µs |
| **方案 E 的公开入口** | **32–34 µs** |
| `torch.empty_like` | ~3 µs |

E = 保留 A 的映射（每 program 一个 `(layer, slot, chunk)`），把形状相关计算按 `(shape, strides, dtype)` 缓存、热路径压成五参调用。框架 bench（quick）实测：**A 5.34× → E 6.92×**（三例 +28~31%），T4 完整开发矩阵（10 例）`full_matrix_passed=True`、负控通过。

**测过并否掉的方案**：

| 变体 | 结果 | 原因 |
|---|---|---|
| D 槽块化反查（53 programs vs 2862） | 84.9 / 381.9 / 3276.9 µs | 砍 program 数 = 砍并行度，单 program 扛 6.4 万元素，慢 3–4× |
| C 索引内核 + 应用 | 69.9 / 81.3 / 591.6 µs | 多一次 launch 约 30 µs，而扫描并不是瓶颈 |
| F E 去掉四个输出步长参数 | 35.9 / 38.1 µs | 少四个参数不多于多一个特化键的代价，更慢 |

**大形状为什么到顶**：`cache=4096` 时张量 16.7M 元素，`clone()` 语义强制读整张 `dst` + 写整张 `out`（约 135 MB），参考实现同样两趟 → 580 µs vs 782 µs（1.35×）就是这条契约的流量比，不是调参空间。

**E 的新增分支与对照**：计划缓存引入"缓存复用"分支，为此补了 `l2-c16-r5-d3-dim8-w3-padded`（**同形状、不同 strides**）——没有它时 `e_stale_plan` 对照**照样通过**（说明分支根本没被考到）；补上后该对照按预期失败。同时删掉了一个不可达的 out-stride 守卫（同一 `dst` 的 `empty_like` 步长恒定）。

全部为**非目标设备**证据；官方分数仍以平台为准。
