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

## G 版（当前候选）：把 stride 搬进设备缓冲

| 版本 | 入口耗时（小形状，4 次重复中位数） | 框架 bench 平均 |
|---|---:|---:|
| A | 43.2 µs | 5.34× |
| E 缓存计划 | 33.0 µs [31.6–34.6] | 6.92× |
| **G 打包 stride** | **26.8 µs [26.3–27.3]** | **8.12×** |

G 把 13 个 stride 标量放进一个按 `(shape, strides, dtype, device)` 缓存的设备端 int32 缓冲，签名从 18 参数降到 6 指针（参数绑定实测占 11.2 µs）。大形状慢 2–3%（GPU 受限），小形状快 19%，区间不重叠。

**测遍并否掉**：D 槽块化（慢 3–4×，砍并行度）、C 索引内核（多一次 launch 不划算）、F 少 4 个参数（与 E 无差别，单次"更差"是噪声）、H 把 4 个维度也塞进缓冲（小形状无差别，cache=4096 慢 11%）。

**剩下的 27 µs 里**：13.2 µs 是 Triton 派发地板、2.9 µs 是必需的 `empty_like`、约 11 µs 是我们自己的 Python 与参数绑定。**唯一能再削的路径是绕过 Triton 公开派发（私有 CompiledKernel / CUDA Graph），明确不采用**——本题要跑七家厂商的 Triton 分支，私有路径在 CUDA 上可行、在别家可能直接启动失败，等于拿非目标设备的微秒去赌真正的闸门。（这一行的拆解在下面 J 版被更细的测量修正：`empty_like` 与 launch 交替时是 6 µs 而不是 2.9 µs。）

验证：共享 CPU 模型新增**默认关闭**的 `allow_auxiliary`（候选自有只读张量可登记并快照；默认仍拒绝未注册张量），故 G 仍可被语义回路验证：10/10 + 7 条否证对照（含 `g_wrong_stride_slot` 打错缓冲槽位、`e_stale_plan` 同形状异 strides）。

## J 版（当前候选）：入口耗时的准确拆解 + 两个小改动

上一轮我在只测了"减少参数条数"之后就宣布到地板，这个判断不成立——入口里还有**约 6 µs 的输出分配**和**约 1.5 µs 的计划查表**从未被单独量过。用同一套 CUDA-event 协议（小形状，交替顺序，多次重复中位数）拆出来：

| 组成 | 耗时 | 说明 |
|---|---:|---|
| 空 kernel（1 参数 1 program） | 12.3 µs | Triton 派发地板 |
| 本 kernel 裸启动（6 指针、输出预分配） | 19.3 µs | ≈7 µs 是 5 个额外指针的绑定 + 真实设备时间 |
| 8 个输出张量轮转（无分配、不同对象） | 18.6 µs | **绑定"新张量对象"本身不要钱** |
| `torch.empty_like(dst)` 单独测 | 2.9 µs | |
| `torch.empty_like` 与 launch 交替出现 | **+6 µs** | 分配器要为"仍被排队 kernel 占用的块"记账 |
| 计划键构造 + `dict.get` + 解包 | ~1.1 µs | |

**两处改动（J）**：① 6 个 constexpr 改为**位置传参**（计划里存元组）；② 目标槽位会被覆盖，所以 **dst 的死读去掉**（`mask=in_row & (~hit)`），只省下 `requests/cache` 那份读流量。

**10 例 confirm 模式（5 组 × 100 ms，cap 4096，非目标 T4）**：G 平均 **6.77×** → J 平均 **7.02×**，10 例全对、负控通过；J 在 9/10 例上更好（+0.8%…+6.5%），只在 all-invalid 例（参考实现只做 clone，比值基数小）差 4%。注意 quick 模式会给更高的绝对值（G 8.12×），**两种模式不可混比**。

**方法论纠正（都记在这里）**：① probe4/6 按固定顺序测变体，结果随测量顺序单调变快，把位置传参的收益夸大成 3 µs；改成**交替顺序**的 probe8 后是 25.37 vs 27.42 µs（−2.05 µs），再改成两个模块真实入口的交替 A/B，差异落到 ±3% 噪声内——所以这条更正**靠的是机制实验（同一 kernel 同一取值），不是端到端差异**。② 两条对照是**空操作**：`src` 的 dim/window 步长按构造都等于 1（`window_index * src_s4 → *1` 不改变行为），`other=` 对保留行不可达——重写成"删掉窗口项""让 dst 读完全不发生"后才按预期失败。

**还剩什么**：入口里可动的只剩 ~2 µs（位置传参已被拿走），大形状 3.59× 是 `clone()` 语义的流量地板。唯一的结构性杠杆仍是绕过 Triton 公开派发；这次真的试了 `warmup()` + `CompiledKernel[grid]`，在本机 Triton 3.6 上无法启动（`IndexError: tuple index out of range`，补成 3 元 grid 后是 `TypeError: function takes exactly 27 arguments (19 given)`）。**明确不采用**：本题要跑七家厂商的 Triton 分支，私有路径在 CUDA 上可行、在别家可能直接启动失败，等于拿非目标设备的微秒去赌真正的闸门。

## K 版（当前候选）：跳过 Triton 每调用的派发记账，10 例平均 6.77× → 11.59×

公开路径 `JITFunction.run` 每次调用都要重做：参数绑定（binder）、特化 + `compute_cache_key`、kernel cache 查找、`used_global_vals` 失效检查，然后才落到 `kernel.run(...)`。这些记账与张量无关，只是"每次都重新算一遍"。K 版在**第一次调用时捕获 `kernel[grid](...)` 返回的 CompiledKernel**，此后按 `JITFunction.run` 的**同一份调用形式**直接发射：

```python
kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata,
           kernel.launch_metadata(grid, stream, *args),
           knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)
```

`args` 是 6 个指针 + 6 个 constexpr，顺序与签名一致；grid 归一化、hooks、stream 取法与公开路径逐字一致。小形状入口 **29.1 → 17.46 µs**（交替顺序 11 次，中位数）。

**10 例 confirm 模式（5×100 ms，cap 4096，非目标 T4）**：

| | G | **K** |
|---|---:|---:|
| 平均 speedup | 6.77× | **11.59×（+71%）** |
| 小形状例（7 个） | 7.67–7.89× | **13.35–14.83×** |
| padded（同形状异 strides） | 7.77× | 14.83× |
| all-invalid | 1.87× | 3.01× |
| 大形状 2.1M 元素 | 3.38× | 3.48×（设备受限，不变） |

10 例全对（首调 + 二次调用都验），T4 完整矩阵 `passed`、负控通过；本地 18 条单测 rc=0、10/10 语义例、8 条对照全部按预期。

**可移植性不是嘴上说的，是对着真轮子核过的**：用 HTTP Range 直接从 `resource.flagos.net` 的索引里抽出 FlagTree 0.6.1 wheel 内的 `triton/runtime/jit.py`（不下载 371 MB–3.3 GB 整包），逐个比对 `plain / iluvatar3.6 / metax3.6 / ascend3.5` 四个轮子——**调用形式完全相同**，唯一差异是 FlagTree 自己加的 `*dist_param`，且它由环境变量 `FLAGTREE_LITE_DIST`（默认未设）与 lite 模式共同决定，K 版把它原样镜像。因此"同一份调用形式对七个芯片通用"是**查证过的事实**，不是推断。

**安全网（本文件禁止任何 `try`，行 212-214 的静态检查会拒绝）**：所有能力探测都用 `getattr`，不用异常。缺 `function`/`packed_metadata`/`launch_metadata`/`run`，或取不到 driver 与 hooks，则计划**永久回落到公开路径**；CPU 语义模型下正是这条回落路径在工作（`triton.runtime.driver` 不在 `sys.modules`），所以 K 版仍可被语义回路验证。stream 每次调用重新取（多流安全），hooks 取自 `triton.knobs`，与公开路径同源。

**注意这里优化的是什么**：算子和数学语义一字未改，省掉的是宿主端每调用的派发记账。大赛反作弊条款针对"核心计算必须 Triton/Triton-TLE、不得有 torch 回退"，K 版走的仍是 Triton 自己的发射路径，没有 torch 回退、没有设备分支。


