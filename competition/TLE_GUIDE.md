# Triton / Triton-TLE 指南吸收与 Task112 全芯片行动清单

来源：[赛事指南：安装 Triton 或 FlagTree、了解 Triton-TLE](https://jwolpxeehx.feishu.cn/wiki/X6CXwJqUoiJURHkw6KIc0y7InTf)（2026-10-01 通过用户已登录的 Arc 阅读）。正文保存在被忽略的本机目录，不提交账号或 Cookie。

## 0. 读取方式的教训

纯文本剪贴板（`pbpaste`）把这份文档按 GBK 落地，**表格里的 ✅ 全部变成 `?` 或空白**，导致第一版吸收误判"表格没内容/昆仑芯不在表里"。同一份内容用 `public.html` 剪贴板格式读取即可完整还原 60 行单元格边界。**结论：读带勾选符号的表格必须用 HTML/富文本，不能信纯文本拷贝。** 本次矩阵即由 HTML 剪贴板解析得到。

## 1. 指南明确的信息

- 允许普通 Triton 或 Triton-TLE；用 TLE 不是强制要求。
- 指南给出的 FlagTree 安装版本（推荐安装版本，**不等于**赛事运行时版本）：

| 芯片 | 指南版本 | Python |
|---|---|---|
| 华为昇腾 | 0.6.1+ascend3.5 | 3.11 |
| 沐曦 | 0.6.1+metax3.6 | 3.12 |
| 海光 | 0.6.1+hcu3.6 | 3.10 |
| 天数 | 0.6.1+iluvatar3.6 | 3.12 |
| 燧原 | 0.6.1+enflame3.6 | 3.12 |
| 国际通用 A | 0.6.1 | 3.12 |

- 包索引 `https://resource.flagos.net/repository/flagos-pypi-hosted/simple`；安装步骤会先卸载 Triton，只在目标隔离环境执行，不动当前共享环境。
- TLE 分三层：TLE-Lite（少量改动扩展 Triton 内核）、TLE-Struct（显式定义计算与数据的结构映射）、TLE-Raw（厂商原生语言）。是不同控制层次，不是收益递增的保证。
- 指南提醒：用 TLE 时昇腾 DSA 与通用 GPU 需不同实现，并提到在同一提交文件内包含两类代码；实现仍须满足本题静态检查与路由契约，不因此引入本题禁止的设备分支或 torch 回退。
- 手写、智能体、KernelGen 都是可选开发方式，指南未要求使用 KernelGen。

## 2. 逐芯片支持矩阵（指南原文，60 行）

原语支持按芯片不同，是**开发前必须核验**的能力表，不是收益承诺。计数：国际通用芯片A 33、燧原 26、华为昇腾 20、天数 19、海光 7、沐曦 2。

注意：**本表的六列是华为昇腾、沐曦、海光、天数、燧原、国际通用芯片A**。昆仑芯与国际通用芯片B **不在表内**——那是"指南未列出"，不是"已验证不支持"。

| 原语 | 说明 | 华为昇腾 | 沐曦 | 海光 | 天数 | 燧原 | 国际通用芯片A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tle.load(is_async=True)` | 从GMEM异步加载 |  |  |  |  | ✅ | ✅ |
| `tle.extract_tile` | 将 tensor 按 sub-tile 网格切分并提取指定坐标的子块 |  | ✅ |  | ✅ | ✅ | ✅ |
| `tle.insert_tile` | 将子块插入 tensor 指定坐标 |  | ✅ |  | ✅ | ✅ | ✅ |
| `tle.range` | 循环范围控制（类似Python range） |  |  |  |  |  | ✅ |
| `tle.device_mesh` | 定义物理设备拓扑结构 |  |  |  |  | ✅ | ✅ |
| `tle.sharding` | 声明 tensor 在 Device Mesh 上的分布状态 |  |  |  |  | ✅ | ✅ |
| `tle.shard_id` | 获取当前 shard 在指定维度的 rank |  |  |  |  | ✅ | ✅ |
| `tle.distributed_barrier` | 子 mesh 同步屏障 |  |  |  |  | ✅ | ✅ |
| `tle.remote` | 获取远程设备上 tensor 的句柄 |  |  |  |  | ✅ | ✅ |
| `tle.reshard` | 在Device Mesh上重新分布tensor |  |  |  |  |  |  |
| `tle.distributed_dot` | 分布式矩阵乘/点积 |  |  |  |  |  |  |
| `tl.load（for local_ptr）` | 通过local_ptr读SMEM |  |  | ✅ | ✅ | ✅ | ✅ |
| `tl.store（for local_ptr）` | 通过local_ptr写SMEM |  |  | ✅ | ✅ | ✅ | ✅ |
| `tl.atomic_add/and/cas/max/min/or/xchg/xor（for local_ptr）` | 通过 local_ptr 进行原子操作 |  |  | ✅ | ✅ | ✅ | ✅ |
| `tle.cumsum` | 排他性累积和 + 总和，返回 (exclusive_sum, total_sum) |  |  | ✅ | ✅ | ✅ | ✅ |
| `tle.argsort` | 返回排序后的索引 |  |  |  |  |  |  |
| `tle.pipe` | 创建显式生产者-消费者数据流管道 |  |  |  | ✅ | ✅ | ✅ |
| `tle.pipe.reader` | 创建管道消费者端点 |  |  |  | ✅ | ✅ | ✅ |
| `tle.pipe.reader.wait` | 消费者等待管道数据就绪 |  |  |  | ✅ | ✅ | ✅ |
| `tle.pipe.reader.release` | 消费者释放已消费的buffer |  |  |  | ✅ | ✅ | ✅ |
| `tle.pipe.writer` | 创建管道生产者端点 |  |  |  | ✅ | ✅ | ✅ |
| `tle.pipe.writer.acquire` | 生产者获取可写buffer |  |  |  | ✅ | ✅ | ✅ |
| `tle.pipe.writer.commit` | 生产者提交已写buffer |  |  |  | ✅ | ✅ | ✅ |
| `tle.pipe.writer.close` | 生产者关闭管道 |  |  |  | ✅ | ✅ | ✅ |
| `tle.gpu.alloc` | 在 SMEM/TMEM 上分配缓冲区 |  |  | ✅ | ✅ | ✅ | ✅ |
| `tle.gpu.alloc(with alias)` | 在SMEM/TMEM上分配缓冲区并启用别名 |  |  |  |  |  |  |
| `tle.gpu.copy` | GMEM ↔ SMEM 数据搬运 |  |  | ✅ | ✅ | ✅ | ✅ |
| `tle.gpu.local_ptr` | 构建 SMEM 缓冲区的指针视图 |  |  | ✅ | ✅ | ✅ | ✅ |
| `tle.gpu.local_ptr(for remote)` | 构建远程GPU SMEM缓冲区的指针视图 |  |  |  |  | ✅ | ✅ |
| `tle.gpu.memory_space` | 声明 GPU 缓冲区所在内存空间（SMEM / TMEM） |  |  |  | ✅ | ✅ | ✅ |
| `tle.gpu.set_layout` | 设置GPU缓冲区内存布局 |  |  |  |  |  |  |
| `tle.gpu.warp_specialize` | warp 特化执行编排，同一 CTA 内不同 warp 分区执行不同 JIT 函数 |  |  |  | ✅ | ✅ | ✅ |
| `tle.gpu.alloc_barrier(s)` | 分配GPU屏障 |  |  |  |  |  | ✅ |
| `tle.gpu.barrier_wait` | 等待GPU屏障 |  |  |  |  |  | ✅ |
| `tle.gpu.barrier_arrive` | 到达GPU屏障（不等待） |  |  |  |  |  | ✅ |
| `tle.gpu.wgmma` | Warp Group Matrix Multiply-Accumulate（Hopper+ WMMA指令） |  |  |  |  |  | ✅ |
| `tle.gpu.wgmma_wait` | 等待WGMMA操作完成 |  |  |  |  |  | ✅ |
| `tle.gpu.copy(with barrier)` | 带屏障的GMEM↔SMEM数据搬运 |  |  |  |  |  | ✅ |
| `tle.dsa.alloc` | 在 DSA 本地内存分配缓冲区 | ✅ |  |  |  |  |  |
| `tle.dsa.copy` | GMEM ↔ 本地缓冲区双向数据搬运· | ✅ |  |  |  |  |  |
| `tle.dsa.local_ptr` | 构建 DSA 本地缓冲区的指针视图 |  |  |  |  |  |  |
| `tle.dsa.local_ptr (for remote)` | 构建 DSA 本地缓冲区的指针视图 |  |  |  |  |  |  |
| `tle.dsa.to_tensor` | DSA 缓冲区 → tensor 视图 | ✅ |  |  |  |  |  |
| `tle.dsa.to_buffer` | tensor → DSA 缓冲区 | ✅ |  |  |  |  |  |
| `tle.dsa.add/sub/mul/div/max/min` | 逐元素加减乘除取最大取最小值 | ✅ |  |  |  |  |  |
| `tle.dsa.pipeline` | DSA 流水线循环 | ✅ |  |  |  |  |  |
| `tle.dsa.parallel` | DSA 并行循环 | ✅ |  |  |  |  |  |
| `tle.dsa.hint` | DSA 编译期提示（context manager） | ✅ |  |  |  |  |  |
| `tle.dsa.extract_slice` | 从缓冲区提取切片 | ✅ |  |  |  |  |  |
| `tle.dsa.insert_slice` | 将切片插入缓冲区 | ✅ |  |  |  |  |  |
| `tle.dsa.extract_element` | 提取单个元素 | ✅ |  |  |  |  |  |
| `tle.dsa.subview` | 子视图 | ✅ |  |  |  |  |  |
| `tle.dsa.ascend.{UB,L1,L0A,L0B,L0C}` | 昇腾五级片上存储 | ✅ |  |  |  |  |  |
| `tle.dsa.ascend.PIPE` | 昇腾DSA流水线控制 | ✅ |  |  |  |  |  |
| `tle.dsa.ascend.sync_block_set` | 昇腾多核块同步设置 | ✅ |  |  |  |  |  |
| `tle.dsa.ascend.sync_block_wait` | 昇腾多核块同步等待 | ✅ |  |  |  |  |  |
| `tle.dsa.ascend.sync_block_all` | 昇腾所有核块全局同步 | ✅ |  |  |  |  |  |
| `tle.dsa.ascend.sub_vec_id` | 昇腾子向量ID | ✅ |  |  |  |  |  |
| `tle.dsa.ascend.sub_vec_num` | 昇腾子向量数量 | ✅ |  |  |  |  |  |
| `tle.dsa.ascend.compile_hint` | 昇腾DSA编译提示 | ✅ |  |  |  |  |  |

## 3. 与 Task112 七个目标的对应

| Task112 目标 | 指南列名 | 支持原语数 | 当前官方值 | decide 结论 |
|---|---|---|---|---|
| ascend | 华为昇腾 | 20（全部 `tle.dsa.*` + 五级片上存储/PIPE/同步） | 2.84 | `rework_behind_reference`，余量最大 |
| metax | 沐曦 | 2（`extract_tile`/`insert_tile`） | 5.38 | `rework_behind_reference` |
| intl_b | 国际通用芯片B | **未列出** | 8.25 | `rework_behind_reference` |
| kunlunxin | 昆仑芯 | **未列出** | Failed | `blocked_no_eligible_result` |
| iluvatar | 天数 | 19（pipe 全套、local_ptr 读写/原子、cumsum、warp_specialize） | 17.92 | `keep` |
| hygon | 海光 | 7（local_ptr 读写/原子、cumsum、gpu.alloc/copy/local_ptr） | 17.86 | `keep` |
| intl_a | 国际通用芯片A | 33（异步 load、pipe 全套、gpu.* 全套、wgmma、barrier、device_mesh） | 10.88 | `keep` |

燧原列有 26 个原语，但**燧原不是 Task112 的目标**（题目目标为七颗，平台表格另有燧原列），对本题无操作价值。

## 4. 不得由指南推断的内容

1. 未列出的芯片（昆仑芯、国际 B）能力记 **unknown**，不写 unsupported。
2. 本机单独下载检查的 FlagTree `0.7.0+xpu3.6` 不是赛事版本；其 `TLE_SUPPORTED_PRIMITIVES = []` 只说明该构建注册表为空。
3. 该构建在 `pm.run(mod, 'make_ttxir')` 异常时统一包装成 `OutOfResources(0, 0, 'uni_sram ...')`；本题报错的 Required=0 / Hardware limit=0 **不证明真实 SRAM 耗尽**。
4. 支持表存在、CPU 语义通过、访存指令数下降都不证明目标编译通过或性能提高。

## 5. 每轮检查清单

### A. 环境与能力先行
- 记录 target、编译器包版本/构建、Python、运行时、硬件、来源；平台未公开的字段写 unknown，不用 wheel 名称补造。
- 逐目标记录原语能力：已验证 / 未知 / 已验证不支持，并附来源。
- 先找匹配版本与目标的官方最小示例，验证 import、JIT 编译与执行；别的芯片成功不能替代本目标验证。

### B. 先兼容，再性能
- 编译失败：保存失败用例、完整 pass 名、定位行与版本；优先缩成最小复现，不把资源包装标签当根因。
- 兼容性假设与性能假设分别记录；一次改一个可识别的结构，并写明什么结果会否定它。
- 兼容性通过后再看瓶颈：普通 Triton → 必要时 Lite → 证据充分时 Struct/Raw；不为用 TLE 而凭空加管道、搬运或同步。

### C. 验证与提交
- CPU 检查数值、净化规则、base-e/base-2、返回 LSE、stride/mask、覆盖与输入不变性，并带针对新分支的有效负控。
- 冻结源码与环境证据，逐目标计算源哈希；非目标芯片保留已通过的字节。
- Failed 单元格点击查看编译详情；评测中不误报终态、不重复上传。

## 6. Task112 的行动优先级（按官方聚合口径）

官方聚合 = 七个目标 speedup 的**算术平均**，且**必须 7/7 才有聚合**。因此：

1. **昆仑芯是资格闸门**：三条记录都是 6/7，`best_eligible` 全为空——不解决昆仑芯，前面所有芯片的分数都不进排名。这是唯一"从无到有"的动作。
2. **昆仑芯之后看绝对增量**：`decide` 给出余量 ascend **+6.73**、metax **+4.19**、intl_b **+1.32**（合计 /7 ≈ +1.7 聚合上界），而 iluvatar/hygon/intl_a 已在参照附近（+1.01/+0.79/+0.17）。低分芯片的绝对增量与高分芯片同权，**优先动 ascend 与 metax**，不是继续堆 iluvatar。
3. **对应杠杆按支持表选**：昇腾可用 `tle.dsa.pipeline` / `parallel` / `hint` / 多级片上存储；沐曦只有 `extract_tile`/`insert_tile`，空间有限；国际 B 未列出，只能普通 Triton。
4. **每一步只改一颗芯片并指名它**：`check` 的沿用计划会阻断未声明的偏离；其余芯片沿用产出其最佳值的文件字节。
5. 昆仑芯仍按第 5 节的先兼容后性能推进，并使用受控的最小复现，而不是盲加未经验证的 TLE 原语。

本清单是人工执行流程，不宣称 CLI 已自动校验 TLE 能力；完整发布证据缺口仍保留，不能由这篇指南补成通过。
