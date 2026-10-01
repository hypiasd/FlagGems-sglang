---
name: flagos-adaptation
description: 将 FlagOS 实验候选适配到官方多芯片契约，检查发布条件、打包、通过 Chrome 提交、回填官方记录，并把获奖提交整理成上游仓库 PR；日常 GPU 优化使用 flagos-experiment。
---

# 比赛适配

读取 [执行手册](../../../competition/WORKFLOW.md) 的比赛部分及本题 profile。题目支持芯片、入口和发布门槛独立维护，不固定为八款，不继承其他题目的数值目标。比赛有两条出口：平台 ZIP（评分）与上游仓库 PR（官方 `docs/CONTRIBUTING.md` 第 9 节的竞赛贡献规则）。

1. `inspect --refresh` 保存官方公开契约证据。提交前通过 Chrome 确认当前任务、批次、登录团队、剩余额度、最近记录及重复包；公共 API 信息不代表提交授权。
2. 从冻结实验用 `prepare` 建立独立适配目录。需要后端变化时编辑适配源码，再填 `release.json`。实验源码不变。
3. `check` 检查包、逐目标结构变化、源码/基线绑定、静态兼容性、独立审查、完整正确性和题目发布门槛。`--package-only` 只证明包有效，不能作为发布通过。缺失证据保持 blocked，不填造预测。
4. 仅在会话已授权提交且完整检查通过后执行 Chrome 上传。上传前将 adaptation.json 写为 `upload_armed`，记录包哈希、任务/批次/团队标识、可见额度及时间；同目录 events.jsonl 追加事件。上传后立即记录平台 record_id 和 submitted 状态。
5. 中断恢复先检查 Chrome 的最近记录；`upload_armed` 或更后不得盲目重复上传。包变化使原检查失效，必须重新检查。沿用超过 120 秒的提交间隔和当前可见额度限制。
6. `record` 回填带平台证据、提交 ID 和包哈希的官方观测；评测中到完成可追加修订，同内容重试幂等。失败、异常及排除原因完整保留。确认全部本题目标通过、官方聚合有效后才比较成绩。
7. 平台评出成绩后，用 `competition.pr` 把获奖包转成上游 PR：`bundle`（tier 映射 + Apache 头 + `__all__` + 官方版本 isort/black + 超长文本折行 + 声明的 lint 改名，全部记进 `bundle.json`）、`materialize`（展开 `upstream/master` 并放入文件）、`check`（structure / hygiene / ast_preservation / style / 官方 ci_checks / import smoke / evidence）、`PR.md`（竞赛口径的 PR 描述，成绩抄自平台记录）。竞赛 PR 不需要 tests/benchmark/docs/operators.yaml；`import_smoke` 在无 torch+triton 的机器上只能报 `unavailable` 并给出设备上的执行命令。
8. `evidence` 复用 KernelGen reliability-gate 的方法论并写成 `evidence.json`：能力三态（`absent` / `configured_unavailable` / `callable`，配置或握手不算可用）、逐目标一个证据状态（缺证据或部分覆盖是 `inconclusive`，不能靠解释升成 `target_validated` / `measured`）、性能只给标签且 `official_score_computed` 恒为 false、平台记录生命周期（`evaluating` 的数值是临时的，interim 行标 `superseded` 保留）、run record 字段纪律（重试换新 `run_id` 并用 `parent_run_id`）、两阶段独立审查收据。闸门里**矛盾 → fail，缺证据 → unavailable**，所以 `ready_for_pr` 始终不等于“已验证”。

本地/T4 通过不是目标芯片通过。KernelGen 工具、配置文件存在或工具握手都不是发布证据（其 MCP 工具本会话未注册，专精知识库只覆盖华为）。`competition.pr check` 的“全过”只覆盖本地能跑的闸门，不等于上游 CI 通过或可以合并。
