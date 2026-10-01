# 历史归档

迁移基线是 `ab800a8`。`migration.json` 给出 114 项原跟踪文件的原路径、归档路径、字节数和 SHA-256；`legacy/` 按原路径保存字节完全相同的源码、ZIP、记录、文档和旧 agent 工作流。`project-contract-before.md` 保存重构前的完整项目恢复合同。

未跟踪文件、旧实验和候选的 788 项本地备份位于忽略目录 `competition/.local/archive/`，清单是 `.local/archive-manifest.json`。这些资产和原未跟踪文件没有自动进入 Git；连接配置、密钥及虚拟环境没有纳入备份清单。源码/包的初始备份是迁移时的快照，后续其他工作可以继续修改原未跟踪文件。

```sh
python -m competition.archive.verify
python -m competition.archive.verify --include-local
```

只校验本地私有备份的第二个命令需要在原设备运行。历史文档中的旧路径和旧命令仅解释当时工作；现行流程见 [执行手册](../WORKFLOW.md)。

## 归档：上游 PR 工具（`pr/`）

`pr/` 是曾经的"第二条出口"（把获奖包整理成上游仓库 PR）的完整实现：tier 映射与打包（`bundle.py`）、PR 描述（`description.py`）、structure / hygiene / ast_preservation / style / 官方 `ci_checks` / import smoke 闸门（`gates.py`）、以及从 KernelGen reliability-gate 方法论落成的证据模型（`evidence.py`、`spec.py`）。

**2026-10-01 按用户指示归档**：PR 是之后的事，不进入当前流程。它不再被任何模块导入，也不再是任何技能或手册的一步；测试文件改名成 `archived_test_*.py`，因此 `unittest discover` 不会自动跑到它们。需要时手动运行：

```sh
python -m competition.archive.pr rules
python -m unittest competition.archive.pr.archived_test_evidence competition.archive.pr.archived_test_pr
```

包内只有相对导入，所以移到 `archive/` 后仍可运行；它写出的产物路径仍是 `.local/pr/`。恢复成现行流程前，先想清楚是否真的要重新引入这条出口。
