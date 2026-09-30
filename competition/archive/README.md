# 历史归档

迁移基线是 `ab800a8`。`migration.json` 给出 114 项原跟踪文件的原路径、归档路径、字节数和 SHA-256；`legacy/` 按原路径保存字节完全相同的源码、ZIP、记录、文档和旧 agent 工作流。`project-contract-before.md` 保存重构前的完整项目恢复合同。

未跟踪文件、旧实验和候选的 788 项本地备份位于忽略目录 `competition/.local/archive/`，清单是 `.local/archive-manifest.json`。这些资产和原未跟踪文件没有自动进入 Git；连接配置、密钥及虚拟环境没有纳入备份清单。源码/包的初始备份是迁移时的快照，后续其他工作可以继续修改原未跟踪文件。

```sh
python -m competition.archive.verify
python -m competition.archive.verify --include-local
```

只校验本地私有备份的第二个命令需要在原设备运行。历史文档中的旧路径和旧命令仅解释当时工作；现行流程见 [执行手册](../WORKFLOW.md)。
