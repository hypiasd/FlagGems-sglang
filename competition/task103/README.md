# Task 103：recompute_w_u

当前选定的算子。公开入口 `recompute_w_u(k, v, beta, g_cumsum, A, cu_seqlens)`；官方来源、七个目标、dtype 和精度见 [profile.json](profile.json)。2026-10-01 公开 API 目标矩阵复核一致。

[adapter.py](adapter.py) 给出参考实现及三个开发用例，包含 BF16 中间舍入、FP32 累加和 GQA head 映射。当前仅支持 `cu_seqlens=None`、`T % BT == 0` 的开发范围，未获得完整官方 checker，不声明官方覆盖。

原生 BF16 路径默认要求 BF16 Tensor Core。T4 不满足时实验返回 `unsupported`；不会自动转换 dtype。发布收益策略未核实，比赛完整检查会阻断发布。

现行工作流见 [执行手册](../WORKFLOW.md)。初始化历史见 [归档 README](../archive/legacy/competition/task103/README.md)。迁移前已有及并行工作新增的未跟踪脚本由原工作保留，本轮未修改或提交。
