# Task 78：concat_and_cast_mha_k

保留现有七份后端实现。契约与八个官方目标见 [profile.json](profile.json)；201 个开发回归用例和三个 quick 代表用例由 [adapter.py](adapter.py) 提供。GPU 输入/输出语义、边界和输入不可变检查独立于计时。

```sh
python -m competition.experiments new --task task78 --source competition/task78 --hypothesis '测量现有实现作为下一轮基线'
python -m competition.experiments bench --run RUN_ID
python -m competition.experiments test --run RUN_ID --all-sources
python -m competition.task78.official_history verify
```

实测 T4：201 × 7 = 1407 个执行全部通过。T4 是非官方目标证据，开发矩阵不等于官方覆盖。[验收记录](../ACCEPTANCE.md) 保存快照与报告 ID。

官方 [账本概览](results.md) 和 [原始记录](results.jsonl) 保留 25 次提交；旧包与版本文档按原路径归档到 [archive](../archive/README.md)。完整操作、发布门槛和 Chrome 恢复流程见 [执行手册](../WORKFLOW.md)。
