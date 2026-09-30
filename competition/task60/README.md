# Task 60：clamp_position

公开入口 `clamp_position(seq_lens)`，计算 `max(seq_lens - 1, 0)`。本题契约和目标矩阵见 [profile.json](profile.json)，开发矩阵见 [adapter.py](adapter.py)，现行命令见 [执行手册](../WORKFLOW.md)。

开发用例覆盖 int32/int64/fp32、空输入、tile 边界、连续与步长视图，共 66 个 GPU 用例；没有声明覆盖官方完整矩阵。受限 CPU 模型可用 `python -m competition.task60.validate_cpu` 执行 132 个通用/Ascend host 语义用例，不证明目标编译或性能。

[results.jsonl](results.jsonl) 保留原官方观察，[历史说明](../archive/legacy/competition/task60/README.md) 保存旧流程。
