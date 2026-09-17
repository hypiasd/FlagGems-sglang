# Task 60 — `clamp_position`

这是 FlagOS S2 赛道一 Task 60 的提交源文件。

`clamp_position.py` 只导出一个公开入口 `clamp_position`，在函数内部按设备类型选择路径：

- `gcu`：燧原 int32 Triton 路径；
- `npu` / 昇腾 PrivateUse1：原生 `torch.clamp_min` 路径；
- 其他设备：通用 Triton 路径。

提交时只需将 `clamp_position.py` 打包到压缩包根目录，不要把本 README 一起上传：

```bash
zip -j clamp_position_submit.zip competition/task60/clamp_position.py
```

本机没有目标加速卡，真实性能以 FlagOS 在线评测为准。
