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

## v14 实验：小输入 1 warp

通用路径在 `BLOCK <= 256` 时使用 1 warp（此前为 1/2/4），更大的
tile 和设备专用路径沿用 v13。仅改变这一项启动参数，不改变计算、
dtype 或输入输出布局。静态检查和模拟 host dispatch 检查通过；
尚无目标芯片性能结果。对照正常基线 v11 的 1.41×，最好重复测量，
确认差异超过运行波动。历史燧原 9.40× 已被确认是异常，不作为基线。
