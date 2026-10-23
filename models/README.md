# policy.onnx

[策略接口与关节映射](../docs/architecture.md) · [训练、导出和替换教程](../docs/training-to-simulation.md)

用户提供的开源示例模型，原始文件内容保持不变。

| 项目 | 值 |
| --- | --- |
| 输入 | `obs`，float32，`[1, 700]` |
| 输出 | `actions`，float32，`[1, 20]` |
| ONNX opset / IR | 18 / 8 |
| 外部权重文件 | 无 |
| SHA-256 | `31ea9bfb9f94270cbc7e2b9a761d8b59354c279adf3f50fdbc7988fdf29f434d` |
| 模型记录的训练运行 | `2026-09-29_10-06-35_linglong_amp_walk` |

原始 ONNX 元数据已完整导出到 `policy.metadata.json`：20 个输出关节依次为左腿 6 轴、
右腿 6 轴、左臂 4 轴、右臂 4 轴。观测项包括角速度、重力投影、指令、关节位置和速度、
上一步动作及步态相位，各项声明历史长度为 10。完整顺序、缩放、默认角度及 PD 参数
以元数据和训练时的观测实现为准。

已通过 `onnx.checker.check_model`，并接入全链仿真的 `rl` plan（键值 4）。
运行副本为 `../loong_gui_sim/runtime/model/onnx/policy.onnx`，两份文件 SHA-256 相同。
预编译插件完成 20 轴策略到 原仓库灵龙的 26 轴仿真的映射，其余关节保持默认姿态；使用说明见
[仿真说明](../loong_gui_sim/README.md)。

输入包历史记录包含 180 帧观测与训练历史缓冲对比、独立 ONNX Runtime 动作对比、PD 和关节映射检查，
以及原 26 轴模型关闭辅助力后的 30 秒进程内站立检查。本次发布实测与历史记录分别见 [VALIDATION.md](../VALIDATION.md)；尚未验证各速度指令下的持续行走。
模型权重保持原样。
