# 验证记录

日期：2026-11-11。使用本包的独立源码和包内动作/策略，Python 3.13、CPU、MuJoCo 3.11.0、MuJoCo Warp 3.11.0、PyTorch 2.9.0+cu128、RSL-RL 5.5.0；其余版本见 `runtime_versions.json`。

## 完整物理仿真播放

运行 `play.sh` 的底层播放入口，使用包内示例策略、固定标称物理、零执行器延迟和无噪声观测。从动作首帧运行 1930 个控制步，共 38.6 秒，包含原动作和 3 秒静止尾段。

- 终止重置次数：**0**；完整动作评测通过。
- 最低 base_link 高度：约 **0.8709 m**。
- 策略输出为有限数值，真实物理仿真由 MuJoCo Warp 执行。
- 评测结果见 `playback_evaluation.json`。零重置不等于每个关节都精确跟踪，报告同时保留最大关节误差。

复现：

```bash
bash play.sh --device cpu --viewer none --report logs/demo_evaluation.json
```

示例策略由原有第 29999 轮训练结果提取。此次验证的是它在精简配置中的播放兼容性；没有将精简配置从头训练到收敛。

## 训练更新与保存

执行 2 个 CPU 环境、2 轮标准 PPO 训练，每轮保存一次。两份 checkpoint 的 actor 参数确实发生变化，保存的 actor 张量均为有限数值。记录见 `training_smoke.json`。

```bash
bash train.sh --device cpu --env.scene.num-envs 2 --agent.max-iterations 2 --agent.save-interval 1
```

这是训练入口、仿真采样、反向传播、参数更新和文件保存的验证，不能代表从头训练两轮便已学会舞蹈。

## 资产与发布检查

`tests/test_release.py` 共 4 项检查通过：固定物理/零噪声/零延迟配置、模型和完整动作及抽样 FK 一致性、示例权重中无实验/优化器内容，以及私有随机化/部署配方未包含在发布包中。

模型为 26 个受控关节，`nq/nv/nu = 33/32/26`。动作含 1781 帧、50 Hz；删除内部路径和转换记录时，所有运动数值数组逐项验证与原数据相同。

已检查源码语法并构建可安装 wheel，确认模型网格、URDF/XML 和 NPZ 随安装包包含。将 wheel 安装到独立目录后，确认实际导入的是该目录的源码、动作路径随安装位置迁移、模型能编译，并用包内策略运行了 50 步物理仿真，终止重置次数为 0。ZIP 内的 `SHA256SUMS` 可校验文件完整性。

此执行环境没有可用 CUDA 设备，也没有可用的 EGL 离屏图形上下文，因此没有声称完成 GPU 训练或 MuJoCo 离屏视频验收。无界面完整物理播放和 CPU PPO 检查均实际执行。可视播放可用 `play.sh --viewer viser`。

## 安装源修复

2026-11-11 根据实际安装反馈调整 `install.sh`：只从 PyTorch CUDA 12.8 源安装 `torch==2.9.0+cu128` 本体（`--no-deps`），其余依赖统一从普通 PyPI 镜像解析，以避免 NVIDIA 运行库的下载被转向连接中断的 `pypi.nvidia.com`。保留 HTTPS 校验，支持中断后重跑，并在安装末尾执行 `pip check` 与运行库导入检查。

已验证 torch 独立安装的解析计划只包含官方 torch wheel；日志中失败的 `nvidia-cuda-nvrtc-cu12==12.8.93` 已实际从清华镜像成功下载；参考仿真环境与发布包要求的依赖一致，`pip check` 通过。

## 2026-11-12 三功能整合复验

在本次项目目录使用本地源码、随包动作和示例 checkpoint，以 RTX 4090 / CUDA 再次验证：

- 已有较新兼容环境（PyTorch 2.14.1+cu130、MuJoCo/MuJoCo Warp 3.16.0、Warp 1.18.0、RSL-RL 5.5.1）运行 4 项资源/配置测试，全部通过。
- 2 个 GPU 环境从头训练 2 轮，生成第 0、1 轮完整 checkpoint；actor MLP 参数最大变化约 `0.0004673`，所有 actor 张量有限。加载该新训练 checkpoint 再追加 1 轮成功，MLP 参数继续变化。
- 随包示例策略完成 1930 步、38.6 秒完整物理播放，终止重置次数为 0，最低基座高度约 0.8709 m。详细数值见 `integration_validation_20261112.json`。

本次修复 `train.sh` / `play.sh` 对 `.venv` 的强制依赖，支持 `MJLAB_PYTHON` 显式选择兼容环境，并优先导入本目录源码，避免与行走工程的同名 `mjlab` 包混淆。安装脚本兼容已经创建但没有 pip 的虚拟环境。训练和播放均实际通过新的入口运行。

短训练验证采样、梯度更新、checkpoint 保存和续训链路；它不代表从零训练已收敛。上述较新环境的兼容性验证与下述精确版本验证分别记录。

随后创建独立 Python 3.12 环境，按 `requirements.txt` 的精确版本安装并安装本工程（PyTorch 2.9.0/CUDA 12.8、MuJoCo/MuJoCo Warp 3.11.0、Warp 1.14.0、RSL-RL 5.5.0）。`uv pip check` 和 `python -m pip check` 均通过，4 项测试全部通过。以默认 `.venv` 入口完成同样的 2 个 GPU 环境、2 轮从头训练，再加载完整新 checkpoint 追加 1 轮；actor MLP 参数均确实更新，权重有限。随包示例策略再次完成 1930 步、38.6 秒播放，0 次终止重置，最低基座高度约 0.87086 m。精确环境和较新环境分别记录在上述 JSON 的两个字段中。

另已重新构建 wheel，核对 30 个 XML/URDF/STL/NPZ 资源全部包含；将 wheel 安装到独立目录后，实际导入该目录的包、动作路径随目录重定位、模型编译均通过。没有执行从头至收敛的长训练。
