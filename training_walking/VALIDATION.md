# 本次训练链路验证（2026-11-12）

在 Linux x86_64、RTX 4090、NVIDIA 驱动 580.65.06 上，使用全新独立虚拟环境按 `uv.lock` 安装。
实际版本为 Python 3.13.5、PyTorch 2.11.0+cu128、MuJoCo / MuJoCo-Warp 3.11.0、Warp 1.17.0、RSL-RL 5.5.0。
`uv pip check` 检查 144 个包通过，训练测试 6 项通过。

从 `/tmp` 调用根目录的 `train.sh walking`，执行平地 AMP 的 16 环境、2 轮训练（768 个采样步），
验证默认本地 TensorBoard、PPO 和 AMP 判别器更新、检查点保存与 ONNX 导出。
actor、critic 和判别器权重均实际变化且全部有限；加载新检查点再追加 1 轮成功，环境步数由 48 继续到 72。
ONNX 结构检查与 ONNX Runtime 的 `obs[1,700] → actions[1,20]` 有限值推理通过。

构建 wheel 后确认全部 30 个机器人及动作资源在包内；将 wheel 解压到另一目录，实际导入搬迁后的包，
机器人实体编译（nq/nv/nu = 27/26/20）与参考动作路径重定位均通过。
根目录的旧任务名称入口及跳舞入口也从 `/tmp` 执行帮助命令成功。

完整测量和小规模覆盖参数见 [验证 JSON](VALIDATION-20261112.json)。两轮训练只验程序链路，
未宣称获得收敛策略；本次实训覆盖单 GPU 平地 AMP，粗糙地形、速度基线与多 GPU 未单独训练。
