# 第一阶段发布检查

检查日期：2026-10-23。此文区分本次实测与输入包的历史记录，不将演示视频等同于当前版本的完整回归。

## 本次验证

- 输入归档：原清单 412 个文件，411 个文件 SHA-256 一致，缺少 `loong_gui_ctrl/loong_gui_ctrl`。第一阶段发布目录不包含独立机器人控制台，已移除其启动入口并在发布清单中明确标记。
- 训练安装：在独立新建虚拟环境执行 `uv sync --locked --extra cu128 --group dev`，144 个包安装成功；uv 锁文件离线一致性检查通过。
- 训练环境：Python 3.13.14、PyTorch 2.11.0+cu128、MuJoCo / MuJoCo-Warp 3.11.0、Warp 1.17.0、rsl-rl-lib 5.5.0；本机 RTX 4070 可被 PyTorch 使用。
- 任务和测试：4 个行走任务可列举；现有测试 `6 passed`，35 条上游 TorchScript 弃用警告。
- 示例策略：ONNX 结构检查及 ONNX Runtime CPU 推理通过；输入 `obs[1,700]`、输出 `actions[1,20]`，零输入推理结果均为有限值。
- UI 系统依赖：Ubuntu 22.04 容器内检查 23 个 x86_64 ELF 文件，没有未解析的动态库。
- UI 离线检查：offscreen 自检通过；独立 Xvfb 显示环境的自检也通过，返回 `compiled_gui`、`native_replay`、`qt_callbacks` 为 true。
- UI 测试边界：两次自检均返回 `native_window_embedded=false`、`native_camera_follow=false`；本次没有据此确认实际 MuJoCo 窗口嵌入、镜头跟随、完整多进程控制链或实机运行。
- 平台边界：宿主机 Ubuntu 20.04 运行 UI 复现 `GLIBC_2.35 not found`；这个预编译 UI 应按 Ubuntu 22.04 平台要求运行，不能仅安装 Python 包来解决系统 ABI 差异。
- 发布清理：仅在独立发布副本清理旧运行日志、编辑器配置；原压缩包保持不变。视频原件保留，并附网页兼容副本。
- 本地 Git 检出：通过独立克隆、恢复本地 LFS 对象与 `git lfs checkout`，420 个文件哈希检查通过，`git lfs fsck` 通过；两个原视频与输入文件逐字节一致。该检查不代表 AtomGit 服务器已接收文件。

小规模 GPU 训练检查通过：单张 RTX 4070、16 个环境、2 次 PPO 迭代，完成 768 个采样步，生成 `model_0.pt`、`model_1.pt` 和 ONNX 导出文件，进程以 0 退出，测试后 GPU 进程已释放。为限制资源占用，将 AMP 预加载量设为 256、回放缓存设为 4096、判别器 mini-batch 数设为 2。

复现命令（整合包根目录）：

```bash
./train.sh Mjlab-Amp-Flat-LingLong \
  --env.scene.num-envs 16 --agent.max-iterations 2 \
  --agent.logger tensorboard \
  --agent.amp.num-preload-transitions 256 \
  --agent.amp.replay-buffer-size 4096 \
  --agent.amp.num-mini-batches 2
```

此测试验证训练、采样、优化、保存与导出链路能够执行，不验证策略已经学会行走。首次 Warp 编译需要额外时间。本机驱动报告 CUDA 12.2，程序提示 CUDA Graphs 因低于 12.4 而禁用，测试使用回退路径完成；不能据此推断较新驱动或 CUDA Graphs 路径也已验证。测试生成的随机初始策略未替换发布示例策略。

## 输入包已有的历史记录

输入包记录：原 26 轴模型曾通过关闭辅助力的 30 秒进程内 RL 站立检查；180 帧观测历史和 ONNX 推理对比通过。这些属于历史验证，本次没有重新执行这些物理对照，也不将它们扩展为持续行走或真机部署结论。

输入包同时记录 SDK 全局析构中的 `Quill::remove_logger` 崩溃，之前的物理检查在释放 MuJoCo、落盘后直接结束工作进程，未覆盖完整析构。本次未修改该 SDK，也未确认该问题已修复。

## 已知限制与发布状态

1. GUI 仅以二进制交付，第三方分发许可依据及模型、动作数据、演示素材的授权范围尚未完整核实，见 [LICENSES.md](LICENSES.md)。
2. 对指定版本执行完整多进程控制链、非零速度行走、停止与退出回归。演示视频可以说明项目能力，不能替代当前检出版本的复现报告。
3. AtomGit SSH 认证已通过，服务器确认账号为 `bjwwww`；目标仓库的推送检查返回 `CH.00905401: You are not allowed to push code to this project`。当前尚未上传，需目标仓库授予该账号写入权限，之后再验证远端 LFS 下载回读。

完整性检查命令：

```bash
git lfs pull
python3 -B check_release.py
```

校验通过表示文件内容和发布配置匹配，不等同于算法性能、GUI 全流程或真机安全已经验证。
