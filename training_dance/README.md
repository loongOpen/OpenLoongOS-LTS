# 灵龙 1.4 舞蹈训练 · OpenLoongOS 仿真版

面向 OpenLoongOS 整理的完整独立 mjlab 训练源码、灵龙 1.4 模型、舞蹈动作和示例策略。解压后即可安装、从头训练，或用示例权重在仿真中播放舞蹈。验证环境为 Linux x86_64。

## 安装

使用 Python 3.12 或 3.13。正式训练使用 NVIDIA CUDA GPU；CPU 可用于播放和少量环境的训练检查。Python、CUDA 驱动和第三方依赖不包含在 ZIP 中。

```bash
cd training_dance
bash install.sh
```

安装脚本创建独立 `.venv`，安装 PyTorch 2.9.0/CUDA 12.8 和验证时使用的依赖版本，再安装本工程。首次安装需要联网，第一次仿真会编译 Warp 内核。若系统的 `python3` 不是 3.12/3.13，可用 `PYTHON=python3.13 bash install.sh`。

安装时只从 PyTorch 官方 CUDA 12.8 源获取 torch 本体，NVIDIA CUDA 运行库和其他依赖从清华 PyPI 镜像获取。安装中断后直接再次执行 `bash install.sh`，会复用已有虚拟环境和下载缓存。脚本保留 HTTPS 证书校验。

如果访问清华镜像不便，可更换普通依赖的下载源：

```bash
MJLAB_PYPI_INDEX_URL=https://pypi.org/simple bash install.sh
```

已有兼容环境也可执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
MJLAB_PYTHON="$(command -v python)" bash train.sh --help
```

`train.sh` 和 `play.sh` 优先使用 `MJLAB_PYTHON`，其次为 `PYTHON`、本目录 `.venv/bin/python`，最后为 `python3`。两种训练工程都使用 `mjlab` 包名，推荐独立虚拟环境；入口会优先加载本目录源码，避免误用行走任务。

## 直接看示例跳舞

```bash
bash play.sh
```

有桌面时打开 MuJoCo 窗口；没有桌面时打开 Viser 服务，在终端提示的网址查看。可显式选择：

```bash
bash play.sh --viewer native
bash play.sh --viewer viser
bash play.sh --device cpu --viewer none --report logs/demo_evaluation.json
```

默认播放 `checkpoints/dance_demo.pt`。示例来自已有灵龙 1.4 第 29999 轮策略，只包含 actor、观测归一化和必要的迭代标记，已剔除 critic、优化器、原实验配置和路径。它用于直接演示，不能作为完整续训 checkpoint。此次没有将精简配置从头训练至收敛，实际验证范围见 `docs/VALIDATION.md`。

示例动作包含 1781 帧、50 Hz、约 35.62 秒，末尾附加 3 秒静止保持。播放使用真实物理仿真与策略推理。到片段末尾后重置并循环播放。无界面评测会跑一整段并报告终止重置次数。

## 从头训练

```bash
bash train.sh --device cuda:0 --env.scene.num-envs 1024 --agent.max-iterations 30000
```

默认 TensorBoard 日志和 checkpoint 保存在 `logs/rsl_rl/linglong14_dance/`，每 250 轮保存一次。查看训练曲线：

```bash
.venv/bin/tensorboard --logdir logs/rsl_rl
```

播放自己训练的权重，或继续自己保存的完整训练 checkpoint：

```bash
bash play.sh logs/rsl_rl/linglong14_dance/你的训练目录/model_29999.pt
bash train.sh --checkpoint logs/rsl_rl/linglong14_dance/你的训练目录/model_1000.pt
```

CPU 短训练检查（验证更新和保存，不用于训练舞蹈效果）：

```bash
bash train.sh --device cpu --env.scene.num-envs 2 --agent.max-iterations 2 --agent.save-interval 1
```

## 发布范围

- 保留 mjlab 仿真框架、标准 RSL-RL PPO、1.4 的 26 关节模型与网格、完整示例动作，以及现有舞蹈观测、奖励和参考动作跟踪逻辑。
- 已移除域随机化源码与任务配置、质量/惯量/COM/摩擦/PD/编码器随机化、外部随机推扰、观测噪声配置、复位姿态/速度/关节扰动和执行器随机延迟配置。
- 移除自适应动作采样、末段加权采样配方、其他机器人任务与资产、内部诊断监督器、实验记录和实机部署工具。
- 标称物理参数固定，执行器延迟为零。训练仍使用普通均匀参考帧采样和 PPO 必需的策略探索、网络初始化、minibatch 洗牌；“去随机化”不表示把 PPO 改为无探索的确定性算法。
- 没有保留训练目录、Git 历史、缓存、虚拟环境、账号配置或原 checkpoint 中的实验信息。包内的通用 mjlab 缓冲区和仿真字段管理属于框架基础设施。

本包以仿真舞蹈训练和演示为目标。保留的舞蹈观测/奖励是保证已有策略兼容的训练逻辑，未宣称所有训练方法或参数都已删除。

## 文件位置

| 内容 | 路径 |
|---|---|
| 训练、播放入口 | `train.sh`、`play.sh` |
| 1.4 环境配置与奖励 | `src/mjlab/tasks/tracking/config/linglong14/` |
| 标称模型与 PD | `src/mjlab/asset_zoo/robots/linglong14/` |
| 完整示例动作 | `src/mjlab/asset_zoo/motions/linglong14/xiao_speed075_modified.npz` |
| 直接播放的策略 | `checkpoints/dance_demo.pt` |
| 检查与评测结果 | `tests/`、`docs/VALIDATION.md` |
| 文件校验 | `SHA256SUMS` |

替换动作时，数据必须匹配 1.4 的关节和刚体命名/顺序、50 Hz 控制周期，以及 wxyz 四元数和连杆原点世界线速度约定。训练参数 `--env.commands.motion.motion-file` 和播放参数 `--motion-file` 可指定新 NPZ。示例权重只用于当前示例动作。

源码基于 mjlab 1.6.0；上游 Apache-2.0 许可、原 Linglong/Isaac Lab 工作区提供的许可和来源说明分别保存在 `LICENSE`、`LICENSE-LINGLONG-SOURCE`、`NOTICE`。本包尚未上传或发布到外部仓库。
