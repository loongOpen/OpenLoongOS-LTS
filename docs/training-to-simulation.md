# 从训练到全链仿真

[返回首页](../README.md) ·[技术说明文档](technical-guide.md) · [快速开始](quickstart.md) · [技术架构](architecture.md) · [UI 使用手册](../loong_gui_sim/README.md)

流程为：安装训练环境 → 小规模检查 → 正式训练 → 回放检查点 → 获取 ONNX → 核对接口 → 在全链仿真中验证。本教程以 `Mjlab-Amp-Flat-LingLong` 为例，使用当前仓库的任务和配置。

## 1. 安装训练环境

先完成 [下载与完整性检查](quickstart.md)。训练使用 Linux x86_64、NVIDIA CUDA GPU 和可用驱动。安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，在仓库根目录执行：

```bash
nvidia-smi
cd mjlab_amp_ovision
uv sync --locked --extra cu128 --group dev
uv run --locked --extra cu128 list-envs
uv run --locked --extra cu128 pytest tests/ -q
cd ..
```

uv 管理 Python 和 Python/CUDA 用户态依赖；NVIDIA 驱动由系统提供。首次安装和 Warp 编译需要时间及磁盘空间。发布检查使用 RTX 4070；环境数和训练速度应以实际显存、配置和日志为准。

| Task ID | 用途 |
| --- | --- |
| `Mjlab-Amp-Flat-LingLong` | 平地 AMP 行走 |
| `Mjlab-Amp-Rough-LingLong` | 起伏地形 AMP 行走 |
| `Mjlab-Velocity-Flat-LingLong` | 平地速度跟踪基线 |
| `Mjlab-Velocity-Rough-LingLong` | 起伏地形速度跟踪基线 |

这四个任务属于同一阶段的行走训练，不是愿景中的四项独立能力。

## 2. 先检查训练链路

仓库根目录运行：

```bash
./train.sh Mjlab-Amp-Flat-LingLong \
  --env.scene.num-envs 16 --agent.max-iterations 2 \
  --agent.logger tensorboard --agent.run-name smoke \
  --agent.amp.num-preload-transitions 256 \
  --agent.amp.replay-buffer-size 4096 \
  --agent.amp.num-mini-batches 2
```

确认完成采样、优化、保存，且没有 NaN、OOM 或 ONNX 导出错误。两轮迭代只验证程序链路，不能得到可用行走策略；此处缩小的 AMP 数据参数也不用于正式训练。

## 3. 正式训练与监控

从仓库根目录启动，按显存调整环境数：

```bash
./train.sh Mjlab-Amp-Flat-LingLong \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.logger tensorboard --agent.run-name walking
```

默认使用一个 GPU；精确参数可通过以下命令查看，不需要启动训练：

```bash
./train.sh Mjlab-Amp-Flat-LingLong --help
```

训练开始时输出实际 device、seed 和日志目录。文件位于：

```text
mjlab_amp_ovision/logs/rsl_rl/linglong_amp/<运行目录>/
├── params/env.yaml          # 本次环境配置
├── params/agent.yaml        # 本次训练配置
├── model_<迭代号>.pt        # 检查点，可续训、可回放
├── <运行目录>.onnx          # 每次保存时更新的策略导出
└── events.out.tfevents.*    # TensorBoard 日志
```

另开终端，进入 `OpenLoongOS-LTS/mjlab_amp_ovision/`：

```bash
uv run --locked --extra cu128 tensorboard --logdir logs/rsl_rl --host 127.0.0.1
```

打开终端给出的本机地址。结合 episode 长度、速度跟踪表现、`amp_style_reward`、`amp_task_reward` 和损失曲线评估训练，不用总奖励一个指标判断是否能走。真实日志键可能带有分类前缀，可按这些名称检索。

训练运行目录是复现实验的基本单位，应一起保存检查点、两份配置、代码提交号和运行命令。停止前确认已有检查点落盘；Ctrl+C 中断后只使用实际存在的检查点，不假设一定自动保存了最后一步。

## 4. 续训

以下命令仍从仓库根目录执行。将引号内的运行目录和文件名替换为自己已有的值：

```bash
./train.sh Mjlab-Amp-Flat-LingLong \
  --env.scene.num-envs 1024 --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run '2026-10-23_14-00-00_walking' \
  --agent.load-checkpoint 'model_1000.pt' \
  --agent.max-iterations 1000 --agent.run-name resume
```

`load-run` 和 `load-checkpoint` 按正则匹配，指定具体名称可以减少选错实验的风险。续训会建立新的运行目录；这里的 `max-iterations` 是本次追加迭代数。

恢复内容包括网络、优化器及 AMP 判别器等状态，并恢复环境步数以继续课程进度。启动配置仍来自当前代码与 CLI，不会自动把旧 `params/env.yaml` 当作完整配置重新加载；因此续训时要保留原配置对应的代码和覆盖参数，不能只拿一个 `.pt` 就假定实验条件一致。

## 5. 回放检查点

进入训练目录，并设置一个实际存在的检查点路径：

```bash
cd mjlab_amp_ovision
checkpoint_file='logs/rsl_rl/linglong_amp/2026-10-23_14-00-00_walking/model_1000.pt'
uv run --locked --extra cu128 play Mjlab-Amp-Flat-LingLong \
  --checkpoint-file "$checkpoint_file" --num-envs 1 --viewer native
```

桌面模式使用 `native`；浏览器查看可改成 `--viewer viser`，按终端打印的地址访问。`play` 使用 `.pt`，不要把 `policy.onnx` 传给 `--checkpoint-file`。仓库示例策略是 ONNX，直接体验它应走 [GUI Demo](quickstart.md)。

当前 `play.py` 会设置八组固定速度命令，按环境编号循环分配；单环境默认第一组为 `(vx=0.7, vy=0, wz=0)`，不是静止测试。终端会打印实际命令。需要对比多种命令时可设置 `--num-envs 8`，具体命令表以 [play.py](../mjlab_amp_ovision/src/mjlab/scripts/play.py) 为准。

回放采用当前任务的 play 配置，也不自动还原检查点目录中的全部环境参数。先确认训练与回放的机器人、观测、动作和任务一致，再分析策略效果。

## 6. 获取并检查 ONNX

本版本在保存检查点时自动导出 ONNX，文件名为 `<运行目录>.onnx`，保存在同一个运行目录。后续保存会覆盖这个 ONNX，所以它对应最近一次成功导出，不是目录中任意一个较早的 `.pt`。需要保留某一阶段的 ONNX 时，应及时复制到独立实验归档中。

导出失败会打印 `[WARN] ONNX export failed (training continues)`；训练继续不代表导出成功。检查日志和文件时间，失败时不要误用之前留下的旧 ONNX。

下面在 `mjlab_amp_ovision/` 中执行。先将 `policy_file` 指向实际导出文件；首次熟悉流程也可以使用仓库示例 `../models/policy.onnx`：

```bash
policy_file='../models/policy.onnx'
policy_file="$(realpath "$policy_file")"
uv run --locked --extra cu128 python - "$policy_file" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

path = Path(sys.argv[1])
model = onnx.load(str(path))
onnx.checker.check_model(model)
session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
assert len(session.get_inputs()) == len(session.get_outputs()) == 1
inp, out = session.get_inputs()[0], session.get_outputs()[0]
assert (inp.name, inp.type, inp.shape) == ('obs', 'tensor(float)', [1, 700])
assert (out.name, out.type, out.shape) == ('actions', 'tensor(float)', [1, 20])
actions = session.run(None, {'obs': np.zeros((1, 700), dtype=np.float32)})[0]
assert np.isfinite(actions).all()
meta = {p.key: p.value for p in model.metadata_props}
baseline = json.loads(Path('../models/policy.metadata.json').read_text())['metadata']
keys = ['joint_names', 'default_joint_pos', 'action_scale',
        'joint_stiffness', 'joint_damping', 'observation_names',
        'observation_terms_scale', 'observation_terms_history_length',
        'observation_terms_flatten_history_dim', 'observation_terms_clip']
for key in keys:
    if key not in meta:
        raise ValueError(f'Missing metadata: {key}')
changed = [key for key in keys if meta[key] != baseline[key]]
print('ONNX structure and zero-input inference: OK')
print('Metadata fields differing from released policy:', changed)
if changed:
    raise SystemExit('Review differing fields against runtime configuration before replacing policy.')
print('Metadata matches the released policy interface; physical behavior still requires simulation.')
PY
```

这项检查验证结构、有限值和元数据一致性，不验证行走性能。字符串比较较严格，格式或舍入变化也可能触发差异；需逐项核实，不能仅凭形状是 700/20 就跳过检查。训练环境使用 Python 3.11–3.13 时，`--group dev` 会安装这里使用的 ONNX Runtime。

## 7. 在全链仿真中替换策略

先退出旧 GUI 和控制链。确认候选模型沿用发布策略的观测项顺序、10 帧历史、动作关节顺序、缩放和默认姿态；具体协议见 [技术架构](architecture.md)。若这些定义发生变化，需要适配运行组件，而不是只替换文件。

继续使用上一步的绝对路径变量 `policy_file`。只有在完成候选策略检查后执行：

```bash
cd ..
backup_dir="$(mktemp -d ../OpenLoongOS-policy-backup.XXXXXX)"
cp loong_gui_sim/runtime/model/onnx/policy.onnx "$backup_dir/policy.onnx"
cp loong_gui_sim/runtime/config/plan_rl.ini "$backup_dir/plan_rl.ini"
cp "$policy_file" loong_gui_sim/runtime/model/onnx/policy.onnx
sha256sum loong_gui_sim/runtime/model/onnx/policy.onnx
./start_sim.sh
```

运行时文件名保持 `policy.onnx`，与 `plan_rl.ini` 的 `model` 设置一致。保留 `models/policy.onnx` 作为发布示例，不需要同步替换它。

先按 Demo 检查辅助恢复、RL 零速度、撤去辅助后的站立，再逐步检查前进、转向与停止。训练模型是 20 轴，运行模型是 26 轴，并使用独立的固定 PD 配置；即使元数据一致，接触、惯量和驱动行为仍需对照验证。

需要回退时，退出 GUI 和控制链，在同一终端执行（新终端需把 `backup_dir` 设为实际备份目录）：

```bash
cp "$backup_dir/policy.onnx" loong_gui_sim/runtime/model/onnx/policy.onnx
cp "$backup_dir/plan_rl.ini" loong_gui_sim/runtime/config/plan_rl.ini
./start_sim.sh
```

替换模型后，`check_release.py` 会按原发布哈希和策略配置报告变化，这是预期行为；不要通过重写发布清单把自定义策略标记成原版。保留实验配置、候选模型哈希与日志即可。

## 8. 使用自己的参考动作

AMP 启动时读取 `src/mjlab/tasks/amp/motions/linglong/` 内的 JSON。每份文件需要 `Frames`（每帧 77 个数）和 `FrameDuration`，可选 `MotionWeight`，缺省为 1。

原始布局为 `root_pos3 | root_rot_xyzw4 | joint_q20 | end12 | lin3 | ang3 | joint_dq20 | end_vel12`；末端顺序为左脚、右脚、左手、右手。加载器提取 53 维 AMP 特征，与 actor 的 700 维输入是两套不同用途的数据。

新动作必须对齐 20 个关节的顺序、单位与坐标系。在线 AMP 特征使用机体系的基座速度和相对根部的脚位置，以及世界系根部高度；仅把 JSON 补足到 77 列并不够。字段切片见 [motion_loader.py](../mjlab_amp_ovision/src/mjlab/tasks/amp/rl/motion_loader.py)，在线特征见 [observations.py](../mjlab_amp_ovision/src/mjlab/tasks/amp/observations.py)。加入新动作后重新做小规模检查，再开始正式训练。
