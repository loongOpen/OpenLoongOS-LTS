# Lite Parkour

Lite 26 自由度人形机器人的独立感知行走项目：PPO + AMP 动作风格奖励、胸前射线深度相机、复杂地形课程，以及 MuJoCo sim2sim。
代码、机器人资源、72 段已经映射好的 Lite 动作数据和一个示例策略均位于本目录。
模型和动作均为实体文件，不需要从其它工程补拷或重新执行 Git LFS 下载。
运行时只需要外部安装的 Isaac Sim / Isaac Lab 等依赖，不需要原工程目录或其它机器人模型。

## 目录

```text
training_parkour/
  pyproject.toml                    可安装的独立 lite_parkour Python 包
  requirements-train.txt            训练相关依赖版本
  requirements-training-cu128.lock 完整联合依赖锁
  requirements-sim.txt              PT/ONNX/MuJoCo 依赖版本
  environment-tested.json           本次实际验证的环境版本与 Isaac Lab 提交
  source/lite_parkour/lite_parkour/
    assets/lite.py                  电机、默认姿态、URDF 导入
    data/Robots/Lite/               26 关节 URDF、MuJoCo XML、27 个网格
    data/MotionData/Lite/amp/reference/
                                   72 段 26 维动作与 manifest
    tasks/amp/lite_env_cfg.py        独立 Lite 环境配置
    tasks/amp/agents/lite_ppo_cfg.py 独立 PPO/AMP 配置
    envs/ managers/ rsl_rl/ sensors/ terrains/ utils/
                                   训练所需的通用实现
  scripts/train_lite.sh             推荐的 headless 训练入口
  scripts/install_training.sh       独立训练环境安装与预检
  patches/                         固定上游提交的两行依赖补丁
  scripts/rsl_rl/train.py, play.py  原生训练 / Isaac Lab 回放入口
  scripts/sim2sim_mujoco_lite.py    PT → ONNX → MuJoCo
  scripts/sim2sim/                  本项目自己的运行库和导出器
  scripts/tools/                    模型重建、FK 和动作数据验证
  checkpoints/lite/                model_10200.pt、参数与导出的 ONNX 示例
  tests/                           Lite 独立配置、数据与部署测试
```

## 一、安装完整训练环境

默认组合为 Linux x86_64、Python **3.12**、Isaac Sim **6.0.0.1**、
Isaac Lab **v3.0.0-beta2**（提交 **28a37cecdd433c22d9eabd6a5954add9f13a8951**，元数据版本 **6.1.11**）、
PyTorch **2.11.0 + CUDA 12.8**、RSL-RL **5.4.1**。本次在 RTX 4090 / 驱动 580.65.06
完成了从头训练和续训检查；具体结果见 [TRAINING_VALIDATION.md](TRAINING_VALIDATION.md)。

先安装 NVIDIA 驱动、Git 和 [uv](https://docs.astral.sh/uv/)，然后：

```bash
cd training_parkour
bash scripts/install_training.sh
.venv/bin/python scripts/list_envs.py
```

脚本创建本目录独立 `.venv`，安装锁定的 233 个第三方依赖，在 `.external/IsaacLab` 获取固定上游提交，
显式安装 13 个 Isaac Lab core 模块和本项目。不会要求原工作区或修改其它 Python 环境。
Isaac Sim 的 RL、asset 和 robot-setup 组件均已纳入锁文件；后两项用于 Lite URDF 导入。
安装结束必须通过 `uv pip check`（247 个包）和 GPU/资源预检才报告完成。

上游该提交有两处与 Sim 6 冲突的依赖声明。安装脚本先核对提交并运行 `git apply --check`，
再应用随包的 [两行元数据补丁](patches/README.md)，保留上游许可证。
它避免使用会安装 Sim 5.1、RSL-RL 5.0.1 或重装 Torch 2.10 的旧 extras/安装入口；
仿真、任务、奖励和物理设置不变。不要再把旧的 `isaaclab.sh -i core` 或 `isaaclab[all]` 混入这个环境。

首次安装需要联网；第一次启动还会从 NVIDIA Kit 注册表下载所需扩展、编译 Warp 内核。
默认按需获取 headless 训练所需内容，不下载完整的 6 GB Kit 扩展缓存包。
因此需要正常访问 `ovextensionsprod.blob.core.windows.net` 等 Isaac Sim 官方服务。
训练入口会在 Kit 启动阶段启用 URDF importer，避免场景创建时才请求缺失扩展。
普通 PyPI 依赖可用 `LITE_PYPI_INDEX_URL=https://你的镜像/simple bash scripts/install_training.sh` 更换下载源；
PyTorch 仍使用锁定的官方 CUDA wheel，所有 HTTPS 连接保留证书校验。

如需指定专用环境和已有的固定 Isaac Lab 源码位置：

```bash
LITE_VENV_DIR=/你的专用环境 LITE_ISAACLAB_DIR=/你的IsaacLab \
  bash scripts/install_training.sh
LITE_PYTHON=/你的专用环境/bin/python bash scripts/train_lite.sh --num_envs 2 --max_iterations 2
```

`list_envs.py` 应只列出 `Lite-AMP-Depth-v0` 和 `Lite-AMP-Depth-Play-v0`。
训练脚本按 `LITE_PYTHON`、本目录 `.venv/bin/python`、当前 `python` 的顺序选择解释器，
不用激活环境也可从仓库根目录执行 `./train.sh parkour ...`。

训练预检把 Python、固定依赖版本、CUDA 实际张量分配，以及 URDF、27 个网格和 72 段动作
的检查结果写入 `logs/environment-check.json`。缺少 URDF 导入组件、旧 Isaac Lab、CPU 版 PyTorch
或 Git LFS 指针都会在启动仿真前退出，并列出修复信息。
脚本自动配置 NVRTC builtins 路径，并使用短的 `/tmp/loong-lite.*` 临时目录，
支持本项目这样的长中文目录；退出时清理本次临时目录。

## 二、训练

在 `training_parkour` 目录执行；使用本目录 `.venv` 时无需激活环境：

```bash
# 4096 个环境、训练 30000 轮
bash scripts/train_lite.sh \
  --num_envs 4096 \
  --max_iterations 30000 \
  --device cuda:0 \
  agent.device=cuda:0

# 新机器先做两轮训练检查；成功后应生成 model_0.pt、model_1.pt 等 checkpoint
bash scripts/train_lite.sh \
  --num_envs 2 --max_iterations 2 --run_name setup_smoke \
  --device cuda:0 agent.device=cuda:0 agent.save_interval=1

# 原包已有验证：32 环境、12 轮短程检查
bash scripts/train_lite.sh \
  --num_envs 32 --max_iterations 12 --run_name smoke \
  --device cuda:0 agent.device=cuda:0

# 续训自己的 Lite 实验；load_run 填完整目录名
bash scripts/train_lite.sh \
  --num_envs 4096 --max_iterations 30000 \
  --resume --load_run <完整run目录名> --checkpoint model_XXX.pt \
  --device cuda:0 agent.device=cuda:0
```

`train_lite.sh` 已经启用 `--headless`。也可通过 `LITE_PYTHON=/path/to/python bash scripts/train_lite.sh ...`
显式指定 Python。训练任务为 `Lite-AMP-Depth-v0`。
日志和模型写入 `logs/rsl_rl/lite_amp_depth/<时间>_<唯一标识>_<run_name>/`，默认每 **200** 轮保存一个 PT，
正常结束时再保存最后一轮；添加 `agent.save_interval=100` 可改为每 100 轮保存。
`--max_iterations` 在续训时表示追加的迭代数。每次启动另保留 `logs/launches/train.*/train.log` 与
`completed.json`。只有本次 nonce 匹配、请求轮数完成、最终 checkpoint 实际保存且大小/时间/哈希
一致，入口才报告训练成功；Kit 即使在异常后返回退出码 0，也不会被当成训练完成。

训练配置保持为：

- 26 个活动关节：双腿 12＋双臂 14，双腕共 6 轴活动，没有活动腰关节。
- 行走 **95%**、站立 **2%**、原地转向 **3%**；每 10 秒重采样。
- 8 类地形各 **12.5%**：上/下台阶、随机箱块、Perlin 粗糙地面、上/下坡面、横栏、重复圆柱。
  不使用踏脚石和同心沟壑。训练地形块 8×8 m，10 级课程、24 列。
- 仿真 200 Hz、策略 50 Hz；使用即时 PD 控制，电机和深度图均不额外加入延迟。
- 相机在 `base_link` 下 `(0.1237, 0, 0.40966)` m，向下 45°。
- AMP 数据保留完整 **72 段 / 7569 帧**；训练判别器使用 12 段行走＋42 段楼梯，
  楼梯单段权重 0.3；机器人每次复位到当前地形原点的固定默认站姿，速度为零。

已移除接触材质、质量/惯量、重心、PD 增益、关节摩擦、电枢惯量、关节零位、
初始状态噪声、参考帧随机复位、外部推扰、策略观测噪声，以及电机/深度图延迟随机化。
物理参数使用机器人定义的固定标称值。地形课程、95%/2%/3% 指令采样和 AMP 示范动作采样继续使用。

随包 `checkpoints/lite/model_10200.pt` 是修改前训练的已有策略示例，其参数记录和权重保持原样，
不代表在当前无域随机化设置下重新训练。导出器兼容其历史参数记录，sim2sim 仍按当前无额外延迟方式执行。
默认训练从头开始，
不会自动加载这个示例。新训练短程检查只验证训练链路，不能表示新策略已经收敛。

## 三、Isaac Lab 回放

```bash
python scripts/rsl_rl/play.py \
  --task Lite-AMP-Depth-Play-v0 --num_envs 1 --viz kit \
  --checkpoint logs/rsl_rl/lite_amp_depth/<run>/model_XXX.pt \
  --kit_args "--enable isaacsim.asset.importer.urdf" \
  --follow_cam --follow_body base_link
```

需要图形桌面。MuJoCo 回放见下一节，不依赖 Isaac Lab 的图形窗口。

## 四、sim2sim：读取 PT，转换 ONNX，再加载

```bash
# 随包示例：每次先导出 PT，然后用 ONNX Runtime 驱动 MuJoCo
python scripts/sim2sim_mujoco_lite.py \
  --checkpoint checkpoints/lite/model_10200.pt \
  --terrain stairs --follow-view

# 自己的训练目录：自动选择轮数最大的 model_<轮数>.pt
python scripts/sim2sim_mujoco_lite.py \
  --checkpoint logs/rsl_rl/lite_amp_depth/<run> \
  --terrain flat --follow-view

# 已有 ONNX，直接推理
python scripts/sim2sim_mujoco_lite.py \
  --onnx checkpoints/lite/exported/policy_10200.onnx \
  --terrain box --follow-view

# 无窗口运行 10 秒（服务器验证）
python scripts/sim2sim_mujoco_lite.py \
  --checkpoint checkpoints/lite/model_10200.pt \
  --terrain stairs --headless --duration 10

# 单独导出，不进入仿真
python scripts/sim2sim/export_lite_pt_to_onnx.py \
  --checkpoint checkpoints/lite/model_10200.pt
```

PT 转换输出位于相邻 `exported/` 目录，文件名为 `policy_<轮数>.onnx` 和同名 `.json`。
JSON 包含关节顺序、电机参数、默认姿态、相机和模型哈希；优先使用 PT 相邻的 `params/env.yaml`。
传 PT 时每次重新导出；`--onnx` 可跳过导出。`--pt` 是 `--checkpoint` 的别名。
目录可以整体移动；路径根据当前脚本位置解析，没有指向原工程的运行时依赖。

支持 `flat`、`stairs`、`box`。sim2sim 的 `stairs` 宽 **2 m**，踏面长 **0.35 m**，
默认台阶高 **0.20 m**（`--step-height 0.10` 可改高），保持浅灰踏面、深灰立面。
颜色装饰不参与接触或策略深度。此处的 2 m 是 sim2sim 测试楼梯的宽度，不改变训练地形块尺寸。

键盘窗口：W/S 前后、A/D 横移、Q/E 转向、按住 R 加速、T 复位。
`--cmd-vx 0 --cmd-dyaw 0` 为站立；`--cmd-vx 0 --cmd-dyaw 0.5` 为原地转向。
`--heading 0` 保持零航向。sim2sim 使用即时电机控制和当前深度帧，不提供额外延迟选项。

模型接口：`obs[1,450] + depth[1,8,18,32] → actions[1,26]`。
物理 200 Hz、策略 50 Hz，默认深度包含自身遮挡，使用训练一致的裁剪、模糊、归一化及历史。

### 只安装 sim2sim 环境

如果对方只运行 PT/ONNX，不训练，可以不安装 Isaac Sim / Isaac Lab：

```bash
conda create -n lite_sim python=3.12 -y
conda activate lite_sim
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-sim.txt
python -m pip install -e .
python scripts/check_environment.py
python scripts/sim2sim_mujoco_lite.py \
  --checkpoint checkpoints/lite/model_10200.pt --terrain flat --headless --duration 10
```

GUI 需要可用的 OpenGL 图形桌面、pygame、mujoco-python-viewer；headless 推理不需要窗口。
保存离屏图片可加 `MUJOCO_GL=egl` 和 `--save-frame logs/preview.png`，需要驱动支持 EGL。

## 五、资源检查与测试

```bash
# 检查 72 段动作、限位、四元数和关键体 FK
python scripts/tools/validate_motion_data.py

# 未安装训练依赖时，先用 Python 标准库检查本地资源完整性
python scripts/check_environment.py --assets-only

# 由本项目 Lite URDF 重新生成 MuJoCo XML，并检查 64 组随机姿态的 FK、质量、惯量
python scripts/tools/build_lite_mujoco.py --check

# 完整测试需要训练环境；纯 sim 环境可只运行两个部署测试文件
python -m pytest -q tests
python -m pytest -q tests/test_lite_sim2sim.py tests/test_lite_onnx_export.py
```

已提供映射好的 26 维动作。修改 URDF 的关节定义、几何或限位后，需要重新适配参考动作，
不能只修改 manifest 的哈希；数据验证和训练环境会检查模型与动作定义的一致性。

## 六、环境与验证范围

训练需要上文固定版本的 Isaac Sim / Isaac Lab；安装脚本负责安装到独立环境。Isaac Sim 6 的 Linux 环境要求 Ubuntu 22.04/24.04 和兼容的 NVIDIA 驱动，本次默认 CUDA 12.8 组合已在驱动 580.65.06 通过。参考 [Isaac Sim 系统要求](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)。

原包已有验证覆盖独立包安装、短训练、ONNX 导出和 MuJoCo 短时推理；短训练不能代表策略已经收敛，短时地形回放不能代表完成全部难度场景。此目录使用独立训练与 sim2sim 入口。2026-11-12 整合复核记录见 [TRAINING_VALIDATION.md](TRAINING_VALIDATION.md)，其中明确区分本次检查与原包已有结果。

文件完整性通过仓库根目录的 `python3 -B check_release.py` 检查。请保留 `LICENSE`、`NOTICE.md` 和各源码的原始许可头；部分上游组件适用 CC BY-NC 4.0 非商业使用条款。
