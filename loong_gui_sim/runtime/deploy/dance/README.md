# 跳舞部署文件与 GUI 运行

2026-11-12 将用户提供的五份文件纳入本发布包，文件原样保留，SHA-256 见 `ASSETS.json`：

- `libplan_bm_taiji_mjlab_a64.so.1`、`plan_bm_taiji_mjlab.ini` 在本目录。
- ONNX、`.refbin` 和原始 `.npz` 在 `runtime/model/dance/`。

原始 `.so.1` 是 **aarch64 / Loong Nabo plan ABI**，依赖 `libloong_loco_sdk_a64.so` 和 ONNX Runtime 1.x。本包桌面 SDK 为 x86-64，不能把 ARM 库重命名成 x64 库加载。原始 ARM 插件的源码未包含在这五份附件中，其硬件运行不属于本次桌面验证。

桌面 GUI 的“跳舞”功能使用 `native/libdance_plan_x64.so`。这是依据已整合的 `training_dance` 和 ONNX 元数据新实现的 C API，不冒充原 ARM 插件。源码 `native/dance_plan.cpp` 和构建脚本 `native/build_dance_plan.sh` 随发布包提供。它直接调用随包 ONNX Runtime 1.21 CPU 推理，不需要 Torch、CUDA、训练 Python 包或独立联网服务。

从 `loong_gui_sim/` 目录运行（最小环境由 `setup_runtime.sh` 安装）：

```bash
bash native/build_dance_plan.sh
runtime/python/.venv/bin/python runtime/python/dance_runtime.py
```

GUI 通过下面的接口运行同一策略，RGB 图像和状态使用原子写入，状态内包含图像 SHA-256：

```bash
runtime/python/.venv/bin/python runtime/python/dance_runtime.py \
  --headless --command-file /tmp/dance-session/command.json \
  --status-file /tmp/dance-session/status.json --snapshot-dir /tmp/dance-session
```

命令格式：`{"enabled":true,"reset_id":0}`；修改 `reset_id` 重新回到轨迹初始状态，`enabled=false` 暂停。跳舞不使用移动速度指令。`stop=true` 可退出。图像文件为 `snapshot.png`，状态包含 `frame`、`steps`、`sim_time`、`base_height`、`fallen`、`image` 和 `image_sha256`。启动前独立检测实际渲染能力：桌面有 `DISPLAY` 时优先 GLFW，服务器无显示时优先 EGL；可通过 `LOONG_RUNTIME_GL` 显式指定后端。

`--config` 默认读取本目录原始 INI，使用其模型、参考文件名、`startFrame` 和 `playbackLoop`，并核对 `1610 / 26 / mujoco_linglong14 / 0.02s`。`--model`、`--reference`、`--motion`、`--start-frame`、`--loop` 可显式覆盖。默认不循环，1931 帧之后保持最后一帧。

桌面仿真重置会将基座位姿、速度和 26 个关节初始化到参考轨迹状态，因此不执行硬件从当前姿态交接用的 `startupSmoothTime`。INI 中头部增益、CPU 绑核、硬件日志选项属于原 ARM 插件，不控制本桌面模型；本训练机器人没有 SDK 的额外头部/腰部控制槽。

策略输入严格为 `21*35 + 35 + 10*84 = 1610`：参考窗口、当前参考和从旧到新的本体历史。重置时历史填满当前观测；输入无二次归一化，输出无额外截断，目标为 `q_reference + action_scale * action`。归一化已内置于用户提供的 ONNX。

MuJoCo 使用训练相同的 URDF 质量/惯量、碰撞体、转子惯量和 0.005s 物理步长；策略步长 0.02s。踝关节在每个物理步执行显式 PD 并限制力矩，其余关节为 MuJoCo 原生位置 PD，沿用训练增益和力矩上限。关节按名字映射，未复用原行走 SDK 的 20 动作映射。

验证命令：

```bash
runtime/python/.venv/bin/python -m unittest discover -s runtime/python -p test_dance_contract.py -v
runtime/python/.venv/bin/python runtime/python/dance_runtime.py --headless --fast --steps 1931
```

2026-11-12 在独立 Python 3.12 / MuJoCo 3.4 最小环境中，三项数值契约检查通过；完整 1931 策略步（38.62s 仿真）无非有限状态、无摔倒，最低基座高度 0.870978m，最后 0.958595m。单独渲染检查成功生成 640×480 实际仿真 RGB。测试证明附件在此桌面实现中的一次完整轨迹运行，不代替硬件验证或所有动作的稳定性评估。

源码版权：桌面新增实现沿用 GUI 的 MIT；URDF 物理重建逻辑及参数来自本发布包 `training_dance`，其 Apache-2.0 和 Linglong 源 MIT 声明见相应目录。原始附件保持用户提供形式，其许可证不因本次整合而被改写。
