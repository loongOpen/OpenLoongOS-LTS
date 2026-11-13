# 越障 Plan：实现、接口与后续训练计划

## 已完成的部署路线

越障训练任务为 **Lite-AMP-Depth-v0**，机器人是 **Lite26**，双腿 12＋双臂 14 个关节。
原桌面行走运行时使用 LingLong 的旧总线和另一份策略，机器人定义与观测语义不同。
因此 GUI 在越障模式启动独立 Lite26 MuJoCo 后端，并加载 `libplan_parkour_x64.so`；
没有将 Lite26 策略冒充旧 Nabo `creat/destroy` 插件，也没有套用原 LingLong 关节映射。

已实现的链路：

1. 从随包 `training_parkour/checkpoints/lite/exported/policy_10200.onnx` 和同名 JSON 读取模型、
   URDF 哈希、26 关节顺序、相机外参、默认姿态、动作缩放、PD、力矩及速度上限。
2. 直接复用 `training_parkour/scripts/sim2sim_mujoco_lite.py` 和 `scripts/sim2sim/runtime.py`
   构建 Lite26 模型、碰撞体、地形、观测和射线深度；不另抄一份容易漂移的训练配置。
3. 原生 `.so` 通过 ONNX Runtime C API 运行双输入策略；Python 使用 `ctypes` 调用原生 Plan，
   不以 Python ORT 推理作为运行时兜底。CPU 单线程，无 CUDA/Isaac Lab/PyTorch 的 GUI 运行依赖。
4. GUI 通过会话目录发送速度、暂停和复位，将原生 MuJoCo GLFW 窗口嵌入与行走相同的视口宿主，同时接收最新策略输入深度及状态。
   地图内物体参与接触碰撞，深度由当前 MuJoCo 场景射线求交产生。
5. 推理输入非有限、维度不符、深度范围错误或输出非有限时明确报错；不会输出伪造动作。
   MuJoCo 每一步检查状态有限。底座高度低于 0.55 m 或躯干竖直轴 Z 分量低于 0.4 时
   输出 `fallen` 并返回退出码 2，避免将倒地误报为正常越障。

## 策略与控制契约

| 项目 | 当前实现 |
|---|---|
| 原生 ABI | `loong_parkour_abi_version/create/infer/destroy/contract`，版本 1；头文件 `loong_gui_sim/native/parkour_plan.h` |
| 输入 | float32 `obs[1,450]`、`depth[1,8,18,32]` |
| 输出 | float32 `actions[1,26]`，动作顺序来自部署 JSON |
| 本体历史 | 各项分别保留 5 帧，再按项拼接：底座角速度 3、去航向旋转表示 6、速度指令 3、关节角 26、关节速度 26、上次动作 26 |
| 深度相机 | `base_link`，位置 `(0.1237,0,0.40966)` m，向下 45°，64×36 射线；光学 Z 距离，包含自身可视网格遮挡 |
| 深度预处理 | 下半幅、左右各裁 16 列，3×3 高斯模糊，裁至 `[0,2.5]` m，除以 2.5；最终 18×32 |
| 深度历史 | 37 帧缓冲，取滞后 `35,30,25,20,15,10,5,0` 的 8 帧，最老在前 |
| 频率 | 物理 200 Hz；策略/相机历史 50 Hz；无额外电机或深度延迟 |
| 控制 | `target=q0+action_scale*actions`；`torque=clip(kp*(target-q)-kd*dq,±effort)`；超过转速上限时禁用继续加速的力矩 |
| 数据错误 | 原生返回错误消息和非零状态；不修改调用方动作输出缓冲 |

## 文件与重建

```text
loong_gui_sim/native/parkour_plan.{h,cpp}       原生 Plan 源码（MIT）
loong_gui_sim/native/build_parkour_plan.sh      当前平台构建（g++、C++17）
loong_gui_sim/native/vendor/onnxruntime/        C API 21 头文件与上游 MIT 许可
loong_gui_sim/runtime/module/parkour/
  libplan_parkour_x64.so                       已构建的 Linux x86_64 Plan
loong_gui_sim/runtime/python/native_policy.py  ctypes 绑定、当前 Python ORT 动态库定位
loong_gui_sim/runtime/python/parkour_runtime.py GUI 后端
training_parkour/                             模型、部署 JSON、URDF、网格、训练/部署 helpers
```

在交付根目录：

```bash
bash loong_gui_sim/native/build_parkour_plan.sh
loong_gui_sim/runtime/python/.venv/bin/python \
  loong_gui_sim/runtime/python/parkour_runtime.py --help
loong_gui_sim/runtime/python/.venv/bin/python \
  loong_gui_sim/runtime/python/parkour_runtime.py --self-test --depth-dir /tmp/loong-parkour-smoke
loong_gui_sim/runtime/python/.venv/bin/python -m unittest discover \
  -s loong_gui_sim/native/tests -p 'test_parkour_plan.py'
```

`.so` 在当前进程中通过 `dlopen` 加载外部运行环境安装的 ONNX Runtime C 库，
要求 API 21 或更新；不包含编译机器的绝对路径，不需要系统安装 ORT 开发库。
构建脚本也能在 aarch64 本机输出 `libplan_parkour_a64.so`，但本次只验证了 x86_64，
不能据此声称 ARM 已验收。部署协议是仿真 ABI；真机需要额外实时总线、相机与安全接口验收。

## GUI 会话协议

```bash
loong_gui_sim/runtime/python/.venv/bin/python \
  loong_gui_sim/runtime/python/parkour_runtime.py \
  --native-viewer --terrain box --duration 600 \
  --depth-dir /tmp/loong-session --status-dir /tmp/loong-session \
  --command-file /tmp/loong-session/command.json
```

`command.json` 采用写临时文件再改名的方式更新：

```json
{"vx":0.6,"vy":0.0,"yaw_rate":0.0,"reset_id":0,"enabled":true}
```

速度范围为 `vx[-0.8,1.5]`、`vy[-0.5,0.5]` m/s、`yaw_rate[-1,1]` rad/s。
`enabled=false` 冻结仿真和策略，保留画面；`reset_id` 改变复位模型、动作和历史。
非法命令会暂停并上报错误。可用 `--start-paused` 从暂停开始。

深度约 10 Hz 输出 `depth.pgm`（近白远黑）、`depth.f32`（32×18、小端 float32、米）及
`depth.json`。JSON 包含尺寸、距离范围、统计、帧号、感知时间 `depth_sim_time`、当前模拟时间、
单调时钟、状态、指令、原生推理次数、模型类型和文件 SHA-256。
显示的是 `depth` 输入中的**最新一帧**，不是另一个相机或人为生成的图。
先换图片、再原子换 JSON；消费者必须校验完整文件的 SHA-256，遇到不同帧就跳过重读。
常量文件名只占用固定空间，长时间运行不会无限积累帧文件。

GUI 主画面使用原生 GLFW 窗口，实际帧缓冲随宿主像素尺寸变化，鼠标旋转、缩放及跟随状态保存在 `viewer.json`。深度面板仍显示策略实际使用的 18×32 数据。

独立无界面导出时可显式使用 `--headless --snapshot-dir ...`，RGB 最多 5 Hz 输出 `scene.ppm`（320×240）及带 SHA-256 的 `scene.json`。
RGB 的低阴影/抗锯齿设置仅影响预览，不改变策略深度与碰撞。GUI 原生模式检测并使用 GLFW／X11；独立无界面导出可自动检测或使用 `LOONG_RUNTIME_GL` 指定后端。无离屏 OpenGL 时去掉
`--headless --snapshot-dir ...`，使用 MuJoCo 桌面窗口；深度可继续显示于 GUI。
结束时写 `result.json`，包括步数、原生调用次数、有限状态、最小高度、起终点、双脚位置与箱体通过量测。

## 本次验证与能力边界

实际验证环境为独立 GUI 最小环境 Python 3.12.14、MuJoCo 3.4.0、ONNX Runtime 1.21.0、
NumPy 2.1.3；其中没有 torch、Isaac Lab 或 Qt。另用系统 MuJoCo/ORT 1.26 检查了同样链路。

- `--self-test`：200 个物理步、50 次真实观测推理，逐帧与原 Python ORT 对照，最大动作绝对误差 **0**，状态有限。
- 100 组随机本体/深度输入的原生与 Python ORT 比较通过；错误形状、NaN、深度越界、旧模型被拒绝，错误时输出缓冲保持原样。
- 平地 10 s：2000 步、500 次原生调用，根部前进至 x=6.155 m，最低高度 0.937 m，无倒地。
- 箱体 10 s：根部从 `(0,0,0.99)` 到 `(5.912,0.089,1.021)` m。第一箱高 0.20 m，
  中心 x=4.5 m、范围 x=[4.1,4.9] m、y=[-0.75,0.75] m，接触碰撞启用。
  双脚踝最终 x=5.928/5.897 m，各减 0.30 m 保守余量仍超过箱体远边。
  根部在箱体 x 范围内共 267 步，最大横向偏移 0.223 m，路径经过箱体上方；不是绕行。
- 独立复核箱体 20 s 到 x=10.945 m，越过前两个 0.20/0.30 m 箱体；尚未覆盖第三个 0.40 m 箱体。
- **0.20 m 台阶、vx=0.6**：5.740 s、287 次推理后躯干倾倒，返回 `fallen`/退出码 2。
  **0.10 m 台阶、vx=0.6**：6.975 s 倾倒。默认演示采用箱体，台阶保留为明确的挑战选项。
  不将这些失败改写为“台阶已通过”或“所有难度均已验收”。
- 单独的低难度台阶演示 `--terrain stairs --step-height .05 --cmd-vx .4 --duration 35`：
  7000 步、1750 次原生调用，完成 x=[2,8.5] m 的上台阶/平台/下台阶，到 x=10.160 m。
  台阶范围内根部最大横向偏移 0.480 m，小于 1 m 半宽；双脚踝 x=10.166/10.176 m，
  各减 0.30 m 余量仍超过 8.5 m 末端，最低根部高度 0.958 m。驶出台阶后有横向漂移，
  不代表长期航向保持通过。该演示不修改训练定义或高难度失败记录。
- 其它低速尝试：10/20 cm 台阶、vx=0.3 在 20 s 分别停在 x=1.856/1.877 m，
  尚未进入台阶，不能算越障成功；10 cm、vx=0.4 在 15.040 s 倾倒，vx=0.5 在 7.645 s 倾倒。
- EGL 实际 RGB 出图、实际深度及 SHA-256 帧配对已经检查。

随包 `model_10200.pt` 是历史训练结果，不代表当前去域随机化设置重新收敛后的策略。
GUI 支持加载自己的新 ONNX，但必须同时使用同名部署 JSON；模型/URDF 校验失败会停止。

## 下一阶段训练与验收计划

1. **固定对照基线**：保留当前模型哈希和上述失败轨迹，以现有配置/动作示范完整启动 PPO+AMP。
   用训练目录内环境检查、动作校验和短训练确认数据闭包；短训练只验链路，不做能力声明。
2. **训练台阶课程**：保持既定 8 类地形和 26 关节定义，从课程低难度升至目标高度。
   记录每类地形完成率、跌倒率、速度误差、关节峰值力矩、动作与深度历史；使用不同训练种子。
   诊断当前台阶失效时区分感知、落脚、力矩饱和和策略泛化，先复现数据再改奖励或课程。
3. **导出与部署回归**：每个候选 PT 通过现有 exporter 生成 ONNX/JSON，重新执行原生/Python
   逐帧动作对比、完整关节/相机/PD 对照、10/20/30 cm 台阶与 20/30/40 cm 箱体测试。
   保留碰撞、真实深度和倒地判定；不得以减弱碰撞、置零深度或放宽倒地阈值取得“通过”。
4. **发布门槛**：按场景×速度×种子生成成功率表，明确通过率与轨迹长度，收录失败样本。
   只有候选模型完成全部目标场景并经 GUI 长时回归，才替换默认示例并声称该难度可稳定越障。
5. **真机另行接入**：在已验证 Lite26 总线、相机时间戳/外参、单位和历史同步后添加设备适配层，
   单独验证时延、掉帧、限幅和急停；本次 GUI 仿真产物不代替真机验收。
