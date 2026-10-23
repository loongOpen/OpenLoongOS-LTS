# 全链仿真使用说明

面向 Ubuntu 22.04 / Linux x86_64，需 X11 或 XWayland 图形桌面。程序已包含 Python 和 PyQt5，无需编译。
默认采用原仓库灵龙 **26 轴模型**（双腿 12＋双臂 14，头腰不参与仿真），RL 使用内置 `policy.onnx`。
请完整保留 `loong_gui_sim/`，其中的 `runtime/` 和 `native/` 是必需资源。

## 1. 启动

首次使用安装系统依赖：

```bash
sudo apt install x11-utils libgl1 libegl1 libx11-6 libxcb-xinerama0 libxkbcommon-x11-0
```

进入整合包根目录运行：

```bash
cd loong_open_framework
./start_sim.sh
```

GUI 自动打开仿真。点击 **“一键启动控制链”**，按提示输入本机 sudo 密码，等待 driver、interface、locomotion 启动完成。

## 2. 让机器人站起来

1. 保持默认辅助力开启，速度指令保持为零。
2. 在“行走控制”页点击 **EN → RC**，等待恢复完成、进入 IDLE。
3. 点击 **RL**，待姿态稳定后，点击仿真画面并按 **F** 关闭辅助力。
4. 结束时点击 **“停止本次进程”**；重新开始前按空格清零速度。

**不要在 RC / IDLE 阶段提前关闭辅助力：它们只恢复、保持关节姿态，不提供主动平衡。**
RL 自动加载唯一策略，无需额外发送模型选择键值 41。

## 3. 常用操作

| 操作 | 功能 |
| --- | --- |
| 控制区 W / S | 增减前后速度 |
| 控制区 A / D | 增减左右速度 |
| 控制区 Q / E | 增减转向速度 |
| 控制区空格 | 清零全部速度指令 |
| 仿真画面 F | 开关辅助力 |
| 仿真画面 G / H | 降低 / 升高辅助高度 |
| “镜头跟随” | 跟随机器人 |
| PLOT / “打开日志回放” | 查看实时曲线 / 导入 driver 文本日志 |

速度按键会累加指令，松开按键不会自动清零。键盘无响应时，点击对应区域，或使用控制区同名按钮。
回放期间暂停实时控制发送。插件键值为：`rc=2`、`rl=4`、`jntTest=31`。

## 4. 配置与日志

以下路径相对于 `loong_gui_sim/`：

| 文件 / 目录 | 用途 |
| --- | --- |
| `runtime/model/scene.xml` | 场景入口，引用 `linglong/linglong_26dof.xml` |
| `runtime/config/driver.ini`、`driver_mujoco.ini` | 关节映射、固定 PD 与辅助力 |
| `runtime/config/plan_rl.ini` | RL 观测、动作映射及推理周期 |
| `runtime/model/onnx/policy.onnx` | 当前策略，与整合包 `models/policy.onnx` 一致 |
| `runtime/log/`、`runtime_logs/` | SDK 日志、各进程启动与运行输出 |

当前配置为 `drvNums=31`、`activeDrvNumUser=26`、`pdMode=3`、`useJntMotMap=0`、`simAnkleMap=none`；所有有效轴使用 MuJoCo 固定 PD。
更换模型时，两份 driver 的关节映射必须一致；更换策略须匹配当前 `obs[1,700] → actions[1,20]` 接口及观测定义。
修改模型、驱动或策略配置后，**退出旧 GUI 和控制链，再重新启动**。

## 5. 排查与自检

- **窗口打不开**：确认图形桌面及 `DISPLAY` 可用，检查系统依赖是否安装。
- **控制链启动失败**：查看对应进程日志，确认 sudo 授权完成。
- **恢复后倒地**：检查是否在进入 RL 前关闭了辅助力，按第 2 节重新操作。
- **曲线无数据**：确认 driver 正常运行，`log2udpFlag=1`、`log2udpIp=127.0.0.1`。

在整合包根目录执行离线自检，不启动控制链：

```bash
QT_QPA_PLATFORM=offscreen ./start_sim.sh --self-test
```

已通过 RL 零速度、无辅助力的 30 秒进程内闭环站立验证。完整多进程运行和各速度下持续行走尚未验证，详见[验证记录](../VALIDATION.md)。
