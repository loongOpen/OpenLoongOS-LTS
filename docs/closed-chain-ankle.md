# 并联脚踝闭链仿真

在 Ubuntu 22.04 x86_64 远程桌面运行 `./start_openloong_gui.sh`。
GUI、同一仿真环境及行走 4 / 跳舞 28 / 越障 43 的操作不变。

## 运行结构

- `loong_gui_sim/runtime/module/loong_driver_sdk/libjoint_map_x64.so`：按指定 `linglongv2.0` 源码编译的 x64 解算库。部署包 v4.2.7 提供 ARM64 二进制，不能直接加载到 x64 电脑。
- `loong_gui_sim/runtime/module/loong_driver_sdk/SOURCE.json`：源码版本、配置来源和二进制校验信息。
- `loong_gui_sim/runtime/config/joint_map.ini`：部署包连杆参数；踝轴中心、方向与限位明确绑定当前 1.4 XML，不依赖外部 URDF。
- `loong_gui_sim/runtime/config/driver.ini`：`useJntMotMap=1`、`simAnkleMap=closed_chain`、`pdMode=1`。
- `loong_gui_sim/runtime/model/linglong/linglong_26dof.xml`：左右各两组曲柄、球铰连杆和连接约束；踝 pitch/roll 为被动关节，执行器作用于电机轴。

命令路径：关节目标/关节 PD → 实机解算库 → 电机力矩 → 闭链机构。
反馈路径：电机轴位置/速度/力矩 → 实机解算库 → 策略关节反馈。
仿真端不再进行虚拟电机适配，避免重复转换。控制总线保持 31 槽、26 个执行器。
曲线按执行器映射显示；回放直接读取电机坐标，通过相同解算库恢复被动踝关节及连杆姿态。

## 几何与惯量边界

曲柄长度 66 mm；电机轴高度为踝中心上方 186 mm、297 mm；踝中心为膝关节下方 420 mm。
部署库外部电机顺序均为长杆、短杆，对应左侧槽 23/24、右侧槽 29/30。
四个 `equality/connect` 约束参与实际动力学求解，不是仅绘制连杆。

资料未提供独立曲柄、连杆的实测质量惯量。当前每个曲柄 0.04 kg、每根连杆 0.06 kg 为仿真估计；
这些质量从原小腿组件中重新分配，使零位的组件总质量、质心和惯量保持原值。
电机 armature 复用原模型数值，尚未独立标定。因此当前验证证明几何与接口可运行，不等于实物动力学完全复现。
训练源码和已训练策略未修改；全链仿真的脚踝动力学已变化。

## 验证范围

2026-11-17 在 Ubuntu 22.04 远端完成：

- 双踝 8 秒动力学摆动：最大闭合误差 7.90e-8 m；真实解算与物理关节角最大差 1.70e-6 rad；无 MuJoCo 警告。
- SDK 三模式 45 秒回归：含行走、跳舞、越障行进与切换，未跌倒。
- 隔离四进程链约 72 秒：包含完整一段跳舞、越障行进及前后切回行走；718 个姿态样本，最低高度 0.8947 m，进程正常退出。使用正式模拟器/驱动及仿真编译的 interface/locomotion，未接真实硬件。
- 闭链回放 80 帧：最大闭合误差 7.14e-8 m，最大关节角差 1.33e-6 rad。

运行 `python3 -B check_release.py` 检查文件、闭链、电机顺序和真实解算开关。
