# 组件许可与来源

本项目是多个组件的整合发布，不使用单一许可证覆盖所有内容。

- `mjlab_amp_ovision/`：保留 Apache-2.0 许可与源码版权声明，基于 mjlab，强化学习依赖 rsl_rl。
- `loong_gui_sim/runtime/license`：原 SDK 的许可文本，保留原作者与版权信息。
- `loong_gui_sim/licenses/`：仿真 SDK、driver、关节映射和 MuJoCo / GLFW 等第三方材料。
- `licenses/loong_gui_ctrl/`：沿用输入包的目录命名，保存 GUI 使用的 Python、PyQt5 / Qt、PyInstaller 等材料；目录存在不代表本次交付机器人控制台。
- `models/`、机器人网格、参考动作及演示素材：保留输入内容及已有声明；尚未在输入包中发现覆盖全部资源的独立再分发授权说明，维护者应补齐来源与授权范围。

## GUI 发布前待核实

当前输入包的说明明确使用 GPL v3 版 PyQt5，UI 业务模块以预编译程序交付，没有对应 UI 源码和构建材料，也没有附带可确认的商业许可依据。因此，现有许可文件不足以确认这一 GUI 二进制的公开分发条件已经完整满足。

维护者需确定适用的发布方式并补齐相应材料，例如采用符合 GPL 要求的源码交付方式，或确认适用的商业授权。仅保留一份 GPL 文本不能替代对具体分发条件的核实。

参考：[PyQt 官方许可说明](https://riverbankcomputing.com/software/pyqt/)、[官方许可 FAQ](https://riverbankcomputing.com/commercial/license-faq)。此项是基于当前交付材料的发布检查，不替代对项目实际授权情况的确认。
